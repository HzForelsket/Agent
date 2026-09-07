#ifndef PREFIX_GROUPER_NPU_ATTENTION_COMMON_H
#define PREFIX_GROUPER_NPU_ATTENTION_COMMON_H
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "shared_prefix_attention_tiling.h"

namespace shared_prefix {
using namespace AscendC;
using Bf = bfloat16_t;
using A = MatmulType<TPosition::GM, CubeFormat::ND, Bf>;
using AT = MatmulType<TPosition::GM, CubeFormat::ND, Bf, true>;
using C = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using ScoreMatmul = Matmul<A, AT, C>;
using ValueMatmul = Matmul<A, A, C>;
using TransposeMatmul = Matmul<AT, A, C>;

template <HardEvent event> __aicore__ inline void Fence()
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}
__aicore__ inline uint64_t Min(uint64_t a, uint64_t b) { return a < b ? a : b; }
__aicore__ inline uint32_t Align(uint32_t n) { return (n + 15) / 16 * 16; }
struct Block {
    GlobalTensor<Bf> gm;
    uint64_t offset;
    uint64_t stride;
    uint32_t rows;
    uint32_t cols;
};

// AIV coordinates the pipeline; CANN's registered Matmul service owns the AIC.
// User queues/events never reuse Matmul's cross-core notification IDs.
class Attention {
public:
    TPipe pipe;
    ScoreMatmul scoreMm;
    ValueMatmul valueMm;
    TransposeMatmul transposeMm;
    SharedPrefixAttentionTilingData t;
    GlobalTensor<Bf> q, k, v, grad;
    GlobalTensor<float> output, dk, dv, lse, delta;
    GlobalTensor<int32_t> prefixStart, prefixEnd, sequenceStart, sequenceEnd, groupEnd;
    LocalTensor<float> scores, dp, probability, ds, state, metadata, reduce, oldRow;

    __aicore__ inline void Init(const SharedPrefixAttentionTilingData& tiling, GM_ADDR workspace)
    {
        t = tiling;
        b = t.tile;
        square = b * b;
        pipe.InitBuffer(inputQueue, 2, 2 * square * sizeof(Bf));
        pipe.InitBuffer(resultQueue, 2, square * sizeof(float));
        pipe.InitBuffer(scoreBuf, square * sizeof(float));
        pipe.InitBuffer(dpBuf, square * sizeof(float));
        pipe.InitBuffer(probBuf, square * sizeof(float));
        pipe.InitBuffer(dsBuf, square * sizeof(float));
        pipe.InitBuffer(weightBuf, square * sizeof(Bf));
        pipe.InitBuffer(stateBuf, b * 16 * sizeof(float));
        pipe.InitBuffer(metaBuf, 2 * b * 16 * sizeof(float));
        pipe.InitBuffer(reduceBuf, b * sizeof(float));
        pipe.InitBuffer(oldBuf, b * sizeof(float));
        scores = scoreBuf.Get<float>(); dp = dpBuf.Get<float>();
        probability = probBuf.Get<float>(); ds = dsBuf.Get<float>();
        state = stateBuf.Get<float>(); metadata = metaBuf.Get<float>();
        reduce = reduceBuf.Get<float>(); oldRow = oldBuf.Get<float>();
        auto* base = reinterpret_cast<__gm__ uint8_t*>(workspace) + GetBlockIdx() * t.core_workspace_bytes;
        for (uint32_t i = 0; i < 2; ++i) {
            aSlot[i].SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(base + i * square * 8));
            bSlot[i].SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(base + i * square * 8 + square * 2));
            cSlot[i].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base + i * square * 8 + square * 4));
        }
        weights[0].SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(base + square * 16));
        weights[1].SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(base + square * 18));
    }

    __aicore__ inline Block Input(GlobalTensor<Bf>& gm, uint64_t token, uint64_t head,
                                 uint64_t heads, uint32_t rows, uint64_t d)
    {
        return {gm, (token * heads + head) * t.head_dim + d, heads * t.head_dim,
                rows, static_cast<uint32_t>(Min(b, t.head_dim - d))};
    }
    __aicore__ inline Block Weight(uint32_t which)
    {
        return {weights[which], 0, b, b, b};
    }
    __aicore__ inline void CopyBlock(LocalTensor<Bf> local, const Block& src)
    {
        // Initialize absent rows; DataCopyPad explicitly zeros the last D/KV block.
        Duplicate(local, static_cast<Bf>(0.0f), square);
        Fence<HardEvent::V_MTE2>();
        for (uint32_t row = 0; row < src.rows; ++row) {
            DataCopyExtParams cp{1, src.cols * static_cast<uint32_t>(sizeof(Bf)), 0, 0, 0};
            DataCopyPadExtParams<Bf> pad{true, 0, static_cast<uint8_t>(Align(src.cols) - src.cols),
                                        static_cast<Bf>(0.0f)};
            DataCopyPad(local[row * b], src.gm[src.offset + row * src.stride], cp, pad);
        }
    }
    __aicore__ inline void Stage(uint32_t slot, const Block& left, const Block& right)
    {
        auto input = inputQueue.AllocTensor<Bf>();
        CopyBlock(input, left);
        CopyBlock(input[square], right);
        inputQueue.EnQue(input);
        input = inputQueue.DeQue<Bf>();
        Fence<HardEvent::MTE2_MTE3>();
        DataCopy(aSlot[slot], input, square);
        DataCopy(bSlot[slot], input[square], square);
        // Publish the operands before notifying the Cube service, and before reusing UB.
        Fence<HardEvent::MTE3_MTE2>();
        inputQueue.FreeTensor(input);
    }
    template <class MM> __aicore__ inline void Launch(MM& mm, uint32_t slot, bool ta, bool tb)
    {
        mm.SetOrgShape(b, b, b);
        mm.SetSingleShape(b, b, b);
        mm.SetTensorA(aSlot[slot], ta);
        mm.SetTensorB(bSlot[slot], tb);
        // Async GM output must request the completion event consumed by WaitIterateAll.
        mm.template IterateAll<false>(cSlot[slot], 0, false, true);
    }
    template <class MM> __aicore__ inline void Wait(MM& mm)
    {
        mm.WaitIterateAll();
        mm.End();
    }
    __aicore__ inline LocalTensor<float> Result(uint32_t slot)
    {
        auto result = resultQueue.AllocTensor<float>();
        DataCopy(result, cSlot[slot], square);
        resultQueue.EnQue(result);
        return resultQueue.DeQue<float>();
    }
    __aicore__ inline void Release(LocalTensor<float> result)
    {
        resultQueue.FreeTensor(result);
    }

    // The next operands are staged while Cube is running. After it completes,
    // launch the next product BEFORE consuming/accumulating the previous result.
    __aicore__ inline void Dot(LocalTensor<float> target, GlobalTensor<Bf>& left,
                              GlobalTensor<Bf>& right, uint64_t qt, uint64_t kt,
                              uint64_t qh, uint64_t kh, uint32_t qr, uint32_t kr)
    {
        Duplicate(target, 0.0f, square);
        Stage(0, Input(left, qt, qh, t.q_heads, qr, 0), Input(right, kt, kh, t.kv_heads, kr, 0));
        Launch(scoreMm, 0, false, true);
        uint32_t slot = 0;
        for (uint64_t d = 0; d < t.head_dim; d += b) {
            const bool next = d + b < t.head_dim;
            if (next) Stage(slot ^ 1, Input(left, qt, qh, t.q_heads, qr, d + b),
                            Input(right, kt, kh, t.kv_heads, kr, d + b));
            Wait(scoreMm);
            if (next) Launch(scoreMm, slot ^ 1, false, true);
            auto part = Result(slot);
            Add(target, target, part, square);
            PipeBarrier<PIPE_V>();
            Release(part);
            slot ^= 1;
        }
    }
    __aicore__ inline void SaveWeights(LocalTensor<float> src, uint32_t index)
    {
        auto bf = weightBuf.Get<Bf>();
        Cast(bf, src, RoundMode::CAST_RINT, square);
        Fence<HardEvent::V_MTE3>();
        DataCopy(weights[index], bf, square);
        Fence<HardEvent::MTE3_V>();
    }
    __aicore__ inline float Reduce(LocalTensor<float> row, bool maximum)
    {
        if (maximum) ReduceMax(reduce, row, reduce, b);
        else ReduceSum(reduce, row, reduce, b);
        Fence<HardEvent::V_S>();
        const float value = reduce.GetValue(0);
        Fence<HardEvent::S_V>();
        return value;
    }
    __aicore__ inline float Exponential(float x)
    {
        Duplicate(oldRow, x, 8);
        PipeBarrier<PIPE_V>();
        Exp(oldRow, oldRow, 8);
        Fence<HardEvent::V_S>();
        const float value = oldRow.GetValue(0);
        Fence<HardEvent::S_V>();
        return value;
    }
    __aicore__ inline void MaskTail(LocalTensor<float> row, uint32_t count, float value)
    {
        // Scalar tail stores support arbitrary causal boundaries; Vector row bases remain aligned.
        Fence<HardEvent::V_S>();
        for (uint32_t i = count; i < b; ++i) row.SetValue(i, value);
        Fence<HardEvent::S_V>();
    }
    __aicore__ inline void StartRows(uint32_t rows)
    {
        Duplicate(state, 0.0f, b * 16);
        Fence<HardEvent::V_S>();
        for (uint32_t r = 0; r < rows; ++r) state.SetValue(r * 16, -3.402823466e+38F);
        Fence<HardEvent::S_V>();
    }
    __aicore__ inline uint32_t Allowed(uint64_t qt, uint64_t kt, uint32_t kr)
    {
        return qt < kt ? 0 : static_cast<uint32_t>(Min(kr, qt - kt + 1));
    }
    __aicore__ inline void Softmax(uint64_t qt, uint64_t kt, uint32_t qr, uint32_t kr)
    {
        Muls(scores, scores, t.scale, square);
        PipeBarrier<PIPE_V>();
        Duplicate(probability, 0.0f, square);
        for (uint32_t r = 0; r < qr; ++r) {
            const uint32_t count = Allowed(qt + r, kt, kr);
            Fence<HardEvent::V_S>();
            if (!count) { state.SetValue(r * 16 + 2, 1.0f); continue; }
            float m = state.GetValue(r * 16), sum = state.GetValue(r * 16 + 1);
            auto row = scores[r * b];
            if (count < b) MaskTail(row, count, -3.402823466e+38F);
            PipeBarrier<PIPE_V>();
            const float tileMax = Reduce(row, true);
            const float nextMax = m > tileMax ? m : tileMax;
            const float alpha = sum == 0.0f ? 0.0f : Exponential(m - nextMax);
            auto p = probability[r * b];
            Adds(p, row, -nextMax, b);
            PipeBarrier<PIPE_V>();
            Exp(p, p, b);
            PipeBarrier<PIPE_V>();
            if (count < b) MaskTail(p, count, 0.0f);
            PipeBarrier<PIPE_V>();
            sum = sum * alpha + Reduce(p, false);
            state.SetValue(r * 16, nextMax);
            state.SetValue(r * 16 + 1, sum);
            state.SetValue(r * 16 + 2, alpha);
        }
        Fence<HardEvent::S_V>();
        SaveWeights(probability, 0);
    }
    __aicore__ inline void LoadStats(uint64_t qt, uint64_t qh, uint32_t qr)
    {
        Fence<HardEvent::S_MTE2>();
        for (uint32_t r = 0; r < qr; ++r) {
            DataCopyExtParams cp{1, sizeof(float), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            DataCopyPad(metadata[r * 16], lse[(qt + r) * t.q_heads + qh], cp, pad);
            DataCopy(metadata[b * 16 + r * 16], delta[((qt + r) * t.q_heads + qh) * 16], 16);
        }
        Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline void GradientWeights(uint64_t qt, uint64_t kt, uint32_t qr, uint32_t kr)
    {
        Muls(scores, scores, t.scale, square);
        PipeBarrier<PIPE_V>();
        Duplicate(probability, 0.0f, square);
        Duplicate(ds, 0.0f, square);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < qr; ++r) {
            const uint32_t count = Allowed(qt + r, kt, kr);
            if (!count) continue;
            auto p = probability[r * b];
            Adds(p, scores[r * b], -metadata.GetValue(r * 16), b);
            PipeBarrier<PIPE_V>();
            Exp(p, p, b);
            PipeBarrier<PIPE_V>();
            if (count < b) MaskTail(p, count, 0.0f);
            Adds(ds[r * b], dp[r * b], -metadata.GetValue(b * 16 + r * 16), b);
            PipeBarrier<PIPE_V>();
            Mul(ds[r * b], ds[r * b], p, b);
            PipeBarrier<PIPE_V>();
            Muls(ds[r * b], ds[r * b], t.scale, b);
        }
        PipeBarrier<PIPE_V>();
        SaveWeights(probability, 0);
        SaveWeights(ds, 1);
    }
    __aicore__ inline void Update(LocalTensor<float> part, GlobalTensor<float>& target,
                                 uint64_t token, uint64_t head, uint64_t heads,
                                 uint32_t rows, uint64_t d, bool first, bool rescale)
    {
        const uint32_t n = static_cast<uint32_t>(Min(b, t.padded_dim - d));
        for (uint32_t r = 0; r < rows; ++r) {
            const uint64_t offset = ((token + r) * heads + head) * t.padded_dim + d;
            if (!first) {
                DataCopy(oldRow, target[offset], n);
                Fence<HardEvent::MTE2_V>();
                if (rescale) {
                    Muls(oldRow, oldRow, state.GetValue(r * 16 + 2), n);
                    PipeBarrier<PIPE_V>();
                }
                Add(part[r * b], part[r * b], oldRow, n);
                Fence<HardEvent::V_MTE2>();
            }
            Fence<HardEvent::V_MTE3>();
            DataCopy(target[offset], part[r * b], n);
        }
        Fence<HardEvent::MTE3_V>();
    }
    template <class MM> __aicore__ inline void Apply(MM& mm, uint32_t weight,
        GlobalTensor<Bf>& right, uint64_t rt, uint64_t rh, uint64_t rightHeads, uint32_t rr,
        GlobalTensor<float>& target, uint64_t ot, uint64_t oh, uint64_t outHeads,
        uint32_t outRows, bool first, bool transpose, bool rescale = false)
    {
        Stage(0, Weight(weight), Input(right, rt, rh, rightHeads, rr, 0));
        Launch(mm, 0, transpose, false);
        uint32_t slot = 0;
        for (uint64_t d = 0; d < t.head_dim; d += b) {
            const bool next = d + b < t.head_dim;
            if (next) Stage(slot ^ 1, Weight(weight), Input(right, rt, rh, rightHeads, rr, d + b));
            Wait(mm);
            if (next) Launch(mm, slot ^ 1, transpose, false);
            auto part = Result(slot);
            Update(part, target, ot, oh, outHeads, outRows, d, first, rescale);
            Release(part);
            slot ^= 1;
        }
    }
    __aicore__ inline void FinishRows(uint64_t qt, uint64_t qh, uint32_t qr)
    {
        for (uint32_t r = 0; r < qr; ++r) {
            const float sum = state.GetValue(r * 16 + 1);
            Duplicate(oldRow, sum, 8);
            PipeBarrier<PIPE_V>();
            Ln(oldRow, oldRow, 8);
            Fence<HardEvent::V_S>();
            state.SetValue(r * 16, state.GetValue(r * 16) + oldRow.GetValue(0));
            Fence<HardEvent::S_MTE2>();
            for (uint64_t d = 0; d < t.padded_dim; d += b) {
                const uint32_t n = static_cast<uint32_t>(Min(b, t.padded_dim - d));
                const uint64_t off = ((qt + r) * t.q_heads + qh) * t.padded_dim + d;
                DataCopy(oldRow, output[off], n);
                Fence<HardEvent::MTE2_V>();
                Muls(oldRow, oldRow, 1.0f / sum, n);
                Fence<HardEvent::V_MTE3>();
                DataCopy(output[off], oldRow, n);
                Fence<HardEvent::MTE3_MTE2>();
            }
            Fence<HardEvent::S_MTE3>();
            DataCopy(lse[((qt + r) * t.q_heads + qh) * 16], state[r * 16], 16);
        }
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::MTE3_S>();
    }
    uint32_t b = 0, square = 0;
private:
    TQue<QuePosition::VECIN, 2> inputQueue, resultQueue;
    TBuf<QuePosition::VECCALC> scoreBuf, dpBuf, probBuf, dsBuf, weightBuf;
    TBuf<QuePosition::VECCALC> stateBuf, metaBuf, reduceBuf, oldBuf;
    GlobalTensor<Bf> aSlot[2], bSlot[2], weights[2];
    GlobalTensor<float> cSlot[2];
};
}
#endif
