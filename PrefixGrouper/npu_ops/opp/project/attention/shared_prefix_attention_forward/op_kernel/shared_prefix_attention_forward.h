#ifndef PREFIX_GROUPER_SHARED_PREFIX_ATTENTION_FORWARD_H
#define PREFIX_GROUPER_SHARED_PREFIX_ATTENTION_FORWARD_H
#include "../../common/op_kernel/attention_common.h"
#include "shared_prefix_attention_forward_tiling_data.h"
#include "lib/activation/softmaxflashv2.h"

namespace shared_prefix {
// CANN 9.0 FA schedule: BMM1(i), Vec1(i-1), BMM2(i-1), Vec2(i-2).
// Contexts span KV intervals AND query tasks; each owns its output rescale state.
class SharedPrefixAttentionForwardKernel {
public:
    TPipe pipe;
    ScoreMatmul scoreMm;
    ValueMatmul valueMm;
    __aicore__ inline void Init(
        GM_ADDR qAddr, GM_ADDR kAddr, GM_ADDR vAddr,
        GM_ADDR prefixStartAddr, GM_ADDR prefixEndAddr, GM_ADDR sequenceStartAddr,
        GM_ADDR sequenceEndAddr, GM_ADDR groupEndAddr, GM_ADDR outAddr,
        GM_ADDR lseAddr, GM_ADDR workspaceAddr,
        const SharedPrefixAttentionForwardTilingData& tilingData)
    {
        t = tilingData;
        q.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(qAddr));
        k.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(kAddr));
        v.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(vAddr));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(outAddr));
        lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lseAddr));
        prefixStart.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefixStartAddr));
        prefixEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefixEndAddr));
        sequenceStart.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequenceStartAddr));
        sequenceEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequenceEndAddr));
        elements = t.q_tile * t.kv_tile;
        statsSize = t.q_tile * 8;
        pipe.InitBuffer(scoreBuf, elements * sizeof(float));
        pipe.InitBuffer(probBuf, elements * sizeof(Bf));
        pipe.InitBuffer(statsBuf, 9 * statsSize * sizeof(float));
        pipe.InitBuffer(softmaxBuf, t.softmax_tmp_bytes);
        const uint32_t indexCount = (t.kv_tile + 63) / 64 * 64;
        pipe.InitBuffer(indexBuf, indexCount * sizeof(float));
        pipe.InitBuffer(maskBuf, 32);
        pipe.InitBuffer(outputBuf, 2 * t.q_tile * outputChunk * sizeof(float));
        scores = scoreBuf.Get<float>();
        probability = probBuf.Get<Bf>();
        stats = statsBuf.Get<float>();
        indices = indexBuf.Get<float>();
        mask = maskBuf.Get<uint8_t>();
        part = outputBuf.Get<float>();
        old = part[t.q_tile * outputChunk];
        CreateVecIndex(indices, 0.0f, indexCount);
        const uint64_t slotBytes = t.q_tile * (t.kv_tile * 6ULL + t.padded_dim * 4ULL);
        auto* base = reinterpret_cast<__gm__ uint8_t*>(GetUserWorkspace(workspaceAddr)) +
                     GetBlockIdx() * t.core_workspace_bytes;
        for (uint32_t s = 0; s < 3; ++s) {
            scoreGm[s].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base + s * slotBytes));
            probGm[s].SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(base + s * slotBytes + elements * 4));
            valueGm[s].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base + s * slotBytes + elements * 6));
        }
    }

    __aicore__ inline void Process()
    {
        const auto& range = t.ranges[GetBlockIdx()];
        uint64_t qt = range.query_start, qh = range.query_head;
        for (uint64_t task = 0; task < range.task_count; ++task) {
            const uint64_t kh = qh / (t.q_heads / t.kv_heads);
            const uint64_t segmentEnd = Min(qt + t.q_tile, sequenceEnd.GetValue(qt));
            const uint32_t qr = segmentEnd - qt;
            const uint64_t ps = prefixStart.GetValue(qt), pe = prefixEnd.GetValue(qt);
            const uint64_t ss = sequenceStart.GetValue(qt);
            bool first = true;
            const uint32_t intervals = ss == ps ? 1 : 2;
            for (uint32_t interval = 0; interval < intervals; ++interval) {
                const uint64_t begin = interval == 0 ? ps : ss;
                const uint64_t limit = ss == ps || interval == 1 ? segmentEnd : pe;
                for (uint64_t kt = begin; kt < limit; kt += t.kv_tile) {
                    const uint32_t kr = Min(t.kv_tile, limit - kt);
                    Push({qt, kt, qh, kh, qr, kr, first,
                          interval + 1 == intervals && kt + kr == limit});
                    first = false;
                }
            }
            if (++qh == t.q_heads) {
                qh = 0;
                qt = segmentEnd;
            }
        }
        // Drain only stages still in flight, without dummy Matmul requests.
        if (issued) {
            Wait(scoreMm);
            Vec1((issued - 1) % 3);
            if (issued > 1) Wait(valueMm);
            Bmm2((issued - 1) % 3);
            if (issued > 1) Vec2((issued - 2) % 3);
            Wait(valueMm);
            Vec2((issued - 1) % 3);
        }
    }

private:
    struct Context {
        uint64_t qt, kt, qh, kh;
        uint32_t qr, kr;
        bool first, last;
    } contexts[3];
    template <class MM> __aicore__ inline void Wait(MM& mm)
    {
        mm.WaitIterateAll();
        mm.End();
    }
    __aicore__ inline void Push(const Context& ctx)
    {
        if (issued) Wait(scoreMm);
        const uint32_t s = issued % 3;
        contexts[s] = ctx;
        Bmm1(s);
        if (issued) Vec1((issued - 1) % 3);
        if (issued > 1) Wait(valueMm);
        if (issued) Bmm2((issued - 1) % 3);
        if (issued > 1) Vec2((issued - 2) % 3);
        ++issued;
    }
    __aicore__ inline void Bmm1(uint32_t s)
    {
        const auto& c = contexts[s];
        // Independent A/B strides read TND/GQA directly, including nonaligned D.
        scoreMm.SetOrgShape(c.qr, t.kv_heads * t.head_dim,
                            t.q_heads * t.head_dim, t.kv_heads * t.head_dim, t.kv_tile);
        scoreMm.SetTensorA(q[(c.qt * t.q_heads + c.qh) * t.head_dim]);
        scoreMm.SetTensorB(k[(c.kt * t.kv_heads + c.kh) * t.head_dim], true);
        scoreMm.SetTail(c.qr, c.kr, t.head_dim);
        scoreMm.IterateAll<false>(scoreGm[s], 0, false, true);
    }
    __aicore__ inline void Bmm2(uint32_t s)
    {
        const auto& c = contexts[s];
        Fence<HardEvent::MTE3_MTE2>();
        valueMm.SetOrgShape(c.qr, t.kv_heads * t.head_dim,
                            t.kv_tile, t.kv_heads * t.head_dim, t.padded_dim);
        valueMm.SetTensorA(probGm[s]);
        valueMm.SetTensorB(v[(c.kt * t.kv_heads + c.kh) * t.head_dim]);
        valueMm.SetTail(c.qr, t.head_dim, c.kr);
        valueMm.IterateAll<false>(valueGm[s], 0, false, true);
    }
    __aicore__ inline LocalTensor<float> Sum(uint32_t s) { return stats[s * 3 * statsSize]; }
    __aicore__ inline LocalTensor<float> Max(uint32_t s) { return stats[(s * 3 + 1) * statsSize]; }
    __aicore__ inline LocalTensor<float> Alpha(uint32_t s) { return stats[(s * 3 + 2) * statsSize]; }
    __aicore__ inline void Vec1(uint32_t s)
    {
        const auto& c = contexts[s];
        const uint32_t columns = (c.kr + 15) / 16 * 16;
        const uint32_t count = c.qr * columns;
        Duplicate(scores, 0.0f, count);
        Fence<HardEvent::V_MTE2>();
        const uint32_t aligned = (c.kr + 7) / 8 * 8;
        DataCopyExtParams cp{static_cast<uint16_t>(c.qr), c.kr * 4,
                             (t.kv_tile - c.kr) * 4, (columns - aligned) / 8, 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(aligned - c.kr), 0.0f};
        DataCopyPad(scores, scoreGm[s], cp, pad);
        Fence<HardEvent::MTE2_V>();
        Muls(scores, scores, t.scale, count);
        PipeBarrier<PIPE_V>();
        // Vector masks replace per-element scalar causal-tail stores.
        for (uint32_t r = 0; r < c.qr; ++r) {
            const uint32_t valid = c.qt + r < c.kt ? 0 : Min(c.kr, c.qt + r - c.kt + 1);
            if (valid == columns) continue;
            // CompareScalar's count must cover whole 256-byte repeats on 910B.
            // Select still consumes only the live columns of the padded KV tail.
            CompareScalar(mask, indices, static_cast<float>(static_cast<int32_t>(valid)), CMPMODE::LT,
                          (columns + 63) / 64 * 64);
            PipeBarrier<PIPE_V>();
            Select(scores[r * columns], mask, scores[r * columns], -3.402823466e+38F,
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, columns);
            PipeBarrier<PIPE_V>();
        }
        auto tmp = softmaxBuf.Get<uint8_t>();
        // Match FA's tail-aware device tiling and use the available API scratch.
        const SoftMaxShapeInfo shape{c.qr, columns, c.qr, c.kr};
        const auto softmaxTiling = SoftMaxFlashV2TilingFunc(
            shape, sizeof(float), sizeof(float), t.softmax_tmp_bytes, !c.first, false);
        if (c.first) {
            SoftmaxFlashV2<float, false, true, false>(scores, Sum(s), Max(s), scores, Alpha(s),
                                                     Sum(s), Max(s), tmp, softmaxTiling, shape);
        } else {
            const uint32_t prev = (s + 2) % 3;
            SoftmaxFlashV2<float, true, true, false>(scores, Sum(s), Max(s), scores, Alpha(s),
                                                    Sum(prev), Max(prev), tmp, softmaxTiling, shape);
        }
        PipeBarrier<PIPE_V>();
        Cast(probability, scores, RoundMode::CAST_RINT, count);
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams outCopy{static_cast<uint16_t>(c.qr), columns * 2, 0,
                                  (t.kv_tile - columns) * 2, 0};
        DataCopyPad(probGm[s], probability, outCopy);
        Fence<HardEvent::MTE3_V>();
    }
    // A DMA descriptor spans rows unless its byte stride exceeds the ISA field.
    __aicore__ inline void ReadRows(LocalTensor<float> dst, GlobalTensor<float> src,
                                    uint32_t rows, uint32_t width, uint64_t stride)
    {
        const uint32_t batch = stride - width <= UINT32_MAX / 4 ? rows : 1;
        const uint32_t aligned = (width + 7) / 8 * 8;
        for (uint32_t r = 0; r < rows; r += batch) {
            DataCopyExtParams cp{static_cast<uint16_t>(batch), width * 4,
                                 batch > 1 ? static_cast<uint32_t>((stride - width) * 4) : 0,
                                 (outputChunk - aligned) / 8, 0};
            DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(aligned - width), 0.0f};
            DataCopyPad(dst[r * outputChunk], src[r * stride], cp, pad);
        }
    }
    template <class T>
    __aicore__ inline void WriteRows(GlobalTensor<T> dst, LocalTensor<T> src,
                                     uint32_t rows, uint32_t width, uint64_t stride,
                                     uint32_t localStride)
    {
        constexpr uint32_t bytes = sizeof(T), perBlock = 32 / bytes;
        const uint32_t batch = stride - width <= UINT32_MAX / bytes ? rows : 1;
        for (uint32_t r = 0; r < rows; r += batch) {
            // UB rounds each block to 32 bytes; GM writes only width elements.
            DataCopyExtParams cp{static_cast<uint16_t>(batch), width * bytes, (localStride - width) / perBlock,
                                 batch > 1 ? static_cast<uint32_t>((stride - width) * bytes) : 0, 0};
            DataCopyPad(dst[r * stride], src[r * localStride], cp);
        }
    }
    __aicore__ inline void Vec2(uint32_t s)
    {
        const auto& c = contexts[s];
        auto alpha = Alpha(s), sum = Sum(s), maximum = Max(s);
        const uint64_t stride = t.q_heads * t.head_dim;
        // Like FA's Vec2, broadcast an eight-float statistic block per row.
        BinaryRepeatParams rescale;
        rescale.src0BlkStride = 0;
        rescale.src0RepStride = 1;
        rescale.src1RepStride = outputChunk / 8;
        rescale.dstRepStride = outputChunk / 8;
        BinaryRepeatParams normalize;
        normalize.src0RepStride = outputChunk / 8;
        normalize.src1BlkStride = 0;
        normalize.src1RepStride = 1;
        normalize.dstRepStride = outputChunk / 8;
        for (uint64_t d = 0; d < t.padded_dim; d += outputChunk) {
            const uint32_t n = Min(outputChunk, t.padded_dim - d);
            const uint32_t valid = d < t.head_dim ? Min(n, t.head_dim - d) : 0;
            const uint64_t off = (c.qt * t.q_heads + c.qh) * t.head_dim + d;
            Duplicate(part, 0.0f, c.qr * outputChunk);
            Fence<HardEvent::V_MTE2>();
            if (valid) ReadRows(part, valueGm[s][d], c.qr, valid, t.padded_dim);
            // At Vec2(s), BMM2 is writing s+1. The previous accumulator in
            // s-1 is read before BMM2 reuses that slot on the next iteration.
            if (!c.first) ReadRows(old, valueGm[(s + 2) % 3][d], c.qr, n, t.padded_dim);
            Fence<HardEvent::MTE2_V>();
            if (!c.first) {
                Mul(old, alpha, old, n, c.qr, rescale);
                PipeBarrier<PIPE_V>();
                Add(part, part, old, n, c.qr,
                    {1, 1, 1, outputChunk / 8, outputChunk / 8, outputChunk / 8});
                PipeBarrier<PIPE_V>();
            }
            if (c.last) {
                Div(part, part, sum, n, c.qr, normalize);
                PipeBarrier<PIPE_V>();
                // The old FP32 values are dead after Add. Reuse their UB for
                // the final BF16 rows, without reserving another cast buffer.
                auto packed = old.ReinterpretCast<Bf>();
                Cast(packed, part, RoundMode::CAST_RINT, c.qr * outputChunk);
                Fence<HardEvent::V_MTE3>();
                WriteRows(output[off], packed, c.qr, valid, stride, outputChunk);
            } else {
                Fence<HardEvent::V_MTE3>();
                WriteRows(valueGm[s][d], part, c.qr, n, t.padded_dim, outputChunk);
            }
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }
        if (c.last) {
            // The next query's Softmax can run before this task drains.
            Ln(sum, sum, c.qr * 8);
            PipeBarrier<PIPE_V>();
            Add(sum, sum, maximum, c.qr * 8);
            Fence<HardEvent::V_MTE3>();
            // Narrow DMA writes, not cached scalar GM stores: exactly one
            // FP32 value per owned (query, head), including interleaved heads.
            WriteRows(lse[c.qt * t.q_heads + c.qh], sum, c.qr, 1, t.q_heads, 8);
            Fence<HardEvent::MTE3_V>();
        }
    }
    static constexpr uint32_t outputChunk = 64;
    SharedPrefixAttentionForwardTilingData t;
    uint64_t issued = 0;
    uint32_t elements = 0, statsSize = 0;
    GlobalTensor<Bf> q, k, v, output, probGm[3];
    GlobalTensor<float> lse, scoreGm[3], valueGm[3];
    GlobalTensor<int32_t> prefixStart, prefixEnd, sequenceStart, sequenceEnd;
    TBuf<QuePosition::VECCALC> scoreBuf, probBuf, statsBuf, softmaxBuf, indexBuf, maskBuf, outputBuf;
    LocalTensor<float> scores, stats, indices, part, old;
    LocalTensor<Bf> probability;
    LocalTensor<uint8_t> mask;
};
}
#endif
