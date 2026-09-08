#ifndef PREFIX_GROUPER_SHARED_PREFIX_ATTENTION_FORWARD_H
#define PREFIX_GROUPER_SHARED_PREFIX_ATTENTION_FORWARD_H
#include "../../common/op_kernel/attention_common.h"

namespace shared_prefix {
class SharedPrefixAttentionForwardKernel : public AttentionBase<true> {
public:
    __aicore__ inline void Init(
        GM_ADDR qAddr, GM_ADDR kAddr, GM_ADDR vAddr,
        GM_ADDR prefixStartAddr, GM_ADDR prefixEndAddr, GM_ADDR sequenceStartAddr,
        GM_ADDR sequenceEndAddr, GM_ADDR groupEndAddr, GM_ADDR outAddr,
        GM_ADDR lseAddr, GM_ADDR workspaceAddr,
        const SharedPrefixAttentionTilingData& tilingData)
    {
        InitBuffers(tilingData, GetUserWorkspace(workspaceAddr));
        q.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(qAddr));
        k.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(kAddr));
        v.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(vAddr));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outAddr));
        lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lseAddr));
        prefixStart.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefixStartAddr));
        prefixEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefixEndAddr));
        sequenceStart.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequenceStartAddr));
        sequenceEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequenceEndAddr));
        groupEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groupEndAddr));
    }

    __aicore__ inline void Process()
    {
        for (uint64_t task = GetBlockIdx(); task < t.task_count; task += t.vector_cores) {
            const uint64_t qh = task % t.q_heads, kh = qh / (t.q_heads / t.kv_heads);
            const uint64_t end = Min((task / t.q_heads + 1) * b, t.total_tokens);
            for (uint64_t qt = task / t.q_heads * b; qt < end;) {
                const uint64_t segmentEnd = Min(end, sequenceEnd.GetValue(qt));
                const uint32_t qr = segmentEnd - qt;
                StartRows(qr);
                bool first = true;
                const uint64_t ps = prefixStart.GetValue(qt), pe = prefixEnd.GetValue(qt);
                const uint64_t ss = sequenceStart.GetValue(qt);
                // Prefix queries have one causal interval; suffix queries have two disjoint intervals.
                for (uint32_t interval = 0; interval < (ss == ps ? 1U : 2U); ++interval) {
                    const uint64_t begin = interval == 0 ? ps : ss;
                    const uint64_t limit = ss == ps || interval == 1 ? segmentEnd : pe;
                    for (uint64_t kt = begin; kt < limit; kt += b) {
                        const uint32_t kr = Min(b, limit - kt);
                        Dot(scores, q, k, qt, kt, qh, kh, qr, kr);
                        Softmax(qt, kt, qr, kr);
                        Apply(valueMm, 0, v, kt, kh, t.kv_heads, kr,
                                output, qt, qh, t.q_heads, qr, first, false, true);
                        first = false;
                    }
                }
                FinishRows(qt, qh, qr);
                qt = segmentEnd;
            }
        }
    }

private:
    __aicore__ inline void StartRows(uint32_t rows)
    {
        Duplicate(state, 0.0f, b * 16);
        Fence<HardEvent::V_S>();
        for (uint32_t r = 0; r < rows; ++r) state.SetValue(r * 16, -3.402823466e+38F);
        Fence<HardEvent::S_V>();
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
};
}
#endif
