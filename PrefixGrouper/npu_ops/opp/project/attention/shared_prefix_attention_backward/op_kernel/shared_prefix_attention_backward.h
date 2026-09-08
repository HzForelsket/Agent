#ifndef PREFIX_GROUPER_SHARED_PREFIX_ATTENTION_BACKWARD_H
#define PREFIX_GROUPER_SHARED_PREFIX_ATTENTION_BACKWARD_H
#include "../../common/op_kernel/attention_common.h"

namespace shared_prefix {
class SharedPrefixAttentionBackwardKernel : public AttentionBase<false> {
public:
    __aicore__ inline void Init(
        GM_ADDR gradOutAddr, GM_ADDR qAddr, GM_ADDR kAddr,
        GM_ADDR vAddr, GM_ADDR deltaAddr, GM_ADDR lseAddr,
        GM_ADDR prefixStartAddr, GM_ADDR prefixEndAddr, GM_ADDR sequenceStartAddr,
        GM_ADDR sequenceEndAddr, GM_ADDR groupEndAddr, GM_ADDR dqAddr,
        GM_ADDR dkAddr, GM_ADDR dvAddr, GM_ADDR workspaceAddr,
        const SharedPrefixAttentionTilingData& tilingData)
    {
        InitBuffers(tilingData, GetUserWorkspace(workspaceAddr));
        q.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(qAddr));
        k.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(kAddr));
        v.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(vAddr));
        grad.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(gradOutAddr));
        delta.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(deltaAddr));
        lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lseAddr));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dqAddr));
        dk.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dkAddr));
        dv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dvAddr));
        prefixStart.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefixStartAddr));
        prefixEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefixEndAddr));
        sequenceStart.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequenceStartAddr));
        sequenceEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequenceEndAddr));
        groupEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groupEndAddr));
    }

    __aicore__ inline void Process()
    {
        const uint64_t qTasks = ((t.total_tokens + b - 1) / b) * t.q_heads;
        for (uint64_t task = GetBlockIdx(); task < t.task_count; task += t.vector_cores) {
            if (task < qTasks) {
                const uint64_t qh = task % t.q_heads, kh = qh / (t.q_heads / t.kv_heads);
                const uint64_t end = Min((task / t.q_heads + 1) * b, t.total_tokens);
                for (uint64_t qt = task / t.q_heads * b; qt < end;) {
                    const uint64_t segmentEnd = Min(end, sequenceEnd.GetValue(qt));
                    const uint32_t qr = segmentEnd - qt;
                    LoadStats(qt, qh, qr);
                    bool first = true;
                    const uint64_t ps = prefixStart.GetValue(qt), pe = prefixEnd.GetValue(qt);
                    const uint64_t ss = sequenceStart.GetValue(qt);
                    for (uint32_t interval = 0; interval < (ss == ps ? 1U : 2U); ++interval) {
                        const uint64_t begin = interval == 0 ? ps : ss;
                        const uint64_t limit = ss == ps || interval == 1 ? segmentEnd : pe;
                        for (uint64_t kt = begin; kt < limit; kt += b) {
                            const uint32_t kr = Min(b, limit - kt);
                            Dot(scores, q, k, qt, kt, qh, kh, qr, kr);
                            Dot(dp, grad, v, qt, kt, qh, kh, qr, kr);
                            GradientWeights(qt, kt, qr, kr);
                            Apply(valueMm, 1, k, kt, kh, t.kv_heads, kr,
                                    output, qt, qh, t.q_heads, qr, first, false);
                            first = false;
                        }
                    }
                    qt = segmentEnd;
                }
            } else {
                const uint64_t job = task - qTasks, kh = job % t.kv_heads;
                const uint64_t end = Min((job / t.kv_heads + 1) * b, t.total_tokens);
                for (uint64_t kt = job / t.kv_heads * b; kt < end;) {
                    const uint64_t segmentEnd = Min(end, sequenceEnd.GetValue(kt));
                    const uint32_t kr = segmentEnd - kt;
                    const bool prefix = prefixStart.GetValue(kt) == sequenceStart.GetValue(kt);
                    const uint64_t queryEnd = prefix ? groupEnd.GetValue(kt) : sequenceEnd.GetValue(kt);
                    bool first = true;
                    // One owner reduces all visible queries and GQA heads for this KV block.
                    for (uint64_t qt = kt; qt < queryEnd; qt += b) {
                        const uint32_t qr = Min(b, queryEnd - qt);
                        const uint64_t ratio = t.q_heads / t.kv_heads;
                        for (uint64_t qh = kh * ratio; qh < (kh + 1) * ratio; ++qh) {
                            LoadStats(qt, qh, qr);
                            Dot(scores, q, k, qt, kt, qh, kh, qr, kr);
                            Dot(dp, grad, v, qt, kt, qh, kh, qr, kr);
                            GradientWeights(qt, kt, qr, kr);
                            Apply(transposeMm, 1, q, qt, qh, t.q_heads, qr,
                                    dk, kt, kh, t.kv_heads, kr, first, true);
                            Apply(transposeMm, 0, grad, qt, qh, t.q_heads, qr,
                                    dv, kt, kh, t.kv_heads, kr, first, true);
                            first = false;
                        }
                    }
                    kt = segmentEnd;
                }
            }
        }
    }

private:
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
};
}
#endif
