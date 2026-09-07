#include "shared_prefix_attention_common.h"
using namespace shared_prefix;

extern "C" __global__ __aicore__ void shared_prefix_attention_backward(
    GM_ADDR grad_out, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR delta, GM_ADDR lse,
    GM_ADDR prefix_start, GM_ADDR prefix_end, GM_ADDR sequence_start, GM_ADDR sequence_end,
    GM_ADDR group_end, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(SharedPrefixAttentionTilingData);
    GET_TILING_DATA(t, tiling);
    Attention a;
    REGIST_MATMUL_OBJ(&a.pipe, GetSysWorkSpacePtr(), a.scoreMm, &t.score_mm,
                      a.valueMm, &t.value_mm, a.transposeMm, &t.transpose_mm);
    a.Init(t, GetUserWorkspace(workspace));
    a.q.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(q));
    a.k.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(k));
    a.v.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(v));
    a.grad.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(grad_out));
    a.delta.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(delta));
    a.lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
    a.output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dq));
    a.dk.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dk));
    a.dv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dv));
    a.prefixStart.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefix_start));
    a.prefixEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefix_end));
    a.sequenceStart.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequence_start));
    a.sequenceEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequence_end));
    a.groupEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(group_end));
    const uint64_t qTasks = ((t.total_tokens + a.b - 1) / a.b) * t.q_heads;
    for (uint64_t task = GetBlockIdx(); task < t.task_count; task += t.vector_cores) {
        if (task < qTasks) {
            const uint64_t qh = task % t.q_heads, kh = qh / (t.q_heads / t.kv_heads);
            const uint64_t end = Min((task / t.q_heads + 1) * a.b, t.total_tokens);
            for (uint64_t qt = task / t.q_heads * a.b; qt < end;) {
                const uint64_t segmentEnd = Min(end, a.sequenceEnd.GetValue(qt));
                const uint32_t qr = segmentEnd - qt;
                a.LoadStats(qt, qh, qr);
                bool first = true;
                const uint64_t ps = a.prefixStart.GetValue(qt), pe = a.prefixEnd.GetValue(qt);
                const uint64_t ss = a.sequenceStart.GetValue(qt);
                for (uint32_t interval = 0; interval < (ss == ps ? 1U : 2U); ++interval) {
                    const uint64_t begin = interval == 0 ? ps : ss;
                    const uint64_t limit = ss == ps || interval == 1 ? segmentEnd : pe;
                    for (uint64_t kt = begin; kt < limit; kt += a.b) {
                        const uint32_t kr = Min(a.b, limit - kt);
                        a.Dot(a.scores, a.q, a.k, qt, kt, qh, kh, qr, kr);
                        a.Dot(a.dp, a.grad, a.v, qt, kt, qh, kh, qr, kr);
                        a.GradientWeights(qt, kt, qr, kr);
                        a.Apply(a.valueMm, 1, a.k, kt, kh, t.kv_heads, kr,
                                a.output, qt, qh, t.q_heads, qr, first, false);
                        first = false;
                    }
                }
                qt = segmentEnd;
            }
        } else {
            const uint64_t job = task - qTasks, kh = job % t.kv_heads;
            const uint64_t end = Min((job / t.kv_heads + 1) * a.b, t.total_tokens);
            for (uint64_t kt = job / t.kv_heads * a.b; kt < end;) {
                const uint64_t segmentEnd = Min(end, a.sequenceEnd.GetValue(kt));
                const uint32_t kr = segmentEnd - kt;
                const bool prefix = a.prefixStart.GetValue(kt) == a.sequenceStart.GetValue(kt);
                const uint64_t queryEnd = prefix ? a.groupEnd.GetValue(kt) : a.sequenceEnd.GetValue(kt);
                bool first = true;
                // One owner reduces all visible queries and GQA heads for this KV block.
                for (uint64_t qt = kt; qt < queryEnd; qt += a.b) {
                    const uint32_t qr = Min(a.b, queryEnd - qt);
                    const uint64_t ratio = t.q_heads / t.kv_heads;
                    for (uint64_t qh = kh * ratio; qh < (kh + 1) * ratio; ++qh) {
                        a.LoadStats(qt, qh, qr);
                        a.Dot(a.scores, a.q, a.k, qt, kt, qh, kh, qr, kr);
                        a.Dot(a.dp, a.grad, a.v, qt, kt, qh, kh, qr, kr);
                        a.GradientWeights(qt, kt, qr, kr);
                        a.Apply(a.transposeMm, 1, a.q, qt, qh, t.q_heads, qr,
                                a.dk, kt, kh, t.kv_heads, kr, first, true);
                        a.Apply(a.transposeMm, 0, a.grad, qt, qh, t.q_heads, qr,
                                a.dv, kt, kh, t.kv_heads, kr, first, true);
                        first = false;
                    }
                }
                kt = segmentEnd;
            }
        }
    }
}
