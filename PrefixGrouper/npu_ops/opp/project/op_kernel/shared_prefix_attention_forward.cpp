#include "shared_prefix_attention_common.h"
using namespace shared_prefix;

extern "C" __global__ __aicore__ void shared_prefix_attention_forward(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR prefix_start, GM_ADDR prefix_end,
    GM_ADDR sequence_start, GM_ADDR sequence_end, GM_ADDR group_end,
    GM_ADDR out, GM_ADDR lse, GM_ADDR workspace, GM_ADDR tiling)
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
    a.output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
    a.lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
    a.prefixStart.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefix_start));
    a.prefixEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefix_end));
    a.sequenceStart.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequence_start));
    a.sequenceEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequence_end));
    a.groupEnd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(group_end));
    for (uint64_t task = GetBlockIdx(); task < t.task_count; task += t.vector_cores) {
        const uint64_t qh = task % t.q_heads, kh = qh / (t.q_heads / t.kv_heads);
        const uint64_t end = Min((task / t.q_heads + 1) * a.b, t.total_tokens);
        for (uint64_t qt = task / t.q_heads * a.b; qt < end;) {
            const uint64_t segmentEnd = Min(end, a.sequenceEnd.GetValue(qt));
            const uint32_t qr = segmentEnd - qt;
            a.StartRows(qr);
            bool first = true;
            const uint64_t ps = a.prefixStart.GetValue(qt), pe = a.prefixEnd.GetValue(qt);
            const uint64_t ss = a.sequenceStart.GetValue(qt);
            // Prefix queries have one causal interval; suffix queries have two disjoint intervals.
            for (uint32_t interval = 0; interval < (ss == ps ? 1U : 2U); ++interval) {
                const uint64_t begin = interval == 0 ? ps : ss;
                const uint64_t limit = ss == ps || interval == 1 ? segmentEnd : pe;
                for (uint64_t kt = begin; kt < limit; kt += a.b) {
                    const uint32_t kr = Min(a.b, limit - kt);
                    a.Dot(a.scores, a.q, a.k, qt, kt, qh, kh, qr, kr);
                    a.Softmax(qt, kt, qr, kr);
                    a.Apply(a.valueMm, 0, a.v, kt, kh, t.kv_heads, kr,
                            a.output, qt, qh, t.q_heads, qr, first, false, true);
                    first = false;
                }
            }
            a.FinishRows(qt, qh, qr);
            qt = segmentEnd;
        }
    }
}
