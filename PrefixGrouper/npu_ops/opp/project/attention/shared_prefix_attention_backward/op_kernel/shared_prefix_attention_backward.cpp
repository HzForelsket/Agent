#include "shared_prefix_attention_backward.h"
using namespace shared_prefix;

extern "C" __global__ __aicore__ void shared_prefix_attention_backward(
    GM_ADDR grad_out, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR delta, GM_ADDR lse,
    GM_ADDR prefix_start, GM_ADDR prefix_end, GM_ADDR sequence_start, GM_ADDR sequence_end,
    GM_ADDR group_end, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(SharedPrefixAttentionTilingData);
    GET_TILING_DATA(t, tiling);
    SharedPrefixAttentionBackwardKernel op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.scoreMm, &t.score_mm,
                      op.valueMm, &t.value_mm, op.transposeMm, &t.transpose_mm);
    op.Init(grad_out, q, k, v, delta, lse, prefix_start, prefix_end, sequence_start, sequence_end, group_end, dq, dk, dv, workspace, t);
    op.Process();
}
