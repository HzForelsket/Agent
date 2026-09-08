#include "shared_prefix_attention_delta.h"
using namespace shared_prefix;

extern "C" __global__ __aicore__ void shared_prefix_attention_delta(
    GM_ADDR out, GM_ADDR grad, GM_ADDR delta, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SharedPrefixVectorTilingData);
    GET_TILING_DATA(t, tiling);
    SharedPrefixAttentionDeltaKernel kernel;
    kernel.Init(out, grad, delta, t);
    kernel.Process();
}
