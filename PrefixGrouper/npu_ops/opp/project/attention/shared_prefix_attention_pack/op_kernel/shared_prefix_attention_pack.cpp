#include "shared_prefix_attention_pack.h"
using namespace shared_prefix;

extern "C" __global__ __aicore__ void shared_prefix_attention_pack(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SharedPrefixVectorTilingData);
    GET_TILING_DATA(t, tiling);
    SharedPrefixAttentionPackKernel kernel;
    kernel.Init(input, output, t);
    kernel.Process();
}
