#include "../../common/op_host/attention_tiling.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    return shared_prefix_host::AttentionTiling(context, false);
}
IMPL_OP_OPTILING(SharedPrefixAttentionForward).Tiling(TilingFunc);
}
