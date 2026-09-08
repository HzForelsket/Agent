#include "../../common/op_host/attention_tiling.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    return shared_prefix_host::VectorTiling(context, true);
}
IMPL_OP_OPTILING(SharedPrefixAttentionPack).Tiling(TilingFunc);
}
