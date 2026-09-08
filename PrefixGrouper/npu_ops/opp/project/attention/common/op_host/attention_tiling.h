#ifndef PREFIX_GROUPER_ATTENTION_TILING_H
#define PREFIX_GROUPER_ATTENTION_TILING_H
#include "register/op_impl_registry.h"

namespace shared_prefix_host {
ge::graphStatus AttentionTiling(gert::TilingContext* context, bool backward);
ge::graphStatus VectorTiling(gert::TilingContext* context, bool pack);
}
#endif
