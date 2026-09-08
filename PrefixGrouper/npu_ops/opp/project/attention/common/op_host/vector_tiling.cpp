#include "attention_tiling.h"
#include "shape_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace shared_prefix_host {
ge::graphStatus VectorTiling(gert::TilingContext* context, bool pack)
{
    auto* shape = context->GetInputShape(0);
    if (!shape || !context->GetPlatformInfo()) return ge::GRAPH_FAILED;
    const auto& s = shape->GetStorageShape();
    if (s.GetDimNum() != 3 || s.GetDim(0) <= 0 || s.GetDim(1) <= 0 || s.GetDim(2) <= 0 ||
        s.GetDim(2) > INT64_MAX - 15) return ge::GRAPH_FAILED;
    auto* t = context->GetTilingData<SharedPrefixVectorTilingData>();
    *t = {};
    if (!shared_prefix_host::Multiply(s.GetDim(0), s.GetDim(1), t->rows)) return ge::GRAPH_FAILED;
    t->padded_dim = s.GetDim(2);
    t->head_dim = s.GetDim(2);
    if (pack) {
        const auto* attrs = context->GetAttrs();
        const auto* dim = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
        if (!dim || *dim <= 0 || *dim > s.GetDim(2) ||
            shared_prefix_host::Align(*dim) != static_cast<uint64_t>(s.GetDim(2))) return ge::GRAPH_FAILED;
        t->head_dim = *dim;
    }
    uint64_t elements;
    if (!shared_prefix_host::Multiply(t->rows, t->head_dim, elements)) return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    if (ub < 16 * 1024 || !platform.GetCoreNumAiv()) return ge::GRAPH_FAILED;
    // Pack includes two worst-case D=1 padded spans, gather offsets, FP32 and BF16 output.
    const uint64_t budget = (ub - 4096) / (pack ? 144 : 24);
    t->tile = std::min<uint64_t>(budget / 32 * 32, UINT16_MAX / 32 * 32);
    if (!t->tile) return ge::GRAPH_FAILED;
    t->tasks = pack ? (elements - 1) / t->tile + 1 : t->rows;
    t->cores = std::min<uint64_t>(platform.GetCoreNumAiv(), t->tasks);
    context->SetBlockDim(t->cores);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}
