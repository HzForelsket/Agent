#include "shared_prefix_attention_host.h"

namespace {
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
ge::graphStatus PackShape(gert::InferShapeContext* context)
{
    auto* input = context->GetInputShape(0);
    auto* output = context->GetOutputShape(0);
    auto* attrs = context->GetAttrs();
    auto* dim = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    if (!input || !output || !dim || input->GetDimNum() != 3 || *dim <= 0) return ge::GRAPH_FAILED;
    *output = *input;
    output->SetDim(2, *dim);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus DeltaShape(gert::InferShapeContext* context)
{
    auto* input = context->GetInputShape(0);
    auto* output = context->GetOutputShape(0);
    if (!input || !output || input->GetDimNum() != 3) return ge::GRAPH_FAILED;
    *output = *input;
    output->SetDim(2, kSharedPrefixRowAlignment);
    return ge::GRAPH_SUCCESS;
}
}
namespace ops {
class SharedPrefixAttentionPack : public OpDef {
public:
    explicit SharedPrefixAttentionPack(const char* name) : OpDef(name)
    {
        Input("accumulator").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Attr("head_dim").Int();
        SetInferShape(PackShape).SetInferDataType([](gert::InferDataTypeContext* c) {
            c->SetOutputDataType(0, ge::DT_BF16); return ge::GRAPH_SUCCESS;
        });
        AICore().SetTiling([](gert::TilingContext* c) { return VectorTiling(c, true); }).AddConfig("ascend910b");
    }
};
class SharedPrefixAttentionDelta : public OpDef {
public:
    explicit SharedPrefixAttentionDelta(const char* name) : OpDef(name)
    {
        Input("attention").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Input("grad_out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("delta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        SetInferShape(DeltaShape).SetInferDataType([](gert::InferDataTypeContext* c) {
            c->SetOutputDataType(0, ge::DT_FLOAT); return ge::GRAPH_SUCCESS;
        });
        AICore().SetTiling([](gert::TilingContext* c) { return VectorTiling(c, false); }).AddConfig("ascend910b");
    }
};
OP_ADD(SharedPrefixAttentionPack);
OP_ADD(SharedPrefixAttentionDelta);
}
