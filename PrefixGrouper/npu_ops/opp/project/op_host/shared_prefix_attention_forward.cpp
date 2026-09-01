#include "../op_kernel/shared_prefix_attention_tiling.h"
#include "register/op_def_registry.h"

#include <algorithm>
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* q_shape = context->GetInputShape(0);
    const gert::StorageShape* k_shape = context->GetInputShape(1);
    const auto* attrs = context->GetAttrs();
    if (q_shape == nullptr || k_shape == nullptr || attrs == nullptr ||
        q_shape->GetStorageShape().GetDimNum() != 3 ||
        k_shape->GetStorageShape().GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<SharedPrefixAttentionTilingData>();
    tiling->total_tokens = static_cast<uint32_t>(q_shape->GetStorageShape().GetDim(0));
    tiling->q_heads = static_cast<uint32_t>(q_shape->GetStorageShape().GetDim(1));
    tiling->kv_heads = static_cast<uint32_t>(k_shape->GetStorageShape().GetDim(1));
    tiling->head_dim = static_cast<uint32_t>(q_shape->GetStorageShape().GetDim(2));
    tiling->task_count = tiling->total_tokens * tiling->q_heads;
    const float* scale = attrs->GetAttrPointer<float>(0);
    if (scale == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->scale = *scale;
    context->SetBlockDim(std::min<uint32_t>(20, std::max<uint32_t>(1, tiling->task_count)));
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* q_shape = context->GetInputShape(0);
    gert::Shape* out_shape = context->GetOutputShape(0);
    gert::Shape* lse_shape = context->GetOutputShape(1);
    if (q_shape == nullptr || out_shape == nullptr || lse_shape == nullptr || q_shape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }
    *out_shape = *q_shape;
    lse_shape->SetDimNum(2);
    lse_shape->SetDim(0, q_shape->GetDim(0));
    lse_shape->SetDim(1, q_shape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class SharedPrefixAttentionForward : public OpDef {
public:
    explicit SharedPrefixAttentionForward(const char* name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("prefix_start").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("prefix_end").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sequence_start").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("lse").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scale").Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SharedPrefixAttentionForward);
}
