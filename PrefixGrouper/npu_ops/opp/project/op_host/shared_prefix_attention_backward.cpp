#include "../op_kernel/shared_prefix_attention_tiling.h"
#include "register/op_def_registry.h"

#include <algorithm>
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* q_shape = context->GetInputShape(1);
    const gert::StorageShape* k_shape = context->GetInputShape(2);
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
    tiling->task_count = tiling->total_tokens * (tiling->q_heads + tiling->kv_heads);
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
    const gert::Shape* q_shape = context->GetInputShape(1);
    const gert::Shape* k_shape = context->GetInputShape(2);
    const gert::Shape* v_shape = context->GetInputShape(3);
    if (q_shape == nullptr || k_shape == nullptr || v_shape == nullptr) {
        return GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *q_shape;
    *context->GetOutputShape(1) = *k_shape;
    *context->GetOutputShape(2) = *v_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(1));
    context->SetOutputDataType(1, context->GetInputDataType(2));
    context->SetOutputDataType(2, context->GetInputDataType(3));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class SharedPrefixAttentionBackward : public OpDef {
public:
    explicit SharedPrefixAttentionBackward(const char* name) : OpDef(name)
    {
        this->Input("grad_out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lse").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("prefix_start").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("prefix_end").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sequence_start").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dq").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dk").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dv").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scale").Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SharedPrefixAttentionBackward);
}
