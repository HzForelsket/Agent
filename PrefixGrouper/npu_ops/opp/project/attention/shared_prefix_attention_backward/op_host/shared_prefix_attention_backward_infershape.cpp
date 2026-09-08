#include "../../common/op_host/shape_utils.h"

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
    for (uint32_t i = 0; i < 3; ++i) {
        auto* out = context->GetOutputShape(i);
        if (out->GetDimNum() != 3 || out->GetDim(2) <= 0 || out->GetDim(2) > INT64_MAX - 15) return GRAPH_FAILED;
        out->SetDim(2, shared_prefix_host::Align(out->GetDim(2)));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SharedPrefixAttentionBackward).InferShape(InferShape).InferDataType(InferDataType);
}
