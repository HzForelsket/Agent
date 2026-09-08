#include "../../common/op_host/shape_utils.h"

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* q_shape = context->GetInputShape(0);
    gert::Shape* out_shape = context->GetOutputShape(0);
    gert::Shape* lse_shape = context->GetOutputShape(1);
    if (q_shape == nullptr || out_shape == nullptr || lse_shape == nullptr || q_shape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }
    if (q_shape->GetDim(2) <= 0 || q_shape->GetDim(2) > INT64_MAX - 15) return GRAPH_FAILED;
    *out_shape = *q_shape;
    lse_shape->SetDimNum(2);
    lse_shape->SetDim(0, q_shape->GetDim(0));
    lse_shape->SetDim(1, q_shape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SharedPrefixAttentionForward).InferShape(InferShape).InferDataType(InferDataType);
}
