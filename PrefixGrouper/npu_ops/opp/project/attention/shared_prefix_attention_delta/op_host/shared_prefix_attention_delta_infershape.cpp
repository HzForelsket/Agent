#include "../../common/op_host/shape_utils.h"

namespace {
ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto* input = context->GetInputShape(0);
    auto* output = context->GetOutputShape(0);
    if (!input || !output || input->GetDimNum() != 3) return ge::GRAPH_FAILED;
    *output = *input;
    output->SetDim(2, kSharedPrefixRowAlignment);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}
IMPL_OP_INFERSHAPE(SharedPrefixAttentionDelta).InferShape(InferShape).InferDataType(InferDataType);
