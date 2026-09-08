#include "../../common/op_host/shape_utils.h"

namespace {
ge::graphStatus InferShape(gert::InferShapeContext* context)
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
ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}
}
IMPL_OP_INFERSHAPE(SharedPrefixAttentionPack).InferShape(InferShape).InferDataType(InferDataType);
