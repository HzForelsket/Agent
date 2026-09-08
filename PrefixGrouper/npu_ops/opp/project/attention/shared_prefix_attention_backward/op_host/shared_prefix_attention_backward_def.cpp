#include "register/op_def_registry.h"

namespace ops {
class SharedPrefixAttentionBackward : public OpDef {
public:
    explicit SharedPrefixAttentionBackward(const char* name) : OpDef(name)
    {
        this->Input("grad_out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("delta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lse").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("prefix_start").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("prefix_end").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sequence_start").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sequence_end").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("group_end").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dq").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dk").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dv").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scale").Float();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SharedPrefixAttentionBackward);
}
