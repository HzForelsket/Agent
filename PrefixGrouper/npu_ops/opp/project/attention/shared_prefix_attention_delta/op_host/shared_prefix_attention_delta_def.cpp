#include "register/op_def_registry.h"

namespace ops {
class SharedPrefixAttentionDelta : public OpDef {
public:
    explicit SharedPrefixAttentionDelta(const char* name) : OpDef(name)
    {
        Input("attention").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Input("grad_out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("delta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SharedPrefixAttentionDelta);
}
