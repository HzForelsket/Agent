#include "register/op_def_registry.h"

namespace ops {
class SharedPrefixAttentionPack : public OpDef {
public:
    explicit SharedPrefixAttentionPack(const char* name) : OpDef(name)
    {
        Input("accumulator").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Attr("head_dim").Int();
        AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SharedPrefixAttentionPack);
}
