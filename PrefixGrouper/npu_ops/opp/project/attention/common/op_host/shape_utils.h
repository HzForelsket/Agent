#ifndef PREFIX_GROUPER_ATTENTION_SHAPE_UTILS_H
#define PREFIX_GROUPER_ATTENTION_SHAPE_UTILS_H
#include "../op_kernel/shared_prefix_attention_tiling_data.h"
#include "register/op_impl_registry.h"

namespace shared_prefix_host {
inline bool Multiply(uint64_t a, uint64_t b, uint64_t& result)
{
    if (b && a > static_cast<uint64_t>(INT64_MAX) / b) return false;
    result = a * b;
    return true;
}
inline uint64_t Align(uint64_t n, uint64_t alignment = kSharedPrefixRowAlignment)
{
    return (n + alignment - 1) / alignment * alignment;
}
inline bool Shape(const gert::Shape& q, const gert::Shape& k)
{
    return q.GetDimNum() == 3 && k.GetDimNum() == 3 && q.GetDim(0) > 0 &&
        q.GetDim(0) <= INT32_MAX && q.GetDim(0) == k.GetDim(0) &&
        q.GetDim(1) > 0 && k.GetDim(1) > 0 && q.GetDim(1) % k.GetDim(1) == 0 &&
        q.GetDim(2) > 0 && q.GetDim(2) <= INT64_MAX - 15 && q.GetDim(2) == k.GetDim(2);
}
}
#endif
