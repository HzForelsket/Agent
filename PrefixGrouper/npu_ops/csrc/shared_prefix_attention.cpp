#include <ATen/DeviceGuard.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "npu_cpp_extension.h"
#include "aclnn_shared_prefix_attention_backward.h"
#include "aclnn_shared_prefix_attention_forward.h"
#include "aclnn_shared_prefix_attention_pack.h"
#include "aclnn_shared_prefix_attention_delta.h"
#include <cmath>
#include <limits>
#include <tuple>

namespace {
constexpr int64_t kRowAlignment = 16;
int64_t padded_dim(int64_t dim)
{
    TORCH_CHECK(dim > 0 && dim <= INT64_MAX - kRowAlignment + 1, "head_dim is outside the addressable range");
    return (dim + kRowAlignment - 1) / kRowAlignment * kRowAlignment;
}
void check_shapes(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v)
{
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k and v must use rank-3 compact TND tensors");
    TORCH_CHECK(q.size(0) > 0 && q.size(0) <= INT32_MAX && q.size(0) == k.size(0) && k.sizes() == v.sizes(),
                "q, k and v must have the same positive token count fitting int32 and matching k/v shapes");
    TORCH_CHECK(q.size(2) > 0 && q.size(2) == k.size(2), "q, k and v must have the same positive head_dim");
    TORCH_CHECK(q.size(1) > 0 && k.size(1) > 0 && q.size(1) % k.size(1) == 0,
                "positive Hq must be divisible by positive Hkv");
    const int64_t padded = padded_dim(q.size(2));
    TORCH_CHECK(q.size(1) <= INT64_MAX / q.size(0) &&
                q.size(0) * q.size(1) <= INT64_MAX / sizeof(float) / padded,
                "padded attention workspace exceeds the addressable range");
}
void check_metadata(const at::Tensor& tensor, const at::Tensor& q, const char* name)
{
    TORCH_CHECK(tensor.device() == q.device(), name, " must be on the same NPU as q");
    TORCH_CHECK(tensor.scalar_type() == at::kInt && tensor.dim() == 1 && tensor.size(0) == q.size(0) && tensor.is_contiguous(),
                name, " must be contiguous int32 [T]");
}
void check_inputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& ps, const at::Tensor& pe, const at::Tensor& ss,
    const at::Tensor& se, const at::Tensor& ge, float scale)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "shared_prefix_attention is NPU-only and has no CPU fallback");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(), "q, k and v must be on the same NPU");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
                "q, k and v must have dtype torch.bfloat16");
    check_shapes(q, k, v);
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q, k and v must be contiguous");
    TORCH_CHECK(std::isfinite(scale) && scale > 0.0f, "softmax scale must be finite and positive");
    check_metadata(ps, q, "prefix_start"); check_metadata(pe, q, "prefix_end");
    check_metadata(ss, q, "sequence_start"); check_metadata(se, q, "sequence_end"); check_metadata(ge, q, "group_end");
}
at::Tensor accumulator(const at::Tensor& like)
{
    return at::empty({like.size(0), like.size(1), padded_dim(like.size(2))}, like.options().dtype(at::kFloat));
}
std::tuple<at::Tensor, at::Tensor> forward_npu(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& ps, const at::Tensor& pe, const at::Tensor& ss,
    const at::Tensor& se, const at::Tensor& ge, double scale)
{
    const float fp_scale = static_cast<float>(scale);
    check_inputs(q, k, v, ps, pe, ss, se, ge, fp_scale);
    scale = static_cast<double>(fp_scale);
    const c10::OptionalDeviceGuard guard(device_of(q));
    auto acc = accumulator(q);
    auto lse_rows = at::empty({q.size(0), q.size(1), kRowAlignment}, q.options().dtype(at::kFloat));
    auto out = at::empty_like(q);
    EXEC_NPU_CMD_EXT(aclnnSharedPrefixAttentionForward, q, k, v, ps, pe, ss, se, ge, scale, acc, lse_rows);
    const int64_t dim = q.size(2);
    EXEC_NPU_CMD_EXT(aclnnSharedPrefixAttentionPack, acc, dim, out);
    // Native strided-to-contiguous copy has independent aligned output ownership.
    // The attention kernel never makes concurrent scalar stores to adjacent LSE values.
    auto lse = lse_rows.select(2, 0).contiguous();
    return {out, lse};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_npu(
    const at::Tensor& grad_out, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& out, const at::Tensor& lse, const at::Tensor& ps, const at::Tensor& pe,
    const at::Tensor& ss, const at::Tensor& se, const at::Tensor& ge, double scale)
{
    const float fp_scale = static_cast<float>(scale);
    check_inputs(q, k, v, ps, pe, ss, se, ge, fp_scale);
    scale = static_cast<double>(fp_scale);
    for (const auto* tensor : {&grad_out, &out})
        TORCH_CHECK(tensor->device() == q.device() && tensor->scalar_type() == at::kBFloat16 &&
                    tensor->sizes() == q.sizes() && tensor->is_contiguous(), "out and grad_out must match contiguous BF16 q");
    TORCH_CHECK(lse.device() == q.device() && lse.scalar_type() == at::kFloat && lse.dim() == 2 &&
                lse.size(0) == q.size(0) && lse.size(1) == q.size(1) && lse.is_contiguous(), "saved lse tensor is invalid");
    const c10::OptionalDeviceGuard guard(device_of(q));
    auto delta = at::empty({q.size(0), q.size(1), kRowAlignment}, q.options().dtype(at::kFloat));
    auto aq = accumulator(q), ak = accumulator(k), av = accumulator(v);
    auto dq = at::empty_like(q), dk = at::empty_like(k), dv = at::empty_like(v);
    EXEC_NPU_CMD_EXT(aclnnSharedPrefixAttentionDelta, out, grad_out, delta);
    EXEC_NPU_CMD_EXT(aclnnSharedPrefixAttentionBackward, grad_out, q, k, v, delta, lse,
                     ps, pe, ss, se, ge, scale, aq, ak, av);
    const int64_t dim = q.size(2);
    EXEC_NPU_CMD_EXT(aclnnSharedPrefixAttentionPack, aq, dim, dq);
    EXEC_NPU_CMD_EXT(aclnnSharedPrefixAttentionPack, ak, dim, dk);
    EXEC_NPU_CMD_EXT(aclnnSharedPrefixAttentionPack, av, dim, dv);
    return {dq, dk, dv};
}
std::tuple<at::Tensor, at::Tensor> forward_meta(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, double)
{
    check_shapes(q, k, v);
    return {at::empty_like(q), at::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat))};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_meta(
    const at::Tensor&, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, double)
{
    check_shapes(q, k, v);
    return {at::empty_like(q), at::empty_like(k), at::empty_like(v)};
}
}
TORCH_LIBRARY(prefix_grouper_npu, m) {
    m.def("shared_prefix_attention_forward(Tensor q, Tensor k, Tensor v, Tensor prefix_start, Tensor prefix_end, "
          "Tensor sequence_start, Tensor sequence_end, Tensor group_end, float scale) -> (Tensor, Tensor)");
    m.def("shared_prefix_attention_backward(Tensor grad_out, Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, "
          "Tensor prefix_start, Tensor prefix_end, Tensor sequence_start, Tensor sequence_end, Tensor group_end, "
          "float scale) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(prefix_grouper_npu, PrivateUse1, m) {
    m.impl("shared_prefix_attention_forward", &forward_npu);
    m.impl("shared_prefix_attention_backward", &backward_npu);
}
TORCH_LIBRARY_IMPL(prefix_grouper_npu, Meta, m) {
    m.impl("shared_prefix_attention_forward", &forward_meta);
    m.impl("shared_prefix_attention_backward", &backward_meta);
}
