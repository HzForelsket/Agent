#include <ATen/DeviceGuard.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "npu_cpp_extension.h"
#include "aclnn_shared_prefix_attention_backward.h"
#include "aclnn_shared_prefix_attention_forward.h"

#include <cmath>
#include <tuple>

namespace {
constexpr int64_t kHeadDim = 128;

void check_metadata(const at::Tensor& tensor, const at::Tensor& q, const char* name)
{
    TORCH_CHECK(tensor.device() == q.device(), name, " must be on the same NPU as q");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must have dtype int32");
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == q.size(0),
                name, " must have shape [T]");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_inputs(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& prefix_start, const at::Tensor& prefix_end,
    const at::Tensor& sequence_start, double scale)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1,
                "shared_prefix_attention is NPU-only and has no CPU fallback");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(),
                "q, k and v must be on the same NPU");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
                k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
                "q, k and v must have dtype torch.bfloat16");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "q, k and v must use compact TND tensors with rank 3");
    TORCH_CHECK(q.size(0) > 0 && q.size(0) == k.size(0) && k.sizes() == v.sizes(),
                "q, k and v must have the same positive token count and matching k/v shapes");
    TORCH_CHECK(q.size(2) == kHeadDim && k.size(2) == kHeadDim,
                "only head_dim=128 is supported");
    TORCH_CHECK(k.size(1) > 0 && q.size(1) % k.size(1) == 0,
                "Hq must be divisible by Hkv");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q, k and v must be contiguous");
    TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
                "softmax scale must be finite and positive");
    check_metadata(prefix_start, q, "prefix_start");
    check_metadata(prefix_end, q, "prefix_end");
    check_metadata(sequence_start, q, "sequence_start");
}

std::tuple<at::Tensor, at::Tensor> forward_npu(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& prefix_start, const at::Tensor& prefix_end,
    const at::Tensor& sequence_start, double scale)
{
    check_inputs(q, k, v, prefix_start, prefix_end, sequence_start, scale);
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    at::Tensor out = at::empty_like(q);
    at::Tensor lse = at::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
    EXEC_NPU_CMD_EXT(aclnnSharedPrefixAttentionForward,
                     q, k, v, prefix_start, prefix_end, sequence_start,
                     scale, out, lse);
    return {out, lse};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_npu(
    const at::Tensor& grad_out, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& out, const at::Tensor& lse,
    const at::Tensor& prefix_start, const at::Tensor& prefix_end,
    const at::Tensor& sequence_start, double scale)
{
    check_inputs(q, k, v, prefix_start, prefix_end, sequence_start, scale);
    TORCH_CHECK(grad_out.device() == q.device() && grad_out.scalar_type() == at::kBFloat16 &&
                grad_out.sizes() == q.sizes() && grad_out.is_contiguous(),
                "grad_out must be a contiguous BF16 tensor matching q");
    TORCH_CHECK(out.device() == q.device() && out.scalar_type() == at::kBFloat16 &&
                out.sizes() == q.sizes() && out.is_contiguous(),
                "saved out tensor is invalid");
    TORCH_CHECK(lse.device() == q.device() && lse.scalar_type() == at::kFloat &&
                lse.dim() == 2 && lse.size(0) == q.size(0) && lse.size(1) == q.size(1) &&
                lse.is_contiguous(), "saved lse tensor is invalid");
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    at::Tensor dq = at::empty_like(q);
    at::Tensor dk = at::empty_like(k);
    at::Tensor dv = at::empty_like(v);
    EXEC_NPU_CMD_EXT(aclnnSharedPrefixAttentionBackward,
                     grad_out, q, k, v, out, lse,
                     prefix_start, prefix_end, sequence_start,
                     scale, dq, dk, dv);
    return {dq, dk, dv};
}

std::tuple<at::Tensor, at::Tensor> forward_meta(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, double)
{
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "q, k and v must have rank 3");
    TORCH_CHECK(q.size(2) == kHeadDim && k.size(2) == kHeadDim && v.size(2) == kHeadDim,
                "only head_dim=128 is supported");
    return {at::empty_like(q), at::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat))};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_meta(
    const at::Tensor&, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, double)
{
    return {at::empty_like(q), at::empty_like(k), at::empty_like(v)};
}
}

TORCH_LIBRARY(prefix_grouper_npu, m) {
    m.def("shared_prefix_attention_forward(Tensor q, Tensor k, Tensor v, "
          "Tensor prefix_start, Tensor prefix_end, Tensor sequence_start, float scale) -> (Tensor, Tensor)");
    m.def("shared_prefix_attention_backward(Tensor grad_out, Tensor q, Tensor k, Tensor v, "
          "Tensor out, Tensor lse, Tensor prefix_start, Tensor prefix_end, "
          "Tensor sequence_start, float scale) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(prefix_grouper_npu, PrivateUse1, m) {
    m.impl("shared_prefix_attention_forward", &forward_npu);
    m.impl("shared_prefix_attention_backward", &backward_npu);
}

TORCH_LIBRARY_IMPL(prefix_grouper_npu, Meta, m) {
    m.impl("shared_prefix_attention_forward", &forward_meta);
    m.impl("shared_prefix_attention_backward", &backward_meta);
}
