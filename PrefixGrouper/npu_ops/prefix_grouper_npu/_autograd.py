"""Autograd registration for the native shared-prefix operator."""
import torch


def _setup_context(ctx, inputs, output):
    q, k, v, ps, pe, ss, se, ge, scale, _, _, _ = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, ps, pe, ss, se, ge)
    ctx.scale = scale
    # LSE is saved state for the attention backward, not a differentiable output.
    ctx.mark_non_differentiable(lse)
    ctx.set_materialize_grads(False)


def _backward(ctx, grad_out, grad_lse):
    if grad_out is None:
        return (None,) * 12
    q, k, v, out, lse, ps, pe, ss, se, ge = ctx.saved_tensors
    dq, dk, dv = torch.ops.prefix_grouper_npu.shared_prefix_attention_backward.default(
        grad_out.contiguous(), q, k, v, out, lse, ps, pe, ss, se, ge, ctx.scale
    )
    return dq, dk, dv, None, None, None, None, None, None, None, None, None


def register_autograd() -> None:
    torch.library.register_autograd(
        "prefix_grouper_npu::shared_prefix_attention_forward",
        _backward,
        setup_context=_setup_context,
    )
