"""
PyTorch autograd functions.
"""

import torch
from torch.autograd import Function
from .utils.typing import Tuple

IndicesTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
LinearIndicesTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _swap_head_sequence_dims(x: torch.Tensor) -> torch.Tensor:
    """Swap the head and sequence dimensions and materialize an ND tensor."""
    permutation = (0, 2, 1, *range(3, x.ndim))
    return x.permute(permutation).contiguous()


def _scatter_selected_rows(
    output: torch.Tensor,
    source: torch.Tensor,
    source_indices: torch.Tensor,
    output_indices: torch.Tensor,
) -> None:
    """Copy selected rows with device-native single-axis gather and scatter ops."""
    if source_indices.numel() == 0:
        return

    updates = torch.index_select(source, 0, source_indices)
    if output.device.type == "npu":
        import torch_npu

        torch_npu.scatter_update_(output, output_indices, updates, axis=0)
    elif output.device.type == "cuda":
        output.index_copy_(0, output_indices, updates)
    else:
        raise RuntimeError(
            f"single-axis ungroup is only implemented for NPU and CUDA, got {output.device.type}"
        )


class UngroupFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,  # NOTE: Shape: [b, num_heads (or 1 for common tensors), seq, ...]
        indices: IndicesTuple,  # 4 non-zero mask index tensors
        shapes: Tuple[
            torch.Size, torch.Size
        ],  # shapes of ungrouped prefix and ungrouped suffix
    ):
        # NOTE: This function can accept [b, num_heads (or 1 for common tensors), seq, ...]
        assert x.ndim >= 3
        input_shape = x.shape
        (
            ungrouped_prefix_indices,
            ungrouped_suffix_indices,
            grouped_prefix_indices,
            grouped_suffix_indices,
        ) = indices
        (
            prefix_x_shape,
            suffix_x_shape,
        ) = shapes
        ctx.save_for_backward(*indices)
        ctx.input_shape = input_shape
        # Split the grouped inputs into prefix and suffix tensors.
        prefix_x = torch.zeros(
            prefix_x_shape[0],
            input_shape[1],
            prefix_x_shape[1],
            *input_shape[3:],
            dtype=x.dtype,
            device=x.device,
        )
        suffix_x = torch.zeros(
            suffix_x_shape[0],
            input_shape[1],
            suffix_x_shape[1],
            *input_shape[3:],
            dtype=x.dtype,
            device=x.device,
        )
        prefix_x[ungrouped_prefix_indices[:, 0], :, ungrouped_prefix_indices[:, 1]] = x[
            grouped_prefix_indices[:, 0], :, grouped_prefix_indices[:, 1]
        ]
        suffix_x[ungrouped_suffix_indices[:, 0], :, ungrouped_suffix_indices[:, 1]] = x[
            grouped_suffix_indices[:, 0], :, grouped_suffix_indices[:, 1]
        ]
        return prefix_x, suffix_x

    @staticmethod
    def backward(ctx, grad_prefix_x: torch.Tensor, grad_suffix_x: torch.Tensor):
        (
            ungrouped_prefix_indices,
            ungrouped_suffix_indices,
            grouped_prefix_indices,
            grouped_suffix_indices,
        ) = ctx.saved_tensors
        input_shape = ctx.input_shape
        # Concat the prefix and suffix grad into a single tensor.
        grad_x = torch.zeros(
            input_shape, dtype=grad_prefix_x.dtype, device=grad_prefix_x.device
        )
        grad_x[grouped_prefix_indices[:, 0], :, grouped_prefix_indices[:, 1]] = (
            grad_prefix_x[
                ungrouped_prefix_indices[:, 0], :, ungrouped_prefix_indices[:, 1]
            ]
        )
        grad_x[grouped_suffix_indices[:, 0], :, grouped_suffix_indices[:, 1]] = (
            grad_suffix_x[
                ungrouped_suffix_indices[:, 0], :, ungrouped_suffix_indices[:, 1]
            ]
        )
        return grad_x, None, None


class LinearUngroupFunction(Function):
    """Ungroup with contiguous layouts and single-axis device indexing."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        linear_indices: LinearIndicesTuple,
        shapes: Tuple[torch.Size, torch.Size],
    ):
        assert x.ndim >= 3
        (
            ungrouped_prefix_indices,
            ungrouped_suffix_indices,
            grouped_prefix_indices,
            grouped_suffix_indices,
        ) = linear_indices
        prefix_x_shape, suffix_x_shape = shapes

        ctx.save_for_backward(*linear_indices)
        ctx.input_shape = x.shape

        x_sequence_major = _swap_head_sequence_dims(x)
        x_flat = x_sequence_major.flatten(0, 1)
        trailing_shape = x_sequence_major.shape[2:]

        prefix_flat = x.new_zeros(
            (prefix_x_shape[0] * prefix_x_shape[1], *trailing_shape)
        )
        _scatter_selected_rows(
            prefix_flat,
            x_flat,
            grouped_prefix_indices,
            ungrouped_prefix_indices,
        )
        prefix_sequence_major = prefix_flat.reshape(
            prefix_x_shape[0], prefix_x_shape[1], *trailing_shape
        )
        prefix_x = _swap_head_sequence_dims(prefix_sequence_major)

        suffix_flat = x.new_zeros(
            (suffix_x_shape[0] * suffix_x_shape[1], *trailing_shape)
        )
        _scatter_selected_rows(
            suffix_flat,
            x_flat,
            grouped_suffix_indices,
            ungrouped_suffix_indices,
        )
        suffix_sequence_major = suffix_flat.reshape(
            suffix_x_shape[0], suffix_x_shape[1], *trailing_shape
        )
        suffix_x = _swap_head_sequence_dims(suffix_sequence_major)
        return prefix_x, suffix_x

    @staticmethod
    def backward(ctx, grad_prefix_x: torch.Tensor, grad_suffix_x: torch.Tensor):
        (
            ungrouped_prefix_indices,
            ungrouped_suffix_indices,
            grouped_prefix_indices,
            grouped_suffix_indices,
        ) = ctx.saved_tensors
        input_shape = ctx.input_shape

        grad_prefix_sequence_major = _swap_head_sequence_dims(grad_prefix_x)
        grad_suffix_sequence_major = _swap_head_sequence_dims(grad_suffix_x)
        grad_prefix_flat = grad_prefix_sequence_major.flatten(0, 1)
        grad_suffix_flat = grad_suffix_sequence_major.flatten(0, 1)

        grad_x_sequence_major_shape = (
            input_shape[0],
            input_shape[2],
            input_shape[1],
            *input_shape[3:],
        )
        grad_x_flat = grad_prefix_x.new_zeros(
            (
                input_shape[0] * input_shape[2],
                input_shape[1],
                *input_shape[3:],
            )
        )
        _scatter_selected_rows(
            grad_x_flat,
            grad_prefix_flat,
            ungrouped_prefix_indices,
            grouped_prefix_indices,
        )
        _scatter_selected_rows(
            grad_x_flat,
            grad_suffix_flat,
            ungrouped_suffix_indices,
            grouped_suffix_indices,
        )
        grad_x_sequence_major = grad_x_flat.reshape(grad_x_sequence_major_shape)
        grad_x = _swap_head_sequence_dims(grad_x_sequence_major)
        return grad_x, None, None


class GroupFunction(Function):
    @staticmethod
    def forward(
        ctx,
        prefix_x: torch.Tensor,  # NOTE: Shape [b, seq, ...]
        suffix_x: torch.Tensor,  # NOTE: Shape [b, seq, ...]
        indices: IndicesTuple,  # 4 non-zero mask index tensors
        x_shape: torch.Size,  # shape of grouped input x
    ):
        # NOTE: This function can accept [b, seq, ...]
        assert prefix_x.ndim == suffix_x.ndim >= 2
        prefix_shape, suffix_shape = prefix_x.shape, suffix_x.shape
        (
            ungrouped_prefix_indices,
            ungrouped_suffix_indices,
            grouped_prefix_indices,
            grouped_suffix_indices,
        ) = indices
        ctx.save_for_backward(*indices)
        ctx.prefix_shape, ctx.suffix_shape = prefix_shape, suffix_shape
        # Concat the prefix and suffix inputs into a single grouped input tensor
        x = torch.zeros(
            *x_shape[:2],
            *prefix_shape[2:],
            dtype=prefix_x.dtype,
            device=prefix_x.device,
        )
        x[grouped_prefix_indices[:, 0], grouped_prefix_indices[:, 1]] = prefix_x[
            ungrouped_prefix_indices[:, 0], ungrouped_prefix_indices[:, 1]
        ]
        x[grouped_suffix_indices[:, 0], grouped_suffix_indices[:, 1]] = suffix_x[
            ungrouped_suffix_indices[:, 0], ungrouped_suffix_indices[:, 1]
        ]
        return x

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor):
        (
            ungrouped_prefix_indices,
            ungrouped_suffix_indices,
            grouped_prefix_indices,
            grouped_suffix_indices,
        ) = ctx.saved_tensors
        prefix_shape, suffix_shape = ctx.prefix_shape, ctx.suffix_shape
        # Split the grad into prefix grad and suffix grad
        grad_prefix_x = torch.zeros(
            prefix_shape, dtype=grad_x.dtype, device=grad_x.device
        )
        grad_prefix_x[
            ungrouped_prefix_indices[:, 0], ungrouped_prefix_indices[:, 1]
        ] = grad_x[grouped_prefix_indices[:, 0], grouped_prefix_indices[:, 1]]
        grad_suffix_x = torch.zeros(
            suffix_shape, dtype=grad_x.dtype, device=grad_x.device
        )
        grad_suffix_x[
            ungrouped_suffix_indices[:, 0], ungrouped_suffix_indices[:, 1]
        ] = grad_x[grouped_suffix_indices[:, 0], grouped_suffix_indices[:, 1]]
        return grad_prefix_x, grad_suffix_x, None, None


class ConvertPaddingFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,  # NOTE: Shape: [b, seq, ...]
        indices: Tuple[torch.Tensor, torch.Tensor],  # 2 non-zero mask index tensors
        o_shape: torch.Size,  # shape of converted output tensor
    ):
        input_shape = x.shape
        ctx.input_shape = input_shape
        x_indices, o_indices = indices
        ctx.save_for_backward(*indices)
        o = torch.zeros(
            *o_shape[:2],
            *input_shape[2:],
            dtype=x.dtype,
            device=x.device,
        )
        o[o_indices[:, 0], o_indices[:, 1]] = x[x_indices[:, 0], x_indices[:, 1]]
        return o

    @staticmethod
    def backward(ctx, grad_o: torch.Tensor):
        x_indices, o_indices = ctx.saved_tensors
        grad_x = torch.zeros(
            ctx.input_shape,
            dtype=grad_o.dtype,
            device=grad_o.device,
        )
        grad_x[x_indices[:, 0], x_indices[:, 1]] = grad_o[
            o_indices[:, 0], o_indices[:, 1]
        ]
        return grad_x, None, None
