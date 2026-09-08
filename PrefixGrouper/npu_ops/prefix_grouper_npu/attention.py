from __future__ import annotations

import numbers
from dataclasses import dataclass
from threading import Lock
from typing import Iterable, Sequence

import torch

from ._extension import load_extension
from .profiling import host_stage


_PLAN_CACHE: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], str], "SharedPrefixPlan"] = {}
_PLAN_LOCK = Lock()


def _positive_ints(values: Sequence[int] | torch.Tensor | Iterable[int], name: str) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError(f"{name} tensor must be on CPU; plan metadata is host-generated")
        values = values.tolist()
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional integer sequence") from exc
    if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) for value in raw):
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    result = tuple(int(value) for value in raw)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain only positive lengths")
    return result


@dataclass(frozen=True, slots=True)
class SharedPrefixPlan:
    prefix_start: torch.Tensor
    prefix_end: torch.Tensor
    sequence_start: torch.Tensor
    sequence_end: torch.Tensor
    group_end: torch.Tensor
    prefix_lens: tuple[int, ...]
    suffix_lens: tuple[int, ...]
    group_sizes: tuple[int, ...]
    total_tokens: int

    @property
    def device(self) -> torch.device:
        return self.prefix_start.device


def build_shared_prefix_plan(
    prefix_lens: Sequence[int] | torch.Tensor | Iterable[int],
    suffix_lens: Sequence[int] | torch.Tensor | Iterable[int],
    group_sizes: Sequence[int] | torch.Tensor | Iterable[int],
    *,
    device: torch.device | str,
) -> SharedPrefixPlan:
    prefixes = _positive_ints(prefix_lens, "prefix_lens")
    suffixes = _positive_ints(suffix_lens, "suffix_lens")
    groups = _positive_ints(group_sizes, "group_sizes")
    if len(prefixes) != len(groups):
        raise ValueError("prefix_lens and group_sizes must have one entry per group")
    if sum(groups) != len(suffixes):
        raise ValueError("sum(group_sizes) must equal len(suffix_lens)")
    if sum(prefixes) + sum(suffixes) > torch.iinfo(torch.int32).max:
        raise ValueError("plan token offsets must fit int32")

    target_device = torch.device(device)
    key = (prefixes, suffixes, groups, str(target_device))
    with _PLAN_LOCK:
        cached = _PLAN_CACHE.get(key)
        if cached is not None:
            return cached

    prefix_start: list[int] = []
    prefix_end: list[int] = []
    sequence_start: list[int] = []
    sequence_end: list[int] = []
    group_end: list[int] = []
    token_offset = 0
    suffix_index = 0
    for prefix_len, group_size in zip(prefixes, groups, strict=True):
        group_prefix_start = token_offset
        group_prefix_end = group_prefix_start + prefix_len
        prefix_start.extend([group_prefix_start] * prefix_len)
        prefix_end.extend([group_prefix_end] * prefix_len)
        sequence_start.extend([group_prefix_start] * prefix_len)
        sequence_end.extend([group_prefix_end] * prefix_len)
        token_offset = group_prefix_end
        for _ in range(group_size):
            suffix_len = suffixes[suffix_index]
            suffix_index += 1
            suffix_start = token_offset
            prefix_start.extend([group_prefix_start] * suffix_len)
            prefix_end.extend([group_prefix_end] * suffix_len)
            sequence_start.extend([suffix_start] * suffix_len)
            sequence_end.extend([suffix_start + suffix_len] * suffix_len)
            token_offset += suffix_len
        group_end.extend([token_offset] * (token_offset - group_prefix_start))

    plan = SharedPrefixPlan(
        prefix_start=torch.tensor(prefix_start, dtype=torch.int32, device=target_device).contiguous(),
        prefix_end=torch.tensor(prefix_end, dtype=torch.int32, device=target_device).contiguous(),
        sequence_start=torch.tensor(sequence_start, dtype=torch.int32, device=target_device).contiguous(),
        sequence_end=torch.tensor(sequence_end, dtype=torch.int32, device=target_device).contiguous(),
        group_end=torch.tensor(group_end, dtype=torch.int32, device=target_device).contiguous(),
        prefix_lens=prefixes,
        suffix_lens=suffixes,
        group_sizes=groups,
        total_tokens=token_offset,
    )
    with _PLAN_LOCK:
        return _PLAN_CACHE.setdefault(key, plan)


def _is_npu(device: torch.device) -> bool:
    return device.type in {"npu", "privateuseone"}


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, plan: SharedPrefixPlan) -> None:
    if not _is_npu(q.device):
        raise ValueError("shared_prefix_attention is NPU-only and has no CPU fallback")
    if q.device != k.device or q.device != v.device or plan.device != q.device:
        raise ValueError("q, k, v and plan metadata must be on the same NPU")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("q, k and v must have dtype torch.bfloat16")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k and v must use compact [T, H, D] TND layout")
    if q.shape[0] == 0 or q.shape[0] != k.shape[0] or k.shape != v.shape:
        raise ValueError("q, k and v must have the same positive token count and matching k/v shapes")
    if q.shape[2] <= 0 or q.shape[2] != k.shape[2]:
        raise ValueError("q, k and v must have the same positive head_dim")
    if q.shape[1] == 0 or k.shape[1] == 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError("Hq must be divisible by Hkv")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q, k and v must be contiguous")
    if q.shape[0] != plan.total_tokens:
        raise ValueError(f"plan expects {plan.total_tokens} tokens, got {q.shape[0]}")
    for name in ("prefix_start", "prefix_end", "sequence_start", "sequence_end", "group_end"):
        tensor = getattr(plan, name)
        if tensor.device != q.device:
            raise ValueError(f"plan.{name} must be on the same NPU as q")
        if tensor.dtype != torch.int32 or tensor.ndim != 1 or tensor.numel() != plan.total_tokens:
            raise ValueError(f"plan.{name} must be contiguous int32 [T]")
        if not tensor.is_contiguous():
            raise ValueError(f"plan.{name} must be contiguous int32 [T]")


class _SharedPrefixAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prefix_start: torch.Tensor,
        prefix_end: torch.Tensor,
        sequence_start: torch.Tensor,
        sequence_end: torch.Tensor,
        group_end: torch.Tensor,
        scale: float,
        prefix_lens: tuple[int, ...],
        suffix_lens: tuple[int, ...],
        group_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        out, lse = torch.ops.prefix_grouper_npu.shared_prefix_attention_forward(
            q, k, v, prefix_start, prefix_end, sequence_start, sequence_end, group_end, scale,
            prefix_lens, suffix_lens, group_sizes
        )
        ctx.save_for_backward(q, k, v, out, lse, prefix_start, prefix_end, sequence_start, sequence_end, group_end)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor):
        q, k, v, out, lse, prefix_start, prefix_end, sequence_start, sequence_end, group_end = ctx.saved_tensors
        dq, dk, dv = torch.ops.prefix_grouper_npu.shared_prefix_attention_backward(
            grad_out.contiguous(), q, k, v, out, lse,
            prefix_start, prefix_end, sequence_start, sequence_end, group_end, ctx.scale
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def shared_prefix_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: SharedPrefixPlan,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    with host_stage("pg_host/custom/python_validate"):
        _validate(q, k, v, plan)
    with host_stage("pg_host/custom/python_scale"):
        scale_fp32 = (
            torch.tensor(q.shape[2], dtype=torch.float32, device="cpu").rsqrt()
            if softmax_scale is None
            else torch.tensor(softmax_scale, dtype=torch.float32, device="cpu")
        )
        if not torch.isfinite(scale_fp32).item() or not (scale_fp32 > 0).item():
            raise ValueError("softmax_scale must be finite and positive")
        # Python/PyTorch transport scalars as doubles; computed in FP32.
        scale = scale_fp32.item()
    with host_stage("pg_host/custom/load_extension"):
        load_extension()
    with host_stage("pg_host/custom/autograd_apply"):
        return _SharedPrefixAttention.apply(
            q, k, v, plan.prefix_start, plan.prefix_end, plan.sequence_start, plan.sequence_end, plan.group_end, scale,
            plan.prefix_lens, plan.suffix_lens, plan.group_sizes
        )
