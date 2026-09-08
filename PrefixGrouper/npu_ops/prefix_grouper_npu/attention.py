from __future__ import annotations

import numbers
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Iterable, Sequence

import torch

from ._extension import get_forward_op
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


def _validate(q: torch.Tensor, plan: SharedPrefixPlan) -> None:
    # Only checks needed before dispatch/scale resolution belong in Python.
    # Tensor dtype, shape, layout and metadata checks are owned by C++.
    if q.device.type not in {"npu", "privateuseone"}:
        raise ValueError("shared_prefix_attention is NPU-only and has no CPU fallback")
    if q.ndim != 3:
        raise ValueError("q, k and v must use compact [T, H, D] TND layout")
    if q.shape[0] != plan.total_tokens:
        raise ValueError(f"plan expects {plan.total_tokens} tokens, got {q.shape[0]}")


@lru_cache(maxsize=128)
def _default_scale(head_dim: int) -> float:
    # Preserve the existing FP32 rsqrt result, including its rounding, once per D.
    # Warm calls only look up a Python float; no per-call CPU tensor operations.
    return torch.tensor(head_dim, dtype=torch.float32, device="cpu").rsqrt().item()


def shared_prefix_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: SharedPrefixPlan,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    with host_stage("pg_host/custom/python_validate"):
        _validate(q, plan)
    with host_stage("pg_host/custom/python_scale"):
        # C++ converts to FP32 and validates the scalar at the dispatch boundary.
        scale = _default_scale(q.shape[2]) if softmax_scale is None else float(softmax_scale)
    with host_stage("pg_host/custom/dispatch"):
        # Resolve the overload once; PyTorch's registered autograd implementation
        # only creates a backward context when gradients are actually required.
        out, _ = get_forward_op()(
            q, k, v, plan.prefix_start, plan.prefix_end, plan.sequence_start, plan.sequence_end, plan.group_end, scale,
            plan.prefix_lens, plan.suffix_lens, plan.group_sizes
        )
        return out
