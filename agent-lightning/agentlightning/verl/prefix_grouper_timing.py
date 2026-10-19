# Copyright (c) Microsoft. All rights reserved.

"""Opt-in PrefixGrouper timings; accelerator events are read only at batch end."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from typing import Any, Generator

import torch

# VERL runs one engine batch at a time per worker process. Use process scope:
# Python custom backward can execute on autograd threads, where ContextVars
# from the calling thread are absent.
_active_timing: PrefixTiming | None = None
_OPERATIONS: set[str] = {
    "forward/batch_prepare",
    "forward/group_masks_to_cpu",
    "forward/group_metadata_cpu",
    "forward/group_metadata_to_device",
    "forward/kv_head_expand",
    "forward/causal_mask",
    "forward/prefix_attention",
    "forward/suffix_attention",
    "forward/prefix_repeat",
    "forward/prefix_suffix_cat",
    "forward/restore_order",
}


class PrefixTiming:
    """Sum calls within one local engine batch, including checkpoint recomputation.

    Event intervals include stream stalls and host submission gaps. They are not
    kernel-only time. Nested ranges are inclusive and must not be added together.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.host: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.events: list[tuple[str, Any, Any]] = []

    @contextmanager
    def activate(self) -> Generator[None, None, None]:
        global _active_timing
        if _active_timing is not None:
            raise RuntimeError("Prefix timing requires non-overlapping engine batches within a worker process.")
        _active_timing = self
        try:
            yield
        finally:
            _active_timing = None

    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        start = end = None
        if self.device.type in {"cuda", "npu"}:
            backend = torch.get_device_module(self.device)
            start = backend.Event(enable_timing=True)
            end = backend.Event(enable_timing=True)
            start.record()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.host[name] += time.perf_counter() - started
            self.calls[name] += 1
            if end is not None:
                end.record()
                self.events.append((name, start, end))

    def finish(self) -> dict[str, float]:
        device_seconds: dict[str, float] = defaultdict(float)
        started = time.perf_counter()
        # Synchronize recorded ends after all micro-batches/backward, never in
        # measure(). This also handles events recorded on different streams.
        for name, start, end in self.events:
            end.synchronize()
            device_seconds[name] += start.elapsed_time(end) / 1000.0
        result = {"prefix_timing/event_readback_host_s": time.perf_counter() - started}
        # Emit zero for unused operations so DP ranks with no shareable groups
        # participate in the same mean instead of silently disappearing.
        for name in sorted(_OPERATIONS):
            root = f"prefix_timing/{name}"
            result[f"{root}/host_s"] = self.host[name]
            result[f"{root}/calls"] = float(self.calls[name])
            if self.device.type in {"cuda", "npu"}:
                result[f"{root}/device_interval_s"] = device_seconds[name]
        return result


@contextmanager
def measure_prefix(name: str) -> Generator[None, None, None]:
    """Measure an existing code region only while a timing session is active."""
    timing = _active_timing
    if timing is None:
        yield
    else:
        with timing.measure(name):
            yield


def timed_prefix(name: str, function: Any) -> Any:
    """Wrap the imported implementation without changing its inputs or outputs."""
    _OPERATIONS.add(name)

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if _active_timing is None:
            return function(*args, **kwargs)
        with measure_prefix(name):
            return function(*args, **kwargs)

    return wrapped


def install_prefix_timing_hooks(prefix_grouper_class: type) -> None:
    """Instrument the actually imported package, including its custom backward."""
    import prefix_grouper.function as functions

    for method, name in {
        "ungroup": "forward/qkv_ungroup",
        "group": "forward/attention_output_group",
        "concat_input": "forward/input_pack",
        "split_output": "forward/logits_split",
    }.items():
        setattr(prefix_grouper_class, method, timed_prefix(name, getattr(prefix_grouper_class, method)))
    setattr(
        prefix_grouper_class,
        "convert_padding",
        staticmethod(timed_prefix("forward/completion_padding", getattr(prefix_grouper_class, "convert_padding"))),
    )
    for name, cls in vars(functions).items():
        if isinstance(cls, type) and issubclass(cls, torch.autograd.Function) and cls.__module__ == functions.__name__:
            cls.forward = staticmethod(timed_prefix(f"forward/{name}", cls.forward))
            cls.backward = staticmethod(timed_prefix(f"backward/{name}", cls.backward))
