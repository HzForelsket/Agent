# Copyright (c) Microsoft. All rights reserved.

"""Accelerator selection and timing helpers for VERL benchmarks."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

import torch

Backend = Literal["gpu", "npu"]

__all__ = ["AcceleratorRuntime", "Backend", "choose_backend", "select_accelerator"]


def _npu_module() -> ModuleType | None:
    if not hasattr(torch, "npu"):
        try:
            importlib.import_module("torch_npu")
        except Exception:
            return None
    return getattr(torch, "npu", None)


def choose_backend(requested: str, *, npu_available: bool, cuda_available: bool) -> Backend:
    """Choose an available accelerator, preferring Ascend for ``auto``."""
    normalized = requested.strip().lower()
    if normalized in {"cuda", "gpu"}:
        if not cuda_available:
            raise RuntimeError("已指定 GPU，但 torch.cuda.is_available() 为 False。")
        return "gpu"
    if normalized == "npu":
        if not npu_available:
            raise RuntimeError("已指定 NPU，但 torch.npu.is_available() 为 False；请检查 torch-npu、CANN 和设备权限。")
        return "npu"
    if normalized != "auto":
        raise ValueError(f"不支持的设备 {requested!r}；可选值为 auto、gpu/cuda 或 npu。")

    platform_override = os.environ.get("VERL_PLATFORM", "").strip().lower()
    if platform_override in {"huawei", "npu"}:
        return choose_backend("npu", npu_available=npu_available, cuda_available=cuda_available)
    if platform_override in {"nvidia", "cuda", "gpu"}:
        return choose_backend("gpu", npu_available=npu_available, cuda_available=cuda_available)
    if npu_available:
        return "npu"
    if cuda_available:
        return "gpu"
    raise RuntimeError("未发现可用的昇腾 NPU 或 NVIDIA GPU。")


@dataclass(frozen=True)
class AcceleratorRuntime:
    """Uniform CUDA/NPU operations needed by the performance benchmark."""

    backend: Backend
    device: torch.device
    module: ModuleType

    @property
    def device_type(self) -> str:
        return "npu" if self.backend == "npu" else "cuda"

    @property
    def display_backend(self) -> str:
        return "NPU" if self.backend == "npu" else "GPU"

    def set_device(self) -> None:
        self.module.set_device(self.device)

    def synchronize(self) -> None:
        self.module.synchronize(self.device)

    def empty_cache(self) -> None:
        self.module.empty_cache()

    def manual_seed_all(self, seed: int) -> None:
        self.module.manual_seed_all(seed)

    def event(self) -> Any:
        return self.module.Event(enable_timing=True)

    def memory_allocated(self) -> int:
        return int(self.module.memory_allocated(self.device))

    def reset_peak_memory_stats(self) -> None:
        self.module.reset_peak_memory_stats(self.device)

    def max_memory_allocated(self) -> int:
        return int(self.module.max_memory_allocated(self.device))

    def profiler(
        self,
        output_dir: str,
        worker_name: str,
        *,
        record_shapes: bool,
        profile_memory: bool,
    ) -> Any:
        """Create the native profiler for a worker without changing the profiled workload."""
        if self.backend == "npu":
            profiler: Any = importlib.import_module("torch_npu.profiler")
            experimental_config = profiler._ExperimentalConfig(
                profiler_level=profiler.ProfilerLevel.Level1,
                aic_metrics=profiler.AiCMetrics.PipeUtilization,
                data_simplification=True,
                export_type=[profiler.ExportType.Text, profiler.ExportType.Db],
            )
            return profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                on_trace_ready=profiler.tensorboard_trace_handler(output_dir, worker_name=worker_name),
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=False,
                experimental_config=experimental_config,
            )
        return torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir, worker_name=worker_name),
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=False,
        )

    def device_name(self) -> str:
        get_device_name = getattr(self.module, "get_device_name", None)
        if get_device_name is not None:
            return str(get_device_name(self.device))
        properties = self.module.get_device_properties(self.device)
        return str(getattr(properties, "name", properties))


def select_accelerator(spec: str = "auto") -> AcceleratorRuntime:
    """Resolve ``auto``, a backend name, or an indexed device such as ``npu:1``."""
    requested, separator, index_text = spec.strip().lower().partition(":")
    if separator:
        try:
            index = int(index_text)
        except ValueError as exc:
            raise ValueError(f"设备编号必须是整数：{spec!r}") from exc
        if index < 0:
            raise ValueError(f"设备编号不能为负数：{spec!r}")
    else:
        index = 0

    npu_module = _npu_module()
    npu_available = bool(npu_module is not None and npu_module.is_available())
    cuda_available = bool(torch.cuda.is_available())
    backend = choose_backend(requested, npu_available=npu_available, cuda_available=cuda_available)
    module = npu_module if backend == "npu" else torch.cuda
    assert module is not None
    device_type = "npu" if backend == "npu" else "cuda"
    os.environ["VERL_PLATFORM"] = "huawei" if backend == "npu" else "nvidia"
    runtime = AcceleratorRuntime(backend=backend, device=torch.device(f"{device_type}:{index}"), module=module)
    if index >= int(module.device_count()):
        raise RuntimeError(f"{runtime.display_backend} 设备 {index} 不存在，可用设备数为 {module.device_count()}。")
    return runtime
