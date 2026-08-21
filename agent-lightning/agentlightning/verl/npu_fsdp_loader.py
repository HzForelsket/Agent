# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""Memory-bounded FSDP checkpoint loading for Huawei NPU workers."""

from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from verl.utils.fsdp_utils import parallel_init_module_fn
from verl.utils.model import convert_weight_keys
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine.fsdp import transformer_impl as fsdp_transformer_impl
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

__all__ = [
    "NPUShardedLoadActorRolloutRefWorker",
    "NPUShardedLoadFSDPEngineWithLMHead",
    "NPUShardedLoadFSDPEngineWithValueHead",
    "NPUShardedLoadTrainingWorker",
]


def _checkpoint_weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"无效的 safetensors 索引：{index_path}")
        result = {str(name): str(filename) for name, filename in weight_map.items()}
    else:
        checkpoint_path = model_path / "model.safetensors"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"NPU 分片加载需要 {index_path.name} 或 {checkpoint_path.name}，但 {model_path} 中均不存在。"
            )
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
            result = {name: checkpoint_path.name for name in checkpoint.keys()}

    missing_files = sorted({filename for filename in result.values() if not (model_path / filename).is_file()})
    if missing_files:
        raise FileNotFoundError(f"模型 checkpoint 文件不完整：{missing_files}")
    return result


def _balanced_parameter_sources(model_path: Path, weight_map: dict[str, str], world_size: int) -> dict[str, int]:
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name, filename in weight_map.items():
        names_by_file[filename].append(name)

    parameter_sizes: dict[str, int] = {}
    for filename, names in names_by_file.items():
        with safe_open(str(model_path / filename), framework="pt", device="cpu") as checkpoint:
            for name in names:
                shape = checkpoint.get_slice(name).get_shape()
                parameter_sizes[name] = int(torch.Size(shape).numel())

    rank_sizes = [0] * world_size
    sources: dict[str, int] = {}
    for name in sorted(parameter_sizes, key=lambda item: (-parameter_sizes[item], item)):
        source = min(range(world_size), key=lambda rank: (rank_sizes[rank], rank))
        sources[name] = source
        rank_sizes[source] += parameter_sizes[name]
    return sources


def _load_rank_checkpoint_state(model_path: Path) -> dict[str, torch.Tensor | int]:
    """Load only this rank's balanced subset of safetensors into CPU memory."""
    weight_map = _checkpoint_weight_map(model_path)
    rank = dist.get_rank()
    sources = _balanced_parameter_sources(model_path, weight_map, dist.get_world_size())
    state: dict[str, torch.Tensor | int] = {name: source for name, source in sources.items()}
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name, filename in weight_map.items():
        if sources[name] == rank:
            names_by_file[filename].append(name)
    for filename, names in names_by_file.items():
        with safe_open(str(model_path / filename), framework="pt", device="cpu") as checkpoint:
            for name in names:
                state[name] = checkpoint.get_tensor(name)
    return state


def _nonpersistent_buffers(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    persistent_names = set(module.state_dict())
    return {
        name: buffer.detach().cpu().clone() for name, buffer in module.named_buffers() if name not in persistent_names
    }


def _nonpersistent_buffer_names(module: torch.nn.Module) -> set[str]:
    persistent_names = set(module.state_dict())
    return {name for name, _buffer in module.named_buffers() if name not in persistent_names}


class _NPUShardedLoadFSDPMixin:
    """Build a meta model and materialize one wrapped layer at a time on NPU."""

    _parallel_init_fn: Any

    def _build_module(self) -> torch.nn.Module:
        module = super()._build_module()
        if self.model_config.lora_rank > 0:
            raise ValueError("NPU 分片加载当前不支持 LoRA；请使用完整参数训练。")
        if self._qat_enabled:
            raise ValueError("NPU 分片加载当前不支持 QAT。")

        state = _load_rank_checkpoint_state(Path(self.model_config.local_path))
        state = convert_weight_keys(state, module)

        rank = dist.get_rank()
        persistent_state = module.state_dict()
        missing_names = set(persistent_state).difference(state)
        fallback_state = (
            {name: persistent_state[name].detach().cpu().clone() for name in missing_names}
            if rank == 0
            else {}
        )
        buffer_names = _nonpersistent_buffer_names(module)
        buffers = _nonpersistent_buffers(module) if rank == 0 else {}

        module.to_empty(device="meta")
        if hasattr(module, "tie_weights"):
            module.tie_weights()
        del persistent_state
        gc.collect()

        for name in missing_names:
            state[name] = fallback_state[name] if rank == 0 else 0
        for name in buffer_names:
            state[name] = buffers[name] if rank == 0 else 0

        self._parallel_init_fn = parallel_init_module_fn(module, state)
        return module

    def _build_fsdp_module(self, module: torch.nn.Module) -> torch.nn.Module:
        original_init_fn = fsdp_transformer_impl.init_fn
        fsdp_transformer_impl.init_fn = self._parallel_init_fn
        try:
            return super()._build_fsdp_module(module)
        finally:
            fsdp_transformer_impl.init_fn = original_init_fn
            del self._parallel_init_fn


@EngineRegistry.register(model_type="language_model", backend="fsdp", device="npu", vendor="huawei")
class NPUShardedLoadFSDPEngineWithLMHead(_NPUShardedLoadFSDPMixin, FSDPEngineWithLMHead):
    """Ascend FSDP language-model engine with distributed checkpoint staging."""


@EngineRegistry.register(model_type="value_model", backend="fsdp", device="npu", vendor="huawei")
class NPUShardedLoadFSDPEngineWithValueHead(
    _NPUShardedLoadFSDPMixin, fsdp_transformer_impl.FSDPEngineWithValueHead
):
    """Ascend FSDP value-model engine with distributed checkpoint staging."""


class NPUShardedLoadTrainingWorker(TrainingWorker):
    """Training worker whose module import registers both Ascend FSDP engines."""


class NPUShardedLoadActorRolloutRefWorker(ActorRolloutRefWorker):
    """Actor/rollout/reference worker using the Ascend sharded-load engine."""

    actor_worker_cls = NPUShardedLoadTrainingWorker
    ref_worker_cls = NPUShardedLoadTrainingWorker
