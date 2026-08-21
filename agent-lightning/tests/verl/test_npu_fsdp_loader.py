# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""Tests for memory-bounded Ascend FSDP checkpoint loading."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from verl.workers.engine import EngineRegistry

from agentlightning.verl import npu_fsdp_loader


def test_safetensor_parameters_are_balanced_across_rank_cpu_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensors = {
        "model.large.weight": torch.arange(48, dtype=torch.float32).reshape(8, 6),
        "model.medium.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "model.small.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
    }
    save_file(tensors, tmp_path / "model.safetensors")
    monkeypatch.setattr(npu_fsdp_loader.dist, "get_world_size", lambda: 2)

    states: list[dict[str, torch.Tensor | int]] = []
    for rank in range(2):
        monkeypatch.setattr(npu_fsdp_loader.dist, "get_rank", lambda rank=rank: rank)
        states.append(npu_fsdp_loader._load_rank_checkpoint_state(tmp_path))

    local_names = [{name for name, value in state.items() if torch.is_tensor(value)} for state in states]
    assert local_names[0].isdisjoint(local_names[1])
    assert local_names[0] | local_names[1] == set(tensors)
    for rank, state in enumerate(states):
        for name, expected in tensors.items():
            value = state[name]
            if name in local_names[rank]:
                torch.testing.assert_close(value, expected)
            else:
                assert value in {0, 1}


def test_huawei_npu_uses_sharded_load_engine() -> None:
    language_engine = EngineRegistry._engines["language_model"]["fsdp"][("npu", "huawei")]
    value_engine = EngineRegistry._engines["value_model"]["fsdp"][("npu", "huawei")]
    assert language_engine is npu_fsdp_loader.NPUShardedLoadFSDPEngineWithLMHead
    assert value_engine is npu_fsdp_loader.NPUShardedLoadFSDPEngineWithValueHead


def test_npu_worker_classes_register_sharded_loading_for_actor_ref_and_critic() -> None:
    actor_worker = npu_fsdp_loader.NPUShardedLoadActorRolloutRefWorker
    assert actor_worker.actor_worker_cls is npu_fsdp_loader.NPUShardedLoadTrainingWorker
    assert actor_worker.ref_worker_cls is npu_fsdp_loader.NPUShardedLoadTrainingWorker


def test_sharded_loader_requires_safetensors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="NPU 分片加载需要"):
        npu_fsdp_loader._checkpoint_weight_map(tmp_path)
