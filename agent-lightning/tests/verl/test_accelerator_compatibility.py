# Copyright (c) Microsoft. All rights reserved.

"""Tests for PrefixGrouper CUDA/Ascend compatibility selection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentlightning.verl.accelerator import choose_backend, select_accelerator
from agentlightning.verl.entrypoint import (
    configure_accelerator,
    invocation_directory,
    prepare_model_for_accelerator,
)
from agentlightning.verl.model_download import ModelMaterialization

_STACK_SPEC = importlib.util.spec_from_file_location(
    "prefix_grouper_stack", Path(__file__).resolve().parents[2] / "scripts" / "prefix_grouper_stack.py"
)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
_STACK_MODULE: Any = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = _STACK_MODULE
_STACK_SPEC.loader.exec_module(_STACK_MODULE)
NPU_CANN_VERSION = _STACK_MODULE.NPU_CANN_VERSION
REQUIRED_STACKS = _STACK_MODULE.REQUIRED_STACKS
build_install_plan = _STACK_MODULE.build_install_plan
detect_backend = _STACK_MODULE.detect_backend


def test_auto_backend_prefers_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERL_PLATFORM", raising=False)
    assert choose_backend("auto", npu_available=True, cuda_available=True) == "npu"
    assert choose_backend("auto", npu_available=False, cuda_available=True) == "gpu"


def test_backend_override_and_unavailable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERL_PLATFORM", "nvidia")
    assert choose_backend("auto", npu_available=True, cuda_available=True) == "gpu"
    with pytest.raises(RuntimeError, match="torch.npu.is_available"):
        choose_backend("npu", npu_available=False, cuda_available=True)


def test_hardware_probe_prefers_npu() -> None:
    calls: list[str] = []

    def probe(command: str, *_args: str) -> bool:
        calls.append(command)
        return command in {"npu-smi", "nvidia-smi"}

    assert detect_backend(probe) == "npu"
    assert calls == ["npu-smi"]


def test_hardware_probe_falls_back_to_gpu() -> None:
    def probe(command: str, *_args: str) -> bool:
        return command == "nvidia-smi"

    assert detect_backend(probe) == "gpu"


def test_install_matrices_and_npu_vllm_source_plan() -> None:
    assert REQUIRED_STACKS["gpu"]["torch"] == "2.11.0"
    assert REQUIRED_STACKS["npu"]["torch"] == "2.10.0"
    assert REQUIRED_STACKS["npu"]["vllm_ascend"] == "0.22.1rc1"
    assert NPU_CANN_VERSION == "9.0.0"

    plan = build_install_plan("npu", Path("/repo"), python="/python")
    arctic_index = next(index for index, command in enumerate(plan) if "arctic-inference==0.1.1" in command.argv)
    vllm_command = next(
        command for command in plan if any("git+https://github.com/vllm-project/vllm" in arg for arg in command.argv)
    )
    requirements_index = next(
        index
        for index, command in enumerate(plan)
        if "/repo/scripts/requirements_prefix_grouper_npu.txt" in command.argv
    )
    assert arctic_index < requirements_index
    assert "--no-build-isolation" in plan[arctic_index].argv
    assert vllm_command.env == {"VLLM_TARGET_DEVICE": "empty"}
    assert "--no-build-isolation" in vllm_command.argv


def test_npu_install_plan_supports_aarch64(monkeypatch: pytest.MonkeyPatch) -> None:
    def aarch64() -> str:
        return "aarch64"

    monkeypatch.setattr(_STACK_MODULE.platform, "machine", aarch64)
    torch_command = build_install_plan("npu", Path("/repo"), python="/python")[0]
    assert "https://mirrors.huaweicloud.com/ascend/repos/pypi/variant" in torch_command.argv
    assert "torch==2.10.0" in torch_command.argv


def test_agentlightning_delegates_device_selection_to_verl(monkeypatch: pytest.MonkeyPatch) -> None:
    import verl.utils.device

    config = SimpleNamespace(trainer=SimpleNamespace(device="cuda"))

    def fake_auto_set_device(received: SimpleNamespace) -> None:
        assert received is config
        received.trainer.device = "npu"

    monkeypatch.setattr(verl.utils.device, "auto_set_device", fake_auto_set_device)
    assert configure_accelerator(config) == "npu"


def test_invocation_directory_without_hydra_is_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert invocation_directory() == tmp_path.resolve()


def test_npu_prepares_model_before_workers_and_gpu_skips_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentlightning.verl import model_download

    config = SimpleNamespace(
        agentlightning={"npu_model_download": {"enabled": True, "local_files_only": True}},
        actor_rollout_ref=SimpleNamespace(model=SimpleNamespace(path="org/model")),
    )
    calls: list[tuple[Any, Path, bool]] = []
    expected = ModelMaterialization(
        model_ref="org/model",
        local_path=str(tmp_path / "org--model"),
        source="git-wget",
        scope="full-repository",
        tls_verification=None,
    )

    def fake_materialize(received: Any, root: Path, *, local_files_only: bool) -> list[ModelMaterialization]:
        calls.append((received, root, local_files_only))
        return [expected]

    monkeypatch.setattr(model_download, "materialize_npu_model_config", fake_materialize)
    assert prepare_model_for_accelerator(config, "npu", tmp_path) == [expected]
    assert calls == [(config, tmp_path, True)]
    assert prepare_model_for_accelerator(config, "cuda", tmp_path) == []
    assert calls == [(config, tmp_path, True)]


def test_current_host_runtime_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERL_PLATFORM", raising=False)
    runtime = select_accelerator("auto")
    assert runtime.backend in {"gpu", "npu"}
    assert runtime.module.is_available()
