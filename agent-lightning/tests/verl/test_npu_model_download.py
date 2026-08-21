# Copyright (c) Microsoft. All rights reserved.

# pyright: reportPrivateUsage=false

"""Tests for ModelScope Git/wget downloads on NPU hosts without CA certificates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest
from omegaconf import OmegaConf

from agentlightning.verl import model_download


def _pointer(content: bytes) -> bytes:
    return (
        b"version https://git-lfs.github.com/spec/v1\n"
        + f"oid sha256:{hashlib.sha256(content).hexdigest()}\nsize {len(content)}\n".encode()
    )


def test_remote_model_is_materialized_under_command_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Path, bool, bool]] = []

    def fake_download(
        model_ref: str,
        local_dir: Path,
        *,
        local_files_only: bool,
        download_weights: bool,
    ) -> str:
        calls.append((model_ref, local_dir, local_files_only, download_weights))
        (local_dir / ".git").mkdir(parents=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(model_download, "_download_repository", fake_download)
    result = model_download.materialize_model_for_npu(
        "Qwen/Qwen2.5-0.5B-Instruct",
        tmp_path,
        download_weights=False,
    )

    expected = (tmp_path / "Qwen--Qwen2.5-0.5B-Instruct").resolve()
    assert result.local_path == str(expected)
    assert result.source == "git-wget"
    assert result.scope == "git-metadata"
    assert result.tls_verification is False
    assert calls == [
        (
            "Qwen/Qwen2.5-0.5B-Instruct",
            expected,
            False,
            False,
        )
    ]


def test_existing_local_model_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    def unexpected_download(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("existing local directories must not be downloaded")

    monkeypatch.setattr(model_download, "_download_repository", unexpected_download)
    result = model_download.materialize_model_for_npu(str(model_dir), tmp_path)
    assert result.local_path == str(model_dir.resolve())
    assert result.source == "existing-local"

    with pytest.raises(FileNotFoundError, match="本地模型目录不存在"):
        model_download.materialize_model_for_npu("./missing-model", tmp_path)


def test_local_files_only_requires_completed_git_checkout(tmp_path: Path) -> None:
    local_dir = tmp_path / "org--model"
    (local_dir / ".git").mkdir(parents=True)
    (local_dir / "config.json").write_text("{}", encoding="utf-8")
    assert model_download._download_repository(
        "org/model",
        local_dir,
        local_files_only=True,
        download_weights=False,
    ) == str(local_dir)

    (local_dir / "model.safetensors").write_bytes(_pointer(b"weights"))
    with pytest.raises(RuntimeError, match="Git LFS 指针"):
        model_download._download_repository("org/model", local_dir, local_files_only=True, download_weights=True)


def test_proxy_credentials_are_encoded_for_git_and_wget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.delenv(model_download.NPU_MODEL_PROXY_ENV, raising=False)
    monkeypatch.setenv(model_download.NPU_MODEL_PROXY_USERNAME_ENV, "domain/user")
    monkeypatch.setenv(model_download.NPU_MODEL_PROXY_PASSWORD_ENV, "p@ss word")

    environment, secrets = model_download._download_environment()
    expected = "http://domain%2Fuser:p%40ss%20word@proxy.internal:8080"
    assert environment["https_proxy"] == expected
    assert environment["http_proxy"] == expected
    assert "HTTPS_PROXY" not in environment
    assert environment["GIT_LFS_SKIP_SMUDGE"] == "1"
    assert expected in secrets


def test_proxy_credentials_must_be_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(model_download.NPU_MODEL_PROXY_ENV, "http://proxy.invalid:8080")
    monkeypatch.setenv(model_download.NPU_MODEL_PROXY_USERNAME_ENV, "user")
    monkeypatch.delenv(model_download.NPU_MODEL_PROXY_PASSWORD_ENV, raising=False)
    with pytest.raises(RuntimeError, match="必须同时设置"):
        model_download._model_proxy()


def test_download_command_redacts_proxy_and_explains_407(monkeypatch: pytest.MonkeyPatch) -> None:
    proxy = "http://user:password@proxy.internal:8080"

    def rejected(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=f"Connecting to {proxy}: 407 Proxy Authentication Required",
        )

    monkeypatch.setattr(model_download.subprocess, "run", rejected)
    with pytest.raises(RuntimeError) as caught:
        model_download._run_download_command(("wget", "url"), environment={}, secrets=(proxy,))
    message = str(caught.value)
    assert "代理返回 407" in message
    assert "password" not in message


def test_lfs_pointer_is_parsed_and_download_is_verified(tmp_path: Path) -> None:
    content = b"verified model weights"
    pointer_path = tmp_path / "model.safetensors"
    pointer_path.write_bytes(_pointer(content))
    pointer = model_download._parse_lfs_pointer(pointer_path)
    assert pointer is not None
    assert pointer == model_download.LfsPointer(sha256=hashlib.sha256(content).hexdigest(), size=len(content))

    downloaded = tmp_path / "model.safetensors.partial"
    downloaded.write_bytes(content)
    model_download._verify_file(downloaded, pointer)
    downloaded.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="完整性校验失败"):
        model_download._verify_file(downloaded, pointer)


def test_git_clone_disables_ssl_and_lfs_smudge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Sequence[str], dict[str, str]]] = []

    def fake_run(
        command: Sequence[str],
        *,
        environment: dict[str, str],
        secrets: Sequence[str],
        cwd: Path | None = None,
    ) -> str:
        del secrets, cwd
        calls.append((command, environment))
        return ""

    def fake_which(name: str) -> str:
        return f"/usr/bin/{name}"

    monkeypatch.setattr(model_download.shutil, "which", fake_which)
    monkeypatch.setattr(model_download, "_run_download_command", fake_run)
    monkeypatch.delenv(model_download.NPU_MODEL_BASE_URL_ENV, raising=False)
    model_download._clone_repository("Qwen/Test", tmp_path / "model", {}, ())

    command = calls[0][0]
    assert "http.sslVerify=false" in command
    assert "filter.lfs.smudge=" in command
    assert "filter.lfs.process=" in command
    assert "https://www.modelscope.cn/Qwen/Test.git" in command


def test_wget_uses_modelscope_api_and_replaces_verified_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"model weights from modelscope"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    target = model_dir / "weights" / "model.safetensors"
    target.parent.mkdir()
    target.write_bytes(_pointer(content))
    calls: list[Sequence[str]] = []

    def fake_run(
        command: Sequence[str],
        *,
        environment: dict[str, str],
        secrets: Sequence[str],
        cwd: Path | None = None,
    ) -> str:
        del environment, secrets, cwd
        calls.append(command)
        output_index = command.index("--output-document") + 1
        Path(command[output_index]).write_bytes(content)
        return ""

    def fake_which(_name: str) -> str:
        return "/usr/bin/wget"

    monkeypatch.setattr(model_download.shutil, "which", fake_which)
    monkeypatch.setattr(model_download, "_run_download_command", fake_run)
    monkeypatch.delenv(model_download.NPU_MODEL_BASE_URL_ENV, raising=False)
    model_download._download_lfs_files("Qwen/Test", model_dir, "master", {}, ())

    assert target.read_bytes() == content
    url = calls[0][-1]
    assert url.startswith("https://www.modelscope.cn/api/v1/models/Qwen/Test/repo?")
    assert "Revision=master" in url
    assert "FilePath=weights%2Fmodel.safetensors" in url
    assert "--no-check-certificate" in calls[0]


def test_verl_model_config_reuses_download_and_rewrites_all_remote_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": "Qwen/main-model",
                    "hf_config_path": None,
                    "tokenizer_path": "Qwen/main-model",
                    "lora_adapter_path": "Qwen/adapter",
                }
            }
        }
    )
    calls: list[tuple[str, bool]] = []

    def fake_materialize(
        model_ref: str,
        download_root: Path,
        *,
        local_files_only: bool = False,
        download_weights: bool = True,
    ) -> model_download.ModelMaterialization:
        del download_weights
        calls.append((model_ref, local_files_only))
        return model_download.ModelMaterialization(
            model_ref=model_ref,
            local_path=str((download_root / model_ref.replace("/", "--")).resolve()),
            source="git-wget",
            scope="full-repository",
            tls_verification=False,
        )

    monkeypatch.setattr(model_download, "materialize_model_for_npu", fake_materialize)
    results = model_download.materialize_npu_model_config(config, tmp_path, local_files_only=True)

    main_path = str((tmp_path / "Qwen--main-model").resolve())
    adapter_path = str((tmp_path / "Qwen--adapter").resolve())
    assert config.actor_rollout_ref.model.path == main_path
    assert config.actor_rollout_ref.model.hf_config_path == main_path
    assert config.actor_rollout_ref.model.tokenizer_path == main_path
    assert config.actor_rollout_ref.model.lora_adapter_path == adapter_path
    assert calls == [("Qwen/main-model", True), ("Qwen/adapter", True)]
    assert [item.model_ref for item in results] == ["Qwen/main-model", "Qwen/adapter"]


def test_verl_model_config_materializes_enabled_critic_and_reward_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": "Qwen/shared-model",
                    "hf_config_path": None,
                    "tokenizer_path": None,
                    "lora_adapter_path": None,
                }
            },
            "critic": {
                "enable": True,
                "model": {
                    "path": "Qwen/shared-model",
                    "hf_config_path": None,
                    "tokenizer_path": None,
                    "lora_adapter_path": None,
                },
            },
            "reward": {
                "reward_model": {
                    "enable": True,
                    "model_path": "Qwen/reward-model",
                }
            },
            "algorithm": {"adv_estimator": "gae"},
        }
    )
    calls: list[str] = []

    def fake_materialize(
        model_ref: str,
        download_root: Path,
        *,
        local_files_only: bool = False,
        download_weights: bool = True,
    ) -> model_download.ModelMaterialization:
        del local_files_only, download_weights
        calls.append(model_ref)
        return model_download.ModelMaterialization(
            model_ref=model_ref,
            local_path=str((download_root / model_ref.replace("/", "--")).resolve()),
            source="git-wget",
            scope="full-repository",
            tls_verification=False,
        )

    monkeypatch.setattr(model_download, "materialize_model_for_npu", fake_materialize)
    results = model_download.materialize_npu_model_config(config, tmp_path)

    shared_path = str((tmp_path / "Qwen--shared-model").resolve())
    reward_path = str((tmp_path / "Qwen--reward-model").resolve())
    assert config.actor_rollout_ref.model.path == shared_path
    assert config.critic.model.path == shared_path
    assert config.critic.model.hf_config_path == shared_path
    assert config.critic.model.tokenizer_path == shared_path
    assert config.reward.reward_model.model_path == reward_path
    assert calls == ["Qwen/shared-model", "Qwen/reward-model"]
    assert [item.model_ref for item in results] == ["Qwen/shared-model", "Qwen/reward-model"]
