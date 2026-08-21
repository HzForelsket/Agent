# Copyright (c) Microsoft. All rights reserved.

# pyright: reportPrivateUsage=false

"""Tests for NPU model materialization on hosts without CA certificates."""

from __future__ import annotations

import ssl
from pathlib import Path
from typing import Any

import huggingface_hub.constants as hub_constants
import huggingface_hub.utils._http as hub_http
import pytest
from omegaconf import OmegaConf

from agentlightning.verl import model_download


def test_remote_model_is_materialized_under_command_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Path, bool, tuple[str, ...] | None]] = []

    def fake_download(
        model_ref: str,
        local_dir: Path,
        *,
        local_files_only: bool,
        allow_patterns: tuple[str, ...] | None,
    ) -> str:
        calls.append((model_ref, local_dir, local_files_only, allow_patterns))
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(model_download, "_download_snapshot", fake_download)
    result = model_download.materialize_model_for_npu(
        "Qwen/Qwen2.5-0.5B-Instruct",
        tmp_path,
        allow_patterns=model_download.CONFIG_ONLY_PATTERNS,
    )

    expected = (tmp_path / "Qwen--Qwen2.5-0.5B-Instruct").resolve()
    assert result.local_path == str(expected)
    assert result.source == "huggingface-snapshot"
    assert result.scope == "config-only"
    assert result.tls_verification is False
    assert calls == [
        (
            "Qwen/Qwen2.5-0.5B-Instruct",
            expected,
            False,
            model_download.CONFIG_ONLY_PATTERNS,
        )
    ]


def test_existing_local_model_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    def unexpected_download(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("existing local directories must not be downloaded")

    monkeypatch.setattr(model_download, "_download_snapshot", unexpected_download)
    result = model_download.materialize_model_for_npu(str(model_dir), tmp_path)
    assert result.local_path == str(model_dir.resolve())
    assert result.source == "existing-local"
    assert result.tls_verification is None

    with pytest.raises(FileNotFoundError, match="本地模型目录不存在"):
        model_download.materialize_model_for_npu("./missing-model", tmp_path)


def test_local_files_only_copies_global_hub_cache_into_command_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached_snapshot = tmp_path / "hub-cache" / "snapshot"
    cached_snapshot.mkdir(parents=True)
    (cached_snapshot / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    local_dir = tmp_path / "command" / "org--model"
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(cached_snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    result = model_download._download_snapshot(
        "org/model",
        local_dir,
        local_files_only=True,
        allow_patterns=model_download.CONFIG_ONLY_PATTERNS,
    )

    assert result == str(local_dir)
    assert (local_dir / "config.json").read_text(encoding="utf-8") == '{"model_type": "test"}'
    assert calls == [
        {
            "repo_id": "org/model",
            "local_files_only": True,
            "allow_patterns": list(model_download.CONFIG_ONLY_PATTERNS),
        }
    ]


def test_insecure_client_disables_verification_only_inside_download_scope() -> None:
    previous_factory = hub_http._GLOBAL_CLIENT_FACTORY
    previous_xet = hub_constants.HF_HUB_DISABLE_XET

    with model_download._insecure_huggingface_client():
        client = hub_http.get_session()
        transport: Any = client._transport
        ssl_context: Any = transport._pool._ssl_context
        assert ssl_context.verify_mode == ssl.CERT_NONE
        assert ssl_context.check_hostname is False
        assert hub_constants.HF_HUB_DISABLE_XET is True

    assert hub_http._GLOBAL_CLIENT_FACTORY is previous_factory
    assert hub_constants.HF_HUB_DISABLE_XET is previous_xet


def test_verl_model_config_reuses_download_and_rewrites_all_remote_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": "org/main-model",
                    "hf_config_path": None,
                    "tokenizer_path": "org/main-model",
                    "lora_adapter_path": "org/adapter",
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
        allow_patterns: tuple[str, ...] | None = None,
    ) -> model_download.ModelMaterialization:
        del allow_patterns
        calls.append((model_ref, local_files_only))
        return model_download.ModelMaterialization(
            model_ref=model_ref,
            local_path=str((download_root / model_ref.replace("/", "--")).resolve()),
            source="huggingface-snapshot",
            scope="full-snapshot",
            tls_verification=False,
        )

    monkeypatch.setattr(model_download, "materialize_model_for_npu", fake_materialize)
    results = model_download.materialize_npu_model_config(config, tmp_path, local_files_only=True)

    main_path = str((tmp_path / "org--main-model").resolve())
    adapter_path = str((tmp_path / "org--adapter").resolve())
    assert config.actor_rollout_ref.model.path == main_path
    assert config.actor_rollout_ref.model.hf_config_path == main_path
    assert config.actor_rollout_ref.model.tokenizer_path == main_path
    assert config.actor_rollout_ref.model.lora_adapter_path == adapter_path
    assert calls == [("org/main-model", True), ("org/adapter", True)]
    assert [item.model_ref for item in results] == ["org/main-model", "org/adapter"]
