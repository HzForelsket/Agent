# Copyright (c) Microsoft. All rights reserved.

# pyright: reportUnknownVariableType=false

"""Materialize Hugging Face models for Ascend hosts without CA certificates."""

from __future__ import annotations

import logging
import os
import shutil
import ssl
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Sequence

logger = logging.getLogger(__name__)

NPU_MODEL_PROXY_ENV = "AGENTLIGHTNING_NPU_MODEL_PROXY"
NPU_MODEL_PROXY_USERNAME_ENV = "AGENTLIGHTNING_NPU_MODEL_PROXY_USERNAME"
NPU_MODEL_PROXY_PASSWORD_ENV = "AGENTLIGHTNING_NPU_MODEL_PROXY_PASSWORD"

_STANDARD_PROXY_ENV_VARS = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)

CONFIG_ONLY_PATTERNS = (
    "*.json",
    "*.py",
)

_MODEL_PATH_FIELDS = (
    "path",
    "hf_config_path",
    "tokenizer_path",
    "lora_adapter_path",
)

__all__ = [
    "CONFIG_ONLY_PATTERNS",
    "ModelMaterialization",
    "NPU_MODEL_PROXY_ENV",
    "NPU_MODEL_PROXY_PASSWORD_ENV",
    "NPU_MODEL_PROXY_USERNAME_ENV",
    "materialize_model_for_npu",
    "materialize_npu_model_config",
]


@dataclass(frozen=True)
class ModelMaterialization:
    """Resolved local model path and the security mode used to obtain it."""

    model_ref: str
    local_path: str
    source: str
    scope: str
    tls_verification: bool | None


def _looks_like_missing_local_path(model_ref: str) -> bool:
    expanded = Path(model_ref).expanduser()
    return expanded.is_absolute() or model_ref.startswith(("./", "../", "~/"))


def _model_directory(download_root: Path, model_ref: str) -> Path:
    directory_name = model_ref.replace("/", "--").replace("\\", "--")
    if not directory_name or directory_name in {".", ".."}:
        raise ValueError(f"无效的 Hugging Face 模型 ID：{model_ref!r}")
    resolved_root = download_root.resolve()
    local_dir = (resolved_root / directory_name).resolve()
    if not local_dir.is_relative_to(resolved_root):
        raise ValueError(f"模型目录不能位于下载目录之外：{local_dir}")
    return local_dir


def _model_proxy() -> tuple[str | None, tuple[str, str] | None]:
    proxy_url = os.environ.get(NPU_MODEL_PROXY_ENV)
    if not proxy_url:
        proxy_url = next((os.environ[name] for name in _STANDARD_PROXY_ENV_VARS if os.environ.get(name)), None)

    username = os.environ.get(NPU_MODEL_PROXY_USERNAME_ENV)
    password = os.environ.get(NPU_MODEL_PROXY_PASSWORD_ENV)
    if bool(username) != bool(password):
        raise RuntimeError(f"{NPU_MODEL_PROXY_USERNAME_ENV} 和 {NPU_MODEL_PROXY_PASSWORD_ENV} 必须同时设置。")
    if username and password and proxy_url is None:
        raise RuntimeError(f"已设置代理账号密码，但没有设置 {NPU_MODEL_PROXY_ENV} 或 HTTPS_PROXY。")
    return proxy_url, (username, password) if username and password else None


def _is_proxy_authentication_error(error: BaseException) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        message = str(current)
        if "407" in message and "Proxy Authentication Required" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


@contextmanager
def _insecure_huggingface_client() -> Generator[None, None, None]:
    """Temporarily use an httpx client without CA verification and without Xet."""
    import httpx
    import huggingface_hub.constants as hub_constants
    import huggingface_hub.utils._http as hub_http
    from huggingface_hub import set_client_factory

    previous_factory = hub_http._GLOBAL_CLIENT_FACTORY  # pyright: ignore[reportPrivateUsage]
    previous_disable_xet = hub_constants.HF_HUB_DISABLE_XET
    previous_disable_xet_env = os.environ.get("HF_HUB_DISABLE_XET")
    proxy_url, proxy_auth = _model_proxy()

    def insecure_client_factory() -> httpx.Client:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        proxy = (
            httpx.Proxy(
                proxy_url,
                ssl_context=ssl_context if proxy_url.startswith("https://") else None,
                auth=proxy_auth,
            )
            if proxy_url
            else None
        )
        return httpx.Client(
            verify=ssl_context,
            trust_env=False,
            proxy=proxy,
            event_hooks={"request": [hub_http.hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
        )

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    hub_constants.HF_HUB_DISABLE_XET = True
    set_client_factory(insecure_client_factory)
    try:
        yield
    finally:
        set_client_factory(previous_factory)
        hub_constants.HF_HUB_DISABLE_XET = previous_disable_xet
        if previous_disable_xet_env is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = previous_disable_xet_env


def _download_snapshot(
    model_ref: str,
    local_dir: Path,
    *,
    local_files_only: bool,
    allow_patterns: Sequence[str] | None,
) -> str:
    from huggingface_hub import snapshot_download

    patterns = list(allow_patterns) if allow_patterns is not None else None
    if local_files_only:
        cached_path = snapshot_download(
            repo_id=model_ref,
            local_files_only=True,
            allow_patterns=patterns,
        )
        shutil.copytree(cached_path, local_dir, dirs_exist_ok=True)
        return str(local_dir)

    proxy_url, _ = _model_proxy()
    logger.warning(
        "NPU 主机没有可用 CA 证书：下载 %s 时仅在当前进程内关闭 TLS 证书校验%s；不会修改系统证书。",
        model_ref,
        "并使用已配置的代理" if proxy_url else "",
    )
    with _insecure_huggingface_client():
        return snapshot_download(
            repo_id=model_ref,
            local_dir=local_dir,
            local_files_only=False,
            allow_patterns=patterns,
        )


def materialize_model_for_npu(
    model_ref: str,
    download_root: Path,
    *,
    local_files_only: bool = False,
    allow_patterns: Sequence[str] | None = None,
) -> ModelMaterialization:
    """Resolve a local path or download a Hub repository beneath ``download_root``."""
    expanded = Path(model_ref).expanduser()
    if expanded.exists():
        local_path = expanded.resolve()
        if not local_path.is_dir():
            raise ValueError(f"模型路径必须是目录：{local_path}")
        return ModelMaterialization(
            model_ref=model_ref,
            local_path=str(local_path),
            source="existing-local",
            scope="local",
            tls_verification=None,
        )
    if _looks_like_missing_local_path(model_ref):
        raise FileNotFoundError(f"本地模型目录不存在：{expanded}")

    root = download_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    local_dir = _model_directory(root, model_ref)
    scope = "config-only" if allow_patterns is not None else "full-snapshot"
    try:
        downloaded_path = Path(
            _download_snapshot(
                model_ref,
                local_dir,
                local_files_only=local_files_only,
                allow_patterns=allow_patterns,
            )
        ).resolve()
    except Exception as exc:
        mode = "仅本地缓存" if local_files_only else "已关闭当前下载客户端的 TLS 证书校验"
        proxy_hint = (
            f"；代理返回 407，请通过 {NPU_MODEL_PROXY_USERNAME_ENV} 和 " f"{NPU_MODEL_PROXY_PASSWORD_ENV} 提供有效凭据"
            if _is_proxy_authentication_error(exc)
            else ""
        )
        raise RuntimeError(
            f"无法把 NPU 模型 {model_ref!r} 下载到 {local_dir}（{mode}{proxy_hint}）。"
            "请检查网络、HF_TOKEN、仓库权限和磁盘空间。"
        ) from exc
    return ModelMaterialization(
        model_ref=model_ref,
        local_path=str(downloaded_path),
        source="huggingface-snapshot",
        scope=scope,
        tls_verification=None if local_files_only else False,
    )


def materialize_npu_model_config(
    config: Any,
    download_root: Path,
    *,
    local_files_only: bool = False,
) -> list[ModelMaterialization]:
    """Download all explicit VERL model references and rewrite them to local paths."""
    model_config = config.actor_rollout_ref.model
    original_main_path = str(model_config.path)
    materialized_by_ref: dict[str, ModelMaterialization] = {}

    for field in _MODEL_PATH_FIELDS:
        value = model_config.get(field)
        if value is None:
            continue
        model_ref = str(value)
        result = materialized_by_ref.get(model_ref)
        if result is None:
            result = materialize_model_for_npu(
                model_ref,
                download_root,
                local_files_only=local_files_only,
            )
            materialized_by_ref[model_ref] = result
        model_config[field] = result.local_path

    # VERL derives these paths from ``path`` only when they are null. Preserve that
    # behavior after replacing the main repository ID with an absolute local path.
    main_result = materialized_by_ref[original_main_path]
    for field in ("hf_config_path", "tokenizer_path"):
        if model_config.get(field) is None:
            model_config[field] = main_result.local_path

    return list(materialized_by_ref.values())
