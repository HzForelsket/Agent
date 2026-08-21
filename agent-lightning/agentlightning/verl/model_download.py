# Copyright (c) Microsoft. All rights reserved.

"""Materialize model Git repositories on Ascend hosts without CA certificates."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

NPU_MODEL_BASE_URL_ENV = "AGENTLIGHTNING_NPU_MODEL_BASE_URL"
NPU_MODEL_PROXY_ENV = "AGENTLIGHTNING_NPU_MODEL_PROXY"
NPU_MODEL_PROXY_USERNAME_ENV = "AGENTLIGHTNING_NPU_MODEL_PROXY_USERNAME"
NPU_MODEL_PROXY_PASSWORD_ENV = "AGENTLIGHTNING_NPU_MODEL_PROXY_PASSWORD"

DEFAULT_MODEL_BASE_URL = "https://www.modelscope.cn"

_STANDARD_PROXY_ENV_VARS = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)
_MODEL_PATH_FIELDS = ("path", "hf_config_path", "tokenizer_path", "lora_adapter_path")
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"

__all__ = [
    "DEFAULT_MODEL_BASE_URL",
    "ModelMaterialization",
    "NPU_MODEL_BASE_URL_ENV",
    "NPU_MODEL_PROXY_ENV",
    "NPU_MODEL_PROXY_PASSWORD_ENV",
    "NPU_MODEL_PROXY_USERNAME_ENV",
    "materialize_model_for_npu",
    "materialize_npu_model_config",
]


@dataclass(frozen=True)
class ModelMaterialization:
    """Resolved local model path and the transport used to obtain it."""

    model_ref: str
    local_path: str
    source: str
    scope: str
    tls_verification: bool | None


@dataclass(frozen=True)
class LfsPointer:
    sha256: str
    size: int


def _looks_like_missing_local_path(model_ref: str) -> bool:
    expanded = Path(model_ref).expanduser()
    return expanded.is_absolute() or model_ref.startswith(("./", "../", "~/"))


def _model_directory(download_root: Path, model_ref: str) -> Path:
    directory_name = model_ref.removesuffix(".git").replace("/", "--").replace("\\", "--")
    if not directory_name or directory_name in {".", ".."}:
        raise ValueError(f"无效的模型仓库 ID：{model_ref!r}")
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


def _authenticated_proxy_url(proxy_url: str, auth: tuple[str, str] | None) -> str:
    if auth is None:
        return proxy_url
    parsed = urlsplit(proxy_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("模型下载代理必须是 http:// 或 https:// URL。")
    username, password = auth
    hostname = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = f"{quote(username, safe='')}:{quote(password, safe='')}@{hostname}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _download_environment() -> tuple[dict[str, str], tuple[str, ...]]:
    environment = os.environ.copy()
    proxy_url, proxy_auth = _model_proxy()
    for name in _STANDARD_PROXY_ENV_VARS:
        environment.pop(name, None)

    secrets: list[str] = []
    if proxy_url:
        authenticated_proxy = _authenticated_proxy_url(proxy_url, proxy_auth)
        environment["https_proxy"] = authenticated_proxy
        environment["http_proxy"] = authenticated_proxy
        secrets.extend(value for value in (authenticated_proxy, proxy_url) if value)

    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    return environment, tuple(secrets)


def _redact(text: str, secrets: Sequence[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _run_download_command(
    command: Sequence[str],
    *,
    environment: dict[str, str],
    secrets: Sequence[str],
    cwd: Path | None = None,
) -> str:
    completed = subprocess.run(list(command), cwd=cwd, env=environment, capture_output=True, text=True, check=False)
    if completed.returncode:
        details = _redact((completed.stderr or completed.stdout).strip(), secrets)
        if "407" in details and "Proxy Authentication Required" in details:
            details = (
                f"代理返回 407；请通过 {NPU_MODEL_PROXY_USERNAME_ENV} 和 "
                f"{NPU_MODEL_PROXY_PASSWORD_ENV} 提供有效凭据"
            )
        raise RuntimeError(details or f"下载命令退出码为 {completed.returncode}")
    return completed.stdout.strip()


def _parse_lfs_pointer(path: Path) -> LfsPointer | None:
    if not path.is_file() or path.stat().st_size > 1024:
        return None
    content = path.read_bytes()
    if not content.startswith(_LFS_HEADER):
        return None
    values: dict[str, str] = {}
    for line in content.decode("ascii").splitlines()[1:]:
        key, _, value = line.partition(" ")
        values[key] = value
    oid = values.get("oid", "")
    size = values.get("size", "")
    if not oid.startswith("sha256:") or not size.isdigit():
        raise ValueError(f"无效的 Git LFS 指针：{path}")
    return LfsPointer(sha256=oid.removeprefix("sha256:"), size=int(size))


def _lfs_files(local_dir: Path) -> list[tuple[Path, LfsPointer]]:
    pointers: list[tuple[Path, LfsPointer]] = []
    for path in local_dir.rglob("*"):
        if ".git" in path.relative_to(local_dir).parts:
            continue
        pointer = _parse_lfs_pointer(path)
        if pointer is not None:
            pointers.append((path, pointer))
    return pointers


def _verify_file(path: Path, pointer: LfsPointer) -> None:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    if size != pointer.size or digest.hexdigest() != pointer.sha256:
        raise RuntimeError(
            f"下载文件完整性校验失败：{path.name}，实际 size={size} sha256={digest.hexdigest()}，"
            f"期望 size={pointer.size} sha256={pointer.sha256}"
        )


def _clone_repository(model_ref: str, local_dir: Path, environment: dict[str, str], secrets: Sequence[str]) -> None:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("自动下载模型需要 git，但当前 PATH 中未找到 git。")
    base_url = os.environ.get(NPU_MODEL_BASE_URL_ENV, DEFAULT_MODEL_BASE_URL).rstrip("/")
    repository_url = f"{base_url}/{quote(model_ref.removesuffix('.git'), safe='/')}.git"
    _run_download_command(
        (
            git,
            "-c",
            "http.sslVerify=false",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.required=false",
            "clone",
            "--depth",
            "1",
            "--no-tags",
            repository_url,
            str(local_dir),
        ),
        environment=environment,
        secrets=secrets,
    )


def _git_revision(local_dir: Path, environment: dict[str, str], secrets: Sequence[str]) -> str:
    git = shutil.which("git")
    assert git is not None
    return _run_download_command(
        (git, "symbolic-ref", "--short", "HEAD"), cwd=local_dir, environment=environment, secrets=secrets
    )


def _download_lfs_files(
    model_ref: str,
    local_dir: Path,
    revision: str,
    environment: dict[str, str],
    secrets: Sequence[str],
) -> None:
    wget = shutil.which("wget")
    if wget is None:
        raise RuntimeError("下载模型权重需要 wget，但当前 PATH 中未找到 wget。")
    base_url = os.environ.get(NPU_MODEL_BASE_URL_ENV, DEFAULT_MODEL_BASE_URL).rstrip("/")
    repo_path = quote(model_ref.removesuffix(".git"), safe="/")
    pointers = _lfs_files(local_dir)
    for path, pointer in pointers:
        relative_path = path.relative_to(local_dir).as_posix()
        query = urlencode({"Revision": revision, "FilePath": relative_path})
        file_url = f"{base_url}/api/v1/models/{repo_path}/repo?{query}"
        partial_path = path.with_name(f"{path.name}.partial")
        logger.info("wget 从 ModelScope 下载模型文件：%s", relative_path)
        _run_download_command(
            (
                wget,
                "--no-check-certificate",
                "--continue",
                "--output-document",
                str(partial_path),
                file_url,
            ),
            environment=environment,
            secrets=secrets,
        )
        _verify_file(partial_path, pointer)
        os.replace(partial_path, path)


def _download_repository(
    model_ref: str,
    local_dir: Path,
    *,
    local_files_only: bool,
    download_weights: bool,
) -> str:
    if local_files_only:
        if not (local_dir / ".git").is_dir():
            raise FileNotFoundError(f"本地 Git 模型目录不存在：{local_dir}")
        if download_weights and _lfs_files(local_dir):
            raise RuntimeError(f"本地模型 {local_dir} 仍包含未下载的 Git LFS 指针。")
        return str(local_dir)

    environment, secrets = _download_environment()
    if local_dir.exists() and not (local_dir / ".git").is_dir() and any(local_dir.iterdir()):
        raise RuntimeError(f"模型目录已存在但不是 Git 仓库：{local_dir}")
    if not (local_dir / ".git").is_dir():
        _clone_repository(model_ref, local_dir, environment, secrets)

    revision = _git_revision(local_dir, environment, secrets)
    if download_weights:
        _download_lfs_files(model_ref, local_dir, revision, environment, secrets)
    return str(local_dir)


def materialize_model_for_npu(
    model_ref: str,
    download_root: Path,
    *,
    local_files_only: bool = False,
    download_weights: bool = True,
) -> ModelMaterialization:
    """Resolve a local path or obtain a model repository with Git and wget."""
    expanded = Path(model_ref).expanduser()
    if expanded.exists():
        local_path = expanded.resolve()
        if not local_path.is_dir():
            raise ValueError(f"模型路径必须是目录：{local_path}")
        if download_weights and _lfs_files(local_path):
            raise RuntimeError(f"本地模型 {local_path} 仍包含未下载的 Git LFS 指针。")
        return ModelMaterialization(model_ref, str(local_path), "existing-local", "local", None)
    if _looks_like_missing_local_path(model_ref):
        raise FileNotFoundError(f"本地模型目录不存在：{expanded}")

    root = download_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    local_dir = _model_directory(root, model_ref)
    scope = "git-metadata" if not download_weights else "full-repository"
    try:
        downloaded_path = Path(
            _download_repository(
                model_ref,
                local_dir,
                local_files_only=local_files_only,
                download_weights=download_weights,
            )
        ).resolve()
    except Exception as exc:
        raise RuntimeError(
            f"无法用 Git/wget 把 NPU 模型 {model_ref!r} 下载到 {local_dir}。"
            f"请检查代理凭据、{NPU_MODEL_BASE_URL_ENV}、仓库权限和磁盘空间：{exc}"
        ) from exc
    return ModelMaterialization(
        model_ref=model_ref,
        local_path=str(downloaded_path),
        source="git-wget",
        scope=scope,
        tls_verification=None if local_files_only else False,
    )


def materialize_npu_model_config(
    config: Any,
    download_root: Path,
    *,
    local_files_only: bool = False,
) -> list[ModelMaterialization]:
    """Download explicit VERL model repositories and rewrite them to local paths."""
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
            result = materialize_model_for_npu(model_ref, download_root, local_files_only=local_files_only)
            materialized_by_ref[model_ref] = result
        model_config[field] = result.local_path

    main_result = materialized_by_ref[original_main_path]
    for field in ("hf_config_path", "tokenizer_path"):
        if model_config.get(field) is None:
            model_config[field] = main_result.local_path
    return list(materialized_by_ref.values())
