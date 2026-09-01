# Copyright (c) Microsoft. All rights reserved.

"""Install the pinned PrefixGrouper stack after auto-detecting NPU or GPU."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import os
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

Backend = Literal["gpu", "npu"]

REQUIRED_STACKS: dict[Backend, dict[str, str]] = {
    "gpu": {
        "torch": "2.11.0",
        "transformers": "5.10.4",
        "vllm": "0.22.1",
        "verl": "0.9.0",
        "prefix_grouper": "0.0.1.post1",
    },
    "npu": {
        "torch": "2.10.0",
        "torch_npu": "2.10.0",
        "transformers": "5.5.4",
        "triton_ascend": "3.2.1",
        "vllm": "0.22.1",
        "vllm_ascend": "0.22.1rc1",
        "verl": "0.9.0",
        "prefix_grouper": "0.0.1.post1",
    },
}
NPU_CANN_VERSION = "9.0.0"


def _empty_environment() -> dict[str, str]:
    return {}


@dataclass(frozen=True)
class InstallCommand:
    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=_empty_environment)
    action: Literal["run", "normalize-triton-metadata"] = "run"

    def display(self) -> str:
        if self.action == "normalize-triton-metadata":
            return "normalize triton-ascend 3.2.1 numpy metadata for the CPU development environment"
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in self.env.items())
        command = shlex.join(self.argv)
        return f"{prefix} {command}" if prefix else command


def _normalize_triton_ascend_metadata() -> None:
    """Apply the project's NumPy override while preserving the pinned wheel code."""
    distribution = importlib.metadata.distribution("triton-ascend")
    if distribution.version != "3.2.1":
        raise RuntimeError(f"Expected triton-ascend 3.2.1, found {distribution.version}.")
    files = distribution.files or ()
    metadata_entry = next((item for item in files if str(item).endswith(".dist-info/METADATA")), None)
    record_entry = next((item for item in files if str(item).endswith(".dist-info/RECORD")), None)
    if metadata_entry is None or record_entry is None:
        raise RuntimeError("Cannot locate triton-ascend installation metadata.")

    metadata_path = Path(distribution.locate_file(metadata_entry))
    record_path = Path(distribution.locate_file(record_entry))
    original = "Requires-Dist: numpy==1.26.4"
    replacement = "Requires-Dist: numpy>=2.0.0,<2.3.0"
    contents = metadata_path.read_text(encoding="utf-8")
    if replacement not in contents:
        if original not in contents:
            raise RuntimeError("Unexpected triton-ascend NumPy dependency metadata.")
        metadata_path.write_text(contents.replace(original, replacement, 1), encoding="utf-8")

    payload = metadata_path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")
    with record_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    metadata_record = str(metadata_entry)
    for row in rows:
        if row and row[0] == metadata_record:
            row[1:] = [f"sha256={digest}", str(len(payload))]
            break
    else:
        raise RuntimeError("triton-ascend RECORD does not contain its METADATA entry.")
    with record_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)


def _command_succeeds(command: str, *args: str) -> bool:
    executable = shutil.which(command)
    if executable is None:
        return False
    try:
        completed = subprocess.run(
            [executable, *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def detect_backend(probe: Callable[..., bool] = _command_succeeds) -> Backend:
    """Detect usable hardware, preferring Ascend when both tools are visible."""
    if probe("npu-smi", "info"):
        return "npu"
    if probe("nvidia-smi", "-L"):
        return "gpu"
    raise RuntimeError("未发现可用的昇腾 NPU 或 NVIDIA GPU；可用 --backend 显式指定目标。")


def build_install_plan(
    backend: Backend,
    repo_root: Path,
    python: str = sys.executable,
    *,
    cpu_dev: bool = False,
) -> list[InstallCommand]:
    if cpu_dev and backend != "npu":
        raise ValueError("无设备 CPU 开发模式仅支持 NPU 后端。")

    requirements = repo_root / "scripts" / f"requirements_prefix_grouper_{backend}.txt"
    pip = (python, "-m", "pip")
    commands: list[InstallCommand] = []
    if backend == "gpu":
        commands.append(InstallCommand((*pip, "install", "-r", str(requirements))))
    else:
        if platform.machine() == "x86_64":
            torch_command = (
                *pip,
                "install",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                "torch==2.10.0",
                "torchvision==0.25.0",
                "torchaudio==2.10.0",
            )
        else:
            torch_command = (
                *pip,
                "install",
                "--extra-index-url",
                "https://mirrors.huaweicloud.com/ascend/repos/pypi/variant",
                "torch==2.10.0",
                "torchvision==0.25.0",
                "torchaudio==2.10.0",
            )
        commands.extend(
            [
                InstallCommand(torch_command),
                InstallCommand(
                    (
                        *pip,
                        "install",
                        "setuptools>=77.0.3,<81.0.0",
                        "setuptools-scm>=8.0",
                        "setuptools-rust>=1.9.0",
                        "cmake>=3.26.1",
                        "ninja",
                        "wheel",
                        "jinja2",
                        "packaging>=24.2",
                        "nanobind==2.9.2",
                        "protobuf==7.36.0",
                        "grpcio-tools==1.83.1",
                    )
                ),
                InstallCommand((*pip, "install", "--no-build-isolation", "arctic-inference==0.1.1")),
            ]
        )
        if cpu_dev:
            ascend_indexes = (
                "--extra-index-url",
                "https://mirrors.huaweicloud.com/ascend/repos/pypi/variant",
                "--extra-index-url",
                "https://mirrors.huaweicloud.com/ascend/repos/pypi",
            )
            commands.extend(
                [
                    InstallCommand((*pip, "install", "--no-deps", *ascend_indexes, "triton-ascend==3.2.1")),
                    InstallCommand((), action="normalize-triton-metadata"),
                    InstallCommand(
                        (
                            *pip,
                            "install",
                            "--extra-index-url",
                            "https://download.pytorch.org/whl/cpu",
                            *ascend_indexes,
                            "torch-npu==2.10.0",
                            "transformers==5.5.4",
                            "arctic-inference==0.1.1",
                            "vllm-ascend==0.22.1rc1",
                            "verl==0.9.0",
                            "prefix_grouper==0.0.1.post1",
                        )
                    ),
                ]
            )
        else:
            commands.append(InstallCommand((*pip, "install", "-r", str(requirements))))
        commands.append(
            InstallCommand(
                (
                    *pip,
                    "install",
                    "--no-build-isolation",
                    "vllm @ git+https://github.com/vllm-project/vllm.git@v0.22.1",
                ),
                env={"VLLM_TARGET_DEVICE": "empty"},
            )
        )
    if cpu_dev:
        # These versions are compatible with both the repository lock and the
        # older upper bounds published by the pinned vLLM Ascend/VERL stack.
        commands.extend(
            [
                InstallCommand(
                    (
                        *pip,
                        "install",
                        "litellm[proxy]==1.80.0",
                        "fastapi==0.121.2",
                        "starlette==0.49.3",
                        "packaging==24.2",
                        "protobuf==7.36.0",
                        "grpcio-tools==1.83.1",
                        "typer==0.20.1",
                        "uvicorn-worker==0.2.0",
                        "tensordict==0.8.3",
                    )
                ),
                InstallCommand((*pip, "uninstall", "-y", "pyvers")),
            ]
        )
        commands.extend(
            [
                InstallCommand((*pip, "install", "-e", str(repo_root))),
                InstallCommand((*pip, "install", "pytest")),
            ]
        )
    else:
        commands.append(InstallCommand((*pip, "install", "--no-deps", "-e", str(repo_root))))
    commands.append(InstallCommand((*pip, "check")))
    return commands


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动发现昇腾 NPU 或 NVIDIA GPU，并安装兼容的 PrefixGrouper 栈")
    parser.add_argument("--backend", choices=("auto", "npu", "gpu"), default="auto")
    parser.add_argument(
        "--cpu-dev",
        action="store_true",
        help="安装无 NPU 设备的 CPU 开发依赖；必须显式指定 --backend npu",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅输出将执行的命令")
    args = parser.parse_args(argv)
    if args.cpu_dev and args.backend != "npu":
        parser.error("--cpu-dev 必须与 --backend npu 一起使用")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    backend: Backend = detect_backend() if args.backend == "auto" else args.backend
    print(f"检测后端：{'昇腾 NPU' if backend == 'npu' else 'NVIDIA GPU'}")
    if backend == "npu":
        print(f"要求 CANN {NPU_CANN_VERSION}；Python 必须为 >=3.10,<3.13。")
        if not (sys.version_info >= (3, 10) and sys.version_info < (3, 13)):
            raise RuntimeError(f"NPU 栈不支持 Python {sys.version.split()[0]}，请使用 Python 3.10-3.12。")
        if args.cpu_dev:
            print("无设备 CPU 开发模式：安装完整 NPU Python 包，但不探测或初始化 NPU。")

    for command in build_install_plan(backend, repo_root, cpu_dev=args.cpu_dev):
        print(f"+ {command.display()}", flush=True)
        if args.dry_run:
            continue
        if command.action == "normalize-triton-metadata":
            _normalize_triton_ascend_metadata()
            continue
        environment = os.environ.copy()
        if args.cpu_dev:
            venv_bin = str(Path(sys.executable).parent)
            environment["PATH"] = os.pathsep.join(
                part for part in (venv_bin, environment.get("PATH", "")) if part
            )
            environment.pop("PYTHONPATH", None)
        environment.update(command.env)
        subprocess.run(command.argv, cwd=repo_root, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
