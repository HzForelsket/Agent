# Copyright (c) Microsoft. All rights reserved.

"""Install the pinned PrefixGrouper stack after auto-detecting NPU or GPU."""

from __future__ import annotations

import argparse
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
        "huggingface_hub": "1.5.0",
        "httpx": "0.28.1",
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

    def display(self) -> str:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in self.env.items())
        command = shlex.join(self.argv)
        return f"{prefix} {command}" if prefix else command


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


def build_install_plan(backend: Backend, repo_root: Path, python: str = sys.executable) -> list[InstallCommand]:
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
                        "protobuf==5.29.5",
                        "grpcio-tools",
                    )
                ),
                InstallCommand((*pip, "install", "--no-build-isolation", "arctic-inference==0.1.1")),
                InstallCommand((*pip, "install", "-r", str(requirements))),
                InstallCommand(
                    (
                        *pip,
                        "install",
                        "--no-build-isolation",
                        "vllm @ git+https://github.com/vllm-project/vllm.git@v0.22.1",
                    ),
                    env={"VLLM_TARGET_DEVICE": "empty"},
                ),
            ]
        )
    commands.extend(
        [
            InstallCommand((*pip, "install", "--no-deps", "-e", str(repo_root))),
            InstallCommand((*pip, "check")),
        ]
    )
    return commands


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动发现昇腾 NPU 或 NVIDIA GPU，并安装兼容的 PrefixGrouper 栈")
    parser.add_argument("--backend", choices=("auto", "npu", "gpu"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="仅输出将执行的命令")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    backend: Backend = detect_backend() if args.backend == "auto" else args.backend
    print(f"检测后端：{'昇腾 NPU' if backend == 'npu' else 'NVIDIA GPU'}")
    if backend == "npu":
        print(f"要求 CANN {NPU_CANN_VERSION}；Python 必须为 >=3.10,<3.13。")
        if not (sys.version_info >= (3, 10) and sys.version_info < (3, 13)):
            raise RuntimeError(f"NPU 栈不支持 Python {sys.version.split()[0]}，请使用 Python 3.10-3.12。")

    for command in build_install_plan(backend, repo_root):
        print(f"+ {command.display()}", flush=True)
        if args.dry_run:
            continue
        environment = os.environ.copy()
        environment.update(command.env)
        subprocess.run(command.argv, cwd=repo_root, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
