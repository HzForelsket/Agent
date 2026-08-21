# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""Benchmark standard attention against PrefixGrouper on VERL 0.9.

By default only model configurations are downloaded and full architectures are
initialized with random weights.  Runtime and memory are therefore representative
for dense kernels without downloading checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from packaging.version import Version
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from prefix_grouper_stack import NPU_CANN_VERSION, REQUIRED_STACKS
from transformers import AutoConfig, AutoModelForCausalLM
from verl.trainer.ppo.prefix_grouper_utils import build_position_ids_for_prefix_grouper
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from verl.workers.engine.fsdp.utils import apply_npu_fsdp_patches

from agentlightning.verl.accelerator import AcceleratorRuntime, select_accelerator
from agentlightning.verl.model_download import materialize_model_for_npu
from agentlightning.verl.prefix_grouper import apply_prefix_grouper_patch, build_prefix_grouper

DEFAULT_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
DIST_NAMES = {
    "torch_npu": "torch-npu",
    "triton_ascend": "triton-ascend",
    "vllm_ascend": "vllm-ascend",
}


@dataclass(frozen=True)
class Case:
    prompt_length: int
    group_size: int


@dataclass(frozen=True)
class DistributedContext:
    """FSDP model mesh and host-side control collectives for the PPA runner."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    model_mesh: DeviceMesh | None = None
    host_group: dist.ProcessGroup | None = None

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier(group=self._host_group())

    def _host_group(self) -> dist.ProcessGroup:
        if self.host_group is None:
            raise RuntimeError("多卡 PPA 缺少 Gloo 主机通信组。")
        return self.host_group

    def max_value(self, value: float) -> float:
        if not self.enabled:
            return float(value)
        tensor = torch.tensor(float(value), dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self._host_group())
        return float(tensor.item())

    def numeric_rows(self, values: Sequence[float]) -> list[list[float]]:
        row = torch.tensor([float(value) for value in values], dtype=torch.float64)
        if not self.enabled:
            return [row.cpu().tolist()]
        rows = [torch.empty_like(row) for _ in range(self.world_size)]
        dist.all_gather(rows, row, group=self._host_group())
        return [item.tolist() for item in rows]

    def broadcast_bool(self, value: bool) -> bool:
        if not self.enabled:
            return value
        tensor = torch.tensor(int(value), dtype=torch.uint8)
        dist.broadcast(tensor, src=0, group=self._host_group())
        return bool(tensor.item())

    def close(self) -> None:
        if not self.enabled:
            return
        dist.destroy_process_group(self._host_group())
        dist.destroy_process_group()


@dataclass(frozen=True)
class PowerProbe:
    """Read whole-device power without coupling the benchmark to vendor Python bindings."""

    backend: str
    device_index: int
    chip_id: int = 0

    @property
    def command(self) -> tuple[str, ...]:
        if self.backend == "npu":
            return ("npu-smi", "info", "-t", "power", "-i", str(self.device_index), "-c", str(self.chip_id))
        return (
            "nvidia-smi",
            f"--id={self.device_index}",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        )

    def sample(self) -> tuple[float | None, str | None]:
        executable = self.command[0]
        if shutil.which(executable) is None:
            return None, f"{executable} 不在 PATH 中"
        try:
            completed = subprocess.run(
                self.command,
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, str(exc)
        if completed.returncode:
            message = completed.stderr.strip() or completed.stdout.strip() or f"exit={completed.returncode}"
            return None, message
        value = parse_power_watts(self.backend, completed.stdout)
        if value is None:
            return None, f"无法解析 {executable} 功耗输出"
        return value, None


def parse_power_watts(backend: str, output: str) -> float | None:
    """Parse the stable power field emitted by npu-smi or nvidia-smi."""
    if backend == "npu":
        match = re.search(
            r"(?:NPU Real-time Power|Power Dissipation)\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            output,
        )
    else:
        match = re.search(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*$", output, flags=re.MULTILINE)
    return float(match.group(1)) if match else None


def percentile(sorted_samples: Sequence[float], fraction: float) -> float:
    if not sorted_samples:
        raise ValueError("Cannot calculate a percentile from an empty sample set.")
    position = (len(sorted_samples) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_samples[lower])
    weight = position - lower
    return float(sorted_samples[lower] * (1.0 - weight) + sorted_samples[upper] * weight)


def sample_statistics(samples: Sequence[float]) -> dict[str, float | int]:
    """Return stable distribution metrics for latency and power samples."""
    if not samples:
        raise ValueError("Cannot summarize an empty sample set.")
    ordered = sorted(float(sample) for sample in samples)
    mean = statistics.mean(ordered)
    stdev = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean,
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "stdev": stdev,
        "coefficient_of_variation": stdev / mean if mean else 0.0,
    }


def parse_case(text: str) -> Case:
    try:
        prompt_length, group_size = (int(value) for value in text.split(":", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case 格式必须是 PROMPT长度:共享组大小，例如 512:4") from exc
    if prompt_length <= 0 or group_size <= 1:
        raise argparse.ArgumentTypeError("prompt 长度必须为正数，组大小必须大于 1")
    return Case(prompt_length, group_size)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="输出 PrefixGrouper 添加前后的速度、显存和正确性对比")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--case", action="append", type=parse_case, dest="cases")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--response-length", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="auto", help="auto、gpu/cuda、npu，或带编号的设备（如 npu:1）")
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=0,
        help="本机并行进程数；0 表示使用全部可见设备，1 表示单卡",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weights", choices=("random", "pretrained"), default="random")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--power",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="采集整卡功耗；默认在 NPU 开启、GPU 关闭",
    )
    parser.add_argument("--power-duration", type=float, default=2.0, help="每条路径的功耗采集时长（秒）")
    parser.add_argument("--power-interval", type=float, default=0.1, help="功耗采样间隔（秒）")
    parser.add_argument("--power-idle-samples", type=int, default=3, help="每个模型加载后的空闲功耗采样数")
    parser.add_argument("--npu-chip-id", type=int, default=0, help="npu-smi 功耗查询使用的 chip id")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output-json", type=Path, default=Path("prefix_grouper_results.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("prefix_grouper_results.md"))
    args = parser.parse_args(argv)
    args.cases = args.cases or [Case(512, 4), Case(1024, 8)]
    if args.batch_size <= 0 or args.response_length <= 0 or args.repeats <= 0 or args.warmup < 0:
        parser.error("batch、response、repeats 必须为正数，warmup 不能为负数")
    if args.power_duration <= 0 or args.power_interval <= 0 or args.power_idle_samples <= 0:
        parser.error("power-duration、power-interval 和 power-idle-samples 必须为正数")
    if args.npu_chip_id < 0:
        parser.error("npu-chip-id 不能为负数")
    if args.nproc_per_node < 0:
        parser.error("nproc-per-node 不能为负数")
    if ":" in args.device and args.nproc_per_node not in {0, 1}:
        parser.error("指定带编号设备时 nproc-per-node 只能为 0 或 1")
    for case in args.cases:
        if case.group_size > args.batch_size or args.batch_size % case.group_size:
            parser.error(f"组大小 {case.group_size} 必须整除 batch-size={args.batch_size}")
    return args


def launch_distributed_if_needed(
    args: argparse.Namespace,
    argv: Sequence[str] | None,
    accelerator: AcceleratorRuntime,
) -> int | None:
    """Relaunch the benchmark with torchrun so a normal command uses every visible device."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or ":" in args.device:
        return None
    process_count = args.nproc_per_node or int(accelerator.module.device_count())
    if process_count <= 1:
        return None
    available = int(accelerator.module.device_count())
    if process_count > available:
        raise ValueError(f"nproc-per-node={process_count} 超过可见设备数 {available}。")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rendezvous_socket:
        rendezvous_socket.bind(("127.0.0.1", 0))
        master_port = int(rendezvous_socket.getsockname()[1])
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master-addr=127.0.0.1",
        f"--master-port={master_port}",
        f"--nproc-per-node={process_count}",
        str(Path(__file__).resolve()),
        *(list(argv) if argv is not None else sys.argv[1:]),
    ]
    return subprocess.run(command, check=False).returncode


def initialize_distributed(accelerator: AcceleratorRuntime) -> tuple[AcceleratorRuntime, DistributedContext]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        accelerator.set_device()
        return accelerator, DistributedContext()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device_spec = f"{accelerator.backend}:{local_rank}"
    accelerator = select_accelerator(device_spec)
    accelerator.set_device()
    dist.init_process_group(backend="hccl" if accelerator.backend == "npu" else "nccl")
    model_mesh = init_device_mesh(
        accelerator.device_type,
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",),
    )
    host_group = dist.new_group(backend="gloo")
    return accelerator, DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        model_mesh=model_mesh,
        host_group=host_group,
    )


def _distribution_version(name: str) -> str:
    try:
        return version(DIST_NAMES.get(name, name))
    except PackageNotFoundError:
        return "未安装"


def installed_stack(backend: str) -> dict[str, str]:
    stack = {
        "torch": torch.__version__.split("+")[0],
        "vllm": _distribution_version("vllm"),
        "verl": _distribution_version("verl"),
        "transformers": _distribution_version("transformers"),
        "prefix_grouper": _distribution_version("prefix_grouper"),
    }
    if backend == "npu":
        stack.update(
            {
                "torch_npu": _distribution_version("torch_npu"),
                "triton_ascend": _distribution_version("triton_ascend"),
                "vllm_ascend": _distribution_version("vllm_ascend"),
            }
        )
    return stack


def check_stack(stack: dict[str, str], backend: str) -> None:
    required = REQUIRED_STACKS[backend]
    mismatches = [
        f"{name}={stack[name]}（需要 {wanted}）"
        for name, wanted in required.items()
        if stack.get(name) == "未安装" or Version(stack[name]).public != Version(wanted).public
    ]
    if mismatches:
        raise RuntimeError("测试环境版本不符合要求：" + "，".join(mismatches))


def load_model(
    model_ref: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    *,
    force_local: bool = False,
    distributed: DistributedContext = DistributedContext(),
):
    local_files_only = args.local_files_only or force_local
    config = AutoConfig.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    config.use_cache = False
    kwargs = {
        "config": config,
        "attn_implementation": "sdpa",
        "dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if distributed.enabled:
        if distributed.model_mesh is None:
            raise RuntimeError("多卡 benchmark 缺少 FSDP 设备网格。")
        if device.type == "npu":
            apply_npu_fsdp_patches()
        init_context_factory = get_init_weight_context_manager(
            use_meta_tensor=not bool(getattr(config, "tie_word_embeddings", False)),
            mesh=distributed.model_mesh,
        )
    else:
        init_context_factory = nullcontext

    with init_context_factory():
        if args.weights == "pretrained":
            model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_config(**kwargs)
        model.to(dtype=dtype)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if distributed.enabled:
        model = FullyShardedDataParallel(
            model,
            param_init_fn=init_fn,
            auto_wrap_policy=get_fsdp_wrap_policy(model),
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
            sync_module_states=True,
            device_mesh=distributed.model_mesh,
            use_orig_params=False,
            limit_all_gathers=True,
        )
    else:
        model.to(device=device)
    return model.eval(), config, parameter_count


def make_batch(
    config: Any,
    *,
    batch_size: int,
    prompt_length: int,
    response_length: int,
    group_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed + prompt_length + group_size)
    group_count = batch_size // group_size
    representatives = torch.randint(
        1, int(config.vocab_size), (group_count, prompt_length), generator=generator, device=device
    )
    prompts = representatives.repeat_interleave(group_size, dim=0)
    responses = torch.randint(
        1, int(config.vocab_size), (batch_size, response_length), generator=generator, device=device
    )
    prefix_mask = torch.ones_like(representatives, dtype=torch.bool)
    response_mask = torch.ones_like(responses, dtype=torch.bool)
    grouper = build_prefix_grouper(
        prefix_mask=prefix_mask,
        suffix_mask=response_mask,
        group_sizes=[group_size] * group_count,
        device=device,
    )
    grouped_ids = grouper.concat_input(representatives, prefix_mask, responses, response_mask)
    return {
        "input_ids": torch.cat((prompts, responses), dim=-1),
        "position_ids": torch.arange(prompt_length + response_length, device=device).expand(batch_size, -1),
        "responses": responses,
        "grouper": grouper,
        "grouped_ids": grouped_ids,
        "grouped_position_ids": build_position_ids_for_prefix_grouper(grouper),
    }


def workload_metrics(batch: dict[str, Any]) -> dict[str, int | float]:
    """Describe the dense-token and causal-attention work represented by a batch."""
    response_length = int(batch["responses"].shape[1])
    prompt_length = int(batch["input_ids"].shape[1] - response_length)
    batch_size = int(batch["input_ids"].shape[0])
    grouper = batch["grouper"]

    baseline_dense_tokens = int(batch["input_ids"].numel())
    grouped_dense_tokens = int(batch["grouped_ids"].numel())
    baseline_attention_pairs = (
        batch_size * (prompt_length + response_length) * (prompt_length + response_length + 1) // 2
    )
    grouped_attention_pairs = 0
    for info in grouper.group_info:
        grouped_attention_pairs += int(info.prefix_len * (info.prefix_len + 1) // 2)
        for suffix_length in info.suffix_lens:
            grouped_attention_pairs += int(suffix_length * info.prefix_len + suffix_length * (suffix_length + 1) // 2)
    return {
        "response_tokens": int(batch["responses"].numel()),
        "baseline_dense_model_tokens": baseline_dense_tokens,
        "prefix_grouper_dense_model_tokens": grouped_dense_tokens,
        "dense_model_token_saved_ratio": 1.0 - grouped_dense_tokens / baseline_dense_tokens,
        "baseline_causal_attention_pairs": baseline_attention_pairs,
        "prefix_grouper_causal_attention_pairs": grouped_attention_pairs,
        "causal_attention_pair_saved_ratio": 1.0 - grouped_attention_pairs / baseline_attention_pairs,
    }


def response_logits(model: torch.nn.Module, batch: dict[str, Any], grouped: bool) -> torch.Tensor:
    response_length = batch["responses"].shape[1]
    prompt_length = batch["input_ids"].shape[1] - response_length
    if grouped:
        output = model(
            input_ids=batch["grouped_ids"],
            attention_mask=None,
            position_ids=batch["grouped_position_ids"],
            prefix_grouper=batch["grouper"],
            use_cache=False,
        )
        _, _, suffix_logits, _ = batch["grouper"].split_output(output.logits, include_prefix_last=1)
        return suffix_logits[:, :-1]
    output = model(
        input_ids=batch["input_ids"],
        # Synthetic benchmark rows are fully valid, so no padding mask is
        # required. ``None`` lets SDPA use its causal flag without reading a
        # zero-dimensional mask reduction back from the accelerator.
        attention_mask=None,
        position_ids=batch["position_ids"],
        use_cache=False,
    )
    return output.logits[:, prompt_length - 1 : prompt_length + response_length - 1]


def timed(
    function: Callable[[], torch.Tensor],
    warmup: int,
    repeats: int,
    accelerator: AcceleratorRuntime,
    distributed: DistributedContext = DistributedContext(),
) -> tuple[float, list[float], list[list[float]]]:
    for _ in range(warmup):
        function()
    accelerator.synchronize()
    distributed.barrier()
    local_samples = []
    samples = []
    for _ in range(repeats):
        distributed.barrier()
        start, end = accelerator.event(), accelerator.event()
        start.record()
        function()
        end.record()
        accelerator.synchronize()
        local_elapsed = float(start.elapsed_time(end))
        local_samples.append(local_elapsed)
        samples.append(distributed.max_value(local_elapsed))
    per_rank_samples = distributed.numeric_rows(local_samples)
    return statistics.median(samples), samples, per_rank_samples


def idle_power(probe: PowerProbe | None, samples: int, interval: float) -> dict[str, Any]:
    if probe is None:
        return {"enabled": False, "available": False, "reason": "功耗采集已关闭"}
    values: list[float] = []
    errors: list[str] = []
    for index in range(samples):
        value, error = probe.sample()
        if value is not None:
            values.append(value)
        elif error is not None:
            errors.append(error)
        if index + 1 < samples:
            time.sleep(interval)
    if not values:
        return {
            "enabled": True,
            "available": False,
            "reason": errors[0] if errors else "没有获得功耗样本",
            "source": " ".join(probe.command),
        }
    return {
        "enabled": True,
        "available": True,
        "source": " ".join(probe.command),
        "samples_watts": values,
        "watts": sample_statistics(values),
        "errors": sorted(set(errors)),
    }


def workload_power(
    function: Callable[[], torch.Tensor],
    *,
    probe: PowerProbe | None,
    duration: float,
    interval: float,
    accelerator: AcceleratorRuntime,
    distributed: DistributedContext = DistributedContext(),
) -> dict[str, Any]:
    """Collect power in a dedicated sustained run so latency samples remain uncontaminated."""
    if probe is None:
        return {"enabled": False, "available": False, "reason": "功耗采集已关闭"}

    values: list[float] = []
    errors: list[str] = []
    stop = threading.Event()

    def sample_loop() -> None:
        while not stop.is_set():
            value, error = probe.sample()
            if value is not None:
                values.append(value)
            elif error is not None:
                errors.append(error)
            stop.wait(interval)

    sampler = threading.Thread(target=sample_loop, name="prefix-grouper-power", daemon=True)
    distributed.barrier()
    sampler.start()
    started = time.monotonic()
    iterations = 0
    try:
        while True:
            if distributed.enabled:
                should_continue = distributed.broadcast_bool(
                    distributed.is_main and time.monotonic() - started < duration
                )
                if not should_continue:
                    break
            elif time.monotonic() - started >= duration:
                break
            function()
            accelerator.synchronize()
            iterations += 1
    finally:
        accelerator.synchronize()
        elapsed = time.monotonic() - started
        stop.set()
        sampler.join(timeout=max(3.0, interval * 2))

    if not values:
        return {
            "enabled": True,
            "available": False,
            "reason": errors[0] if errors else "没有获得功耗样本",
            "source": " ".join(probe.command),
            "measurement_duration_seconds": elapsed,
            "workload_iterations": iterations,
        }
    return {
        "enabled": True,
        "available": True,
        "source": " ".join(probe.command),
        "measurement_duration_seconds": elapsed,
        "workload_iterations": iterations,
        "samples_watts": values,
        "watts": sample_statistics(values),
        "errors": sorted(set(errors)),
    }


def aggregate_device_power(
    measurement: dict[str, Any],
    distributed: DistributedContext,
) -> dict[str, Any]:
    """Aggregate per-rank whole-device power into job-level power."""
    if not distributed.enabled:
        return measurement
    available = bool(measurement.get("available"))
    watts = measurement.get("watts", {})
    fields = ("min", "max", "mean", "p50", "p90", "p95", "p99", "stdev")
    values = [float(available), float(watts.get("count", 0))]
    values.extend(float(watts.get(field, 0.0)) for field in fields)
    values.extend(
        [
            float(measurement.get("measurement_duration_seconds", 0.0)),
            float(measurement.get("workload_iterations", 0)),
        ]
    )
    rows = distributed.numeric_rows(values)
    per_rank = []
    for rank, row in enumerate(rows):
        rank_available = bool(row[0])
        rank_watts = {field: row[index + 2] for index, field in enumerate(fields)}
        rank_watts["count"] = int(row[1])
        per_rank.append(
            {
                "rank": rank,
                "device_index": rank,
                "available": rank_available,
                "watts": rank_watts if rank_available else None,
                "measurement_duration_seconds": row[-2],
                "workload_iterations": int(row[-1]),
            }
        )
    if not all(item["available"] for item in per_rank):
        return {
            "enabled": bool(measurement.get("enabled", True)),
            "available": False,
            "reason": "至少一个 rank 未获得功耗样本",
            "per_rank": per_rank,
        }

    aggregate_watts = {field: sum(float(item["watts"][field]) for item in per_rank) for field in fields}
    aggregate_watts["count"] = min(int(item["watts"]["count"]) for item in per_rank)
    aggregate_watts["coefficient_of_variation"] = (
        aggregate_watts["stdev"] / aggregate_watts["mean"] if aggregate_watts["mean"] else 0.0
    )
    result = {
        "enabled": True,
        "available": True,
        "source": f"所有 {distributed.world_size} 张设备的整卡功耗之和",
        "watts": aggregate_watts,
        "per_rank": per_rank,
    }
    if any(item["measurement_duration_seconds"] for item in per_rank):
        result["measurement_duration_seconds"] = max(
            float(item["measurement_duration_seconds"]) for item in per_rank
        )
        result["workload_iterations"] = min(int(item["workload_iterations"]) for item in per_rank)
    return result


def add_energy_efficiency(
    power: dict[str, Any],
    *,
    latency_ms: float,
    response_tokens: int,
    idle_watts: float | None,
) -> dict[str, Any]:
    if not power.get("available"):
        return power
    average_watts = float(power["watts"]["mean"])
    energy_joules = average_watts * latency_ms / 1000.0
    dynamic_watts = max(0.0, average_watts - idle_watts) if idle_watts is not None else None
    dynamic_energy_joules = dynamic_watts * latency_ms / 1000.0 if dynamic_watts is not None else None
    return {
        **power,
        "estimated_energy_joules_per_step": energy_joules,
        "response_tokens_per_joule": response_tokens / energy_joules if energy_joules else None,
        "idle_watts": idle_watts,
        "dynamic_watts": dynamic_watts,
        "estimated_dynamic_energy_joules_per_step": dynamic_energy_joules,
        "response_tokens_per_dynamic_joule": (
            response_tokens / dynamic_energy_joules if dynamic_energy_joules else None
        ),
    }


def peak_memory(
    function: Callable[[], torch.Tensor],
    accelerator: AcceleratorRuntime,
    distributed: DistributedContext = DistributedContext(),
) -> dict[str, Any]:
    gc.collect()
    accelerator.empty_cache()
    distributed.barrier()
    baseline = accelerator.memory_allocated()
    accelerator.reset_peak_memory_stats()
    function()
    accelerator.synchronize()
    peak = accelerator.max_memory_allocated()
    rows = distributed.numeric_rows([peak / 2**20, (peak - baseline) / 2**20])
    peaks = [row[0] for row in rows]
    incremental = [row[1] for row in rows]
    return {
        "peak_mib": max(peaks),
        "mean_peak_mib": statistics.mean(peaks),
        "sum_peak_mib": sum(peaks),
        "incremental_peak_mib": max(incremental),
        "mean_incremental_peak_mib": statistics.mean(incremental),
        "per_rank": [
            {"rank": rank, "peak_mib": row[0], "incremental_peak_mib": row[1]}
            for rank, row in enumerate(rows)
        ],
    }


def benchmark_mode(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    mode: str,
    warmup: int,
    repeats: int,
    accelerator: AcceleratorRuntime,
    power_probe: PowerProbe | None,
    power_duration: float,
    power_interval: float,
    idle_watts: float | None,
    distributed: DistributedContext = DistributedContext(),
) -> dict[str, Any]:
    def step(grouped: bool) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        logits = response_logits(model, batch, grouped)
        if mode == "forward-backward":
            loss = F.cross_entropy(logits.flatten(0, 1), batch["responses"].flatten())
            loss.backward()
            return loss
        return logits

    context = torch.no_grad() if mode == "forward" else torch.enable_grad()
    with context:
        baseline_ms, baseline_samples, baseline_rank_samples = timed(
            lambda: step(False), warmup, repeats, accelerator, distributed
        )
        grouped_ms, grouped_samples, grouped_rank_samples = timed(
            lambda: step(True), warmup, repeats, accelerator, distributed
        )
        baseline_power = aggregate_device_power(
            workload_power(
                lambda: step(False),
                probe=power_probe,
                duration=power_duration,
                interval=power_interval,
                accelerator=accelerator,
                distributed=distributed,
            ),
            distributed,
        )
        grouped_power = aggregate_device_power(
            workload_power(
                lambda: step(True),
                probe=power_probe,
                duration=power_duration,
                interval=power_interval,
                accelerator=accelerator,
                distributed=distributed,
            ),
            distributed,
        )
        baseline_memory = peak_memory(lambda: step(False), accelerator, distributed)
        grouped_memory = peak_memory(lambda: step(True), accelerator, distributed)
    tokens = batch["responses"].numel() * distributed.world_size
    return {
        "mode": mode,
        "performance": {
            "speedup": baseline_ms / grouped_ms,
            "baseline": {
                "latency_ms": {
                    **sample_statistics(baseline_samples),
                    "samples": baseline_samples,
                    "per_rank_samples": baseline_rank_samples,
                },
                "response_tokens_per_second": tokens * 1000 / baseline_ms,
            },
            "prefix_grouper": {
                "latency_ms": {
                    **sample_statistics(grouped_samples),
                    "samples": grouped_samples,
                    "per_rank_samples": grouped_rank_samples,
                },
                "response_tokens_per_second": tokens * 1000 / grouped_ms,
            },
        },
        "power": {
            "baseline": add_energy_efficiency(
                baseline_power,
                latency_ms=baseline_ms,
                response_tokens=tokens,
                idle_watts=idle_watts,
            ),
            "prefix_grouper": add_energy_efficiency(
                grouped_power,
                latency_ms=grouped_ms,
                response_tokens=tokens,
                idle_watts=idle_watts,
            ),
        },
        "memory": {
            "baseline": {
                **baseline_memory,
                "peak_bytes_per_response_token": baseline_memory["sum_peak_mib"] * 2**20 / tokens,
            },
            "prefix_grouper": {
                **grouped_memory,
                "peak_bytes_per_response_token": grouped_memory["sum_peak_mib"] * 2**20 / tokens,
            },
            "saved_mib": baseline_memory["peak_mib"] - grouped_memory["peak_mib"],
            "saved_ratio": 1 - grouped_memory["peak_mib"] / baseline_memory["peak_mib"],
        },
    }


def correctness(
    model: torch.nn.Module,
    batch: dict[str, Any],
    accelerator: AcceleratorRuntime,
    distributed: DistributedContext = DistributedContext(),
) -> dict[str, Any]:
    with torch.no_grad():
        baseline_logits = response_logits(model, batch, False)
        accelerator.synchronize()
        grouped_logits = response_logits(model, batch, True)
        accelerator.synchronize()
        response_ids = batch["responses"].unsqueeze(-1)
        baseline_log_probs = baseline_logits.float().log_softmax(-1).gather(-1, response_ids).squeeze(-1)
        grouped_log_probs = grouped_logits.float().log_softmax(-1).gather(-1, response_ids).squeeze(-1)

    logit_difference = (baseline_logits.float() - grouped_logits.float()).abs()
    log_prob_difference = (baseline_log_probs - grouped_log_probs).abs()
    low_precision = baseline_logits.dtype in {torch.bfloat16, torch.float16}
    max_tolerance = 0.2 if low_precision else 0.03
    mean_tolerance = 0.05 if low_precision else 0.01
    local_values = [
        log_prob_difference.max().item(),
        log_prob_difference.mean().item(),
        logit_difference.max().item(),
        logit_difference.mean().item(),
        (baseline_logits.argmax(dim=-1) == grouped_logits.argmax(dim=-1)).float().mean().item(),
    ]
    rows = distributed.numeric_rows(local_values)
    return {
        "passed": bool(
            max(row[0] for row in rows) <= max_tolerance
            and max(row[1] for row in rows) <= mean_tolerance
        ),
        "max_response_log_prob_error": max(row[0] for row in rows),
        "mean_response_log_prob_error": statistics.mean(row[1] for row in rows),
        "max_logit_error": max(row[2] for row in rows),
        "mean_logit_error": statistics.mean(row[3] for row in rows),
        "top1_agreement": min(row[4] for row in rows),
        "per_rank": [
            {
                "rank": rank,
                "max_response_log_prob_error": row[0],
                "mean_response_log_prob_error": row[1],
                "max_logit_error": row[2],
                "mean_logit_error": row[3],
                "top1_agreement": row[4],
            }
            for rank, row in enumerate(rows)
        ],
        "max_tolerance": max_tolerance,
        "mean_tolerance": mean_tolerance,
    }


def markdown(results: dict[str, Any]) -> str:
    lines = [
        "# PrefixGrouper PPA 对比",
        "",
        f"- 加速器：{results['accelerator']['backend']} / {results['accelerator']['name']}",
        f"- 并行：{results['accelerator']['parallelism']}，world size={results['accelerator']['world_size']}，"
        f"每 rank/global batch={results['batch_size_per_rank']}/{results['global_batch_size']}，"
        f"统计通信={results['accelerator']['statistics_backend']}",
        f"- 软件栈：torch {results['stack']['torch']} / vLLM {results['stack']['vllm']} / VERL {results['stack']['verl']}",
        f"- 精度：{results['dtype']}，权重：{results['weights']}",
        f"- NPU 模型下载目录：{results['model_download_root'] or '不适用（GPU）'}",
        "- PPA 口径：步延迟取最慢 rank、吞吐汇总所有 rank、功耗汇总所有设备、峰值显存取 rank 最大值并在 JSON 保留逐 rank 明细。",
        "",
        "## 工作量",
        "",
        "| 模型 | Prompt | 共享组 | Dense token 前/后 | Dense token 减少 | Causal pair 前/后 | Attention pair 减少 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in results["models"]:
        for case in model["cases"]:
            work = case["workload"]
            lines.append(
                f"| {model['label']} | {case['prompt_length']} | {case['group_size']} | "
                f"{work['baseline_dense_model_tokens']}/{work['prefix_grouper_dense_model_tokens']} | "
                f"{work['dense_model_token_saved_ratio']:.1%} | "
                f"{work['baseline_causal_attention_pairs']}/{work['prefix_grouper_causal_attention_pairs']} | "
                f"{work['causal_attention_pair_saved_ratio']:.1%} |"
            )

    lines.extend(
        [
            "",
            "## Performance",
            "",
            "| 模型 | 模式 | Prompt | 共享组 | 前 p50/p95/p99 ms | 后 p50/p95/p99 ms | 加速比 | 前 tok/s | 后 tok/s |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in results["models"]:
        for case in model["cases"]:
            for item in case["measurements"]:
                performance = item["performance"]
                baseline = performance["baseline"]
                grouped = performance["prefix_grouper"]
                baseline_latency = baseline["latency_ms"]
                grouped_latency = grouped["latency_ms"]
                lines.append(
                    f"| {model['label']} | {item['mode']} | {case['prompt_length']} | {case['group_size']} | "
                    f"{baseline_latency['p50']:.3f}/{baseline_latency['p95']:.3f}/{baseline_latency['p99']:.3f} | "
                    f"{grouped_latency['p50']:.3f}/{grouped_latency['p95']:.3f}/{grouped_latency['p99']:.3f} | "
                    f"{performance['speedup']:.2f}x | {baseline['response_tokens_per_second']:.0f} | "
                    f"{grouped['response_tokens_per_second']:.0f} |"
                )

    lines.extend(
        [
            "",
            "## 峰值显存",
            "",
            "| 模型 | 模式 | Prompt | 共享组 | 前峰值 MiB | 后峰值 MiB | 节省 MiB | 节省比例 | 前/后增量峰值 MiB |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in results["models"]:
        for case in model["cases"]:
            for item in case["measurements"]:
                memory = item["memory"]
                baseline = memory["baseline"]
                grouped = memory["prefix_grouper"]
                lines.append(
                    f"| {model['label']} | {item['mode']} | {case['prompt_length']} | {case['group_size']} | "
                    f"{baseline['peak_mib']:.1f} | {grouped['peak_mib']:.1f} | {memory['saved_mib']:.1f} | "
                    f"{memory['saved_ratio']:.1%} | {baseline['incremental_peak_mib']:.1f}/"
                    f"{grouped['incremental_peak_mib']:.1f} |"
                )

    lines.extend(
        [
            "",
            "## Power",
            "",
            "| 模型 | 模式 | Prompt | 共享组 | 空闲 W | 前 mean/p95/max W | 后 mean/p95/max W | 前/后 J/step | 前/后 tok/J | 能效比 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in results["models"]:
        idle = model["idle_power"]
        idle_text = f"{idle['watts']['mean']:.1f}" if idle.get("available") else "不可用"
        for case in model["cases"]:
            for item in case["measurements"]:
                baseline = item["power"]["baseline"]
                grouped = item["power"]["prefix_grouper"]
                if baseline.get("available") and grouped.get("available"):
                    baseline_watts = baseline["watts"]
                    grouped_watts = grouped["watts"]
                    baseline_power_text = (
                        f"{baseline_watts['mean']:.1f}/{baseline_watts['p95']:.1f}/{baseline_watts['max']:.1f}"
                    )
                    grouped_power_text = (
                        f"{grouped_watts['mean']:.1f}/{grouped_watts['p95']:.1f}/{grouped_watts['max']:.1f}"
                    )
                    energy_text = (
                        f"{baseline['estimated_energy_joules_per_step']:.3f}/"
                        f"{grouped['estimated_energy_joules_per_step']:.3f}"
                    )
                    efficiency_text = (
                        f"{baseline['response_tokens_per_joule']:.1f}/{grouped['response_tokens_per_joule']:.1f}"
                    )
                    efficiency_ratio = grouped["response_tokens_per_joule"] / baseline["response_tokens_per_joule"]
                    ratio_text = f"{efficiency_ratio:.2f}x"
                else:
                    reason = baseline.get("reason") or grouped.get("reason") or "不可用"
                    baseline_power_text = grouped_power_text = energy_text = efficiency_text = ratio_text = reason
                lines.append(
                    f"| {model['label']} | {item['mode']} | {case['prompt_length']} | {case['group_size']} | "
                    f"{idle_text} | {baseline_power_text} | {grouped_power_text} | {energy_text} | "
                    f"{efficiency_text} | {ratio_text} |"
                )

    lines.extend(
        [
            "",
            "## Accuracy",
            "",
            "| 模型 | Prompt | 共享组 | 结果 | log-prob max/mean error | logit max/mean error | top-1 一致率 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in results["models"]:
        for case in model["cases"]:
            accuracy = case["accuracy"]
            lines.append(
                f"| {model['label']} | {case['prompt_length']} | {case['group_size']} | "
                f"{'通过' if accuracy['passed'] else '失败'} | "
                f"{accuracy['max_response_log_prob_error']:.6f}/{accuracy['mean_response_log_prob_error']:.6f} | "
                f"{accuracy['max_logit_error']:.6f}/{accuracy['mean_logit_error']:.6f} | "
                f"{accuracy['top1_agreement']:.2%} |"
            )
    return "\n".join(lines) + "\n"


def write_report(results: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(results), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    accelerator = select_accelerator(args.device)
    stack = installed_stack(accelerator.backend)
    check_stack(stack, accelerator.backend)
    launched = launch_distributed_if_needed(args, argv, accelerator)
    if launched is not None:
        return launched
    accelerator, distributed = initialize_distributed(accelerator)
    device = accelerator.device
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    accelerator.manual_seed_all(args.seed)
    if accelerator.backend == "gpu":
        torch.backends.cuda.matmul.allow_tf32 = True
    apply_prefix_grouper_patch()
    power_enabled = args.power if args.power is not None else accelerator.backend == "npu"
    power_probe = (
        PowerProbe(
            backend=accelerator.backend,
            device_index=int(device.index or 0),
            chip_id=args.npu_chip_id,
        )
        if power_enabled
        else None
    )
    model_download_root = Path.cwd().resolve() if accelerator.backend == "npu" else None

    results: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stack": stack,
        "accelerator": {
            "backend": accelerator.display_backend,
            "device": str(device),
            "name": accelerator.device_name(),
            "world_size": distributed.world_size,
            "parallelism": "fully-sharded-data-parallel" if distributed.enabled else "single-device",
            "statistics_backend": "gloo-cpu" if distributed.enabled else "local-cpu",
        },
        "required_cann": NPU_CANN_VERSION if accelerator.backend == "npu" else None,
        "dtype": args.dtype,
        "weights": args.weights,
        "model_download_root": str(model_download_root) if model_download_root is not None else None,
        "batch_size_per_rank": args.batch_size,
        "global_batch_size": args.batch_size * distributed.world_size,
        "rank_seed_stride": 1_000_003 if distributed.enabled else 0,
        "response_length": args.response_length,
        "ppa_definition": {
            "performance": "步延迟取所有 rank 最大值；response token 吞吐汇总所有 rank",
            "power": "逐卡 npu-smi/nvidia-smi 整卡功耗求和；J/step 由总平均功耗乘以全局 p50 延迟估算",
            "accuracy": "添加前后 response log-prob、logit 和 top-1 一致性",
            "area": "芯片物理面积不是运行时指标，不输出；显存同时输出逐 rank、最大值、平均值与总和",
        },
        "power_sampling": {
            "enabled": power_enabled,
            "duration_seconds_per_path": args.power_duration if power_enabled else None,
            "interval_seconds": args.power_interval if power_enabled else None,
            "npu_chip_id": args.npu_chip_id if accelerator.backend == "npu" else None,
        },
        "models": [],
    }
    for index, model_ref in enumerate(args.models, 1):
        if distributed.is_main:
            print(f"[{index}/{len(args.models)}] {model_ref}", flush=True)
        started = time.monotonic()
        materialization = None
        resolved_model_ref = model_ref
        if model_download_root is not None:
            if distributed.is_main:
                materialization = materialize_model_for_npu(
                    model_ref,
                    model_download_root,
                    local_files_only=args.local_files_only,
                    download_weights=args.weights == "pretrained",
                )
            distributed.barrier()
            if not distributed.is_main:
                materialization = materialize_model_for_npu(
                    model_ref,
                    model_download_root,
                    local_files_only=True,
                    download_weights=args.weights == "pretrained",
                )
            assert materialization is not None
            resolved_model_ref = materialization.local_path
            if distributed.is_main:
                print(f"NPU 模型已就绪：{model_ref} -> {resolved_model_ref}", flush=True)
        model, config, parameter_count = load_model(
            resolved_model_ref,
            args,
            device,
            dtype,
            force_local=materialization is not None,
            distributed=distributed,
        )
        idle_measurement = aggregate_device_power(
            idle_power(power_probe, args.power_idle_samples, args.power_interval),
            distributed,
        )
        idle_watts = float(idle_measurement["watts"]["mean"]) if idle_measurement.get("available") else None
        model_result = {
            "model": model_ref,
            "resolved_model_path": resolved_model_ref,
            "model_materialization": asdict(materialization) if materialization is not None else None,
            "label": Path(model_ref).name,
            "model_type": config.model_type,
            "parameters": parameter_count,
            "idle_power": idle_measurement,
            "cases": [],
        }
        for case in args.cases:
            max_positions = getattr(config, "max_position_embeddings", None)
            if max_positions and case.prompt_length + args.response_length > max_positions:
                raise ValueError(f"{model_ref} 的上下文长度不足以运行 {case}")
            batch = make_batch(
                config,
                batch_size=args.batch_size,
                prompt_length=case.prompt_length,
                response_length=args.response_length,
                group_size=case.group_size,
                device=device,
                seed=args.seed + distributed.rank * 1_000_003,
            )
            check = correctness(model, batch, accelerator, distributed)
            if not check["passed"]:
                raise RuntimeError(f"{model_ref} {case} 正确性检查失败：{check}")
            modes = ["forward", "forward-backward"] if args.backward else ["forward"]
            measurements = [
                benchmark_mode(
                    model,
                    batch,
                    mode=mode,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    accelerator=accelerator,
                    power_probe=power_probe,
                    power_duration=args.power_duration,
                    power_interval=args.power_interval,
                    idle_watts=idle_watts,
                    distributed=distributed,
                )
                for mode in modes
            ]
            model_result["cases"].append(
                {
                    **asdict(case),
                    "workload": workload_metrics(batch),
                    "accuracy": check,
                    "measurements": measurements,
                }
            )
            del batch
            gc.collect()
            accelerator.empty_cache()
        model_result["elapsed_seconds"] = time.monotonic() - started
        results["models"].append(model_result)
        if distributed.is_main:
            write_report(results, args.output_json, args.output_markdown)
        del model, config
        gc.collect()
        accelerator.empty_cache()

    if distributed.is_main:
        print("\n" + markdown(results))
        print(f"JSON: {args.output_json.resolve()}")
        print(f"Markdown: {args.output_markdown.resolve()}")
    distributed.barrier()
    distributed.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
