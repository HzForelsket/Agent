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
import re
import shutil
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
import torch.nn.functional as F
from packaging.version import Version
from prefix_grouper import PrefixGrouper
from prefix_grouper_stack import NPU_CANN_VERSION, REQUIRED_STACKS
from transformers import AutoConfig, AutoModelForCausalLM
from verl.trainer.ppo.prefix_grouper_utils import build_position_ids_for_prefix_grouper

from agentlightning.verl.accelerator import AcceleratorRuntime, select_accelerator
from agentlightning.verl.model_download import materialize_model_for_npu
from agentlightning.verl.prefix_grouper import apply_prefix_grouper_patch

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
    for case in args.cases:
        if case.group_size > args.batch_size or args.batch_size % case.group_size:
            parser.error(f"组大小 {case.group_size} 必须整除 batch-size={args.batch_size}")
    return args


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
    if args.weights == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            local_files_only=local_files_only,
            **kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_config(**kwargs)
    model.to(device=device, dtype=dtype).eval()
    return model, config


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
    grouper = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=prefix_mask,
        suffix_mask=response_mask,
        group_sizes=[group_size] * group_count,
        padding_mode="right",
        device=device,
    )
    grouped_ids = grouper.concat_input(representatives, prefix_mask, responses, response_mask)
    return {
        "input_ids": torch.cat((prompts, responses), dim=-1),
        "attention_mask": torch.ones((batch_size, prompt_length + response_length), dtype=torch.bool, device=device),
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
            attention_mask=batch["grouper"].padding_mask,
            position_ids=batch["grouped_position_ids"],
            prefix_grouper=batch["grouper"],
            use_cache=False,
        )
        _, _, suffix_logits, _ = batch["grouper"].split_output(output.logits, include_prefix_last=1)
        return suffix_logits[:, :-1]
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
    )
    return output.logits[:, prompt_length - 1 : prompt_length + response_length - 1]


def timed(
    function: Callable[[], torch.Tensor],
    warmup: int,
    repeats: int,
    accelerator: AcceleratorRuntime,
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        function()
    accelerator.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = accelerator.event(), accelerator.event()
        start.record()
        function()
        end.record()
        accelerator.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), samples


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
    sampler.start()
    started = time.monotonic()
    iterations = 0
    try:
        while time.monotonic() - started < duration:
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


def peak_memory(function: Callable[[], torch.Tensor], accelerator: AcceleratorRuntime) -> tuple[float, float]:
    gc.collect()
    accelerator.empty_cache()
    baseline = accelerator.memory_allocated()
    accelerator.reset_peak_memory_stats()
    function()
    accelerator.synchronize()
    peak = accelerator.max_memory_allocated()
    return peak / 2**20, (peak - baseline) / 2**20


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
) -> dict[str, Any]:
    def step(grouped: bool) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        logits = response_logits(model, batch, grouped)
        if mode == "forward-backward":
            loss = F.cross_entropy(logits.flatten(0, 1), batch["responses"].flatten())
            loss.backward()
            return loss
        return logits

    context = torch.inference_mode() if mode == "forward" else torch.enable_grad()
    with context:
        baseline_ms, baseline_samples = timed(lambda: step(False), warmup, repeats, accelerator)
        grouped_ms, grouped_samples = timed(lambda: step(True), warmup, repeats, accelerator)
        baseline_power = workload_power(
            lambda: step(False),
            probe=power_probe,
            duration=power_duration,
            interval=power_interval,
            accelerator=accelerator,
        )
        grouped_power = workload_power(
            lambda: step(True),
            probe=power_probe,
            duration=power_duration,
            interval=power_interval,
            accelerator=accelerator,
        )
        baseline_peak, baseline_incremental = peak_memory(lambda: step(False), accelerator)
        grouped_peak, grouped_incremental = peak_memory(lambda: step(True), accelerator)
    tokens = batch["responses"].numel()
    return {
        "mode": mode,
        "performance": {
            "speedup": baseline_ms / grouped_ms,
            "baseline": {
                "latency_ms": {**sample_statistics(baseline_samples), "samples": baseline_samples},
                "response_tokens_per_second": tokens * 1000 / baseline_ms,
            },
            "prefix_grouper": {
                "latency_ms": {**sample_statistics(grouped_samples), "samples": grouped_samples},
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
                "peak_mib": baseline_peak,
                "incremental_peak_mib": baseline_incremental,
                "peak_bytes_per_response_token": baseline_peak * 2**20 / tokens,
            },
            "prefix_grouper": {
                "peak_mib": grouped_peak,
                "incremental_peak_mib": grouped_incremental,
                "peak_bytes_per_response_token": grouped_peak * 2**20 / tokens,
            },
            "saved_mib": baseline_peak - grouped_peak,
            "saved_ratio": 1 - grouped_peak / baseline_peak,
        },
    }


def correctness(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    with torch.inference_mode():
        baseline_logits = response_logits(model, batch, False)
        grouped_logits = response_logits(model, batch, True)
        response_ids = batch["responses"].unsqueeze(-1)
        baseline_log_probs = baseline_logits.float().log_softmax(-1).gather(-1, response_ids).squeeze(-1)
        grouped_log_probs = grouped_logits.float().log_softmax(-1).gather(-1, response_ids).squeeze(-1)

    logit_difference = (baseline_logits.float() - grouped_logits.float()).abs()
    log_prob_difference = (baseline_log_probs - grouped_log_probs).abs()
    low_precision = baseline_logits.dtype in {torch.bfloat16, torch.float16}
    max_tolerance = 0.2 if low_precision else 0.03
    mean_tolerance = 0.05 if low_precision else 0.01
    return {
        "passed": bool(log_prob_difference.max() <= max_tolerance and log_prob_difference.mean() <= mean_tolerance),
        "max_response_log_prob_error": log_prob_difference.max().item(),
        "mean_response_log_prob_error": log_prob_difference.mean().item(),
        "max_logit_error": logit_difference.max().item(),
        "mean_logit_error": logit_difference.mean().item(),
        "top1_agreement": (baseline_logits.argmax(dim=-1) == grouped_logits.argmax(dim=-1)).float().mean().item(),
        "max_tolerance": max_tolerance,
        "mean_tolerance": mean_tolerance,
    }


def markdown(results: dict[str, Any]) -> str:
    lines = [
        "# PrefixGrouper PPA 对比",
        "",
        f"- 加速器：{results['accelerator']['backend']} / {results['accelerator']['name']}",
        f"- 软件栈：torch {results['stack']['torch']} / vLLM {results['stack']['vllm']} / VERL {results['stack']['verl']}",
        f"- 精度：{results['dtype']}，权重：{results['weights']}",
        f"- NPU 模型下载目录：{results['model_download_root'] or '不适用（GPU）'}",
        "- PPA 口径：Performance（延迟/吞吐）、Power（整卡功耗/估算能耗）、Accuracy（数值一致性）；另列峰值显存资源占用。芯片 Area 不属于运行时可测指标。",
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
    device = accelerator.device
    dtype = getattr(torch, args.dtype)
    accelerator.set_device()
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
        },
        "required_cann": NPU_CANN_VERSION if accelerator.backend == "npu" else None,
        "dtype": args.dtype,
        "weights": args.weights,
        "model_download_root": str(model_download_root) if model_download_root is not None else None,
        "batch_size": args.batch_size,
        "response_length": args.response_length,
        "ppa_definition": {
            "performance": "设备事件计时的延迟分布与 response token 吞吐",
            "power": "npu-smi/nvidia-smi 整卡功耗；J/step 由平均功耗乘以 p50 延迟估算",
            "accuracy": "添加前后 response log-prob、logit 和 top-1 一致性",
            "area": "芯片物理面积不是运行时指标，不输出；以峰值显存作为资源占用补充",
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
        print(f"[{index}/{len(args.models)}] {model_ref}", flush=True)
        started = time.monotonic()
        materialization = None
        resolved_model_ref = model_ref
        if model_download_root is not None:
            materialization = materialize_model_for_npu(
                model_ref,
                model_download_root,
                local_files_only=args.local_files_only,
                download_weights=args.weights == "pretrained",
            )
            resolved_model_ref = materialization.local_path
            print(f"NPU 模型已就绪：{model_ref} -> {resolved_model_ref}", flush=True)
        model, config = load_model(
            resolved_model_ref,
            args,
            device,
            dtype,
            force_local=materialization is not None,
        )
        idle_measurement = idle_power(power_probe, args.power_idle_samples, args.power_interval)
        idle_watts = float(idle_measurement["watts"]["mean"]) if idle_measurement.get("available") else None
        model_result = {
            "model": model_ref,
            "resolved_model_path": resolved_model_ref,
            "model_materialization": asdict(materialization) if materialization is not None else None,
            "label": Path(model_ref).name,
            "model_type": config.model_type,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
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
                seed=args.seed,
            )
            check = correctness(model, batch)
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
        write_report(results, args.output_json, args.output_markdown)
        del model, config
        gc.collect()
        accelerator.empty_cache()

    print("\n" + markdown(results))
    print(f"JSON: {args.output_json.resolve()}")
    print(f"Markdown: {args.output_markdown.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
