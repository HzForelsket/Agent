# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""Distributed PrefixGrouper benchmark workers built on VERL's Ray/FSDP stack."""

from __future__ import annotations

import gc
import math
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import ray
import torch
import torch.nn.functional as F
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.config import CheckpointConfig
from verl.trainer.ppo.prefix_grouper_utils import build_position_ids_for_prefix_grouper
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig, TrainingWorkerConfig

from .accelerator import AcceleratorRuntime
from .prefix_grouper import PrefixGrouperTrainingWorker, build_prefix_grouper

__all__ = [
    "Case",
    "DistributedBenchmarkTaskRunner",
    "PowerProbe",
    "add_energy_efficiency",
    "parse_power_watts",
    "percentile",
    "sample_statistics",
    "workload_metrics",
]


@dataclass(frozen=True)
class Case:
    prompt_length: int
    group_size: int


@dataclass(frozen=True)
class PowerProbe:
    """Read whole-device power without vendor Python bindings."""

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
    baseline_logits_to_keep = torch.arange(
        prompt_length - 1,
        prompt_length + response_length - 1,
        device=device,
    )
    grouped_suffix_logits_to_keep = (
        prompt_length
        + torch.arange(group_size * response_length, device=device).view(group_size, response_length)[:, :-1].flatten()
    )
    grouped_logits_to_keep = torch.cat(
        (torch.tensor([prompt_length - 1], device=device), grouped_suffix_logits_to_keep)
    )
    return {
        "input_ids": torch.cat((prompts, responses), dim=-1),
        "position_ids": torch.arange(prompt_length + response_length, device=device).expand(batch_size, -1),
        "responses": responses,
        "grouper": grouper,
        "grouped_ids": grouped_ids,
        "grouped_position_ids": build_position_ids_for_prefix_grouper(grouper),
        "baseline_logits_to_keep": baseline_logits_to_keep,
        "grouped_logits_to_keep": grouped_logits_to_keep,
        "group_count": group_count,
        "group_size": group_size,
    }


def workload_metrics(batch: dict[str, Any]) -> dict[str, int | float]:
    """Describe the dense-token and causal-attention work represented by a local batch."""
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


def _response_logits(model: torch.nn.Module, batch: dict[str, Any], grouped: bool) -> torch.Tensor:
    response_length = batch["responses"].shape[1]
    if grouped:
        output = model(
            input_ids=batch["grouped_ids"],
            attention_mask=None,
            position_ids=batch["grouped_position_ids"],
            prefix_grouper=batch["grouper"],
            use_cache=False,
            logits_to_keep=batch["grouped_logits_to_keep"],
        )
        prefix_last_logits = output.logits[:, :1].unsqueeze(1).expand(-1, batch["group_size"], -1, -1)
        suffix_logits = output.logits[:, 1:].view(
            batch["group_count"],
            batch["group_size"],
            response_length - 1,
            output.logits.shape[-1],
        )
        return torch.cat((prefix_last_logits, suffix_logits), dim=2).flatten(0, 1).contiguous()
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=None,
        position_ids=batch["position_ids"],
        use_cache=False,
        logits_to_keep=batch["baseline_logits_to_keep"],
    )
    return output.logits.contiguous()


def _timed(
    function: Callable[[], torch.Tensor],
    warmup: int,
    repeats: int,
    accelerator: AcceleratorRuntime,
) -> list[float]:
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
    return samples


def _idle_power(probe: PowerProbe | None, samples: int, interval: float) -> dict[str, Any]:
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


def _workload_power(
    function: Callable[[], torch.Tensor],
    *,
    probe: PowerProbe | None,
    repeats: int,
    interval: float,
    accelerator: AcceleratorRuntime,
) -> dict[str, Any]:
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
    try:
        for _ in range(repeats):
            function()
            accelerator.synchronize()
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
            "workload_iterations": repeats,
        }
    return {
        "enabled": True,
        "available": True,
        "source": " ".join(probe.command),
        "measurement_duration_seconds": elapsed,
        "workload_iterations": repeats,
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


def _peak_memory(function: Callable[[], torch.Tensor], accelerator: AcceleratorRuntime) -> tuple[float, float]:
    gc.collect()
    accelerator.empty_cache()
    baseline = accelerator.memory_allocated()
    accelerator.reset_peak_memory_stats()
    function()
    accelerator.synchronize()
    peak = accelerator.max_memory_allocated()
    return peak / 2**20, (peak - baseline) / 2**20


def _profile_comparison(
    step: Callable[[bool], torch.Tensor],
    *,
    case: Case,
    mode: str,
    settings: dict[str, Any],
    accelerator: AcceleratorRuntime,
) -> dict[str, Any]:
    if not settings["profile_enabled"]:
        return {"enabled": False}

    rank = torch.distributed.get_rank()
    output_dir = (
        Path(settings["profile_output_dir"])
        / f"prompt_{case.prompt_length}_group_{case.group_size}"
        / mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_name = f"{os.uname().nodename}_rank_{rank}"
    output_dirs: dict[str, str] = {}
    labels: dict[str, str] = {}
    for path_name, grouped in (("baseline", False), ("prefix_grouper", True)):
        capture_dir = output_dir / path_name
        capture_dir.mkdir(parents=True, exist_ok=True)
        label = f"prefix_grouper/{mode}/{path_name}"
        with accelerator.profiler(
            str(capture_dir),
            worker_name,
            record_shapes=bool(settings["profile_record_shapes"]),
            profile_memory=bool(settings["profile_memory"]),
        ) as profiler:
            with torch.profiler.record_function(label):
                step(grouped)
            profiler.step()
        output_dirs[path_name] = str(capture_dir)
        labels[path_name] = label
    return {
        "enabled": True,
        "output_dirs": output_dirs,
        "worker_name": worker_name,
        "labels": labels,
        "record_shapes": bool(settings["profile_record_shapes"]),
        "profile_memory": bool(settings["profile_memory"]),
    }


def _benchmark_mode(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    case: Case,
    mode: str,
    settings: dict[str, Any],
    accelerator: AcceleratorRuntime,
    power_probe: PowerProbe | None,
    idle_watts: float | None,
) -> dict[str, Any]:
    model.train(mode == "forward-backward")

    def step(grouped: bool) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        logits = _response_logits(model, batch, grouped)
        if mode == "forward-backward":
            loss = F.cross_entropy(logits.flatten(0, 1), batch["responses"].flatten())
            loss.backward()
            return loss
        return logits

    # Multi-rank FSDP needs normal tensor version counters while installing its
    # pre/post-forward hooks; inference_mode removes them and breaks the hook path.
    context = torch.no_grad() if mode == "forward" else torch.enable_grad()
    with context:
        baseline_samples = _timed(lambda: step(False), settings["warmup"], settings["repeats"], accelerator)
        grouped_samples = _timed(lambda: step(True), settings["warmup"], settings["repeats"], accelerator)
        baseline_power = _workload_power(
            lambda: step(False),
            probe=power_probe,
            repeats=settings["power_repeats"],
            interval=settings["power_interval"],
            accelerator=accelerator,
        )
        grouped_power = _workload_power(
            lambda: step(True),
            probe=power_probe,
            repeats=settings["power_repeats"],
            interval=settings["power_interval"],
            accelerator=accelerator,
        )
        baseline_peak, baseline_incremental = _peak_memory(lambda: step(False), accelerator)
        grouped_peak, grouped_incremental = _peak_memory(lambda: step(True), accelerator)
        profile = _profile_comparison(
            step,
            case=case,
            mode=mode,
            settings=settings,
            accelerator=accelerator,
        )
    tokens = batch["responses"].numel()
    baseline_ms = percentile(sorted(baseline_samples), 0.5)
    grouped_ms = percentile(sorted(grouped_samples), 0.5)
    return {
        "mode": mode,
        "profile": profile,
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


def _correctness(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    with torch.no_grad():
        baseline_logits = _response_logits(model, batch, False)
        grouped_logits = _response_logits(model, batch, True)
        response_ids = batch["responses"].unsqueeze(-1)
        baseline_log_probs = baseline_logits.float().log_softmax(-1).gather(-1, response_ids).squeeze(-1)
        grouped_log_probs = grouped_logits.float().log_softmax(-1).gather(-1, response_ids).squeeze(-1)

    logit_difference = (baseline_logits.float() - grouped_logits.float()).abs()
    log_prob_difference = (baseline_log_probs - grouped_log_probs).abs()
    low_precision = baseline_logits.dtype in {torch.bfloat16, torch.float16}
    max_tolerance = 2.0 if low_precision else 0.3
    mean_tolerance = 1.0 if low_precision else 0.1
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


def _worker_accelerator() -> AcceleratorRuntime:
    device_name = get_device_name()
    if device_name not in {"cuda", "npu"}:
        raise RuntimeError(f"PrefixGrouper benchmark requires cuda or npu workers, found {device_name!r}.")
    return AcceleratorRuntime(
        backend="npu" if device_name == "npu" else "gpu",
        device=torch.device(device_name, get_device_id()),
        module=get_torch_device(),
    )


def _assigned_physical_device() -> int:
    accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
    for resource_name in ("NPU", "GPU"):
        values = accelerator_ids.get(resource_name, [])
        if values:
            try:
                return int(str(values[0]).split(".", 1)[0])
            except ValueError:
                break
    return int(get_device_id())


def _local_parameter_count(model: torch.nn.Module) -> int:
    """Count this rank's local shards for either FSDP1 or FSDP2."""
    from torch.distributed.tensor import DTensor

    return sum(
        parameter.to_local().numel() if isinstance(parameter, DTensor) else parameter.numel()
        for parameter in model.parameters()
    )


class DistributedPrefixGrouperBenchmarkWorker(PrefixGrouperTrainingWorker):
    """Run one local PrefixGrouper group on every VERL FSDP rank."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def run_benchmark(self, settings: dict[str, Any]) -> dict[str, Any]:
        accelerator = _worker_accelerator()
        accelerator.set_device()
        rank = torch.distributed.get_rank()
        seed = int(settings["seed"]) + rank * 1_000_003
        torch.manual_seed(seed)
        accelerator.manual_seed_all(seed)
        model = self.engine.module
        model.eval()
        config = self.model_config.hf_config

        power_probe = None
        if settings["power_enabled"]:
            power_probe = PowerProbe(
                backend=accelerator.backend,
                device_index=_assigned_physical_device(),
                chip_id=int(settings["npu_chip_id"]),
            )
        idle_measurement = _idle_power(
            power_probe,
            int(settings["power_idle_samples"]),
            float(settings["power_interval"]),
        )
        idle_watts = float(idle_measurement["watts"]["mean"]) if idle_measurement.get("available") else None

        cases = []
        for case_data in settings["cases"]:
            case = Case(**case_data)
            max_positions = getattr(config, "max_position_embeddings", None)
            if max_positions and case.prompt_length + settings["response_length"] > max_positions:
                raise ValueError(
                    f"{self.model_config.path} context is too short for "
                    f"{case.prompt_length}+{settings['response_length']}."
                )
            batch = make_batch(
                config,
                batch_size=int(settings["batch_size_per_rank"]),
                prompt_length=case.prompt_length,
                response_length=int(settings["response_length"]),
                group_size=case.group_size,
                device=accelerator.device,
                seed=seed,
            )
            check = _correctness(model, batch)
            if not check["passed"]:
                raise RuntimeError(f"rank={rank} {case} correctness failed: {check}")
            modes = ["forward", "forward-backward"] if settings["backward"] else ["forward"]
            measurements = [
                _benchmark_mode(
                    model,
                    batch,
                    case=case,
                    mode=mode,
                    settings=settings,
                    accelerator=accelerator,
                    power_probe=power_probe,
                    idle_watts=idle_watts,
                )
                for mode in modes
            ]
            cases.append(
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

        return {
            "rank": rank,
            "world_size": torch.distributed.get_world_size(),
            "hostname": os.uname().nodename,
            "device": str(accelerator.device),
            "physical_device_id": _assigned_physical_device(),
            "device_name": accelerator.device_name(),
            "local_parameter_count": _local_parameter_count(model),
            "idle_power": idle_measurement,
            "model_type": config.model_type,
            "cases": cases,
        }


@ray.remote(num_cpus=1)
class DistributedBenchmarkTaskRunner:
    """Mirror Agent Lightning's TaskRunner and let VERL own the worker group."""

    def run(self, model_path: str, settings: dict[str, Any]) -> list[dict[str, Any]]:
        from ray.util.placement_group import remove_placement_group
        from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager

        model_config = HFModelConfig(
            path=model_path,
            load_tokenizer=False,
            trust_remote_code=bool(settings["trust_remote_code"]),
            override_config={"attn_implementation": "sdpa", "use_cache": False},
            enable_gradient_checkpointing=bool(settings["gradient_checkpointing"]),
            use_remove_padding=False,
            use_fused_kernels=False,
        )
        dtype = str(settings["dtype"])
        engine_config = FSDPEngineConfig(
            strategy=str(settings["strategy"]),
            dtype=dtype,
            model_dtype=dtype,
            mixed_precision={
                "param_dtype": dtype,
                "reduce_dtype": "float32",
                "buffer_dtype": "float32",
            },
            wrap_policy={"min_num_params": 0},
            reshard_after_forward=True,
            fsdp_size=-1,
            forward_only=False,
            use_dynamic_bsz=False,
            micro_batch_size_per_gpu=int(settings["batch_size_per_rank"]),
            ulysses_sequence_parallel_size=1,
            use_torch_compile=False,
            use_remove_padding=False,
            seed=int(settings["seed"]),
        )
        worker_config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=FSDPOptimizerConfig(
                lr=1e-6,
                total_training_steps=1,
                lr_warmup_steps=0,
                weight_decay=0.0,
            ),
            checkpoint_config=CheckpointConfig(),
            profiler_config=None,
        )
        role = 0
        pool_name = f"benchmark_pool_{uuid.uuid4().hex}_"
        pool_manager = ResourcePoolManager(
            resource_pool_spec={pool_name: [int(settings["n_devices_per_node"])] * int(settings["nnodes"])},
            mapping={role: pool_name},
            max_colocate_count=1,
        )
        pool_manager.create_resource_pool()
        resource_pool = pool_manager.get_resource_pool(role)
        try:
            worker_group = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=RayClassWithInitArgs(
                    ray.remote(DistributedPrefixGrouperBenchmarkWorker),
                    config=worker_config,
                ),
                device_name=str(settings["device_name"]),
            )
            worker_group.reset()
            return worker_group.run_benchmark(settings)
        finally:
            for placement_group in resource_pool.pgs or []:
                remove_placement_group(placement_group)
