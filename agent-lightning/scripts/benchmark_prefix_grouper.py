# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""Benchmark standard attention against PrefixGrouper through VERL's Ray/FSDP entrypoint.

The controller never initializes a process group or assigns accelerator devices.
Agent Lightning selects the platform, while VERL's ``RayWorkerGroup`` owns worker
placement, ranks, HCCL/NCCL initialization, and FSDP model/gradient sharding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import ray
import torch
from omegaconf import OmegaConf
from packaging.version import Version
from prefix_grouper_stack import NPU_CANN_VERSION, REQUIRED_STACKS

from agentlightning.verl.accelerator import select_accelerator
from agentlightning.verl.entrypoint import configure_accelerator
from agentlightning.verl.model_download import materialize_model_for_npu
from agentlightning.verl.prefix_grouper_benchmark import (
    Case,
    DistributedBenchmarkTaskRunner as _DistributedBenchmarkTaskRunner,
    add_energy_efficiency,
    parse_power_watts,
    percentile,
    sample_statistics,
    workload_metrics,
)

__all__ = [
    "add_energy_efficiency",
    "parse_power_watts",
    "sample_statistics",
    "workload_metrics",
]

DEFAULT_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
DIST_NAMES = {
    "torch_npu": "torch-npu",
    "triton_ascend": "triton-ascend",
    "vllm_ascend": "vllm-ascend",
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
    parser = argparse.ArgumentParser(description="通过 Agent Lightning + VERL Ray/FSDP 输出 PrefixGrouper PPA 对比")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--case", action="append", type=parse_case, dest="cases")
    parser.add_argument(
        "--batch-size-per-rank",
        type=int,
        default=8,
        help="每个 FSDP data-parallel rank 的 batch；共享前缀组不会跨 rank 拆分",
    )
    parser.add_argument("--response-length", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", choices=("auto", "gpu", "cuda", "npu"), default="auto")
    parser.add_argument("--strategy", choices=("fsdp", "fsdp2"), default="fsdp")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument(
        "--n-devices-per-node",
        type=int,
        help="每节点 Ray worker 数；单节点默认使用当前进程可见的全部加速器",
    )
    parser.add_argument("--ray-address", help="传给 ray.init(address=...) 的集群地址")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="由 VERL/Hugging Face 模型启用激活重计算；默认开启以降低 backward 峰值显存",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--power",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="采集整卡功耗；默认在 NPU 开启、GPU 关闭",
    )
    parser.add_argument(
        "--power-repeats",
        type=int,
        default=5,
        help="功耗采集时各 rank 同步执行的固定 step 数；固定次数可避免分布式 collective 失配",
    )
    parser.add_argument("--power-interval", type=float, default=0.1, help="功耗采样间隔（秒）")
    parser.add_argument("--power-idle-samples", type=int, default=3, help="每个模型加载后的空闲功耗采样数")
    parser.add_argument("--npu-chip-id", type=int, default=0, help="npu-smi 功耗查询使用的 chip id")
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="在每个 Ray worker 内对 baseline 和 PrefixGrouper 各采集一个相同范围的 step",
    )
    parser.add_argument("--profile-dir", type=Path, default=Path("prefix_grouper_profiles"))
    parser.add_argument(
        "--profile-record-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在 profile 中记录算子输入 shape；默认开启",
    )
    parser.add_argument(
        "--profile-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="在 profile 中记录算子内存；会显著增大采集开销和输出体积",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output-json", type=Path, default=Path("prefix_grouper_results.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("prefix_grouper_results.md"))
    args = parser.parse_args(argv)
    args.cases = args.cases or [Case(512, 4), Case(1024, 8)]
    if args.batch_size_per_rank <= 0 or args.response_length <= 0 or args.repeats <= 0 or args.warmup < 0:
        parser.error("batch、response、repeats 必须为正数，warmup 不能为负数")
    if args.power_repeats <= 0 or args.power_interval <= 0 or args.power_idle_samples <= 0:
        parser.error("power-repeats、power-interval 和 power-idle-samples 必须为正数")
    if args.nnodes <= 0 or (args.n_devices_per_node is not None and args.n_devices_per_node <= 0):
        parser.error("nnodes 和 n-devices-per-node 必须为正数")
    if args.npu_chip_id < 0:
        parser.error("npu-chip-id 不能为负数")
    for case in args.cases:
        if case.group_size > args.batch_size_per_rank or args.batch_size_per_rank % case.group_size:
            parser.error(
                f"组大小 {case.group_size} 必须整除 batch-size-per-rank={args.batch_size_per_rank}；"
                "PrefixGrouper 共享组不能跨 data-parallel rank"
            )
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


def _multiply_workload(workload: dict[str, Any], world_size: int) -> dict[str, Any]:
    result = dict(workload)
    for key in (
        "response_tokens",
        "baseline_dense_model_tokens",
        "prefix_grouper_dense_model_tokens",
        "baseline_causal_attention_pairs",
        "prefix_grouper_causal_attention_pairs",
    ):
        result[key] = int(result[key]) * world_size
    result["scope"] = "global"
    result["world_size"] = world_size
    return result


def _aggregate_accuracy(rank_values: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "passed": all(value["passed"] for value in rank_values),
        "max_response_log_prob_error": max(value["max_response_log_prob_error"] for value in rank_values),
        "mean_response_log_prob_error": statistics.mean(value["mean_response_log_prob_error"] for value in rank_values),
        "max_logit_error": max(value["max_logit_error"] for value in rank_values),
        "mean_logit_error": statistics.mean(value["mean_logit_error"] for value in rank_values),
        "top1_agreement": statistics.mean(value["top1_agreement"] for value in rank_values),
        "max_tolerance": rank_values[0]["max_tolerance"],
        "mean_tolerance": rank_values[0]["mean_tolerance"],
        "per_rank": rank_values,
    }


def _aggregate_power(
    rank_values: list[dict[str, Any]],
    *,
    latency_ms: float,
    response_tokens: int,
    idle_watts: float | None,
) -> dict[str, Any]:
    if not rank_values or not all(value.get("available") for value in rank_values):
        reasons = sorted({str(value.get("reason", "设备功耗不可用")) for value in rank_values})
        return {
            "enabled": any(value.get("enabled") for value in rank_values),
            "available": False,
            "reason": "；".join(reasons),
            "per_rank": rank_values,
        }
    statistic_keys = ("min", "max", "mean", "p50", "p90", "p95", "p99")
    watts = {key: sum(float(value["watts"][key]) for value in rank_values) for key in statistic_keys}
    watts.update(
        {
            "count": min(int(value["watts"]["count"]) for value in rank_values),
            "stdev": math.sqrt(sum(float(value["watts"]["stdev"]) ** 2 for value in rank_values)),
        }
    )
    watts["coefficient_of_variation"] = watts["stdev"] / watts["mean"] if watts["mean"] else 0.0
    return add_energy_efficiency(
        {
            "enabled": True,
            "available": True,
            "scope": "all_devices_sum",
            "watts": watts,
            "per_rank": rank_values,
        },
        latency_ms=latency_ms,
        response_tokens=response_tokens,
        idle_watts=idle_watts,
    )


def _aggregate_idle_power(rank_values: list[dict[str, Any]]) -> dict[str, Any]:
    if not rank_values or not all(value.get("available") for value in rank_values):
        return {
            "enabled": any(value.get("enabled") for value in rank_values),
            "available": False,
            "reason": "；".join(sorted({str(value.get("reason", "设备功耗不可用")) for value in rank_values})),
            "per_rank": rank_values,
        }
    watts = {
        key: sum(float(value["watts"][key]) for value in rank_values)
        for key in ("min", "max", "mean", "p50", "p90", "p95", "p99")
    }
    watts["count"] = min(int(value["watts"]["count"]) for value in rank_values)
    watts["stdev"] = math.sqrt(sum(float(value["watts"]["stdev"]) ** 2 for value in rank_values))
    watts["coefficient_of_variation"] = watts["stdev"] / watts["mean"] if watts["mean"] else 0.0
    return {"enabled": True, "available": True, "scope": "all_devices_sum", "watts": watts, "per_rank": rank_values}


def _aggregate_measurement(
    rank_values: list[dict[str, Any]],
    *,
    global_response_tokens: int,
    idle_watts: float | None,
) -> dict[str, Any]:
    paths = ("baseline", "prefix_grouper")
    performance: dict[str, Any] = {}
    for path in paths:
        samples_by_rank = [value["performance"][path]["latency_ms"]["samples"] for value in rank_values]
        critical = [max(samples) for samples in zip(*samples_by_rank, strict=True)]
        median_ms = percentile(sorted(critical), 0.5)
        performance[path] = {
            "latency_ms": {**sample_statistics(critical), "samples": critical, "scope": "slowest_rank"},
            "response_tokens_per_second": global_response_tokens * 1000 / median_ms,
        }
    performance["speedup"] = (
        performance["baseline"]["latency_ms"]["p50"] / performance["prefix_grouper"]["latency_ms"]["p50"]
    )

    memory_paths: dict[str, Any] = {}
    for path in paths:
        per_rank = [value["memory"][path] for value in rank_values]
        memory_paths[path] = {
            "peak_mib": max(value["peak_mib"] for value in per_rank),
            "incremental_peak_mib": max(value["incremental_peak_mib"] for value in per_rank),
            "aggregate_peak_mib": sum(value["peak_mib"] for value in per_rank),
            "aggregate_incremental_peak_mib": sum(value["incremental_peak_mib"] for value in per_rank),
            "peak_bytes_per_response_token": sum(value["peak_mib"] for value in per_rank)
            * 2**20
            / global_response_tokens,
            "per_rank": per_rank,
        }
    baseline_memory = memory_paths["baseline"]
    grouped_memory = memory_paths["prefix_grouper"]
    memory = {
        **memory_paths,
        "saved_mib": baseline_memory["peak_mib"] - grouped_memory["peak_mib"],
        "saved_ratio": 1 - grouped_memory["aggregate_peak_mib"] / baseline_memory["aggregate_peak_mib"],
        "aggregate_saved_mib": baseline_memory["aggregate_peak_mib"] - grouped_memory["aggregate_peak_mib"],
    }

    power = {}
    for path in paths:
        latency_ms = float(performance[path]["latency_ms"]["p50"])
        power[path] = _aggregate_power(
            [value["power"][path] for value in rank_values],
            latency_ms=latency_ms,
            response_tokens=global_response_tokens,
            idle_watts=idle_watts,
        )
    return {"mode": rank_values[0]["mode"], "performance": performance, "power": power, "memory": memory}


def aggregate_model_result(
    model_ref: str,
    resolved_model_path: str,
    materialization: Any,
    rank_results: list[dict[str, Any]],
    *,
    batch_size_per_rank: int,
) -> dict[str, Any]:
    if not rank_results:
        raise RuntimeError("VERL worker group did not return benchmark results.")
    rank_results = sorted(rank_results, key=lambda value: value["rank"])
    world_size = len(rank_results)
    if any(result["world_size"] != world_size for result in rank_results):
        raise RuntimeError("VERL worker world-size metadata is inconsistent.")
    idle_measurement = _aggregate_idle_power([result["idle_power"] for result in rank_results])
    idle_watts = float(idle_measurement["watts"]["mean"]) if idle_measurement.get("available") else None

    cases = []
    for case_index, first_case in enumerate(rank_results[0]["cases"]):
        rank_cases = [result["cases"][case_index] for result in rank_results]
        workload = _multiply_workload(first_case["workload"], world_size)
        global_response_tokens = int(workload["response_tokens"])
        cases.append(
            {
                "prompt_length": first_case["prompt_length"],
                "group_size": first_case["group_size"],
                "workload": workload,
                "accuracy": _aggregate_accuracy([case["accuracy"] for case in rank_cases]),
                "measurements": [
                    _aggregate_measurement(
                        [case["measurements"][index] for case in rank_cases],
                        global_response_tokens=global_response_tokens,
                        idle_watts=idle_watts,
                    )
                    for index in range(len(first_case["measurements"]))
                ],
            }
        )
    return {
        "model": model_ref,
        "resolved_model_path": resolved_model_path,
        "model_materialization": asdict(materialization) if materialization is not None else None,
        "label": Path(model_ref).name,
        "model_type": rank_results[0]["model_type"],
        "parameters": sum(result["local_parameter_count"] for result in rank_results),
        "world_size": world_size,
        "batch_size_per_rank": batch_size_per_rank,
        "global_batch_size": batch_size_per_rank * world_size,
        "workers": [
            {
                key: result[key]
                for key in ("rank", "hostname", "device", "physical_device_id", "device_name", "local_parameter_count")
            }
            for result in rank_results
        ],
        "idle_power": idle_measurement,
        "cases": cases,
    }


def markdown(results: dict[str, Any]) -> str:
    lines = [
        "# PrefixGrouper PPA 对比",
        "",
        f"- 加速器：{results['accelerator']['backend']} / {results['accelerator']['name']}，"
        f"world size={results['distributed']['world_size']}（{results['distributed']['nnodes']} 节点 × "
        f"{results['distributed']['n_devices_per_node']} 设备）",
        f"- 软件栈：torch {results['stack']['torch']} / vLLM {results['stack']['vllm']} / VERL {results['stack']['verl']}",
        f"- 分布式入口：Agent Lightning accelerator selection + VERL RayWorkerGroup/{results['distributed']['strategy']}",
        f"- 精度：{results['dtype']}，权重：pretrained",
        f"- Batch：每 rank {results['batch_size_per_rank']}，全局 {results['global_batch_size']}",
        f"- NPU 模型下载目录：{results['model_download_root'] or '不适用（GPU）'}",
        "- PPA 口径：延迟取所有 rank 的慢端值，吞吐按全局 response token 计算；功耗为所有设备之和；显存同时报告最坏 rank 和设备总和。",
    ]
    if results["profile"]["enabled"]:
        lines.append(f"- Profile：{results['profile']['output_dir']}；{results['profile']['scope']}")
    lines.extend(
        [
            "",
            "## 工作量",
            "",
            "| 模型 | Prompt | 共享组 | Dense token 前/后 | Dense token 减少 | Causal pair 前/后 | Attention pair 减少 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
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
            "| 模型 | 模式 | Prompt | 共享组 | 最坏 rank 前/后 MiB | 全设备前/后 MiB | 全设备节省 MiB | 节省比例 | 最坏 rank 增量前/后 MiB |",
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
                    f"{baseline['peak_mib']:.1f}/{grouped['peak_mib']:.1f} | "
                    f"{baseline['aggregate_peak_mib']:.1f}/{grouped['aggregate_peak_mib']:.1f} | "
                    f"{memory['aggregate_saved_mib']:.1f} | {memory['saved_ratio']:.1%} | "
                    f"{baseline['incremental_peak_mib']:.1f}/"
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
    accelerator.set_device()
    if accelerator.backend == "gpu":
        torch.backends.cuda.matmul.allow_tf32 = True

    platform_config = OmegaConf.create({"trainer": {"device": accelerator.device_type}})
    device_name = configure_accelerator(platform_config)
    n_devices_per_node = args.n_devices_per_node or int(accelerator.module.device_count())
    if n_devices_per_node > int(accelerator.module.device_count()) and args.nnodes == 1:
        raise ValueError(f"请求 {n_devices_per_node} 个设备，但当前节点只发现 {accelerator.module.device_count()} 个。")
    world_size = args.nnodes * n_devices_per_node
    power_enabled = args.power if args.power is not None else accelerator.backend == "npu"
    profile_root = args.profile_dir.resolve() if args.profile else None
    model_download_root = Path.cwd().resolve() if accelerator.backend == "npu" else None
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    settings: dict[str, Any] = {
        "cases": [asdict(case) for case in args.cases],
        "batch_size_per_rank": args.batch_size_per_rank,
        "response_length": args.response_length,
        "dtype": args.dtype,
        "strategy": args.strategy,
        "nnodes": args.nnodes,
        "n_devices_per_node": n_devices_per_node,
        "device_name": device_name,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "backward": args.backward,
        "gradient_checkpointing": args.gradient_checkpointing,
        "trust_remote_code": args.trust_remote_code,
        "power_enabled": power_enabled,
        "power_repeats": args.power_repeats,
        "power_interval": args.power_interval,
        "power_idle_samples": args.power_idle_samples,
        "npu_chip_id": args.npu_chip_id,
        "profile_enabled": args.profile,
        "profile_output_dir": None,
        "profile_record_shapes": args.profile_record_shapes,
        "profile_memory": args.profile_memory,
        "seed": args.seed,
    }

    results: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stack": stack,
        "accelerator": {
            "backend": accelerator.display_backend,
            "device": device_name,
            "name": accelerator.device_name(),
        },
        "distributed": {
            "entrypoint": "agentlightning.verl.entrypoint.configure_accelerator + verl.single_controller.ray.RayWorkerGroup",
            "strategy": args.strategy,
            "gradient_checkpointing": args.gradient_checkpointing,
            "nnodes": args.nnodes,
            "n_devices_per_node": n_devices_per_node,
            "world_size": world_size,
        },
        "required_cann": NPU_CANN_VERSION if accelerator.backend == "npu" else None,
        "dtype": args.dtype,
        "weights": "pretrained",
        "model_download_root": str(model_download_root) if model_download_root is not None else None,
        "batch_size_per_rank": args.batch_size_per_rank,
        "global_batch_size": args.batch_size_per_rank * world_size,
        "response_length": args.response_length,
        "ppa_definition": {
            "performance": "各 rank 设备事件延迟的慢端值；吞吐按全局 response token 计算",
            "power": "所有 rank 对应设备的 npu-smi/nvidia-smi 功耗之和",
            "accuracy": "所有 rank 的 response log-prob、logit 和 top-1 一致性",
            "memory": "最坏 rank 峰值与所有 rank 峰值之和",
        },
        "power_sampling": {
            "enabled": power_enabled,
            "workload_repeats_per_path": args.power_repeats if power_enabled else None,
            "interval_seconds": args.power_interval if power_enabled else None,
            "npu_chip_id": args.npu_chip_id if accelerator.backend == "npu" else None,
        },
        "profile": {
            "enabled": args.profile,
            "output_dir": str(profile_root) if profile_root is not None else None,
            "scope": "每个 case/mode 在正常计时后额外采集 baseline 与 PrefixGrouper 各一个 step",
            "record_shapes": args.profile_record_shapes if args.profile else None,
            "profile_memory": args.profile_memory if args.profile else None,
            "npu_level": "Level1/PipeUtilization/Text+Db" if args.profile and accelerator.backend == "npu" else None,
        },
        "models": [],
    }
    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray_init_kwargs = {"address": args.ray_address} if args.ray_address else {}
        ray.init(**ray_init_kwargs)
    try:
        for index, model_ref in enumerate(args.models, 1):
            print(f"[{index}/{len(args.models)}] {model_ref}，启动 {world_size} 个 VERL FSDP worker", flush=True)
            started = time.monotonic()
            materialization = None
            resolved_model_ref = model_ref
            if model_download_root is not None:
                materialization = materialize_model_for_npu(
                    model_ref,
                    model_download_root,
                    local_files_only=args.local_files_only,
                    download_weights=True,
                )
                resolved_model_ref = materialization.local_path
                print(f"NPU 模型已就绪：{model_ref} -> {resolved_model_ref}", flush=True)

            runner = _DistributedBenchmarkTaskRunner.remote()
            try:
                model_settings = {
                    **settings,
                    "profile_output_dir": (
                        str(profile_root / f"{index:02d}_{Path(model_ref).name}")
                        if profile_root is not None
                        else None
                    ),
                }
                rank_results = ray.get(runner.run.remote(resolved_model_ref, model_settings))
            finally:
                ray.kill(runner)
            model_result = aggregate_model_result(
                model_ref,
                resolved_model_ref,
                materialization,
                rank_results,
                batch_size_per_rank=args.batch_size_per_rank,
            )
            model_result["elapsed_seconds"] = time.monotonic() - started
            results["models"].append(model_result)
            write_report(results, args.output_json, args.output_markdown)
    finally:
        if owns_ray:
            ray.shutdown()

    print("\n" + markdown(results))
    print(f"JSON: {args.output_json.resolve()}")
    print(f"Markdown: {args.output_markdown.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
