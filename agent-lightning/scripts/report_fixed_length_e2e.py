#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Compare completed no-share/simple fixed-length E2E runs without launching training.

Usage::

    python scripts/report_fixed_length_e2e.py --no-share-dir /runs/no-share \
        --simple-dir /runs/simple --output-dir /runs/comparison
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

Record = dict[str, Any]
REQUIRED_TIMINGS = (
    "step",
    "gen",
    "trace_conversion",
    "old_log_prob",
    "ref",
    "adv",
    "update_actor",
)
EXPECTED_STAGES = {"rollout", "trace_conversion", "actor_input", "step_metrics", "step_passed"}


def read_object(path: Path) -> Record:
    """Load a required JSON object with a useful error location."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return cast(Record, value)


def number(value: Any) -> float | None:
    """Reject booleans, strings and nonfinite measurements."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def positive_int(value: Any, name: str) -> int:
    """Validate a declared count or token length."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def normalize_config(config: Record, mode: str, *, resolved: bool) -> Record:
    """Remove only the intended sharing switch/backend and output path."""
    config = copy.deepcopy(config)
    lightning = config["agentlightning"]
    grouping = lightning.get("prefix_grouper", {})
    if grouping.get("enabled", False) is not (mode == "simple"):
        raise ValueError(f"{mode}: configuration does not match the requested sharing mode")
    grouping.pop("enabled", None)
    if grouping:
        raise ValueError(f"{mode}: unsupported additional prefix_grouper configuration")
    lightning.pop("prefix_grouper", None)
    actor = config["actor_rollout_ref"]["actor"]
    if resolved:
        if actor.get("use_prefix_grouper") is not (mode == "simple"):
            raise ValueError(f"{mode}: resolved actor sharing flag is inconsistent")
        actor.pop("use_prefix_grouper")
    override = config["actor_rollout_ref"]["model"]["override_config"]
    if mode == "simple":
        if override.pop("prefix_grouper_npu_backend", None) != "fusion":
            raise ValueError("simple must use the shared pg-verl-ppa fusion implementation")
    elif "prefix_grouper_npu_backend" in override:
        raise ValueError("no-share must not configure a PrefixGrouper backend")
    config["trainer"].pop("default_local_dir", None)
    return config


def load_run(directory: Path, mode: str) -> Record:
    """Require successful, complete runs and verify the raw rollout evidence."""
    if (directory / "failure.json").exists():
        raise ValueError(f"{directory}: failed/partial runs cannot be compared")
    result = read_object(directory / "result.json")
    if result.get("status") != "passed" or result.get("mode") != mode:
        raise ValueError(f"{directory}: expected a passed {mode} run; prepared runs are not measurements")
    run = read_object(directory / "run.json")
    args: Record = run["args"]
    if args.get("mode") != mode or args.get("prepare_only") is not False:
        raise ValueError(f"{directory}: run arguments do not describe a completed {mode} training run")
    steps = positive_int(args["steps"], "steps")
    batch_size = positive_int(args["train_batch_size"], "train_batch_size")
    repeats = positive_int(args["rollouts_per_sample"], "rollouts_per_sample")
    prompt_length = positive_int(args["prompt_tokens"], "prompt_tokens")
    response_length = positive_int(args["response_tokens"], "response_tokens")
    expected_rollouts = steps * batch_size * repeats
    for key, expected in (
        ("steps", steps),
        ("rollouts", expected_rollouts),
        ("prompt_tokens", prompt_length),
        ("response_tokens", response_length),
        ("reward_policy", "synthetic-rollout-index"),
        ("nonzero_actor_gradient", True),
    ):
        if result.get(key) != expected:
            raise ValueError(f"{directory}: result.{key} does not match the workload")

    dataset: list[Record] = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    prompts = {row["sample_id"]: row["prompt_token_ids"] for row in dataset}
    if len(dataset) != batch_size or set(prompts) != set(range(batch_size)):
        raise ValueError(f"{directory}: invalid input dataset")
    if any(len(tokens) != prompt_length for tokens in prompts.values()):
        raise ValueError(f"{directory}: input prompt lengths differ from the configured length")

    events: list[Record] = []
    with (directory / "events.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event: Record = json.loads(line)
            if event.get("mode") != mode or event.get("stage") not in EXPECTED_STAGES:
                raise ValueError(f"{directory}/events.jsonl:{line_number}: unexpected mode or stage")
            events.append(event)
    for stage in EXPECTED_STAGES - {"rollout"}:
        stage_events = [event for event in events if event["stage"] == stage]
        if sorted(event["step"] for event in stage_events) != list(range(1, steps + 1)):
            raise ValueError(f"{directory}: incomplete or duplicate {stage} steps")
        if stage in {"trace_conversion", "actor_input"}:
            if any(event.get("rows") != batch_size * repeats for event in stage_events):
                raise ValueError(f"{directory}: incorrect {stage} batch size")
    rollouts = [event for event in events if event["stage"] == "rollout"]
    if len(rollouts) != expected_rollouts or len({event["rollout_id"] for event in rollouts}) != expected_rollouts:
        raise ValueError(f"{directory}: incomplete or duplicate rollout records")
    counts = Counter((event["sample_id"], event["rollout_index"]) for event in rollouts)
    if counts != Counter({(sample, index): steps for sample in range(batch_size) for index in range(repeats)}):
        raise ValueError(f"{directory}: incomplete rollout groups")
    for event in rollouts:
        tokens: list[Any] = event["response_token_ids"]
        if (
            event["prompt_token_ids"] != prompts[event["sample_id"]]
            or event.get("prompt_tokens") != prompt_length
            or event.get("response_tokens") != response_length
            or len(tokens) != response_length
            or not all(type(token) is int for token in tokens)
            or event.get("finish_reason") != "length"
        ):
            raise ValueError(f"{directory}: rollout does not satisfy the exact token-length contract")
    step_events = sorted((event for event in events if event["stage"] == "step_metrics"), key=lambda x: x["step"])
    for event in step_events:
        for stage in REQUIRED_TIMINGS:
            value = number(event["metrics"].get(f"timing_s/{stage}"))
            if value is None or value <= 0:
                raise ValueError(f"{directory}: missing/nonpositive {stage} timing at step {event['step']}")
        for metric in ("actor/grad_norm", "actor/lr", "actor/pg_loss"):
            value = number(event["metrics"].get(metric))
            if value is None or (metric != "actor/pg_loss" and value <= 0):
                raise ValueError(f"{directory}: invalid {metric} at step {event['step']}")
    device = read_object(directory / "device.json")
    if not device.get("backend") or not device.get("name") or not run.get("versions") or not run.get("platform"):
        raise ValueError(f"{directory}: missing device/software metadata")
    return {
        "directory": str(directory),
        "run": run,
        "result": result,
        "dataset": dataset,
        "device": device,
        "config": normalize_config(read_object(directory / "config.json"), mode, resolved=False),
        "resolved_config": normalize_config(read_object(directory / "resolved_config.json"), mode, resolved=True),
        "steps": step_events,
        "request_identity": Counter(
            (event["sample_id"], event["rollout_index"], event["request_seed"], event["reward"]) for event in rollouts
        ),
    }


def validate_comparable(baseline: Record, simple: Record) -> Record:
    """Reject different workloads, environments or effective training settings."""
    args = copy.deepcopy(baseline["run"]["args"])
    other_args = copy.deepcopy(simple["run"]["args"])
    for values in (args, other_args):
        for key in ("mode", "output_dir", "device"):
            values.pop(key, None)
    mismatches: list[str] = []
    if args != other_args:
        mismatches.append("run arguments")
    for key in ("dataset", "device", "config", "resolved_config", "request_identity"):
        if baseline[key] != simple[key]:
            mismatches.append(key)
    for key in ("versions", "platform"):
        if baseline["run"][key] != simple["run"][key]:
            mismatches.append(key)
    if mismatches:
        raise ValueError("Runs are not comparable; mismatched: " + ", ".join(mismatches))
    return {"args": args, "device": baseline["device"], "versions": baseline["run"]["versions"]}


def summarize(values: list[float]) -> Record:
    """Retain raw samples as well as summary statistics."""
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples": values,
    }


def compare_window(baseline: list[Record], simple: list[Record], response_tokens_per_step: int) -> Record:
    """Compare a fixed step window, with missing metrics explicitly listed."""
    if not baseline:
        return {
            "step_numbers": [],
            "step_summary": None,
            "per_step": [],
            "metrics": {},
            "unavailable_metrics": [],
            "response_throughput": None,
        }
    names: set[str] = set()
    for event in baseline + simple:
        names.update(event["metrics"])
    comparisons: Record = {}
    unavailable: list[str] = []
    for name in sorted(names):
        samples = [[number(event["metrics"].get(name)) for event in events] for events in (baseline, simple)]
        if any(value is None for values in samples for value in values):
            unavailable.append(name)
            continue
        base_stats, simple_stats = [summarize(cast(list[float], values)) for values in samples]
        base_mean, simple_mean = float(base_stats["mean"]), float(simple_stats["mean"])
        is_time = name.startswith("timing_s/")
        is_memory = "memory" in name and (name.endswith("_gb") or name.endswith("_mib"))
        comparisons[name] = {
            "no_share": base_stats,
            "simple": simple_stats,
            "simple_minus_no_share": simple_mean - base_mean,
            "reduction_percent": (
                (100 * (base_mean - simple_mean) / base_mean) if (is_time or is_memory) and base_mean > 0 else None
            ),
            "speedup": base_mean / simple_mean if is_time and base_mean > 0 and simple_mean > 0 else None,
        }
    base_time = sum(float(event["metrics"]["timing_s/step"]) for event in baseline)
    simple_time = sum(float(event["metrics"]["timing_s/step"]) for event in simple)
    per_step: list[Record] = []
    for base_event, simple_event in zip(baseline, simple, strict=True):
        if base_event["step"] != simple_event["step"]:
            raise ValueError("Cannot compare timings for different step numbers")
        base_seconds = float(base_event["metrics"]["timing_s/step"])
        simple_seconds = float(simple_event["metrics"]["timing_s/step"])
        per_step.append(
            {
                "step": base_event["step"],
                "no_share_seconds": base_seconds,
                "simple_seconds": simple_seconds,
                "saved_seconds": base_seconds - simple_seconds,
                "reduction_percent": 100 * (base_seconds - simple_seconds) / base_seconds,
                "speedup": base_seconds / simple_seconds,
            }
        )
    step_count = len(baseline)
    token_count = len(baseline) * response_tokens_per_step
    return {
        "step_numbers": [event["step"] for event in baseline],
        "step_summary": {
            "timing_key": "timing_s/step",
            "count": step_count,
            "no_share": {
                "total_seconds": base_time,
                "mean_seconds": base_time / step_count,
                "steps_per_second": step_count / base_time,
            },
            "simple": {
                "total_seconds": simple_time,
                "mean_seconds": simple_time / step_count,
                "steps_per_second": step_count / simple_time,
            },
            "saved_seconds": base_time - simple_time,
            "reduction_percent": 100 * (base_time - simple_time) / base_time,
            "speedup": base_time / simple_time,
        },
        "per_step": per_step,
        "metrics": comparisons,
        "unavailable_metrics": unavailable,
        "response_throughput": {
            "response_tokens": token_count,
            "no_share_step_seconds": base_time,
            "simple_step_seconds": simple_time,
            "no_share_tokens_per_second": token_count / base_time,
            "simple_tokens_per_second": token_count / simple_time,
            "speedup": base_time / simple_time,
        },
    }


def build_report(baseline: Record, simple: Record) -> Record:
    """Build all-step and step-1-excluded comparisons from validated runs."""
    invariants = validate_comparable(baseline, simple)
    args = invariants["args"]
    tokens = args["train_batch_size"] * args["rollouts_per_sample"] * args["response_tokens"]
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison_valid": True,
        "sources": {"no_share": baseline["directory"], "simple": simple["directory"]},
        "invariants": invariants,
        "windows": {
            "all_steps": compare_window(baseline["steps"], simple["steps"], tokens),
            "excluding_step_1": compare_window(baseline["steps"][1:], simple["steps"][1:], tokens),
        },
        "raw_steps": {"no_share": baseline["steps"], "simple": simple["steps"]},
        "limitations": [
            "step 耗时包含 rollout 和训练阶段，不包含模型加载、Ray 初始化等启动开销；阶段可能嵌套，不能相加。",
            "response 吞吐 = 窗口内实际生成的 response token 总数 / 完整 step 耗时总和，未按设备数归一化。",
            "显存为生产 trainer 原始指标；不代表所有设备的显存总和，也不保证是单 step 重置后的峰值。",
            "排除 step 1 只是固定统计窗口，不证明已经达到稳态；仅有 1 step 时该窗口无数据。",
            "在线采样和策略更新可能产生不同 response；本报告不证明两种实现逐元素数值等价或 kernel 派发一致。",
            "模型按已记录的本地路径比较；现有运行元数据不包含权重内容、完整源码或驱动的指纹。",
        ],
    }


def render_markdown(report: Record) -> str:
    """Render a compact Chinese comparison with explicit metric definitions."""
    args = report["invariants"]["args"]
    lines = [
        "# 固定长度 E2E：no-share 与 simple 对比",
        "",
        f"输入：{args['prompt_tokens']} prompt tokens / {args['response_tokens']} response tokens；"
        f"每步 {args['train_batch_size']} 个 prompt × {args['rollouts_per_sample']} 条 rollout；"
        f"共 {args['steps']} steps，{args['n_devices']} 个设备。",
        "",
        f"no-share：`{report['sources']['no_share']}`",
        f"simple：`{report['sources']['simple']}`",
        "",
        "两次运行均完整通过，已核对输入 token IDs、采样 seed、软件与设备记录、配置和训练步骤。",
        "",
        "耗时加速比 = no-share / simple；降低比例 = (no-share − simple) / no-share × 100%。"
        "加速比大于 1 表示 simple 更快，负的节省时间或降低比例表示 simple 更慢。",
        "",
        "完整 step 直接使用 `timing_s/step`，包含 rollout 和训练，不包含模型加载和 Ray 初始化。"
        "总耗时为所选 steps 的计时之和，不是整个进程的墙钟时间；不累加内部阶段耗时。"
        "整体加速比使用总耗时之比，不对逐 step 加速比取平均。",
        "",
    ]
    for name, title in (("all_steps", "全部 steps"), ("excluding_step_1", "排除 step 1")):
        window = report["windows"][name]
        lines.extend([f"## {title}", ""])
        if not window["step_numbers"]:
            lines.extend(["无数据：运行只有一个训练 step。", ""])
            continue
        summary = window["step_summary"]
        base_step, simple_step = summary["no_share"], summary["simple"]
        lines.extend(
            [
                "### 完整 step 速度与耗时",
                "",
                f"统计 **{summary['count']} steps**；simple 整体加速比 **{summary['speedup']:.3f}×**，"
                f"节省 **{summary['saved_seconds']:.3f} s**（**{summary['reduction_percent']:.2f}%**）。",
                "",
                "| 指标 | no-share | simple |",
                "| --- | ---: | ---: |",
                f"| step 总耗时（s） | {base_step['total_seconds']:.6f} | {simple_step['total_seconds']:.6f} |",
                f"| 平均每步耗时（s/step） | {base_step['mean_seconds']:.6f} | {simple_step['mean_seconds']:.6f} |",
                f"| step 速度（steps/s） | {base_step['steps_per_second']:.6f} | "
                f"{simple_step['steps_per_second']:.6f} |",
                "",
            ]
        )
        if name == "all_steps":
            lines.extend(
                [
                    "### 逐 step 总耗时对比",
                    "",
                    "| Step | no-share（s） | simple（s） | 节省（s） | 耗时降低 | 加速比 |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in window["per_step"]:
                lines.append(
                    f"| {row['step']} | {row['no_share_seconds']:.6f} | {row['simple_seconds']:.6f} | "
                    f"{row['saved_seconds']:.6f} | {row['reduction_percent']:.2f}% | {row['speedup']:.3f}× |"
                )
            lines.append("")
        lines.extend(["### 阶段及其他指标（逐 step 均值）", ""])
        throughput = window["response_throughput"]
        lines.extend(
            [
                f"response 吞吐：no-share **{throughput['no_share_tokens_per_second']:.3f} tokens/s**，"
                f"simple **{throughput['simple_tokens_per_second']:.3f} tokens/s**，"
                f"比值 **{throughput['speedup']:.3f}×**。",
                "",
                "| 指标（原始 key） | no-share 均值 | simple 均值 | 差值 | 降低比例 | 耗时加速比 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric, values in window["metrics"].items():
            reduction = values["reduction_percent"]
            speedup = values["speedup"]
            lines.append(
                f"| `{metric}` | {values['no_share']['mean']:.6g} | {values['simple']['mean']:.6g} | "
                f"{values['simple_minus_no_share']:.6g} | "
                + (f"{reduction:.2f}%" if reduction is not None else "—")
                + " | "
                + (f"{speedup:.3f}×" if speedup is not None else "—")
                + " |"
            )
        lines.append("")
        if window["unavailable_metrics"]:
            lines.extend(["缺失或非有限指标（未比较）：" + ", ".join(window["unavailable_metrics"]), ""])
    lines.extend(["## 统计边界", "", *[f"- {note}" for note in report["limitations"]], ""])
    lines.append("`report.json` 保留均值、中位数、标准差、最小/最大值、原始样本与逐 step 指标。\n")
    return "\n".join(lines)


def main() -> None:
    """Write reports only after both runs have passed the comparison checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-share-dir", type=Path, required=True)
    parser.add_argument("--simple-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New report directory; never overwrites results."
    )
    args = parser.parse_args()
    baseline_dir, simple_dir, output_dir = (
        path.expanduser().resolve() for path in (args.no_share_dir, args.simple_dir, args.output_dir)
    )
    if baseline_dir == simple_dir:
        parser.error("The two modes must come from separate run directories")
    if output_dir.exists():
        parser.error("--output-dir must not already exist")
    try:
        report = build_report(load_run(baseline_dir, "no-share"), load_run(simple_dir, "simple"))
        markdown = render_markdown(report)
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(serialized, encoding="utf-8")
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(f"REPORT_JSON={output_dir / 'report.json'}")
    print(f"REPORT_MARKDOWN={output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
