#!/usr/bin/env python3
"""Compare baseline/simple outputs from benchmark_multiturn_online_e2e.py."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

BENCHMARK_ID = "agl-multiturn-online-e2e"
RESULT_SCHEMA_VERSION = 2
Direction = Literal["higher", "lower", "neutral"]
Record = dict[str, Any]
IDENTITY_FIELDS = {
    "record_type",
    "schema_version",
    "benchmark_id",
    "task",
    "mode",
    "backend",
    "global_step",
}
COMPARABLE_FIELDS = (
    "schema_version",
    "benchmark_id",
    "task",
    "backend",
    "device_name",
    "required_cann",
    "steps",
    "tasks",
    "train_batch_size",
    "rollouts_per_sample",
    "micro_batch_size_per_device",
    "n_devices_per_node",
    "tensor_model_parallel_size",
    "n_runners",
    "max_prompt_length",
    "max_response_length",
    "rollout_max_model_len",
    "rollout_max_tokens",
    "temperature",
    "learning_rate",
    "save_freq",
    "seed",
    "model",
    "model_name",
    "dataset",
    "task_settings",
    "stack",
)


def _load_jsonl(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            records.append(value)
    return records


def load_run(directory: Path, mode: str) -> tuple[Record, list[Record]]:
    path = directory.expanduser().resolve() / "metrics.jsonl"
    records = _load_jsonl(path)
    unexpected = {record.get("record_type") for record in records} - {"run", "step"}
    if unexpected:
        raise ValueError(f"{path} contains unexpected record types: {sorted(map(str, unexpected))}.")
    runs = [record for record in records if record.get("record_type") == "run"]
    steps = [record for record in records if record.get("record_type") == "step"]
    if len(runs) != 1:
        raise ValueError(f"{path} must contain exactly one run record; found {len(runs)}.")
    run = runs[0]
    for record in records:
        if record.get("benchmark_id") != BENCHMARK_ID:
            raise ValueError(f"{path} contains a different benchmark ID.")
        if record.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError(f"{path} contains an unsupported schema version.")
        if record.get("mode") != mode:
            raise ValueError(f"{path} contains mode {record.get('mode')!r}; expected {mode!r}.")
    expected_steps = run.get("steps")
    if not isinstance(expected_steps, int) or expected_steps <= 0 or len(steps) != expected_steps:
        raise ValueError(f"{path} has {len(steps)} step records but declares {expected_steps!r}.")
    steps.sort(key=lambda record: record.get("global_step", -1))
    if [record.get("global_step") for record in steps] != list(range(1, expected_steps + 1)):
        raise ValueError(f"{path} global steps are not contiguous from 1.")
    return run, steps


def validate_comparable(baseline: Record, simple: Record) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    invariants: dict[str, Any] = {}
    for field in COMPARABLE_FIELDS:
        if baseline.get(field) != simple.get(field):
            mismatches[field] = {"baseline": baseline.get(field), "simple": simple.get(field)}
        else:
            invariants[field] = baseline.get(field)
    if mismatches:
        raise ValueError("Runs are not comparable: " + json.dumps(mismatches, ensure_ascii=False, sort_keys=True))
    return invariants


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> Record:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
    }


def common_metrics(baseline: list[Record], simple: list[Record]) -> list[str]:
    names = (set(baseline[0]) & set(simple[0])) - IDENTITY_FIELDS
    for record in baseline[1:] + simple[1:]:
        names &= set(record)
    return sorted(name for name in names if all(_number(record.get(name)) is not None for record in baseline + simple))


def _direction(metric: str) -> Direction:
    if metric.startswith("timing_") or metric.startswith("actor/perf/"):
        return "lower"
    if metric == "perf/throughput" or metric.startswith("perf/mfu/"):
        return "higher"
    return "neutral"


def compare_window(baseline: list[Record], simple: list[Record], metrics: list[str]) -> Record:
    result: Record = {}
    for metric in metrics:
        baseline_summary = summarize([float(record[metric]) for record in baseline])
        simple_summary = summarize([float(record[metric]) for record in simple])
        baseline_mean = float(baseline_summary["mean"])
        simple_mean = float(simple_summary["mean"])
        direction = _direction(metric)
        speedup = None
        improvement = None
        if direction == "lower" and simple_mean:
            speedup = baseline_mean / simple_mean
            improvement = (baseline_mean - simple_mean) / baseline_mean * 100 if baseline_mean else None
        elif direction == "higher" and baseline_mean:
            speedup = simple_mean / baseline_mean
            improvement = (simple_mean - baseline_mean) / baseline_mean * 100
        result[metric] = {
            "direction": direction,
            "baseline": baseline_summary,
            "simple": simple_summary,
            "simple_minus_baseline": simple_mean - baseline_mean,
            "simple_over_baseline": simple_mean / baseline_mean if baseline_mean else None,
            "speedup": speedup,
            "improvement_percent": improvement,
        }
    return result


def build_report(
    baseline_run: Record,
    baseline_steps: list[Record],
    simple_run: Record,
    simple_steps: list[Record],
) -> Record:
    invariants = validate_comparable(baseline_run, simple_run)
    metrics = common_metrics(baseline_steps, simple_steps)
    if "timing_s/step" not in metrics or "training/reward" not in metrics:
        raise ValueError("Required timing_s/step or training/reward metric is missing.")
    baseline_wall = float(baseline_run["wall_seconds"])
    simple_wall = float(simple_run["wall_seconds"])
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison_valid": True,
        "invariants": invariants,
        "run_wall_seconds": {
            "baseline": baseline_wall,
            "simple": simple_wall,
            "simple_minus_baseline": simple_wall - baseline_wall,
            "speedup": baseline_wall / simple_wall if simple_wall else None,
        },
        "windows": {
            "all_steps": {
                "step_numbers": list(range(1, len(baseline_steps) + 1)),
                "metrics": compare_window(baseline_steps, simple_steps, metrics),
            },
            "steady_state": {
                "step_numbers": list(range(2, len(baseline_steps) + 1)),
                "metrics": (
                    compare_window(baseline_steps[1:], simple_steps[1:], metrics) if len(baseline_steps) > 1 else {}
                ),
            },
        },
        "metric_coverage": metrics,
        "notes": [
            "steady_state 固定排除 global step 1",
            "lower 指标的 speedup=baseline/simple；higher 指标的 speedup=simple/baseline",
            "reward 为 neutral，只报告原始统计和差值，不声明性能改善",
            "在线 rollout 使用相同 seed 和采样配置，但随机生成可能产生不同响应；报告同时保留 reward",
            "在线请求预留输出预算，超长输入默认从左侧截断；训练使用实际服务 token IDs",
        ],
    }


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6g}"


def render_markdown(report: Record) -> str:
    invariants = report["invariants"]
    metrics = report["windows"]["steady_state"]["metrics"]
    preferred = (
        "timing_s/step",
        "timing_s/rollout_execution",
        "timing_s/trace_conversion",
        "timing_s/old_log_prob",
        "timing_s/ref",
        "timing_s/update_actor",
        "perf/throughput",
        "actor/perf/max_memory_allocated_gb",
        "training/reward",
    )
    shown = [name for name in preferred if name in metrics]
    lines = [
        f"# {invariants['task']} 多轮端到端 baseline/simple 对比",
        "",
        f"模型：`{invariants['model_name']}`；后端：`{invariants['backend']}`；"
        f"step：{invariants['steps']}；rollout：{invariants['rollouts_per_sample']}。",
        "",
        "下表为 steady-state（固定排除 step 1）。",
        "",
        "| 指标 | baseline mean | simple mean | 差值 | 改善 | speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in shown:
        item = metrics[name]
        improvement = item["improvement_percent"]
        lines.append(
            f"| `{name}` | {_fmt(item['baseline']['mean'])} | {_fmt(item['simple']['mean'])} | "
            f"{_fmt(item['simple_minus_baseline'])} | "
            f"{'N/A' if improvement is None else f'{float(improvement):+.3f}%'} | {_fmt(item['speedup'])} |"
        )
    wall = report["run_wall_seconds"]
    lines.extend(
        [
            "",
            f"进程墙钟时间：baseline {_fmt(wall['baseline'])} s，simple {_fmt(wall['simple'])} s，"
            f"speedup {_fmt(wall['speedup'])}。",
            "",
            "完整逐指标统计、all-steps 窗口和可比性字段见 `report.json`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--simple-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline_run, baseline_steps = load_run(args.baseline_dir, "baseline")
    simple_run, simple_steps = load_run(args.simple_dir, "simple")
    report = build_report(baseline_run, baseline_steps, simple_run, simple_steps)
    report["sources"] = {
        "baseline": str(args.baseline_dir.expanduser().resolve()),
        "simple": str(args.simple_dir.expanduser().resolve()),
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(report)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
