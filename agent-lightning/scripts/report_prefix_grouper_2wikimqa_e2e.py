#!/usr/bin/env python3
"""Build JSON and Markdown reports for the maintained 2WikiMQA E2E benchmark.

Example:
    python scripts/report_prefix_grouper_2wikimqa_e2e.py \
        --baseline-dir /results/2wikimqa/gpu-baseline \
        --prefix-grouper-dir /results/2wikimqa/gpu-prefix-grouper \
        --output-dir /results/2wikimqa/gpu-report
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

BENCHMARK_ID = "pg-2wikimqa-e2e"
RESULT_SCHEMA_VERSION = 3
Direction = Literal["higher", "lower", "neutral"]
Record = dict[str, Any]

REQUIRED_STAGE_METRICS = (
    "timing_s/step",
    "timing_s/gen",
    "timing_s/weight_sync",
    "timing_s/rollout_setup",
    "timing_s/rollout_execution",
    "timing_s/trace_conversion",
    "timing_s/rollout_cleanup",
    "timing_s/prompt_group_reorder",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/reward",
    "timing_s/adv",
    "timing_s/batch_postprocess",
    "timing_s/update_actor",
)
PER_TOKEN_METRICS = (
    "timing_per_token_ms/gen",
    "timing_per_token_ms/ref",
    "timing_per_token_ms/adv",
    "timing_per_token_ms/update_actor",
)
PERFORMANCE_METRICS = (
    "perf/throughput",
    "perf/mfu/old_log_prob",
    "perf/mfu/actor",
    "actor/perf/max_memory_allocated_gb",
    "actor/perf/max_memory_reserved_gb",
    "actor/perf/cpu_memory_used_gb",
)
QUALITY_METRICS = ("training/reward",)
COMPARABLE_RUN_FIELDS = (
    "schema_version",
    "benchmark_id",
    "dataset",
    "backend",
    "device_name",
    "steps",
    "dataset_rows",
    "train_batch_size",
    "micro_batch_size_per_device",
    "rollouts_per_sample",
    "input_policy",
    "response_policy",
    "min_prompt_tokens",
    "max_prompt_tokens",
    "max_response_tokens",
    "n_devices_per_node",
    "tensor_model_parallel_size",
    "n_runners",
    "seed",
    "model_ref",
    "model_name",
    "model_path",
    "dataset_source",
    "dataset_path",
    "stack",
    "required_cann",
)
DISPLAY_NAMES = {
    "timing_s/step": "完整训练 step (s)",
    "timing_s/gen": "rollout / gen (s)",
    "timing_s/weight_sync": "权重同步 (s)",
    "timing_s/rollout_setup": "rollout 准备 (s)",
    "timing_s/rollout_execution": "agent rollout 执行 (s)",
    "timing_s/trace_conversion": "trace 转换 (s)",
    "timing_s/rollout_cleanup": "rollout 清理 (s)",
    "timing_s/prompt_group_reorder": "共享前缀重排 (s)",
    "timing_s/old_log_prob": "old log-prob (s)",
    "timing_s/ref": "reference log-prob (s)",
    "timing_s/reward": "reward (s)",
    "timing_s/adv": "advantage (s)",
    "timing_s/batch_postprocess": "batch 后处理 (s)",
    "timing_s/update_actor": "actor update (s)",
    "timing_per_token_ms/gen": "rollout / gen (ms/token)",
    "timing_per_token_ms/ref": "reference (ms/token)",
    "timing_per_token_ms/adv": "advantage (ms/token)",
    "timing_per_token_ms/update_actor": "actor update (ms/token)",
    "perf/throughput": "吞吐 (token/s)",
    "perf/mfu/old_log_prob": "old log-prob MFU",
    "perf/mfu/actor": "actor MFU",
    "actor/perf/max_memory_allocated_gb": "actor 峰值已分配显存 (GiB)",
    "actor/perf/max_memory_reserved_gb": "actor 峰值保留显存 (GiB)",
    "actor/perf/cpu_memory_used_gb": "actor CPU 内存 (GiB)",
    "training/reward": "step reward",
}


@dataclass(frozen=True)
class RunArtifacts:
    """One benchmark mode's validated raw artifacts."""

    directory: Path
    metrics_path: Path
    responses_path: Path
    run: Record
    steps: list[Record]
    responses: list[Record]


def parse_args() -> argparse.Namespace:
    """Parse report arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--prefix-grouper-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace report.json and report.md.")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[Record]:
    if not path.is_file():
        raise FileNotFoundError(f"Required benchmark artifact does not exist: {path}")
    records: list[Record] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            records.append(cast(Record, value))
    return records


def _require_int(record: Record, key: str, context: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} requires integer field {key!r}.")
    return value


def _require_float(record: Record, key: str, context: str) -> float:
    value = _finite_number(record.get(key))
    if value is None:
        raise ValueError(f"{context} requires finite numeric field {key!r}.")
    return value


def _require_mode_record(record: Record, *, record_type: str, mode: str, context: str) -> None:
    if record.get("record_type") != record_type:
        raise ValueError(f"{context} has unexpected record_type {record.get('record_type')!r}.")
    if record.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"{context} does not use result schema {RESULT_SCHEMA_VERSION}.")
    if record.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError(f"{context} is not a {BENCHMARK_ID} artifact.")
    if record_type == "step" and record.get("mode") != mode:
        raise ValueError(f"{context} has mode {record.get('mode')!r}; expected {mode!r}.")


def load_artifacts(directory: Path, expected_mode: str) -> RunArtifacts:
    """Load one mode and enforce the maintained result schema."""
    resolved = directory.resolve()
    metrics_path = resolved / "metrics.jsonl"
    responses_path = resolved / "responses.jsonl"
    metric_records = _load_jsonl(metrics_path)
    unexpected_types = sorted({str(record.get("record_type")) for record in metric_records} - {"run", "step"})
    if unexpected_types:
        raise ValueError(f"{metrics_path} contains unexpected record types: {unexpected_types}")
    run_records = [record for record in metric_records if record.get("record_type") == "run"]
    step_records = [record for record in metric_records if record.get("record_type") == "step"]
    if len(run_records) != 1:
        raise ValueError(f"{metrics_path} must contain exactly one run record; found {len(run_records)}.")
    run = run_records[0]
    _require_mode_record(run, record_type="run", mode=expected_mode, context=str(metrics_path))
    if run.get("mode") != expected_mode:
        raise ValueError(f"{metrics_path} has mode {run.get('mode')!r}; expected {expected_mode!r}.")

    expected_steps = _require_int(run, "steps", str(metrics_path))
    if len(step_records) != expected_steps:
        raise ValueError(f"{metrics_path} declares {expected_steps} steps but contains {len(step_records)}.")
    ordered_steps = sorted(step_records, key=lambda record: _require_int(record, "global_step", str(metrics_path)))
    global_steps = [_require_int(record, "global_step", str(metrics_path)) for record in ordered_steps]
    if global_steps != list(range(1, expected_steps + 1)):
        raise ValueError(f"{metrics_path} has non-contiguous global steps: {global_steps}")
    for index, record in enumerate(ordered_steps, start=1):
        _require_mode_record(record, record_type="step", mode=expected_mode, context=f"{metrics_path} step {index}")
        if record.get("backend") != run.get("backend"):
            raise ValueError(f"{metrics_path} step {index} backend does not match its run record.")

    responses = _load_jsonl(responses_path)
    for index, record in enumerate(responses, start=1):
        _require_mode_record(
            record,
            record_type="rollout",
            mode=expected_mode,
            context=f"{responses_path} record {index}",
        )
    expected_responses = (
        expected_steps
        * _require_int(run, "train_batch_size", str(metrics_path))
        * _require_int(run, "rollouts_per_sample", str(metrics_path))
    )
    if len(responses) != expected_responses:
        raise ValueError(
            f"{responses_path} contains {len(responses)} rollouts; expected {expected_responses} "
            "from steps × train_batch_size × rollouts_per_sample."
        )
    return RunArtifacts(resolved, metrics_path, responses_path, run, ordered_steps, responses)


def validate_comparable(baseline: RunArtifacts, prefix: RunArtifacts) -> dict[str, Any]:
    """Reject a comparison unless every controlled run field matches."""
    mismatches: dict[str, dict[str, Any]] = {}
    invariants: dict[str, Any] = {}
    for field in COMPARABLE_RUN_FIELDS:
        baseline_value = baseline.run.get(field)
        prefix_value = prefix.run.get(field)
        if baseline_value != prefix_value:
            mismatches[field] = {"baseline": baseline_value, "prefix_grouper": prefix_value}
        else:
            invariants[field] = baseline_value
    if mismatches:
        raise ValueError(
            "Benchmark runs are not comparable: " + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    baseline_steps = [_require_int(record, "global_step", str(baseline.metrics_path)) for record in baseline.steps]
    prefix_steps = [_require_int(record, "global_step", str(prefix.metrics_path)) for record in prefix.steps]
    if baseline_steps != prefix_steps:
        raise ValueError("Baseline and PrefixGrouper global-step sets differ.")

    baseline_keys = _response_keys(baseline.responses, baseline.responses_path)
    prefix_keys = _response_keys(prefix.responses, prefix.responses_path)
    if baseline_keys != prefix_keys:
        missing_from_prefix = sorted(baseline_keys - prefix_keys)
        missing_from_baseline = sorted(prefix_keys - baseline_keys)
        raise ValueError(
            "Rollout identity sets differ: "
            + json.dumps(
                {
                    "missing_from_prefix_grouper": missing_from_prefix[:10],
                    "missing_from_baseline": missing_from_baseline[:10],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return invariants


def _response_keys(records: list[Record], path: Path) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    for index, record in enumerate(records, start=1):
        sample_id = record.get("sample_id")
        rollout_index = record.get("rollout_index")
        request_seed = record.get("request_seed")
        if not isinstance(sample_id, str) or not isinstance(rollout_index, int) or not isinstance(request_seed, int):
            raise ValueError(f"{path} record {index} has an invalid rollout identity.")
        key = (sample_id, rollout_index, request_seed)
        if key in keys:
            raise ValueError(f"{path} contains duplicate rollout identity {key!r}.")
        keys.add(key)
    return keys


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _metric_names(records: list[Record]) -> set[str]:
    if not records:
        return set()
    names = {
        key
        for key, value in records[0].items()
        if key not in {"global_step", "schema_version"} and _finite_number(value) is not None
    }
    for record in records[1:]:
        names &= {
            key
            for key, value in record.items()
            if key not in {"global_step", "schema_version"} and _finite_number(value) is not None
        }
    return names


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float | int]:
    """Return fixed descriptive statistics for one metric series."""
    if not values:
        raise ValueError("Cannot summarize an empty metric series.")
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


def _summarize_steps(records: list[Record], metric_names: list[str]) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for metric in metric_names:
        values = [_finite_number(record.get(metric)) for record in records]
        if any(value is None for value in values):
            raise ValueError(f"Metric {metric!r} is missing or non-finite in a step record.")
        summaries[metric] = summarize(cast(list[float], values))
    return summaries


def _direction(metric: str) -> Direction:
    if metric.startswith("timing_") or metric.startswith("actor/perf/"):
        return "lower"
    if metric == "perf/throughput" or metric.startswith("perf/mfu/"):
        return "higher"
    return "neutral"


def compare_summaries(
    baseline: dict[str, dict[str, float | int]],
    prefix: dict[str, dict[str, float | int]],
) -> dict[str, Record]:
    """Compare mean values while preserving both raw descriptive summaries."""
    comparisons: dict[str, Record] = {}
    for metric in sorted(baseline.keys() & prefix.keys()):
        baseline_mean = float(baseline[metric]["mean"])
        prefix_mean = float(prefix[metric]["mean"])
        direction = _direction(metric)
        ratio = prefix_mean / baseline_mean if baseline_mean != 0.0 else None
        speedup: float | None = None
        improvement: float | None = None
        if direction == "lower" and prefix_mean != 0.0:
            speedup = baseline_mean / prefix_mean
            improvement = (baseline_mean - prefix_mean) / baseline_mean * 100.0 if baseline_mean else None
        elif direction == "higher" and baseline_mean != 0.0:
            speedup = prefix_mean / baseline_mean
            improvement = (prefix_mean - baseline_mean) / baseline_mean * 100.0
        comparisons[metric] = {
            "direction": direction,
            "baseline": baseline[metric],
            "prefix_grouper": prefix[metric],
            "prefix_minus_baseline": prefix_mean - baseline_mean,
            "prefix_over_baseline": ratio,
            "speedup": speedup,
            "improvement_percent": improvement,
        }
    return comparisons


def _response_summary(records: list[Record], path: Path) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for metric in ("reward", "exact_reward"):
        values = [_finite_number(record.get(metric)) for record in records]
        if any(value is None for value in values):
            raise ValueError(f"{path} contains a missing or non-finite {metric!r}.")
        result[metric] = summarize(cast(list[float], values))
    return result


def build_report(baseline: RunArtifacts, prefix: RunArtifacts) -> Record:
    """Build a comparison report after strict equivalence checks."""
    invariants = validate_comparable(baseline, prefix)
    baseline_metric_names = _metric_names(baseline.steps)
    prefix_metric_names = _metric_names(prefix.steps)
    common_metric_names = sorted(baseline_metric_names & prefix_metric_names)
    required = set(REQUIRED_STAGE_METRICS + PER_TOKEN_METRICS + PERFORMANCE_METRICS + QUALITY_METRICS)
    missing = sorted(required - set(common_metric_names))
    if missing:
        raise ValueError(f"Required report metrics are missing: {missing}")

    windows: dict[str, Record] = {}
    for name, baseline_steps, prefix_steps in (
        ("all_steps", baseline.steps, prefix.steps),
        ("steady_state", baseline.steps[1:], prefix.steps[1:]),
    ):
        if not baseline_steps or not prefix_steps:
            raise ValueError("At least two training steps are required to compute a steady-state report.")
        baseline_summary = _summarize_steps(baseline_steps, common_metric_names)
        prefix_summary = _summarize_steps(prefix_steps, common_metric_names)
        windows[name] = {
            "step_numbers": [
                _require_int(record, "global_step", str(baseline.metrics_path)) for record in baseline_steps
            ],
            "metrics": compare_summaries(baseline_summary, prefix_summary),
        }

    baseline_responses = _response_summary(baseline.responses, baseline.responses_path)
    prefix_responses = _response_summary(prefix.responses, prefix.responses_path)
    wall_baseline = _require_float(baseline.run, "wall_seconds", str(baseline.metrics_path))
    wall_prefix = _require_float(prefix.run, "wall_seconds", str(prefix.metrics_path))
    return {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison_valid": True,
        "sources": {
            "baseline_metrics": str(baseline.metrics_path),
            "baseline_responses": str(baseline.responses_path),
            "prefix_grouper_metrics": str(prefix.metrics_path),
            "prefix_grouper_responses": str(prefix.responses_path),
        },
        "invariants": invariants,
        "run_wall_seconds": {
            "baseline": wall_baseline,
            "prefix_grouper": wall_prefix,
            "prefix_minus_baseline": wall_prefix - wall_baseline,
            "speedup": wall_baseline / wall_prefix if wall_prefix else None,
        },
        "windows": windows,
        "rollout_quality": {
            "response_count": len(baseline.responses),
            "metrics": compare_summaries(baseline_responses, prefix_responses),
        },
        "metric_coverage": {
            "common": common_metric_names,
            "baseline_only": sorted(baseline_metric_names - prefix_metric_names),
            "prefix_grouper_only": sorted(prefix_metric_names - baseline_metric_names),
        },
        "notes": [
            "steady_state excludes global step 1 as warmup and includes every remaining step",
            "speedup uses baseline/prefix for lower-is-better metrics and prefix/baseline for higher-is-better metrics",
            "the report preserves raw measurements and does not by itself prove hardware correctness or causality",
        ],
    }


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _metric_table(report: Record, window: str, metrics: tuple[str, ...]) -> list[str]:
    window_record = cast(Record, cast(Record, report["windows"])[window])
    comparisons = cast(dict[str, Record], window_record["metrics"])
    lines = [
        "| 指标 | B mean | B p50 | B p95 | PG mean | PG p50 | PG p95 | Mean 差值 | 改善 | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        item = comparisons[metric]
        baseline_stats = cast(Record, item["baseline"])
        prefix_stats = cast(Record, item["prefix_grouper"])
        improvement = item["improvement_percent"]
        improvement_text = "N/A" if improvement is None else f"{float(improvement):+.3f}%"
        speedup = item["speedup"]
        speedup_text = "N/A" if speedup is None else f"{float(speedup):.4f}×"
        lines.append(
            "| "
            + " | ".join(
                (
                    DISPLAY_NAMES.get(metric, metric),
                    _format_number(baseline_stats["mean"]),
                    _format_number(baseline_stats["median"]),
                    _format_number(baseline_stats["p95"]),
                    _format_number(prefix_stats["mean"]),
                    _format_number(prefix_stats["median"]),
                    _format_number(prefix_stats["p95"]),
                    _format_number(item["prefix_minus_baseline"]),
                    improvement_text,
                    speedup_text,
                )
            )
            + " |"
        )
    return lines


def render_markdown(report: Record) -> str:
    """Render the canonical human-readable report."""
    invariants = cast(Record, report["invariants"])
    wall = cast(Record, report["run_wall_seconds"])
    rollout_quality = cast(Record, report["rollout_quality"])
    quality_metrics = cast(dict[str, Record], rollout_quality["metrics"])
    lines = [
        "# PrefixGrouper 2WikiMQA 训练端到端基准报告",
        "",
        f"- Benchmark ID：`{BENCHMARK_ID}`",
        f"- 后端：`{invariants['backend']}`；设备：`{invariants['device_name']}`",
        f"- 模型：`{invariants['model_name']}`（来源：`{invariants['model_ref']}`；本地：`{invariants['model_path']}`）",
        f"- 数据：`{invariants['dataset']}`，{invariants['dataset_rows']} rows",
        f"- 输入：原始 prompt 过滤 {invariants['min_prompt_tokens']}–{invariants['max_prompt_tokens']} tokens；"
        f"输出最多 {invariants['max_response_tokens']} tokens",
        f"- Steps：{invariants['steps']}；steady-state 排除 step 1",
        f"- 比较条件校验：通过（{len(COMPARABLE_RUN_FIELDS)} 个受控字段一致）",
        "",
        "## 整次运行",
        "",
        "| Baseline wall (s) | PrefixGrouper wall (s) | 差值 (s) | Speedup |",
        "|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            (
                _format_number(wall["baseline"]),
                _format_number(wall["prefix_grouper"]),
                _format_number(wall["prefix_minus_baseline"]),
                f"{float(wall['speedup']):.4f}×" if wall["speedup"] is not None else "N/A",
            )
        )
        + " |",
        "",
        "## 阶段时间：全部 steps",
        "",
        *_metric_table(report, "all_steps", REQUIRED_STAGE_METRICS),
        "",
        "## 阶段时间：steady-state",
        "",
        *_metric_table(report, "steady_state", REQUIRED_STAGE_METRICS),
        "",
        "## 每 token 时间：steady-state",
        "",
        *_metric_table(report, "steady_state", PER_TOKEN_METRICS),
        "",
        "## 吞吐与资源：steady-state",
        "",
        *_metric_table(report, "steady_state", PERFORMANCE_METRICS),
        "",
        "## 质量指标",
        "",
        f"共比较 {rollout_quality['response_count']} 个具有相同 sample、rollout index 和 seed 的响应。",
        "",
        "| 指标 | Baseline mean | PrefixGrouper mean | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for metric, display_name in (("reward", "2WikiMQA F1"), ("exact_reward", "Exact match")):
        item = quality_metrics[metric]
        lines.append(
            f"| {display_name} | {_format_number(cast(Record, item['baseline'])['mean'])} | "
            f"{_format_number(cast(Record, item['prefix_grouper'])['mean'])} | "
            f"{_format_number(item['prefix_minus_baseline'])} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- steady-state 固定排除首个训练 step；JSON 报告仍保留全部 step 的统计。",
            "- `gen` 包含权重同步、rollout 和 trace 收集等合并工作，并非纯解码时间。",
            "- 报告只对受控条件一致的原始结果进行算术比较，不单独构成硬件正确性或因果证据。",
            "",
            "## 原始数据",
            "",
        ]
    )
    for label, path in cast(Record, report["sources"]).items():
        lines.append(f"- `{label}`：`{path}`")
    return "\n".join(lines) + "\n"


def prepare_outputs(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    """Resolve report outputs without silently replacing prior evidence."""
    resolved = output_dir.resolve()
    json_path = resolved / "report.json"
    markdown_path = resolved / "report.md"
    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite reports: " + ", ".join(str(path) for path in existing))
    if overwrite:
        for path in existing:
            path.unlink()
    resolved.mkdir(parents=True, exist_ok=True)
    return json_path, markdown_path


def main() -> None:
    """Validate raw artifacts and write canonical comparison reports."""
    args = parse_args()
    baseline = load_artifacts(args.baseline_dir, "baseline")
    prefix = load_artifacts(args.prefix_grouper_dir, "prefix_grouper")
    report = build_report(baseline, prefix)
    json_path, markdown_path = prepare_outputs(args.output_dir, args.overwrite)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"report_json": str(json_path), "report_markdown": str(markdown_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
