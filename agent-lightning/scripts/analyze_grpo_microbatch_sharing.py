#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Analyze per-task PrefixGrouper sharing under GRPO micro-batch sizes.

The input is one or more workload-labelled ``calls.jsonl`` exports.  The script
reconstructs the same untruncated training segments as
``analyze_multiturn_token_ratio.py``, groups exact prompts within each task, and
reports the duplicate prompt-token work removable at each micro-batch size.
It also exports each rollout's initial-prompt length and final logical
trajectory length, plus task-level length summaries.

Example:
    python scripts/analyze_grpo_microbatch_sharing.py \
      --input q20=/runs/q20/calls.jsonl \
      --input sql=/runs/sql/calls.jsonl \
      --input web=/runs/web/calls.jsonl \
      --micro-batch-sizes 1,2,4,8,16,32,64 \
      --output-dir /runs/grpo-sharing-analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from analyze_multiturn_token_ratio import (
    analyze_trajectory,
    read_jsonl,
    sharing_group_id,
    token_ids,
    trajectory_id,
    turn_index,
)


def parse_input(value: str) -> tuple[str, Path]:
    """Parse ``WORKLOAD=PATH`` command-line values."""
    workload, separator, raw_path = value.partition("=")
    if not separator or not workload.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--input must use WORKLOAD=PATH")
    path = Path(raw_path).expanduser()
    if path.is_dir():
        path = path / "calls.jsonl"
    return workload.strip(), path


def parse_micro_batch_sizes(value: str) -> list[int]:
    """Parse a comma-separated, strictly increasing set of positive sizes."""
    try:
        sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("micro-batch sizes must be integers") from error
    if not sizes or sizes[0] <= 0:
        raise argparse.ArgumentTypeError("micro-batch sizes must be positive")
    return sizes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=parse_input,
        required=True,
        metavar="WORKLOAD=PATH",
        help="Labelled calls.jsonl input; repeat once per workload.",
    )
    parser.add_argument(
        "--micro-batch-sizes",
        type=parse_micro_batch_sizes,
        default=parse_micro_batch_sizes("1,2,4,8,16,32,64"),
        help="Comma-separated per-device GRPO micro-batch sizes (default: 1,2,4,8,16,32,64).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--role",
        default="policy",
        help="Analyze this role when records have a role field (default: policy).",
    )
    parser.add_argument(
        "--group-key",
        choices=("auto", "data_id", "task_id"),
        default="auto",
        help="Field identifying the GRPO task group (default: auto).",
    )
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    """Summarize an unweighted task-level distribution."""
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "task_count": len(finite),
        "mean": statistics.fmean(finite),
        "stddev": statistics.pstdev(finite),
        "min": min(finite),
        "p25": percentile(finite, 0.25),
        "p50": percentile(finite, 0.50),
        "p75": percentile(finite, 0.75),
        "p95": percentile(finite, 0.95),
        "max": max(finite),
        "zero_share_tasks": sum(value == 0 for value in finite),
    }


def load_trajectories(path: Path, role: str, group_key: str) -> list[dict[str, Any]]:
    """Load policy calls and reconstruct trajectory training segments."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fallback, (line_number, record) in enumerate(read_jsonl(path)):
        record_role = record.get("role")
        if role != "all" and record_role is not None and record_role != role:
            continue
        context = f"{path}:{line_number}"
        identifier = trajectory_id(record, context)
        resolved_group_key, group_id = sharing_group_id(record, group_key, context)
        if group_id is None:
            raise ValueError(f"{context} has no task/data group identifier")
        groups[identifier].append(
            {
                "turn": turn_index(record, fallback, context),
                "prompt_ids": token_ids(record, "prompt", context),
                "response_ids": token_ids(record, "response", context),
                "group_key": resolved_group_key,
                "group_id": group_id,
            }
        )
    if not groups:
        raise ValueError(f"No analyzable records found in {path}")
    return [analyze_trajectory(identifier, turns) for identifier, turns in sorted(groups.items())]


def trajectory_length_rows(workload: str, trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Export exact initial and final logical lengths for every trajectory."""
    return [
        {
            "workload": workload,
            "task_id": str(trajectory["group_id"]),
            "trajectory_id": trajectory["trajectory_id"],
            "turns": trajectory["turns"],
            "prefix_breaks": trajectory["prefix_breaks"],
            "training_segments": trajectory["segments"],
            "initial_prompt_length": len(trajectory["_initial_prompt_ids"]),
            "final_trajectory_length": int(trajectory["_logical_total_tokens"]),
        }
        for trajectory in trajectories
    ]


def task_rows(workload: str, trajectories: list[dict[str, Any]], sizes: list[int]) -> list[dict[str, Any]]:
    """Compute micro-batch-constrained sharing metrics for every task."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        by_task[str(trajectory["group_id"])].append(trajectory)

    rows: list[dict[str, Any]] = []
    for task_id, task_trajectories in sorted(by_task.items()):
        initial_prompt_lengths = [len(trajectory["_initial_prompt_ids"]) for trajectory in task_trajectories]
        final_trajectory_lengths = [int(trajectory["_logical_total_tokens"]) for trajectory in task_trajectories]
        exact_prompts: dict[tuple[int, ...], int] = defaultdict(int)
        prompt_tokens = 0
        suffix_tokens = 0
        segment_count = 0
        for trajectory in task_trajectories:
            for segment in trajectory["_sharing_segments"]:
                prompt = tuple(segment["prompt_ids"])
                exact_prompts[prompt] += 1
                prompt_tokens += len(prompt)
                suffix_tokens += int(segment["response_suffix_tokens"])
                segment_count += 1

        independent_tokens = prompt_tokens + suffix_tokens
        if independent_tokens <= 0:
            raise ValueError(f"{workload}/{task_id} has no training tokens")
        repeated_groups = sum(count >= 2 for count in exact_prompts.values())
        repeated_segments = sum(count for count in exact_prompts.values() if count >= 2)
        for size in sizes:
            # Exact-prompt rows are contiguous after production's stable
            # reorder.  A group of n rows occupies ceil(n / size)
            # micro-batches at best, so one prompt is computed per occupied
            # micro-batch and all remaining occurrences are removable.
            saved_tokens = sum(
                (count - math.ceil(count / size)) * len(prompt) for prompt, count in exact_prompts.items() if count >= 2
            )
            grouped_tokens = independent_tokens - saved_tokens
            rows.append(
                {
                    "workload": workload,
                    "task_id": task_id,
                    "micro_batch_size_per_device": size,
                    "trajectories": len(task_trajectories),
                    "initial_prompt_length_min": min(initial_prompt_lengths),
                    "initial_prompt_length_mean": statistics.fmean(initial_prompt_lengths),
                    "initial_prompt_length_max": max(initial_prompt_lengths),
                    "final_trajectory_length_min": min(final_trajectory_lengths),
                    "final_trajectory_length_mean": statistics.fmean(final_trajectory_lengths),
                    "final_trajectory_length_p50": percentile(
                        [float(value) for value in final_trajectory_lengths], 0.50
                    ),
                    "final_trajectory_length_p95": percentile(
                        [float(value) for value in final_trajectory_lengths], 0.95
                    ),
                    "final_trajectory_length_max": max(final_trajectory_lengths),
                    "training_segments": segment_count,
                    "exact_prompt_groups": len(exact_prompts),
                    "repeated_prompt_groups": repeated_groups,
                    "repeated_prompt_segments": repeated_segments,
                    "independent_prompt_tokens": prompt_tokens,
                    "response_suffix_tokens": suffix_tokens,
                    "independent_total_tokens": independent_tokens,
                    "reducible_duplicate_prompt_tokens": saved_tokens,
                    "grouped_total_tokens": grouped_tokens,
                    "sharing_ratio": saved_tokens / independent_tokens,
                    "prompt_deduplication_ratio": saved_tokens / prompt_tokens if prompt_tokens else 0.0,
                    "token_work_ratio": independent_tokens / grouped_tokens,
                }
            )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build task-distribution statistics for every workload and size."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["workload"], row["micro_batch_size_per_device"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (workload, size), members in sorted(grouped.items()):
        stats = distribution(float(row["sharing_ratio"]) for row in members)
        total_independent = sum(int(row["independent_total_tokens"]) for row in members)
        total_saved = sum(int(row["reducible_duplicate_prompt_tokens"]) for row in members)
        summaries.append(
            {
                "workload": workload,
                "micro_batch_size_per_device": size,
                **stats,
                "weighted_sharing_ratio": total_saved / total_independent,
                "total_independent_tokens": total_independent,
                "total_saved_tokens": total_saved,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as a stable UTF-8 CSV."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value: float) -> str:
    """Format a ratio as a percentage."""
    return f"{value * 100:.2f}%"


def render_report(inputs: dict[str, str], rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> str:
    """Render the concise human-readable analysis report."""
    lines = [
        "# GRPO 不同 micro-batch 下的 task 共享比例",
        "",
        "## 口径",
        "",
        "- 共享比例 = 同一 task 内、同一 micro-batch 中可消除的重复 prompt token / 独立执行的训练总 token。",
        "- 训练总 token = 每个 exact-prefix segment 的 prompt token + response suffix token。",
        "- 初始 prompt 长度 = trajectory 第一次 policy 调用的 prompt token 数。",
        "- 最终轨迹长度 = trajectory 最后一次 policy 调用的完整 prompt 加 response token 数。",
        "- 同一 task 的完全相同 prompt 会按生产逻辑连续排列；一个 prompt group 跨越几个 micro-batch，就需计算几次。",
        "- 结果是 task 内连续装箱、且 task 从 micro-batch 边界开始时的可实现上界；不含跨 task 共享，也不含 DP rank 边界损失。",
        "",
        "## 数据覆盖",
        "",
        "| 数据集 | 原始 calls.jsonl | task 数 | trajectory 数 |",
        "|---|---|---:|---:|",
    ]
    workloads = sorted(inputs)
    for workload in workloads:
        selected = [row for row in rows if row["workload"] == workload]
        first_size = min(int(row["micro_batch_size_per_device"]) for row in selected)
        base = [row for row in selected if row["micro_batch_size_per_device"] == first_size]
        lines.append(
            f"| {workload} | `{inputs[workload]}` | {len(base)} | {sum(int(row['trajectories']) for row in base)} |"
        )

    lines.extend(
        [
            "",
            "## 每个 task 的结果",
            "",
            "| 数据集 | task | micro-batch | 初始 prompt 均值 | 最终轨迹均值 | 最终轨迹 P95 | 独立 token | 可省 token | 共享比例 | token work ratio |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {workload} | `{task_id}` | {micro_batch_size_per_device} | "
            "{initial_mean:.2f} | {final_mean:.2f} | {final_p95:.2f} | "
            "{independent_total_tokens} | {reducible_duplicate_prompt_tokens} | {sharing} | {work_ratio:.4f}× |".format(
                **row,
                initial_mean=float(row["initial_prompt_length_mean"]),
                final_mean=float(row["final_trajectory_length_mean"]),
                final_p95=float(row["final_trajectory_length_p95"]),
                sharing=percent(float(row["sharing_ratio"])),
                work_ratio=float(row["token_work_ratio"]),
            )
        )

    lines.extend(
        [
            "",
            "## task 共享比例分布",
            "",
            "| 数据集 | micro-batch | n | 均值 | P50 | P95 | 最大值 | token 加权值 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        lines.append(
            "| {workload} | {micro_batch_size_per_device} | {task_count} | {mean} | {p50} | {p95} | {max_value} | {weighted} |".format(
                workload=summary["workload"],
                micro_batch_size_per_device=summary["micro_batch_size_per_device"],
                task_count=summary["task_count"],
                mean=percent(float(summary["mean"])),
                p50=percent(float(summary["p50"])),
                p95=percent(float(summary["p95"])),
                max_value=percent(float(summary["max"])),
                weighted=percent(float(summary["weighted_sharing_ratio"])),
            )
        )

    task_counts = {summary["workload"]: int(summary["task_count"]) for summary in summaries}
    insufficient = sorted(workload for workload, count in task_counts.items() if count < 2)
    if insufficient:
        lines.extend(
            [
                "",
                "## 限制",
                "",
                f"以下数据集不足 2 个 task：{', '.join(insufficient)}。其分位数、均值、最小值和最大值会退化为单个观测值。",
                "这些数据集的结果可以比较 micro-batch 对现有 task 的影响，但不能代表完整数据集的 task 分布。",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the analysis and write reproducible JSON, CSV, and Markdown outputs."""
    args = parse_args()
    inputs = dict(args.input)
    if len(inputs) != len(args.input):
        raise ValueError("workload labels passed to --input must be unique")
    rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    resolved_inputs: dict[str, str] = {}
    for workload, path in args.input:
        path = path.resolve()
        resolved_inputs[workload] = str(path)
        trajectories = load_trajectories(path, args.role, args.group_key)
        length_rows.extend(trajectory_length_rows(workload, trajectories))
        rows.extend(task_rows(workload, trajectories, args.micro_batch_sizes))
    summaries = summarize_rows(rows)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_trajectory_lengths.csv", length_rows)
    write_csv(output_dir / "per_task_microbatch_sharing.csv", rows)
    write_csv(output_dir / "task_sharing_distribution.csv", summaries)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": resolved_inputs,
        "role": args.role,
        "group_key": args.group_key,
        "micro_batch_sizes_per_device": args.micro_batch_sizes,
        "metric": "reducible_duplicate_prompt_tokens / independent_total_tokens",
        "assumptions": [
            "Exact prompts within each task are contiguous before micro-batch partitioning.",
            "Each task starts at a micro-batch boundary.",
            "Sharing is limited to one task and cannot cross data-parallel rank boundaries.",
        ],
        "per_trajectory_lengths": length_rows,
        "per_task": rows,
        "task_distribution": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_report(resolved_inputs, rows, summaries), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
