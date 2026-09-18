#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Analyze prompt/response token ratios in multi-turn trajectory JSONL.

The input can be either a JSONL file or a run directory containing
``calls.jsonl``. Each record must contain a trajectory/rollout ID and token IDs
in one of these forms:

* ``prompt_token_ids`` and ``response_token_ids``;
* ``prompt_ids`` and ``response_ids``;
* ``prompt.token_ids`` and ``response.token_ids``.

Example:
    python scripts/analyze_multiturn_token_ratio.py --input /runs/sql-prefix-npu/calls.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL file or a directory containing calls.jsonl.")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report; stdout is always printed.")
    parser.add_argument(
        "--include-trajectories",
        action="store_true",
        help="Include every trajectory's metrics in the report instead of summary statistics only.",
    )
    parser.add_argument(
        "--role",
        default="policy",
        help="Analyze this role when records have a role field; use 'all' to keep every role (default: policy).",
    )
    parser.add_argument(
        "--group-key",
        choices=("auto", "data_id", "task_id"),
        default="auto",
        help="Field that identifies rollout groups for prompt sharing; auto prefers data_id, then task_id.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield non-empty JSONL records with their one-based line numbers."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, value


def token_ids(record: dict[str, Any], side: str, context: str) -> list[int]:
    """Read and validate prompt or response token IDs from supported schemas."""
    value = record.get(f"{side}_token_ids")
    if value is None:
        value = record.get(f"{side}_ids")
    if value is None and isinstance(record.get(side), dict):
        value = record[side].get("token_ids")
    if not isinstance(value, list) or any(type(token) is not int or token < 0 for token in value):
        raise ValueError(f"{context} has invalid or missing {side} token IDs")
    return value


def trajectory_id(record: dict[str, Any], context: str) -> str:
    """Resolve the stable trajectory identifier used for grouping turns."""
    value = record.get("trajectory_id", record.get("rollout_id"))
    if not isinstance(value, (str, int)) or str(value) == "":
        raise ValueError(f"{context} has no trajectory_id or rollout_id")
    return str(value)


def sharing_group_id(record: dict[str, Any], group_key: str, context: str) -> tuple[str | None, str | None]:
    """Resolve the task/data group used to keep sibling rollouts together."""
    keys = ("data_id", "task_id") if group_key == "auto" else (group_key,)
    for key in keys:
        value = record.get(key)
        if isinstance(value, (str, int)) and str(value) != "":
            return key, str(value)
    if group_key != "auto":
        raise ValueError(f"{context} has no valid {group_key}")
    return None, None


def turn_index(record: dict[str, Any], fallback: int, context: str) -> int:
    """Resolve a turn index while allowing already ordered exports."""
    value = record.get("turn", record.get("turn_index", fallback))
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} has an invalid turn index: {value!r}")
    return value


def ratio(numerator: int, denominator: int) -> float | None:
    """Return a finite ratio, or None for a zero denominator."""
    return numerator / denominator if denominator else None


def share(part: int, total: int) -> float | None:
    """Return a part-of-total share, or None for an empty total."""
    return part / total if total else None


def percentile(values: list[float], quantile: float) -> float | None:
    """Compute a linearly interpolated percentile without third-party packages."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Iterable[float | None]) -> dict[str, float | int | None]:
    """Summarize per-trajectory ratios without weighting short and long trajectories together."""
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return {
        "count": len(finite),
        "min": min(finite) if finite else None,
        "p50": percentile(finite, 0.50),
        "mean": mean(finite) if finite else None,
        "p95": percentile(finite, 0.95),
        "max": max(finite) if finite else None,
    }


def starts_with(sequence: list[int], prefix: list[int]) -> bool:
    """Check exact token-prefix preservation between adjacent turns."""
    return len(sequence) >= len(prefix) and sequence[: len(prefix)] == prefix


def summarize_counts(prompt: int, response: int, response_name: str) -> dict[str, int | float | None]:
    """Build ratio and share metrics for one pair of token counts."""
    total = prompt + response
    return {
        "prompt_tokens": prompt,
        response_name: response,
        "total_tokens": total,
        "prompt_to_response_ratio": ratio(prompt, response),
        "prompt_share": share(prompt, total),
        "response_share": share(response, total),
    }


def analyze_trajectory(trajectory: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze API-call totals and the trajectory-level training representation."""
    ordered = sorted(turns, key=lambda row: row["turn"])
    indices = [row["turn"] for row in ordered]
    if len(indices) != len(set(indices)):
        raise ValueError(f"trajectory {trajectory!r} contains duplicate turn indices")
    group_keys = {(row["group_key"], row["group_id"]) for row in ordered}
    if len(group_keys) != 1:
        raise ValueError(f"trajectory {trajectory!r} changes sharing group across turns: {sorted(group_keys)!r}")
    group_key, group_id = next(iter(group_keys))

    api_prompt = sum(len(row["prompt_ids"]) for row in ordered)
    model_response = sum(len(row["response_ids"]) for row in ordered)

    # Match the trajectory aggregator's untruncated representation. The first
    # prompt is the training prompt. Later prompt deltas (tool/environment
    # results included) and model outputs form the cumulative response suffix.
    segments: list[dict[str, Any]] = []
    segment_prompt = ordered[0]["prompt_ids"]
    segment_start_turn = ordered[0]["turn"]
    previous_context: list[int] = []
    segment_policy_response = 0
    prefix_breaks = 0

    for row in ordered:
        full_context = row["prompt_ids"] + row["response_ids"]
        if previous_context and not starts_with(full_context, previous_context):
            segments.append(
                {
                    "start_turn": segment_start_turn,
                    "prompt_tokens": len(segment_prompt),
                    "response_suffix_tokens": len(previous_context) - len(segment_prompt),
                    "model_response_tokens": segment_policy_response,
                    "prompt_ids": segment_prompt,
                }
            )
            prefix_breaks += 1
            segment_prompt = row["prompt_ids"]
            segment_start_turn = row["turn"]
            segment_policy_response = 0
        previous_context = full_context
        segment_policy_response += len(row["response_ids"])

    segments.append(
        {
            "start_turn": segment_start_turn,
            "prompt_tokens": len(segment_prompt),
            "response_suffix_tokens": len(previous_context) - len(segment_prompt),
            "model_response_tokens": segment_policy_response,
            "prompt_ids": segment_prompt,
        }
    )

    training_prompt = sum(segment["prompt_tokens"] for segment in segments)
    response_suffix = sum(segment["response_suffix_tokens"] for segment in segments)
    environment_tokens = response_suffix - model_response
    if environment_tokens < 0:
        raise ValueError(f"trajectory {trajectory!r} produced a negative environment-token count")

    return {
        "trajectory_id": trajectory,
        "group_key": group_key,
        "group_id": group_id,
        "turns": len(ordered),
        "prefix_breaks": prefix_breaks,
        "segments": len(segments),
        "api_call": summarize_counts(api_prompt, model_response, "model_response_tokens"),
        "training_trajectory": {
            **summarize_counts(training_prompt, response_suffix, "response_suffix_tokens"),
            "model_response_tokens": model_response,
            "environment_tokens": environment_tokens,
            "prompt_to_model_response_ratio": ratio(training_prompt, model_response),
            "policy_loss_token_share_of_suffix": share(model_response, response_suffix),
        },
        "_sharing_segments": segments,
    }


def analyze_sharing(trajectories: list[dict[str, Any]], requested_group_key: str) -> dict[str, Any]:
    """Measure exact-prompt sharing potential using the production grouping rule."""
    resolved_keys = {row["group_key"] for row in trajectories}
    if resolved_keys == {None}:
        return {
            "available": False,
            "reason": "Records contain neither data_id nor task_id; sharing groups cannot be reconstructed.",
        }
    if None in resolved_keys or len(resolved_keys) != 1:
        raise ValueError(f"Inconsistent sharing group keys across trajectories: {sorted(map(str, resolved_keys))}")
    resolved_group_key = next(iter(resolved_keys))

    exact_prompt_groups: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
    prompt_tokens = suffix_tokens = 0
    segment_count = 0
    for trajectory in trajectories:
        group_id = trajectory["group_id"]
        for segment in trajectory["_sharing_segments"]:
            prompt = segment["prompt_ids"]
            exact_prompt_groups[(group_id, tuple(prompt))].append(segment)
            prompt_tokens += len(prompt)
            suffix_tokens += segment["response_suffix_tokens"]
            segment_count += 1

    shared = [segments for segments in exact_prompt_groups.values() if len(segments) >= 2]
    shareable_occurrences = sum(len(segments[0]["prompt_ids"]) * len(segments) for segments in shared)
    shared_once = sum(len(segments[0]["prompt_ids"]) for segments in shared)
    reducible_duplicates = shareable_occurrences - shared_once
    independent_tokens = prompt_tokens + suffix_tokens
    grouped_tokens = independent_tokens - reducible_duplicates
    return {
        "available": True,
        "requested_group_key": requested_group_key,
        "resolved_group_key": resolved_group_key,
        "training_segments": segment_count,
        "shared_groups": len(shared),
        "shared_segments": sum(len(segments) for segments in shared),
        "independent_prompt_tokens": prompt_tokens,
        "response_suffix_tokens": suffix_tokens,
        "independent_total_tokens": independent_tokens,
        "shareable_prompt_token_occurrences": shareable_occurrences,
        "shared_prompt_tokens_computed_once": shared_once,
        "reducible_duplicate_prompt_tokens": reducible_duplicates,
        "prompt_shareable_ratio": share(shareable_occurrences, prompt_tokens),
        "prompt_deduplication_ratio": share(reducible_duplicates, prompt_tokens),
        "shareable_ratio": share(reducible_duplicates, independent_tokens),
        "grouped_total_tokens": grouped_tokens,
        "token_work_ratio": ratio(independent_tokens, grouped_tokens),
        "assumption": (
            "All exact-prompt segments from the same group are present in one PrefixGrouper micro-batch; "
            "realized sharing cannot cross micro-batch or data-parallel-rank boundaries."
        ),
    }


def analyze(path: Path, role: str, group_key: str, include_trajectories: bool) -> dict[str, Any]:
    """Load calls, group turns, and aggregate weighted and per-trajectory statistics."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_records = 0
    skipped_roles: dict[str, int] = defaultdict(int)
    for fallback, (line_number, record) in enumerate(read_jsonl(path)):
        record_role = record.get("role")
        if role != "all" and record_role is not None and record_role != role:
            skipped_roles[str(record_role)] += 1
            continue
        context = f"{path}:{line_number}"
        identifier = trajectory_id(record, context)
        resolved_group_key, group_id = sharing_group_id(record, group_key, context)
        groups[identifier].append(
            {
                "turn": turn_index(record, fallback, context),
                "prompt_ids": token_ids(record, "prompt", context),
                "response_ids": token_ids(record, "response", context),
                "group_key": resolved_group_key,
                "group_id": group_id,
            }
        )
        selected_records += 1

    if not groups:
        raise ValueError(f"No analyzable records found in {path}")

    trajectories = [analyze_trajectory(identifier, turns) for identifier, turns in sorted(groups.items())]
    api_prompt = sum(row["api_call"]["prompt_tokens"] for row in trajectories)
    model_response = sum(row["api_call"]["model_response_tokens"] for row in trajectories)
    training_prompt = sum(row["training_trajectory"]["prompt_tokens"] for row in trajectories)
    response_suffix = sum(row["training_trajectory"]["response_suffix_tokens"] for row in trajectories)
    environment_tokens = sum(row["training_trajectory"]["environment_tokens"] for row in trajectories)

    sharing = analyze_sharing(trajectories, group_key)
    report = {
        "input": str(path.resolve()),
        "role": role,
        "selected_records": selected_records,
        "skipped_records_by_role": dict(sorted(skipped_roles.items())),
        "trajectories": len(trajectories),
        "turns": sum(row["turns"] for row in trajectories),
        "prefix_breaks": sum(row["prefix_breaks"] for row in trajectories),
        "overall_weighted": {
            "api_call": summarize_counts(api_prompt, model_response, "model_response_tokens"),
            "training_trajectory": {
                **summarize_counts(training_prompt, response_suffix, "response_suffix_tokens"),
                "model_response_tokens": model_response,
                "environment_tokens": environment_tokens,
                "prompt_to_model_response_ratio": ratio(training_prompt, model_response),
                "policy_loss_token_share_of_suffix": share(model_response, response_suffix),
            },
        },
        "prefix_sharing": sharing,
        "per_trajectory_distribution": {
            "api_prompt_to_model_response_ratio": distribution(
                row["api_call"]["prompt_to_response_ratio"] for row in trajectories
            ),
            "training_prompt_to_response_suffix_ratio": distribution(
                row["training_trajectory"]["prompt_to_response_ratio"] for row in trajectories
            ),
            "training_prompt_to_model_response_ratio": distribution(
                row["training_trajectory"]["prompt_to_model_response_ratio"] for row in trajectories
            ),
        },
        "definitions": {
            "api_call": "Sum every turn's full prompt and model response; repeated history is counted repeatedly.",
            "training_prompt": "The first prompt of each exact-prefix segment, matching trajectory aggregation before truncation.",
            "response_suffix": "Everything after the training prompt: model outputs plus later tool/environment prompt deltas.",
            "model_response": "Only tokens generated by the policy model; these are the policy-loss tokens.",
            "ratio": "prompt_tokens / response_tokens; shares divide by prompt_tokens + response_tokens.",
            "prefix_break": "Adjacent turns did not preserve the prior full token context and were analyzed as separate segments.",
            "prompt_shareable_ratio": (
                "Prompt-token occurrences belonging to exact-prompt groups of size >= 2, divided by all prompt tokens."
            ),
            "prompt_deduplication_ratio": (
                "Duplicate prompt tokens removable by computing each shared prompt once, divided by all prompt tokens."
            ),
            "shareable_ratio": (
                "Duplicate prompt tokens removable by exact-prefix sharing, divided by independent prompt + suffix tokens."
            ),
        },
    }
    if include_trajectories:
        report["per_trajectory"] = [
            {key: value for key, value in trajectory.items() if not key.startswith("_")}
            for trajectory in trajectories
        ]
    return report


def main() -> None:
    """Run the analyzer and emit a reproducible JSON report."""
    args = parse_args()
    input_path = args.input / "calls.jsonl" if args.input.is_dir() else args.input
    report = analyze(input_path, args.role, args.group_key, args.include_trajectories)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
