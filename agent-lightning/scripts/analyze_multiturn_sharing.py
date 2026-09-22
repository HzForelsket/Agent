#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

r"""Analyze multi-turn prefix sharing through one offline entrypoint.

The default training view reconstructs untruncated exact-prefix segments and
reports token composition and micro-batch-constrained exact-prompt savings.
The calls and trajectory views compare common-prefix and trie sharing using
explicitly different statistical units. No model or accelerator is required.

Usage:
    python scripts/analyze_multiturn_sharing.py \
        --input sql=/runs/sql/calls.jsonl --input q20=/runs/q20/calls.jsonl \
        --micro-batch-sizes 1,2,4,8,16,32,64 --output-dir /runs/sharing
    python scripts/analyze_multiturn_sharing.py \
        --view calls --input /runs/sql --output-dir /runs/sql/analysis

See MULTITURN_SHARING.md for input contracts and metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

CALL_UNIT = "original_workflow_call_slots"
TRAJECTORY_UNIT = "one_complete_trajectory_sequence"
COSTS = (
    "separate_tokens",
    "separate_causal_pairs",
    "common_prefix_tokens",
    "simple_tokens",
    "simple_causal_pairs",
    "merged_tokens",
    "merged_causal_pairs",
)


def read_records(path: Path, diagnostics: list[str]) -> list[dict[str, Any]]:
    """Read durable JSONL records, reporting an interrupted final write explicitly."""
    if not path.exists():
        diagnostics.append(f"Missing file: {path.name}")
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                diagnostics.append(f"Ignored interrupted final record: {path.name}:{index + 1}")
            else:
                raise
    return records


def lcp(left: list[int], right: list[int]) -> int:
    """Return the length of the exact common token prefix."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def tree_cost(sequences: Iterable[list[int]]) -> tuple[int, int]:
    """Count trie nodes and causal query/key pairs, without constructing dense masks."""
    previous: list[int] = []
    tokens = pairs = 0
    for sequence in sorted(sequences):
        common = lcp(previous, sequence)
        length = len(sequence)
        tokens += length - common
        pairs += (length * (length + 1) - common * (common + 1)) // 2
        previous = sequence
    return tokens, pairs


def valid_call(call: dict[str, Any]) -> bool:
    """Recheck actual token IDs, HTTP status and lengths rather than trusting a flag."""
    prompt, response = call.get("prompt_token_ids"), call.get("response_token_ids")
    usage = call.get("response", {}).get("usage") or {}
    choices = call.get("response", {}).get("choices", [])
    return bool(
        call.get("token_ids_valid")
        and call.get("http_status") == 200
        and isinstance(prompt, list)
        and isinstance(response, list)
        and prompt
        and response
        and all(type(token) is int and token >= 0 for token in prompt + response)
        and len(prompt) == usage.get("prompt_tokens")
        and len(response) == usage.get("completion_tokens")
        and len(choices) == 1
        and choices[0].get("finish_reason") in {"tool_calls", "stop"}
    )


def metrics(separate: int, merged: int, separate_pairs: int, merged_pairs: int) -> dict[str, Any]:
    return {
        "separate_tokens": separate,
        "merged_tokens": merged,
        "saved_tokens": separate - merged,
        "token_reduction": 1 - merged / separate if separate else None,
        "token_work_ratio": separate / merged if merged else None,
        "separate_causal_pairs": separate_pairs,
        "merged_causal_pairs": merged_pairs,
        "causal_pair_reduction": 1 - merged_pairs / separate_pairs if separate_pairs else None,
    }


def complete_sequence(calls: list[dict[str, Any]]) -> list[int]:
    """Return one full-history sequence after checking that prior messages and actions remain present."""
    for previous, following in zip(calls, calls[1:]):
        history = previous["request"]["messages"]
        following_history = following["request"]["messages"]
        if len(following_history) <= len(history) or following_history[: len(history)] != history:
            raise ValueError("message_history_not_preserved")
        action = previous["response"]["choices"][0]["message"]
        recorded_action = following_history[len(history)]
        if any(
            recorded_action.get(key) != action[key]
            for key in ("role", "content", "tool_calls", "reasoning")
            if action.get(key) is not None
        ):
            raise ValueError("assistant_action_not_preserved")
    final = calls[-1]
    return final["prompt_token_ids"] + final["response_token_ids"]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sharing_costs(sequences: list[list[int]]) -> dict[str, int]:
    """Compare independent trajectories, one common prefix, and all trie branches."""
    common = len(sequences[0])
    for sequence in sequences[1:]:
        common = min(common, lcp(sequences[0], sequence))
    separate = sum(len(sequence) for sequence in sequences)
    separate_pairs = sum(len(sequence) * (len(sequence) + 1) // 2 for sequence in sequences)
    tree_tokens, tree_pairs = tree_cost(sequences)
    return {
        "separate_tokens": separate,
        "separate_causal_pairs": separate_pairs,
        "common_prefix_tokens": common,
        "simple_tokens": separate - (len(sequences) - 1) * common,
        "simple_causal_pairs": separate_pairs - (len(sequences) - 1) * common * (common + 1) // 2,
        "merged_tokens": tree_tokens,
        "merged_causal_pairs": tree_pairs,
    }


def comparisons(costs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Use the same complete groups for every comparison and its denominator."""
    return {
        "simple_sharing": metrics(
            costs["separate_tokens"],
            costs["simple_tokens"],
            costs["separate_causal_pairs"],
            costs["simple_causal_pairs"],
        ),
        "cross_trajectory_sharing": metrics(
            costs["separate_tokens"],
            costs["merged_tokens"],
            costs["separate_causal_pairs"],
            costs["merged_causal_pairs"],
        ),
        "tree_over_simple": metrics(
            costs["simple_tokens"], costs["merged_tokens"], costs["simple_causal_pairs"], costs["merged_causal_pairs"]
        ),
    }


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
        "prompt_share": ratio(prompt, total),
        "response_share": ratio(response, total),
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
    initial_prompt_ids = ordered[0]["prompt_ids"]
    final_context_ids = ordered[-1]["prompt_ids"] + ordered[-1]["response_ids"]
    logical_response = len(final_context_ids) - len(initial_prompt_ids)
    if logical_response < 0:
        raise ValueError(f"trajectory {trajectory!r} ends with fewer tokens than its initial prompt")

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
            "policy_loss_token_share_of_suffix": ratio(model_response, response_suffix),
        },
        "_initial_prompt_ids": initial_prompt_ids,
        "_logical_response_tokens": logical_response,
        "_logical_total_tokens": len(final_context_ids),
        "_sharing_segments": segments,
    }


def analyze_sharing(trajectories: list[dict[str, Any]], requested_group_key: str) -> dict[str, Any]:
    """Measure exact-prompt sharing potential using the production grouping rule."""
    for trajectory in trajectories:
        trajectory["_shared_length"] = None
        trajectory["_training_shared_length"] = None
    resolved_keys = {row["group_key"] for row in trajectories}
    if resolved_keys == {None}:
        return {
            "available": False,
            "reason": "Records contain neither data_id nor task_id; sharing groups cannot be reconstructed.",
        }
    if None in resolved_keys or len(resolved_keys) != 1:
        raise ValueError(f"Inconsistent sharing group keys across trajectories: {sorted(map(str, resolved_keys))}")
    resolved_group_key = next(iter(resolved_keys))

    initial_prompt_groups: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
    exact_prompt_groups: dict[tuple[str, tuple[int, ...]], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(
        list
    )
    prompt_tokens = suffix_tokens = 0
    segment_count = 0
    for trajectory in trajectories:
        group_id = trajectory["group_id"]
        initial_prompt_groups[(group_id, tuple(trajectory["_initial_prompt_ids"]))].append(trajectory)
        trajectory["_shared_length"] = 0
        trajectory["_training_shared_length"] = 0
        for segment in trajectory["_sharing_segments"]:
            prompt = segment["prompt_ids"]
            exact_prompt_groups[(group_id, tuple(prompt))].append((trajectory, segment))
            prompt_tokens += len(prompt)
            suffix_tokens += segment["response_suffix_tokens"]
            segment_count += 1

    for group in initial_prompt_groups.values():
        if len(group) >= 2:
            shared_length = len(group[0]["_initial_prompt_ids"])
            for trajectory in group:
                trajectory["_shared_length"] = shared_length

    shared = [segments for segments in exact_prompt_groups.values() if len(segments) >= 2]
    for group in shared:
        shared_length = len(group[0][1]["prompt_ids"])
        for trajectory, _ in group:
            trajectory["_training_shared_length"] += shared_length
    shareable_occurrences = sum(len(segments[0][1]["prompt_ids"]) * len(segments) for segments in shared)
    shared_once = sum(len(segments[0][1]["prompt_ids"]) for segments in shared)
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
        "shareable_prompt_fraction": ratio(shareable_occurrences, prompt_tokens),
        "prompt_deduplication_ratio": ratio(reducible_duplicates, prompt_tokens),
        "token_reduction": ratio(reducible_duplicates, independent_tokens),
        "grouped_total_tokens": grouped_tokens,
        "token_work_ratio": ratio(independent_tokens, grouped_tokens),
        "assumption": (
            "All exact-prompt segments from the same group are present in one PrefixGrouper micro-batch; "
            "realized sharing cannot cross micro-batch or data-parallel-rank boundaries."
        ),
    }


def trajectory_record(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Build the required per-trajectory record plus training diagnostics."""
    initial_prompt_length = len(trajectory["_initial_prompt_ids"])
    response_length = trajectory["_logical_response_tokens"]
    total_length = trajectory["_logical_total_tokens"]
    shared_length = trajectory["_shared_length"]
    training = trajectory["training_trajectory"]
    training_shared_length = trajectory["_training_shared_length"]
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "group_key": trajectory["group_key"],
        "group_id": trajectory["group_id"],
        "turns": trajectory["turns"],
        "shared_prompt_fraction": ratio(shared_length, total_length) if shared_length is not None else None,
        "response_length": response_length,
        "initial_prompt_length": initial_prompt_length,
        "shared_length": shared_length,
        "total_length": total_length,
        "model_response_length": training["model_response_tokens"],
        "prefix_breaks": trajectory["prefix_breaks"],
        "training_segments": trajectory["segments"],
        "training_prompt_length": training["prompt_tokens"],
        "training_response_length": training["response_suffix_tokens"],
        "training_shared_length": training_shared_length,
        "training_total_length": training["total_tokens"],
        "training_shared_prompt_fraction": (
            ratio(training_shared_length, training["total_tokens"]) if training_shared_length is not None else None
        ),
    }


def field_mean(records: list[dict[str, Any]], field: str) -> float | None:
    """Return the arithmetic mean for a numeric per-trajectory field."""
    values = [record[field] for record in records if record[field] is not None]
    return mean(values) if values else None


def distribution(values: Iterable[float | None]) -> dict[str, Any]:
    """Summarize an unweighted distribution; aggregate token ratios are separate."""
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return {
        "count": len(finite),
        "mean": mean(finite) if finite else None,
        "stddev": pstdev(finite) if finite else None,
        "min": min(finite) if finite else None,
        "p25": percentile(finite, 0.25),
        "p50": percentile(finite, 0.50),
        "p75": percentile(finite, 0.75),
        "p95": percentile(finite, 0.95),
        "max": max(finite) if finite else None,
        "zero_count": sum(value == 0 for value in finite),
    }


def load_training_trajectories(
    path: Path, role: str, group_key: str
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    """Read ordered token calls once for all training and micro-batch metrics."""
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
        if resolved_group_key is None:
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
        selected_records += 1

    if not groups:
        raise ValueError(f"No analyzable records found in {path}")

    trajectories = [analyze_trajectory(identifier, turns) for identifier, turns in sorted(groups.items())]
    return trajectories, selected_records, dict(skipped_roles)


def summarize_training(
    path: Path,
    role: str,
    group_key: str,
    trajectories: list[dict[str, Any]],
    selected_records: int,
    skipped_roles: dict[str, int],
) -> dict[str, Any]:
    """Summarize token composition and exact-prompt sharing before batch limits."""
    api_prompt = sum(row["api_call"]["prompt_tokens"] for row in trajectories)
    model_response = sum(row["api_call"]["model_response_tokens"] for row in trajectories)
    training_prompt = sum(row["training_trajectory"]["prompt_tokens"] for row in trajectories)
    response_suffix = sum(row["training_trajectory"]["response_suffix_tokens"] for row in trajectories)
    environment_tokens = sum(row["training_trajectory"]["environment_tokens"] for row in trajectories)

    sharing = analyze_sharing(trajectories, group_key)
    per_trajectory = [trajectory_record(trajectory) for trajectory in trajectories]
    mean_fields = (
        "shared_prompt_fraction",
        "response_length",
        "initial_prompt_length",
        "shared_length",
        "total_length",
        "model_response_length",
        "training_prompt_length",
        "training_response_length",
        "training_shared_length",
        "training_total_length",
        "training_shared_prompt_fraction",
    )
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
                "policy_loss_token_share_of_suffix": ratio(model_response, response_suffix),
            },
        },
        "prefix_sharing": sharing,
        "per_trajectory_mean": {field: field_mean(per_trajectory, field) for field in mean_fields},
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
            "shareable_prompt_fraction": (
                "Prompt-token occurrences belonging to exact-prompt groups of size >= 2, divided by all prompt tokens."
            ),
            "prompt_deduplication_ratio": (
                "Duplicate prompt tokens removable by computing each shared prompt once, divided by all prompt tokens."
            ),
            "token_reduction": (
                "Duplicate prompt tokens removable by exact-prefix sharing, divided by independent prompt + suffix tokens."
            ),
            "per_trajectory.response_length": (
                "Final full-history prompt plus final response, minus the first-turn prompt; includes model and environment tokens."
            ),
            "per_trajectory.shared_length": (
                "Initial-prompt tokens shared when at least two trajectories in the same group have exactly equal prompts."
            ),
            "per_trajectory.shared_prompt_fraction": "shared_length / total_length for that logical trajectory.",
            "per_trajectory.training_*": (
                "Exact-prefix-segment metrics matching PrefixGrouper's untruncated training representation."
            ),
        },
        "per_trajectory": per_trajectory,
    }
    return report


def parse_micro_batch_sizes(value: str) -> list[int]:
    """Parse a comma-separated, strictly increasing set of positive sizes."""
    try:
        sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("micro-batch sizes must be integers") from error
    if not sizes or sizes[0] <= 0:
        raise argparse.ArgumentTypeError("micro-batch sizes must be positive")
    return sizes


def task_rows(workload: str, trajectories: list[dict[str, Any]], sizes: list[int]) -> list[dict[str, Any]]:
    """Compute micro-batch-constrained sharing metrics for every task."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        by_task[str(trajectory["group_id"])].append(trajectory)

    rows: list[dict[str, Any]] = []
    for task_id, task_trajectories in sorted(by_task.items()):
        interaction_rounds = [int(trajectory["turns"]) for trajectory in task_trajectories]
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
            # Each prompt group is independently packed into ceil(n / size)
            # batches. This is an optimistic bound: different groups may
            # start partway through a batch, and DP ranks split groups.
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
                    "interaction_rounds_min": min(interaction_rounds),
                    "interaction_rounds_mean": mean(interaction_rounds),
                    "interaction_rounds_p50": percentile([float(value) for value in interaction_rounds], 0.50),
                    "interaction_rounds_p95": percentile([float(value) for value in interaction_rounds], 0.95),
                    "interaction_rounds_max": max(interaction_rounds),
                    "initial_prompt_length_min": min(initial_prompt_lengths),
                    "initial_prompt_length_mean": mean(initial_prompt_lengths),
                    "initial_prompt_length_max": max(initial_prompt_lengths),
                    "final_trajectory_length_min": min(final_trajectory_lengths),
                    "final_trajectory_length_mean": mean(final_trajectory_lengths),
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
                    "token_reduction": saved_tokens / independent_tokens,
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
        stats = distribution(float(row["token_reduction"]) for row in members)
        total_independent = sum(int(row["independent_total_tokens"]) for row in members)
        total_saved = sum(int(row["reducible_duplicate_prompt_tokens"]) for row in members)
        summaries.append(
            {
                "workload": workload,
                "micro_batch_size_per_device": size,
                **{
                    ("task_count" if key == "count" else "zero_share_tasks" if key == "zero_count" else key): value
                    for key, value in stats.items()
                },
                "weighted_token_reduction": total_saved / total_independent,
                "total_independent_tokens": total_independent,
                "total_saved_tokens": total_saved,
            }
        )
    return summaries


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
        "- 交互轮数 = 一条 trajectory 中 policy 模型的调用次数。",
        "- 初始 prompt 长度 = trajectory 第一次 policy 调用的 prompt token 数。",
        "- 最终轨迹长度 = trajectory 最后一次 policy 调用的完整 prompt 加 response token 数。",
        "- 同一 task 的完全相同 prompt 会按生产逻辑连续排列；一个 prompt group 跨越几个 micro-batch，就需计算几次。",
        "- 结果是每个相同 prompt 组独立对齐 micro-batch 边界时的乐观上界；不模拟实际装箱偏移、DP rank 分配或训练截断，不是实测加速比。",
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
            "| 数据集 | task | micro-batch | 交互轮数均值 | 交互轮数 P95 | 初始 prompt 均值 | 最终轨迹均值 | 最终轨迹 P95 | 独立 token | 可省 token | 共享比例 | token work ratio |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {workload} | `{task_id}` | {micro_batch_size_per_device} | "
            "{rounds_mean:.2f} | {rounds_p95:.2f} | {initial_mean:.2f} | {final_mean:.2f} | {final_p95:.2f} | "
            "{independent_total_tokens} | {reducible_duplicate_prompt_tokens} | {sharing} | {work_ratio:.4f}× |".format(
                **row,
                rounds_mean=float(row["interaction_rounds_mean"]),
                rounds_p95=float(row["interaction_rounds_p95"]),
                initial_mean=float(row["initial_prompt_length_mean"]),
                final_mean=float(row["final_trajectory_length_mean"]),
                final_p95=float(row["final_trajectory_length_p95"]),
                sharing=percent(float(row["token_reduction"])),
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
                weighted=percent(float(summary["weighted_token_reduction"])),
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


def analyze_raw(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate raw calls and select complete trajectory groups."""
    config = json.loads((root / "config.json").read_text())
    group_size = config["rollouts_per_task"]
    if type(group_size) is not int or group_size < 2:
        raise ValueError("Cross-trajectory sharing requires rollouts_per_task >= 2")
    selected = json.loads((root / "selected_tasks.json").read_text())
    diagnostics: list[str] = []
    raw = read_records(root / "calls.jsonl", diagnostics)
    trajectories = read_records(root / "trajectories.jsonl", diagnostics)
    if len({row["trajectory_id"] for row in trajectories}) != len(trajectories):
        raise ValueError("Duplicate trajectory records; do not mix collection runs")
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in raw:
        by_trajectory[call["trajectory_id"]].append(call)
    for trajectory in trajectories:
        by_task[trajectory["task_id"]].append(trajectory)
    task_ids = [str(task["id"]) for task in selected]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Duplicate selected task IDs")
    if set(by_task) - set(task_ids):
        raise ValueError("Trajectory task IDs do not match selected_tasks.json")
    per_task = []
    excluded = []
    trajectory_sequences = []
    included_calls = 0
    for task_id in task_ids:
        group = by_task[task_id]
        reason = None
        if len(group) != group_size or {row["sample_index"] for row in group} != set(range(group_size)):
            reason = "missing_or_duplicate_samples"
        elif any(row["status"] != "completed" for row in group):
            reason = "incomplete_trajectory"
        calls_in_group = 0
        group_sequences = []
        for trajectory in group:
            calls = sorted(by_trajectory[trajectory["trajectory_id"]], key=lambda row: row["turn"])
            if (
                not calls
                or len(calls) != trajectory["model_calls"]
                or [row["turn"] for row in calls] != list(range(len(calls)))
                or any(
                    row["task_id"] != task_id
                    or row["sample_index"] != trajectory["sample_index"]
                    or not valid_call(row)
                    for row in calls
                )
            ):
                reason = reason or "missing_invalid_or_truncated_calls"
                continue
            if calls[-1]["response"]["choices"][0]["finish_reason"] != "stop":
                reason = reason or "no_final_model_answer"
            try:
                sequence = complete_sequence(calls)
            except ValueError as error:
                reason = reason or str(error)
                continue
            calls_in_group += len(calls)
            group_sequences.append(
                {
                    "task_id": task_id,
                    "trajectory_id": trajectory["trajectory_id"],
                    "sample_index": trajectory["sample_index"],
                    "token_ids": sequence,
                    "messages": calls[-1]["request"]["messages"] + [calls[-1]["response"]["choices"][0]["message"]],
                }
            )
        if reason:
            excluded.append({"task_id": task_id, "reason": reason, "recorded_trajectories": len(group)})
            continue
        per_task.append(
            {
                "task_id": task_id,
                "trajectories": group_size,
                "model_calls": calls_in_group,
            }
        )
        trajectory_sequences.extend(group_sequences)
        included_calls += calls_in_group
    return (
        per_task,
        trajectory_sequences,
        {
            "input": str(root),
            "group_size": group_size,
            "selected_tasks": len(selected),
            "complete_groups": len(per_task),
            "excluded_groups": len(excluded),
            "included_trajectories": len(per_task) * group_size,
            "included_model_calls": included_calls,
            "recorded_model_calls": len(raw),
            "trajectory_status": dict(Counter(t["status"] for t in trajectories)),
            "excluded": excluded,
            "diagnostics": diagnostics,
            "validation_source": "raw_model_calls_and_trajectory_status",
        },
    )


def analyze_export(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Recompute from a saved analysis export without accessing the original run or model."""
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("statistics_unit") != "one_complete_trajectory_sequence"
        or summary.get("sequence_representation") != "final_prompt_token_ids_plus_final_response_token_ids"
    ):
        raise ValueError("Analysis input must contain complete trajectory sequences, not individual model calls")
    group_size = summary["group_size"]
    if type(group_size) is not int or group_size < 2:
        raise ValueError("Analysis group_size must be an integer >= 2")
    diagnostics: list[str] = []
    sequences = read_records(root / "trajectory_sequences.jsonl", diagnostics)
    if diagnostics:
        raise ValueError("Incomplete analysis export: " + "; ".join(diagnostics))
    with (root / "per_task.csv").open(encoding="utf-8-sig", newline="") as handle:
        previous = list(csv.DictReader(handle))
    rows = [
        {"task_id": row["task_id"], "trajectories": int(row["trajectories"]), "model_calls": int(row["model_calls"])}
        for row in previous
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence in sequences:
        tokens = sequence.get("token_ids")
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Analysis contains missing or invalid full-trajectory token IDs")
        groups[sequence["task_id"]].append(sequence)
    if len({row["trajectory_id"] for row in sequences}) != len(sequences):
        raise ValueError("Duplicate trajectory IDs in analysis export")
    if len({row["task_id"] for row in rows}) != len(rows) or set(groups) != {row["task_id"] for row in rows}:
        raise ValueError("Analysis per_task.csv does not match trajectory_sequences.jsonl")
    for row, old in zip(rows, previous):
        group = groups[row["task_id"]]
        if (
            row["trajectories"] != group_size
            or len(group) != group_size
            or {sequence["sample_index"] for sequence in group} != set(range(group_size))
            or row["model_calls"] < group_size
        ):
            raise ValueError(f"Incomplete or inconsistent exported group: {row['task_id']}")
        lengths = [len(sequence["token_ids"]) for sequence in group]
        if sum(lengths) != int(old["separate_tokens"]) or sum(n * (n + 1) // 2 for n in lengths) != int(
            old["separate_causal_pairs"]
        ):
            raise ValueError(f"Exported token sequences disagree with saved baseline: {row['task_id']}")
    if (
        len(rows) != summary["complete_groups"]
        or len(sequences) != summary["included_trajectories"]
        or sum(row["model_calls"] for row in rows) != summary["included_model_calls"]
        or len(rows) + summary["excluded_groups"] != summary["selected_tasks"]
    ):
        raise ValueError("Analysis coverage counts disagree with exported records")
    keys = (
        "input",
        "group_size",
        "selected_tasks",
        "complete_groups",
        "excluded_groups",
        "included_trajectories",
        "included_model_calls",
        "recorded_model_calls",
        "trajectory_status",
        "excluded",
        "diagnostics",
    )
    context = {key: summary[key] for key in keys}
    context["validation_source"] = "previously_validated_complete_sequence_export"
    return rows, sequences, context


def load_call_sequences(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select complete groups, preserving every policy call as an independent context."""
    config = json.loads((root / "config.json").read_text())
    if config.get("statistics_unit") != CALL_UNIT:
        raise ValueError(
            "This analyzer requires original-workflow call-slot collection, not RAG single-sequence traces"
        )
    size = config["rollouts_per_task"]
    if type(size) is not int or size < 2:
        raise ValueError("rollouts_per_task must be >= 2")
    diagnostics: list[str] = []
    calls = read_records(root / "calls.jsonl", diagnostics)
    trajectories = read_records(root / "trajectories.jsonl", diagnostics)
    environment = (
        read_records(root / "environment_calls.jsonl", diagnostics)
        if (root / "environment_calls.jsonl").exists()
        else []
    )
    selected = json.loads((root / "selected_tasks.json").read_text())
    task_ids = [str(task["id"]) for task in selected]
    if len(set(task_ids)) != len(task_ids) or len({row["trajectory_id"] for row in trajectories}) != len(trajectories):
        raise ValueError("Duplicate task or trajectory IDs")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    auxiliary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        by_task[trajectory["task_id"]].append(trajectory)
    if set(by_task) - set(task_ids):
        raise ValueError("Unknown task IDs in trajectories")
    for call in calls:
        by_trajectory[call["trajectory_id"]].append(call)
    for call in environment:
        auxiliary[call["trajectory_id"]].append(call)
    exported = []
    excluded = []
    for task_id in task_ids:
        group = by_task[task_id]
        reason = None
        if len(group) != size or {t["sample_index"] for t in group} != set(range(size)):
            reason = "missing_or_duplicate_samples"
        elif any(t["status"] != "completed" for t in group):
            reason = "incomplete_trajectory"
        records = []
        for trajectory in group:
            policy = sorted(by_trajectory[trajectory["trajectory_id"]], key=lambda c: c["turn"])
            if (
                not policy
                or len(policy) != trajectory["model_calls"]
                or [c["turn"] for c in policy] != list(range(len(policy)))
                or any(
                    c.get("role") != "policy"
                    or c["task_id"] != task_id
                    or c["sample_index"] != trajectory["sample_index"]
                    or not valid_call(c)
                    for c in policy
                )
            ):
                reason = reason or "invalid_missing_or_truncated_policy_calls"
                continue
            if policy[-1]["response"]["choices"][0]["finish_reason"] != "stop":
                reason = reason or "no_final_policy_response"
            aux = auxiliary[trajectory["trajectory_id"]]
            for role in ("answerer", "search"):
                role_calls = sorted((c for c in aux if c.get("role") == role), key=lambda c: c["turn"])
                expected = trajectory["role_model_calls"][role]
                if len(role_calls) != expected or [c["turn"] for c in role_calls] != list(range(expected)):
                    reason = reason or "missing_or_duplicate_environment_calls"
            if config["agent"] == "q20" and not any(c.get("role") == "answerer" for c in aux):
                reason = reason or "missing_answerer_calls"
            if any(
                not valid_call(c) or c["task_id"] != task_id or c["sample_index"] != trajectory["sample_index"]
                for c in aux
            ):
                reason = reason or "invalid_or_truncated_environment_call"
            for call in policy:
                records.append(
                    {
                        "task_id": task_id,
                        "trajectory_id": trajectory["trajectory_id"],
                        "sample_index": trajectory["sample_index"],
                        "role": "policy",
                        "turn": call["turn"],
                        "token_ids": call["prompt_token_ids"] + call["response_token_ids"],
                        "request": call["request"],
                        "response": call["response"],
                    }
                )
        if reason:
            excluded.append({"task_id": task_id, "reason": reason})
        else:
            exported.extend(records)
    context = {
        "statistics_unit": CALL_UNIT,
        "input": str(root),
        "agent": config["agent"],
        "model": config["model"],
        "workflow_implementation": config["workflow_implementation"],
        "group_size": size,
        "selected_tasks": len(selected),
        "complete_groups": len(selected) - len(excluded),
        "excluded_groups": len(excluded),
        "included_trajectories": (len(selected) - len(excluded)) * size,
        "included_model_calls": len(exported),
        "recorded_model_calls": len(calls),
        "environment_model_calls": len(environment),
        "valid_environment_tokens": sum(
            len(c["prompt_token_ids"]) + len(c["response_token_ids"]) for c in environment if valid_call(c)
        ),
        "trajectory_status": dict(Counter(t["status"] for t in trajectories)),
        "excluded": excluded,
        "diagnostics": diagnostics,
        "collection_settings": {
            k: config[k] for k in ("q20_search", "seed", "concurrency", "max_model_calls", "trajectory_timeout")
        },
        "validation_source": "raw_calls_and_original_workflow_completion",
    }
    return exported, context


def analyze_calls(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    """Write per-task and aggregate reports with one sequence per real call, never concatenate histories."""
    if (root / "config.json").is_file():
        records, context = load_call_sequences(root)
    else:
        context = json.loads((root / "summary.json").read_text())
        if context.get("statistics_unit") != CALL_UNIT:
            raise ValueError("Expected a call-slot analysis directory")
        diagnostics: list[str] = []
        records = read_records(root / "call_sequences.jsonl", diagnostics)
        if diagnostics:
            raise ValueError("; ".join(diagnostics))
        context["validation_source"] = "previously_validated_call_sequence_export"
    size = context["group_size"]
    if type(size) is not int or size < 2:
        raise ValueError("Invalid group size")
    groups: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    trajectory_ids: dict[tuple[str, int], str] = {}
    for row in records:
        key = (row["task_id"], row["sample_index"], row["turn"])
        tokens = row["token_ids"]
        if key in seen or row["role"] != "policy" or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("Invalid or duplicate exported call sequence")
        seen.add(key)
        trajectory_key = (row["task_id"], row["sample_index"])
        if trajectory_key in trajectory_ids and trajectory_ids[trajectory_key] != row["trajectory_id"]:
            raise ValueError("Mixed trajectory IDs for one sample")
        trajectory_ids[trajectory_key] = row["trajectory_id"]
        groups[row["task_id"]][row["sample_index"]].append(row)
    if (
        len(groups) != context["complete_groups"]
        or len(records) != context["included_model_calls"]
        or len(trajectory_ids) != context["included_trajectories"]
    ):
        raise ValueError("Export coverage differs from summary")
    per_task = []
    for task_id, samples in groups.items():
        if set(samples) != set(range(size)):
            raise ValueError("Incomplete sample group in export")
        sequences = []
        for sample in range(size):
            calls = sorted(samples[sample], key=lambda row: row["turn"])
            if [row["turn"] for row in calls] != list(range(len(calls))):
                raise ValueError("Missing or duplicated call ordinals")
            sequences.append([row["token_ids"] for row in calls])
        costs = dict.fromkeys(COSTS, 0)
        for slot in range(max(map(len, sequences))):
            # Empty entries preserve G-way simple sharing: a missing call means no prefix common to all G.
            slot_costs = sharing_costs([trajectory[slot] if slot < len(trajectory) else [] for trajectory in sequences])
            for key in COSTS:
                costs[key] += slot_costs[key]
        row = {
            "task_id": task_id,
            "trajectories": size,
            "model_calls": sum(map(len, sequences)),
            "call_slots": max(map(len, sequences)),
            **costs,
        }
        for name, value in comparisons(costs).items():
            row.update({f"{name}_{key}": v for key, v in value.items()})
        per_task.append(row)
    totals = {key: sum(row[key] for row in per_task) for key in COSTS}
    results = comparisons(totals)
    summary = {
        **context,
        **results,
        "analysis_input": str(root),
        "definitions": {
            "baseline": "Sum prompt+response length (and triangular causal pairs) for every real policy call in each complete trajectory.",
            "alignment": "Within each question, align policy calls by zero-based ordinal. Each slot contains at most one call from each trajectory.",
            "simple": "Share the G-way longest common prefix in each slot once. A missing call is an empty sequence, giving common prefix zero.",
            "tree": "One trie per call slot across trajectories; never share between slots or within a trajectory. Not a global optimum across all possible call alignments.",
            "environment": "Q20 answerer/search calls are saved separately and excluded from policy sharing. No original prompts, nodes, histories or termination rules are changed.",
            "scope": "A sum of independent model contexts, not a flattened dialogue, measured speedup, or a claim that training uses one concatenated forward pass.",
        },
    }
    return per_task, records, summary, totals


def analyze_complete_trajectories(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    """Compare complete full-history sequences using the common cost functions."""
    if (root / "config.json").is_file():
        per_task, records, context = analyze_raw(root)
    else:
        per_task, records, context = analyze_export(root)
    groups: dict[str, list[list[int]]] = defaultdict(list)
    for record in records:
        groups[record["task_id"]].append(record["token_ids"])
    for row in per_task:
        row.update(sharing_costs(groups[row["task_id"]]))
        for name, comparison in comparisons(row).items():
            row.update({f"{name}_{key}": value for key, value in comparison.items()})
    totals = {key: sum(row[key] for row in per_task) for key in COSTS}
    summary = {
        **context,
        "statistics_unit": TRAJECTORY_UNIT,
        "sequence_representation": "final_prompt_token_ids_plus_final_response_token_ids",
        "analysis_input": str(root),
        **comparisons(totals),
        "definitions": {
            "baseline": "One final full-history prompt plus final response per complete trajectory.",
            "simple": "Share the exact prefix common to all G trajectories of one task once.",
            "tree": "Merge exact prefixes using a trie per task, including subgroup branches; never merge after divergence.",
            "scope": "History and actions must be preserved. Not a sum of API calls or a simulation of training batches.",
        },
    }
    return per_task, records, summary, totals


def write_json(path: Path, value: Any) -> None:
    """Write finite, human-readable metrics."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_sequence_report(root: Path, output: Path, view: str) -> None:
    """Use one report format for the two explicitly named sequence views."""
    analyzer = analyze_calls if view == "calls" else analyze_complete_trajectories
    per_task, records, summary, totals = analyzer(root)
    summary["view"] = view
    summary["definitions"].update(
        {
            "token_reduction": "Removed token positions divided by the independent baseline token positions.",
            "tree_over_simple": "Additional reductions use the simple-sharing cost as denominator.",
            "causal_pairs": "Full causal attention pairs, including the diagonal: L*(L+1)/2.",
            "limitation": "Structural estimate, not measured speedup or implementation support. Sharing requires matching model state, positions and masks; preserve per-sample loss and advantage weights.",
        }
    )
    table = [
        {
            "scheme": name,
            **metrics(totals["separate_tokens"], totals[token], totals["separate_causal_pairs"], totals[pairs]),
        }
        for name, token, pairs in (
            ("independent", "separate_tokens", "separate_causal_pairs"),
            ("simple_sharing", "simple_tokens", "simple_causal_pairs"),
            ("cross_trajectory_sharing", "merged_tokens", "merged_causal_pairs"),
        )
    ]
    lines = [
        f"# 多轮共享分析：{view}",
        "",
        f"输入：`{root}`；统计单位：`{summary['statistics_unit']}`。",
        f"完整题组 {summary['complete_groups']}/{summary['selected_tasks']}，"
        f"纳入 {summary['included_trajectories']} 条轨迹、{summary['included_model_calls']} 次调用。",
        f"验证来源：`{summary['validation_source']}`。",
        "",
        "| 方案 | token 位置数 | token 减少比例 | 因果注意力对 | 注意力对减少比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table:
        reduction = "N/A" if row["token_reduction"] is None else percent(row["token_reduction"])
        pairs = "N/A" if row["causal_pair_reduction"] is None else percent(row["causal_pair_reduction"])
        lines.append(
            f"| {row['scheme']} | {row['merged_tokens']} | {reduction} | {row['merged_causal_pairs']} | {pairs} |"
        )
    lines.extend(["", "## 统计口径", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in summary["definitions"].items())
    extra = summary["tree_over_simple"]
    lines.extend(["", f"前缀树相对简单共享额外减少 {extra['saved_tokens']} 个 token；详细比例见 summary.json。"])
    if view == "calls":
        lines.append(
            f"环境模型调用 {summary['environment_model_calls']} 次，有效 token 数 {summary['valid_environment_tokens']}，不计入 policy 共享。"
        )
    lines.append("导出目录重算复用此前完整组筛选；不能重新验证缺失的原始调用或恢复排除组。")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    sequence_file = "call_sequences.jsonl" if view == "calls" else "trajectory_sequences.jsonl"
    with (output / sequence_file).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_csv(output / "per_task.csv", per_task, list(per_task[0]) if per_task else ["task_id", *COSTS])
    write_csv(output / "benefit.csv", table, list(table[0]))
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_report(inputs: dict[str, Path], output: Path, role: str, group_key: str, sizes: list[int]) -> None:
    """Write token composition and micro-batch savings from the same loaded calls."""
    workloads = {}
    task_metrics = []
    trajectory_metrics = []
    for workload, source in inputs.items():
        path = source / "calls.jsonl" if source.is_dir() else source
        trajectories, count, skipped = load_training_trajectories(path, role, group_key)
        report = summarize_training(path, role, group_key, trajectories, count, skipped)
        report["validation_source"] = "token_calls_only; completion and sampling-group coverage are not validated"
        workloads[workload] = report
        trajectory_metrics.extend({"workload": workload, **row} for row in report["per_trajectory"])
        task_metrics.extend(task_rows(workload, trajectories, sizes))
    summaries = summarize_rows(task_metrics)
    resolved_inputs = {label: str(path) for label, path in inputs.items()}
    summary = {
        "view": "training",
        "statistics_unit": "untruncated_training_segments",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": resolved_inputs,
        "role": role,
        "group_key": group_key,
        "micro_batch_sizes_per_device": sizes,
        "metric": "token_reduction = reducible_duplicate_prompt_tokens / independent_total_tokens",
        "assumptions": [
            "Each exact-prompt group is independently aligned to a micro-batch boundary: optimistic upper bound.",
            "No sharing across tasks or data-parallel ranks; actual packing, rank splits and truncation are not simulated.",
            "All supplied token calls are included; this view does not validate completed trajectories or full sampling groups.",
            "Structural token work only, not measured training speedup or memory savings.",
        ],
        "workloads": workloads,
        "per_task": task_metrics,
        "task_distribution": summaries,
    }
    text = render_report(resolved_inputs, task_metrics, summaries)
    text += "\n## 轨迹 token 构成\n\n"
    text += "训练视图纳入提供的 token 调用，不检查题组是否完整或轨迹是否正常完成。\n\n"
    text += "| 工作负载 | 轨迹数 | policy 调用数 | 前缀中断 | 训练分段 | prompt token | suffix token |\n"
    text += "|---|---:|---:|---:|---:|---:|---:|\n"
    for label, report in workloads.items():
        training = report["overall_weighted"]["training_trajectory"]
        text += f"| {label} | {report['trajectories']} | {report['turns']} | {report['prefix_breaks']} | {report['prefix_sharing']['training_segments']} | {training['prompt_tokens']} | {training['response_suffix_tokens']} |\n"
    text += "\n逐轨迹的 shared_prompt_fraction 是 prompt 占比；token_reduction 才是消除重复计算的比例，二者不能互换。\n"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    for filename, rows in (
        ("per_trajectory.csv", trajectory_metrics),
        ("per_task_microbatch.csv", task_metrics),
        ("task_distribution.csv", summaries),
    ):
        write_csv(output / filename, rows, list(rows[0]))
    (output / "report.md").write_text(text, encoding="utf-8")


def parse_input(value: str) -> tuple[str, Path]:
    """Accept PATH or LABEL=PATH without changing the input representation."""
    label, separator, raw_path = value.partition("=")
    path = Path(raw_path if separator else value).expanduser().resolve()
    if not separator:
        label = path.parent.name if path.is_file() else path.name
    if not label.strip() or (separator and not raw_path.strip()):
        raise argparse.ArgumentTypeError("--input must be PATH or LABEL=PATH")
    return label.strip(), path


def main() -> None:
    """Dispatch all multi-turn analysis through one CLI and one output option."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", type=parse_input, required=True, metavar="[LABEL=]PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=("training", "calls", "trajectory"), default="training")
    parser.add_argument(
        "--micro-batch-sizes", type=parse_micro_batch_sizes, help="Training only; default: 1,2,4,8,16,32,64"
    )
    parser.add_argument("--role", help="Training only; default: policy; use all to retain all roles")
    parser.add_argument("--group-key", choices=("auto", "data_id", "task_id"), help="Training only; default: auto")
    args = parser.parse_args()
    inputs = dict(args.input)
    if len(inputs) != len(args.input):
        parser.error("input labels must be unique; use LABEL=PATH for repeated directory names")
    if args.view != "training" and (
        len(inputs) != 1 or any(value is not None for value in (args.role, args.group_key, args.micro_batch_sizes))
    ):
        parser.error("calls/trajectory views take one input and no training-only options")
    output = args.output_dir.expanduser().resolve()
    if (output / "config.json").exists() or (output / "calls.jsonl").exists():
        parser.error("output cannot be a raw collection directory")
    if (output / "summary.json").exists():
        parser.error("output already contains a report; choose a fresh --output-dir")
    try:
        if args.view == "training":
            write_training_report(
                inputs,
                output,
                args.role or "policy",
                args.group_key or "auto",
                args.micro_batch_sizes or [1, 2, 4, 8, 16, 32, 64],
            )
        else:
            write_sequence_report(next(iter(inputs.values())), output, args.view)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
