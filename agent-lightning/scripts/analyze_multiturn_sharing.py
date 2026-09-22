#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

r"""Analyze multi-turn prefix sharing through one offline entrypoint.

The default view compares the first N collected rollouts of each task. Its
independent token count sums one final context per selected rollout, not all
training segments. Training-segment diagnostics are reported separately.
The calls and trajectory views compare common-prefix and trie sharing using
explicitly different statistical units. No model or accelerator is required.

Usage:
    python scripts/analyze_multiturn_sharing.py \
        --input sql=/runs/sql/calls.jsonl --input q20=/runs/q20/calls.jsonl \
        --rollout-counts 1,2,4,8,16,32,64 --output-dir /runs/sharing
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
        "_final_context_ids": final_context_ids,
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
    """Read token calls, retaining collection order and optional rollout indices."""
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
        sample_index = record.get("sample_index", record.get("rollout_index"))
        if sample_index is not None and (type(sample_index) is not int or sample_index < 0):
            raise ValueError(f"{context} has an invalid sample_index or rollout_index")
        groups[identifier].append(
            {
                "turn": turn_index(record, fallback, context),
                "prompt_ids": token_ids(record, "prompt", context),
                "response_ids": token_ids(record, "response", context),
                "group_key": resolved_group_key,
                "group_id": group_id,
                "sample_index": sample_index,
            }
        )
        selected_records += 1

    if not groups:
        raise ValueError(f"No analyzable records found in {path}")

    trajectories = []
    for collection_order, (identifier, turns) in enumerate(groups.items()):
        indices = {turn["sample_index"] for turn in turns if turn["sample_index"] is not None}
        if len(indices) > 1:
            raise ValueError(f"trajectory {identifier!r} changes rollout index across turns")
        trajectory = analyze_trajectory(identifier, turns)
        trajectory["sample_index"] = next(iter(indices)) if indices else None
        trajectory["collection_order"] = collection_order
        trajectories.append(trajectory)
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


def parse_rollout_counts(value: str) -> list[int]:
    """Parse positive rollout cohort sizes in ascending order."""
    try:
        counts = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("rollout counts must be integers") from error
    if not counts or counts[0] <= 0:
        raise argparse.ArgumentTypeError("rollout counts must be positive")
    return counts


def task_rows(
    workload: str, trajectories: list[dict[str, Any]], counts: list[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select nested rollout cohorts and sum their final contexts exactly once."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        by_task[str(trajectory["group_id"])].append(trajectory)
    rows = []
    skipped = []
    for task_id, task_trajectories in sorted(by_task.items()):
        indices = [trajectory["sample_index"] for trajectory in task_trajectories]
        indexed = all(index is not None for index in indices) and len(set(indices)) == len(indices)
        order_key = "sample_index" if indexed else "collection_order"
        ordered = sorted(task_trajectories, key=lambda trajectory: trajectory[order_key])
        for count in counts:
            if len(ordered) < count:
                skipped.append(
                    {
                        "workload": workload,
                        "task_id": task_id,
                        "rollout_count": count,
                        "available_rollouts": len(ordered),
                        "reason": "insufficient_rollouts",
                    }
                )
                continue
            selected = ordered[:count]
            rounds = [trajectory["turns"] for trajectory in selected]
            prompt_lengths = [len(trajectory["_initial_prompt_ids"]) for trajectory in selected]
            final_lengths = [trajectory["_logical_total_tokens"] for trajectory in selected]
            exact_prompts: Counter[tuple[int, ...]] = Counter()
            for trajectory in selected:
                prompt = trajectory["_initial_prompt_ids"]
                # A rewritten final context may no longer contain the initial
                # prompt. Never subtract tokens absent from this denominator.
                if starts_with(trajectory["_final_context_ids"], prompt):
                    exact_prompts[tuple(prompt)] += 1
            independent_tokens = sum(final_lengths)
            if independent_tokens <= 0:
                raise ValueError(f"{workload}/{task_id} has no final-context tokens")
            saved_tokens = sum((occurrences - 1) * len(prompt) for prompt, occurrences in exact_prompts.items())
            grouped_tokens = independent_tokens - saved_tokens
            rows.append(
                {
                    "workload": workload,
                    "task_id": task_id,
                    "rollout_count": count,
                    "available_rollouts": len(ordered),
                    "selection_order": order_key,
                    "selected_rollout_ids": [trajectory["trajectory_id"] for trajectory in selected],
                    "interaction_rounds_mean": mean(rounds),
                    "interaction_rounds_p50": percentile([float(value) for value in rounds], 0.50),
                    "interaction_rounds_p95": percentile([float(value) for value in rounds], 0.95),
                    "initial_prompt_length_mean": mean(prompt_lengths),
                    "final_trajectory_length_mean": mean(final_lengths),
                    "final_trajectory_length_p50": percentile([float(value) for value in final_lengths], 0.50),
                    "final_trajectory_length_p95": percentile([float(value) for value in final_lengths], 0.95),
                    "prefix_breaks": sum(trajectory["prefix_breaks"] for trajectory in selected),
                    "initial_prompt_preserved_rollouts": sum(exact_prompts.values()),
                    "repeated_prompt_groups": sum(occurrences >= 2 for occurrences in exact_prompts.values()),
                    "independent_total_tokens": independent_tokens,
                    "reducible_duplicate_prompt_tokens": saved_tokens,
                    "grouped_total_tokens": grouped_tokens,
                    "token_reduction": saved_tokens / independent_tokens,
                    "token_work_ratio": independent_tokens / grouped_tokens,
                }
            )
    return rows, skipped


def summarize_rows(rows: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report coverage and token-weighted savings for every requested cohort size."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    excluded: Counter[tuple[str, int]] = Counter()
    for row in rows:
        grouped[(row["workload"], row["rollout_count"])].append(row)
    for row in skipped:
        excluded[(row["workload"], row["rollout_count"])] += 1
    summaries = []
    for workload, count in sorted(set(grouped) | set(excluded)):
        members = grouped[(workload, count)]
        stats = distribution(row["token_reduction"] for row in members)
        independent = sum(row["independent_total_tokens"] for row in members)
        saved = sum(row["reducible_duplicate_prompt_tokens"] for row in members)
        summaries.append(
            {
                "workload": workload,
                "rollout_count": count,
                "task_count": len(members),
                "insufficient_tasks": excluded[(workload, count)],
                "selected_rollouts": len(members) * count,
                **{key: value for key, value in stats.items() if key != "count"},
                "weighted_token_reduction": ratio(saved, independent),
                "total_independent_tokens": independent,
                "total_saved_tokens": saved,
            }
        )
    return summaries


def percent(value: float | None) -> str:
    """Format a measured fraction, keeping missing cohorts distinct from zero."""
    return "N/A" if value is None else f"{value * 100:.2f}%"


def render_report(workloads: dict[str, Any], rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> str:
    """Explain rollout selection and use one final sequence per selected rollout."""
    lines = [
        "# 不同 rollout 数量下的多轮轨迹共享分析",
        "",
        "- 每个 task 固定顺序取前 N 条 rollout；编号完整且唯一时按 sample_index/rollout_index 排序，否则按文件首次出现顺序。",
        "- 各档重新计算所选 rollout 的轮数、初始 prompt、最终轨迹均值和共享率。",
        "- 独立 token = 所选 N 条 rollout 的最终轨迹长度之和 = N × 最终轨迹长度均值（未四舍五入）。",
        "- 最终轨迹长度 = 最后一次调用的 prompt + response 长度；不累加历史调用或训练分段。",
        "- 仅共享所选 rollout 中完全相同、且仍保留在最终上下文开头的初始 prompt；各相同 prompt 组只保留一份。",
        "- 本报告比较采集样本数量，不模拟训练 micro-batch、设备分配或截断，不是实测加速比。",
        "- 前缀中断时，最终上下文不一定包含完整交互历史；训练分段诊断另列于 summary.json，不作为本表分母。",
        "",
        "## 数据覆盖",
        "",
        "| 数据集 | 输入 | 已采集 rollout | 调用数 |",
        "|---|---|---:|---:|",
    ]
    for workload, report in workloads.items():
        lines.append(
            f"| {workload} | `{report['input']}` | {report['available_rollouts']} | {report['selected_records']} |"
        )
    lines.extend(
        [
            "",
            "## 每个 task 的结果",
            "",
            "| 数据集 | task | rollout 数 | 已采集数 | 交互轮数均值 | 初始 prompt 均值 | 最终轨迹均值 | 最终轨迹 P95 | 独立 token（所选轨迹之和） | 可省 token | 共享比例 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['workload']} | `{row['task_id']}` | {row['rollout_count']} | {row['available_rollouts']} | "
            f"{row['interaction_rounds_mean']:.2f} | {row['initial_prompt_length_mean']:.2f} | "
            f"{row['final_trajectory_length_mean']:.2f} | {row['final_trajectory_length_p95']:.2f} | "
            f"{row['independent_total_tokens']} | {row['reducible_duplicate_prompt_tokens']} | {percent(row['token_reduction'])} |"
        )
    lines.extend(
        [
            "",
            "## 按 rollout 数量汇总",
            "",
            "| 数据集 | 每 task 的 rollout 数 | 纳入 task | 数量不足 task | 所选 rollout 总数 | 独立 token 总和 | task 均值 | P50 | P95 | token 加权共享率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['workload']} | {row['rollout_count']} | {row['task_count']} | {row['insufficient_tasks']} | "
            f"{row['selected_rollouts']} | {row['total_independent_tokens']} | {percent(row['mean'])} | "
            f"{percent(row['p50'])} | {percent(row['p95'])} | {percent(row['weighted_token_reduction'])} |"
        )
    lines.extend(
        [
            "",
            "数量不足的 task 跳过该档位，不复制样本或用较小数量冒充；无可用 task 时共享率为 N/A。",
            "不同档位可能覆盖不同 task，须结合纳入 task 数比较；单 task 的分位数不能代表完整数据集。",
            "所选 rollout ID 和不足数量明细见 summary.json；采集调用不等于已验证正常完成的轨迹。",
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


def write_rollout_report(inputs: dict[str, Path], output: Path, role: str, group_key: str, counts: list[int]) -> None:
    """Write nested rollout-count comparisons with separate training diagnostics."""
    workloads = {}
    task_metrics = []
    trajectory_metrics = []
    skipped_cohorts = []
    for workload, source in inputs.items():
        path = source / "calls.jsonl" if source.is_dir() else source
        trajectories, count, skipped_roles = load_training_trajectories(path, role, group_key)
        diagnostics = summarize_training(path, role, group_key, trajectories, count, skipped_roles)
        workloads[workload] = {
            "input": str(path),
            "available_rollouts": len(trajectories),
            "selected_records": count,
            "skipped_roles": skipped_roles,
            "validation_source": "token_calls_only; completion and sampling-group coverage are not validated",
            "training_segment_diagnostics": diagnostics,
        }
        for trajectory in trajectories:
            trajectory_metrics.append(
                {
                    "workload": workload,
                    "task_id": trajectory["group_id"],
                    "trajectory_id": trajectory["trajectory_id"],
                    "sample_index": trajectory["sample_index"],
                    "collection_order": trajectory["collection_order"],
                    "interaction_rounds": trajectory["turns"],
                    "initial_prompt_length": len(trajectory["_initial_prompt_ids"]),
                    "final_trajectory_length": trajectory["_logical_total_tokens"],
                    "prefix_breaks": trajectory["prefix_breaks"],
                }
            )
        rows, skipped = task_rows(workload, trajectories, counts)
        task_metrics.extend(rows)
        skipped_cohorts.extend(skipped)
    summaries = summarize_rows(task_metrics, skipped_cohorts)
    summary = {
        "view": "training",
        "statistics_unit": "selected_rollout_final_contexts",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "group_key": group_key,
        "rollout_counts": counts,
        "definitions": {
            "selection": "First N rollouts per task; unique complete sample indices first, otherwise first appearance in the input file. Larger cohorts contain smaller cohorts.",
            "independent_total_tokens": "Sum of final prompt+response lengths for exactly N selected rollouts; never a sum over training segments.",
            "sharing": "Identical initial prompts within the selected task cohort are counted once, only for rollouts whose final context still starts with that prompt.",
            "token_reduction": "reducible_duplicate_prompt_tokens / independent_total_tokens",
            "coverage": "Tasks with fewer than N rollouts are excluded from that cohort size, without replacement.",
            "scope": "Rollout-count comparison, not a micro-batch or DP-rank simulation. Final contexts may omit earlier history after prefix breaks. No measured speedup claim.",
            "training_segment_diagnostics": "All-input training-segment statistics, provided separately; these are not the denominators of the rollout-count report.",
        },
        "workloads": workloads,
        "per_task": task_metrics,
        "task_distribution": summaries,
        "skipped_cohorts": skipped_cohorts,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_csv(output / "per_trajectory.csv", trajectory_metrics, list(trajectory_metrics[0]))
    csv_tasks = [{**row, "selected_rollout_ids": json.dumps(row["selected_rollout_ids"])} for row in task_metrics]
    write_csv(
        output / "per_task_rollout_counts.csv",
        csv_tasks,
        list(csv_tasks[0]) if csv_tasks else ["workload", "task_id", "rollout_count", "independent_total_tokens"],
    )
    write_csv(output / "task_distribution.csv", summaries, list(summaries[0]))
    (output / "report.md").write_text(render_report(workloads, task_metrics, summaries), encoding="utf-8")


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
        "--rollout-counts",
        type=parse_rollout_counts,
        help="Number of collected rollouts per task; default: 1,2,4,8,16,32,64",
    )
    parser.add_argument("--role", help="Training only; default: policy; use all to retain all roles")
    parser.add_argument("--group-key", choices=("auto", "data_id", "task_id"), help="Training only; default: auto")
    args = parser.parse_args()
    inputs = dict(args.input)
    if len(inputs) != len(args.input):
        parser.error("input labels must be unique; use LABEL=PATH for repeated directory names")
    if args.view != "training" and (
        len(inputs) != 1 or any(value is not None for value in (args.role, args.group_key, args.rollout_counts))
    ):
        parser.error("calls/trajectory views take one input and no training-only options")
    output = args.output_dir.expanduser().resolve()
    if (output / "config.json").exists() or (output / "calls.jsonl").exists():
        parser.error("output cannot be a raw collection directory")
    if (output / "summary.json").exists():
        parser.error("output already contains a report; choose a fresh --output-dir")
    try:
        if args.view == "training":
            write_rollout_report(
                inputs,
                output,
                args.role or "policy",
                args.group_key or "auto",
                args.rollout_counts or [1, 2, 4, 8, 16, 32, 64],
            )
        else:
            write_sequence_report(next(iter(inputs.values())), output, args.view)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
