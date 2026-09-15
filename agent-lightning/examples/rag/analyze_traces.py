# Copyright (c) Microsoft. All rights reserved.

"""Estimate sharing across complete trajectories: python analyze_traces.py --input RUN_DIR_OR_ANALYSIS_DIR."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


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


def main() -> None:
    """Analyze raw trajectories or a standalone analysis export and write all three schemes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Collection directory or existing analysis directory."
    )
    parser.add_argument(
        "--output", type=Path, help="Destination; defaults to RUN_DIR/analysis or the input analysis directory."
    )
    args = parser.parse_args()
    root = args.input.resolve()
    if (root / "config.json").is_file():
        per_task, sequences, context = analyze_raw(root)
        default_output = root / "analysis"
    elif (root / "summary.json").is_file():
        per_task, sequences, context = analyze_export(root)
        default_output = root
    else:
        parser.error(
            "input must be a collection directory with config.json, or an analysis directory with summary.json, per_task.csv and trajectory_sequences.jsonl"
        )
    output = (args.output or default_output).resolve()
    if (output / "config.json").exists() or (output / "calls.jsonl").exists():
        parser.error("output cannot be a raw collection directory; choose its analysis subdirectory")
    groups: dict[str, list[list[int]]] = defaultdict(list)
    for sequence in sequences:
        groups[sequence["task_id"]].append(sequence["token_ids"])
    cost_fields = (
        "separate_tokens",
        "separate_causal_pairs",
        "common_prefix_tokens",
        "simple_tokens",
        "simple_causal_pairs",
        "merged_tokens",
        "merged_causal_pairs",
    )
    for row in per_task:
        row.update(sharing_costs(groups[row["task_id"]]))
        for name, comparison in comparisons(row).items():
            if name == "cross_trajectory_sharing":
                row.update(comparison)
            else:
                row.update({f"{name}_{key}": value for key, value in comparison.items()})
    totals = {key: sum(row[key] for row in per_task) for key in cost_fields}
    comparison = comparisons(totals)
    summary = {
        **context,
        "statistics_unit": "one_complete_trajectory_sequence",
        "sequence_representation": "final_prompt_token_ids_plus_final_response_token_ids",
        "analysis_input": str(root),
        **comparison,
        "definitions": {
            "baseline": "Sum one complete sequence per trajectory: final full-history prompt plus final response.",
            "simple_sharing": "Share the longest exact token prefix common to all G trajectories of the same question once; all suffixes are independent, without subgroup sharing.",
            "comparison": "Share all exact prefixes across the same G complete trajectories using a prefix tree; never merge across questions or after divergence.",
            "tree_over_simple": "Additional reductions use the simple-sharing cost as denominator, not the independent baseline.",
            "causal_pairs": "Full causal attention: L*(L+1)/2 for each sequence. A shared prefix of length P saves (G-1)*P*(P+1)/2 pairs.",
            "scope": "Identical complete groups for all schemes. Global ratios use summed costs. Analysis exports reuse prior raw-call validation; excluded groups are not recovered.",
            "loss": "Retain separate per-trajectory loss multiplicities, advantages and clipping terms; tool outputs are context only.",
            "limitation": "Structural estimate, not measured training speedup or a claim of PrefixGrouper implementation support. Matching positions, masks and model state required.",
        },
    }
    table = []
    for name, token_key, pair_key in (
        ("每条完整轨迹独立计算（基线）", "separate_tokens", "separate_causal_pairs"),
        ("简单共享（全组最长公共前缀，后缀独立）", "simple_tokens", "simple_causal_pairs"),
        (f"同题{context['group_size']}条完整轨迹合并前缀树", "merged_tokens", "merged_causal_pairs"),
    ):
        value = metrics(totals["separate_tokens"], totals[token_key], totals["separate_causal_pairs"], totals[pair_key])
        table.append(
            {
                "方案": name,
                "token位置数": value["merged_tokens"],
                "减少token位置数": value["saved_tokens"],
                "token减少比例": value["token_reduction"],
                "token工作量缩减倍数": value["token_work_ratio"],
                "causal_attention_pairs": value["merged_causal_pairs"],
                "attention_pair减少比例": value["causal_pair_reduction"],
            }
        )

    def percent(value: float | None) -> str:
        return f"{value:.2%}" if value is not None else "N/A"

    def ratio(value: float | None) -> str:
        return f"{value:.2f}x" if value is not None else "N/A"

    table_text = "\n".join(
        f"| {row['方案']} | {row['token位置数']:,} | {percent(row['token减少比例'])} | {ratio(row['token工作量缩减倍数'])} | {row['causal_attention_pairs']:,} | {percent(row['attention_pair减少比例'])} |"
        for row in table
    )
    extra = comparison["tree_over_simple"]
    report = f"""# 完整轨迹跨轨迹共享收益估算

原采集目录：`{context['input']}`。本次分析输入：`{root}`。
每题 {context['group_size']} 条轨迹，完整组 {context['complete_groups']}/{context['selected_tasks']}；
纳入 {context['included_trajectories']} 条轨迹、{context['included_model_calls']} 次模型调用，排除 {context['excluded_groups']} 组。
验证来源：`{context['validation_source']}`；三种方案使用完全相同的完整题组。

| 方案 | token 位置数 | 相对独立基线减少比例 | token 工作量缩减倍数 | causal attention pairs | 相对独立基线 pair 减少比例 |
|---|---:|---:|---:|---:|---:|
{table_text}

前缀树相对简单共享额外减少 **{extra['saved_tokens']:,}** 个 token 位置，
相对简单共享减少 **{percent(extra['token_reduction'])}**，工作量缩减倍数 **{ratio(extra['token_work_ratio'])}**；
额外减少 **{extra['separate_causal_pairs'] - extra['merged_causal_pairs']:,}** 个 attention pairs，
相对简单共享减少 **{percent(extra['causal_pair_reduction'])}**。

统计单位固定为一条完整轨迹、一条 token 序列，包含初始问题、全部模型动作、工具结果和最终回复。
序列使用最终完整历史请求的 prompt token ID 加最终 response token ID。
简单共享只保存同题全组最长公共 token 前缀一次，第一次分叉后各条后缀独立；不再对子组共享。
设该前缀长度为 P、轨迹数为 G：简单共享 token 工作量 = 独立基线 − (G−1)×P；
attention pairs = 独立基线 pairs − (G−1)×P×(P+1)/2。
前缀树还允许部分轨迹在共同分支上继续共享，但分叉后相同文本不重新合并。
简单共享是这里定义的单公共前缀对照，不代表已实测 PrefixGrouper 的分组行为。
所有汇总比例按工作量之和计算，不对各题百分比取平均。

原始轨迹输入会逐轮检查消息历史、模型动作、完成状态及真实 token ID。
分析目录输入复用此前已通过检查的完整序列，并核对题组、编号、基线 token 数与覆盖率；
无法仅凭导出目录重新验证原始调用，也不会补回此前排除的题组。

这是结构上的计算量估算，不是 NPU 训练耗时、显存节省或实测加速比。
共享要求位置编码、attention mask 和模型状态一致；每条轨迹的 loss、advantage、权重和 clipping 保持独立并累加梯度。
工具结果只作为上下文。训练接入仍须保证 tokenizer、loss mask 和 rollout log-prob 位置映射一致。
详细数据见 `per_task.csv`、`benefit.csv`、`summary.json` 和 `trajectory_sequences.jsonl`。
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "trajectory_sequences.jsonl").open("w", encoding="utf-8") as handle:
        for sequence in sequences:
            handle.write(json.dumps(sequence, ensure_ascii=False) + "\n")
    fields = list(per_task[0]) if per_task else ["task_id", "trajectories", "model_calls", *cost_fields]
    write_csv(output / "per_task.csv", per_task, fields)
    write_csv(output / "benefit.csv", table, list(table[0]))
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved benefit table: {output / 'benefit.csv'}")


if __name__ == "__main__":
    main()
