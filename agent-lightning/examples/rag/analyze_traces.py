# Copyright (c) Microsoft. All rights reserved.

"""Estimate sharing across complete trajectories: python analyze_traces.py --input RUN_DIR."""

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


def main() -> None:
    """Write an aggregate benefit table and per-question results for complete groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to RUN_DIR/analysis; raw traces are never modified.")
    args = parser.parse_args()
    root = args.input.resolve()
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
        sequences: list[list[int]] = []
        separate_tokens = separate_pairs = calls_in_group = 0
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
            length = len(sequence)
            separate_tokens += length
            separate_pairs += length * (length + 1) // 2
            calls_in_group += len(calls)
            sequences.append(sequence)
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
        merged_tokens, merged_pairs = tree_cost(sequences)
        per_task.append(
            {
                "task_id": task_id,
                "trajectories": group_size,
                "model_calls": calls_in_group,
                **metrics(separate_tokens, merged_tokens, separate_pairs, merged_pairs),
            }
        )
        trajectory_sequences.extend(group_sequences)
        included_calls += calls_in_group
    aggregate = metrics(
        *(
            sum(row[key] for row in per_task)
            for key in ("separate_tokens", "merged_tokens", "separate_causal_pairs", "merged_causal_pairs")
        )
    )
    summary = {
        "statistics_unit": "one_complete_trajectory_sequence",
        "sequence_representation": "final_prompt_token_ids_plus_final_response_token_ids",
        "input": str(root),
        "group_size": group_size,
        "selected_tasks": len(selected),
        "complete_groups": len(per_task),
        "excluded_groups": len(excluded),
        "included_trajectories": len(per_task) * group_size,
        "included_model_calls": included_calls,
        "recorded_model_calls": len(raw),
        "trajectory_status": dict(Counter(t["status"] for t in trajectories)),
        "cross_trajectory_sharing": aggregate,
        "excluded": excluded,
        "diagnostics": diagnostics,
        "definitions": {
            "baseline": "Sum the lengths of one complete token sequence per trajectory: final full-history prompt plus final response.",
            "comparison": "Build one prefix tree from exactly G complete trajectory sequences for each question; no sharing across questions.",
            "causal_pairs": "Sum of visible ancestor keys including self for every unique token node; full causal attention assumed.",
            "scope": "Only complete groups with successful, non-truncated, exact-token calls and all prior messages/actions preserved; global ratios use summed costs.",
            "loss": "Shared states retain separate per-trajectory loss multiplicities, advantages and clipping terms.",
            "limitation": "Structural work estimate, not measured training speedup or support already implemented in PrefixGrouper. Requires matching positions, masks and model state.",
        },
    }
    output = (args.output or root / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "trajectory_sequences.jsonl").open("w", encoding="utf-8") as handle:
        for trajectory in trajectory_sequences:
            handle.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
    fields = ["task_id", "trajectories", "model_calls", *aggregate]
    write_csv(output / "per_task.csv", per_task, fields)
    table = [
        {
            "方案": "每条完整轨迹独立计算（基线）",
            "token位置数": aggregate["separate_tokens"],
            "减少token位置数": 0,
            "token减少比例": 0 if per_task else "N/A",
            "token工作量缩减倍数": 1 if per_task else "N/A",
            "causal_attention_pairs": aggregate["separate_causal_pairs"],
            "attention_pair减少比例": 0 if per_task else "N/A",
        },
        {
            "方案": f"同题{group_size}条完整轨迹合并前缀树",
            "token位置数": aggregate["merged_tokens"],
            "减少token位置数": aggregate["saved_tokens"],
            "token减少比例": aggregate["token_reduction"],
            "token工作量缩减倍数": aggregate["token_work_ratio"],
            "causal_attention_pairs": aggregate["merged_causal_pairs"],
            "attention_pair减少比例": aggregate["causal_pair_reduction"],
        },
    ]
    write_csv(output / "benefit.csv", table, list(table[0]))

    def percent(value: float | None) -> str:
        return f"{value:.2%}" if value is not None else "N/A"

    ratio = aggregate["token_work_ratio"]
    ratio_text = f"{ratio:.2f}x" if ratio is not None else "N/A"
    report = f"""# 完整轨迹跨轨迹共享收益估算

采集目录：`{root}`。每题 {group_size} 条轨迹，完整组 {len(per_task)}/{len(selected)}；
纳入 {len(per_task) * group_size} 条轨迹、{included_calls} 次模型调用，排除 {len(excluded)} 个不完整或无效组。

| 方案 | token 位置数 | token 减少比例 | token 工作量缩减倍数 | causal attention pairs | pair 减少比例 |
|---|---:|---:|---:|---:|---:|
| 每条完整轨迹独立计算（基线） | {aggregate['separate_tokens']:,} | {'0%' if per_task else 'N/A'} | {'1.00x' if per_task else 'N/A'} | {aggregate['separate_causal_pairs']:,} | {'0%' if per_task else 'N/A'} |
| 同题 {group_size} 条完整轨迹合并前缀树 | {aggregate['merged_tokens']:,} | {percent(aggregate['token_reduction'])} | {ratio_text} | {aggregate['merged_causal_pairs']:,} | {percent(aggregate['causal_pair_reduction'])} |

统计单位固定为一条完整轨迹、一条 token 序列：初始问题、全部模型动作和工具结果、最终回复。
序列取最后一次完整历史请求的 prompt token ID 加最终 response token ID；逐轮检查历史消息和模型动作均被保留。
基线 = 同题 {group_size} 条完整序列的长度之和；共享后 = 这 {group_size} 条序列合成前缀树的节点数。
减少比例 = 1 − 共享后 / 基线。汇总使用工作量之和，非各题百分比的算术平均。

这是结构上的计算量估算，不是 NPU 训练耗时、显存节省或实测加速比，也不表示 PrefixGrouper 已支持该树。
共享还要求位置编码、attention mask 和模型状态一致；分叉后相同文本不重新合并。
各轨迹的 response loss、advantage、权重与 clipping 仍须分别保留并正确累加梯度。
工具结果作为上下文，不作为模型生成动作计算 policy loss。
这是按最终完整历史序列估计的训练结构；实际训练还需保证 tokenizer、loss mask 和 rollout log-prob 的位置映射一致。

详细数据见 `per_task.csv`、`benefit.csv`、`summary.json`；实际纳入的完整序列见 `trajectory_sequences.jsonl`。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved benefit table: {output / 'benefit.csv'}")


if __name__ == "__main__":
    main()
