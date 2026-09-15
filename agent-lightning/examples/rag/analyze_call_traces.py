# Copyright (c) Microsoft. All rights reserved.

"""Compare sharing without changing original workflows: --input RUN_OR_ANALYSIS_DIR [--output DIR]."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_traces import comparisons, read_records, sharing_costs, valid_call, write_csv

UNIT = "original_workflow_call_slots"
COSTS = (
    "separate_tokens",
    "separate_causal_pairs",
    "common_prefix_tokens",
    "simple_tokens",
    "simple_causal_pairs",
    "merged_tokens",
    "merged_causal_pairs",
)


def raw_sequences(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select complete groups, preserving every policy call as an independent context."""
    config = json.loads((root / "config.json").read_text())
    if config.get("statistics_unit") != UNIT:
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
        "statistics_unit": UNIT,
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


def analyze(root: Path, output: Path) -> None:
    """Write per-task and aggregate reports with one sequence per real call, never concatenate histories."""
    if (root / "config.json").is_file():
        records, context = raw_sequences(root)
    else:
        context = json.loads((root / "summary.json").read_text())
        if context.get("statistics_unit") != UNIT:
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
    table = []
    for label, result in (
        ("独立调用（完整轨迹汇总基线）", None),
        ("简单共享（同调用位置的全组公共前缀）", results["simple_sharing"]),
        ("前缀树共享（仅同调用位置跨轨迹）", results["cross_trajectory_sharing"]),
    ):
        table.append(
            {
                "方案": label,
                "token位置数": result["merged_tokens"] if result else totals["separate_tokens"],
                "token减少比例": result["token_reduction"] if result else (0.0 if per_task else None),
                "causal_attention_pairs": result["merged_causal_pairs"] if result else totals["separate_causal_pairs"],
                "pair减少比例": result["causal_pair_reduction"] if result else (0.0 if per_task else None),
            }
        )

    def percent(value: Any) -> str:
        return "N/A" if value is None else f"{value:.2%}"

    lines = [
        f"| {row['方案']} | {row['token位置数']:,} | {percent(row['token减少比例'])} | {row['causal_attention_pairs']:,} | {percent(row['pair减少比例'])} |"
        for row in table
    ]
    extra = results["tree_over_simple"]
    report = f"""# 原流程多上下文轨迹共享对比：{context['agent']}

模型：`{context['model']}`；原流程：`{context['workflow_implementation']}`。
输入：`{root}`；每题 {size} 条轨迹，完整组 {context['complete_groups']}/{context['selected_tasks']}，
纳入 {context['included_trajectories']} 条完整轨迹、{len(records)} 次 policy 调用。
原始流程结束但答案错误的轨迹仍计为完成；失败、缺失或长度截断的组排除。

| 方案 | token 位置数 | 相对独立基线减少 | attention pairs | 相对独立基线减少 |
|---|---:|---:|---:|---:|
{chr(10).join(lines)}

前缀树相对简单共享额外减少 {extra['saved_tokens']:,} 个 token 位置（{percent(extra['token_reduction'])}），
额外减少 {extra['separate_causal_pairs'] - extra['merged_causal_pairs']:,} 个 attention pairs（{percent(extra['causal_pair_reduction'])}）。

每条轨迹保留原流程的多个模型上下文。每次调用使用真实 prompt+response token，工作量累加到完整轨迹。
只对齐同题 policy 角色的第 k 次调用：每个位置每条轨迹至多贡献一条序列；不跨位置共享，不做轨迹内部去重。
某条轨迹没有第 k 次调用时视为空序列，因此该位置的全组公共前缀为零；树仍可共享剩余轨迹的前缀。
序号配对不保证 SQL 语义阶段或 Q20 游戏轮次相同，也不声称达到所有跨调用配对的最大共享收益。
简单共享节省 (G−1)×公共前缀长度；pairs 按完整 causal attention 的三角数估算，各位置分别求和。
Q20 的 Answerer/Search 共记录 {context['environment_model_calls']} 次调用，有效调用 token 合计 {context['valid_environment_tokens']:,}，单独保存，不计入 policy 收益。
不会修改原有提示词、历史重建、检查修正或终止条件；模型部署和采样条件以 config.json/calls.jsonl 为准。

这与 RAG 的“最后一次输入＋输出代表一条完整序列”口径不同，不直接横向排名。
结果是结构工作量估算，不是 NPU 实测训练加速；共享仍要求模型、位置和 mask 一致，并分别保留 loss/advantage 权重。
验证来源：`{context['validation_source']}`。导出目录输入复用此前筛选结果，不能重新验证缺失的原始调用。
"""
    if (output / "config.json").exists() or (output / "calls.jsonl").exists():
        raise ValueError("Output must not overwrite a raw collection directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (output / "call_sequences.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(
        output / "per_task.csv",
        per_task,
        list(per_task[0]) if per_task else ["task_id", "trajectories", "model_calls", "call_slots", *COSTS],
    )
    write_csv(output / "benefit.csv", table, list(table[0]))
    (output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.input.resolve()
    analyze(root, (args.output or (root / "analysis" if (root / "config.json").is_file() else root)).resolve())
