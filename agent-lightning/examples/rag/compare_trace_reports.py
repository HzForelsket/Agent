# Copyright (c) Microsoft. All rights reserved.

"""Summarize SQL/Q20 reports: --inputs SQL_ANALYSIS Q20_ANALYSIS --output COMPARISON_DIR."""

import argparse
import json
from pathlib import Path

from analyze_call_traces import UNIT
from analyze_traces import write_csv


def main() -> None:
    """Keep per-workload denominators and coverage visible; do not claim measured training speed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    summaries = []
    for path in args.inputs:
        root = path.resolve()
        if (root / "config.json").exists():
            root = root / "analysis"
        summary = json.loads((root / "summary.json").read_text())
        if summary.get("statistics_unit") != UNIT:
            raise ValueError(f"{root} has a different statistics unit; do not mix RAG flattened-sequence reports")
        summaries.append({"source": str(root), "summary": summary})
        simple, tree, extra = (
            summary[key] for key in ("simple_sharing", "cross_trajectory_sharing", "tree_over_simple")
        )
        rows.append(
            {
                "agent": summary["agent"],
                "model": summary["model"],
                "group_size": summary["group_size"],
                "complete_groups": summary["complete_groups"],
                "selected_tasks": summary["selected_tasks"],
                "policy_calls": summary["included_model_calls"],
                "baseline_tokens": tree["separate_tokens"],
                "simple_tokens": simple["merged_tokens"],
                "simple_reduction": simple["token_reduction"],
                "tree_tokens": tree["merged_tokens"],
                "tree_reduction": tree["token_reduction"],
                "tree_over_simple_reduction": extra["token_reduction"],
                "baseline_pairs": tree["separate_causal_pairs"],
                "simple_pairs": simple["merged_causal_pairs"],
                "tree_pairs": tree["merged_causal_pairs"],
                "environment_calls": summary["environment_model_calls"],
            }
        )
    output = args.output.resolve()
    if output in {Path(item["source"]) for item in summaries} or (output / "config.json").exists():
        raise ValueError("Comparison output must be separate from input reports and raw traces")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparison.csv", rows, list(rows[0]))
    (output / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")

    def pct(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2%}"

    lines = [
        "# SQL / 20 Questions 原流程轨迹共享对比",
        "",
        "同题同角色按调用序号配对；不修改原流程，不做轨迹内部去重。",
        "",
        "| Agent | 模型 | 每题轨迹数 | 完整题组 | 独立 token | 简单共享 token（减少） | 前缀树 token（减少） | 树相对简单共享额外减少 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['agent']} | {row['model']} | {row['group_size']} | {row['complete_groups']}/{row['selected_tasks']} | {row['baseline_tokens']:,} | {row['simple_tokens']:,}（{pct(row['simple_reduction'])}） | {row['tree_tokens']:,}（{pct(row['tree_reduction'])}） | {pct(row['tree_over_simple_reduction'])} |"
        )
    lines.extend(
        [
            "",
            "各行使用自己的完整题组和基线，不跨任务合树或混合百分比。模型、数据、调用数量不同会影响分布，不能据此判断哪个 Agent 更快。",
            "这里只汇总结构工作量；未测量训练耗时、反向传播、通信或显存节省。Q20 回答者/搜索成本单列，不计入 policy 收益。",
            "完整来源、采集设置和排除原因保存在 comparison.json；attention pair 计数见 comparison.csv。",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
