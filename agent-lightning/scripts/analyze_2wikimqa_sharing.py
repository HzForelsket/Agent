# Copyright (c) Microsoft. All rights reserved.

"""Analyze single-turn 2WikiMQA token positions and causal attention pairs offline.

Usage (one run, one model/tokenizer, and one sampling group per sample_id):
    python scripts/analyze_2wikimqa_sharing.py --input responses.jsonl \
        --group-size 8 --sharing prompt --assume-identical-prompts --output analysis
    python scripts/analyze_2wikimqa_sharing.py --input token_records.jsonl \
        --group-size 8 --sharing tree --output analysis

Each JSONL row requires sample_id, rollout_index (0..G-1), and finish_reason
('stop' or 'length'). Supply prompt_token_ids and response_token_ids for exact
analysis. Without IDs, prompt mode requires positive prompt_tokens and
response_tokens plus explicit --assume-identical-prompts. Counts must describe
the actual model context, including chat-template tokens. Text is never retokenized.
Optional count fields are checked against IDs when both are present.

Prompt sharing saves only the identical input, keeping every output independent.
Tree sharing additionally merges exact output prefixes. Both use the same input
records as the independent baseline, never independently sampled comparison runs.
Repeated sample IDs from different steps/runs must be separated before analysis;
duplicate rollout indices fail instead of being silently combined.

Length-terminated outputs remain included as observed workloads and are counted
in the report; their lengths do not describe unrestricted answer generation.
Old response exports without output token counts cannot be analyzed accurately.
No model, training, tokenizer download, or accelerator is needed.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def triangle(length: int) -> int:
    """Count full causal query/key pairs, including the diagonal."""
    return length * (length + 1) // 2


def token_ids(value: Any, name: str) -> list[int]:
    """Require nonempty exact token IDs, rejecting booleans and negative IDs."""
    if not isinstance(value, list) or not value or any(type(t) is not int or t < 0 for t in value):
        raise ValueError(f"{name} must be a nonempty list of nonnegative integer IDs")
    return value


def positive_int(value: Any, name: str) -> int:
    """Validate an actual token count without coercion."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer; text lengths are not token counts")
    return value


def read_groups(args: argparse.Namespace) -> dict[str, dict[int, dict[str, Any]]]:
    """Read one sampling group per question and validate records strictly."""
    groups: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Each row must be a JSON object")
                sample = row.get("sample_id")
                index = row.get("rollout_index")
                if not isinstance(sample, str) or not sample:
                    raise ValueError("sample_id must be a nonempty string")
                if type(index) is not int or not 0 <= index < args.group_size:
                    raise ValueError("rollout_index must be in 0..group-size-1")
                if index in groups[sample]:
                    raise ValueError(f"Duplicate sample/index: {sample}/{index}; separate repeated groups")
                if row.get("finish_reason") not in ("stop", "length"):
                    raise ValueError("finish_reason must be stop or length for a single-turn answer")
                has_ids = "prompt_token_ids" in row or "response_token_ids" in row
                if has_ids:
                    for part in ("prompt", "response"):
                        ids = token_ids(row.get(f"{part}_token_ids"), f"{part}_token_ids")
                        field = f"{part}_tokens"
                        if field in row and positive_int(row[field], field) != len(ids):
                            raise ValueError(f"{field} disagrees with exact token IDs")
                        row[field] = len(ids)
                else:
                    if args.sharing == "tree" or not args.assume_identical_prompts:
                        raise ValueError(
                            "Exact token IDs required; count-only prompt sharing needs " "--assume-identical-prompts"
                        )
                    for part in ("prompt", "response"):
                        positive_int(row.get(f"{part}_tokens"), f"{part}_tokens")
                groups[sample][index] = row
            except (ValueError, TypeError) as error:
                raise ValueError(f"{args.input}:{line_number}: {error}") from error
    if not groups:
        raise ValueError("Input contains no records")
    return groups


def analyze_group(sample: str, rows: list[dict[str, Any]], sharing: str) -> dict[str, Any]:
    """Compute independent and shared structural work on identical sequences."""
    prompt = rows[0]["prompt_tokens"]
    if any(row["prompt_tokens"] != prompt for row in rows):
        raise ValueError(f"{sample}: prompt lengths differ within the group")
    known_prompts = [row["prompt_token_ids"] for row in rows if "prompt_token_ids" in row]
    if known_prompts and any(ids != known_prompts[0] for ids in known_prompts):
        raise ValueError(f"{sample}: prompt token IDs differ within the group")
    lengths = [prompt + row["response_tokens"] for row in rows]
    baseline_tokens = sum(lengths)
    baseline_pairs = sum(map(triangle, lengths))
    if sharing == "prompt":
        shared_tokens = baseline_tokens - (len(rows) - 1) * prompt
        shared_pairs = baseline_pairs - (len(rows) - 1) * triangle(prompt)
    else:
        sequences = sorted(row["prompt_token_ids"] + row["response_token_ids"] for row in rows)
        previous: list[int] = []
        shared_tokens = shared_pairs = 0
        for sequence in sequences:
            common = 0
            for left, right in zip(previous, sequence):
                if left != right:
                    break
                common += 1
            shared_tokens += len(sequence) - common
            shared_pairs += triangle(len(sequence)) - triangle(common)
            previous = sequence
    return {
        "sample_id": sample,
        "sequences": len(rows),
        "length_terminated": sum(row["finish_reason"] == "length" for row in rows),
        "count_only_sequences": sum("prompt_token_ids" not in row for row in rows),
        "baseline_tokens": baseline_tokens,
        "shared_tokens": shared_tokens,
        "baseline_causal_pairs": baseline_pairs,
        "shared_causal_pairs": shared_pairs,
        "token_reduction": 1 - shared_tokens / baseline_tokens,
        "causal_pair_reduction": 1 - shared_pairs / baseline_pairs,
    }


def main() -> None:
    """Write per-question CSV, aggregate JSON and a human-readable report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--sharing", choices=("prompt", "tree"), required=True)
    parser.add_argument(
        "--assume-identical-prompts",
        action="store_true",
        help="Explicitly assert identical actual prompts per sample when only counts exist",
    )
    args = parser.parse_args()
    if args.group_size < 2:
        parser.error("--group-size must be >= 2")
    if args.sharing == "tree" and args.assume_identical_prompts:
        parser.error("Tree sharing requires exact IDs; do not supply --assume-identical-prompts")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory")
    try:
        groups = read_groups(args)
        excluded = [
            {"sample_id": sample, "reason": "incomplete_group", "records": len(rows)}
            for sample, rows in groups.items()
            if len(rows) != args.group_size
        ]
        results = [
            analyze_group(sample, [rows[i] for i in range(args.group_size)], args.sharing)
            for sample, rows in sorted(groups.items())
            if len(rows) == args.group_size
        ]
        if not results:
            raise ValueError(f"No complete groups; incomplete groups: {len(excluded)}")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    keys = (
        "sequences",
        "length_terminated",
        "count_only_sequences",
        "baseline_tokens",
        "shared_tokens",
        "baseline_causal_pairs",
        "shared_causal_pairs",
    )
    totals = {key: sum(row[key] for row in results) for key in keys}
    totals["token_reduction"] = 1 - totals["shared_tokens"] / totals["baseline_tokens"]
    totals["causal_pair_reduction"] = 1 - totals["shared_causal_pairs"] / totals["baseline_causal_pairs"]
    summary = {
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "sharing": args.sharing,
        "group_size": args.group_size,
        "assume_identical_prompts": args.assume_identical_prompts,
        "complete_groups": len(results),
        "observed_groups": len(groups),
        "excluded_groups": excluded,
        **totals,
        "mean_baseline_tokens_per_sequence": totals["baseline_tokens"] / totals["sequences"],
        "definitions": {
            "tokens": "Actual prompt+response positions; no padding, no loss-mask filtering.",
            "pairs": "Sum L*(L+1)/2; includes diagonal, not multiplied by heads or layers.",
            "prompt": "Save (G-1)*P positions and (G-1)*P*(P+1)/2 pairs per group.",
            "tree": "Unique trie nodes; each node at 1-based depth d contributes d causal pairs.",
            "scope": "One single-turn sampling group per sample; no cross-question sharing.",
            "limitation": "Structural estimate, not runtime speedup or measured PrefixGrouper execution. "
            "Sharing requires identical model, positions and masks; preserve per-sample loss weights.",
        },
    }
    report = f"""# 2WikiMQA 单轮问答共享统计

输入：`{args.input.resolve()}`；共享方式：`{args.sharing}`；每题 {args.group_size} 条样本。
完整组 {len(results)}/{len(groups)}；纳入 {totals['sequences']} 条；排除 {len(excluded)} 个不完整组。

| 方案 | token 位置数 | causal attention pairs |
|---|---:|---:|
| 独立基线 | {totals['baseline_tokens']:,} | {totals['baseline_causal_pairs']:,} |
| 共享 | {totals['shared_tokens']:,} | {totals['shared_causal_pairs']:,} |
| 减少比例 | {totals['token_reduction']:.2%} | {totals['causal_pair_reduction']:.2%} |

单条平均基线长度：{summary['mean_baseline_tokens_per_sequence']:.2f} token。
length 结束的样本数：{totals['length_terminated']}，已计入实际工作量，不代表自然完成长度。
仅有长度、依赖用户相同 prompt 声明的样本数：{totals['count_only_sequences']}。

基线每条长度 L = prompt + response，pairs = L×(L+1)/2，包含对角线。
prompt 模式只共享输入；tree 模式共享完整序列的精确前缀，各分支互不可见。
不跨题共享，不计 padding，不按 loss mask 删除上下文，也不乘层数或头数。
这是结构工作量估算，不是实测加速比；共享要求模型、位置和 mask 一致，并保留各样本 loss 权重。
逐题数据见 per_task.csv，排除项和计算定义见 summary.json。
"""
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "per_task.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
