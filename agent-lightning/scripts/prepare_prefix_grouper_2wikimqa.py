#!/usr/bin/env python3
"""Prepare the maintained 2WikiMQA dataset for the PrefixGrouper E2E benchmark.

Example:
    python scripts/prepare_prefix_grouper_2wikimqa.py \
        --input /data/2wikimqa.json \
        --tokenizer /models/Qwen3-30B-A3B-Instruct-2507 \
        --output /data/2wikimqa-2k.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, cast

from transformers import AutoTokenizer

DATASET_NAME = "2WikiMQA"
SYSTEM_PROMPT = (
    "Answer the question using only the supplied Wikipedia passages. "
    "Return only the shortest answer phrase, with no explanation."
)
MAX_PROMPT_TOKENS = 2048
MIN_PROMPT_TOKENS = 1900
MINIMUM_ROWS = 64


def parse_args() -> argparse.Namespace:
    """Parse dataset preparation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="2WikiMQA source JSON files.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file.")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Local Qwen tokenizer/model directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the destination if it already exists.")
    return parser.parse_args()


def render_context(row: dict[str, Any]) -> str:
    """Render Wikipedia documents with supporting documents first."""
    metadata = row["metadata"]
    context = metadata["context"]
    titles = [str(title) for title in context["title"]]
    contents = context["content"]
    supporting_titles = {str(title) for title in metadata["supporting_facts"]["title"]}

    documents: list[tuple[bool, str]] = []
    for title, sentences in zip(titles, contents, strict=True):
        text = " ".join(str(sentence) for sentence in sentences)
        documents.append((title not in supporting_titles, f"Title: {title}\n{text}"))
    documents.sort(key=lambda item: item[0])
    return "\n\n".join(document for _, document in documents)


def build_user_prompt(context: str, question: str) -> str:
    """Build the fixed 2WikiMQA user prompt."""
    return f"Wikipedia passages:\n\n{context}\n\nQuestion: {question}\nAnswer:"


def chat_length(tokenizer: Any, user_prompt: str) -> int:
    """Return the tokenized chat-template length."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(encoded, "keys"):
        return len(encoded["input_ids"])
    return len(encoded)


def fit_prompt(tokenizer: Any, context: str, question: str) -> tuple[str, int, int]:
    """Trim context from the end until the fixed prompt limit is satisfied."""
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    full_prompt = build_user_prompt(context, question)
    full_length = chat_length(tokenizer, full_prompt)
    if full_length <= MAX_PROMPT_TOKENS:
        return full_prompt, full_length, full_length

    low, high = 0, len(context_ids)
    best_prompt = build_user_prompt("", question)
    best_length = chat_length(tokenizer, best_prompt)
    while low <= high:
        midpoint = (low + high) // 2
        candidate_context = tokenizer.decode(context_ids[:midpoint], skip_special_tokens=True)
        candidate_prompt = build_user_prompt(candidate_context, question)
        candidate_length = chat_length(tokenizer, candidate_prompt)
        if candidate_length <= MAX_PROMPT_TOKENS:
            best_prompt = candidate_prompt
            best_length = candidate_length
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best_prompt, best_length, full_length


def load_source_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Load rows from the exported 2WikiMQA JSON representation."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        source_rows = payload.get("rows")
        if not isinstance(source_rows, list):
            raise ValueError(f"{path} does not contain a list-valued 'rows' field.")
        for item in cast(list[Any], source_rows):
            if not isinstance(item, dict):
                raise ValueError(f"{path} contains an invalid 2WikiMQA row wrapper.")
            wrapper = cast(dict[str, Any], item)
            if not isinstance(wrapper.get("row"), dict):
                raise ValueError(f"{path} contains an invalid 2WikiMQA row wrapper.")
            rows.append(cast(dict[str, Any], wrapper["row"]))
    return rows


def main() -> None:
    """Prepare and write the fixed-length 2WikiMQA JSONL dataset."""
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if not args.tokenizer.is_dir():
        raise FileNotFoundError(f"Tokenizer must be a local directory: {args.tokenizer}")

    tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(args.tokenizer, local_files_only=True)
    prepared: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for row in load_source_rows(args.input):
        sample_id = str(row["id"])
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate 2WikiMQA sample ID: {sample_id}")
        seen_sample_ids.add(sample_id)

        context = render_context(row)
        prompt, prompt_tokens, original_prompt_tokens = fit_prompt(tokenizer, context, str(row["question"]))
        if prompt_tokens < MIN_PROMPT_TOKENS:
            continue
        answers = [str(answer) for answer in row["golden_answers"]]
        if not answers:
            raise ValueError(f"2WikiMQA sample {sample_id} has no golden answer.")
        prepared.append(
            {
                "dataset": DATASET_NAME,
                "sample_id": sample_id,
                "prompt": prompt,
                "answers": answers,
                "prompt_tokens": prompt_tokens,
                "original_prompt_tokens": original_prompt_tokens,
            }
        )

    if len(prepared) < MINIMUM_ROWS:
        raise RuntimeError(f"Need at least {MINIMUM_ROWS} eligible 2WikiMQA prompts, found {len(prepared)}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in prepared:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    lengths = [int(row["prompt_tokens"]) for row in prepared]
    original_lengths = [int(row["original_prompt_tokens"]) for row in prepared]
    print(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "rows": len(prepared),
                "prompt_min": min(lengths),
                "prompt_mean": statistics.mean(lengths),
                "prompt_median": statistics.median(lengths),
                "prompt_max": max(lengths),
                "original_prompt_mean": statistics.mean(original_lengths),
                "original_prompt_max": max(original_lengths),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
