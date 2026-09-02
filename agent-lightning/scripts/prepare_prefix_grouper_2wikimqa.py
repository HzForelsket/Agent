#!/usr/bin/env python3
"""Prepare a token-filtered 2WikiMQA workload for the PrefixGrouper E2E benchmark.

Example:
    python scripts/prepare_prefix_grouper_2wikimqa.py \
        --tokenizer /models/Qwen3-8B \
        --output /data/2wikimqa-2k.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, cast

import httpx
import pyarrow.parquet as pq
from transformers import AutoTokenizer

DATASET_NAME = "2WikiMQA"
SYSTEM_PROMPT = (
    "Answer the question using only the supplied Wikipedia passages. "
    "Return only the shortest answer phrase, with no explanation."
)
DEFAULT_MAX_PROMPT_TOKENS = 2048
DEFAULT_MIN_PROMPT_TOKENS = 1900
MINIMUM_ROWS = 64
DATASET_REPOSITORY = "framolfese/2WikiMultihopQA"
DATASET_REVISION = "fe713bfbd1afbca1a65246741a75890405d56a3a"
DATASET_PARQUET_REVISION = "e8992ba04ceb4b6d144e1368cc5af38c5c632e7a"
DATASET_PARQUET_SHA256 = "408e2dbb28edc6c8b9ca3ba0c94d4fc7bf17ffb923766593a3a7f546ab4cba59"
DATASET_PARQUET_SIZE = 29505064
DATASET_CONFIG = "default"
DATASET_SPLIT = "validation"
DATASET_TOTAL_ROWS = 12576
DATASET_PARQUET_URL_ENV = "AGENTLIGHTNING_DATASET_PARQUET_URL"
DEFAULT_DATASET_PARQUET_URL = (
    "https://huggingface.co/datasets/framolfese/2WikiMultihopQA/resolve/"
    f"{DATASET_PARQUET_REVISION}/default/validation/0000.parquet"
)
DEFAULT_DOWNLOAD_DIR = Path(".cache/pg-2wikimqa-e2e/datasets")


def parse_args() -> argparse.Namespace:
    """Parse dataset preparation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", help="Existing pinned 2WikiMQA Parquet source files.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file.")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Local Qwen tokenizer/model directory.")
    parser.add_argument("--min-prompt-tokens", type=int, default=DEFAULT_MIN_PROMPT_TOKENS)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--local-files-only", action="store_true", help="Require an existing source cache.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the destination if it already exists.")
    return parser.parse_args()


def render_context(row: dict[str, Any]) -> str:
    """Render Wikipedia documents with supporting documents first."""
    context = row["context"]
    titles = [str(title) for title in context["title"]]
    contents = context["sentences"]
    supporting_titles = {str(title) for title in row["supporting_facts"]["title"]}

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


def dataset_source() -> dict[str, str | int]:
    """Return the fixed source identity used by the maintained workload."""
    return {
        "repository": DATASET_REPOSITORY,
        "revision": DATASET_REVISION,
        "parquet_revision": DATASET_PARQUET_REVISION,
        "parquet_sha256": DATASET_PARQUET_SHA256,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "rows": "all",
        "total_rows": DATASET_TOTAL_ROWS,
    }


def source_cache_path(download_dir: Path) -> Path:
    """Return the deterministic cache path for the pinned source rows."""
    return download_dir / f"2wikimqa-{DATASET_SPLIT}-{DATASET_REVISION[:12]}-full.parquet"


def _validate_source_file(path: Path) -> None:
    if path.stat().st_size != DATASET_PARQUET_SIZE:
        raise ValueError(f"{path} does not match the pinned 2WikiMQA Parquet size.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != DATASET_PARQUET_SHA256:
        raise ValueError(f"{path} does not match the pinned 2WikiMQA Parquet SHA-256.")
    if pq.ParquetFile(path).metadata.num_rows != DATASET_TOTAL_ROWS:
        raise ValueError(f"{path} does not contain exactly {DATASET_TOTAL_ROWS} rows.")


def download_source_rows(download_dir: Path, *, local_files_only: bool) -> Path:
    """Materialize the complete pinned validation Parquet into an atomic cache."""
    cache_path = source_cache_path(download_dir).resolve()
    if cache_path.is_file():
        _validate_source_file(cache_path)
        return cache_path
    if local_files_only:
        raise FileNotFoundError(f"本地 2WikiMQA 源数据缓存不存在：{cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = cache_path.with_suffix(f"{cache_path.suffix}.partial")
    parquet_url = os.environ.get(DATASET_PARQUET_URL_ENV, DEFAULT_DATASET_PARQUET_URL)
    resume_offset = partial_path.stat().st_size if partial_path.is_file() else 0
    if resume_offset == DATASET_PARQUET_SIZE:
        try:
            _validate_source_file(partial_path)
        except ValueError:
            resume_offset = 0
        else:
            os.replace(partial_path, cache_path)
            return cache_path
    can_resume = 0 < resume_offset < DATASET_PARQUET_SIZE
    headers = {"Range": f"bytes={resume_offset}-"} if can_resume else {}
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        with client.stream("GET", parquet_url, headers=headers) as response:
            response.raise_for_status()
            append = can_resume and response.status_code == 206
            with partial_path.open("ab" if append else "wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
    _validate_source_file(partial_path)
    os.replace(partial_path, cache_path)
    return cache_path


def load_source_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Load rows from the pinned local 2WikiMQA representation."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        _validate_source_file(path)
        source_rows = pq.read_table(path).to_pylist()
        if not all(isinstance(row, dict) for row in source_rows):
            raise ValueError(f"{path} contains an invalid 2WikiMQA source row.")
        rows.extend(cast(list[dict[str, Any]], source_rows))
    return rows


def prepare_dataset(
    source_paths: list[Path],
    tokenizer_path: Path,
    output: Path,
    *,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Filter full, untruncated prompts by tokenized input length."""
    if min_prompt_tokens <= 0 or max_prompt_tokens <= 0:
        raise ValueError("Prompt token bounds must be positive.")
    if min_prompt_tokens > max_prompt_tokens:
        raise ValueError("Minimum prompt tokens cannot exceed maximum prompt tokens.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Tokenizer must be a local directory: {tokenizer_path}")

    tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(tokenizer_path, local_files_only=True)
    prepared: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for row in load_source_rows(source_paths):
        sample_id = str(row["id"])
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate 2WikiMQA sample ID: {sample_id}")
        seen_sample_ids.add(sample_id)

        context = render_context(row)
        prompt = build_user_prompt(context, str(row["question"]))
        prompt_tokens = chat_length(tokenizer, prompt)
        if not min_prompt_tokens <= prompt_tokens <= max_prompt_tokens:
            continue
        answers = [str(row["answer"])]
        if not answers[0]:
            raise ValueError(f"2WikiMQA sample {sample_id} has no golden answer.")
        prepared.append(
            {
                "dataset": DATASET_NAME,
                "sample_id": sample_id,
                "prompt": prompt,
                "answers": answers,
                "prompt_tokens": prompt_tokens,
                "input_policy": "filter-untruncated",
            }
        )

    if len(prepared) < MINIMUM_ROWS:
        raise RuntimeError(f"Need at least {MINIMUM_ROWS} eligible 2WikiMQA prompts, found {len(prepared)}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output.with_suffix(f"{output.suffix}.partial")
    with partial_path.open("w", encoding="utf-8") as stream:
        for row in prepared:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(partial_path, output)

    lengths = [int(row["prompt_tokens"]) for row in prepared]
    return {
        "dataset": DATASET_NAME,
        "source": dataset_source(),
        "rows": len(prepared),
        "prompt_min": min(lengths),
        "prompt_mean": statistics.mean(lengths),
        "prompt_median": statistics.median(lengths),
        "prompt_max": max(lengths),
        "input_policy": "filter-untruncated",
        "output": str(output.resolve()),
    }


def main() -> None:
    """Prepare and write the configured 2WikiMQA JSONL workload."""
    args = parse_args()
    source_paths = args.input or [download_source_rows(args.download_dir, local_files_only=args.local_files_only)]
    summary = prepare_dataset(
        source_paths,
        args.tokenizer,
        args.output,
        min_prompt_tokens=args.min_prompt_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
