#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Compare fixed-length E2E runs using the Wiki report's metrics and windows.

    python scripts/report_fixed_length_e2e.py --no-share-dir /runs/no-share \
        --simple-dir /runs/simple --output-dir /runs/comparison
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from report_prefix_grouper_2wikimqa_e2e import (
    COMPARABLE_RUN_FIELDS,
    RunArtifacts,
    build_report,
    load_artifacts,
    prepare_outputs,
    render_markdown,
)
from test_fixed_length_e2e import BENCHMARK_ID, DATASET_NAME


def load_fixed_artifacts(directory: Path, mode: str) -> RunArtifacts:
    """Check complete Wiki-format results against the prepared exact token IDs."""
    if (directory / "failure.json").exists():
        raise ValueError(f"{directory}: failed or partial runs cannot be compared")
    artifacts = load_artifacts(directory, mode, benchmark_id=BENCHMARK_ID)
    run = artifacts.run
    prompt_length = run.get("min_prompt_tokens")
    response_length = run.get("max_response_tokens")
    if (
        run.get("dataset") != DATASET_NAME
        or run.get("response_policy") != "exact-tokens"
        or type(prompt_length) is not int
        or prompt_length <= 0
        or prompt_length != run.get("max_prompt_tokens")
        or type(response_length) is not int
        or response_length <= 0
    ):
        raise ValueError(f"{directory}: expected a fixed input and exact output length workload")
    dataset_path = Path(run["dataset_path"])
    source = dataset_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != run["dataset_source"].get("sha256"):
        raise ValueError(f"{dataset_path}: prepared dataset no longer matches the run")
    rows = [json.loads(line) for line in source.decode("utf-8").splitlines() if line.strip()]
    prompts = {row["sample_id"]: row["prompt_token_ids"] for row in rows}
    if len(rows) != run["dataset_rows"] or len(prompts) != len(rows):
        raise ValueError(f"{dataset_path}: incomplete or duplicate dataset rows")
    if any(row["prompt_tokens"] != prompt_length or len(row["prompt_token_ids"]) != prompt_length for row in rows):
        raise ValueError(f"{dataset_path}: input lengths differ from the requested length")
    for response in artifacts.responses:
        tokens = response["response_token_ids"]
        if (
            response["prompt_token_ids"] != prompts.get(response["sample_id"])
            or response["prompt_tokens"] != prompt_length
            or response["response_tokens"] != response_length
            or len(tokens) != response_length
            or not all(type(token) is int for token in tokens)
            or response["request_max_tokens"] != response_length
            or response["finish_reason"] != "length"
        ):
            raise ValueError(f"{directory}: serving token IDs do not satisfy the exact length contract")
    return artifacts


def main() -> None:
    """Use the shared Wiki comparison and rendering after verifying input identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-share-dir", type=Path, required=True)
    parser.add_argument("--simple-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.no_share_dir.resolve() == args.simple_dir.resolve():
        parser.error("The two modes must come from separate run directories")
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    try:
        baseline = load_fixed_artifacts(args.no_share_dir, "baseline")
        simple = load_fixed_artifacts(args.simple_dir, "prefix_grouper")
        # Each run prepares its own dataset file. The verified content hash is
        # compared in dataset_source; differing output locations are immaterial.
        fields = tuple(field for field in COMPARABLE_RUN_FIELDS if field != "dataset_path")
        report = build_report(baseline, simple, comparable_run_fields=fields)
        markdown = render_markdown(report)
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    json_path, markdown_path = prepare_outputs(args.output_dir, overwrite=False)
    json_path.write_text(serialized, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"REPORT_JSON={json_path}")
    print(f"REPORT_MARKDOWN={markdown_path}")


if __name__ == "__main__":
    main()
