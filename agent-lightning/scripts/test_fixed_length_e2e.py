#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Prepare fixed-length chat inputs, then run the shared 2WikiMQA E2E pipeline.

    python scripts/test_fixed_length_e2e.py --mode simple --model /path/to/model \
        --prompt-tokens 128 --response-tokens 32 --output-dir /tmp/fixed-length-e2e

Input lengths include the chat template. Output length is exact, with EOS ignored.
``--prepare-only --device gpu`` writes the dataset and resolved configuration
without loading model weights, starting Ray, or running training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

BENCHMARK_ID = "fixed-length-e2e"
DATASET_NAME = "FixedLength"


def write_json(path: Path, value: Any) -> None:
    """Persist run configuration and completion state."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse dataset lengths and the same resource defaults as the Wiki entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("no-share", "simple"), default="no-share")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True, help="Exact input length including chat template.")
    parser.add_argument("--response-tokens", type=int, required=True, help="Exact generated length; EOS is ignored.")
    parser.add_argument("--dataset-size", type=int, default=64)
    parser.add_argument("--steps", type=int, help="Defaults to one full dataset epoch, as in the Wiki entrypoint.")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--rollouts-per-sample", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--n-devices", type=int, help="Defaults to the number of visible devices.")
    parser.add_argument("--tensor-parallel-size", type=int, help="Defaults to --n-devices.")
    parser.add_argument("--n-runners", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "gpu", "npu"), default="auto")
    parser.add_argument(
        "--prefix-timing", action="store_true", help="Record shared-prefix operation timings in simple mode."
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output-dir", type=Path, required=True, help="New run directory; never overwrites a run.")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for name in (
        "prompt_tokens",
        "response_tokens",
        "dataset_size",
        "steps",
        "train_batch_size",
        "rollouts_per_sample",
        "micro_batch_size",
        "n_devices",
        "tensor_parallel_size",
        "n_runners",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.dataset_size < args.train_batch_size:
        parser.error("--dataset-size must be at least --train-batch-size")
    if args.rollouts_per_sample < 2:
        parser.error("GRPO requires --rollouts-per-sample >= 2")
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")
    args.model = args.model.expanduser().resolve()
    if not args.model.is_dir():
        parser.error("--model must point to an existing local pretrained model directory")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def prepare_dataset(args: argparse.Namespace) -> Path:
    """Generate exact-length synthetic QA chats in the Wiki dataset schema."""
    from benchmark_prefix_grouper_2wikimqa_e2e import SYSTEM_PROMPT
    from prepare_prefix_grouper_2wikimqa import build_user_prompt, chat_length
    from transformers import AutoTokenizer

    tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(args.model, local_files_only=True)
    dataset_path = args.output_dir / "dataset.jsonl"
    with dataset_path.open("x", encoding="utf-8") as stream:
        for sample_id in range(args.dataset_size):
            answer = ("river", "forest", "mountain", "ocean")[sample_id % 4]
            context = f"Title: Record {sample_id}\nThe answer word in this record is {answer}.\n"

            def prompt_with_padding(count: int) -> str:
                return build_user_prompt(context + " x" * count, "What is the answer word in this record?")

            minimum_length = chat_length(tokenizer, prompt_with_padding(0))
            if minimum_length > args.prompt_tokens:
                raise ValueError(
                    f"Sample {sample_id} needs at least {minimum_length} prompt tokens including the chat template."
                )
            lower, upper = 0, args.prompt_tokens
            while lower <= upper:
                count = (lower + upper) // 2
                prompt = prompt_with_padding(count)
                length = chat_length(tokenizer, prompt)
                if length == args.prompt_tokens:
                    break
                if length < args.prompt_tokens:
                    lower = count + 1
                else:
                    upper = count - 1
            else:
                raise ValueError(f"Tokenizer cannot render sample {sample_id} at exactly {args.prompt_tokens} tokens.")
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
            if hasattr(prompt_ids, "keys"):
                prompt_ids = prompt_ids["input_ids"]
            row = {
                "dataset": DATASET_NAME,
                "sample_id": str(sample_id),
                "prompt": prompt,
                "answers": [answer],
                "prompt_tokens": len(prompt_ids),
                "prompt_token_ids": [int(token) for token in prompt_ids],
                "input_policy": "filter-untruncated",
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
    return dataset_path


def wiki_arguments(args: argparse.Namespace, dataset_path: Path) -> argparse.Namespace:
    """Translate the fixed-length input options to the common execution contract."""
    return argparse.Namespace(
        **{
            **vars(args),
            "mode": "prefix_grouper" if args.mode == "simple" else "baseline",
            "model": str(args.model),
            "model_ref": str(args.model),
            "model_name": "fixed-length-e2e",
            "dataset_path": dataset_path,
            "dataset_source": {
                "kind": "synthetic-fixed-length-chat",
                "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            },
            "min_prompt_tokens": args.prompt_tokens,
            "max_prompt_tokens": args.prompt_tokens,
            "max_response_tokens": args.response_tokens,
            "micro_batch_size_per_device": args.micro_batch_size,
            "n_devices_per_node": args.n_devices,
            "tensor_model_parallel_size": args.tensor_parallel_size,
            "npu_attention_backend": "fusion",
        }
    )


def run(args: argparse.Namespace) -> None:
    """Reuse Wiki configuration, agent, trainer, rollout path, and metric recording."""
    import benchmark_prefix_grouper_2wikimqa_e2e as wiki
    from omegaconf import OmegaConf

    import agentlightning as agl
    from agentlightning.verl.accelerator import select_accelerator

    dataset_path = prepare_dataset(args)
    common_args = wiki_arguments(args, dataset_path)
    runtime = None if args.prepare_only else select_accelerator(args.device)
    backend = wiki.requested_backend(args.device) if runtime is None else runtime.backend
    available_devices = None if runtime is None else int(runtime.module.device_count())
    stack = wiki.installed_stack(backend)
    wiki.check_stack(backend, stack)
    devices, tensor_parallel_size = wiki.resolve_resources(common_args, available_devices)
    dataset = wiki.load_dataset(
        dataset_path,
        args.seed,
        args.prompt_tokens,
        args.prompt_tokens,
        dataset_name=DATASET_NAME,
        benchmark_id=BENCHMARK_ID,
    )
    wiki.resolve_workload_limits(common_args, dataset, exact_response_length=True)
    config = wiki.build_config(
        args=common_args,
        backend=backend,
        n_devices_per_node=devices,
        tensor_model_parallel_size=tensor_parallel_size,
        metrics_path=args.output_dir / "metrics.jsonl",
        dataset_size=len(dataset),
        benchmark_id=BENCHMARK_ID,
    )
    write_json(args.output_dir / "config.json", config)
    if args.prepare_only:
        # Wiki dry-run also resolves VERL configuration without creating workers.
        algorithm = agl.VERL(config, trainer_cls=wiki.BenchmarkTrainer)
        write_json(args.output_dir / "resolved_config.json", OmegaConf.to_container(algorithm.config, resolve=True))
        write_json(args.output_dir / "result.json", {"status": "prepared", "training_executed": False})
        print(f"PREPARED: {dataset_path}; training was not executed.", flush=True)
        return
    if runtime is None:
        raise RuntimeError("Training requires an accelerator runtime.")
    metrics_path, responses_path = wiki.prepare_outputs(args.output_dir, overwrite=False)
    wiki.run_benchmark(
        common_args,
        runtime,
        dataset,
        config,
        metrics_path,
        responses_path,
        stack,
        benchmark_id=BENCHMARK_ID,
        dataset_name=DATASET_NAME,
        response_policy="exact-tokens",
    )
    # Length/completeness checks run after the timed training pipeline.
    from report_fixed_length_e2e import load_fixed_artifacts

    load_fixed_artifacts(args.output_dir, common_args.mode)
    write_json(
        args.output_dir / "result.json",
        {"status": "passed", "mode": args.mode, "steps": common_args.steps, "response_tokens": args.response_tokens},
    )
    print(f"PASSED: {args.output_dir / 'result.json'}", flush=True)


def main() -> None:
    """Persist preparation metadata and retain completed results on failure."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT_DIR={args.output_dir}", flush=True)
    try:
        versions = {}
        for package in ("agentlightning", "torch", "transformers", "verl", "vllm", "ray"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "missing"
        write_json(
            args.output_dir / "run.json",
            {
                "argv": sys.argv,
                "executable": sys.executable,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "platform": platform.platform(),
                "versions": versions,
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            },
        )
        run(args)
    except BaseException as error:
        completed = any(
            path.exists() and path.stat().st_size
            for path in (args.output_dir / "responses.jsonl", args.output_dir / "metrics.jsonl")
        )
        if completed:
            write_json(
                args.output_dir / "failure.json",
                {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()},
            )
            print(f"FAILED: partial results retained in {args.output_dir}", file=sys.stderr, flush=True)
        else:
            shutil.rmtree(args.output_dir)
            print("FAILED before any rollout/step completed; removed failed output directory.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
