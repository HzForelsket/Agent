#!/usr/bin/env python3
"""Run the maintained 2WikiMQA rollout-to-GRPO PrefixGrouper benchmark.

Baseline and PrefixGrouper are deliberately separate invocations. The script
supports the pinned GPU and Ascend NPU stacks through ``--device`` and records
both per-step VERL metrics and per-rollout 2WikiMQA rewards as JSONL.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import string
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from omegaconf import OmegaConf
from packaging.version import InvalidVersion, Version

import agentlightning as agl
from agentlightning.verl.accelerator import AcceleratorRuntime, Backend, select_accelerator
from agentlightning.verl.trainer import AgentLightningTrainer
from prefix_grouper_stack import NPU_CANN_VERSION, REQUIRED_STACKS

BENCHMARK_ID = "pg-2wikimqa-e2e"
RESULT_SCHEMA_VERSION = 1
DATASET_NAME = "2WikiMQA"
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SYSTEM_PROMPT = (
    "Answer the question using only the supplied Wikipedia passages. "
    "Return only the shortest answer phrase, with no explanation."
)
MAX_PROMPT_TOKENS = 2048
MIN_PROMPT_TOKENS = 1900
MAX_RESPONSE_TOKENS = 64
MINIMUM_DATASET_ROWS = 64
DEFAULT_ROLLOUTS_PER_SAMPLE = 4
DEFAULT_TRAIN_BATCH_SIZE = 8
DEFAULT_MICRO_BATCH_SIZE = 2
DEFAULT_STEPS = 8
DEFAULT_RUNNERS = 16
DEFAULT_SEED = 20260827
Mode = Literal["baseline", "prefix_grouper"]


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one complete JSONL record under a cross-process file lock."""
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _numeric(value: Any) -> int | float | bool | None:
    """Convert scalar metric values to JSON-safe finite numbers."""
    if isinstance(value, (bool, int, float)):
        result = value
    elif hasattr(value, "item"):
        try:
            result = value.item()
        except Exception:
            return None
    else:
        return None
    if isinstance(result, float) and not math.isfinite(result):
        return None
    return result if isinstance(result, (bool, int, float)) else None


class BenchmarkTrainer(AgentLightningTrainer):
    """Record the production trainer's built-in metrics without altering steps."""

    def _train_step(self, batch_dict: dict[str, Any], *, profile_rollout: bool = False) -> dict[str, Any]:
        base_trainer: Any = cast(Any, super())
        metrics = cast(dict[str, Any], base_trainer._train_step(batch_dict, profile_rollout=profile_rollout))
        config: Any = cast(Any, self).config
        record: dict[str, Any] = {
            "record_type": "step",
            "schema_version": RESULT_SCHEMA_VERSION,
            "benchmark_id": BENCHMARK_ID,
            "mode": str(config.trainer.benchmark_mode),
            "backend": str(config.trainer.benchmark_backend),
            "global_step": self.global_steps,
        }
        for key, value in metrics.items():
            converted = _numeric(value)
            if converted is not None:
                record[key] = converted
        _append_jsonl(Path(str(config.trainer.benchmark_metrics_path)), record)
        print("AGL_2WIKIMQA_STEP=" + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        return metrics


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _qa_f1(prediction: str, answer: str) -> float:
    prediction_tokens = _normalize_answer(prediction).split()
    answer_tokens = _normalize_answer(answer).split()
    if not prediction_tokens or not answer_tokens:
        return float(prediction_tokens == answer_tokens)
    common = sum((Counter(prediction_tokens) & Counter(answer_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(answer_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _rollout_seed(task: dict[str, Any], rollout: agl.Rollout) -> tuple[int, int]:
    metadata = rollout.metadata or {}
    rollout_index = metadata.get("rollout_index")
    if not isinstance(rollout_index, int) or rollout_index < 0:
        raise ValueError("The maintained E2E benchmark requires rollout.metadata['rollout_index'].")
    payload = f"{task['benchmark_seed']}:{task['sample_id']}:{rollout_index}".encode("utf-8")
    request_seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF
    return request_seed, rollout_index


@agl.rollout
async def wiki_agent(task: dict[str, Any], llm: agl.LLM, rollout: agl.Rollout) -> float:
    """Answer one 2WikiMQA question and return token-overlap F1 as reward."""
    request_seed, rollout_index = _rollout_seed(task, rollout)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(task["input"])},
    ]
    request = {
        "model": llm.model,
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "seed": request_seed,
        "return_token_ids": True,
    }
    headers = {"Authorization": f"Bearer {llm.api_key or 'local-vllm'}"}
    endpoint = llm.endpoint.rstrip("/") + "/chat/completions"
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(endpoint, headers=headers, json=request)
        response.raise_for_status()
        payload = response.json()

    message = payload["choices"][0]["message"]
    text = str(message.get("content") or message.get("reasoning_content") or "").strip()
    answers = [str(answer) for answer in task["answers"]]
    reward = max((_qa_f1(text, answer) for answer in answers), default=0.0)
    exact_reward = float(any(_normalize_answer(text) == _normalize_answer(answer) for answer in answers))

    responses_path = os.environ.get("AGL_2WIKIMQA_E2E_RESPONSES_PATH")
    if responses_path:
        _append_jsonl(
            Path(responses_path),
            {
                "record_type": "rollout",
                "schema_version": RESULT_SCHEMA_VERSION,
                "benchmark_id": BENCHMARK_ID,
                "sample_id": task["sample_id"],
                "prompt_tokens": task["prompt_tokens"],
                "rollout_index": rollout_index,
                "request_seed": request_seed,
                "answers": answers,
                "response": text,
                "exact_reward": exact_reward,
                "reward": reward,
            },
        )
    return reward


def load_dataset(path: Path, benchmark_seed: int) -> list[dict[str, Any]]:
    """Load and validate the maintained 2WikiMQA JSONL schema."""
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            source = cast(dict[str, Any], json.loads(line))
            if source.get("dataset") != DATASET_NAME:
                raise ValueError(f"{path}:{line_number} is not marked as {DATASET_NAME}.")
            sample_id = str(source["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample ID at {path}:{line_number}: {sample_id}")
            sample_ids.add(sample_id)
            answers = source["answers"]
            if not isinstance(answers, list) or not answers:
                raise ValueError(f"{path}:{line_number} has no golden answers.")
            answer_values = cast(list[Any], answers)
            prompt_tokens = int(source["prompt_tokens"])
            if not MIN_PROMPT_TOKENS <= prompt_tokens <= MAX_PROMPT_TOKENS:
                raise ValueError(
                    f"{path}:{line_number} has {prompt_tokens} prompt tokens; "
                    f"the maintained range is {MIN_PROMPT_TOKENS}–{MAX_PROMPT_TOKENS}."
                )
            rows.append(
                {
                    "input": str(source["prompt"]),
                    "answers": [str(answer) for answer in answer_values],
                    "sample_id": sample_id,
                    "prompt_tokens": prompt_tokens,
                    "benchmark_seed": benchmark_seed,
                }
            )
    return rows


def _normalized_version(value: str) -> str:
    try:
        return Version(value).public.split("+", 1)[0]
    except InvalidVersion:
        return value


def installed_stack(backend: Backend) -> dict[str, str]:
    """Return installed package versions for one pinned benchmark stack."""
    versions: dict[str, str] = {}
    for distribution in REQUIRED_STACKS[backend]:
        try:
            versions[distribution] = _normalized_version(importlib.metadata.version(distribution))
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def check_stack(backend: Backend, versions: dict[str, str]) -> None:
    """Reject software drift before an official benchmark run."""
    mismatches = {
        package: {"required": required, "installed": versions.get(package, "missing")}
        for package, required in REQUIRED_STACKS[backend].items()
        if versions.get(package) != required
    }
    if mismatches:
        raise RuntimeError(
            f"{backend.upper()} benchmark stack does not match the pinned matrix: "
            + json.dumps(mismatches, sort_keys=True)
        )


def build_config(
    *,
    args: argparse.Namespace,
    backend: Backend,
    n_devices_per_node: int,
    tensor_model_parallel_size: int,
    metrics_path: Path,
    dataset_size: int,
) -> dict[str, Any]:
    """Build an accelerator-neutral VERL configuration for the fixed workload."""
    mode: Mode = args.mode
    steps_per_epoch = dataset_size // args.train_batch_size
    total_epochs = math.ceil(args.steps / steps_per_epoch)
    config: dict[str, Any] = {
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "data": {
            "train_batch_size": args.train_batch_size,
            "max_prompt_length": MAX_PROMPT_TOKENS,
            "max_response_length": MAX_RESPONSE_TOKENS,
            "filter_overlong_prompts": False,
        },
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "mode": "async",
                "tensor_model_parallel_size": tensor_model_parallel_size,
                "n": args.rollouts_per_sample,
                "temperature": 1.0,
                "top_p": 1.0,
                "log_prob_micro_batch_size_per_gpu": args.micro_batch_size_per_device,
                "multi_turn": {"enable": False, "format": "hermes"},
                "gpu_memory_utilization": 0.35,
                "max_model_len": MAX_PROMPT_TOKENS + MAX_RESPONSE_TOKENS,
            },
            "actor": {
                "strategy": "fsdp",
                "ppo_mini_batch_size": args.train_batch_size,
                "ppo_micro_batch_size_per_gpu": args.micro_batch_size_per_device,
                "ppo_epochs": 1,
                "optim": {"lr": 1e-6},
                "use_kl_loss": True,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0.0,
                "use_torch_compile": False,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                    "use_torch_compile": False,
                    "model_dtype": "bf16",
                    "mixed_precision": {
                        "param_dtype": "bf16",
                        "reduce_dtype": "fp32",
                        "buffer_dtype": "fp32",
                    },
                    "ulysses_sequence_parallel_size": 1,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": args.micro_batch_size_per_device,
                "fsdp_config": {
                    "param_offload": True,
                    "model_dtype": "bf16",
                    "mixed_precision": {
                        "param_dtype": "bf16",
                        "reduce_dtype": "fp32",
                        "buffer_dtype": "fp32",
                    },
                    "ulysses_sequence_parallel_size": 1,
                },
            },
            "model": {
                "path": args.model,
                "override_config": {"attn_implementation": "sdpa"},
                "use_remove_padding": False,
                "use_fused_kernels": False,
                "external_lib": "agentlightning.verl.benchmark_fsdp_sync",
                "enable_gradient_checkpointing": True,
                "enable_activation_offload": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": n_devices_per_node,
            "nnodes": 1,
            "balance_batch": False,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console"],
            "project_name": "AgentLightningE2EBenchmark",
            "experiment_name": f"2wikimqa_qwen3_30b_{backend}_{mode}",
            "save_freq": -1,
            "test_freq": -1,
            "total_epochs": total_epochs,
            "total_training_steps": args.steps,
            "benchmark_metrics_path": str(metrics_path),
            "benchmark_mode": mode,
            "benchmark_backend": backend,
        },
    }
    if backend == "npu":
        config["agentlightning"] = {"npu_model_download": {"enabled": True, "local_files_only": args.local_files_only}}
    if mode == "prefix_grouper":
        config.setdefault("agentlightning", {})["prefix_grouper"] = {"enabled": True}
    return config


def parse_args() -> argparse.Namespace:
    """Parse maintained benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "prefix_grouper"), required=True)
    parser.add_argument("--device", choices=("auto", "gpu", "cuda", "npu"), default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pretrained model ID or local model directory.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--micro-batch-size-per-device", type=int, default=DEFAULT_MICRO_BATCH_SIZE)
    parser.add_argument("--rollouts-per-sample", type=int, default=DEFAULT_ROLLOUTS_PER_SAMPLE)
    parser.add_argument("--n-runners", type=int, default=DEFAULT_RUNNERS)
    parser.add_argument("--n-devices-per-node", type=int)
    parser.add_argument("--tensor-model-parallel-size", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--local-files-only", action="store_true", help="Disable NPU model downloads.")
    parser.add_argument("--overwrite", action="store_true", help="Replace benchmark JSONL outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the config without using hardware.")
    return parser.parse_args()


def _requested_backend(device: str) -> Backend:
    normalized = device.lower()
    if normalized in {"gpu", "cuda"}:
        return "gpu"
    if normalized == "npu":
        return "npu"
    raise ValueError("--dry-run requires an explicit --device gpu/cuda or --device npu.")


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def resolve_resources(args: argparse.Namespace, available_devices: int | None) -> tuple[int, int]:
    """Validate the single-node resource shape used by both benchmark modes."""
    if args.n_devices_per_node is None:
        if available_devices is None:
            raise ValueError("--dry-run requires --n-devices-per-node.")
        n_devices_per_node = available_devices
    else:
        n_devices_per_node = args.n_devices_per_node
    _validate_positive("--n-devices-per-node", n_devices_per_node)
    if available_devices is not None and n_devices_per_node > available_devices:
        raise ValueError(f"Requested {n_devices_per_node} devices, but only {available_devices} are visible.")

    tensor_model_parallel_size = args.tensor_model_parallel_size or n_devices_per_node
    _validate_positive("--tensor-model-parallel-size", tensor_model_parallel_size)
    if tensor_model_parallel_size > n_devices_per_node:
        raise ValueError("Tensor parallel size cannot exceed devices per node.")
    if n_devices_per_node % tensor_model_parallel_size != 0:
        raise ValueError("Tensor parallel size must divide devices per node.")
    if args.train_batch_size % n_devices_per_node != 0:
        raise ValueError("Train batch size must be divisible by devices per node to preserve prompt groups.")
    return n_devices_per_node, tensor_model_parallel_size


def prepare_outputs(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    """Create an isolated output directory without silently destroying results."""
    metrics_path = output_dir / "metrics.jsonl"
    responses_path = output_dir / "responses.jsonl"
    existing = [path for path in (metrics_path, responses_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite benchmark outputs: " + ", ".join(map(str, existing)))
    if overwrite:
        for path in existing:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    return metrics_path.resolve(), responses_path.resolve()


def run_benchmark(
    args: argparse.Namespace,
    runtime: AcceleratorRuntime,
    dataset: list[dict[str, Any]],
    config: dict[str, Any],
    metrics_path: Path,
    responses_path: Path,
    stack: dict[str, str],
) -> None:
    """Launch the production Agent Lightning VERL training entrypoint."""
    runtime.set_device()
    os.environ["AGL_2WIKIMQA_E2E_RESPONSES_PATH"] = str(responses_path)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    algorithm = agl.VERL(config, trainer_cls=BenchmarkTrainer)
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=args.n_runners,
        tracer=agl.OtelTracer(),
        adapter=agl.LlmProxyTraceToTriplet(),
    )
    trainer.fit(wiki_agent, train_dataset=dataset, val_dataset=dataset[: args.train_batch_size])
    record = {
        "record_type": "run",
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "dataset": DATASET_NAME,
        "mode": args.mode,
        "backend": runtime.backend,
        "device_name": runtime.device_name(),
        "started_at": started_at,
        "wall_seconds": time.perf_counter() - started,
        "steps": args.steps,
        "dataset_rows": len(dataset),
        "train_batch_size": args.train_batch_size,
        "micro_batch_size_per_device": args.micro_batch_size_per_device,
        "rollouts_per_sample": args.rollouts_per_sample,
        "min_prompt_tokens": MIN_PROMPT_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "n_devices_per_node": config["trainer"]["n_gpus_per_node"],
        "tensor_model_parallel_size": config["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"],
        "n_runners": args.n_runners,
        "seed": args.seed,
        "model": args.model,
        "dataset_path": str(args.dataset_path.resolve()),
        "stack": stack,
        "required_cann": NPU_CANN_VERSION if runtime.backend == "npu" else None,
    }
    _append_jsonl(metrics_path, record)
    print("AGL_2WIKIMQA_RUN=" + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    """Validate the formal workload and optionally launch it."""
    args = parse_args()
    for name in (
        "--steps",
        "--train-batch-size",
        "--micro-batch-size-per-device",
        "--rollouts-per-sample",
        "--n-runners",
    ):
        _validate_positive(name, int(getattr(args, name[2:].replace("-", "_"))))
    if not args.dataset_path.is_file():
        raise FileNotFoundError(f"Prepared 2WikiMQA JSONL does not exist: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path, args.seed)
    if len(dataset) < MINIMUM_DATASET_ROWS:
        raise ValueError(
            f"Dataset has {len(dataset)} rows; the maintained workload requires at least {MINIMUM_DATASET_ROWS}."
        )
    if len(dataset) < args.train_batch_size:
        raise ValueError(f"Dataset has {len(dataset)} rows, fewer than train batch size {args.train_batch_size}.")

    runtime: AcceleratorRuntime | None = None
    if args.dry_run:
        backend = _requested_backend(args.device)
        available_devices = None
    else:
        runtime = select_accelerator(args.device)
        backend = runtime.backend
        available_devices = int(runtime.module.device_count())
    stack = installed_stack(backend)
    check_stack(backend, stack)
    n_devices_per_node, tensor_model_parallel_size = resolve_resources(args, available_devices)

    output_dir = args.output_dir.resolve()
    metrics_path = output_dir / "metrics.jsonl"
    responses_path = output_dir / "responses.jsonl"
    config = build_config(
        args=args,
        backend=backend,
        n_devices_per_node=n_devices_per_node,
        tensor_model_parallel_size=tensor_model_parallel_size,
        metrics_path=metrics_path,
        dataset_size=len(dataset),
    )
    if args.dry_run:
        merged_algorithm = agl.VERL(config, trainer_cls=BenchmarkTrainer)
        merged_config = cast(dict[str, Any], OmegaConf.to_container(merged_algorithm.config, resolve=True))
        print(
            json.dumps(
                {
                    "benchmark_id": BENCHMARK_ID,
                    "dataset": DATASET_NAME,
                    "dataset_rows": len(dataset),
                    "backend": backend,
                    "stack": stack,
                    "required_cann": NPU_CANN_VERSION if backend == "npu" else None,
                    "config": merged_config,
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    assert runtime is not None
    metrics_path, responses_path = prepare_outputs(output_dir, args.overwrite)
    run_benchmark(args, runtime, dataset, config, metrics_path, responses_path, stack)


if __name__ == "__main__":
    main()
