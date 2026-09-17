#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Train the Spider SQL agent end to end with trajectory-level PrefixGrouper.

Examples:
    python scripts/train_multiturn_prefix_grouper.py --device gpu --model /models/Qwen3-30B \
        --output-dir /runs/sql-prefix-gpu
    python scripts/train_multiturn_prefix_grouper.py --device npu --model /models/Qwen3-30B \
        --output-dir /runs/sql-prefix-npu
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
from packaging.version import InvalidVersion, Version
from transformers import AutoConfig
from verl.workers.rollout.utils import get_max_position_embeddings

import agentlightning as agl
from agentlightning.verl.accelerator import AcceleratorRuntime, Backend, select_accelerator
from prefix_grouper_stack import NPU_CANN_VERSION, REQUIRED_STACKS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_DATA = REPO_ROOT / "examples" / "spider" / "data" / "train_spider.parquet"
DEFAULT_VAL_DATA = REPO_ROOT / "examples" / "spider" / "data" / "test_dev_500.parquet"


def parse_args() -> argparse.Namespace:
    """Parse the shared GPU/NPU training interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("gpu", "npu"), required=True)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint directory.")
    parser.add_argument("--model-name", help="Stable logical name exposed to the rollout service.")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-devices-per-node", type=int, default=4)
    parser.add_argument("--tensor-model-parallel-size", type=int)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size-per-device", type=int, default=4)
    parser.add_argument("--rollouts-per-sample", type=int, default=4)
    parser.add_argument("--n-runners", type=int, default=10)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument(
        "--max-response-length",
        type=int,
        help="Cumulative trajectory suffix limit; defaults to model context minus --max-prompt-length.",
    )
    parser.add_argument("--total-epochs", type=int, default=2)
    parser.add_argument("--total-training-steps", type=int)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active-agent", help="Optional adapter agent-name filter.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Forbid model downloads when the NPU launcher materializes model references.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the selected software stack and print the merged config without using an accelerator.",
    )
    return parser.parse_args()


def requested_backend(device: str) -> Backend:
    """Map the public device argument to the common backend type."""
    return cast(Backend, device)


def normalized_version(value: str) -> str:
    """Normalize local-version suffixes before checking the pinned stack."""
    try:
        return Version(value).public.split("+", 1)[0]
    except InvalidVersion:
        return value


def installed_stack(backend: Backend) -> dict[str, str]:
    """Read package metadata without importing either accelerator runtime."""
    versions: dict[str, str] = {}
    for distribution in REQUIRED_STACKS[backend]:
        try:
            versions[distribution] = normalized_version(importlib.metadata.version(distribution))
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def check_stack(backend: Backend, versions: dict[str, str]) -> None:
    """Reject drift from the project's pinned PrefixGrouper stacks."""
    mismatches = {
        package: {"required": required, "installed": versions.get(package, "missing")}
        for package, required in REQUIRED_STACKS[backend].items()
        if versions.get(package) != required
    }
    if mismatches:
        raise RuntimeError(
            f"{backend.upper()} PrefixGrouper stack does not match the pinned matrix: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_args(args: argparse.Namespace) -> None:
    """Validate resource and grouping constraints shared by GPU and NPU."""
    positive = (
        "n_devices_per_node",
        "train_batch_size",
        "micro_batch_size_per_device",
        "rollouts_per_sample",
        "n_runners",
        "max_prompt_length",
        "total_epochs",
        "save_freq",
    )
    for name in positive:
        value = int(getattr(args, name))
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}.")
    if args.total_training_steps is not None and args.total_training_steps <= 0:
        raise ValueError("--total-training-steps must be positive.")
    if args.rollouts_per_sample < 2:
        raise ValueError("Simple shared-prefix training requires at least two rollouts per sample.")
    if args.train_batch_size % args.n_devices_per_node:
        raise ValueError("--train-batch-size must be divisible by --n-devices-per-node.")
    if args.micro_batch_size_per_device % args.rollouts_per_sample:
        raise ValueError(
            "--micro-batch-size-per-device must be a multiple of --rollouts-per-sample "
            "so a complete prefix group is not split."
        )
    tensor_parallel = args.tensor_model_parallel_size or args.n_devices_per_node
    if tensor_parallel <= 0 or args.n_devices_per_node % tensor_parallel:
        raise ValueError("--tensor-model-parallel-size must be positive and divide --n-devices-per-node.")
    args.tensor_model_parallel_size = tensor_parallel


def resolve_model_limits(args: argparse.Namespace) -> None:
    """Resolve a valid trajectory capacity from the selected model config."""
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    context_length = int(get_max_position_embeddings(model_config))
    if args.max_prompt_length >= context_length:
        raise ValueError(
            f"--max-prompt-length={args.max_prompt_length} must be smaller than the model context "
            f"length {context_length}."
        )
    if args.max_response_length is None:
        args.max_response_length = context_length - args.max_prompt_length
    elif args.max_response_length <= 0:
        raise ValueError("--max-response-length must be positive.")
    elif args.max_prompt_length + args.max_response_length > context_length:
        raise ValueError(
            f"Configured max_model_len={args.max_prompt_length + args.max_response_length} exceeds the "
            f"model context length {context_length}; use --max-response-length no larger than "
            f"{context_length - args.max_prompt_length}."
        )
    args.model_context_length = context_length


def logical_model_name(model: str, configured_name: str | None) -> str:
    """Keep the rollout service name independent from a local weight path."""
    name = configured_name
    if name is None:
        name = Path(model).name if Path(model).is_absolute() else model
    name = name.strip()
    if not name or Path(name).is_absolute():
        raise ValueError("--model-name must be a non-empty logical name, not an absolute path.")
    return name


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build one accelerator-neutral online rollout and training configuration."""
    model_name = logical_model_name(args.model, args.model_name)
    config: dict[str, Any] = {
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "agentlightning": {
            "model_name": model_name,
            "npu_model_download": {
                "enabled": True,
                "local_files_only": args.local_files_only,
            },
            "prefix_grouper": {"enabled": True},
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": args.max_prompt_length,
                "trajectory_max_response_length": args.max_response_length,
                "debug": False,
            },
        },
        "data": {
            "train_batch_size": args.train_batch_size,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "filter_overlong_prompts": False,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "mode": "async",
                "tensor_model_parallel_size": args.tensor_model_parallel_size,
                "n": args.rollouts_per_sample,
                "temperature": 1.0,
                "top_p": 1.0,
                "log_prob_micro_batch_size_per_gpu": args.micro_batch_size_per_device,
                # Agent Lightning executes the SQL workflow's multiple model calls. VERL's
                # separate native agent-loop switch remains disabled; the trace aggregator
                # below merges those real calls into one masked trajectory sample.
                "multi_turn": {"enable": False, "format": "hermes"},
                "gpu_memory_utilization": 0.35,
                "max_model_len": args.max_prompt_length + args.max_response_length,
                "prometheus": {"served_model_name": model_name},
            },
            "actor": {
                "strategy": "fsdp",
                "ppo_mini_batch_size": args.train_batch_size,
                "ppo_micro_batch_size_per_gpu": args.micro_batch_size_per_device,
                "ppo_epochs": 1,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
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
                "enable_gradient_checkpointing": True,
                "enable_activation_offload": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": args.n_devices_per_node,
            "nnodes": 1,
            "balance_batch": False,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console"],
            "project_name": "AgentLightningMultiturnPrefixGrouper",
            "experiment_name": "spider_sql_multiturn_shared_prefix",
            "default_local_dir": str(args.output_dir.expanduser().resolve()),
            "save_freq": args.save_freq,
            "test_freq": -1,
            "total_epochs": args.total_epochs,
            "seed": args.seed,
        },
    }
    if args.total_training_steps is not None:
        config["trainer"]["total_training_steps"] = args.total_training_steps
    return config


def run_training(args: argparse.Namespace, runtime: AcceleratorRuntime, config: dict[str, Any]) -> None:
    """Run real SQL rollouts and update the policy through the common VERL path."""
    train_path = args.train_data.expanduser().resolve()
    val_path = args.val_data.expanduser().resolve()
    for path in (train_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(f"Spider parquet file does not exist: {path}")

    import pandas as pd

    spider_dir = REPO_ROOT / "examples" / "spider"
    sys.path.insert(0, str(spider_dir))
    from sql_agent import LitSQLAgent

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Multiturn PrefixGrouper training output: {output_dir}", flush=True)
    launch = {
        "command": sys.argv,
        "backend": runtime.backend,
        "required_cann": NPU_CANN_VERSION if runtime.backend == "npu" else None,
        "stack": installed_stack(runtime.backend),
        "train_data": str(train_path),
        "val_data": str(val_path),
        "model_context_length": args.model_context_length,
        "config": config,
    }
    (output_dir / "launch.json").write_text(
        json.dumps(launch, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime.set_device()

    train_data = pd.read_parquet(train_path).to_dict(orient="records")
    val_data = pd.read_parquet(val_path).to_dict(orient="records")
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(
        n_runners=args.n_runners,
        algorithm=algorithm,
        adapter={"agent_match": args.active_agent},
    )
    trainer.fit(LitSQLAgent(), train_dataset=train_data, val_dataset=val_data)


def main() -> None:
    """Validate the common configuration and optionally launch training."""
    args = parse_args()
    validate_args(args)
    resolve_model_limits(args)
    backend = requested_backend(args.device)
    stack = installed_stack(backend)
    check_stack(backend, stack)
    config = build_config(args)

    if args.dry_run:
        merged = agl.VERL(config).config
        print(
            json.dumps(
                {
                    "backend": backend,
                    "required_cann": NPU_CANN_VERSION if backend == "npu" else None,
                    "stack": stack,
                    "config": OmegaConf.to_container(merged, resolve=True),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    runtime = select_accelerator(args.device)
    if runtime.backend != backend:
        raise RuntimeError(f"Requested {backend}, selected {runtime.backend}.")
    available_devices = int(runtime.module.device_count())
    if args.n_devices_per_node > available_devices:
        raise RuntimeError(f"Requested {args.n_devices_per_node} devices, but only {available_devices} are visible.")
    run_training(args, runtime, config)


if __name__ == "__main__":
    main()
