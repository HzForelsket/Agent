#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Test real rollout-to-GRPO training with exact prompt/response token lengths.

Usage (from the repository root, on a machine with a usable accelerator)::

    python scripts/test_fixed_length_e2e.py --model /path/to/model \
        --prompt-tokens 128 --response-tokens 32 --output-dir /tmp/fixed-length-e2e

``--prepare-only`` validates the local tokenizer/context and writes inputs and
configuration without starting Ray, loading model weights, or running rollout.
See FIXED_LENGTH_E2E.md for the acceptance criteria and synthetic reward policy.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast


def write_json(path: Path, value: Any) -> None:
    """Write a human-readable configuration or final result."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_event(path: Path, record: dict[str, Any]) -> None:
    """Persist each completed stage across runner and Ray processes."""
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    """Parse and reject invalid resource/length configurations before setup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local pretrained model/tokenizer directory.")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--response-tokens", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--rollouts-per-sample", type=int, default=2)
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--n-runners", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "gpu", "npu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory; never overwrites an existing run."
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for name in (
        "prompt_tokens",
        "response_tokens",
        "steps",
        "train_batch_size",
        "rollouts_per_sample",
        "n_devices",
        "tensor_parallel_size",
        "micro_batch_size",
        "n_runners",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.rollouts_per_sample < 2:
        parser.error("GRPO requires --rollouts-per-sample >= 2 for a nonzero advantage")
    if args.n_devices % args.tensor_parallel_size:
        parser.error("--tensor-parallel-size must divide --n-devices")
    if args.train_batch_size * args.rollouts_per_sample % (args.n_devices * args.micro_batch_size):
        parser.error("train-batch-size * rollouts-per-sample must be divisible by n-devices * micro-batch-size")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be finite and positive")
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")
    args.model = args.model.expanduser().resolve()
    if not args.model.is_dir():
        parser.error("--model must point to an existing local pretrained model directory")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def prepare_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Create deterministic token-native synthetic prompts without retokenization."""
    from transformers import AutoConfig, AutoTokenizer

    model_config: Any = cast(Any, AutoConfig).from_pretrained(args.model, local_files_only=True)
    text_config = model_config.get_text_config()
    context_length = getattr(text_config, "max_position_embeddings", None)
    if not isinstance(context_length, int):
        raise ValueError("The model must declare max_position_embeddings for context validation.")
    if args.prompt_tokens + args.response_tokens > context_length:
        raise ValueError(
            f"prompt + response = {args.prompt_tokens + args.response_tokens} exceeds model context {context_length}"
        )
    tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(args.model, local_files_only=True)
    rows: list[dict[str, Any]] = []
    for sample_id in range(args.train_batch_size):
        source_ids: list[int] = tokenizer.encode(
            f"Sample {sample_id}. Continue the following text: The river flows through the quiet valley. ",
            add_special_tokens=False,
        )
        source_ids = [int(token) for token in source_ids if token not in tokenizer.all_special_ids]
        if not source_ids:
            raise ValueError("Tokenizer produced no ordinary prompt tokens.")
        prompt_ids = (source_ids * math.ceil(args.prompt_tokens / len(source_ids)))[: args.prompt_tokens]
        rows.append({"sample_id": sample_id, "prompt_token_ids": prompt_ids})
    return rows


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Use the standard Agent Lightning/VERL FSDP training path."""
    return {
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "agentlightning": {"model_name": "fixed-length-e2e", "trace_aggregator": {"level": "transition"}},
        "data": {
            "train_batch_size": args.train_batch_size,
            "max_prompt_length": args.prompt_tokens,
            "max_response_length": args.response_tokens,
            "filter_overlong_prompts": False,
            "seed": args.seed,
        },
        "actor_rollout_ref": {
            "model": {
                "path": str(args.model),
                "override_config": {"attn_implementation": "sdpa"},
                "use_remove_padding": False,
                "use_fused_kernels": False,
                "enable_gradient_checkpointing": True,
            },
            "rollout": {
                "name": "vllm",
                "mode": "async",
                "tensor_model_parallel_size": args.tensor_parallel_size,
                "n": args.rollouts_per_sample,
                "temperature": 1.0,
                "top_p": 1.0,
                "seed": args.seed,
                "prompt_length": args.prompt_tokens,
                "response_length": args.response_tokens,
                "max_model_len": args.prompt_tokens + args.response_tokens,
                "gpu_memory_utilization": 0.35,
                "log_prob_micro_batch_size_per_gpu": args.micro_batch_size,
                "multi_turn": {"enable": False},
            },
            "actor": {
                "strategy": "fsdp",
                "ppo_mini_batch_size": args.train_batch_size,
                "ppo_micro_batch_size_per_gpu": args.micro_batch_size,
                "ppo_epochs": 1,
                "optim": {"lr": 1e-6},
                "use_kl_loss": True,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0.0,
                "use_torch_compile": False,
                "fsdp_config": {"param_offload": True, "optimizer_offload": True, "model_dtype": "bf16"},
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": args.micro_batch_size,
                "fsdp_config": {"param_offload": True, "model_dtype": "bf16"},
            },
        },
        "trainer": {
            "n_gpus_per_node": args.n_devices,
            "nnodes": 1,
            "balance_batch": False,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console"],
            "project_name": "FixedLengthE2E",
            "experiment_name": "fixed-length",
            "save_freq": -1,
            "test_freq": -1,
            "resume_mode": "disable",
            "total_epochs": args.steps,
            "total_training_steps": args.steps,
            "default_local_dir": str(args.output_dir / "checkpoints"),
        },
    }


def runtime_components(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Build test observers using the public trainer/daemon extension points."""
    import httpx
    import torch

    import agentlightning as agl
    from agentlightning.verl.daemon import AgentModeDaemon
    from agentlightning.verl.trainer import AgentLightningTrainer

    events_path = args.output_dir / "events.jsonl"
    expected_rows = args.train_batch_size * args.rollouts_per_sample

    def check_batch(batch: Any) -> None:
        tensors = batch.batch
        if len(batch) != expected_rows:
            raise AssertionError(f"Expected {expected_rows} training rows, got {len(batch)}")
        if tuple(tensors["prompts"].shape) != (expected_rows, args.prompt_tokens):
            raise AssertionError("Training prompt tensor shape does not match the requested length")
        if tuple(tensors["responses"].shape) != (expected_rows, args.response_tokens):
            raise AssertionError("Training response tensor shape does not match the requested length")
        if not bool((tensors["attention_mask"] == 1).all()):
            raise AssertionError("Training batch contains padding or missing response tokens")
        if bool(tensors["is_drop_mask"].any()):
            raise AssertionError("Training batch contains a dropped/truncated prompt")
        counts: Counter[str] = Counter(cast(list[str], batch.non_tensor_batch["data_id_list"]))
        if len(counts) != args.train_batch_size or set(counts.values()) != {args.rollouts_per_sample}:
            raise AssertionError(f"Incomplete GRPO groups: {dict(counts)}")

    class FixedLengthDaemon(AgentModeDaemon):
        """Expose token-native completions and observe real trace conversion."""

        def _build_rollout_resources(
            self, llm_resource: agl.LLM, server_addresses: list[str], is_train: bool
        ) -> agl.NamedResources:
            resources = super()._build_rollout_resources(llm_resource, server_addresses, is_train)
            if not server_addresses:
                raise RuntimeError("No live rollout servers")
            # Hosted-vLLM LiteLLM routes transform token prompts into chat text.
            # Use the real training servers' native completion API, then emit the
            # exact serving IDs through the standard tracer and triplet adapter.
            for index, address in enumerate(server_addresses):
                resources[f"completion_{index}"] = agl.LLM(
                    endpoint=f"http://{address}/v1",
                    model=self.model_name,
                )
            return resources

        def get_train_data_batch(
            self,
            max_prompt_length: int,
            max_response_length: int,
            device: torch.device,
            global_steps: int,
            *,
            prepare_prefix_groups: bool = False,
        ) -> Any:
            batch, metrics = super().get_train_data_batch(
                max_prompt_length,
                max_response_length,
                device,
                global_steps,
                prepare_prefix_groups=prepare_prefix_groups,
            )
            check_batch(batch)
            append_event(
                events_path,
                {
                    "stage": "trace_conversion",
                    "step": global_steps,
                    "rows": len(batch),
                    "prompt_tokens": args.prompt_tokens,
                    "response_tokens": args.response_tokens,
                },
            )
            return batch, metrics

    class FixedLengthAgent(agl.LitAgent[dict[str, Any]]):
        """Generate real tokens and supply a deterministic, nonconstant test reward."""

        async def rollout_async(
            self, task: dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout
        ) -> float:
            metadata = rollout.metadata or {}
            index = metadata.get("rollout_index")
            if type(index) is not int or not 0 <= index < args.rollouts_per_sample:
                raise AssertionError("Missing or invalid rollout_index")
            endpoints = [
                value
                for key, value in sorted(resources.items())
                if key.startswith("completion_") and isinstance(value, agl.LLM)
            ]
            if not endpoints:
                raise RuntimeError("No completion LLM resource")
            server_index = (int(task["sample_id"]) * int(args.rollouts_per_sample) + index) % len(endpoints)
            llm: agl.LLM = endpoints[server_index]
            prompt_ids = [int(token) for token in task["prompt_token_ids"]]
            seed_bytes = f"{args.seed}:{task['sample_id']}:{index}".encode()
            request_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:4], "big") % 2**31
            request = {
                "model": llm.model,
                "prompt": prompt_ids,
                "max_tokens": args.response_tokens,
                "min_tokens": args.response_tokens,
                "ignore_eos": True,
                "stop": [],
                "stop_token_ids": [],
                "add_special_tokens": False,
                "return_token_ids": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "seed": request_seed,
                "n": 1,
                "stream": False,
                "echo": False,
            }
            async with httpx.AsyncClient(timeout=args.request_timeout) as client:
                response = await client.post(llm.endpoint.rstrip("/") + "/completions", json=request)
                response.raise_for_status()
                payload = response.json()
            choices = payload.get("choices", [])
            if len(choices) != 1:
                raise AssertionError(f"Expected one completion, got {len(choices)}")
            choice = choices[0]
            actual_prompt = choice.get("prompt_token_ids")
            raw_response_ids: Any = choice.get("token_ids")
            if actual_prompt != prompt_ids or len(prompt_ids) != args.prompt_tokens:
                raise AssertionError("Serving prompt token IDs differ from the exact input")
            if (
                not isinstance(raw_response_ids, list)
                or len(cast(list[Any], raw_response_ids)) != args.response_tokens
                or not all(type(token) is int for token in cast(list[Any], raw_response_ids))
            ):
                raise AssertionError(f"Serving response must contain exactly {args.response_tokens} token IDs")
            response_ids = cast(list[int], raw_response_ids)
            if choice.get("finish_reason") != "length":
                raise AssertionError(f"Unexpected finish_reason: {choice.get('finish_reason')}")
            usage = payload.get("usage", {})
            if (
                usage.get("prompt_tokens") != args.prompt_tokens
                or usage.get("completion_tokens") != args.response_tokens
            ):
                raise AssertionError(f"Serving usage does not match the requested lengths: {usage}")
            tracer = agl.get_active_tracer()
            if tracer is None:
                raise RuntimeError("No active Agent Lightning tracer")
            tracer.create_span(
                "litellm_request",
                attributes={
                    "gen_ai.request.model": llm.model,
                    "gen_ai.response.id": payload["id"],
                    "prompt_token_ids": actual_prompt,
                    "response_token_ids": response_ids,
                },
            )
            # Deliberately synthetic: distinct group rewards exercise GRPO even
            # when the model emits identical completions. This is not a quality score.
            reward = index / (args.rollouts_per_sample - 1)
            append_event(
                events_path,
                {
                    "stage": "rollout",
                    "rollout_id": rollout.rollout_id,
                    "sample_id": int(task["sample_id"]),
                    "rollout_index": index,
                    "request_seed": request_seed,
                    "reward": reward,
                    "prompt_token_ids": actual_prompt,
                    "response_token_ids": response_ids,
                    "prompt_tokens": len(actual_prompt),
                    "response_tokens": len(response_ids),
                    "finish_reason": choice["finish_reason"],
                },
            )
            return reward

    class FixedLengthTrainer(AgentLightningTrainer):
        """Check the batch immediately before the production actor update."""

        def _update_actor(self, batch: Any) -> Any:
            check_batch(batch)
            for name in ("old_log_probs", "ref_log_prob", "advantages", "returns"):
                if not bool(torch.isfinite(batch.batch[name]).all()):
                    raise AssertionError(f"Nonfinite training tensor: {name}")
            if not bool((batch.batch["response_mask"] == 1).all()):
                raise AssertionError("Actor response mask excludes generated tokens")
            if not bool(batch.batch["advantages"].abs().sum() > 0):
                raise AssertionError("GRPO produced only zero advantages")
            append_event(events_path, {"stage": "actor_input", "step": self.global_steps, "rows": len(batch)})
            return super()._update_actor(batch)

        def _train_step(self, batch_dict: dict[str, Any], *, profile_rollout: bool = False) -> dict[str, Any]:
            metrics: dict[str, Any] = cast(Any, super())._train_step(batch_dict, profile_rollout=profile_rollout)
            # Save available measurements before evaluating the pass criteria.
            numeric = {}
            for key, value in metrics.items():
                if hasattr(value, "item"):
                    value = value.item()
                if isinstance(value, (int, float)):
                    numeric[key] = value if math.isfinite(value) else str(value)
            append_event(events_path, {"stage": "step_metrics", "step": self.global_steps, "metrics": numeric})
            for stage in ("gen", "trace_conversion", "old_log_prob", "ref", "adv", "update_actor"):
                value = float(metrics[f"timing_s/{stage}"])
                if not math.isfinite(value) or value <= 0:
                    raise AssertionError(f"Training stage did not complete: {stage}")
            grad_norm = float(metrics["actor/grad_norm"])
            if not math.isfinite(grad_norm) or grad_norm <= 0:
                raise AssertionError(f"Expected a finite nonzero actor gradient, got {grad_norm}")
            for key in ("actor/pg_loss", "actor/lr"):
                if not math.isfinite(float(metrics[key])):
                    raise AssertionError(f"Nonfinite actor metric: {key}")
            if float(metrics["actor/lr"]) <= 0:
                raise AssertionError("Actor learning rate is zero")
            append_event(events_path, {"stage": "step_passed", "step": self.global_steps})
            return metrics

    return FixedLengthAgent, FixedLengthDaemon, FixedLengthTrainer


def run_training(args: argparse.Namespace, dataset: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Run the ordinary trainer and require all requested steps and rollouts."""
    import agentlightning as agl
    from agentlightning.verl.accelerator import select_accelerator

    runtime = select_accelerator(args.device)
    if runtime.module.device_count() < args.n_devices:
        raise ValueError(f"Requested {args.n_devices} devices, found {runtime.module.device_count()}")
    runtime.set_device()
    write_json(args.output_dir / "device.json", {"backend": runtime.backend, "name": runtime.device_name()})
    agent_cls, daemon_cls, trainer_cls = runtime_components(args)
    algorithm = agl.VERL(config, trainer_cls=trainer_cls, daemon_cls=daemon_cls)
    from omegaconf import OmegaConf

    write_json(args.output_dir / "resolved_config.json", OmegaConf.to_container(algorithm.config, resolve=True))
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=args.n_runners,
        tracer=agl.OtelTracer(),
        adapter=agl.LlmProxyTraceToTriplet(),
    )
    trainer.fit(agent_cls(), train_dataset=dataset, val_dataset=dataset)
    events = [json.loads(line) for line in (args.output_dir / "events.jsonl").read_text().splitlines()]
    for stage in ("trace_conversion", "actor_input", "step_metrics", "step_passed"):
        steps = [event["step"] for event in events if event["stage"] == stage]
        if sorted(steps) != list(range(1, args.steps + 1)):
            raise AssertionError(f"Incomplete or duplicate {stage} stages: {steps}")
    rollouts = [event for event in events if event["stage"] == "rollout"]
    expected = args.steps * args.train_batch_size * args.rollouts_per_sample
    if len(rollouts) != expected or len({event["rollout_id"] for event in rollouts}) != expected:
        raise AssertionError(f"Expected {expected} distinct successful rollouts, got {len(rollouts)} records")
    write_json(
        args.output_dir / "result.json",
        {
            "status": "passed",
            "steps": args.steps,
            "rollouts": expected,
            "prompt_tokens": args.prompt_tokens,
            "response_tokens": args.response_tokens,
            "reward_policy": "synthetic-rollout-index",
            "nonzero_actor_gradient": True,
        },
    )


def main() -> None:
    """Prepare durable run metadata before any model/runtime initialization."""
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
        dataset = prepare_dataset(args)
        config = build_config(args)
        write_json(args.output_dir / "dataset.json", dataset)
        write_json(args.output_dir / "config.json", config)
        if args.prepare_only:
            write_json(args.output_dir / "result.json", {"status": "prepared", "training_executed": False})
            print("PREPARED: tokenizer, exact prompt lengths, context budget and config only; E2E not executed.")
            return
        run_training(args, dataset, config)
        print(f"PASSED: {args.output_dir / 'result.json'}", flush=True)
    except BaseException as error:
        events_path = args.output_dir / "events.jsonl"
        if events_path.exists() and events_path.stat().st_size:
            write_json(
                args.output_dir / "failure.json",
                {
                    "status": "failed",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "completed_stages": "See events.jsonl; partial results are not a passed E2E test.",
                },
            )
            print(f"FAILED: partial stage evidence retained in {args.output_dir}", file=sys.stderr, flush=True)
        else:
            shutil.rmtree(args.output_dir)
            print(
                "FAILED before any rollout/training stage completed; removed failed output directory.", file=sys.stderr
            )
        raise


if __name__ == "__main__":
    main()
