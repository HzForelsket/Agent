#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""统一运行 SQL、20 Questions 和 Web RAG 的多轮在线端到端训练。

每次进程只运行一个任务和一种模式。``baseline`` 使用标准 VERL 前向，
``simple`` 仅共享同题多条 rollout 完全相同的初始 prompt；多轮 suffix 保持独立。
两种模式都覆盖在线 rollout、轨迹级聚合、GRPO 和 actor update。
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import importlib.metadata
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, cast

from omegaconf import OmegaConf
from packaging.version import InvalidVersion, Version
from transformers import AutoConfig
from verl.workers.rollout.utils import get_max_position_embeddings

import agentlightning as agl
from agentlightning.verl.accelerator import AcceleratorRuntime, Backend, select_accelerator
from agentlightning.verl.daemon import AgentModeDaemon
from agentlightning.verl.model_download import materialize_model
from agentlightning.verl.trainer import AgentLightningTrainer
from prefix_grouper_stack import NPU_CANN_VERSION, REQUIRED_STACKS

BENCHMARK_ID = "agl-multiturn-online-e2e"
RESULT_SCHEMA_VERSION = 3
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_STEPS = 10
DEFAULT_TASKS = 32
DEFAULT_TRAIN_BATCH_SIZE = 8
DEFAULT_ROLLOUTS = 4
DEFAULT_MICRO_BATCH_SIZE = 4
DEFAULT_RUNNERS = 10
DEFAULT_SEED = 20260920
REPO_ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = REPO_ROOT / "examples" / "rag"
SPIDER_DIR = REPO_ROOT / "examples" / "spider"
Q20_DIR = REPO_ROOT / "examples" / "tinker"

Mode = Literal["baseline", "simple"]
TaskName = Literal["sql", "q20", "web"]


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one complete metrics record safely from the Ray trainer process."""
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _numeric(value: Any) -> int | float | bool | None:
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
    """Persist the production trainer's per-step metrics without changing a step."""

    def _train_step(self, batch_dict: dict[str, Any], *, profile_rollout: bool = False) -> dict[str, Any]:
        base_trainer: Any = cast(Any, super())
        metrics = cast(dict[str, Any], base_trainer._train_step(batch_dict, profile_rollout=profile_rollout))
        config: Any = cast(Any, self).config
        record: dict[str, Any] = {
            "record_type": "step",
            "schema_version": RESULT_SCHEMA_VERSION,
            "benchmark_id": BENCHMARK_ID,
            "task": str(config.trainer.benchmark_task),
            "mode": str(config.trainer.benchmark_mode),
            "backend": str(config.trainer.benchmark_backend),
            "global_step": self.global_steps,
        }
        for key, value in metrics.items():
            converted = _numeric(value)
            if converted is not None:
                record[key] = converted
        _append_jsonl(Path(str(config.trainer.benchmark_metrics_path)), record)
        print("AGL_MULTITURN_STEP=" + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        return metrics


class LocalModelEnvironmentDaemon(AgentModeDaemon):
    """Expose the single local rollout model as an untraced environment LLM."""

    def _build_rollout_resources(
        self,
        llm_resource: agl.LLM,
        server_addresses: list[str],
        is_train: bool,
    ) -> agl.NamedResources:
        resources = super()._build_rollout_resources(llm_resource, server_addresses, is_train)
        if len(server_addresses) != 1:
            raise RuntimeError(
                "The pure-local Q20 answerer requires exactly one rollout-model server; "
                "set tensor model parallel size equal to devices per node."
            )
        resources["environment_llm"] = agl.LLM(
            endpoint=f"http://{server_addresses[0]}/v1",
            model=self.model_name,
            sampling_parameters={"temperature": 0.0},
        )
        return resources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("sql", "q20", "web"), required=True)
    parser.add_argument("--mode", choices=("baseline", "simple"), required=True)
    parser.add_argument("--device", choices=("auto", "gpu", "cuda", "npu"), default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local checkpoint.")
    parser.add_argument("--model-name", help="Stable logical name exposed to the rollout service.")
    parser.add_argument("--dataset", type=Path, help="Optional task-specific CSV/Parquet dataset.")
    parser.add_argument("--sql-database-dir", type=Path, help="Spider root or database directory.")
    parser.add_argument("--output-dir", type=Path, help="Fresh result directory; generated automatically when omitted.")
    parser.add_argument("--download-dir", type=Path, default=REPO_ROOT / ".cache" / "multiturn-artifacts")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASKS)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--rollouts-per-sample", type=int, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--micro-batch-size-per-device", type=int, default=DEFAULT_MICRO_BATCH_SIZE)
    parser.add_argument("--n-runners", type=int, default=DEFAULT_RUNNERS)
    parser.add_argument("--n-devices-per-node", type=int, default=4)
    parser.add_argument("--tensor-model-parallel-size", type=int)
    parser.add_argument("--max-prompt-length", type=int, default=4096, help="Training trajectory prompt capacity.")
    parser.add_argument("--max-response-length", type=int, help="Training trajectory suffix capacity.")
    parser.add_argument(
        "--rollout-max-model-len", type=int, help="Serving context capacity; defaults to the model context length."
    )
    parser.add_argument(
        "--rollout-max-tokens", type=int, default=2048, help="Maximum output tokens per online model request."
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-freq", type=int, default=-1)
    parser.add_argument("--sql-max-turns", type=int, default=3)
    parser.add_argument("--web-max-turns", type=int, default=10)
    parser.add_argument("--web-mcp-url", help="Use an existing MCP SSE endpoint instead of starting one.")
    parser.add_argument("--web-mcp-port", type=int, default=8099)
    parser.add_argument("--web-mcp-startup-timeout", type=int, default=300)
    parser.add_argument("--web-data-dir", type=Path)
    parser.add_argument("--web-embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--web-embedding-cache", type=Path)
    parser.add_argument("--q20-search", action="store_true")
    parser.add_argument(
        "--q20-request-timeout", type=float, default=120.0, help="Timeout in seconds for each Q20 model request."
    )
    parser.add_argument("--npu-attention-backend", choices=("fusion", "custom"), default="fusion")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--insecure-download", action="store_true")
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture selected complete training steps with VERL's worker and rollout profilers.",
    )
    parser.add_argument("--profile-steps", type=int, nargs="+", help="1-based steps to capture; default: 1.")
    parser.add_argument("--profile-dir", type=Path, help="Trace directory; default: <output-dir>/profile.")
    parser.add_argument(
        "--profile-record-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record operator input shapes when profiling (default: enabled).",
    )
    parser.add_argument(
        "--profile-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record operator memory when profiling (default: disabled).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Prepare data and print config without hardware use.")
    return parser.parse_args()


def _normalized_version(value: str) -> str:
    try:
        return Version(value).public.split("+", 1)[0]
    except InvalidVersion:
        return value


def installed_stack(backend: Backend) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in REQUIRED_STACKS[backend]:
        try:
            versions[distribution] = _normalized_version(importlib.metadata.version(distribution))
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def check_stack(backend: Backend, versions: dict[str, str]) -> None:
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


def _requested_backend(device: str) -> Backend:
    if device in {"gpu", "cuda"}:
        return "gpu"
    if device == "npu":
        return "npu"
    raise ValueError("--dry-run requires an explicit --device gpu/cuda or --device npu.")


def _logical_model_name(model: str, configured: str | None) -> str:
    name = configured or (Path(model).name if Path(model).is_absolute() else model)
    name = name.strip()
    if not name or Path(name).is_absolute():
        raise ValueError("--model-name must be a non-empty logical name, not an absolute path.")
    return name


def validate_args(args: argparse.Namespace) -> None:
    if not args.profile and (args.profile_steps is not None or args.profile_dir is not None or args.profile_memory):
        raise ValueError("--profile-steps, --profile-dir and --profile-memory require --profile.")
    if args.profile:
        args.profile_steps = sorted(set(args.profile_steps or [1]))
        if any(step < 1 or step > args.steps for step in args.profile_steps):
            raise ValueError("--profile-steps must be between 1 and --steps (inclusive).")
    if not math.isfinite(args.q20_request_timeout) or args.q20_request_timeout <= 0:
        raise ValueError("--q20-request-timeout must be finite and positive.")
    for name in (
        "steps",
        "tasks",
        "train_batch_size",
        "rollouts_per_sample",
        "micro_batch_size_per_device",
        "n_runners",
        "n_devices_per_node",
        "max_prompt_length",
        "rollout_max_tokens",
        "sql_max_turns",
        "web_max_turns",
        "web_mcp_port",
        "web_mcp_startup_timeout",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.rollouts_per_sample < 2:
        raise ValueError("GRPO baseline/simple comparison requires at least two rollouts per sample.")
    if args.tasks < args.train_batch_size or args.tasks % args.train_batch_size:
        raise ValueError("--tasks must be at least one train batch and divisible by --train-batch-size.")
    if args.train_batch_size % args.n_devices_per_node:
        raise ValueError("--train-batch-size must be divisible by --n-devices-per-node.")
    if args.micro_batch_size_per_device % args.rollouts_per_sample:
        raise ValueError("--micro-batch-size-per-device must be a multiple of --rollouts-per-sample.")
    if (args.train_batch_size * args.rollouts_per_sample) % (
        args.n_devices_per_node * args.micro_batch_size_per_device
    ):
        raise ValueError("The rollout batch must be divisible by devices * micro-batch-size-per-device.")
    tensor_parallel = args.tensor_model_parallel_size or args.n_devices_per_node
    if tensor_parallel <= 0 or args.n_devices_per_node % tensor_parallel:
        raise ValueError("--tensor-model-parallel-size must be positive and divide --n-devices-per-node.")
    args.tensor_model_parallel_size = tensor_parallel
    if args.task == "q20" and tensor_parallel != args.n_devices_per_node:
        raise ValueError(
            "Pure-local Q20 requires --tensor-model-parallel-size to equal --n-devices-per-node "
            "so player and untraced answerer share one local model server."
        )
    if args.temperature <= 0 or args.learning_rate <= 0:
        raise ValueError("--temperature and --learning-rate must be positive.")
    if args.save_freq == 0 or args.save_freq < -1:
        raise ValueError("--save-freq must be -1 (disabled) or a positive integer.")
    if args.max_response_length is not None and args.max_response_length <= 0:
        raise ValueError("--max-response-length must be positive.")
    if args.rollout_max_model_len is not None and args.rollout_max_model_len <= 0:
        raise ValueError("--rollout-max-model-len must be positive.")


def resolve_model_limits(args: argparse.Namespace) -> None:
    args.model_ref = args.model
    model_ref = DEFAULT_MODEL if args.model == "Qwen3-8B" and not Path(args.model).exists() else args.model
    args.model_name = _logical_model_name(args.model, args.model_name)
    args.model = materialize_model(
        model_ref,
        args.download_dir.expanduser().resolve() / "models",
        local_files_only=args.local_files_only,
    ).local_path
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    context_length = int(get_max_position_embeddings(model_config))
    if args.max_prompt_length >= context_length:
        raise ValueError("--max-prompt-length must be smaller than the model context length.")
    if args.max_response_length is None:
        args.max_response_length = context_length - args.max_prompt_length
    elif args.max_prompt_length + args.max_response_length > context_length:
        raise ValueError("prompt + response capacity exceeds the model context length.")
    args.model_context_length = context_length
    if args.rollout_max_model_len is None:
        args.rollout_max_model_len = context_length
    elif args.rollout_max_model_len > context_length:
        raise ValueError("--rollout-max-model-len exceeds the model context length.")
    if args.rollout_max_tokens >= args.rollout_max_model_len:
        raise ValueError("--rollout-max-tokens must be smaller than --rollout-max-model-len.")


def prepare_workload(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sys.path.insert(0, str(RAG_DIR))
    from trace_tasks import prepare_tasks

    config = {
        "agent": "rag" if args.task == "web" else args.task,
        "dataset": str(args.dataset.expanduser().resolve()) if args.dataset else None,
        "sql_database_dir": (str(args.sql_database_dir.expanduser().resolve()) if args.sql_database_dir else None),
        "tasks": args.tasks,
        "seed": args.seed,
        "insecure_download": args.insecure_download,
    }
    tasks, metadata = prepare_tasks(config)
    if args.task == "q20":
        for task in tasks:
            task["search_enabled"] = args.q20_search
    return tasks, metadata


def configure_profiler(args: argparse.Namespace, backend: Backend, config: dict[str, Any]) -> None:
    """Use the production trainer's step boundaries for all worker processes."""
    if not args.profile:
        return
    tool = "npu" if backend == "npu" else "torch"
    output_dir = (args.profile_dir or (args.output_dir / "profile")).expanduser().resolve()
    contents = ["cpu", "npu" if backend == "npu" else "cuda"]
    if args.profile_record_shapes:
        contents.append("shapes")
    if args.profile_memory:
        contents.append("memory")
    config["global_profiler"] = {
        "tool": tool,
        "steps": args.profile_steps,
        "profile_continuous_steps": False,
        "save_path": str(output_dir),
    }
    for role in ("actor", "ref", "rollout"):
        tool_config: dict[str, Any] = {"contents": list(contents), "discrete": role == "rollout"}
        if backend == "npu":
            tool_config.update(level="level1", analysis=False)
        config["actor_rollout_ref"][role]["profiler"] = {
            "tool": tool,
            "enable": True,
            "all_ranks": True,
            "save_path": str(output_dir / role),
            "tool_config": {tool: tool_config},
        }


def build_config(
    args: argparse.Namespace,
    backend: Backend,
    metrics_path: Path,
    dataset_size: int,
) -> dict[str, Any]:
    steps_per_epoch = dataset_size // args.train_batch_size
    total_epochs = math.ceil(args.steps / steps_per_epoch)
    config: dict[str, Any] = {
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "agentlightning": {
            "model_name": args.model_name,
            "npu_model_download": {
                "enabled": False,
                "local_files_only": args.local_files_only,
            },
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
                "temperature": args.temperature,
                "top_p": 1.0,
                "log_prob_micro_batch_size_per_gpu": args.micro_batch_size_per_device,
                "multi_turn": {"enable": False, "format": "hermes"},
                "gpu_memory_utilization": 0.35,
                "max_model_len": args.rollout_max_model_len,
                "prompt_length": args.rollout_max_model_len - args.rollout_max_tokens,
                "response_length": args.rollout_max_tokens,
                "prometheus": {"served_model_name": args.model_name},
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                        "middleware": ["agentlightning.verl.rollout_context.RolloutContextMiddleware"],
                    }
                },
            },
            "actor": {
                "strategy": "fsdp",
                "ppo_mini_batch_size": args.train_batch_size,
                "ppo_micro_batch_size_per_gpu": args.micro_batch_size_per_device,
                "ppo_epochs": 1,
                "optim": {"lr": args.learning_rate},
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
            "device": "cuda" if backend == "gpu" else "npu",
            "n_gpus_per_node": args.n_devices_per_node,
            "nnodes": 1,
            "balance_batch": False,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console"],
            "project_name": "AgentLightningMultiturnE2E",
            "experiment_name": f"{args.task}_{args.model_name.replace('/', '_')}_{backend}_{args.mode}",
            "default_local_dir": str(args.output_dir.expanduser().resolve()),
            "save_freq": args.save_freq,
            "test_freq": -1,
            "total_epochs": total_epochs,
            "total_training_steps": args.steps,
            "seed": args.seed,
            "benchmark_metrics_path": str(metrics_path),
            "benchmark_task": args.task,
            "benchmark_mode": args.mode,
            "benchmark_backend": backend,
        },
    }
    if args.mode == "simple":
        config["agentlightning"]["prefix_grouper"] = {"enabled": True}
        config["actor_rollout_ref"]["model"]["override_config"][
            "prefix_grouper_npu_backend"
        ] = args.npu_attention_backend
    configure_profiler(args, backend, config)
    return config


class Q20Agent(agl.LitAgent[dict[str, Any]]):
    """Train only the player; answerer/search reuse the untraced local model."""

    def __init__(self, search_enabled: bool, request_timeout: float, max_tokens: int) -> None:
        super().__init__()
        self.search_enabled = search_enabled
        self.request_timeout = request_timeout
        self.max_tokens = max_tokens

    async def rollout_async(self, task: dict[str, Any], resources: Any, rollout: Any) -> float:
        sys.path.insert(0, str(Q20_DIR))
        from crewai import LLM as CrewLLM
        from opentelemetry.instrumentation.utils import suppress_instrumentation
        from q20_agent import AnswererResponse, SearchTool, TwentyQuestionsFlow

        class UntracedCrewLLM(CrewLLM):
            """Keep local environment-model calls out of the policy trace."""

            def call(self, *call_args: Any, **call_kwargs: Any) -> Any:
                with suppress_instrumentation():
                    return super().call(*call_args, **call_kwargs)

        llm = resources["main_llm"]
        environment_llm = resources["environment_llm"]
        base_url = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
        player = CrewLLM(
            model="openai/" + llm.model,
            base_url=base_url,
            api_key="dummy",
            extra_body={"return_token_ids": True},
            max_tokens=self.max_tokens,
            timeout=self.request_timeout,
        )
        answerer = UntracedCrewLLM(
            model="openai/" + environment_llm.model,
            base_url=environment_llm.endpoint,
            api_key="dummy",
            temperature=0.0,
            response_format=AnswererResponse,
            max_tokens=self.max_tokens,
            timeout=self.request_timeout,
        )
        search = None
        if self.search_enabled:
            search = SearchTool(
                model=UntracedCrewLLM(
                    model="openai/" + environment_llm.model,
                    base_url=environment_llm.endpoint,
                    api_key="dummy",
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    timeout=self.request_timeout,
                )
            )
        flow = TwentyQuestionsFlow(player_llm=player, answer_llm=answerer, search_tool=search)
        await asyncio.to_thread(
            lambda: asyncio.run(flow.kickoff_async({"answer": task["answer"], "category": task["category"]}))
        )
        return 1.0 if flow.state.correct else 0.0


def make_agent(args: argparse.Namespace, tasks: list[dict[str, Any]], mcp_url: str | None) -> Any:
    if args.task == "sql":
        sys.path.insert(0, str(SPIDER_DIR))
        from sql_agent import LitSQLAgent

        roots = {str(task["spider_dir"]) for task in tasks}
        if len(roots) != 1:
            raise ValueError("All selected SQL tasks must use one Spider root.")
        agent = LitSQLAgent(max_turns=args.sql_max_turns)
        agent.spider_dir = roots.pop()
        return agent
    if args.task == "q20":
        return Q20Agent(args.q20_search, args.q20_request_timeout, args.rollout_max_tokens)
    if not mcp_url:
        raise ValueError("Web RAG requires an MCP URL.")
    sys.path.insert(0, str(RAG_DIR))
    from rag_agent import RAGAgent

    return RAGAgent(mcp_server_url=mcp_url, max_turns=args.web_max_turns)


def _wait_for_port(process: subprocess.Popen[Any], port: int, timeout: int, log_path: Path) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"Web MCP exited with code {code}; inspect {log_path}.")
        with socket.socket() as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(1.0)
    raise TimeoutError(f"Web MCP did not listen on port {port} within {timeout}s; inspect {log_path}.")


def _require_free_port(port: int) -> None:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"Web MCP port {port} is already in use; choose --web-mcp-port.") from error


@contextmanager
def web_mcp(args: argparse.Namespace, output_dir: Path) -> Iterator[str | None]:
    if args.task != "web":
        yield None
        return
    if args.web_mcp_url:
        yield args.web_mcp_url
        return

    _require_free_port(args.web_mcp_port)
    data_dir = (args.web_data_dir or (REPO_ROOT.parent / "data" / "cache" / "rag")).expanduser().resolve()
    embedding_cache = (args.web_embedding_cache or (data_dir / "embedding-models")).expanduser().resolve()
    command = [
        sys.executable,
        "-u",
        str(RAG_DIR / "wiki_retriever_mcp.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.web_mcp_port),
        "--device",
        "cpu",
        "--data-dir",
        str(data_dir),
        "--embedding-model",
        args.web_embedding_model,
        "--embedding-cache",
        str(embedding_cache),
    ]
    if args.local_files_only:
        command.append("--local-files-only")
    if args.insecure_download:
        command.append("--insecure-download")
    log_path = output_dir / "web_mcp.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_port(process, args.web_mcp_port, args.web_mcp_startup_timeout, log_path)
            yield f"http://127.0.0.1:{args.web_mcp_port}/sse"
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()


def run_training(
    args: argparse.Namespace,
    runtime: AcceleratorRuntime,
    tasks: list[dict[str, Any]],
    dataset_metadata: dict[str, Any],
    config: dict[str, Any],
    stack: dict[str, str],
) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    metrics_path = output_dir / "metrics.jsonl"
    (output_dir / "selected_tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(dataset_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runtime.set_device()
    with web_mcp(args, output_dir) as mcp_url:
        launch = {
            "benchmark_id": BENCHMARK_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "command": sys.argv,
            "task": args.task,
            "mode": args.mode,
            "backend": runtime.backend,
            "device_name": runtime.device_name(),
            "required_cann": NPU_CANN_VERSION if runtime.backend == "npu" else None,
            "stack": stack,
            "model_ref": args.model_ref,
            "model_context_length": args.model_context_length,
            "mcp_url": mcp_url,
            "config": config,
        }
        (output_dir / "launch.json").write_text(
            json.dumps(launch, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        agent = make_agent(args, tasks, mcp_url)
        algorithm = agl.VERL(
            config,
            trainer_cls=BenchmarkTrainer,
            daemon_cls=LocalModelEnvironmentDaemon if args.task == "q20" else None,
        )
        trainer = agl.Trainer(
            n_runners=args.n_runners,
            algorithm=algorithm,
            tracer=agl.OtelTracer(),
            adapter=agl.LlmProxyTraceToTriplet(),
        )
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        trainer.fit(agent, train_dataset=tasks, val_dataset=tasks[: args.train_batch_size])
        run_record = {
            "record_type": "run",
            "schema_version": RESULT_SCHEMA_VERSION,
            "benchmark_id": BENCHMARK_ID,
            "task": args.task,
            "mode": args.mode,
            "backend": runtime.backend,
            "device_name": runtime.device_name(),
            "required_cann": NPU_CANN_VERSION if runtime.backend == "npu" else None,
            "started_at": started_at,
            "wall_seconds": time.perf_counter() - started,
            "steps": args.steps,
            "tasks": len(tasks),
            "train_batch_size": args.train_batch_size,
            "rollouts_per_sample": args.rollouts_per_sample,
            "micro_batch_size_per_device": args.micro_batch_size_per_device,
            "n_devices_per_node": args.n_devices_per_node,
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "n_runners": args.n_runners,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "rollout_max_model_len": args.rollout_max_model_len,
            "rollout_max_tokens": args.rollout_max_tokens,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "save_freq": args.save_freq,
            "seed": args.seed,
            "model": args.model,
            "model_ref": args.model_ref,
            "model_name": args.model_name,
            "dataset": dataset_metadata,
            "task_settings": {
                "sql_max_turns": args.sql_max_turns if args.task == "sql" else None,
                "web_max_turns": args.web_max_turns if args.task == "web" else None,
                "web_mcp_url": mcp_url if args.task == "web" else None,
                "q20_environment_model": "local_actor_backend" if args.task == "q20" else None,
                "q20_search": args.q20_search if args.task == "q20" else None,
                "q20_request_timeout": args.q20_request_timeout if args.task == "q20" else None,
            },
            "npu_attention_backend": args.npu_attention_backend if args.mode == "simple" else None,
            "stack": stack,
            "profile": {
                "enabled": args.profile,
                "steps": args.profile_steps if args.profile else [],
                "record_shapes": args.profile_record_shapes if args.profile else False,
                "profile_memory": args.profile_memory if args.profile else False,
                "scope": "selected-online-training-steps",
            },
            "profile_output_dir": config.get("global_profiler", {}).get("save_path"),
        }
        _append_jsonl(metrics_path, run_record)
        print("AGL_MULTITURN_RUN=" + json.dumps(run_record, ensure_ascii=False, sort_keys=True), flush=True)
        if args.profile and runtime.backend == "npu":
            profile_dir = str(config["global_profiler"]["save_path"])
            print(f"解析 NPU profile：{profile_dir}", flush=True)
            runtime.analyse_profiles(profile_dir)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = REPO_ROOT / ".cache" / f"multiturn-{args.task}-{args.mode}-{stamp}-{uuid.uuid4().hex[:8]}"
    if not args.dry_run:
        args.output_dir = args.output_dir.expanduser().resolve()
        args.output_dir.mkdir(parents=True, exist_ok=False)
        print(f"AGL_MULTITURN_OUTPUT_DIR={args.output_dir}", flush=True)
        invocation = {
            "command": sys.argv,
            "python": sys.executable,
            "arguments": vars(args),
            "visible_devices": {
                name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")
            },
        }
        (args.output_dir / "invocation.json").write_text(
            json.dumps(invocation, default=str, indent=2) + "\n", encoding="utf-8"
        )

    runtime: AcceleratorRuntime | None = None
    if args.dry_run:
        backend = _requested_backend(args.device)
    else:
        runtime = select_accelerator(args.device)
        backend = runtime.backend
        available = int(runtime.module.device_count())
        if args.n_devices_per_node > available:
            raise RuntimeError(f"Requested {args.n_devices_per_node} devices, but only {available} are visible.")
    if args.npu_attention_backend == "custom" and (backend != "npu" or args.mode != "simple"):
        raise ValueError("--npu-attention-backend custom requires --device npu --mode simple.")
    stack = installed_stack(backend)
    check_stack(backend, stack)
    resolve_model_limits(args)
    tasks, dataset_metadata = prepare_workload(args)
    metrics_path = args.output_dir.expanduser().resolve() / "metrics.jsonl"
    config = build_config(args, backend, metrics_path, len(tasks))
    if args.profile:
        print(f"AGL_MULTITURN_PROFILE_DIR={config['global_profiler']['save_path']}", flush=True)

    if args.dry_run:
        merged = agl.VERL(
            config,
            trainer_cls=BenchmarkTrainer,
            daemon_cls=LocalModelEnvironmentDaemon if args.task == "q20" else None,
        ).config
        print(
            json.dumps(
                {
                    "benchmark_id": BENCHMARK_ID,
                    "task": args.task,
                    "mode": args.mode,
                    "backend": backend,
                    "selected_tasks": len(tasks),
                    "dataset": dataset_metadata,
                    "stack": stack,
                    "required_cann": NPU_CANN_VERSION if backend == "npu" else None,
                    "config": OmegaConf.to_container(merged, resolve=True),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    assert runtime is not None
    run_training(args, runtime, tasks, dataset_metadata, config, stack)


if __name__ == "__main__":
    main()
