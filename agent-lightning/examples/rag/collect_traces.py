# Copyright (c) Microsoft. All rights reserved.

"""Start NPU vLLM and CPU MCP, collect complete RAG trajectories, and generate the benefit table."""

import argparse
import asyncio
import contextvars
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_data import DEFAULT_DATA_DIR, add_download_argument, ensure_example_data
from trace_services import Processes, check_ports, service_commands

CURRENT: contextvars.ContextVar[str] = contextvars.ContextVar("trajectory_id")


def append(root: Path, name: str, record: dict[str, Any]) -> None:
    """Persist one record under a process lock before proceeding."""
    with (root / name).open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    """Persist run configuration or metadata."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory; existing paths are rejected.")
    parser.add_argument("--model", default="Qwen3-30B-A3B-Instruct-2507", help="Served model name, not weight path.")
    parser.add_argument("--model-path", type=Path, help="Local BF16 Qwen3-30B-A3B-Instruct-2507 weight directory.")
    parser.add_argument("--vllm-python", default=sys.executable, help="Python executable in the NPU vLLM environment.")
    parser.add_argument(
        "--npu-devices",
        default=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        help="NPU chip IDs, e.g. 0,1,2,3; TP equals their count.",
    )
    parser.add_argument("--vllm-port", type=positive, default=18030)
    parser.add_argument("--mcp-port", type=positive, default=8099)
    parser.add_argument("--startup-timeout", type=positive, default=1800)
    parser.add_argument("--max-model-len", type=positive, default=32768)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85, help="vLLM memory fraction (also named gpu on NPU)."
    )
    parser.add_argument("--retrieval-data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_DATA_DIR / "embedding-models")
    parser.add_argument("--local-files-only", action="store_true", help="Require cached/local embedding model files.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA_DIR / "dataset_tiny.parquet")
    add_download_argument(parser)
    parser.add_argument("--tasks", type=positive, default=32)
    parser.add_argument("--rollouts-per-task", type=positive, default=4)
    parser.add_argument("--concurrency", type=positive, default=4)
    parser.add_argument("--max-model-calls", type=positive, default=8)
    parser.add_argument("--max-tokens-per-call", type=positive, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--proxy-port", type=positive, default=18031)
    parser.add_argument("--trajectory-timeout", type=positive, default=900)
    parser.add_argument(
        "--server-metadata", type=Path, help="JSON describing serving hardware, versions and launch command."
    )
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        return args
    if not args.model_path or not (args.model_path / "config.json").is_file():
        parser.error("--model-path must point to the local 30B weight directory containing config.json")
    devices = (args.npu_devices or "").split(",")
    if not all(device.strip().isdigit() for device in devices) or len({int(device) for device in devices}) != len(
        devices
    ):
        parser.error("set --npu-devices (or ASCEND_RT_VISIBLE_DEVICES) to distinct chip IDs, e.g. 0,1,2,3")
    args.npu_devices = ",".join(str(int(device)) for device in devices)
    executable = shutil.which(args.vllm_python)
    if executable is None:
        parser.error(f"vLLM Python executable not found: {args.vllm_python}")
    args.vllm_python = str(Path(executable).absolute())
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be in (0, 1)")
    if args.max_tokens_per_call >= args.max_model_len:
        parser.error("max-tokens-per-call must be smaller than max-model-len")
    if args.rollouts_per_task < 2:
        parser.error("rollouts-per-task must be at least 2 for cross-trajectory sharing")
    if not 0 <= args.temperature <= 2 or not 0 <= args.seed < 2**32:
        parser.error("temperature must be in [0, 2] and seed in [0, 2**32)")
    if max(args.vllm_port, args.mcp_port, args.proxy_port + args.concurrency - 1) > 65535:
        parser.error("proxy port range exceeds 65535")
    if args.worker is None and args.dataset.name != "dataset_tiny.parquet" and not args.dataset.is_file():
        parser.error(
            f"Custom dataset not found: {args.dataset.resolve()}. "
            "Only the bundled example filename dataset_tiny.parquet can be downloaded automatically."
        )
    return args


async def run_worker(root: Path, config: dict[str, Any], worker_id: int) -> None:
    # Each process owns a Lightning tracer; concurrent rollouts in one thread would conflict.
    os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
    os.environ["OPENAI_API_KEY"] = "local-capture-proxy"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import httpx
    import uvicorn
    from agents import RunHooks, set_tracing_disabled
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from rag_agent import RAGAgent

    import agentlightning as agl

    set_tracing_disabled(True)
    counts: dict[str, int] = defaultdict(int)
    valid_counts: dict[str, int] = defaultdict(int)
    finish_reasons: dict[str, list[str]] = defaultdict(list)
    identities: dict[str, dict[str, Any]] = {}
    app = FastAPI()
    headers = {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"} if os.environ.get("VLLM_API_KEY") else {}

    @app.post("/{trajectory_id}/v1/chat/completions")
    async def forward(trajectory_id: str, request: Request) -> JSONResponse:
        if trajectory_id not in identities:
            return JSONResponse({"error": {"message": "unknown trajectory"}}, status_code=404)
        turn = counts[trajectory_id]
        counts[trajectory_id] += 1
        if turn >= config["max_model_calls"]:
            return JSONResponse({"error": {"message": "model call cap exceeded"}}, status_code=400)
        payload = await request.json()
        identity = identities[trajectory_id]
        payload["return_token_ids"] = True
        sample = identity["task_index"] * config["rollouts_per_task"] + identity["sample_index"]
        payload["seed"] = (config["seed"] + sample * config["max_model_calls"] + turn) % (2**63)
        record = {
            "trajectory_id": trajectory_id,
            **identity,
            "turn": turn,
            "started_at": time.time(),
            "request": payload,
            "token_ids_valid": False,
        }
        try:
            async with httpx.AsyncClient(timeout=config["trajectory_timeout"], trust_env=False) as client:
                result = await client.post(config["endpoint"] + "/chat/completions", json=payload, headers=headers)
            record["http_status"] = result.status_code
            response = result.json()
            record["response"] = response
            if result.is_success:
                prompt = response.get("prompt_token_ids")
                choices = response.get("choices", [])
                completion = choices[0].get("token_ids") if len(choices) == 1 else None
                usage = response.get("usage") or {}
                record["prompt_token_ids"] = prompt
                record["response_token_ids"] = completion
                record["token_ids_valid"] = (
                    isinstance(prompt, list)
                    and isinstance(completion, list)
                    and len(prompt) == usage.get("prompt_tokens")
                    and len(completion) == usage.get("completion_tokens")
                    and bool(prompt)
                    and bool(completion)
                    and all(type(token) is int and token >= 0 for token in prompt + completion)
                )
                if not record["token_ids_valid"]:
                    raise ValueError("vLLM must return exact prompt_token_ids and choices[0].token_ids matching usage")
                valid_counts[trajectory_id] += 1
                finish_reasons[trajectory_id].append(choices[0]["finish_reason"])
            return JSONResponse(response, status_code=result.status_code)
        except Exception as error:
            record["error_type"] = type(error).__name__
            record["error"] = str(error)
            logging.exception("Model call failed: %s turn %s", trajectory_id, turn)
            return JSONResponse({"error": {"message": str(error), "type": "capture_error"}}, status_code=502)
        finally:
            record["finished_at"] = time.time()
            append(root, "calls.jsonl", record)

    class Hooks(RunHooks[Any]):
        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            append(
                root,
                "events.jsonl",
                {"trajectory_id": CURRENT.get(), "event": "tool_start", "tool": tool.name, "time": time.time()},
            )

        async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
            append(
                root,
                "events.jsonl",
                {
                    "trajectory_id": CURRENT.get(),
                    "event": "tool_end",
                    "tool": tool.name,
                    "result": result,
                    "time": time.time(),
                },
            )

        async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
            append(
                root,
                "events.jsonl",
                {"trajectory_id": CURRENT.get(), "event": "final_answer", "output": output, "time": time.time()},
            )

    port = config["proxy_port"] + worker_id
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    server_task = asyncio.create_task(server.serve())
    runner = agl.LitAgentRunner(agl.OtelTracer())
    store = agl.InMemoryLightningStore()
    try:
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError("Capture proxy did not start")
            await asyncio.sleep(0.1)
        runner.init(RAGAgent(mcp_server_url=config["mcp_url"], max_turns=config["max_model_calls"], hooks=Hooks()))
        runner.init_worker(worker_id, store)
        tasks = json.loads((root / "selected_tasks.json").read_text())
        for task_index, task in enumerate(tasks):
            for sample_index in range(config["rollouts_per_task"]):
                if (task_index * config["rollouts_per_task"] + sample_index) % config["concurrency"] != worker_id:
                    continue
                trajectory_id = f"q{task_index:03d}-r{sample_index}"
                CURRENT.set(trajectory_id)
                identities[trajectory_id] = {
                    "task_index": task_index,
                    "sample_index": sample_index,
                    "task_id": str(task["id"]),
                }
                info = {
                    "trajectory_id": trajectory_id,
                    **identities[trajectory_id],
                    "question": task["question"],
                    "answer": task["answer"],
                    "started_at": time.time(),
                    "status": "failed",
                }
                resource = agl.LLM(
                    endpoint=f"http://127.0.0.1:{port}/{trajectory_id}/v1",
                    model=config["model"],
                    sampling_parameters={
                        "temperature": config["temperature"],
                        "max_tokens": config["max_tokens_per_call"],
                    },
                )
                try:
                    rollout = await asyncio.wait_for(
                        runner.step(task, resources={"main_llm": resource}), timeout=config["trajectory_timeout"]
                    )
                    info["rollout"] = rollout.model_dump(mode="json")
                    info["status"] = "completed" if rollout.status == "succeeded" else "failed"
                    if "length" in finish_reasons[trajectory_id]:
                        info["status"] = "truncated"
                    for span in await store.query_spans(rollout.rollout_id):
                        append(
                            root, "spans.jsonl", {"trajectory_id": trajectory_id, "span": span.model_dump(mode="json")}
                        )
                except BaseException as error:
                    info["error_type"] = type(error).__name__
                    info["error"] = str(error)
                    info["status"] = "max_turns" if type(error).__name__ == "MaxTurnsExceeded" else "failed"
                    logging.exception("Trajectory stopped: %s", trajectory_id)
                    if not isinstance(error, Exception):
                        raise
                finally:
                    info.update(
                        finished_at=time.time(),
                        model_calls=counts[trajectory_id],
                        valid_model_calls=valid_counts[trajectory_id],
                    )
                    append(root, "trajectories.jsonl", info)
                    print(f"{trajectory_id}: {info['status']}, calls={counts[trajectory_id]}", flush=True)
                if valid_counts[trajectory_id] == 0:
                    raise RuntimeError(f"No valid model call for {trajectory_id}; inspect worker log and calls.jsonl")
    finally:
        try:
            runner.teardown()
        finally:
            server.should_exit = True
            await server_task


async def collect(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    if args.worker is not None:
        await run_worker(root, json.loads((root / "config.json").read_text()), args.worker)
        return
    root.mkdir(parents=True, exist_ok=False)
    print(f"Trace output: {root}", flush=True)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "worker"
    }
    config.update(
        output=str(root),
        dataset=str(args.dataset.resolve()),
        schema_version=1,
        endpoint=f"http://127.0.0.1:{args.vllm_port}/v1",
        mcp_url=f"http://127.0.0.1:{args.mcp_port}/sse",
        model_path=str(args.model_path.resolve()),
        retrieval_data_dir=str(args.retrieval_data_dir.resolve()),
        embedding_cache=str(args.embedding_cache.resolve()),
        created_at=time.time(),
    )
    write_json(root / "config.json", config)
    processes = Processes(root)
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    if task is not None:
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    # Local SSE and model traffic must not be sent through download proxies.
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = ",".join(filter(None, [os.environ.get(key), "127.0.0.1", "localhost"]))
    try:
        if args.server_metadata:
            write_json(root / "server_metadata.json", json.loads(args.server_metadata.read_text()))
        versions = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True, capture_output=True, check=True
        ).stdout.strip()
        write_json(
            root / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "client_packages": versions,
                "git_revision": revision,
                "source_sha256": {
                    name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                    for name in (
                        "collect_traces.py",
                        "trace_services.py",
                        "rag_agent.py",
                        "wiki_retriever_mcp.py",
                        "rag_data.py",
                        "embedding_download.py",
                    )
                },
            },
        )
        check_ports([args.vllm_port, args.mcp_port, *range(args.proxy_port, args.proxy_port + args.concurrency)])
        import pandas as pd

        if args.dataset.name == "dataset_tiny.parquet":
            ensure_example_data(args.dataset.parent, insecure=args.insecure_download)
        frame = pd.read_parquet(args.dataset)
        if not {"id", "question", "answer"}.issubset(frame.columns) or len(frame) < args.tasks:
            raise ValueError("Dataset must have id/question/answer columns and at least --tasks rows")
        tasks = frame.sample(n=args.tasks, random_state=args.seed)[["id", "question", "answer"]].to_dict("records")
        tasks = [{key: str(value) for key, value in task.items()} for task in tasks]
        if len({task["id"] for task in tasks}) != len(tasks):
            raise ValueError("Selected task IDs must be unique")
        write_json(root / "selected_tasks.json", tasks)
        write_json(
            root / "dataset_metadata.json",
            {
                "path": str(args.dataset.resolve()),
                "sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "rows": len(frame),
            },
        )
        commands, overrides = service_commands(config)
        write_json(
            root / "services.json",
            {
                "commands": commands,
                "vllm_environment": {
                    **{key: value for key, value in os.environ.items() if key.startswith(("HCCL_", "ASCEND_"))},
                    **overrides,
                },
                "reference_stack": {"CANN": "9.0.0", "vllm": "0.22.1", "vllm-ascend": "0.22.1rc1"},
                "started_at": time.time(),
            },
        )
        await processes.start("mcp", commands["mcp"])
        await processes.start("vllm", commands["vllm"], env={**os.environ, **overrides})
        await processes.ready(config)
        write_json(
            root / "services_ready.json",
            {"time": time.time(), "pids": {name: child.pid for name, child in processes.children.items()}},
        )
        workers = []
        for worker_id in range(args.concurrency):
            workers.append(
                await processes.start(
                    f"worker-{worker_id}",
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        "--output",
                        str(root),
                        "--worker",
                        str(worker_id),
                    ],
                )
            )
        codes = await processes.wait_workers(workers)
        write_json(root / "completion.json", {"worker_exit_codes": codes, "finished_at": time.time()})
        await processes.stop()
        analysis = await processes.start(
            "analysis", [sys.executable, str(Path(__file__).with_name("analyze_traces.py")), "--input", str(root)]
        )
        if await analysis.wait():
            raise RuntimeError("Capture finished, but analysis failed; raw traces are retained")
        print(f"Benefit table: {root / 'analysis' / 'report.md'}", flush=True)
    except BaseException as error:
        await processes.stop()
        print(f"Collection stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        for path in root.glob("*.log"):
            with path.open("rb") as log:
                log.seek(max(0, path.stat().st_size - 6000))
                print(f"{path.name}:\n{log.read().decode(errors='replace')}", file=sys.stderr)
        usable = False
        if (root / "calls.jsonl").exists():
            for line in (root / "calls.jsonl").read_text().splitlines():
                try:
                    usable |= bool(json.loads(line).get("token_ids_valid"))
                except json.JSONDecodeError:
                    continue
        if usable:
            write_json(
                root / "failure.json",
                {"error_type": type(error).__name__, "error": str(error), "partial": True, "time": time.time()},
            )
            print(f"Partial traces retained: {root}", file=sys.stderr, flush=True)
        else:
            shutil.rmtree(root)
            print("No usable model calls; removed the failed run directory.", file=sys.stderr, flush=True)
        raise
    finally:
        await processes.stop()
        loop.remove_signal_handler(signal.SIGTERM)


if __name__ == "__main__":
    asyncio.run(collect(arguments()))
