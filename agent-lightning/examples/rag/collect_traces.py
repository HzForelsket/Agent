# Copyright (c) Microsoft. All rights reserved.

"""Collect complete RAG trajectories through a vLLM endpoint; see TRACE_COLLECTION.md."""

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
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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
    parser.add_argument("--endpoint", default="http://127.0.0.1:18030/v1", help="vLLM OpenAI base URL, ending in /v1.")
    parser.add_argument("--model", default="Qwen3-30B-A3B-Instruct-2507", help="Served model name, not weight path.")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "data/dataset_tiny.parquet")
    parser.add_argument("--mcp-url", default="http://127.0.0.1:8099/sse")
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
    if args.rollouts_per_task < 2:
        parser.error("rollouts-per-task must be at least 2 for cross-trajectory sharing")
    if not 0 <= args.temperature <= 2 or not 0 <= args.seed < 2**32:
        parser.error("temperature must be in [0, 2] and seed in [0, 2**32)")
    if args.proxy_port + args.concurrency - 1 > 65535:
        parser.error("proxy port range exceeds 65535")
    endpoint = urlsplit(args.endpoint)
    if endpoint.scheme not in {"http", "https"} or not endpoint.netloc or not endpoint.path.endswith("/v1"):
        parser.error("endpoint must be an HTTP(S) base URL ending in /v1")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        parser.error("put API credentials in VLLM_API_KEY, not the endpoint URL")
    if args.worker is None and not args.dataset.is_file():
        parser.error(
            f"Dataset not found: {args.dataset.resolve()}\n"
            "Example data is not included in Git and is not downloaded automatically. "
            "Follow TRACE_COLLECTION.md, section 2, to download the dataset and retrieval corpus, "
            "or pass --dataset /absolute/path/to/dataset_tiny.parquet."
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
        endpoint=args.endpoint.rstrip("/"),
        created_at=time.time(),
    )
    write_json(root / "config.json", config)
    processes: list[asyncio.subprocess.Process] = []
    logs: list[Any] = []
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
                    for name in ("collect_traces.py", "rag_agent.py", "wiki_retriever_mcp.py")
                },
            },
        )
        import pandas as pd

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
        for worker_id in range(args.concurrency):
            log = (root / f"worker-{worker_id}.log").open("w")
            logs.append(log)
            processes.append(
                await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--output",
                    str(root),
                    "--worker",
                    str(worker_id),
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                )
            )
        codes = await asyncio.gather(*(process.wait() for process in processes))
        if any(codes):
            raise RuntimeError(f"Worker exit codes: {codes}")
        write_json(root / "completion.json", {"worker_exit_codes": codes, "finished_at": time.time()})
        analysis = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).with_name("analyze_traces.py")), "--input", str(root)
        )
        if await analysis.wait():
            raise RuntimeError("Capture finished, but analysis failed; raw traces are retained")
    except BaseException as error:
        for process in processes:
            if process.returncode is None:
                process.terminate()
        try:
            await asyncio.wait_for(asyncio.gather(*(process.wait() for process in processes)), timeout=10)
        except asyncio.TimeoutError:
            for process in processes:
                if process.returncode is None:
                    process.kill()
            await asyncio.gather(*(process.wait() for process in processes))
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
            for log in logs:
                log.flush()
            for path in root.glob("worker-*.log"):
                print(f"{path.name}:\n{path.read_text()[-6000:]}", file=sys.stderr)
            shutil.rmtree(root)
            print("No usable model calls; removed the failed run directory.", file=sys.stderr, flush=True)
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    asyncio.run(collect(arguments()))
