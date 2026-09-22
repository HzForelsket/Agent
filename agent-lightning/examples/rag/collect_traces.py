# Copyright (c) Microsoft. All rights reserved.

"""Collect RAG, original SQL or original 20 Questions trajectories with managed NPU vLLM."""

import argparse
import asyncio
import contextvars
import faulthandler
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag_data import DEFAULT_DATA_DIR, add_download_argument
from trace_services import WORKER_TIMEOUT_EXIT_CODE, Processes, check_ports, service_commands
from trace_tasks import prepare_tasks

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
    parser.add_argument("--agent", choices=("rag", "sql", "q20"), default="rag")
    parser.add_argument(
        "--sql-database-dir", type=Path, help="Directory containing original Spider database/DB_ID/DB_ID.sqlite files."
    )
    parser.add_argument(
        "--q20-search", action="store_true", help="Enable the original optional Q20 simulated-search tool."
    )
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
    parser.add_argument("--dataset", type=Path, help="Default: cached RAG/Spider parquet or bundled Q20 nouns CSV.")
    add_download_argument(parser)
    parser.add_argument("--tasks", type=positive, default=32)
    parser.add_argument("--rollouts-per-task", type=positive, default=4)
    parser.add_argument("--concurrency", type=positive, default=4)
    parser.add_argument(
        "--max-model-calls",
        type=positive,
        help="Safety cap per role: RAG 8; SQL/Q20 128. Does not replace original flow limits.",
    )
    parser.add_argument("--max-tokens-per-call", type=positive, default=2048)
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="RAG/SQL sampling; Q20 preserves original CrewLLM defaults."
    )
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--proxy-port", type=positive, default=18031)
    parser.add_argument(
        "--trajectory-timeout",
        type=positive,
        default=900,
        help="Seconds for a whole trajectory, including all model calls, tools and evaluation (default: 900).",
    )
    parser.add_argument(
        "--server-metadata", type=Path, help="JSON describing serving hardware, versions and launch command."
    )
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        return args
    if args.max_model_calls is None:
        args.max_model_calls = 8 if args.agent == "rag" else 128
    if args.agent == "sql" and args.max_tokens_per_call != 2048:
        parser.error("Original SQL workflow fixes max_tokens=2048; capture does not change it")
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
    if (
        args.dataset
        and not args.dataset.is_file()
        and not (args.agent == "rag" and args.dataset.name == "dataset_tiny.parquet")
    ):
        parser.error(
            f"Custom dataset not found: {args.dataset.resolve()}. "
            "Omit --dataset to prepare the bundled dataset automatically."
        )
    return args


async def run_worker(root: Path, config: dict[str, Any], worker_id: int) -> None:
    # Each process owns a Lightning tracer; concurrent rollouts in one thread would conflict.
    os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
    os.environ["OPENAI_API_KEY"] = "local-capture-proxy"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CREWAI_TELEMETRY_DISABLED"] = "true"
    os.environ["CREWAI_TRACING_ENABLED"] = "false"
    import httpx
    import uvicorn
    from agents import RunHooks, set_tracing_disabled
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    import agentlightning as agl

    set_tracing_disabled(True)
    role_counts: dict[tuple[str, str], int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    valid_counts: dict[str, int] = defaultdict(int)
    finish_reasons: dict[str, list[str]] = defaultdict(list)
    identities: dict[str, dict[str, Any]] = {}
    app = FastAPI()
    headers = {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"} if os.environ.get("VLLM_API_KEY") else {}

    async def forward(trajectory_id: str, request: Request, role: str) -> JSONResponse:
        if trajectory_id not in identities or role not in {"policy", "answerer", "search"}:
            return JSONResponse({"error": {"message": "unknown trajectory"}}, status_code=404)
        turn = role_counts[trajectory_id, role]
        role_counts[trajectory_id, role] += 1
        if role == "policy":
            counts[trajectory_id] += 1
        payload = await request.json()
        identity = identities[trajectory_id]
        payload["return_token_ids"] = True
        sample = identity["task_index"] * config["rollouts_per_task"] + identity["sample_index"]
        payload["seed"] = (
            config["seed"]
            + sample * config["max_model_calls"]
            + turn
            + ("policy", "answerer", "search").index(role) * 2**40
        ) % (2**63)
        record = {
            "trajectory_id": trajectory_id,
            **identity,
            "turn": turn,
            "role": role,
            "started_at": time.time(),
            "request": payload,
            "token_ids_valid": False,
        }
        try:
            if turn >= config["max_model_calls"]:
                raise RuntimeError("Model call safety cap exceeded; original workflow did not finish")
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
                if role == "policy":
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
            append(root, "calls.jsonl" if role == "policy" else "environment_calls.jsonl", record)

    @app.post("/{trajectory_id}/v1/chat/completions")
    async def forward_policy(trajectory_id: str, request: Request) -> JSONResponse:
        return await forward(trajectory_id, request, "policy")

    @app.post("/{trajectory_id}/{role}/v1/chat/completions")
    async def forward_role(trajectory_id: str, role: str, request: Request) -> JSONResponse:
        return await forward(trajectory_id, request, role)

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
        if config["agent"] == "rag":
            from rag_agent import RAGAgent

            agent = RAGAgent(mcp_server_url=config["mcp_url"], max_turns=config["max_model_calls"], hooks=Hooks())
        else:
            from trace_workflows import make_agent

            def record_workflow(name: str, value: dict[str, Any]) -> None:
                append(root, name, {"trajectory_id": CURRENT.get(), "time": time.time(), **value})

            agent = make_agent(config, record_workflow)
        runner.init(agent)
        runner.init_worker(worker_id, store)
        tasks = json.loads((root / "selected_tasks.json").read_text())
        resume_path = root / f"worker-{worker_id}-resume.json"
        next_sample = json.loads(resume_path.read_text())["next_sample"] if resume_path.exists() else worker_id
        print(f"Worker {worker_id}: starting at sample index {next_sample}", flush=True)
        for task_index, task in enumerate(tasks):
            for sample_index in range(config["rollouts_per_task"]):
                sample = task_index * config["rollouts_per_task"] + sample_index
                if sample < next_sample or sample % config["concurrency"] != worker_id:
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
                trajectory_started = time.monotonic()
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
                except asyncio.TimeoutError as error:
                    elapsed = time.monotonic() - trajectory_started
                    info["status"] = "failed"
                    info["error_type"] = type(error).__name__
                    info["timeout_seconds"] = config["trajectory_timeout"]
                    info["error"] = (
                        f"Trajectory timed out after {elapsed:.1f}s "
                        f"(whole-trajectory limit: {config['trajectory_timeout']}s; "
                        f"policy requests received: {counts[trajectory_id]}, "
                        f"valid responses: {valid_counts[trajectory_id]}). "
                        f"Cause: {error!r}. Inspect calls.jsonl and worker log thread stacks "
                        "for model, tool or evaluation delays."
                    )
                    logging.exception("Trajectory stopped: %s: %s", trajectory_id, info["error"])
                    # Cancellation of to_thread leaves the original workflow running. Capture its
                    # actual stack before exiting, rather than only the cancelled await's traceback.
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
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
                        elapsed_seconds=time.monotonic() - trajectory_started,
                        model_calls=counts[trajectory_id],
                        valid_model_calls=valid_counts[trajectory_id],
                        role_model_calls={
                            role: role_counts[trajectory_id, role] for role in ("policy", "answerer", "search")
                        },
                    )
                    append(root, "trajectories.jsonl", info)
                    print(f"{trajectory_id}: {info['status']}, calls={counts[trajectory_id]}", flush=True)
                if info.get("error_type") == "TimeoutError" and config["agent"] != "rag":
                    # Persist progress only after the failed trajectory is durable. A new process
                    # must replace this one because cancellation cannot stop its workflow thread.
                    write_json(
                        resume_path,
                        {
                            "worker_id": worker_id,
                            "trajectory_id": trajectory_id,
                            "next_sample": sample + config["concurrency"],
                        },
                    )
                    print(
                        "Original workflow timed out; requesting worker replacement for remaining samples", flush=True
                    )
                    os._exit(WORKER_TIMEOUT_EXIT_CODE)
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
        dataset=str(args.dataset.resolve()) if args.dataset else None,
        statistics_unit="one_complete_trajectory_sequence" if args.agent == "rag" else "original_workflow_call_slots",
        workflow_implementation={
            "rag": "rag.RAGAgent",
            "sql": "spider.LitSQLAgent",
            "q20": "tinker.TwentyQuestionsFlow",
        }[args.agent],
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
                        "trace_tasks.py",
                        "trace_workflows.py",
                        "../spider/sql_agent.py",
                        "../tinker/q20_agent.py",
                        "rag_agent.py",
                        "wiki_retriever_mcp.py",
                        "rag_data.py",
                        "embedding_download.py",
                    )
                },
            },
        )
        ports = [args.vllm_port, *range(args.proxy_port, args.proxy_port + args.concurrency)]
        if args.agent == "rag":
            ports.append(args.mcp_port)
        check_ports(ports)
        if args.agent != "rag":
            dependency_check = await processes.start(
                "dependencies",
                [sys.executable, "-u", str(Path(__file__).with_name("trace_workflows.py")), "--agent", args.agent],
            )
            if await dependency_check.wait():
                raise RuntimeError(
                    "Original workflow imports failed; install requirements-workflow-traces.txt in the collector environment"
                )
        tasks, dataset_metadata = prepare_tasks(config)
        write_json(root / "selected_tasks.json", tasks)
        write_json(root / "dataset_metadata.json", dataset_metadata)
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
        if "mcp" in commands:
            await processes.start("mcp", commands["mcp"])
        await processes.start("vllm", commands["vllm"], env={**os.environ, **overrides})
        await processes.ready(config)
        write_json(
            root / "services_ready.json",
            {"time": time.time(), "pids": {name: child.pid for name, child in processes.children.items()}},
        )

        async def start_worker(worker_id: int) -> asyncio.subprocess.Process:
            return await processes.start(
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

        next_samples = list(range(args.concurrency))
        worker_timeout_counts = [0] * args.concurrency

        async def restart_worker(worker_id: int) -> asyncio.subprocess.Process:
            resume = json.loads((root / f"worker-{worker_id}-resume.json").read_text())
            next_sample = resume["next_sample"]
            # Require durable forward progress so a stale checkpoint cannot cause a restart loop.
            if (
                resume["worker_id"] != worker_id
                or type(next_sample) is not int
                or next_sample <= next_samples[worker_id]
                or next_sample >= len(tasks) * args.rollouts_per_task + args.concurrency
                or next_sample % args.concurrency != worker_id
            ):
                raise RuntimeError(f"Invalid timeout checkpoint for worker {worker_id}: {resume}")
            await processes.stop([f"worker-{worker_id}"])
            processes.check_services()
            next_samples[worker_id] = next_sample
            worker_timeout_counts[worker_id] += 1
            append(
                root, "worker_restarts.jsonl", {**resume, "exit_code": WORKER_TIMEOUT_EXIT_CODE, "time": time.time()}
            )
            print(f"Skipping timed-out trajectory {resume['trajectory_id']}; replacing worker {worker_id}", flush=True)
            return await start_worker(worker_id)

        workers = [await start_worker(worker_id) for worker_id in range(args.concurrency)]
        codes = await processes.wait_workers(workers, restart_worker)
        with (root / "trajectories.jsonl").open() as handle:
            trajectory_status = Counter(json.loads(line)["status"] for line in handle)
        write_json(
            root / "completion.json",
            {
                "worker_exit_codes": codes,
                "worker_timeout_counts": worker_timeout_counts,
                "trajectory_status": dict(trajectory_status),
                "partial": any(status != "completed" for status in trajectory_status),
                "finished_at": time.time(),
            },
        )
        await processes.stop()
        analyzer = Path(__file__).resolve().parents[2] / "scripts" / "analyze_multiturn_sharing.py"
        analysis = await processes.start(
            "analysis",
            [
                sys.executable,
                str(analyzer),
                "--view",
                "trajectory" if args.agent == "rag" else "calls",
                "--input",
                str(root),
                "--output-dir",
                str(root / "analysis"),
            ],
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
