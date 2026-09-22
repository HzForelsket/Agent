# Copyright (c) Microsoft. All rights reserved.

"""Owned subprocesses and service readiness for collect_traces.py on an NPU host."""

import asyncio
import os
import signal
import socket
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

WORKER_TIMEOUT_EXIT_CODE = 75


def check_ports(ports: list[int]) -> None:
    """Reject occupied ports before launching any service; never stop an existing listener."""
    if len(set(ports)) != len(ports):
        raise ValueError("vLLM, MCP and capture proxy ports must not overlap")
    for port in ports:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(
                    f"Local port {port} is unavailable; stop its owner or select another port"
                ) from error


class Processes:
    """Keep process groups and logs under the collection run's lifetime."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.children: dict[str, asyncio.subprocess.Process] = {}
        self.logs: dict[str, Any] = {}

    async def start(
        self, name: str, command: list[str], env: dict[str, str] | None = None
    ) -> asyncio.subprocess.Process:
        """Launch one owned process group with a continuously written log."""
        if name in self.children or name in self.logs:
            raise RuntimeError(f"Stop {name} before reusing its process slot")
        log = (self.root / f"{name}.log").open("a")
        self.logs[name] = log
        # Collection children are noninteractive. Inheriting the terminal can leave
        # CrewAI's trace-viewing input thread holding stdin during interpreter exit.
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.children[name] = process
        print(f"Started {name}: pid={process.pid}, log={self.root / (name + '.log')}", flush=True)
        return process

    def check_services(self) -> None:
        """Fail immediately if either serving process has exited."""
        for name in (name for name in ("mcp", "vllm") if name in self.children):
            process = self.children[name]
            if process.returncode is not None:
                raise RuntimeError(
                    f"{name} exited with code {process.returncode}; inspect {self.root / (name + '.log')}"
                )

    async def ready(self, config: dict[str, Any]) -> None:
        """Wait for the served model and the MCP retrieve tool before creating workers."""
        import httpx
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        deadline = time.monotonic() + config["startup_timeout"]
        next_report = 0.0
        pending = {"mcp", "vllm"} if config["agent"] == "rag" else {"vllm"}
        errors: dict[str, str] = {}
        headers = {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"} if os.environ.get("VLLM_API_KEY") else {}
        async with httpx.AsyncClient(timeout=5, trust_env=False, headers=headers) as client:
            while pending:
                self.check_services()
                for name in tuple(pending):
                    try:
                        if name == "vllm":
                            health = await client.get(config["endpoint"].removesuffix("/v1") + "/health")
                            health.raise_for_status()
                            models = await client.get(config["endpoint"] + "/models")
                            models.raise_for_status()
                            if config["model"] not in {model["id"] for model in models.json()["data"]}:
                                raise ValueError(f"Expected served model {config['model']} is absent")
                        else:
                            # The outer timeout also bounds SSE connection/initialization and teardown.
                            async def list_tools() -> list[Any]:
                                async with sse_client(config["mcp_url"], timeout=5, sse_read_timeout=5) as streams:
                                    async with ClientSession(*streams) as session:
                                        await session.initialize()
                                        return (await session.list_tools()).tools

                            names = {tool.name for tool in await asyncio.wait_for(list_tools(), timeout=8)}
                            if "retrieve" not in names:
                                raise ValueError("MCP does not expose the retrieve tool")
                        pending.remove(name)
                        print(f"Ready: {name}", flush=True)
                    except Exception as error:
                        errors[name] = f"{type(error).__name__}: {error}"
                self.check_services()
                if pending and time.monotonic() >= deadline:
                    raise TimeoutError(f"Service startup timed out: { {name: errors.get(name) for name in pending} }")
                if pending:
                    if time.monotonic() >= next_report:
                        print(f"Waiting for {', '.join(sorted(pending))}; see mcp.log / vllm.log", flush=True)
                        next_report = time.monotonic() + 30
                    await asyncio.sleep(2)

    async def wait_workers(
        self,
        workers: list[asyncio.subprocess.Process],
        restart_worker: Callable[[int], Awaitable[asyncio.subprocess.Process]],
    ) -> list[int]:
        """Replace workers after recorded trajectory timeouts; fail on other process errors."""
        while True:
            self.check_services()
            codes = [worker.returncode for worker in workers]
            if any(code is not None and code not in (0, WORKER_TIMEOUT_EXIT_CODE) for code in codes):
                raise RuntimeError(f"Worker exit codes: {codes}")
            if WORKER_TIMEOUT_EXIT_CODE in codes:
                for worker_id, code in enumerate(codes):
                    if code == WORKER_TIMEOUT_EXIT_CODE:
                        workers[worker_id] = await restart_worker(worker_id)
                continue
            if all(code is not None for code in codes):
                return [await worker.wait() for worker in workers]
            await asyncio.sleep(1)

    async def stop(self, names: list[str] | None = None) -> None:
        """Terminate selected owned groups and descendants, or all groups when names is omitted."""
        selected = set(names) if names is not None else self.children.keys() | self.logs.keys()
        children = {name: self.children[name] for name in selected if name in self.children}
        groups = {process.pid for process in children.values()}
        for pid in groups:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 15
        while groups and time.monotonic() < deadline:
            for pid in tuple(groups):
                try:
                    os.killpg(pid, 0)
                except ProcessLookupError:
                    groups.remove(pid)
            if groups:
                await asyncio.sleep(0.2)
        for pid in groups:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await asyncio.gather(*(process.wait() for process in children.values()))
        for name in selected:
            self.children.pop(name, None)
            log = self.logs.pop(name, None)
            if log is not None:
                log.flush()
                os.fsync(log.fileno())
                log.close()


def service_commands(config: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Build the pinned vLLM Ascend launch and CPU retrieval commands without invoking a shell."""
    import sys

    vllm = [
        config["vllm_python"],
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config["model_path"],
        "--served-model-name",
        config["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(config["vllm_port"]),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(len(config["npu_devices"].split(","))),
        "--enable-expert-parallel",
        "--distributed-executor-backend",
        "mp",
        "--max-model-len",
        str(config["max_model_len"]),
        "--max-num-seqs",
        str(config["concurrency"]),
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--no-enable-prefix-caching",
        "--enforce-eager",
    ]
    mcp = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("wiki_retriever_mcp.py")),
        "--host",
        "127.0.0.1",
        "--port",
        str(config["mcp_port"]),
        "--device",
        "cpu",
        "--data-dir",
        config["retrieval_data_dir"],
        "--embedding-model",
        config["embedding_model"],
        "--embedding-cache",
        config["embedding_cache"],
    ]
    if config["insecure_download"]:
        mcp.append("--insecure-download")
    if config["local_files_only"]:
        mcp.append("--local-files-only")
    overrides = {
        "ASCEND_RT_VISIBLE_DEVICES": config["npu_devices"],
        "OMP_NUM_THREADS": "1",
        "PYTORCH_NPU_ALLOC_CONF": os.environ.get("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    return ({"mcp": mcp, "vllm": vllm} if config["agent"] == "rag" else {"vllm": vllm}), overrides
