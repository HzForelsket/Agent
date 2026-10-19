# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""Agent Lightning instrumentation for VERL 0.9's vLLM HTTP server."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress

import psutil
import ray
from verl.workers.rollout.vllm_rollout.utils import SuppressSignalInThread
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from vllm.entrypoints.cli.serve import run_headless

from agentlightning.instrumentation.vllm import instrument_vllm

logger = logging.getLogger(__name__)
_replicas: list[vLLMReplica] = []


def _unwrap_ray_remote(cls):
    return getattr(cls, "__ray_actor_class__", cls)


@ray.remote(num_cpus=1)
class PatchedvLLMServer(_unwrap_ray_remote(vLLMHttpServer)):
    """Install Agent Lightning tracing before VERL constructs vLLM's OpenAI server."""

    def __init__(self, *args, **kwargs):
        self._closing = False
        self._shutdown_descendants = None
        instrument_vllm()
        super().__init__(*args, **kwargs)

    async def run_server(self, args):
        """Expose the configured logical name instead of the local weight path."""
        served_model_name = self.config.prometheus.served_model_name
        if served_model_name:
            args.served_model_name = [str(served_model_name)]
        await super().run_server(args)

    async def run_headless(self, args) -> None:
        """Keep upstream failure propagation without exiting during teardown."""
        args.api_server_count = 0

        def run() -> None:
            with SuppressSignalInThread():
                run_headless(args)

        def done(future: asyncio.Future) -> None:
            if not self._closing:
                try:
                    future.result()
                except BaseException:
                    logger.exception("Headless vLLM engine failed")
                else:
                    logger.error("Headless vLLM engine exited unexpectedly")
                os._exit(1)

        self.task = asyncio.create_task(asyncio.to_thread(run))
        self.task.add_done_callback(done)

    async def prepare_shutdown(self) -> None:
        """Mark every node as closing before the master stops remote engines."""
        self._closing = True
        if self._shutdown_descendants is None:
            self._shutdown_descendants = psutil.Process().children(recursive=True)

    async def shutdown(self) -> None:
        """Close the engine and reap only this server's own subprocess tree.

        Snapshot descendants before engine shutdown, which may orphan worker
        grandchildren. psutil Process objects retain PID creation times so a
        reused PID cannot cause an unrelated process to be signalled.
        """
        await self.prepare_shutdown()
        descendants = self._shutdown_descendants
        try:
            server_task = getattr(self, "_server_task", None)
            if server_task is not None:
                server_task.cancel()
                await asyncio.wait_for(asyncio.gather(server_task, return_exceptions=True), timeout=5)
            # `engine` is assigned late in upstream startup; initialization
            # failures can leave subprocesses without this attribute.
            engine = getattr(self, "engine", None)
            if engine is not None:
                engine.shutdown(timeout=10)
        finally:
            for process in reversed(descendants):
                with suppress(psutil.NoSuchProcess):
                    process.terminate()
            _, alive = psutil.wait_procs(descendants, timeout=5)
            for process in alive:
                with suppress(psutil.NoSuchProcess):
                    process.kill()
            _, alive = psutil.wait_procs(alive, timeout=5)
            if alive:
                raise RuntimeError(f"vLLM subprocesses did not exit: {[process.pid for process in alive]}")


def shutdown_vllm_servers() -> None:
    """Shut down all replicas created by this dedicated TaskRunner process."""
    replicas = list(_replicas)
    _replicas.clear()
    servers = [server for replica in replicas for server in replica.servers]
    errors: list[Exception] = []
    try:
        # Prepare all nodes before shutting down the master. Otherwise a remote
        # headless server can treat master shutdown as an unexpected engine exit.
        for method in ("prepare_shutdown", "shutdown"):
            pending = []
            for server in servers:
                try:
                    pending.append(getattr(server, method).remote())
                except Exception as error:
                    logger.exception("Failed to dispatch vLLM %s", method)
                    errors.append(error)
            for result in pending:
                try:
                    ray.get(result, timeout=30)
                except Exception as error:
                    logger.exception("Failed to complete vLLM %s", method)
                    errors.append(error)
    finally:
        for server in servers:
            try:
                ray.kill(server, no_restart=True)
            except Exception as error:
                logger.exception("Failed to terminate a vLLM server actor")
                errors.append(error)
        # Standalone reward/teacher replicas may own pools outside the trainer's
        # resource pool manager. Hybrid pools are removed by TaskRunner.close.
        from ray.util.placement_group import remove_placement_group
        from verl.workers.rollout.replica import RolloutMode

        for replica in replicas:
            if replica.rollout_mode == RolloutMode.STANDALONE and replica.resource_pool is not None:
                for placement_group in replica.resource_pool.pgs or []:
                    try:
                        remove_placement_group(placement_group)
                    except Exception as error:
                        logger.exception("Failed to remove a standalone rollout placement group")
                        errors.append(error)
    if errors:
        raise RuntimeError("vLLM cleanup did not complete successfully") from errors[0]


def install_vllm_server_patch() -> None:
    """Make VERL 0.9 replicas use the instrumented HTTP server class.

    VERL 0.9 retains ``custom_async_server`` in its configuration but constructs
    ``vLLMHttpServer`` directly.  Patch the replica constructor until the config
    hook is wired upstream.
    """
    if getattr(vLLMReplica, "_agentlightning_patched", False):
        return
    original_init = vLLMReplica.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.server_class = PatchedvLLMServer
        _replicas.append(self)

    vLLMReplica.__init__ = patched_init
    vLLMReplica._agentlightning_patched = True
