# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""Agent Lightning instrumentation for VERL 0.9's vLLM HTTP server."""

from __future__ import annotations

import ray
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from agentlightning.instrumentation.vllm import instrument_vllm


def _unwrap_ray_remote(cls):
    return getattr(cls, "__ray_actor_class__", cls)


@ray.remote(num_cpus=1)
class PatchedvLLMServer(_unwrap_ray_remote(vLLMHttpServer)):
    """Install Agent Lightning tracing before VERL constructs vLLM's OpenAI server."""

    def __init__(self, *args, **kwargs):
        instrument_vllm()
        super().__init__(*args, **kwargs)


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

    vLLMReplica.__init__ = patched_init
    vLLMReplica._agentlightning_patched = True
