"""Optional Python stage markers for isolated host-profile captures."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import torch

_ENABLED = ContextVar("prefix_grouper_host_profile", default=False)
HOST_PROBE_VERSION = 1


@contextmanager
def profile_host_stages():
    token = _ENABLED.set(True)
    try:
        yield
    finally:
        _ENABLED.reset(token)


def host_stage(name: str):
    # No record_function calls outside the explicit capture context.
    return torch.profiler.record_function(name) if _ENABLED.get() else nullcontext()
