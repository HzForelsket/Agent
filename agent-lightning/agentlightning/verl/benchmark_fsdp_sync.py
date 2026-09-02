# Copyright (c) Microsoft. All rights reserved.

"""VERL external-library hook for bounded-memory end-to-end benchmarks.

VERL 0.9 normally defers FSDP gradient synchronization across PPO micro-batches.
The 30B 2WikiMQA benchmark imports this module through ``external_lib`` so each
micro-batch synchronizes immediately instead of retaining full gradients until
the last micro-batch. Both benchmark modes use the same hook.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


@contextmanager
def _sync_every_micro_batch(self: FSDPEngine, *, is_last_micro_batch: bool) -> Generator[None, None, None]:
    """Reduce-scatter every micro-batch instead of retaining full gradients."""
    del self, is_last_micro_batch
    yield


FSDPEngine._gradient_sync_context = _sync_every_micro_batch  # type: ignore[method-assign]
