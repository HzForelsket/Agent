# Copyright (c) Microsoft. All rights reserved.

"""Compatibility hooks for vLLM token-id instrumentation.

vLLM 0.22 exposes prompt token IDs on ``ChatCompletionResponse`` and response
token IDs on each choice when ``return_token_ids`` is enabled.  Agent
Lightning therefore no longer needs to monkeypatch vLLM's serving internals.
The public functions remain as no-ops so existing callers keep working.
"""

from __future__ import annotations

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)

__all__ = [
    "instrument_vllm",
    "uninstrument_vllm",
]


def instrument_vllm() -> None:
    """Validate the native vLLM 0.22 token-id response fields."""
    if "prompt_token_ids" not in ChatCompletionResponse.model_fields:
        raise RuntimeError("vLLM does not expose ChatCompletionResponse.prompt_token_ids")
    if "token_ids" not in ChatCompletionResponseChoice.model_fields:
        raise RuntimeError("vLLM does not expose ChatCompletionResponseChoice.token_ids")


def uninstrument_vllm() -> None:
    """No-op because vLLM 0.22 provides token IDs without monkeypatching."""
    return None
