# Copyright (c) Microsoft. All rights reserved.

"""Bound online chat requests using the serving model's actual token budget."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, cast

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RolloutContextMiddleware(BaseHTTPMiddleware):
    """Reserve output space before vLLM renders and tokenizes chat requests.

    vLLM applies left truncation to the rendered token sequence, including chat
    template and tool tokens. Returned prompt token IDs therefore describe the
    actual model input even when old context has been discarded. This also
    covers environment-model requests that bypass the training trace proxy.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method != "POST" or request.url.path != "/v1/chat/completions":
            return await call_next(request)
        try:
            raw_payload = await request.json()
        except (ValueError, UnicodeDecodeError):
            return await call_next(request)
        if not isinstance(raw_payload, dict):
            return await call_next(request)
        payload = cast(dict[str, Any], raw_payload)

        model_config = request.app.state.vllm_config.model_config
        context_length = int(model_config.max_model_len)
        output_limit = int(model_config.override_generation_config["max_new_tokens"])
        if not 0 < output_limit < context_length:
            raise ValueError("Rollout max_new_tokens must be positive and smaller than max_model_len.")

        requested = payload.get("max_completion_tokens")
        if requested is None:
            requested = payload.get("max_tokens")
        if requested is not None and (type(requested) is not int or requested <= 0):
            # Leave malformed sampling parameters to vLLM's normal validation.
            return await call_next(request)
        output_tokens = min(requested, output_limit) if requested is not None else output_limit
        input_limit = context_length - output_tokens
        truncation = payload.get("truncate_prompt_tokens")
        if truncation is not None:
            if type(truncation) is not int or truncation < -1:
                return await call_next(request)
            if truncation >= 0:
                input_limit = min(input_limit, truncation)

        payload.pop("max_tokens", None)
        payload["max_completion_tokens"] = output_tokens
        payload["truncate_prompt_tokens"] = input_limit
        if payload.get("truncation_side") is None:
            payload["truncation_side"] = "left"

        body = json.dumps(payload).encode("utf-8")
        request.scope["headers"] = [
            (name, value) for name, value in request.scope["headers"] if name.lower() != b"content-length"
        ] + [(b"content-length", str(len(body)).encode("ascii"))]
        # BaseHTTPMiddleware forwards the cached body to downstream consumers.
        request._body = body  # pyright: ignore[reportPrivateUsage]
        return await call_next(request)
