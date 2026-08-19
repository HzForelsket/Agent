# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""PrefixGrouper integration for VERL's data-parallel actor."""

from __future__ import annotations

import logging
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

import torch
from packaging.version import Version
from prefix_grouper import PrefixGrouper
from torch import nn
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

logger = logging.getLogger(__name__)

PREFIX_GROUPER_ATTENTION = "agentlightning_prefix_grouper"

__all__ = [
    "PREFIX_GROUPER_ATTENTION",
    "PrefixGrouperActorRolloutRefWorker",
    "PrefixGrouperAsyncActorRolloutRefWorker",
    "PrefixGrouperDataParallelPPOActor",
    "register_prefix_grouper_attention",
    "reorder_by_prompt",
]


def _repeat_kv(hidden_states: torch.Tensor, num_attention_heads: int) -> torch.Tensor:
    """Expand grouped-query key/value heads to the number of query heads."""
    num_key_value_heads = hidden_states.shape[1]
    if num_key_value_heads == num_attention_heads:
        return hidden_states
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            f"The number of attention heads ({num_attention_heads}) must be divisible by the number of "
            f"key/value heads ({num_key_value_heads})."
        )
    return hidden_states.repeat_interleave(num_attention_heads // num_key_value_heads, dim=1)


def _causal_padding_mask(
    padding_mask: torch.Tensor,
    query_length: int,
    key_value_length: int,
) -> torch.Tensor:
    """Create PrefixGrouper's causal boolean SDPA mask from its 2-D padding mask."""
    padding_mask = padding_mask.bool()
    suffix_offset = key_value_length - query_length
    if suffix_offset < 0:
        raise ValueError(f"Key/value length ({key_value_length}) cannot be smaller than query length ({query_length}).")

    query_mask = padding_mask[:, -query_length:]
    key_value_mask = padding_mask[:, :key_value_length]
    query_positions = torch.arange(query_length, device=padding_mask.device).unsqueeze(-1)
    key_positions = torch.arange(key_value_length, device=padding_mask.device).unsqueeze(0)
    causal_mask = key_positions <= query_positions + suffix_offset
    return query_mask[:, None, :, None] & key_value_mask[:, None, None, :] & causal_mask[None, None, :, :]


def _prefix_grouper_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *args: Any,
    prefix_grouper: PrefixGrouper | None = None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, None]:
    """Run SDPA normally or split it into shared-prefix and suffix attention."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if prefix_grouper is None:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    def attention_forward(
        inner_query: torch.Tensor,
        inner_key: torch.Tensor,
        inner_value: torch.Tensor,
        padding_mask: torch.Tensor,
        *_args: Any,
        **_kwargs: Any,
    ) -> torch.Tensor:
        inner_key = _repeat_kv(inner_key, inner_query.shape[1])
        inner_value = _repeat_kv(inner_value, inner_query.shape[1])
        causal_mask = _causal_padding_mask(padding_mask, inner_query.shape[2], inner_key.shape[2])
        output = torch.nn.functional.scaled_dot_product_attention(
            inner_query,
            inner_key,
            inner_value,
            attn_mask=causal_mask,
            dropout_p=dropout,
            scale=scaling,
        )
        return output.transpose(1, 2).contiguous()

    return prefix_grouper.forward(attention_forward, query, key, value), None


def register_prefix_grouper_attention() -> None:
    """Register the custom attention and its standard SDPA fallback mask."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(PREFIX_GROUPER_ATTENTION, _prefix_grouper_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        PREFIX_GROUPER_ATTENTION,
        ALL_MASK_ATTENTION_FUNCTIONS["sdpa"],
    )


def _prompt_groups(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_length: int,
) -> List[List[int]]:
    """Group row indices whose non-padding prompt token IDs are identical."""
    prompt_ids = input_ids[:, :-response_length]
    prompt_mask = attention_mask[:, :-response_length].bool()
    groups: OrderedDict[Tuple[int, ...], List[int]] = OrderedDict()
    for row_index in range(input_ids.shape[0]):
        key = tuple(prompt_ids[row_index][prompt_mask[row_index]].detach().cpu().tolist())
        groups.setdefault(key, []).append(row_index)
    return list(groups.values())


def reorder_by_prompt(batch: Any) -> None:
    """Place larger identical-prompt groups first so VERL micro-batches keep them together."""
    response_length = batch.batch["responses"].shape[-1]
    groups = _prompt_groups(batch.batch["input_ids"], batch.batch["attention_mask"], response_length)
    groups.sort(key=lambda indices: -len(indices))
    order = [row_index for indices in groups for row_index in indices]
    batch.reorder(torch.tensor(order, dtype=torch.int32))


class PrefixGrouperDataParallelPPOActor(DataParallelPPOActor):
    """VERL actor that uses one prompt forward for responses sharing exact token prefixes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.use_remove_padding:
            raise ValueError("PrefixGrouper requires actor_rollout_ref.model.use_remove_padding=false.")
        if self.use_fused_kernels:
            raise ValueError("PrefixGrouper does not support VERL fused kernels.")
        if self.ulysses_sequence_parallel_size != 1:
            raise ValueError("PrefixGrouper does not support Ulysses sequence parallelism.")

    def update_policy(self, data: Any) -> Any:
        # VERL's global token balancing can interleave prompt groups. Restore
        # group locality within each data-parallel rank before making micro-batches.
        reorder_by_prompt(data)
        return super().update_policy(data)

    def _forward_micro_batch(
        self,
        micro_batch: Dict[str, Any],
        temperature: float,
        calculate_entropy: bool = False,
    ) -> Tuple[torch.Tensor | None, torch.Tensor]:
        if "multi_modal_inputs" in micro_batch:
            if not getattr(self, "_prefix_grouper_warned_multimodal", False):
                logger.warning("PrefixGrouper currently supports text-only batches; using the standard VERL forward.")
                self._prefix_grouper_warned_multimodal = True
            return super()._forward_micro_batch(micro_batch, temperature, calculate_entropy)

        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.shape[-1]
        prompt_length = input_ids.shape[-1] - response_length
        groups = _prompt_groups(input_ids, attention_mask, response_length)
        if all(len(indices) == 1 for indices in groups):
            return super()._forward_micro_batch(micro_batch, temperature, calculate_entropy)

        order = torch.tensor(
            [row_index for indices in groups for row_index in indices],
            dtype=torch.long,
            device=input_ids.device,
        )
        representatives = torch.tensor(
            [indices[0] for indices in groups],
            dtype=torch.long,
            device=input_ids.device,
        )
        inverse_order = torch.empty_like(order)
        inverse_order[order] = torch.arange(order.shape[0], device=order.device)

        prompt_ids = input_ids[representatives, :prompt_length]
        prompt_mask = attention_mask[representatives, :prompt_length]
        response_ids = responses[order]
        response_mask = attention_mask[order, prompt_length:]
        prefix_grouper = PrefixGrouper.from_ungrouped_masks(
            prefix_mask=prompt_mask,
            suffix_mask=response_mask,
            group_sizes=[len(indices) for indices in groups],
            device=input_ids.device,
            padding_mode="right",
        )
        grouped_input_ids = prefix_grouper.concat_input(prompt_ids, prompt_mask, response_ids, response_mask)
        grouped_position_ids = prefix_grouper.concat_input(
            position_ids[representatives, :prompt_length],
            prompt_mask,
            position_ids[order, prompt_length:],
            response_mask,
        )

        autocast_context = (
            torch.autocast(device_type=self.device_name, dtype=self.param_dtype)
            if self.param_dtype != torch.float32
            else nullcontext()
        )
        with autocast_context:
            output = self.actor_module(
                input_ids=grouped_input_ids,
                attention_mask=prefix_grouper.padding_mask,
                position_ids=grouped_position_ids,
                use_cache=False,
                prefix_grouper=prefix_grouper,
            )
            _, _, suffix_logits, _ = prefix_grouper.split_output(output.logits, include_prefix_last=1)
            suffix_logits = suffix_logits[:, :-1]
            target_ids = prefix_grouper.convert_padding(response_ids, response_mask, padding_mode="right")
            suffix_logits = suffix_logits / temperature
            compact_log_probs = logprobs_from_logits(suffix_logits, target_ids)
            compact_entropy = self.compute_entropy_from_logits(suffix_logits) if calculate_entropy else None

        compact_length = compact_log_probs.shape[-1]
        log_probs = compact_log_probs.new_zeros((input_ids.shape[0], response_length))
        log_probs[:, :compact_length] = compact_log_probs[inverse_order]
        entropy = None
        if compact_entropy is not None:
            entropy = compact_entropy.new_zeros((input_ids.shape[0], response_length))
            entropy[:, :compact_length] = compact_entropy[inverse_order]

        return entropy, log_probs


class _PrefixGrouperWorkerMixin:
    def _build_model_optimizer(self, *args: Any, **kwargs: Any) -> Any:
        override_model_config = dict(kwargs.get("override_model_config", {}))
        override_model_config["attn_implementation"] = PREFIX_GROUPER_ATTENTION
        kwargs["override_model_config"] = override_model_config
        return super()._build_model_optimizer(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        import verl
        import verl.workers.actor

        if Version(verl.__version__) < Version("0.6.0"):
            raise RuntimeError(f"PrefixGrouper requires VERL >= 0.6.0, found {verl.__version__}.")

        register_prefix_grouper_attention()
        original_actor_cls = verl.workers.actor.DataParallelPPOActor
        verl.workers.actor.DataParallelPPOActor = PrefixGrouperDataParallelPPOActor
        try:
            super().init_model()
        finally:
            verl.workers.actor.DataParallelPPOActor = original_actor_cls


class PrefixGrouperActorRolloutRefWorker(_PrefixGrouperWorkerMixin, ActorRolloutRefWorker):
    """Synchronous FSDP worker with PrefixGrouper-enabled actor forwards."""


class PrefixGrouperAsyncActorRolloutRefWorker(_PrefixGrouperWorkerMixin, AsyncActorRolloutRefWorker):
    """Asynchronous FSDP worker with PrefixGrouper-enabled actor forwards."""
