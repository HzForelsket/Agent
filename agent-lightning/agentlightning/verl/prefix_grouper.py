# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""PrefixGrouper support for the VERL 0.9 model-engine worker stack.

VERL 0.9 ships the attention wrapper and basic PrefixGrouper helpers, but its
FSDP language-model engine does not call the grouped forward path.  This module
connects that path to ``FSDPEngineWithLMHead.forward_step`` while preserving the
nested-tensor output layout expected by VERL's PPO losses.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from functools import lru_cache
from itertools import accumulate
from types import MethodType
from typing import Any, Iterable

import torch
from packaging.version import Version
from prefix_grouper import PrefixGrouper
from tensordict import TensorDict
from verl.models.transformers.monkey_patch import apply_prefix_grouper_patch as _apply_verl_prefix_grouper_patch
from verl.trainer.ppo.prefix_grouper_utils import build_position_ids_for_prefix_grouper, pg_forward
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

from .npu_fsdp_loader import NPUShardedLoadFSDPEngineWithLMHead as _NPUShardedLoadFSDPEngineWithLMHead

__all__ = [
    "PrefixGrouperActorRolloutRefWorker",
    "PrefixGrouperTrainingWorker",
    "apply_prefix_grouper_patch",
    "forward_with_prefix_grouper",
    "group_prompt_indices",
    "reorder_by_prompt",
]

_PATCHED = False
_NPU_TORCH_VERSION = Version("2.10.0")
_NPU_COMPRESSED_MASK_SIZE = 2048
_NPU_CAUSAL_MASKS: dict[torch.device, torch.Tensor] = {}


@lru_cache(maxsize=1)
def _torch_npu_module() -> Any:
    import torch_npu

    current = Version(torch_npu.__version__)
    if current.public != _NPU_TORCH_VERSION.public:
        raise RuntimeError(f"The native PrefixGrouper NPU path requires torch-npu 2.10.0, found {current}.")
    return torch_npu


def _npu_compressed_causal_mask(device: torch.device) -> torch.Tensor:
    """Return the reusable 2048x2048 compressed causal mask required by NPU FA."""
    mask = _NPU_CAUSAL_MASKS.get(device)
    if mask is None:
        # The reference/old-policy pass can call this function under
        # ``torch.inference_mode()`` before the actor update.  An inference
        # tensor cannot later be saved by the fused training operator for its
        # backward pass, so the process-wide cache must always hold a normal
        # tensor regardless of the caller's mode.
        with torch.inference_mode(False):
            mask = torch.ones((_NPU_COMPRESSED_MASK_SIZE, _NPU_COMPRESSED_MASK_SIZE), dtype=torch.bool).triu_(1)
            mask = mask.to(device=device)
        _NPU_CAUSAL_MASKS[device] = mask
    return mask


def _pack_bnsd(tensor: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Pack a padded BNSD tensor into the TND layout used by NPU fused attention."""
    if tensor.ndim != 4 or valid_mask.shape != (tensor.shape[0], tensor.shape[2]):
        raise ValueError(
            f"Cannot pack tensor {tuple(tensor.shape)} with mask {tuple(valid_mask.shape)} as BNSD -> TND."
        )
    return tensor.transpose(1, 2)[valid_mask]


def _unpack_tnd(tensor: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Restore a packed TND attention output to padded BSND layout."""
    if tensor.ndim != 3:
        raise ValueError(f"Cannot unpack non-TND tensor with shape {tuple(tensor.shape)}.")
    output = tensor.new_zeros((*valid_mask.shape, *tensor.shape[1:]))
    output[valid_mask] = tensor
    return output


def _npu_attention_lengths(
    prefix_grouper: PrefixGrouper,
    *,
    query_length: int,
    key_length: int,
) -> tuple[list[int], list[int], int]:
    """Return per-sequence Q/KV lengths and the matching NPU causal sparse mode."""
    group_info = list(prefix_grouper.group_info)
    if query_length == key_length:
        prefix_lengths = [int(info.prefix_len) for info in group_info]
        return prefix_lengths, prefix_lengths, 2  # left-up causal

    query_lengths = [int(suffix_length) for info in group_info for suffix_length in info.suffix_lens]
    key_lengths = [int(info.prefix_len + suffix_length) for info in group_info for suffix_length in info.suffix_lens]
    return query_lengths, key_lengths, 3  # right-down causal for suffix Q against prefix + suffix KV


def _npu_tnd_attention(
    prefix_grouper: PrefixGrouper,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    dropout: float,
    scaling: float | None,
) -> torch.Tensor:
    """Run PrefixGrouper attention with torch-npu's packed fused training operator."""
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"Attention dropout must be in [0, 1), found {dropout}.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Native NPU PrefixGrouper attention requires BNSD query, key, and value tensors.")
    if key.shape != value.shape:
        raise ValueError(f"Key and value shapes must match, found {tuple(key.shape)} and {tuple(value.shape)}.")

    query_length, key_length = query.shape[2], key.shape[2]
    if padding_mask.shape != (query.shape[0], key_length):
        raise ValueError(
            f"NPU attention mask must have shape {(query.shape[0], key_length)}, found {tuple(padding_mask.shape)}."
        )
    query_lengths, key_lengths, sparse_mode = _npu_attention_lengths(
        prefix_grouper,
        query_length=query_length,
        key_length=key_length,
    )
    if len(query_lengths) != query.shape[0] or len(key_lengths) != key.shape[0]:
        raise ValueError(
            "PrefixGrouper sequence metadata does not match the NPU attention batch: "
            f"Q={len(query_lengths)}/{query.shape[0]}, KV={len(key_lengths)}/{key.shape[0]}."
        )
    if max(query_lengths) > query_length or max(key_lengths) > key_length:
        raise ValueError(
            f"PrefixGrouper sequence lengths exceed padded shapes: Q={query_lengths}/{query_length}, "
            f"KV={key_lengths}/{key_length}."
        )

    key_mask = padding_mask.bool()
    query_mask = key_mask[:, -query_length:]
    packed_query = _pack_bnsd(query, query_mask)
    packed_key = _pack_bnsd(key, key_mask)
    packed_value = _pack_bnsd(value, key_mask)
    scale = float(scaling) if scaling is not None else query.shape[-1] ** -0.5
    torch_npu = _torch_npu_module()
    packed_output = torch_npu.npu_fusion_attention(
        query=packed_query,
        key=packed_key,
        value=packed_value,
        head_num=query.shape[1],
        input_layout="TND",
        atten_mask=_npu_compressed_causal_mask(query.device),
        scale=scale,
        keep_prob=1.0 - dropout,
        actual_seq_qlen=list(accumulate(query_lengths)),
        actual_seq_kvlen=list(accumulate(key_lengths)),
        sparse_mode=sparse_mode,
        softmax_layout="TND",
    )[0]
    return _unpack_tnd(packed_output, query_mask)


def _repeat_kv(hidden_states: torch.Tensor, query_heads: int) -> torch.Tensor:
    key_value_heads = hidden_states.shape[1]
    if key_value_heads == query_heads:
        return hidden_states
    if query_heads % key_value_heads:
        raise ValueError(f"Query heads ({query_heads}) must be divisible by KV heads ({key_value_heads}).")
    return hidden_states.repeat_interleave(query_heads // key_value_heads, dim=1)


def _causal_sdpa_mask(padding_mask: torch.Tensor, query_length: int, key_length: int) -> torch.Tensor:
    """Expand PrefixGrouper's 2-D padding mask into SDPA's causal mask."""
    suffix_offset = key_length - query_length
    if suffix_offset < 0:
        raise ValueError(f"Key length ({key_length}) cannot be smaller than query length ({query_length}).")
    padding_mask = padding_mask.bool()
    query_valid = padding_mask[:, -query_length:]
    key_valid = padding_mask[:, :key_length]
    query_positions = torch.arange(query_length, device=padding_mask.device).unsqueeze(-1)
    key_positions = torch.arange(key_length, device=padding_mask.device).unsqueeze(0)
    causal = key_positions <= query_positions + suffix_offset
    return query_valid[:, None, :, None] & key_valid[:, None, None, :] & causal[None, None, :, :]


def _sdpa_prefix_grouper_wrapper(original_fn):
    def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
        prefix_grouper = kwargs.pop("prefix_grouper", None)
        if prefix_grouper is None:
            return original_fn(module, query, key, value, attention_mask, *args, **kwargs)

        dropout = kwargs.pop("dropout", 0.0)
        scaling = kwargs.pop("scaling", None)

        def attention_forward(inner_query, inner_key, inner_value, padding_mask, *_args, **_kwargs):
            if inner_query.device.type == "npu":
                return _npu_tnd_attention(
                    prefix_grouper,
                    inner_query,
                    inner_key,
                    inner_value,
                    padding_mask,
                    dropout=dropout,
                    scaling=scaling,
                )
            inner_key = _repeat_kv(inner_key, inner_query.shape[1])
            inner_value = _repeat_kv(inner_value, inner_query.shape[1])
            causal_mask = _causal_sdpa_mask(padding_mask, inner_query.shape[2], inner_key.shape[2])
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

    return wrapped


def apply_prefix_grouper_patch() -> None:
    """Apply VERL's patch plus a causal-mask fix for its SDPA adapter."""
    global _PATCHED
    if _PATCHED:
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    _apply_verl_prefix_grouper_patch()
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _sdpa_prefix_grouper_wrapper(original_sdpa)
    _PATCHED = True


def _check_verl_version() -> None:
    import verl

    current = Version(verl.__version__)
    if not Version("0.9.0") <= current < Version("0.10.0"):
        raise RuntimeError(f"This PrefixGrouper integration requires VERL 0.9.x, found {verl.__version__}.")


def _prompt_key(prompt: torch.Tensor, pad_token_id: int) -> tuple[int, ...]:
    return tuple(prompt[prompt.ne(pad_token_id)].detach().cpu().tolist())


def group_prompt_indices(prompts: torch.Tensor, pad_token_id: int) -> list[list[int]]:
    """Return stable groups of rows with exactly equal non-padding prompts."""
    groups: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()
    for row, prompt in enumerate(prompts):
        groups.setdefault(_prompt_key(prompt, pad_token_id), []).append(row)
    return list(groups.values())


def _group_order(groups: Iterable[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.tensor([row for group in groups for row in group], dtype=torch.long, device=device)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=device)
    return order, inverse


def _as_scalar(value: Any) -> float:
    value = tu.unwrap_non_tensor_data(value)
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("PrefixGrouper currently requires a scalar temperature.")
        return float(value.item())
    return float(value)


def _response_values_to_nested(
    values: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
) -> torch.Tensor:
    """Place response predictions in VERL's full-sequence jagged layout."""
    if not input_ids.is_nested:
        raise ValueError("VERL 0.9 PrefixGrouper expects no-padding nested input_ids.")

    sequence_lengths = input_ids.offsets().diff().tolist()
    rows = []
    for row, (sequence_length, prompt_length, response_length) in enumerate(
        zip(sequence_lengths, prompt_lengths.tolist(), response_lengths.tolist(), strict=True)
    ):
        if prompt_length <= 0:
            raise ValueError("PrefixGrouper requires every sample to contain at least one prompt token.")
        if sequence_length != prompt_length + response_length:
            raise ValueError(
                "Nested input length does not match prompt + response length: "
                f"{sequence_length} != {prompt_length} + {response_length}."
            )
        full_row = values.new_zeros(sequence_length)
        full_row[prompt_length - 1 : prompt_length + response_length - 1] = values[row, :response_length]
        rows.append(full_row)
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def forward_with_prefix_grouper(
    micro_batch: TensorDict | dict[str, Any],
    model: torch.nn.Module,
    *,
    temperature: float,
    calculate_entropy: bool,
    entropy_fn: Any,
) -> dict[str, torch.Tensor] | None:
    """Run a shared-prefix forward and return VERL-compatible model outputs.

    ``None`` means that the batch has no repeated prompt and should use VERL's
    standard forward path.
    """
    if "multi_modal_inputs" in micro_batch:
        return None

    prompts = micro_batch["prompts"]
    responses = micro_batch["responses"]
    response_mask = micro_batch["response_mask"].bool()
    input_ids = micro_batch["input_ids"]

    if prompts.is_nested or responses.is_nested:
        raise ValueError("PrefixGrouper requires the padded prompts and responses retained by VERL 0.9.")

    if "attention_mask" in micro_batch and not micro_batch["attention_mask"].is_nested:
        prompt_mask = micro_batch["attention_mask"][:, : prompts.shape[1]].bool()
    else:
        pad_token_id = int(tu.get_non_tensor_data(micro_batch, "pad_token_id", 0))
        prompt_mask = prompts.ne(pad_token_id)
    grouped: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()
    for row, prompt in enumerate(prompts):
        key = tuple(prompt[prompt_mask[row]].detach().cpu().tolist())
        grouped.setdefault(key, []).append(row)
    groups = list(grouped.values())
    if all(len(group) == 1 for group in groups):
        return None

    order, inverse_order = _group_order(groups, prompts.device)
    representatives = torch.tensor([group[0] for group in groups], dtype=torch.long, device=prompts.device)
    prefix_ids = prompts.index_select(0, representatives)
    prefix_mask = prompt_mask.index_select(0, representatives)
    ordered_responses = responses.index_select(0, order)
    ordered_response_mask = response_mask.index_select(0, order)

    prefix_grouper = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=prefix_mask,
        suffix_mask=ordered_response_mask,
        group_sizes=[len(group) for group in groups],
        padding_mode="right",
        device=prompts.device,
    )
    concat_input_ids = prefix_grouper.concat_input(
        prefix_ids,
        prefix_mask,
        ordered_responses,
        ordered_response_mask,
    )
    position_ids = build_position_ids_for_prefix_grouper(prefix_grouper)

    log_probs, entropy, suffix_mask = pg_forward(
        model=model,
        prefix_grouper=prefix_grouper,
        concat_input_ids=concat_input_ids,
        attention_mask=prefix_grouper.padding_mask,
        position_ids=position_ids,
        completion_ids=ordered_responses,
        completion_mask=ordered_response_mask,
        temperature=temperature,
        padding_mode="right",
        include_prefix_last=1,
        calculate_entropy=calculate_entropy,
        entropy_fn=entropy_fn,
    )

    log_probs = log_probs.masked_fill(~suffix_mask.bool(), 0).index_select(0, inverse_order)
    if entropy is not None:
        entropy = entropy.masked_fill(~suffix_mask.bool(), 0).index_select(0, inverse_order)

    prompt_lengths = prompt_mask.sum(dim=-1)
    response_lengths = response_mask.sum(dim=-1)
    output = {
        "log_probs": _response_values_to_nested(
            log_probs,
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
        )
    }
    if entropy is not None:
        output["entropy"] = _response_values_to_nested(
            entropy,
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
        )
    return output


def _prefix_grouper_forward_step(
    engine: FSDPEngineWithLMHead,
    micro_batch: TensorDict,
    loss_function: Any,
    forward_only: bool,
):
    """FSDP engine hook matching VERL 0.9's ``forward_step`` contract."""
    unsupported = (
        tu.get_non_tensor_data(micro_batch, "use_remove_padding", False)
        or tu.get_non_tensor_data(micro_batch, "use_fused_kernels", False)
        or tu.get_non_tensor_data(micro_batch, "distillation_use_topk", False)
        or tu.get_non_tensor_data(micro_batch, "calculate_sum_pi_squared", False)
    )
    if unsupported or "multi_modal_inputs" in micro_batch:
        return FSDPEngineWithLMHead.forward_step(engine, micro_batch, loss_function, forward_only)

    micro_batch = micro_batch.to(get_device_id())
    temperature = _as_scalar(micro_batch["temperature"])
    calculate_entropy = bool(tu.get_non_tensor_data(micro_batch, "calculate_entropy", False))
    autocast_dtype = getattr(engine, "_autocast_dtype", torch.bfloat16)
    autocast_ctx = (
        nullcontext()
        if autocast_dtype == torch.float32
        else torch.autocast(device_type=get_device_name(), dtype=autocast_dtype)
    )

    with autocast_ctx:
        model_output = forward_with_prefix_grouper(
            micro_batch,
            engine.module,
            temperature=temperature,
            calculate_entropy=calculate_entropy,
            entropy_fn=engine.compute_entropy_from_logits,
        )
        if model_output is None:
            return FSDPEngineWithLMHead.forward_step(engine, micro_batch, loss_function, forward_only)

        if loss_function is not None:
            loss, metrics = loss_function(
                model_output=model_output,
                data=micro_batch,
                dp_group=engine.get_data_parallel_group(),
            )
        else:
            if not forward_only:
                raise AssertionError("forward_only must be true when loss_function is None")
            loss = torch.tensor(1.0, device=get_device_name())
            metrics = {}

        detached_output = {
            key: value.detach() if torch.is_tensor(value) and value.grad_fn is not None else value
            for key, value in model_output.items()
        }
        metadata = {
            "model_output": detached_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }
        return loss, metadata


class PrefixGrouperTrainingWorker(TrainingWorker):
    """VERL 0.9 training worker with PrefixGrouper wired into its FSDP engine."""

    def __init__(self, config: Any):
        _check_verl_version()
        if config.model_type != "language_model":
            raise ValueError("PrefixGrouper is only supported for VERL language-model workers.")
        if config.model_config.get("use_remove_padding", False):
            raise ValueError("PrefixGrouper requires actor_rollout_ref.model.use_remove_padding=false.")
        if config.model_config.get("use_fused_kernels", False):
            raise ValueError("PrefixGrouper does not support VERL fused kernels.")
        if config.engine_config.ulysses_sequence_parallel_size != 1:
            raise ValueError("PrefixGrouper does not support Ulysses sequence parallelism.")
        if config.engine_config.strategy not in {"fsdp", "fsdp2"}:
            raise ValueError("PrefixGrouper requires VERL's FSDP or FSDP2 engine.")

        apply_prefix_grouper_patch()
        super().__init__(config=config)
        if not isinstance(self.engine, FSDPEngineWithLMHead):
            raise TypeError(f"Expected FSDPEngineWithLMHead, found {type(self.engine).__name__}.")
        self.engine.forward_step = MethodType(_prefix_grouper_forward_step, self.engine)


class PrefixGrouperActorRolloutRefWorker(ActorRolloutRefWorker):
    """VERL 0.9 actor/rollout/reference worker using PrefixGrouper for actor and ref forwards."""

    actor_worker_cls = PrefixGrouperTrainingWorker
    ref_worker_cls = PrefixGrouperTrainingWorker


def reorder_by_prompt(batch: Any) -> None:
    """Place equal prompts contiguously before VERL partitions the batch."""
    responses = batch.batch["responses"]
    response_length = responses.shape[-1]
    prompts = batch.batch.get("prompts", batch.batch["input_ids"][:, :-response_length])
    attention_mask = batch.batch["attention_mask"][:, :-response_length]
    grouped: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()
    for row, prompt in enumerate(prompts):
        key = tuple(prompt[attention_mask[row].bool()].detach().cpu().tolist())
        grouped.setdefault(key, []).append(row)
    groups = list(grouped.values())
    groups.sort(key=len, reverse=True)
    order = torch.tensor([row for group in groups for row in group], dtype=torch.int64)
    batch.reorder(order)
