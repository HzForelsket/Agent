# Copyright (c) Microsoft. All rights reserved.

"""Simple G-way call-prefix sharing; imported only by the sharing replay process."""

from __future__ import annotations

from types import MethodType

import torch
from verl.trainer.ppo.prefix_grouper_utils import build_position_ids_for_prefix_grouper
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.engine.fsdp import FSDPEngineWithLMHead

from .prefix_grouper import apply_prefix_grouper_patch, build_prefix_grouper


def shared_forward_step(engine, micro_batch, loss_function, forward_only):
    """Restore every original prediction position, including losses inside the shared prefix."""
    prefix = int(micro_batch["replay_prefix"][0].item())
    if prefix == 0:
        return FSDPEngineWithLMHead.forward_step(engine, micro_batch, loss_function, forward_only)
    device = next(engine.module.parameters()).device
    data = micro_batch.to(device)
    sequences = data["input_ids"].unbind()
    lengths = [len(row) for row in sequences]
    if len(sequences) != 4 or any(length < prefix for length in lengths):
        raise ValueError("A shared call slot must contain four complete sequences")
    packed = torch.cat([sequences[0][:prefix], *(row[prefix:] for row in sequences)])[None, :]
    if max(lengths) == prefix:
        # Four identical complete sequences have no suffix attention work.
        model_kwargs = {
            "attention_mask": torch.ones_like(packed),
            "position_ids": torch.arange(prefix, device=device)[None, :],
        }
    else:
        prefix_mask = torch.ones((1, prefix), dtype=torch.bool)
        suffix_mask = torch.arange(max(lengths) - prefix)[None, :] < torch.tensor(lengths)[:, None] - prefix
        grouper = build_prefix_grouper(
            prefix_mask=prefix_mask,
            suffix_mask=suffix_mask,
            group_sizes=[4],
            device=device,
        )
        model_kwargs = {
            "attention_mask": None,
            "position_ids": build_position_ids_for_prefix_grouper(grouper),
            "prefix_grouper": grouper,
        }
    # Repeated prefix indices deliberately accumulate all trajectory gradients through index_select.
    indices = []
    offset = prefix
    for length in lengths:
        indices.extend(range(prefix))
        indices.extend(range(offset, offset + length - prefix))
        offset += length - prefix
    with torch.autocast(device_type=device.type, dtype=engine._autocast_dtype):
        output = engine.module(input_ids=packed, use_cache=False, **model_kwargs)
        logits = output.logits[0].index_select(0, torch.tensor(indices, device=device))
        temperatures = torch.cat([data["temperature"][i].expand(length) for i, length in enumerate(lengths)])
        logits = logits / temperatures[:, None].to(logits.dtype)
        # Match VERL's packed next-token labels. The last position of each row has zero loss mask.
        labels = torch.roll(data["input_ids"].values(), shifts=-1, dims=0)
        values = logprobs_from_logits(logits=logits, labels=labels)
        model_output = {"log_probs": torch.nested.nested_tensor_from_jagged(values, data["input_ids"].offsets())}
        if loss_function is None:
            if not forward_only:
                raise ValueError("Training requires a loss function")
            loss, metrics = values.new_zeros(()), {}
        else:
            loss, metrics = loss_function(
                model_output=model_output, data=data, dp_group=engine.get_data_parallel_group()
            )
    return loss, {
        "model_output": {key: value.detach() for key, value in model_output.items()},
        "loss": loss.detach().item(),
        "metrics": metrics,
    }


def prepare_shared_model() -> None:
    """Enable the existing GPU/NPU PrefixGrouper attention integration before loading weights."""
    apply_prefix_grouper_patch()


def install_shared_forward(engine) -> None:
    """Change only this replay engine's forward method; leave the original training adapter unchanged."""
    engine.forward_step = MethodType(shared_forward_step, engine)
