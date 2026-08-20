# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""Correctness tests for the VERL 0.9 PrefixGrouper engine path."""

from __future__ import annotations

from copy import deepcopy

import torch
from tensordict import TensorDict
from transformers import Qwen2Config, Qwen2ForCausalLM
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.padding import no_padding_2_padding

from agentlightning.verl.prefix_grouper import (
    PrefixGrouperActorRolloutRefWorker,
    PrefixGrouperTrainingWorker,
    apply_prefix_grouper_patch,
    forward_with_prefix_grouper,
    group_prompt_indices,
)


def _tiny_qwen() -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(config)


def _batch() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    prompts = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 0, 4, 5],
            [0, 0, 4, 5],
        ]
    )
    responses = torch.tensor(
        [
            [6, 7, 0],
            [8, 9, 10],
            [11, 0, 0],
            [12, 13, 0],
        ]
    )
    prompt_mask = prompts.ne(0)
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=torch.bool,
    )
    attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
    padded_input_ids = torch.cat((prompts, responses), dim=-1)
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
    nested_rows = [padded_input_ids[row][attention_mask[row]] for row in range(prompts.shape[0])]
    nested_input_ids = torch.nested.as_nested_tensor(nested_rows, layout=torch.jagged)
    micro_batch = {
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
        "input_ids": nested_input_ids,
        "attention_mask": attention_mask,
    }
    return micro_batch, padded_input_ids, position_ids


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    return -(probabilities * torch.log_softmax(logits, dim=-1)).sum(dim=-1)


def _baseline(model: Qwen2ForCausalLM, padded_input_ids: torch.Tensor, position_ids: torch.Tensor, batch: dict):
    output = model(
        input_ids=padded_input_ids,
        attention_mask=batch["attention_mask"],
        position_ids=position_ids,
        use_cache=False,
    )
    prompt_width = batch["prompts"].shape[1]
    response_width = batch["responses"].shape[1]
    logits = output.logits[:, prompt_width - 1 : prompt_width + response_width - 1]
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, batch["responses"].unsqueeze(-1)).squeeze(-1)
    return log_probs, _entropy(logits)


def _as_tensordict(batch: dict[str, torch.Tensor]) -> TensorDict:
    return TensorDict(batch, batch_size=[batch["prompts"].shape[0]])


def test_worker_uses_verl_090_model_engine() -> None:
    assert issubclass(PrefixGrouperActorRolloutRefWorker, ActorRolloutRefWorker)
    assert PrefixGrouperActorRolloutRefWorker.actor_worker_cls is PrefixGrouperTrainingWorker
    assert PrefixGrouperActorRolloutRefWorker.ref_worker_cls is PrefixGrouperTrainingWorker


def test_group_prompt_indices_ignores_padding() -> None:
    prompts = torch.tensor([[0, 1, 2], [1, 2, 0], [0, 3, 4]])
    assert group_prompt_indices(prompts, pad_token_id=0) == [[0, 1], [2]]


def test_prefix_grouper_matches_qwen2_forward_and_backward() -> None:
    torch.manual_seed(0)
    apply_prefix_grouper_patch()
    baseline_model = _tiny_qwen()
    grouped_model = deepcopy(baseline_model)
    batch, padded_input_ids, position_ids = _batch()

    baseline_log_probs, baseline_entropy = _baseline(baseline_model, padded_input_ids, position_ids, batch)
    baseline_loss = -(baseline_log_probs * batch["response_mask"]).sum()
    baseline_loss.backward()

    grouped_output = forward_with_prefix_grouper(
        batch,
        grouped_model,
        temperature=1.0,
        calculate_entropy=True,
        entropy_fn=_entropy,
    )
    assert grouped_output is not None
    td = _as_tensordict(batch)
    grouped_log_probs = no_padding_2_padding(grouped_output["log_probs"], td)
    grouped_entropy = no_padding_2_padding(grouped_output["entropy"], td)
    grouped_loss = -(grouped_log_probs * batch["response_mask"]).sum()
    grouped_loss.backward()

    valid = batch["response_mask"]
    torch.testing.assert_close(grouped_log_probs[valid], baseline_log_probs[valid], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(grouped_entropy[valid], baseline_entropy[valid], atol=2e-5, rtol=2e-5)
    for (baseline_name, baseline_parameter), (grouped_name, grouped_parameter) in zip(
        baseline_model.named_parameters(), grouped_model.named_parameters(), strict=True
    ):
        assert baseline_name == grouped_name
        torch.testing.assert_close(grouped_parameter.grad, baseline_parameter.grad, atol=3e-5, rtol=3e-5)
