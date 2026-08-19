# Copyright (c) Microsoft. All rights reserved.

# type: ignore

from copy import deepcopy

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from verl.workers.actor import DataParallelPPOActor

from agentlightning.verl.prefix_grouper import (
    PREFIX_GROUPER_ATTENTION,
    PrefixGrouperDataParallelPPOActor,
    register_prefix_grouper_attention,
)


def _tiny_qwen() -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=32,
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


def _batch() -> dict[str, torch.Tensor]:
    prompts = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 0, 4, 5],
            [0, 0, 4, 5],
        ]
    )
    prompt_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
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
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
        ]
    )
    attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
    return {
        "input_ids": torch.cat([prompts, responses], dim=-1),
        "attention_mask": attention_mask,
        "position_ids": torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0),
        "responses": responses,
    }


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    return -(probabilities * torch.log_softmax(logits, dim=-1)).sum(dim=-1)


def _baseline_outputs(model: Qwen2ForCausalLM, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    response_length = batch["responses"].shape[-1]
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
    )
    logits = output.logits[:, -response_length - 1 : -1]
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, batch["responses"].unsqueeze(-1)).squeeze(-1)
    return log_probs, _entropy(logits)


class _FakeData:
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch

    def reorder(self, order: torch.Tensor) -> None:
        self.batch = {key: value[order.long()] for key, value in self.batch.items()}


def test_actor_update_restores_prompt_group_locality(monkeypatch) -> None:
    permutation = torch.tensor([0, 2, 1, 3])
    data = _FakeData({key: value[permutation] for key, value in _batch().items()})

    def fake_update_policy(_actor, update_data: _FakeData) -> list[int]:
        return update_data.batch["responses"][:, 0].tolist()

    monkeypatch.setattr(DataParallelPPOActor, "update_policy", fake_update_policy)
    actor = object.__new__(PrefixGrouperDataParallelPPOActor)

    assert actor.update_policy(data) == [6, 8, 11, 12]


def test_prefix_grouper_matches_qwen2_forward_and_backward() -> None:
    torch.manual_seed(0)
    baseline_model = _tiny_qwen()
    grouped_model = deepcopy(baseline_model)
    grouped_model.config._attn_implementation = PREFIX_GROUPER_ATTENTION
    register_prefix_grouper_attention()
    batch = _batch()

    baseline_log_probs, baseline_entropy = _baseline_outputs(baseline_model, batch)
    baseline_loss = -(baseline_log_probs * batch["attention_mask"][:, -3:]).sum()
    baseline_loss.backward()

    actor = object.__new__(PrefixGrouperDataParallelPPOActor)
    actor.actor_module = grouped_model
    actor.device_name = "cpu"
    actor.param_dtype = torch.float32
    actor.compute_entropy_from_logits = _entropy
    grouped_entropy, grouped_log_probs = actor._forward_micro_batch(batch, temperature=1.0, calculate_entropy=True)
    grouped_loss = -(grouped_log_probs * batch["attention_mask"][:, -3:]).sum()
    grouped_loss.backward()

    valid_response_tokens = batch["attention_mask"][:, -3:].bool()
    assert grouped_entropy is not None
    torch.testing.assert_close(
        grouped_log_probs[valid_response_tokens],
        baseline_log_probs[valid_response_tokens],
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        grouped_entropy[valid_response_tokens],
        baseline_entropy[valid_response_tokens],
        atol=2e-5,
        rtol=2e-5,
    )
    for (baseline_name, baseline_parameter), (grouped_name, grouped_parameter) in zip(
        baseline_model.named_parameters(), grouped_model.named_parameters()
    ):
        assert baseline_name == grouped_name
        torch.testing.assert_close(grouped_parameter.grad, baseline_parameter.grad, atol=3e-5, rtol=3e-5)
