# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""Correctness tests for the VERL 0.9 PrefixGrouper engine path."""

from __future__ import annotations

import sys
from copy import deepcopy
from types import ModuleType
from typing import Iterator

import pytest
import torch
from prefix_grouper import PrefixGrouper
from tensordict import TensorDict
from transformers import Qwen2Config, Qwen2ForCausalLM
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.padding import no_padding_2_padding

from agentlightning.verl import prefix_grouper as prefix_grouper_module
from agentlightning.verl.prefix_grouper import (
    PrefixGrouperActorRolloutRefWorker,
    PrefixGrouperTrainingWorker,
    apply_prefix_grouper_patch,
    forward_with_prefix_grouper,
    group_prompt_indices,
)


@pytest.fixture(autouse=True)
def _clear_native_npu_test_state() -> Iterator[None]:
    prefix_grouper_module._torch_npu_module.cache_clear()
    prefix_grouper_module._NPU_CAUSAL_MASKS.clear()
    yield
    prefix_grouper_module._torch_npu_module.cache_clear()
    prefix_grouper_module._NPU_CAUSAL_MASKS.clear()


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


def test_native_npu_attention_uses_packed_tnd_gqa_and_causal_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    fake_torch_npu = ModuleType("torch_npu")
    fake_torch_npu.__version__ = "2.10.0"

    def fake_npu_fusion_attention(**kwargs):
        calls.append(kwargs)
        return (kwargs["query"], None, None, None, 0, 0, 0)

    fake_torch_npu.npu_fusion_attention = fake_npu_fusion_attention
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    prefix_mask = torch.tensor([[1, 1, 1], [0, 1, 1]], dtype=torch.bool)
    suffix_mask = torch.tensor([[1, 1], [1, 0], [1, 1], [1, 0]], dtype=torch.bool)
    grouper = PrefixGrouper.from_ungrouped_masks(prefix_mask, suffix_mask, group_sizes=[2, 2])

    prefix_query = torch.randn(2, 4, 3, 8)
    prefix_key = torch.randn(2, 2, 3, 8)
    prefix_value = torch.randn(2, 2, 3, 8)
    prefix_output = prefix_grouper_module._npu_tnd_attention(
        grouper,
        prefix_query,
        prefix_key,
        prefix_value,
        grouper.prefix_attn_mask,
        dropout=0.0,
        scaling=None,
    )

    suffix_query = torch.randn(4, 4, 2, 8)
    suffix_key = torch.randn(4, 2, 5, 8)
    suffix_value = torch.randn(4, 2, 5, 8)
    suffix_output = prefix_grouper_module._npu_tnd_attention(
        grouper,
        suffix_query,
        suffix_key,
        suffix_value,
        grouper.suffix_attn_mask,
        dropout=0.1,
        scaling=0.125,
    )

    assert prefix_output.shape == (2, 3, 4, 8)
    assert suffix_output.shape == (4, 2, 4, 8)
    assert len(calls) == 2

    prefix_call, suffix_call = calls
    assert prefix_call["input_layout"] == suffix_call["input_layout"] == "TND"
    assert prefix_call["query"].shape == (5, 4, 8)
    assert prefix_call["key"].shape == (5, 2, 8)
    assert prefix_call["actual_seq_qlen"] == prefix_call["actual_seq_kvlen"] == [3, 5]
    assert prefix_call["sparse_mode"] == 2
    assert prefix_call["keep_prob"] == 1.0

    assert suffix_call["query"].shape == (6, 4, 8)
    assert suffix_call["key"].shape == (16, 2, 8)
    assert suffix_call["actual_seq_qlen"] == [2, 3, 5, 6]
    assert suffix_call["actual_seq_kvlen"] == [5, 9, 13, 16]
    assert suffix_call["sparse_mode"] == 3
    assert suffix_call["keep_prob"] == pytest.approx(0.9)
    assert suffix_call["scale"] == 0.125

    causal_mask = prefix_call["atten_mask"]
    assert causal_mask is suffix_call["atten_mask"]
    assert causal_mask.shape == (2048, 2048)
    assert causal_mask.dtype == torch.bool
    assert not causal_mask[0, 0]
    assert causal_mask[0, 1]
    assert not causal_mask[1, 0]


def test_native_npu_attention_rejects_unpinned_torch_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch_npu = ModuleType("torch_npu")
    fake_torch_npu.__version__ = "2.9.0"
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    with pytest.raises(RuntimeError, match="requires torch-npu 2.10.0"):
        prefix_grouper_module._torch_npu_module()


def test_npu_tnd_pack_and_unpack_preserve_gradients() -> None:
    tensor = torch.randn(2, 3, 4, 5, requires_grad=True)
    valid_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)

    packed = prefix_grouper_module._pack_bnsd(tensor, valid_mask)
    unpacked = prefix_grouper_module._unpack_tnd(packed, valid_mask)
    unpacked.sum().backward()

    expected_grad = valid_mask[:, None, :, None].expand_as(tensor)
    assert tensor.grad is not None
    torch.testing.assert_close(tensor.grad, expected_grad.to(tensor.dtype))
