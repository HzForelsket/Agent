from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

torch_npu = pytest.importorskip("torch_npu")

from prefix_grouper_npu import build_shared_prefix_plan, shared_prefix_attention
from reference import materialized_reference


pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires a real Ascend 910B")

PREFIX_LENS, SUFFIX_LENS, GROUP_SIZES = (1,), (1, 63), (2,)
SEED = 1234


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    error = (actual - expected).abs()
    worst = tuple(index.item() for index in torch.unravel_index(error.argmax(), error.shape))
    return {
        "cosine": F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
        "max_abs": error[worst].item(),
        "worst_index": worst,
        "actual_at_worst": actual[worst].item(),
        "expected_at_worst": expected[worst].item(),
    }


def _make_reference_case():
    torch.manual_seed(SEED)
    total_tokens = sum(PREFIX_LENS) + sum(SUFFIX_LENS)
    # Match the original failing case's RNG order, including BF16 grad generation.
    q_seed = torch.randn(total_tokens, 2, 128, dtype=torch.float32).to(torch.bfloat16)
    k_seed = torch.randn(total_tokens, 2, 128, dtype=torch.float32).to(torch.bfloat16)
    v_seed = torch.randn(total_tokens, 2, 128, dtype=torch.float32).to(torch.bfloat16)
    grad_seed = torch.randn_like(q_seed)
    q_ref = q_seed.float().requires_grad_(True)
    k_ref = k_seed.float().requires_grad_(True)
    v_ref = v_seed.float().requires_grad_(True)
    out_ref = materialized_reference(q_ref, k_ref, v_ref, PREFIX_LENS, SUFFIX_LENS, GROUP_SIZES)
    out_ref.backward(grad_seed.float())
    expected = {"out": out_ref.detach(), "dq": q_ref.grad, "dk": k_ref.grad, "dv": v_ref.grad}

    # Compact rows: prefix 0, first suffix 1, second suffix 2:65. No cross-suffix attention.
    allowed = torch.zeros((total_tokens, total_tokens), dtype=torch.bool)
    allowed[:, 0] = True
    allowed[1, 1] = True
    allowed[2:, 2:] = torch.ones((SUFFIX_LENS[1], SUFFIX_LENS[1]), dtype=torch.bool).tril()
    scale = torch.tensor(128.0, dtype=torch.float32).rsqrt()
    scores = torch.einsum("thd,shd->hts", q_seed.float(), k_seed.float()) * scale
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    expected_lse = scores.logsumexp(dim=-1).transpose(0, 1).contiguous()
    dense_out = torch.einsum("hts,shd->thd", scores.softmax(dim=-1), v_seed.float())
    torch.testing.assert_close(dense_out, expected["out"], rtol=1e-5, atol=1e-6)
    return (q_seed, k_seed, v_seed, grad_seed), expected, expected_lse


def test_original_minimal_random_forward_backward() -> None:
    (q_seed, k_seed, v_seed, grad_seed), expected, expected_lse = _make_reference_case()
    q = q_seed.npu().requires_grad_(True)
    k = k_seed.npu().requires_grad_(True)
    v = v_seed.npu().requires_grad_(True)
    plan = build_shared_prefix_plan(PREFIX_LENS, SUFFIX_LENS, GROUP_SIZES, device=q.device)
    saved_lse: list[torch.Tensor] = []

    def pack_saved(tensor: torch.Tensor) -> torch.Tensor:
        saved = tensor.detach()
        if saved.dtype == torch.float32 and saved.shape == q_seed.shape[:2]:
            saved_lse.append(saved)
        return saved

    # Observe the actual autograd-saved LSE without a second forward or replacement values.
    with torch.autograd.graph.saved_tensors_hooks(pack_saved, lambda tensor: tensor):
        out = shared_prefix_attention(q, k, v, plan)
    out.backward(grad_seed.npu())
    torch.npu.synchronize()

    assert len(saved_lse) == 1
    actual_lse = saved_lse[0].float().cpu()
    tensors = {"out": out, "dq": q.grad, "dk": k.grad, "dv": v.grad}
    actual = {name: tensor.detach().float().cpu() for name, tensor in tensors.items()}
    metrics = {
        "case": {"name": "original_minimal_random", "prefix_lens": PREFIX_LENS,
                 "suffix_lens": SUFFIX_LENS, "group_sizes": GROUP_SIZES,
                 "hq": 2, "hkv": 2, "head_dim": 128, "total_tokens": q_seed.shape[0], "seed": SEED},
        "lse": {"actual": actual_lse.tolist(),
                "expected": expected_lse.tolist(),
                "max_abs": (actual_lse - expected_lse).abs().max().item()},
        **{name: _metric(actual[name], expected[name]) for name in expected},
    }
    print("PREFIX_GROUPER_NPU_RESULT=" + json.dumps(metrics, sort_keys=True), flush=True)

    for name, tensor in tensors.items():
        assert tensor.dtype == torch.bfloat16, name
    assert metrics["out"]["cosine"] >= 0.999, metrics["out"]
    assert metrics["out"]["max_abs"] <= 0.05, metrics["out"]
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-6)
    for name in ("dq", "dk", "dv"):
        assert metrics[name]["cosine"] >= 0.999, (name, metrics[name])
        assert metrics[name]["max_abs"] <= 0.1, (name, metrics[name])
