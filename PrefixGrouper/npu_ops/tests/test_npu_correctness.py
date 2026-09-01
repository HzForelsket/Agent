from __future__ import annotations

import json
import math
import os

import pytest
import torch
import torch.nn.functional as F

torch_npu = pytest.importorskip("torch_npu")

from prefix_grouper_npu import build_shared_prefix_plan, shared_prefix_attention
from reference import materialized_reference


pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires a real Ascend 910B")


CASES = [
    ((1,), (1, 63), (2,), 2, 2),
    ((127,), (64, 65, 1, 63), (4,), 6, 2),
    ((128,), (65,) * 8, (8,), 3, 1),
    ((129,), (1, 64), (2,), 4, 4),
    ((1024,), (63, 65, 64, 1), (4,), 3, 1),
    ((1536,), (1, 63), (2,), 2, 1),
    ((127, 129), (1, 63, 64, 65), (2, 2), 6, 2),
    ((1, 128), (65, 1, 63, 64, 1, 65), (2, 4), 3, 1),
]


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.detach().float().cpu().flatten()
    expected = expected.detach().float().cpu().flatten()
    return {
        "cosine": float(F.cosine_similarity(actual, expected, dim=0)),
        "max_abs": float((actual - expected).abs().max()),
    }


@pytest.mark.parametrize("prefix_lens,suffix_lens,group_sizes,hq,hkv", CASES)
def test_forward_backward_against_materialized_fp32_reference(
    prefix_lens, suffix_lens, group_sizes, hq, hkv, capsys
) -> None:
    torch.manual_seed(1234)
    total_tokens = sum(prefix_lens) + sum(suffix_lens)
    q_seed = torch.randn(total_tokens, hq, 128).to(torch.bfloat16)
    k_seed = torch.randn(total_tokens, hkv, 128).to(torch.bfloat16)
    v_seed = torch.randn(total_tokens, hkv, 128).to(torch.bfloat16)
    grad_seed = torch.randn_like(q_seed)

    q_ref = q_seed.float().requires_grad_(True)
    k_ref = k_seed.float().requires_grad_(True)
    v_ref = v_seed.float().requires_grad_(True)
    out_ref = materialized_reference(
        q_ref, k_ref, v_ref, prefix_lens, suffix_lens, group_sizes
    )
    out_ref.backward(grad_seed.float())

    q = q_seed.npu().requires_grad_(True)
    k = k_seed.npu().requires_grad_(True)
    v = v_seed.npu().requires_grad_(True)
    plan = build_shared_prefix_plan(prefix_lens, suffix_lens, group_sizes, device=q.device)
    out = shared_prefix_attention(q, k, v, plan)
    out.backward(grad_seed.npu())
    torch.npu.synchronize()

    metrics = {
        "case": {
            "prefix_lens": prefix_lens,
            "suffix_lens": suffix_lens,
            "group_sizes": group_sizes,
            "hq": hq,
            "hkv": hkv,
        },
        "out": _metric(out, out_ref),
        "dq": _metric(q.grad, q_ref.grad),
        "dk": _metric(k.grad, k_ref.grad),
        "dv": _metric(v.grad, v_ref.grad),
    }
    print("PREFIX_GROUPER_NPU_RESULT=" + json.dumps(metrics, sort_keys=True))
    assert metrics["out"]["cosine"] >= 0.999
    assert metrics["out"]["max_abs"] <= 0.05
    for name in ("dq", "dk", "dv"):
        assert metrics[name]["cosine"] >= 0.999
        assert metrics[name]["max_abs"] <= 0.1


def test_invalid_tensor_contracts() -> None:
    plan = build_shared_prefix_plan([1], [1, 1], [2], device="npu")
    q = torch.empty((3, 2, 128), device="npu", dtype=torch.bfloat16)
    k = torch.empty((3, 1, 128), device="npu", dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="bfloat16"):
        shared_prefix_attention(q.float(), k, k, plan)
    with pytest.raises(ValueError, match="head_dim=128"):
        shared_prefix_attention(q[:, :, :64].contiguous(), k[:, :, :64].contiguous(), k[:, :, :64].contiguous(), plan)
    noncontiguous_q = q.transpose(0, 1).contiguous().transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous"):
        shared_prefix_attention(noncontiguous_q, k, k, plan)
    bad_heads = torch.empty((3, 3, 128), device="npu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible"):
        shared_prefix_attention(bad_heads, torch.empty((3, 2, 128), device="npu", dtype=torch.bfloat16),
                                torch.empty((3, 2, 128), device="npu", dtype=torch.bfloat16), plan)
