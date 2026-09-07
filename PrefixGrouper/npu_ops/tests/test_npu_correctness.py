from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

torch_npu = pytest.importorskip("torch_npu")

from prefix_grouper_npu import build_shared_prefix_plan, shared_prefix_attention
from reference import materialized_reference


pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires a real Ascend 910B")


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    return {
        "cosine": F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
        "max_abs": (actual - expected).abs().max().item(),
        "actual_first_two": actual[:, 0, :2].tolist(),
        "expected_first_two": expected[:, 0, :2].tolist(),
        "actual_tail_max_abs": actual[:, :, 2:].abs().max().item(),
    }


def test_single_core_minimal_forward_backward() -> None:
    prefix_lens, suffix_lens, group_sizes = (1,), (1, 1), (2,)
    q_seed = torch.zeros((3, 1, 128), dtype=torch.bfloat16)
    k_seed = torch.zeros_like(q_seed)
    v_seed = torch.zeros_like(q_seed)
    grad_seed = torch.zeros_like(q_seed)
    q_seed[:, 0, 0] = torch.tensor([1, 1, 2], dtype=torch.bfloat16)
    k_seed[:, 0, :2] = torch.tensor([[1, 1], [1, -1], [1, 3]], dtype=torch.bfloat16)
    v_seed[:, 0, 0] = torch.tensor([1, 3, -1], dtype=torch.bfloat16)
    grad_seed[:, 0, 0] = 1

    # Allowed keys are {0}, {0, 1}, {0, 2}; each suffix has probabilities (1/2, 1/2).
    scale = torch.tensor(128.0, dtype=torch.float32).rsqrt()
    expected = {name: torch.zeros((3, 1, 128), dtype=torch.float32)
                for name in ("out", "dq", "dk", "dv")}
    expected["out"][:, 0, 0] = torch.tensor([1, 2, 0], dtype=torch.float32)
    expected["dq"][:, 0, 1] = scale * torch.tensor([0, -1, -1], dtype=torch.float32)
    expected["dk"][:, 0, 0] = scale * torch.tensor([0.5, 0.5, -1], dtype=torch.float32)
    expected["dv"][:, 0, 0] = torch.tensor([2, 0.5, 0.5], dtype=torch.float32)
    expected_lse = (scale * torch.tensor([1, 1, 2], dtype=torch.float32)
                    + torch.tensor([1, 2, 2], dtype=torch.float32).log()).reshape(3, 1)

    q_ref = q_seed.float().requires_grad_(True)
    k_ref = k_seed.float().requires_grad_(True)
    v_ref = v_seed.float().requires_grad_(True)
    out_ref = materialized_reference(q_ref, k_ref, v_ref, prefix_lens, suffix_lens, group_sizes)
    out_ref.backward(grad_seed.float())
    for name, value in {"out": out_ref, "dq": q_ref.grad, "dk": k_ref.grad, "dv": v_ref.grad}.items():
        torch.testing.assert_close(value, expected[name], rtol=1e-6, atol=1e-7)

    q = q_seed.npu().requires_grad_(True)
    k = k_seed.npu().requires_grad_(True)
    v = v_seed.npu().requires_grad_(True)
    plan = build_shared_prefix_plan(prefix_lens, suffix_lens, group_sizes, device=q.device)
    saved_lse: list[torch.Tensor] = []

    def pack_saved(tensor: torch.Tensor) -> torch.Tensor:
        saved = tensor.detach()
        if saved.dtype == torch.float32 and saved.shape == (3, 1):
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
        "case": {"name": "single_core_minimal", "prefix_lens": prefix_lens,
                 "suffix_lens": suffix_lens, "group_sizes": group_sizes,
                 "hq": 1, "hkv": 1, "head_dim": 128, "scale": scale.item()},
        "lse": {"actual": actual_lse.flatten().tolist(),
                "expected": expected_lse.flatten().tolist(),
                "max_abs": (actual_lse - expected_lse).abs().max().item()},
        **{name: _metric(actual[name], expected[name]) for name in expected},
    }
    print("PREFIX_GROUPER_NPU_RESULT=" + json.dumps(metrics, sort_keys=True), flush=True)

    for name, tensor in tensors.items():
        assert tensor.dtype == torch.bfloat16, name
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-6)
    for name in expected:
        assert metrics[name]["cosine"] >= 0.999, name
        # Account only for the final BF16 rounding, including every nominally zero element.
        rounded = expected[name].to(torch.bfloat16).float()
        torch.testing.assert_close(actual[name], rounded, rtol=0, atol=1e-5, msg=name)
