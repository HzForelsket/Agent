from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

torch_npu = pytest.importorskip("torch_npu")

from prefix_grouper_npu import build_shared_prefix_plan, shared_prefix_attention
from reference import dense_lse_reference, materialized_reference


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


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = (actual - expected).abs()
    worst = tuple(index.item() for index in torch.unravel_index(error.argmax(), error.shape))
    return {
        "cosine": float(F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)),
        "max_abs": float(error[worst]),
        "worst_index": worst,
        "actual_at_worst": float(actual[worst]),
        "expected_at_worst": float(expected[worst]),
    }


@pytest.mark.parametrize("prefix_lens,suffix_lens,group_sizes,hq,hkv", CASES)
def test_forward_backward_against_materialized_fp32_reference(
    prefix_lens, suffix_lens, group_sizes, hq, hkv
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
    print("PREFIX_GROUPER_NPU_RESULT=" + json.dumps(metrics, sort_keys=True), flush=True)
    assert metrics["out"]["cosine"] >= 0.999
    assert metrics["out"]["max_abs"] <= 0.05
    for name in ("dq", "dk", "dv"):
        assert metrics[name]["cosine"] >= 0.999
        assert metrics[name]["max_abs"] <= 0.1


def _small_inputs(total, hq, hkv, seed, dim=128):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(total, hq, dim, generator=rng, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn(total, hkv, dim, generator=rng, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn(total, hkv, dim, generator=rng, dtype=torch.float32).to(torch.bfloat16)
    grad = torch.randn(q.shape, generator=rng, dtype=torch.bfloat16)
    return q, k, v, grad


def _check_small_call(name, seeds, metadata, buffers, scale=None):
    q_seed, k_seed, v_seed, grad_seed = seeds
    q_ref, k_ref, v_ref = (tensor.float().requires_grad_(True) for tensor in seeds[:3])
    out_ref = materialized_reference(q_ref, k_ref, v_ref, *metadata, scale=scale)
    out_ref.backward(grad_seed.float())
    lse_ref = dense_lse_reference(q_seed, k_seed, *metadata, scale=scale)
    q, k, v = buffers
    with torch.no_grad():
        for target, source in zip(buffers, seeds[:3], strict=True):
            target.copy_(source)
            target.grad = None
    plan = build_shared_prefix_plan(*metadata, device=q.device)
    saved_lse = []

    def pack(tensor):
        saved = tensor.detach()
        if saved.dtype == torch.float32 and saved.shape == q.shape[:2]:
            saved_lse.append(saved)
        return saved

    # Observe the LSE saved by the real forward; do not substitute a reference or rerun it.
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = shared_prefix_attention(q, k, v, plan, softmax_scale=scale)
    out.backward(grad_seed.npu())
    torch.npu.synchronize()
    assert len(saved_lse) == 1
    tensors = {"out": out, "dq": q.grad, "dk": k.grad, "dv": v.grad}
    expected = {"out": out_ref, "dq": q_ref.grad, "dk": k_ref.grad, "dv": v_ref.grad}
    for tensor in tensors.values():
        assert tensor.dtype == torch.bfloat16
    actual = {key: value.detach().float().cpu().clone() for key, value in tensors.items()}
    actual["lse"] = saved_lse[0].cpu().clone()
    metrics = {
        "case": {"name": name, "prefix_lens": metadata[0], "suffix_lens": metadata[1],
                 "group_sizes": metadata[2], "hq": q.shape[1], "hkv": k.shape[1], "head_dim": q.shape[-1], "scale": scale},
        **{key: _metric(actual[key], expected[key]) for key in expected},
        "lse": _metric(actual["lse"], lse_ref),
    }
    print("PREFIX_GROUPER_NPU_RESULT=" + json.dumps(metrics, sort_keys=True), flush=True)
    for key in expected:
        assert metrics[key]["cosine"] >= 0.999, (name, key, metrics[key])
        assert metrics[key]["max_abs"] <= (0.05 if key == "out" else 0.1), (name, key, metrics[key])
    torch.testing.assert_close(actual["lse"], lse_ref, rtol=1e-5, atol=1e-6)
    return actual


@pytest.mark.parametrize("total", [15, 16, 17], ids=["lse-15", "lse-16", "lse-17"])
def test_lse_block_boundaries(total):
    seeds = _small_inputs(total, 1, 1, 1234)
    buffers = tuple(torch.empty_like(tensor, device="npu", requires_grad=True) for tensor in seeds[:3])
    _check_small_call(f"lse_boundary_{total}", seeds, ((1,), (1, total - 2), (2,)), buffers)


def test_same_process_a_b_a_reuses_buffers_and_plans():
    a = _small_inputs(17, 2, 1, 1234, 129)
    b = _small_inputs(17, 2, 1, 5678, 129)
    buffers = tuple(torch.empty_like(tensor, device="npu", requires_grad=True) for tensor in a[:3])
    metadata_a = ((1,), (1, 15), (2,))
    metadata_b = ((2,), (3, 12), (2,))
    plan_a = build_shared_prefix_plan(*metadata_a, device=buffers[0].device)
    first = _check_small_call("A1", a, metadata_a, buffers)
    _check_small_call("B", b, metadata_b, buffers)
    assert build_shared_prefix_plan(*metadata_a, device=buffers[0].device) is plan_a
    last = _check_small_call("A2", a, metadata_a, buffers)
    # A1 and A2 each pass the independent oracle; bitwise determinism is not promised.
    print("PREFIX_GROUPER_NPU_REPEAT=" + json.dumps({
        key: _metric(last[key], first[key]) for key in first
    }, sort_keys=True), flush=True)


def test_projection_attention_loss_autograd_integration():
    rng = torch.Generator(device="cpu").manual_seed(1234)
    metadata = ((1, 2), (1, 2, 1), (1, 2))
    seeds = {
        "hidden": torch.randn(7, 8, generator=rng, dtype=torch.float32),
        "wq": torch.randn(256, 8, generator=rng, dtype=torch.float32) * 0.1,
        "wk": torch.randn(128, 8, generator=rng, dtype=torch.float32) * 0.1,
        "wv": torch.randn(128, 8, generator=rng, dtype=torch.float32) * 0.1,
        "wo": torch.randn(8, 256, generator=rng, dtype=torch.float32) * 0.1,
    }
    seeds = {key: value.to(torch.bfloat16) for key, value in seeds.items()}
    ref = {key: value.float().requires_grad_(True) for key, value in seeds.items()}
    npu = {key: value.npu().requires_grad_(True) for key, value in seeds.items()}
    target = torch.randn(7, 8, generator=rng, dtype=torch.float32)
    plan = build_shared_prefix_plan(*metadata, device=npu["hidden"].device)

    def project(values, attention):
        hidden = values["hidden"]
        q = F.linear(hidden, values["wq"]).view(7, 2, 128)
        k = F.linear(hidden, values["wk"]).view(7, 1, 128)
        v = F.linear(hidden, values["wv"]).view(7, 1, 128)
        return hidden + F.linear(attention(q, k, v).reshape(7, 256), values["wo"])

    expected = project(ref, lambda q, k, v: materialized_reference(q, k, v, *metadata))
    expected_loss = (expected - target).square().mean()
    expected_loss.backward()
    actual = project(npu, lambda q, k, v: shared_prefix_attention(q, k, v, plan))
    loss = (actual.float() - target.npu()).square().mean()
    loss.backward()
    torch.npu.synchronize()
    metrics = {"out": _metric(actual, expected), **{
        key: _metric(npu[key].grad, ref[key].grad) for key in seeds
    }}
    print("PREFIX_GROUPER_NPU_INTEGRATION=" + json.dumps({
        "case": "projection_attention_residual_loss", "metrics": metrics,
        "loss": loss.item(), "expected_loss": expected_loss.item(),
    }, sort_keys=True), flush=True)
    assert actual.dtype == torch.bfloat16 and loss.dtype == torch.float32
    torch.testing.assert_close(loss.cpu(), expected_loss.detach(), rtol=0.02, atol=0.001)
    for name, metric in metrics.items():
        assert metric["cosine"] >= 0.999, (name, metric)
        assert metric["max_abs"] <= (0.02 if name == "out" else 0.01), (name, metric)


def test_invalid_tensor_contracts() -> None:
    plan = build_shared_prefix_plan([1], [1, 1], [2], device="npu")
    q = torch.empty((3, 2, 128), device="npu", dtype=torch.bfloat16)
    k = torch.empty((3, 1, 128), device="npu", dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="bfloat16"):
        shared_prefix_attention(q.float(), k, k, plan)
    with pytest.raises(ValueError, match="same positive head_dim"):
        shared_prefix_attention(q[:, :, :64].contiguous(), k, k, plan)
    with pytest.raises(ValueError, match="same positive head_dim"):
        shared_prefix_attention(q[:, :, :0].contiguous(), k[:, :, :0].contiguous(), k[:, :, :0].contiguous(), plan)
    noncontiguous_q = q.transpose(0, 1).contiguous().transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous"):
        shared_prefix_attention(noncontiguous_q, k, k, plan)
    bad_heads = torch.empty((3, 3, 128), device="npu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible"):
        shared_prefix_attention(bad_heads, torch.empty((3, 2, 128), device="npu", dtype=torch.bfloat16),
                                torch.empty((3, 2, 128), device="npu", dtype=torch.bfloat16), plan)


@pytest.mark.parametrize("dim", [1, 15, 16, 17, 127, 128, 129, 256, 257, 513])
def test_dynamic_head_dimension_forward_lse_and_gradients(dim):
    metadata = ((2, 1), (2, 1, 3), (2, 1))
    seeds = _small_inputs(9, 3, 1, 2468, dim)
    buffers = tuple(torch.empty_like(tensor, device="npu", requires_grad=True) for tensor in seeds[:3])
    _check_small_call(f"dynamic_d_{dim}", seeds, metadata, buffers, scale=0.125 if dim == 17 else None)


@pytest.mark.parametrize("length", [63, 64, 65, 129])
def test_pipeline_boundaries_with_unaligned_dimension(length):
    metadata = ((1,), (length, 1), (2,))
    seeds = _small_inputs(length + 2, 2, 1, 1357, 17)
    buffers = tuple(torch.empty_like(tensor, device="npu", requires_grad=True) for tensor in seeds[:3])
    _check_small_call(f"pipeline_{length}", seeds, metadata, buffers)
