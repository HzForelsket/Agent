from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from reference import _attention, materialized_reference


_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_shared_prefix_attention.py"
_SPEC = importlib.util.spec_from_file_location("shared_prefix_benchmark", _PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_expanded_gradients_accumulate_prefix_copies_without_crossing_groups():
    metadata = ((1, 2), (1, 2, 1), (1, 2))
    dq = torch.arange(7.0).reshape(7, 1, 1)
    dk = torch.arange(12.0).reshape(12, 1, 1)
    dv = torch.ones_like(dk)
    actual = benchmark._fold_baseline_gradients((dq, dk, dv), *metadata)
    expected = (
        dq,
        torch.tensor([1, 2, 17, 20, 7, 8, 11]).reshape(7, 1, 1).float(),
        torch.tensor([2, 1, 3, 3, 1, 1, 1]).reshape(7, 1, 1).float(),
    )
    for grad, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(grad, reference)


def test_gradient_check_compares_same_compact_tokens_and_output_gradient():
    torch.manual_seed(1234)
    metadata = ((1, 2), (1, 2, 1), (1, 2))
    inputs = tuple(torch.randn(7, heads, 4, requires_grad=True) for heads in (2, 1, 1))
    bq, bk, bv, qends, kvends = benchmark._baseline_inputs(*inputs, *metadata)
    expanded = tuple(tensor.detach().requires_grad_(True) for tensor in (bq, bk, bv))

    def fusion_reference():
        parts = []
        qstart = kvstart = 0
        for qend, kvend in zip(qends, kvends, strict=True):
            nq, nkv = qend - qstart, kvend - kvstart
            mask = torch.ones(nq, nkv, dtype=torch.bool).triu(diagonal=nkv - nq + 1)
            parts.append(_attention(
                expanded[0][qstart:qend], expanded[1][kvstart:kvend], expanded[2][kvstart:kvend],
                mask, 128 ** -0.5,
            ))
            qstart, kvstart = qend, kvend
        return torch.cat(parts)

    metrics = benchmark._check_gradients(
        lambda: materialized_reference(*inputs, *metadata), fusion_reference,
        inputs, expanded, torch.randn_like(inputs[0]), metadata,
    )
    for metric in metrics.values():
        assert metric["cosine"] >= 0.99999
        assert metric["max_abs"] < 1e-6
    assert all(tensor.grad is None for tensor in (*inputs, *expanded))


@pytest.mark.parametrize("mode", ["forward", "backward", "forward_backward"])
def test_measurement_modes_rebuild_graph_and_do_not_accumulate_leaf_gradients(mode):
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    grad_output = torch.tensor([0.5, 2.0])
    forwards = []

    def forward():
        forwards.append(torch.is_grad_enabled())
        return x.square()

    for iteration in range(3):
        step = benchmark._prepare_step(forward, (x,), grad_output, mode)
        # Only backward-only runs forward during preparation (outside timing).
        assert len(forwards) == iteration + (mode == "backward")
        actual = step()
        if mode == "forward":
            torch.testing.assert_close(actual, x.detach().square())
            assert not actual.requires_grad
        else:
            torch.testing.assert_close(actual[0], 2 * x.detach() * grad_output)
        assert x.grad is None
        assert len(forwards) == iteration + 1
    assert forwards == [mode != "forward"] * 3


def test_zero_gradients_pass_and_nonfinite_gradients_fail():
    assert benchmark._metric(torch.zeros(3), torch.zeros(3)) == {"cosine": 1.0, "max_abs": 0.0}
    assert benchmark._metric(torch.ones(3), torch.zeros(3))["max_abs"] == 1.0
    for value in (float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="shape/finite"):
            benchmark._metric(torch.tensor([value]), torch.ones(1))
