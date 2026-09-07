from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_shared_prefix_attention.py"
_SPEC = importlib.util.spec_from_file_location("shared_prefix_benchmark", _PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


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
