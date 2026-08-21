# Copyright (c) Microsoft. All rights reserved.

"""Focused tests for PrefixGrouper PPA reporting helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from prefix_grouper import PrefixGrouper

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
_BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "benchmark_prefix_grouper", _SCRIPTS_DIR / "benchmark_prefix_grouper.py"
)
assert _BENCHMARK_SPEC is not None and _BENCHMARK_SPEC.loader is not None
_BENCHMARK_MODULE: Any = importlib.util.module_from_spec(_BENCHMARK_SPEC)
sys.modules[_BENCHMARK_SPEC.name] = _BENCHMARK_MODULE
_BENCHMARK_SPEC.loader.exec_module(_BENCHMARK_MODULE)


def test_power_parsers_and_distribution_statistics() -> None:
    assert _BENCHMARK_MODULE.parse_power_watts("npu", "NPU Real-time Power(W) : 287.4\n") == 287.4
    assert _BENCHMARK_MODULE.parse_power_watts("npu", "Power Dissipation(W) : 301.2\n") == 301.2
    assert _BENCHMARK_MODULE.parse_power_watts("gpu", "312.50\n") == 312.5
    assert _BENCHMARK_MODULE.parse_power_watts("npu", "not supported") is None

    stats = _BENCHMARK_MODULE.sample_statistics([1, 2, 3, 4, 5])
    assert stats["p50"] == 3
    assert stats["p95"] == pytest.approx(4.8)
    assert stats["p99"] == pytest.approx(4.96)
    assert stats["mean"] == 3
    assert stats["coefficient_of_variation"] == pytest.approx(0.5270462767)


def test_energy_efficiency_uses_measured_power_and_idle_baseline() -> None:
    power = {
        "enabled": True,
        "available": True,
        "watts": {"mean": 300.0},
    }
    result = _BENCHMARK_MODULE.add_energy_efficiency(
        power,
        latency_ms=100.0,
        response_tokens=600,
        idle_watts=100.0,
    )
    assert result["estimated_energy_joules_per_step"] == 30.0
    assert result["response_tokens_per_joule"] == 20.0
    assert result["dynamic_watts"] == 200.0
    assert result["estimated_dynamic_energy_joules_per_step"] == 20.0
    assert result["response_tokens_per_dynamic_joule"] == 30.0


def test_workload_metrics_report_dense_tokens_and_causal_pairs() -> None:
    prefix_mask = torch.ones((2, 3), dtype=torch.bool)
    suffix_mask = torch.ones((4, 2), dtype=torch.bool)
    grouper = PrefixGrouper.from_ungrouped_masks(  # pyright: ignore[reportUnknownMemberType]
        prefix_mask, suffix_mask, group_sizes=[2, 2]
    )
    batch = {
        "input_ids": torch.zeros((4, 5), dtype=torch.long),
        "responses": torch.zeros((4, 2), dtype=torch.long),
        "grouped_ids": torch.zeros(grouper.x_shape, dtype=torch.long),
        "grouper": grouper,
    }

    metrics = _BENCHMARK_MODULE.workload_metrics(batch)
    assert metrics["response_tokens"] == 8
    assert metrics["baseline_dense_model_tokens"] == 20
    assert metrics["prefix_grouper_dense_model_tokens"] == 14
    assert metrics["dense_model_token_saved_ratio"] == pytest.approx(0.3)
    assert metrics["baseline_causal_attention_pairs"] == 60
    assert metrics["prefix_grouper_causal_attention_pairs"] == 48
    assert metrics["causal_attention_pair_saved_ratio"] == pytest.approx(0.2)
