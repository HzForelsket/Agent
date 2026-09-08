"""Device-free checks of host-profile accounting, not NPU timing tests."""
import csv
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "benchmarks" / "host_profile_report.py"
_spec = importlib.util.spec_from_file_location("host_profile_report", _path)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def test_nested_host_totals_are_not_added(tmp_path):
    operator = "shared_prefix_attention"
    rows = [
        (f"pg_attention/{operator}/forward/step_000", 0, 1750.77),
        ("pg_attention/device_synchronize", 0, 343.75),
        ("prefix_grouper_npu::shared_prefix_attention_forward", 512.58, 709.74),
    ]
    rows += [(f"pg_host/custom/{name}", 1, 100) for name in report.CUSTOM_CPP_STAGES]
    rows += [(f"pg_host/custom/{name}", 1, 10) for name in report.CUSTOM_PYTHON_STAGES]
    path = tmp_path / "operator_details.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Name", "Host Self Duration(us)", "Host Total Duration(us)"])
        writer.writerows(rows)
    metrics = report.read_host_metrics(path, operator, "forward", 0)
    assert metrics["missing_events"] == []
    assert metrics["derived"]["before_final_sync_us"] == pytest.approx(1407.02)
    assert metrics["derived"]["cpp_outside_stages_us"] == pytest.approx(409.74)


def test_missing_probe_is_unknown_not_zero(tmp_path):
    path = tmp_path / "operator_details.csv"
    path.write_text("Name,Host Self Duration(us),Host Total Duration(us)\n", encoding="utf-8")
    metrics = report.read_host_metrics(path, "shared_prefix_attention", "forward", 0)
    assert "pg_host/custom/attention_bridge" in metrics["missing_events"]
    assert metrics["derived"] == {}
    assert "pg_host/custom/attention_bridge" not in metrics["events"]


def test_pending_report_has_return_checklist_and_no_fabricated_timings():
    result = {
        "profiling": {"status": "pending", "host_probe_version": 3, "captures": []},
        "python": "/python", "package_path": "/package", "torch": "2.10.0",
        "torch_npu": "2.10.0", "input": {}, "timings": {},
    }
    markdown = report.host_profile_markdown(result)
    assert "需要回传的关键数值" in markdown
    assert "缓存命中未经直接证据确认就填“未知”" in markdown
    assert "attention_bridge | 0" not in markdown
