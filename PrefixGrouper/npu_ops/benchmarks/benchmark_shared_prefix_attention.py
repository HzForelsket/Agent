from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch


def _baseline_layout(prefix_lens, suffix_lens, group_sizes, device):
    """Prepare reusable metadata only; K/V gathering stays inside the timed call."""
    kv_rows = []
    q_cumulative, kv_cumulative = [], []
    token_offset = suffix_index = q_total = kv_total = 0
    for prefix_len, group_size in zip(prefix_lens, group_sizes, strict=True):
        prefix_rows = range(token_offset, token_offset + prefix_len)
        kv_rows.extend(prefix_rows)
        q_total += prefix_len; kv_total += prefix_len
        q_cumulative.append(q_total); kv_cumulative.append(kv_total)
        token_offset += prefix_len
        for _ in range(group_size):
            suffix_len = suffix_lens[suffix_index]; suffix_index += 1
            kv_rows.extend(prefix_rows)
            kv_rows.extend(range(token_offset, token_offset + suffix_len))
            q_total += suffix_len; kv_total += prefix_len + suffix_len
            q_cumulative.append(q_total); kv_cumulative.append(kv_total)
            token_offset += suffix_len
    return torch.tensor(kv_rows, dtype=torch.int64, device=device), q_cumulative, kv_cumulative


def _prepare_step(fn, inputs, grad_output, mode):
    if mode == "forward":
        return torch.no_grad()(fn)
    if mode == "backward":
        # Rebuild saved tensors for every sample; forward is outside the timer.
        output = fn()
        return lambda: torch.autograd.grad(output, inputs, grad_output)
    if mode == "forward_backward":
        return lambda: torch.autograd.grad(fn(), inputs, grad_output)
    raise ValueError(f"Unknown measurement mode: {mode}")


def _percentile(samples, fraction):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _measure(operators, mode, grad_output, warmup, iterations, records, save):
    # Alternate AB/BA for both warmup and paired samples. Each step owns a fresh
    # graph, and neither path retains outputs or gradients into the next step.
    for name, _, _ in operators:
        records[name] = {"status": "warming_up", "samples_ms": [], "memory_samples": []}
    save()
    for index in range(warmup):
        order = operators if index % 2 == 0 else operators[::-1]
        for _, fn, inputs in order:
            step = _prepare_step(fn, inputs, grad_output, mode)
            output = step()
            torch.npu.synchronize()
            del output, step
    for index in range(iterations):
        order = operators if index % 2 == 0 else operators[::-1]
        for position, (name, fn, inputs) in enumerate(order):
            torch.npu.synchronize()
            resident = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            step = _prepare_step(fn, inputs, grad_output, mode)
            # Backward graph construction is untimed, but its saved tensors
            # count towards the full step's incremental allocation peak.
            torch.npu.synchronize()
            prepared = torch.npu.memory_allocated()
            start = time.perf_counter()
            output = step()
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            record = records[name]
            record["samples_ms"].append(elapsed)
            record["memory_samples"].append({
                "round": index, "position": position,
                "resident_bytes": resident,
                "prepared_bytes": prepared,
                "peak_increment_bytes": max(0, torch.npu.max_memory_allocated() - resident),
            })
            record["status"] = "running"
            del output, step
        # Summaries and report I/O stay outside the pair of device workloads.
        for name, _, _ in operators:
            record = records[name]
            samples = record["samples_ms"]
            record.update({
                "median_ms": statistics.median(samples),
                "mean_ms": statistics.mean(samples), "min_ms": min(samples), "max_ms": max(samples),
                "p25_ms": _percentile(samples, 0.25), "p75_ms": _percentile(samples, 0.75),
                "p95_ms": _percentile(samples, 0.95),
                "peak_increment_bytes": max(s["peak_increment_bytes"] for s in record["memory_samples"]),
            })
        save()
    for name, _, _ in operators:
        records[name]["status"] = "complete"
    save()


def _metric(actual, expected):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError("benchmark tensor shape/finite check failed")
    return {
        # Identical zero gradients are valid, e.g. a one-token causal query.
        "cosine": (
            1.0 if torch.equal(actual, expected) else
            torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
        ),
        "max_abs": (actual - expected).abs().max().item(),
    }


@torch.no_grad()
def _check_outputs(custom, baseline):
    return _metric(custom(), baseline())


def _check_gradients(custom, baseline, inputs, grad_output):
    actual = torch.autograd.grad(custom(), inputs, grad_output)
    expected = torch.autograd.grad(baseline(), inputs, grad_output)
    return {
        name: _metric(grad, reference)
        for name, grad, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True)
    }


def _profile(operators, modes, grad_output, args, result, save, profiler):
    """Capture isolated calls after timing; backward preparation stays untraced."""
    info = result["profiling"]
    info["status"] = "running"
    result["status"] = "profiling"
    save()
    metrics = (profiler.AiCMetrics.AiCoreNone if args.profile_aic_metrics == "None"
               else getattr(profiler.AiCMetrics, args.profile_aic_metrics))
    try:
        for mode in modes:
            for name, fn, inputs in operators:
                for index in range(args.profile_steps):
                    capture_dir = args.trace_dir / mode / name / f"step_{index:03d}"
                    capture_dir.mkdir(parents=True, exist_ok=False)
                    capture = {
                        "mode": mode, "operator": name, "step": index,
                        "directory": str(capture_dir), "status": "warming_up", "artifacts": {},
                    }
                    info["captures"].append(capture)
                    print(f"NPU profile: {capture_dir}", flush=True)
                    save()
                    # Exporting a previous capture can take time. Warm both
                    # operators by the same count immediately before each capture.
                    for _ in range(args.warmup):
                        step = _prepare_step(fn, inputs, grad_output, mode)
                        output = step()
                        torch.npu.synchronize()
                        del output, step
                    step = _prepare_step(fn, inputs, grad_output, mode)
                    torch.npu.synchronize()
                    capture["status"] = "collecting"
                    save()
                    with profiler.profile(
                        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                        on_trace_ready=profiler.tensorboard_trace_handler(
                            str(capture_dir), analyse_flag=True, async_mode=False,
                        ),
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=args.profile_with_stack,
                        experimental_config=profiler._ExperimentalConfig(
                            profiler_level=profiler.ProfilerLevel.Level1,
                            aic_metrics=metrics,
                            export_type=profiler.ExportType.Text,
                            data_simplification=False,
                        ),
                    ) as prof:
                        prof.add_metadata_json("benchmark", json.dumps({
                            "benchmark_id": result["benchmark_id"],
                            "script_sha256": result["script_sha256"],
                            "input": result["input"],
                            "mode": mode, "operator": name, "step": index,
                            "scope": result["timing_scope"][mode],
                        }))
                        with torch.profiler.record_function(f"pg_attention/{name}/{mode}/step_{index:03d}"):
                            output = step()
                            with torch.profiler.record_function("pg_attention/device_synchronize"):
                                torch.npu.synchronize()
                    del output, step
                    capture["status"] = "checking_export"
                    # torch-npu catches some collection/export exceptions itself.
                    # Do not report successful profiling just because the context exited.
                    for filename in ("trace_view.json", "kernel_details.csv", "operator_details.csv"):
                        paths = list(capture_dir.rglob(filename))
                        if len(paths) != 1 or paths[0].stat().st_size == 0:
                            raise RuntimeError(f"Missing or ambiguous profiler export {filename} under {capture_dir}")
                        capture["artifacts"][filename] = str(paths[0])
                    with Path(capture["artifacts"]["kernel_details.csv"]).open(encoding="utf-8-sig", newline="") as stream:
                        kernels = csv.DictReader(stream)
                        capture["kernel_columns"] = kernels.fieldnames
                        capture["kernel_count"] = sum(1 for _ in kernels)
                    if not capture["kernel_count"]:
                        raise RuntimeError(f"Profiler exported no NPU kernel records under {capture_dir}")
                    capture["status"] = "complete"
                    save()
    except Exception as exc:
        info["status"] = "failed"
        info["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if info["captures"] and info["captures"][-1]["status"] != "complete":
            info["captures"][-1]["status"] = "failed"
        result["status"] = "failed_profiling"
        save()
        raise
    info["status"] = "complete"
    save()


def _markdown_report(result):
    inputs = result["input"]
    status = result["status"]
    lines = [
        "# 共享前缀 Attention 单算子测速", "",
        f"运行状态：{'完成' if status == 'complete' else '未完成'}（`{status}`）。", "",
        f"- 设备：{result.get('device', '尚未获取')}",
        f"- 环境：PyTorch {result['torch']}；torch-npu {result['torch_npu']}；"
        f"prefix-grouper-npu {result['prefix_grouper_npu']}",
        f"- 输入：prefix={inputs['prefix']}，suffixes={inputs['suffixes']}，"
        f"Hq={inputs['hq']}，Hkv={inputs['hkv']}，head_dim=128，BF16。",
        f"- 预热：{inputs['warmup']} 次；每项采样：{inputs['iterations']} 次。", "",
        "## 耗时", "",
        "| 模式 | 算子 | 中位耗时（ms） | P25–P75（ms） | P95（ms） | 已采样次数 | 峰值增量（MiB） | 状态 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    labels = {"forward": "前向", "backward": "后向", "forward_backward": "前向＋后向"}
    for mode in result["timing_modes"]:
        records = result["timings"].get(mode, {})
        for name, label in (
            ("shared_prefix_attention", "custom"),
            ("npu_fusion_attention_compact", "fusion（含展开及梯度归并）"),
        ):
            record = records.get(name, {})
            median = f"{record['median_ms']:.6f}" if "median_ms" in record else "—"
            spread = f"{record['p25_ms']:.6f}–{record['p75_ms']:.6f}" if "p25_ms" in record else "—"
            p95 = f"{record['p95_ms']:.6f}" if "p95_ms" in record else "—"
            peak = f"{record['peak_increment_bytes'] / 2**20:.2f}" if "peak_increment_bytes" in record else "—"
            lines.append(
                f"| {labels[mode]} | {label} | {median} | {spread} | {p95} | {len(record.get('samples_ms', []))} | "
                f"{peak} | {record.get('status', '未开始')} |"
            )
    lines.extend([
        "", "## 正确性检查", "",
        "| 项目 | 余弦相似度 | 最大绝对误差 | 结果 |",
        "|---|---:|---:|---|",
    ])
    checks = [("输出", result.get("correctness"), 0.05)]
    if "backward" in result["timing_modes"]:
        checks.extend((name, result.get("gradient_correctness", {}).get(name), 0.1)
                      for name in ("dq", "dk", "dv"))
    for name, metric, tolerance in checks:
        if metric is None:
            lines.append(f"| {name} | — | — | 未完成 |")
        else:
            passed = metric["cosine"] >= 0.999 and metric["max_abs"] <= tolerance
            lines.append(
                f"| {name} | {metric['cosine']:.6f} | {metric['max_abs']:.6f} | "
                f"{'通过' if passed else '失败'} |"
            )
    lines.extend([
        "", "## 计时范围", "",
        "- 使用设备同步后的主机墙钟耗时；前向使用 no-grad。",
        "- 两边共用紧凑 Q/K/V、上游梯度和精度；每轮交替 AB/BA，预热不计入样本。",
        "- 后向每次重新建图并同步，前向建图不计时；前向＋后向包含两者。",
        "- 双方复用预先构建的 plan/索引/序列长度/掩码；fusion 的 K/V 展开计入前向，梯度归并计入后向。",
        "- 峰值增量为每次准备建图前的驻留分配之上的最大分配量；后向包含未计时前向的保存张量。",
        "- 双方元数据同时驻留；峰值增量不是独立进程总显存，也不包含缓存分配器预留内存。",
        "- 使用相同紧凑输入到输出/梯度的算子路径；不代表 Agent Lightning 或模型端到端加速。",
        "- 输出/梯度保留到计时结束；报告写入在每对采样结束后，profiler 在所有采样结束后运行。",
        "- 未完成的运行仅保留已有采样，不能作为完整对比结果。", "",
    ])
    profiling = result.get("profiling")
    if profiling:
        lines.extend([
            "## NPU Profile", "",
            f"采集状态：`{profiling['status']}`；Level1；请求的 AI Core 指标：`{profiling['aic_metrics']}`。",
            f"每条路径采集 {profiling['steps']} 次，每次采集前额外预热 {inputs['warmup']} 次。", "",
            "- CPU/NPU 时间线、输入形状和内存分配均开启；堆栈采集："
            f"{'开启' if profiling['with_stack'] else '关闭'}。",
            "- 反向建图在采集外完成；时间线中的 pg_attention 范围标识路径及模式。",
            "- Profile 包含采集开销，只用于定位瓶颈；上方耗时来自未开启 profiler 的采样。",
            "- trace_view.json 查看调度与重叠；kernel_details.csv 查看设备任务耗时及可用硬件指标；"
            "operator_details.csv 查看框架算子与设备耗时关联。",
            "- 硬件指标以实机导出列为准；分配记录不能直接说明内核内部 GM/UB 搬运量。", "",
            "| 模式 | 算子 | 采样 | 设备任务数 | 状态 | 文件 |",
            "|---|---|---:|---:|---|---|",
        ])
        for capture in profiling["captures"]:
            links = " · ".join(f"[{name}](<{path}>)" for name, path in capture["artifacts"].items())
            lines.append(
                f"| {labels[capture['mode']]} | {capture['operator']} | {capture['step']} | "
                f"{capture.get('kernel_count', '—')} | {capture['status']} | {links or capture['directory']} |"
            )
        if "error" in profiling:
            lines.extend(["", f"采集失败：{profiling['error']['type']}：{profiling['error']['message']}"])
        lines.append("")
    return "\n".join(lines)


def _save_results(result, json_path, markdown_path):
    reports = []
    if json_path:
        reports.append((json_path, json.dumps(result, indent=2, sort_keys=True) + "\n"))
    if markdown_path:
        reports.append((markdown_path, _markdown_report(result)))
    for path, contents in reports:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=int, default=1024)
    parser.add_argument("--suffixes", type=int, nargs="+", default=[64, 65, 63, 1])
    parser.add_argument("--hq", type=int, default=6)
    parser.add_argument("--hkv", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--backward", action=argparse.BooleanOptionalAction, default=False,
        help="also check gradients and time backward-only and forward+backward",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--output-markdown", type=Path,
        help="Markdown report path; defaults to the --output path with a .md suffix",
    )
    parser.add_argument("--trace-dir", type=Path, help="new directory for CPU/NPU profiles collected after timing")
    parser.add_argument("--profile-steps", type=int, default=1,
                        help="isolated profile captures per operator and mode; requires --trace-dir")
    parser.add_argument("--profile-aic-metrics", default="PipeUtilization",
                        choices=["None", "PipeUtilization", "Memory", "MemoryL0", "MemoryUB", "ResourceConflictRatio"],
                        help="one AI Core metric group per run; requires --trace-dir")
    parser.add_argument("--profile-with-stack", action="store_true",
                        help="include Python stacks in the profile; requires --trace-dir")
    args = parser.parse_args()
    if args.prefix <= 0 or any(length <= 0 for length in args.suffixes):
        parser.error("prefix and suffix lengths must be positive")
    if args.hkv <= 0 or args.hq <= 0 or args.hq % args.hkv:
        parser.error("hq and hkv must be positive, and hq must be divisible by hkv")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations must be positive")
    if args.profile_steps <= 0:
        parser.error("profile-steps must be positive")
    if not args.trace_dir and (args.profile_steps != 1 or args.profile_aic_metrics != "PipeUtilization"
                               or args.profile_with_stack):
        parser.error("profile options require --trace-dir")
    if args.trace_dir:
        args.trace_dir = args.trace_dir.resolve()
        if args.trace_dir.exists():
            parser.error(f"trace directory already exists: {args.trace_dir}; use a fresh path")
        if args.output is None:
            args.output = args.trace_dir / "benchmark.json"
    if args.output_markdown is None and args.output:
        args.output_markdown = args.output.with_suffix(".md")
    if args.output and args.output_markdown and args.output.resolve() == args.output_markdown.resolve():
        parser.error("JSON and Markdown output paths must be different")
    for path in (args.output, args.output_markdown):
        if path and path.exists():
            parser.error(f"output already exists: {path}; use a fresh result path")
        if path and args.trace_dir and (path.resolve() == args.trace_dir or path.resolve() in args.trace_dir.parents):
            parser.error("an output file cannot be the trace directory or its parent")
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=False)
        print(f"NPU profile output: {args.trace_dir}", flush=True)
    for path in (args.output, args.output_markdown):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Benchmark output: {path.resolve()}", flush=True)

    import torch_npu
    import prefix_grouper_npu
    from prefix_grouper_npu import build_shared_prefix_plan, shared_prefix_attention

    modes = ["forward", "backward", "forward_backward"] if args.backward else ["forward"]
    result = {
        "benchmark_id": "pg-ascend-shared-prefix-attention",
        "status": "initializing",
        "command": sys.argv,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.executable,
        "architecture": platform.machine(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "prefix_grouper_npu": prefix_grouper_npu.__version__,
        "package_path": prefix_grouper_npu.__file__,
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
        "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "input": {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()},
        "schema_version": 2,
        "measurement": "synchronized host wall time; alternating paired AB/BA samples",
        "timing_modes": modes,
        "timings": {},
        "timing_scope": {
            "forward": "no-grad compact input to compact output; fusion K/V gathering included",
            "backward": "autograd.grad with a fresh untimed, synchronized forward per sample",
            "forward_backward": "fresh forward plus autograd.grad in one timing window",
            "gradients": "both return compact dq/dk/dv; fusion gather backward and prefix reduction included",
            "setup": "input generation, plan, gather indices, sequence lengths and causal mask excluded for both",
            "memory": "per-sample peak allocated bytes minus residency before graph preparation; "
                "includes untimed backward graph preparation; not isolated process memory",
        },
        "comparison": "equivalent compact-input attention paths, not an Agent Lightning baseline or end-to-end speedup",
    }
    if args.trace_dir:
        result["profiling"] = {
            "status": "pending", "directory": str(args.trace_dir), "steps": args.profile_steps,
            "level": "Level1", "aic_metrics": args.profile_aic_metrics,
            "with_stack": args.profile_with_stack, "record_shapes": True, "profile_memory": True,
            "captures": [],
        }

    def save():
        _save_results(result, args.output, args.output_markdown)

    save()
    if not torch.npu.is_available():
        raise RuntimeError("benchmark requires a real Ascend 910B")
    result["device"] = torch.npu.get_device_name(0)
    torch.manual_seed(1234)
    torch.npu.manual_seed_all(1234)
    result["seed"] = 1234
    save()

    total = args.prefix + sum(args.suffixes)
    q = torch.randn(total, args.hq, 128, device="npu", dtype=torch.bfloat16)
    k = torch.randn(total, args.hkv, 128, device="npu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    plan = build_shared_prefix_plan([args.prefix], args.suffixes, [len(args.suffixes)], device="npu")
    compact_storage = (q.numel() + k.numel() + v.numel()) * q.element_size()

    kv_rows, qlens, kvlens = _baseline_layout([args.prefix], args.suffixes, [len(args.suffixes)], q.device)
    causal = torch.triu(torch.ones((2048, 2048), device="npu", dtype=torch.bool), diagonal=1)
    scale = torch.tensor(128.0, dtype=torch.float32, device="cpu").rsqrt().item()
    custom = lambda: shared_prefix_attention(q, k, v, plan)
    baseline = lambda: torch_npu.npu_fusion_attention(
        q, k.index_select(0, kv_rows), v.index_select(0, kv_rows),
        head_num=args.hq, input_layout="TND", atten_mask=causal,
        scale=scale, keep_prob=1.0, actual_seq_qlen=qlens, actual_seq_kvlen=kvlens,
        sparse_mode=3,
    )[0]

    result.update({
        "compact_input_bytes": compact_storage,
        "fusion_materialized_kv_bytes": kv_rows.numel() * (args.hkv * 128 * 2) * k.element_size(),
        "fusion_index_bytes": kv_rows.numel() * kv_rows.element_size(),
        "fusion_mask_bytes": causal.numel() * causal.element_size(),
        "custom_plan_bytes": sum(getattr(plan, name).numel() * getattr(plan, name).element_size()
                                 for name in ("prefix_start", "prefix_end", "sequence_start", "sequence_end", "group_end")),
    })
    result["status"] = "checking_outputs"
    save()
    result["correctness"] = _check_outputs(custom, baseline)
    save()
    if result["correctness"]["cosine"] < 0.999 or result["correctness"]["max_abs"] > 0.05:
        result["status"] = "failed_correctness"
        save()
        raise RuntimeError(f"benchmark outputs disagree: {result['correctness']}")
    inputs = (q, k, v)
    grad_output = None
    if args.backward:
        for tensor in inputs:
            tensor.requires_grad_(True)
        grad_output = torch.randn_like(q)
        result["status"] = "checking_gradients"
        save()
        result["gradient_correctness"] = _check_gradients(
            custom, baseline, inputs, grad_output,
        )
        save()
        if any(
            metric["cosine"] < 0.999 or metric["max_abs"] > 0.1
            for metric in result["gradient_correctness"].values()
        ):
            result["status"] = "failed_gradient_correctness"
            save()
            raise RuntimeError(f"benchmark gradients disagree: {result['gradient_correctness']}")

    operators = (
        ("shared_prefix_attention", custom, inputs),
        ("npu_fusion_attention_compact", baseline, inputs),
    )
    for mode in modes:
        records = result["timings"].setdefault(mode, {})
        result["status"] = f"measuring:{mode}"
        try:
            _measure(
                operators, mode, grad_output, args.warmup, args.iterations, records, save,
            )
        except Exception as exc:
            result["status"] = "failed_measurement"
            result["error"] = {"mode": mode, "type": type(exc).__name__, "message": str(exc)}
            save()
            raise
    if args.trace_dir:
        _profile(operators, modes, grad_output, args, result, save, torch_npu.profiler)
    result["status"] = "complete"
    save()
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)


if __name__ == "__main__":
    main()
