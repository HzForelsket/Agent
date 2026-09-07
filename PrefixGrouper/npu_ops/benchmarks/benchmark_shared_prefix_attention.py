from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch


def _baseline_inputs(q, k, v, prefix_lens, suffix_lens, group_sizes):
    q_parts, k_parts, v_parts = [], [], []
    q_cumulative, kv_cumulative = [], []
    token_offset = suffix_index = q_total = kv_total = 0
    for prefix_len, group_size in zip(prefix_lens, group_sizes, strict=True):
        p = slice(token_offset, token_offset + prefix_len)
        q_parts.append(q[p]); k_parts.append(k[p]); v_parts.append(v[p])
        q_total += prefix_len; kv_total += prefix_len
        q_cumulative.append(q_total); kv_cumulative.append(kv_total)
        token_offset += prefix_len
        for _ in range(group_size):
            suffix_len = suffix_lens[suffix_index]; suffix_index += 1
            s = slice(token_offset, token_offset + suffix_len)
            q_parts.append(q[s])
            k_parts.append(torch.cat((k[p], k[s]), dim=0))
            v_parts.append(torch.cat((v[p], v[s]), dim=0))
            q_total += suffix_len; kv_total += prefix_len + suffix_len
            q_cumulative.append(q_total); kv_cumulative.append(kv_total)
            token_offset += suffix_len
    return (torch.cat(q_parts), torch.cat(k_parts), torch.cat(v_parts), q_cumulative, kv_cumulative)


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


def _measure(prepare, warmup: int, iterations: int, record, save):
    for _ in range(warmup):
        step = prepare()
        step()
        del step
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    for _ in range(iterations):
        step = prepare()
        # In backward mode, wait for the untimed forward to finish first.
        torch.npu.synchronize()
        start = time.perf_counter()
        step()
        torch.npu.synchronize()
        record["samples_ms"].append((time.perf_counter() - start) * 1000)
        record["median_ms"] = statistics.median(record["samples_ms"])
        record["peak_bytes"] = torch.npu.max_memory_allocated()
        del step
        save()
    record["status"] = "complete"
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


def _fold_baseline_gradients(grads, prefix_lens, suffix_lens, group_sizes):
    """Map expanded fusion gradients to compact tokens, outside measurement."""
    total = sum(prefix_lens) + sum(suffix_lens)
    rows = torch.arange(total)
    q_rows, k_rows, v_rows, _, _ = _baseline_inputs(
        rows, rows, rows, prefix_lens, suffix_lens, group_sizes
    )
    folded = []
    for grad, indices in zip(grads, (q_rows, k_rows, v_rows), strict=True):
        grad = grad.detach().float().cpu()
        compact = grad.new_zeros((total, *grad.shape[1:]))
        folded.append(compact.index_add_(0, indices, grad))
    return tuple(folded)


def _check_gradients(custom, baseline, custom_inputs, baseline_inputs, grad_output, metadata):
    actual = torch.autograd.grad(custom(), custom_inputs, grad_output)
    expanded = torch.autograd.grad(baseline(), baseline_inputs, grad_output)
    expected = _fold_baseline_gradients(expanded, *metadata)
    return {
        name: _metric(grad, reference)
        for name, grad, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True)
    }


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
        "| 模式 | 算子 | 中位耗时（ms） | 已采样次数 | 峰值分配（MiB） | 状态 |",
        "|---|---|---:|---:|---:|---|",
    ]
    labels = {"forward": "前向", "backward": "后向", "forward_backward": "前向＋后向"}
    for mode in result["timing_modes"]:
        records = result if mode == "forward" else result.get(mode, {})
        for name, label in (
            ("shared_prefix_attention", "custom"),
            ("npu_fusion_attention_materialized", "fusion（预展开输入）"),
        ):
            record = records.get(name, {})
            median = f"{record['median_ms']:.6f}" if "median_ms" in record else "—"
            peak = f"{record['peak_bytes'] / 2**20:.2f}" if "peak_bytes" in record else "—"
            lines.append(
                f"| {labels[mode]} | {label} | {median} | {len(record.get('samples_ms', []))} | "
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
        "- 后向每次重新建图并同步，前向建图不计时；前向＋后向包含两者。",
        "- 输入展开和 plan 构建不计时；fusion 的共享前缀梯度累加仅用于正确性检查，不计时。",
        "- 峰值分配为进程级统计，两套输入同时驻留，不代表单个算子的独立显存开销。",
        "- 结果是 custom 与预展开 fusion 的裸算子比较，不代表完整 permute 路径或模型端到端加速。",
        "- 未完成的运行仅保留已有采样，不能作为完整对比结果。", "",
    ])
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
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if args.prefix <= 0 or any(length <= 0 for length in args.suffixes):
        parser.error("prefix and suffix lengths must be positive")
    if args.hkv <= 0 or args.hq <= 0 or args.hq % args.hkv:
        parser.error("hq and hkv must be positive, and hq must be divisible by hkv")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations must be positive")
    if args.output_markdown is None and args.output:
        args.output_markdown = args.output.with_suffix(".md")
    if args.output and args.output_markdown and args.output.resolve() == args.output_markdown.resolve():
        parser.error("JSON and Markdown output paths must be different")
    for path in (args.output, args.output_markdown):
        if path and path.exists():
            parser.error(f"output already exists: {path}; use a fresh result path")
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
        "measurement": "synchronized host wall time; prebuilt inputs and plan",
        "timing_modes": modes,
        "timing_scope": {
            "forward": "no-grad forward; input materialization and plan construction excluded",
            "backward": "autograd.grad with a fresh untimed, synchronized forward per sample",
            "forward_backward": "fresh forward plus autograd.grad in one timing window",
            "gradients": "custom returns compact dq/dk/dv; fusion returns expanded dq/dk/dv; "
                "fusion prefix-gradient reduction is used only for correctness, outside timing",
        },
        "comparison": "raw operator measurements, not an Agent Lightning baseline or end-to-end speedup",
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

    bq, bk, bv, qlens, kvlens = _baseline_inputs(q, k, v, [args.prefix], args.suffixes, [len(args.suffixes)])
    causal = torch.triu(torch.ones((2048, 2048), device="npu", dtype=torch.bool), diagonal=1)
    scale = torch.tensor(128.0, dtype=torch.float32, device="cpu").rsqrt().item()
    custom = lambda: shared_prefix_attention(q, k, v, plan)
    baseline = lambda: torch_npu.npu_fusion_attention(
        bq, bk, bv, head_num=args.hq, input_layout="TND", atten_mask=causal,
        scale=scale, keep_prob=1.0, actual_seq_qlen=qlens, actual_seq_kvlen=kvlens,
        sparse_mode=3,
    )[0]

    result.update({
        "compact_input_bytes": compact_storage,
        "materialized_baseline_input_bytes": (bq.numel() + bk.numel() + bv.numel()) * bq.element_size(),
    })
    result["status"] = "checking_outputs"
    save()
    result["correctness"] = _check_outputs(custom, baseline)
    save()
    if result["correctness"]["cosine"] < 0.999 or result["correctness"]["max_abs"] > 0.05:
        result["status"] = "failed_correctness"
        save()
        raise RuntimeError(f"benchmark outputs disagree: {result['correctness']}")
    custom_inputs, baseline_inputs = (q, k, v), (bq, bk, bv)
    grad_output = None
    if args.backward:
        for tensor in (*custom_inputs, *baseline_inputs):
            tensor.requires_grad_(True)
        grad_output = torch.randn_like(q)
        result["status"] = "checking_gradients"
        save()
        result["gradient_correctness"] = _check_gradients(
            custom, baseline, custom_inputs, baseline_inputs, grad_output,
            ([args.prefix], args.suffixes, [len(args.suffixes)]),
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
        ("shared_prefix_attention", custom, custom_inputs),
        ("npu_fusion_attention_materialized", baseline, baseline_inputs),
    )
    for mode in modes:
        # Preserve the existing top-level forward result keys.
        records = result if mode == "forward" else result.setdefault(mode, {})
        for name, fn, inputs in operators:
            result["status"] = f"{mode}:{name}"
            record = {"status": "running", "samples_ms": []}
            records[name] = record
            save()
            _measure(
                lambda: _prepare_step(fn, inputs, grad_output, mode),
                args.warmup, args.iterations, record, save,
            )
    if args.trace_dir:
        result["status"] = "profiling"
        save()
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        for mode in modes:
            for name, fn, inputs in operators:
                trace_name = "shared_prefix" if name == "shared_prefix_attention" else "materialized_fusion_attention"
                trace_root = args.trace_dir if mode == "forward" else args.trace_dir / mode
                step = _prepare_step(fn, inputs, grad_output, mode)
                torch.npu.synchronize()
                with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_root / trace_name)),
                    record_shapes=True,
                    profile_memory=True,
                ):
                    step()
                    torch.npu.synchronize()
                del step
        result["trace_dir"] = str(args.trace_dir)
    result["status"] = "complete"
    save()
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)


if __name__ == "__main__":
    main()
