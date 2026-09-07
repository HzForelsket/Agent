from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu
import prefix_grouper_npu

from prefix_grouper_npu import build_shared_prefix_plan, shared_prefix_attention


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


def _measure(fn, warmup: int, iterations: int, record, save):
    for _ in range(warmup): fn()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    for _ in range(iterations):
        start = time.perf_counter(); fn(); torch.npu.synchronize()
        record["samples_ms"].append((time.perf_counter() - start) * 1000)
        record["median_ms"] = statistics.median(record["samples_ms"])
        record["peak_bytes"] = torch.npu.max_memory_allocated()
        save()
    record["status"] = "complete"
    save()


def _check_outputs(custom, baseline):
    actual = custom().detach().float().cpu()
    expected = baseline().detach().float().cpu()
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError("benchmark output shape/finite check failed")
    return {
        "cosine": torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
        "max_abs": (actual - expected).abs().max().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=int, default=1024)
    parser.add_argument("--suffixes", type=int, nargs="+", default=[64, 65, 63, 1])
    parser.add_argument("--hq", type=int, default=6)
    parser.add_argument("--hkv", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if args.prefix <= 0 or any(length <= 0 for length in args.suffixes):
        parser.error("prefix and suffix lengths must be positive")
    if args.hkv <= 0 or args.hq <= 0 or args.hq % args.hkv:
        parser.error("hq and hkv must be positive, and hq must be divisible by hkv")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations must be positive")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error("output already exists; use a fresh result path")
        print(f"Benchmark output: {args.output.resolve()}", flush=True)
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
        "measurement": "forward only; synchronized host wall time; prebuilt inputs and plan",
        "comparison": "raw operator measurements, not an Agent Lightning baseline or end-to-end speedup",
    }

    def save():
        if args.output:
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(args.output)

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
    for name, fn in (("shared_prefix_attention", custom), ("npu_fusion_attention_materialized", baseline)):
        result["status"] = name
        record = {"status": "running", "samples_ms": []}
        result[name] = record
        save()
        _measure(fn, args.warmup, args.iterations, record, save)
    if args.trace_dir:
        result["status"] = "profiling"
        save()
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        for name, fn in (("shared_prefix", custom), ("materialized_fusion_attention", baseline)):
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(args.trace_dir / name)
                ),
                record_shapes=True,
                profile_memory=True,
            ):
                fn()
                torch.npu.synchronize()
        result["trace_dir"] = str(args.trace_dir)
    result["status"] = "complete"
    save()
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)


if __name__ == "__main__":
    main()
