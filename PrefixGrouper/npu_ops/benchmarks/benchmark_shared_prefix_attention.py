from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu

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


def _measure(fn, warmup: int, iterations: int):
    for _ in range(warmup): fn()
    torch.npu.synchronize()
    samples = []
    torch.npu.reset_peak_memory_stats()
    for _ in range(iterations):
        start = time.perf_counter(); fn(); torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(samples), "samples_ms": samples,
            "peak_bytes": torch.npu.max_memory_allocated()}


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
    if not torch.npu.is_available():
        raise RuntimeError("benchmark requires a real Ascend 910B")

    total = args.prefix + sum(args.suffixes)
    q = torch.randn(total, args.hq, 128, device="npu", dtype=torch.bfloat16)
    k = torch.randn(total, args.hkv, 128, device="npu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    plan = build_shared_prefix_plan([args.prefix], args.suffixes, [len(args.suffixes)], device="npu")
    compact_storage = (q.numel() + k.numel() + v.numel()) * q.element_size()

    bq, bk, bv, qlens, kvlens = _baseline_inputs(q, k, v, [args.prefix], args.suffixes, [len(args.suffixes)])
    causal = torch.triu(torch.ones((2048, 2048), device="npu", dtype=torch.bool), diagonal=1)
    scale = 1.0 / math.sqrt(128.0)
    custom = lambda: shared_prefix_attention(q, k, v, plan)
    baseline = lambda: torch_npu.npu_fusion_attention(
        bq, bk, bv, head_num=args.hq, input_layout="TND", atten_mask=causal,
        scale=scale, keep_prob=1.0, actual_seq_qlen=qlens, actual_seq_kvlen=kvlens,
        sparse_mode=3,
    )[0]

    result = {
        "command": sys.argv,
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "cann": "9.0.0",
        "input": vars(args) | {"output": str(args.output) if args.output else None},
        "compact_input_bytes": compact_storage,
        "materialized_baseline_input_bytes": (bq.numel() + bk.numel() + bv.numel()) * bq.element_size(),
        "shared_prefix_attention": _measure(custom, args.warmup, args.iterations),
        "npu_fusion_attention_materialized": _measure(baseline, args.warmup, args.iterations),
    }
    if args.trace_dir:
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
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
