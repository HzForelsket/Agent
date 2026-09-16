#!/usr/bin/env python
# Copyright (c) Microsoft. All rights reserved.

"""Prepare or run GPU/NPU independent-vs-simple-sharing SQL trace training updates."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from prefix_grouper_stack import NPU_CANN_VERSION, REQUIRED_STACKS
from sql_trace_replay_data import BENCHMARK_ID, file_hash, prepare_workload, write_json


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Original SQL capture directory, including events.jsonl")
    parser.add_argument("--model", type=Path, help="Local pretrained checkpoint with tokenizer files")
    parser.add_argument("--backend", choices=("gpu", "npu"))
    parser.add_argument(
        "--devices", type=int, default=4, help="Visible devices; one complete question per rank per step"
    )
    parser.add_argument(
        "--steps", type=int, default=4, help="Includes warmup; consumes steps * devices complete questions"
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--check-samples", type=int, default=4096, help="Entries per local parameter shard; 0 checks all entries"
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate real traces on CPU without importing torch or loading weights",
    )
    parser.add_argument("--report-only", action="store_true", help="Rebuild the report in an existing run directory")
    parser.add_argument(
        "--worker",
        choices=("baseline-check", "simple-check", "baseline-measure", "simple-measure"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_results(root: Path, devices: int) -> dict:
    """Gate performance on explicitly scoped numerical checks from all ranks."""
    checks = []
    for rank in range(devices):
        base = read_lines(root / "baseline-check" / f"rank-{rank}" / "checks.jsonl")
        shared = read_lines(root / "simple-check" / f"rank-{rank}" / "checks.jsonl")
        if len(base) != 1 or len(shared) != 1 or base[0]["step"] != shared[0]["step"]:
            raise ValueError("Verification step mismatch")
        identity = lambda record: (record["kind"], record.get("parameter"), record.get("slot"), record["elements"])
        if list(map(identity, base[0]["results"])) != list(map(identity, shared[0]["results"])):
            raise ValueError("Parameter layout or verification scope differs")
        checks.append({"rank": rank, **shared[0]})
    nonzero_gradient = any(
        row["kind"] == "gradient" and row["reference_l2"] > 0 for check in checks for row in check["results"]
    )
    nonzero_update = any(
        row["kind"] == "parameter_update" and row["reference_l2"] > 0 for check in checks for row in check["results"]
    )
    result = {
        "ranks": checks,
        "nonzero_gradient": nonzero_gradient,
        "nonzero_update": nonzero_update,
        "passed": all(check["passed"] for check in checks) and nonzero_gradient and nonzero_update,
    }
    write_json(root / "numerical_checks.json", result)
    return result


def report(root: Path) -> None:
    """Use slowest-rank step latency and global original response tokens in both modes."""
    settings = json.loads((root / "settings.json").read_text())
    checks = check_results(root, settings["devices"])
    manifest_hash = file_hash(root / "settings.json")
    workload_hash = file_hash(root / "workload.json")
    for rank in range(settings["devices"]):
        environments = [
            read_lines(root / stage / f"rank-{rank}" / "environment.jsonl")
            for stage in ("baseline-check", "simple-check", "baseline-measure", "simple-measure")
        ]
        if any(len(rows) != 1 for rows in environments):
            raise ValueError("Missing or repeated worker environment records")
        if any(rows[0] != environments[0][0] for rows in environments[1:]):
            raise ValueError("Worker hardware, settings or workload changed between stages")
        if (
            environments[0][0]["settings_sha256"] != manifest_hash
            or environments[0][0]["workload_sha256"] != workload_hash
        ):
            raise ValueError("Run configuration or workload changed since execution")
    results = {}
    for mode in ("baseline", "simple"):
        ranks = [
            read_lines(root / f"{mode}-measure" / f"rank-{rank}" / "steps.jsonl") for rank in range(settings["devices"])
        ]
        if any([row["step"] for row in records] != list(range(settings["steps"])) for records in ranks):
            raise ValueError("Missing, repeated or reordered steps; cannot publish a complete comparison")
        steps = []
        for index in range(settings["steps"]):
            rows = [rank[index] for rank in ranks]
            steps.append(
                {
                    "step": index,
                    "task_ids": [row["task_id"] for row in rows],
                    **{
                        key: max(row[key] for row in rows)
                        for key in (
                            "e2e_seconds",
                            "forward_seconds",
                            "backward_sync_seconds",
                            "optimizer_seconds",
                            "peak_allocated_bytes",
                        )
                    },
                    **{
                        key: sum(row[key] for row in rows)
                        for key in ("response_tokens", "independent_tokens", "simple_tokens")
                    },
                    "loss": sum(row["loss_dp_scaled"] for row in rows) / settings["devices"],
                }
            )
        measured = steps[settings["warmup"] :]
        results[mode] = {
            "steps": steps,
            "measured_steps": len(measured),
            "mean_step_seconds": statistics.mean(row["e2e_seconds"] for row in measured),
            "median_step_seconds": statistics.median(row["e2e_seconds"] for row in measured),
            "response_tokens_per_second": sum(row["response_tokens"] for row in measured)
            / sum(row["e2e_seconds"] for row in measured),
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in measured),
            **{
                f"mean_{key}": statistics.mean(row[key] for row in measured)
                for key in ("forward_seconds", "backward_sync_seconds", "optimizer_seconds")
            },
        }
    for base, shared in zip(results["baseline"]["steps"], results["simple"]["steps"], strict=True):
        if any(
            base[key] != shared[key] for key in ("task_ids", "response_tokens", "independent_tokens", "simple_tokens")
        ):
            raise ValueError("The two modes did not execute the same workload")
    speedup = (
        results["baseline"]["mean_step_seconds"] / results["simple"]["mean_step_seconds"] if checks["passed"] else None
    )
    result = {
        "benchmark_id": BENCHMARK_ID,
        "settings": settings,
        "numerical_checks_passed": checks["passed"],
        "speedup": speedup,
        "results": results,
    }
    write_json(root / "report.json", result)
    table = [
        {"mode": mode, **{key: value for key, value in metrics.items() if key != "steps"}}
        for mode, metrics in results.items()
    ]
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    rows = [
        "| 方式 | 平均完整训练步 / s | 中位数 / s | response token/s | 最大单卡 allocated / GiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, value in results.items():
        rows.append(
            f"| {mode} | {value['mean_step_seconds']:.6f} | {value['median_step_seconds']:.6f} | "
            f"{value['response_tokens_per_second']:.2f} | {value['peak_allocated_bytes'] / 2**30:.3f} |"
        )
    text = (
        f"# SQL 固定轨迹完整训练步对比（{settings['backend']}）\n\n" + "\n".join(rows) + "\n\n"
        f"数值检查通过：{checks['passed']}；加速比：{speedup if speedup is not None else 'N/A'}。"
        f"排除前 {settings['warmup']} 个更新，其余 {settings['steps'] - settings['warmup']} 个更新参与性能汇总。\n\n"
        "完整步包含设备数据搬运、共享打包、前向/loss、反向/梯度同步、裁剪、AdamW 更新、调度器及配置的 offload。"
        "计时包含同步采样开销；各阶段的最慢 rank 可能不同，阶段最大值不可直接相加。\n\n"
        "不包含模型加载、初始 old-policy log-prob 准备、SQL rollout、验证文件写入。"
        "这是固定轨迹离线回放，不是在线采样至更新的训练流水线，不评价训练收敛。\n\n"
        f"检查范围：一个含非零 advantage 的初始模型更新；回复 log-prob 全量，参数/梯度/更新"
        f"{'全量本地分片' if settings['check_samples'] == 0 else '每个本地参数分片最多 ' + str(settings['check_samples']) + ' 个固定采样位置'}。"
        "采样通过不能证明全部梯度逐元素一致；其余训练步保留原始 loss 和时延。\n"
    )
    (root / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = arguments()
    root = args.output.resolve()
    if args.report_only:
        report(root)
        return
    if args.worker:
        settings = json.loads((root / "settings.json").read_text())
        workload = json.loads((root / "workload.json").read_text())
        mode, phase = args.worker.split("-")
        # Register torch.npu before VERL freezes module-level device availability flags.
        if settings["backend"] == "npu":
            import torch_npu  # noqa: F401
        from agentlightning.verl.sql_trace_replay import run

        run(settings, workload, root / args.worker, mode=mode, phase=phase)
        return
    if not args.input or not args.backend or (not args.prepare_only and not args.model):
        raise ValueError("--input and --backend are required; execution also requires --model")
    if (
        args.devices < 1
        or not 0 <= args.warmup < args.steps
        or args.check_samples < 0
        or not 0 < args.clip_ratio < 1
        or args.lr <= 0
        or args.atol < 0
        or args.rtol < 0
        or not all(math.isfinite(value) for value in (args.lr, args.clip_ratio, args.atol, args.rtol))
    ):
        raise ValueError("Invalid device, step, warmup, learning-rate or numerical-check configuration")
    root.mkdir(parents=True, exist_ok=False)
    print(f"SQL replay output: {root}", flush=True)
    settings = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"worker", "report_only", "prepare_only", "output"}
    }
    settings["benchmark_id"] = BENCHMARK_ID
    settings["command"] = sys.argv
    settings["world_size"] = args.devices
    write_json(root / "settings.json", settings)
    prepared = False
    active = "prepare"
    try:
        workload = prepare_workload(args.input.resolve(), steps=args.steps, world_size=args.devices)
        write_json(root / "workload.json", workload)
        prepared = True
        if args.prepare_only:
            print(f"Validated {len(workload['groups'])} complete SQL groups; no accelerator initialized", flush=True)
            return
        active = "environment"
        expected = REQUIRED_STACKS[args.backend]
        versions = {name: importlib.metadata.version(name) for name in expected}
        mismatches = {
            name: (versions[name], version)
            for name, version in expected.items()
            if versions[name].split("+")[0] != version
        }
        if mismatches:
            raise RuntimeError(f"Pinned stack mismatch: {mismatches}")
        if not args.model.is_dir() or not (args.model / "config.json").exists():
            raise ValueError("--model must be a complete local pretrained model directory")
        config = json.loads((args.model / "config.json").read_text())
        for group in workload["groups"]:
            for slot in group["slots"]:
                for row in slot["rows"]:
                    if row and (
                        max(row["tokens"]) >= config["vocab_size"]
                        or len(row["tokens"]) > config["max_position_embeddings"]
                    ):
                        raise ValueError("Capture tokens or sequence length exceed the local model configuration")
        print("Fingerprinting local checkpoint files before launching workers", flush=True)
        model_files = sorted(path for path in args.model.iterdir() if path.is_file())
        settings["model_sha256"] = {path.name: file_hash(path) for path in model_files}
        repo = Path(__file__).resolve().parents[1]
        source_paths = [
            Path(__file__),
            Path(__file__).with_name("sql_trace_replay_data.py"),
            repo / "agentlightning/verl/sql_trace_replay.py",
            repo / "agentlightning/verl/sql_trace_replay_shared.py",
            repo / "agentlightning/verl/prefix_grouper.py",
        ]
        settings["source_sha256"] = {str(path.relative_to(repo)): file_hash(path) for path in source_paths}
        settings["versions"] = versions
        settings["required_cann"] = NPU_CANN_VERSION if args.backend == "npu" else None
        settings["workload_sha256"] = file_hash(root / "workload.json")
        settings["environment"] = {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA_VISIBLE", "ASCEND_RT_VISIBLE", "HCCL_", "NCCL_"))
        }
        write_json(root / "settings.json", settings)
        env = {
            **os.environ,
            "VERL_PLATFORM": "huawei" if args.backend == "npu" else "nvidia",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        for stage in ("baseline-check", "simple-check", "baseline-measure", "simple-measure"):
            active = stage
            (root / stage).mkdir()
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={args.devices}",
                str(Path(__file__).resolve()),
                "--worker",
                stage,
                "--output",
                str(root),
            ]
            print(f"Starting {stage}: log={root / (stage + '.log')}", flush=True)
            with (root / f"{stage}.log").open("w") as log:
                subprocess.run(
                    command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            if stage == "simple-check" and not check_results(root, args.devices)["passed"]:
                raise RuntimeError("Numerical checks failed or update was zero; performance stages were not run")
        active = "report"
        report(root)
        write_json(root / "completion.json", {"finished_at": time.time(), "stages": 4})
        print(f"Comparison report: {root / 'report.md'}", flush=True)
    except BaseException as error:
        if prepared:
            write_json(root / "failure.json", {"stage": active, "error": str(error), "partial": True})
            print(f"Validated workload and any completed results retained: {root}", file=sys.stderr)
        else:
            shutil.rmtree(root)
            print("No usable workload; removed failed output directory", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
