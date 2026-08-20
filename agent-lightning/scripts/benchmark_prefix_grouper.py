# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""Benchmark standard attention against PrefixGrouper on VERL 0.9.

By default only model configurations are downloaded and full architectures are
initialized with random weights.  Runtime and memory are therefore representative
for dense kernels without downloading checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from packaging.version import Version
from prefix_grouper import PrefixGrouper
from transformers import AutoConfig, AutoModelForCausalLM
from verl.trainer.ppo.prefix_grouper_utils import build_position_ids_for_prefix_grouper

from agentlightning.verl.accelerator import AcceleratorRuntime, select_accelerator
from agentlightning.verl.prefix_grouper import apply_prefix_grouper_patch
from prefix_grouper_stack import NPU_CANN_VERSION, REQUIRED_STACKS

DEFAULT_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"]
DIST_NAMES = {
    "torch_npu": "torch-npu",
    "triton_ascend": "triton-ascend",
    "vllm_ascend": "vllm-ascend",
}


@dataclass(frozen=True)
class Case:
    prompt_length: int
    group_size: int


def parse_case(text: str) -> Case:
    try:
        prompt_length, group_size = (int(value) for value in text.split(":", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case 格式必须是 PROMPT长度:共享组大小，例如 512:4") from exc
    if prompt_length <= 0 or group_size <= 1:
        raise argparse.ArgumentTypeError("prompt 长度必须为正数，组大小必须大于 1")
    return Case(prompt_length, group_size)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="输出 PrefixGrouper 添加前后的速度、显存和正确性对比")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--case", action="append", type=parse_case, dest="cases")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--response-length", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="auto", help="auto、gpu/cuda、npu，或带编号的设备（如 npu:1）")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weights", choices=("random", "pretrained"), default="random")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output-json", type=Path, default=Path("prefix_grouper_results.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("prefix_grouper_results.md"))
    args = parser.parse_args(argv)
    args.cases = args.cases or [Case(512, 4), Case(1024, 8)]
    if args.batch_size <= 0 or args.response_length <= 0 or args.repeats <= 0 or args.warmup < 0:
        parser.error("batch、response、repeats 必须为正数，warmup 不能为负数")
    for case in args.cases:
        if case.group_size > args.batch_size or args.batch_size % case.group_size:
            parser.error(f"组大小 {case.group_size} 必须整除 batch-size={args.batch_size}")
    return args


def _distribution_version(name: str) -> str:
    try:
        return version(DIST_NAMES.get(name, name))
    except PackageNotFoundError:
        return "未安装"


def installed_stack(backend: str) -> dict[str, str]:
    stack = {
        "torch": torch.__version__.split("+")[0],
        "vllm": _distribution_version("vllm"),
        "verl": _distribution_version("verl"),
        "transformers": _distribution_version("transformers"),
        "prefix_grouper": _distribution_version("prefix_grouper"),
    }
    if backend == "npu":
        stack.update(
            {
                "torch_npu": _distribution_version("torch_npu"),
                "triton_ascend": _distribution_version("triton_ascend"),
                "vllm_ascend": _distribution_version("vllm_ascend"),
            }
        )
    return stack


def check_stack(stack: dict[str, str], backend: str) -> None:
    required = REQUIRED_STACKS[backend]
    mismatches = [
        f"{name}={stack[name]}（需要 {wanted}）"
        for name, wanted in required.items()
        if stack.get(name) == "未安装" or Version(stack[name]).public != Version(wanted).public
    ]
    if mismatches:
        raise RuntimeError("测试环境版本不符合要求：" + "，".join(mismatches))


def load_model(model_ref: str, args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    config = AutoConfig.from_pretrained(
        model_ref,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    config.use_cache = False
    kwargs = {
        "config": config,
        "attn_implementation": "sdpa",
        "dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.weights == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            local_files_only=args.local_files_only,
            **kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_config(**kwargs)
    model.to(device=device, dtype=dtype).eval()
    return model, config


def make_batch(
    config: Any,
    *,
    batch_size: int,
    prompt_length: int,
    response_length: int,
    group_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed + prompt_length + group_size)
    group_count = batch_size // group_size
    representatives = torch.randint(
        1, int(config.vocab_size), (group_count, prompt_length), generator=generator, device=device
    )
    prompts = representatives.repeat_interleave(group_size, dim=0)
    responses = torch.randint(
        1, int(config.vocab_size), (batch_size, response_length), generator=generator, device=device
    )
    prefix_mask = torch.ones_like(representatives, dtype=torch.bool)
    response_mask = torch.ones_like(responses, dtype=torch.bool)
    grouper = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=prefix_mask,
        suffix_mask=response_mask,
        group_sizes=[group_size] * group_count,
        padding_mode="right",
        device=device,
    )
    grouped_ids = grouper.concat_input(representatives, prefix_mask, responses, response_mask)
    return {
        "input_ids": torch.cat((prompts, responses), dim=-1),
        "attention_mask": torch.ones((batch_size, prompt_length + response_length), dtype=torch.bool, device=device),
        "position_ids": torch.arange(prompt_length + response_length, device=device).expand(batch_size, -1),
        "responses": responses,
        "grouper": grouper,
        "grouped_ids": grouped_ids,
        "grouped_position_ids": build_position_ids_for_prefix_grouper(grouper),
    }


def response_logits(model: torch.nn.Module, batch: dict[str, Any], grouped: bool) -> torch.Tensor:
    response_length = batch["responses"].shape[1]
    prompt_length = batch["input_ids"].shape[1] - response_length
    if grouped:
        output = model(
            input_ids=batch["grouped_ids"],
            attention_mask=batch["grouper"].padding_mask,
            position_ids=batch["grouped_position_ids"],
            prefix_grouper=batch["grouper"],
            use_cache=False,
        )
        _, _, suffix_logits, _ = batch["grouper"].split_output(output.logits, include_prefix_last=1)
        return suffix_logits[:, :-1]
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
    )
    return output.logits[:, prompt_length - 1 : prompt_length + response_length - 1]


def timed(
    function: Callable[[], torch.Tensor],
    warmup: int,
    repeats: int,
    accelerator: AcceleratorRuntime,
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        function()
    accelerator.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = accelerator.event(), accelerator.event()
        start.record()
        function()
        end.record()
        accelerator.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), samples


def peak_memory(function: Callable[[], torch.Tensor], accelerator: AcceleratorRuntime) -> tuple[float, float]:
    gc.collect()
    accelerator.empty_cache()
    baseline = accelerator.memory_allocated()
    accelerator.reset_peak_memory_stats()
    function()
    accelerator.synchronize()
    peak = accelerator.max_memory_allocated()
    return peak / 2**20, (peak - baseline) / 2**20


def benchmark_mode(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    mode: str,
    warmup: int,
    repeats: int,
    accelerator: AcceleratorRuntime,
) -> dict[str, Any]:
    def step(grouped: bool) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        logits = response_logits(model, batch, grouped)
        if mode == "forward-backward":
            loss = F.cross_entropy(logits.flatten(0, 1), batch["responses"].flatten())
            loss.backward()
            return loss
        return logits

    context = torch.inference_mode() if mode == "forward" else torch.enable_grad()
    with context:
        baseline_ms, baseline_samples = timed(lambda: step(False), warmup, repeats, accelerator)
        grouped_ms, grouped_samples = timed(lambda: step(True), warmup, repeats, accelerator)
        baseline_peak, baseline_incremental = peak_memory(lambda: step(False), accelerator)
        grouped_peak, grouped_incremental = peak_memory(lambda: step(True), accelerator)
    tokens = batch["responses"].numel()
    return {
        "mode": mode,
        "baseline_ms": baseline_ms,
        "prefix_grouper_ms": grouped_ms,
        "speedup": baseline_ms / grouped_ms,
        "baseline_response_tokens_per_second": tokens * 1000 / baseline_ms,
        "prefix_grouper_response_tokens_per_second": tokens * 1000 / grouped_ms,
        "baseline_peak_mib": baseline_peak,
        "prefix_grouper_peak_mib": grouped_peak,
        "peak_memory_saved_ratio": 1 - grouped_peak / baseline_peak,
        "baseline_incremental_mib": baseline_incremental,
        "prefix_grouper_incremental_mib": grouped_incremental,
        "baseline_samples_ms": baseline_samples,
        "prefix_grouper_samples_ms": grouped_samples,
    }


def correctness(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    with torch.inference_mode():
        baseline_logits = response_logits(model, batch, False)
        grouped_logits = response_logits(model, batch, True)
        response_ids = batch["responses"].unsqueeze(-1)
        baseline_log_probs = baseline_logits.float().log_softmax(-1).gather(-1, response_ids).squeeze(-1)
        grouped_log_probs = grouped_logits.float().log_softmax(-1).gather(-1, response_ids).squeeze(-1)

    logit_difference = (baseline_logits.float() - grouped_logits.float()).abs()
    log_prob_difference = (baseline_log_probs - grouped_log_probs).abs()
    low_precision = baseline_logits.dtype in {torch.bfloat16, torch.float16}
    max_tolerance = 0.2 if low_precision else 0.03
    mean_tolerance = 0.05 if low_precision else 0.01
    return {
        "passed": bool(log_prob_difference.max() <= max_tolerance and log_prob_difference.mean() <= mean_tolerance),
        "max_response_log_prob_error": log_prob_difference.max().item(),
        "mean_response_log_prob_error": log_prob_difference.mean().item(),
        "max_logit_error": logit_difference.max().item(),
        "mean_logit_error": logit_difference.mean().item(),
        "top1_agreement": (baseline_logits.argmax(dim=-1) == grouped_logits.argmax(dim=-1)).float().mean().item(),
        "max_tolerance": max_tolerance,
        "mean_tolerance": mean_tolerance,
    }


def markdown(results: dict[str, Any]) -> str:
    lines = [
        "# PrefixGrouper 性能对比",
        "",
        f"- 加速器：{results['accelerator']['backend']} / {results['accelerator']['name']}",
        f"- 软件栈：torch {results['stack']['torch']} / vLLM {results['stack']['vllm']} / VERL {results['stack']['verl']}",
        f"- 精度：{results['dtype']}，权重：{results['weights']}",
        "",
        "| 模型 | 模式 | Prompt | 共享组 | 添加前 ms | 添加后 ms | 加速比 | 添加前 tok/s | 添加后 tok/s | 峰值显存节省 | 正确性 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in results["models"]:
        for case in model["cases"]:
            for item in case["measurements"]:
                lines.append(
                    f"| {model['label']} | {item['mode']} | {case['prompt_length']} | {case['group_size']} | "
                    f"{item['baseline_ms']:.3f} | {item['prefix_grouper_ms']:.3f} | {item['speedup']:.2f}x | "
                    f"{item['baseline_response_tokens_per_second']:.0f} | "
                    f"{item['prefix_grouper_response_tokens_per_second']:.0f} | "
                    f"{item['peak_memory_saved_ratio']:.1%} | {'通过' if case['correctness']['passed'] else '失败'} |"
                )
    return "\n".join(lines) + "\n"


def write_report(results: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(results), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    accelerator = select_accelerator(args.device)
    stack = installed_stack(accelerator.backend)
    check_stack(stack, accelerator.backend)
    device = accelerator.device
    dtype = getattr(torch, args.dtype)
    accelerator.set_device()
    torch.manual_seed(args.seed)
    accelerator.manual_seed_all(args.seed)
    if accelerator.backend == "gpu":
        torch.backends.cuda.matmul.allow_tf32 = True
    apply_prefix_grouper_patch()

    results: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stack": stack,
        "accelerator": {
            "backend": accelerator.display_backend,
            "device": str(device),
            "name": accelerator.device_name(),
        },
        "required_cann": NPU_CANN_VERSION if accelerator.backend == "npu" else None,
        "dtype": args.dtype,
        "weights": args.weights,
        "batch_size": args.batch_size,
        "response_length": args.response_length,
        "models": [],
    }
    for index, model_ref in enumerate(args.models, 1):
        print(f"[{index}/{len(args.models)}] {model_ref}", flush=True)
        started = time.monotonic()
        model, config = load_model(model_ref, args, device, dtype)
        model_result = {
            "model": model_ref,
            "label": Path(model_ref).name,
            "model_type": config.model_type,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "cases": [],
        }
        for case in args.cases:
            max_positions = getattr(config, "max_position_embeddings", None)
            if max_positions and case.prompt_length + args.response_length > max_positions:
                raise ValueError(f"{model_ref} 的上下文长度不足以运行 {case}")
            batch = make_batch(
                config,
                batch_size=args.batch_size,
                prompt_length=case.prompt_length,
                response_length=args.response_length,
                group_size=case.group_size,
                device=device,
                seed=args.seed,
            )
            check = correctness(model, batch)
            if not check["passed"]:
                raise RuntimeError(f"{model_ref} {case} 正确性检查失败：{check}")
            modes = ["forward", "forward-backward"] if args.backward else ["forward"]
            measurements = [
                benchmark_mode(
                    model,
                    batch,
                    mode=mode,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    accelerator=accelerator,
                )
                for mode in modes
            ]
            model_result["cases"].append({**asdict(case), "correctness": check, "measurements": measurements})
            del batch
            gc.collect()
            accelerator.empty_cache()
        model_result["elapsed_seconds"] = time.monotonic() - started
        results["models"].append(model_result)
        write_report(results, args.output_json, args.output_markdown)
        del model, config
        gc.collect()
        accelerator.empty_cache()

    print("\n" + markdown(results))
    print(f"JSON: {args.output_json.resolve()}")
    print(f"Markdown: {args.output_markdown.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
