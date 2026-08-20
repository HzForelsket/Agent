# Copyright (c) Microsoft. All rights reserved.

"""Compare standard Hugging Face attention with Agent Lightning's PrefixGrouper path.

The benchmark uses complete model architectures. By default, it downloads only each
model's configuration and initializes random weights because dense-model runtime and
memory usage do not depend on the learned weight values. Pass ``--weights pretrained``
to load pretrained weights instead.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import transformers
from prefix_grouper import PrefixGrouper
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from agentlightning.verl.prefix_grouper import (
    PREFIX_GROUPER_ATTENTION,
    register_prefix_grouper_attention,
)


@dataclass(frozen=True)
class BenchmarkCase:
    """One prompt-length and shared-prefix group-size combination."""

    prompt_length: int
    group_size: int


TensorBatch = Dict[str, Any]
BenchmarkFunction = Callable[[], torch.Tensor]


def parse_case(value: str) -> BenchmarkCase:
    """Parse a benchmark case written as PROMPT_LENGTH:GROUP_SIZE."""
    try:
        prompt_length_text, group_size_text = value.split(":", maxsplit=1)
        prompt_length = int(prompt_length_text)
        group_size = int(group_size_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Cases must use PROMPT_LENGTH:GROUP_SIZE, for example 1024:4.") from exc
    if prompt_length <= 0 or group_size <= 0:
        raise argparse.ArgumentTypeError("Prompt length and group size must both be positive.")
    return BenchmarkCase(prompt_length=prompt_length, group_size=group_size)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PrefixGrouper against standard SDPA and output a comparison report."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Hugging Face model IDs or local model/config directories.",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        dest="cases",
        help="Benchmark case as PROMPT_LENGTH:GROUP_SIZE. Repeat for multiple cases.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--response-length", type=int, default=64)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("forward", "forward-backward"),
        default=("forward", "forward-backward"),
        help="Benchmark modes to run.",
    )
    parser.add_argument(
        "--weights",
        choices=("random", "pretrained"),
        default="random",
        help="Use full-size random weights (config download only) or pretrained weights.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2, help="Forward warmup iterations.")
    parser.add_argument("--repeats", type=int, default=5, help="Forward timed iterations.")
    parser.add_argument("--backward-warmup", type=int, default=1)
    parser.add_argument("--backward-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--atol", type=float, default=0.1, help="Absolute tolerance for BF16 correctness checks.")
    parser.add_argument("--rtol", type=float, default=0.05, help="Relative tolerance for correctness checks.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("prefix_grouper_benchmark.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("prefix_grouper_benchmark.md"))
    args = parser.parse_args(argv)
    if not args.cases:
        args.cases = [BenchmarkCase(prompt_length=1024, group_size=4), BenchmarkCase(1536, 8)]
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.response_length <= 0:
        parser.error("--batch-size and --response-length must be positive.")
    for name in ("warmup", "backward_warmup"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative.")
    for name in ("repeats", "backward_repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    for case in args.cases:
        if case.group_size > args.batch_size or args.batch_size % case.group_size:
            parser.error(
                f"Case {case.prompt_length}:{case.group_size} is invalid: group size must divide "
                f"batch size {args.batch_size}."
            )
    if not args.device.startswith("cuda"):
        parser.error("This benchmark currently requires a CUDA device.")


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_label(model_ref: str) -> str:
    path = Path(model_ref)
    return path.name if path.name else model_ref


def load_model(
    model_ref: str,
    *,
    weights: str,
    dtype: torch.dtype,
    device: torch.device,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Tuple[PreTrainedModel, PretrainedConfig]:
    config = AutoConfig.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    config.use_cache = False
    common_kwargs = {
        "attn_implementation": PREFIX_GROUPER_ATTENTION,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if weights == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            config=config,
            local_files_only=local_files_only,
            **common_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, **common_kwargs)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, config


def validate_context_length(config: PretrainedConfig, case: BenchmarkCase, response_length: int) -> None:
    max_positions = getattr(config, "max_position_embeddings", None)
    if isinstance(max_positions, int) and case.prompt_length + response_length > max_positions:
        raise ValueError(
            f"Prompt {case.prompt_length} + response {response_length} exceeds the model's "
            f"max_position_embeddings={max_positions}."
        )


def make_batch(
    config: PretrainedConfig,
    *,
    batch_size: int,
    prompt_length: int,
    response_length: int,
    group_size: int,
    device: torch.device,
    seed: int,
) -> TensorBatch:
    groups = batch_size // group_size
    generator = torch.Generator(device=device).manual_seed(seed + prompt_length + group_size)
    vocab_size = int(config.vocab_size)
    prompt_representatives = torch.randint(
        0,
        vocab_size,
        (groups, prompt_length),
        device=device,
        generator=generator,
    )
    prompt_ids = prompt_representatives.repeat_interleave(group_size, dim=0)
    response_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, response_length),
        device=device,
        generator=generator,
    )
    input_ids = torch.cat((prompt_ids, response_ids), dim=1)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(prompt_length + response_length, device=device).expand(batch_size, -1)

    prefix_mask = torch.ones((groups, prompt_length), dtype=torch.long, device=device)
    response_mask = torch.ones((batch_size, response_length), dtype=torch.long, device=device)
    grouper = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=prefix_mask,
        suffix_mask=response_mask,
        group_sizes=[group_size] * groups,
        device=device,
        padding_mode="right",
    )
    grouped_input_ids = grouper.concat_input(
        prompt_representatives,
        prefix_mask,
        response_ids,
        response_mask,
    )
    prefix_position_ids = torch.arange(prompt_length, device=device).expand(groups, -1)
    response_position_ids = torch.arange(
        prompt_length,
        prompt_length + response_length,
        device=device,
    ).expand(batch_size, -1)
    grouped_position_ids = grouper.concat_input(
        prefix_position_ids,
        prefix_mask,
        response_position_ids,
        response_mask,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_ids": response_ids,
        "grouper": grouper,
        "grouped_input_ids": grouped_input_ids,
        "grouped_position_ids": grouped_position_ids,
    }


def response_logits(model: PreTrainedModel, batch: TensorBatch, *, grouped: bool) -> torch.Tensor:
    response_length = batch["response_ids"].shape[1]
    prompt_length = batch["input_ids"].shape[1] - response_length
    if grouped:
        output = model(
            input_ids=batch["grouped_input_ids"],
            attention_mask=batch["grouper"].padding_mask,
            position_ids=batch["grouped_position_ids"],
            use_cache=False,
            prefix_grouper=batch["grouper"],
        )
        _, _, suffix_logits, _ = batch["grouper"].split_output(output.logits, include_prefix_last=1)
        return suffix_logits[:, :-1]
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
    )
    return output.logits[:, prompt_length - 1 : prompt_length + response_length - 1]  # noqa: E203


def cuda_time_ms(function: BenchmarkFunction, *, warmup: int, repeats: int) -> Tuple[float, List[float]]:
    for _ in range(warmup):
        value = function()
        del value
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = function()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
        del value
    return statistics.median(samples), samples


def peak_memory_mib(function: BenchmarkFunction, device: torch.device) -> Tuple[float, float]:
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    value = function()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)
    del value
    return peak / (1024**2), (peak - baseline) / (1024**2)


def comparison_result(
    *,
    mode: str,
    case: BenchmarkCase,
    response_length: int,
    batch_size: int,
    baseline_ms: float,
    grouped_ms: float,
    baseline_samples: List[float],
    grouped_samples: List[float],
    baseline_peak: float,
    grouped_peak: float,
    baseline_incremental: float,
    grouped_incremental: float,
) -> Dict[str, Any]:
    return {
        "mode": mode,
        "prompt_length": case.prompt_length,
        "response_length": response_length,
        "batch_size": batch_size,
        "group_size": case.group_size,
        "baseline_ms": baseline_ms,
        "prefix_grouper_ms": grouped_ms,
        "speedup": baseline_ms / grouped_ms,
        "baseline_peak_mib": baseline_peak,
        "prefix_grouper_peak_mib": grouped_peak,
        "peak_memory_reduction": 1.0 - grouped_peak / baseline_peak,
        "baseline_incremental_mib": baseline_incremental,
        "prefix_grouper_incremental_mib": grouped_incremental,
        "baseline_samples_ms": baseline_samples,
        "prefix_grouper_samples_ms": grouped_samples,
    }


def benchmark_forward(
    model: PreTrainedModel,
    batch: TensorBatch,
    case: BenchmarkCase,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    def baseline_function() -> torch.Tensor:
        return response_logits(model, batch, grouped=False)

    def grouped_function() -> torch.Tensor:
        return response_logits(model, batch, grouped=True)

    with torch.inference_mode():
        baseline_ms, baseline_samples = cuda_time_ms(
            baseline_function,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        grouped_ms, grouped_samples = cuda_time_ms(
            grouped_function,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        baseline_peak, baseline_incremental = peak_memory_mib(baseline_function, device)
        grouped_peak, grouped_incremental = peak_memory_mib(grouped_function, device)
    return comparison_result(
        mode="forward",
        case=case,
        response_length=args.response_length,
        batch_size=args.batch_size,
        baseline_ms=baseline_ms,
        grouped_ms=grouped_ms,
        baseline_samples=baseline_samples,
        grouped_samples=grouped_samples,
        baseline_peak=baseline_peak,
        grouped_peak=grouped_peak,
        baseline_incremental=baseline_incremental,
        grouped_incremental=grouped_incremental,
    )


def benchmark_forward_backward(
    model: PreTrainedModel,
    batch: TensorBatch,
    case: BenchmarkCase,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    def step(grouped: bool) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        logits = response_logits(model, batch, grouped=grouped)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["response_ids"].reshape(-1),
        )
        loss.backward()
        return loss

    def baseline_function() -> torch.Tensor:
        return step(False)

    def grouped_function() -> torch.Tensor:
        return step(True)

    baseline_ms, baseline_samples = cuda_time_ms(
        baseline_function,
        warmup=args.backward_warmup,
        repeats=args.backward_repeats,
    )
    model.zero_grad(set_to_none=True)
    grouped_ms, grouped_samples = cuda_time_ms(
        grouped_function,
        warmup=args.backward_warmup,
        repeats=args.backward_repeats,
    )
    model.zero_grad(set_to_none=True)
    baseline_peak, baseline_incremental = peak_memory_mib(baseline_function, device)
    model.zero_grad(set_to_none=True)
    grouped_peak, grouped_incremental = peak_memory_mib(grouped_function, device)
    model.zero_grad(set_to_none=True)
    return comparison_result(
        mode="forward-backward",
        case=case,
        response_length=args.response_length,
        batch_size=args.batch_size,
        baseline_ms=baseline_ms,
        grouped_ms=grouped_ms,
        baseline_samples=baseline_samples,
        grouped_samples=grouped_samples,
        baseline_peak=baseline_peak,
        grouped_peak=grouped_peak,
        baseline_incremental=baseline_incremental,
        grouped_incremental=grouped_incremental,
    )


def check_correctness(
    model: PreTrainedModel,
    config: PretrainedConfig,
    *,
    device: torch.device,
    seed: int,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    batch = make_batch(
        config,
        batch_size=4,
        prompt_length=64,
        response_length=16,
        group_size=4,
        device=device,
        seed=seed,
    )
    with torch.inference_mode():
        baseline = response_logits(model, batch, grouped=False).float()
        grouped = response_logits(model, batch, grouped=True).float()
    difference = (baseline - grouped).abs()
    result = {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "atol": atol,
        "rtol": rtol,
        "passed": torch.allclose(baseline, grouped, atol=atol, rtol=rtol),
    }
    del batch, baseline, grouped, difference
    return result


def markdown_report(results: Dict[str, Any]) -> str:
    lines = [
        "# PrefixGrouper benchmark",
        "",
        f"- Device: {results['device']}",
        f"- Dtype: {results['dtype']}",
        f"- Weights: {results['weights']}",
        f"- Batch / response length: {results['batch_size']} / {results['response_length']}",
        "",
        "## Correctness",
        "",
        "| Model | Architecture | Parameters | Max abs error | Mean abs error | Passed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in results["models"]:
        correctness = model["correctness"]
        lines.append(
            f"| {model['label']} | {model['model_type']} | {model['parameters']:,} | "
            f"{correctness['max_abs_error']:.6f} | {correctness['mean_abs_error']:.6f} | "
            f"{'yes' if correctness['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Performance comparison",
            "",
            "| Model | Mode | Prompt | Group | Baseline ms | PrefixGrouper ms | Speedup | "
            "Baseline peak MiB | PrefixGrouper peak MiB | Memory saved |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in results["models"]:
        for benchmark in model["benchmarks"]:
            lines.append(
                f"| {model['label']} | {benchmark['mode']} | {benchmark['prompt_length']} | "
                f"{benchmark['group_size']} | {benchmark['baseline_ms']:.3f} | "
                f"{benchmark['prefix_grouper_ms']:.3f} | {benchmark['speedup']:.3f}x | "
                f"{benchmark['baseline_peak_mib']:.1f} | {benchmark['prefix_grouper_peak_mib']:.1f} | "
                f"{benchmark['peak_memory_reduction']:.1%} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_results(results: Dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(results), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() returned False.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    register_prefix_grouper_attention()

    results: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "prefix_grouper_version": package_version("prefix_grouper"),
        "dtype": args.dtype,
        "weights": args.weights,
        "batch_size": args.batch_size,
        "response_length": args.response_length,
        "models": [],
    }

    for model_index, model_ref in enumerate(args.models):
        print(f"[{model_index + 1}/{len(args.models)}] Loading {model_ref}...", flush=True)
        started = time.monotonic()
        model, config = load_model(
            model_ref,
            weights=args.weights,
            dtype=dtype,
            device=device,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        model_result: Dict[str, Any] = {
            "model": model_ref,
            "label": model_label(model_ref),
            "model_type": config.model_type,
            "parameters": parameter_count(model),
            "correctness": check_correctness(
                model,
                config,
                device=device,
                seed=args.seed,
                atol=args.atol,
                rtol=args.rtol,
            ),
            "benchmarks": [],
        }
        if not model_result["correctness"]["passed"]:
            raise RuntimeError(
                f"Correctness check failed for {model_ref}: {model_result['correctness']}. "
                "Performance results would not be trustworthy."
            )
        for case in args.cases:
            validate_context_length(config, case, args.response_length)
            batch = make_batch(
                config,
                batch_size=args.batch_size,
                prompt_length=case.prompt_length,
                response_length=args.response_length,
                group_size=case.group_size,
                device=device,
                seed=args.seed,
            )
            print(
                f"  prompt={case.prompt_length}, group={case.group_size}, modes={','.join(args.modes)}",
                flush=True,
            )
            if "forward" in args.modes:
                model_result["benchmarks"].append(benchmark_forward(model, batch, case, args, device))
            if "forward-backward" in args.modes:
                model_result["benchmarks"].append(benchmark_forward_backward(model, batch, case, args, device))
            del batch
            gc.collect()
            torch.cuda.empty_cache()
        model_result["elapsed_seconds"] = time.monotonic() - started
        results["models"].append(model_result)
        write_results(results, args.output_json, args.output_markdown)
        del model, config
        gc.collect()
        torch.cuda.empty_cache()

    report = markdown_report(results)
    print("\n" + report)
    print(f"JSON: {args.output_json.resolve()}")
    print(f"Markdown: {args.output_markdown.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
