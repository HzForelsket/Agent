# Copyright (c) Microsoft. All rights reserved.

"""Fixed SQL trace GRPO updates through Agent Lightning's VERL FSDP engine.

The baseline never imports PrefixGrouper. Device selection happens before this
module is imported by the benchmark launcher. No rollout service is started.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from tensordict import TensorDict
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.fsdp import FSDPEngineWithLMHead

from .accelerator import AcceleratorRuntime


def persist(path: Path, value: dict) -> None:
    """Flush rank-local results after each independently completed step."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def make_batch(slot: dict | None, pad: int, denominator: int, world: int) -> TensorDict:
    """Preserve prompt/response boundaries; absent calls are zero-loss synchronization padding."""
    rows = slot["rows"] if slot else [None] * 4
    ids, masks, advantages, positions, temperatures = [], [], [], [], []
    for row in rows:
        tokens = row["tokens"] if row else [pad]
        ids.append(torch.tensor(tokens, dtype=torch.long))
        positions.append(torch.arange(len(tokens)))
        mask = torch.zeros(len(tokens))
        if row:
            mask[row["prompt_length"] - 1 : len(tokens) - 1] = 1
        masks.append(mask)
        advantages.append(torch.full((len(tokens),), row["advantage"] if row else 0.0))
        temperatures.append(row["temperature"] if row else 1.0)

    def nested(values):
        return torch.nested.as_nested_tensor(values, layout=torch.jagged)

    data = TensorDict(
        {
            "input_ids": nested(ids),
            "position_ids": nested(positions),
            "loss_mask": nested(masks),
            "advantages": nested(advantages),
            "temperature": torch.tensor(temperatures),
            "replay_prefix": torch.full((4,), slot["common_prefix"] if slot else 0, dtype=torch.long),
            "normalization": torch.full((4,), world / denominator),
        },
        batch_size=[4],
    )
    tu.assign_non_tensor(
        data,
        use_remove_padding=False,
        use_fused_kernels=False,
        pad_token_id=pad,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
    )
    return data


def grpo_loss(*, model_output, data, dp_group):
    """Clipped PPO surrogate using one fixed GRPO advantage per original trajectory."""
    del dp_group
    current = model_output["log_probs"].values().float()
    old = data["old_log_probs"].values().float()
    mask = data["loss_mask"].values().bool()
    advantage = data["advantages"].values()
    # Select before exp: zero-loss prompt/padding positions cannot create overflow or NaN gradients.
    ratio = (current[mask] - old[mask]).exp()
    clip = data["clip_ratio"][0]
    loss = -torch.minimum(ratio * advantage[mask], ratio.clamp(1 - clip, 1 + clip) * advantage[mask]).sum()
    loss = loss * data["normalization"][0]
    return loss, {}


def parameter_sample(parameter: torch.Tensor, count: int) -> torch.Tensor:
    """Take deterministic local-shard entries; count=0 retains every entry."""
    values = parameter.detach().reshape(-1)
    if count and values.numel() > count:
        indices = torch.arange(count, device=values.device, dtype=torch.int64) * values.numel() // count
        values = values.index_select(0, indices)
    return values.float().cpu()


def compare_tensor(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict:
    """Report elementwise tolerance failures, including all nonfinite values."""
    if actual.shape != expected.shape:
        raise ValueError(f"Numerical evidence shape mismatch: {actual.shape} != {expected.shape}")
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    error = (actual - expected).abs()
    failed = ~finite | (error > atol + rtol * expected.abs())
    return {
        "elements": actual.numel(),
        "failed": int(failed.sum()),
        "max_abs": float(error[finite].max()) if finite.any() else 0.0,
        "reference_l2": float(expected.double().norm()),
    }


def verify_evidence(root: Path, reference: Path | None, name: str, value: torch.Tensor, settings: dict) -> dict:
    """Write baseline evidence or compare the corresponding sharing tensor outside timing."""
    if reference is None:
        torch.save(value, root / f"{name}.pt")
        return {"elements": value.numel(), "failed": 0, "max_abs": 0.0, "reference_l2": float(value.double().norm())}
    expected = torch.load(reference / f"{name}.pt", map_location="cpu", weights_only=True)
    atol, rtol = (0.0, 0.0) if name.startswith(("initial-", "old-logprobs-")) else (settings["atol"], settings["rtol"])
    if name.startswith("update-"):
        atol = min(atol, settings["lr"] * 1e-3)
    return compare_tensor(value, expected, atol, rtol)


def run(settings: dict, workload: dict, output: Path, *, mode: str, phase: str) -> None:
    """Run identical complete update schedules on GPU or NPU under torchrun."""
    backend = settings["backend"]
    local_rank, rank, world = (int(os.environ[key]) for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
    if backend == "npu":
        import torch_npu  # noqa: F401

        device_module, device_type, collective = torch.npu, "npu", "hccl"
    else:
        device_module, device_type, collective = torch.cuda, "cuda", "nccl"
    runtime = AcceleratorRuntime(backend, torch.device(f"{device_type}:{local_rank}"), device_module)
    runtime.set_device()
    dist.init_process_group(collective)
    torch.manual_seed(settings["seed"])
    runtime.manual_seed_all(settings["seed"])
    rank_root = output / f"rank-{rank}"
    rank_root.mkdir()
    persist(
        rank_root / "environment.jsonl",
        {
            "device": runtime.device_name(),
            "rank": rank,
            "world": world,
            "settings_sha256": hashlib.sha256((output.parent / "settings.json").read_bytes()).hexdigest(),
            "workload_sha256": hashlib.sha256((output.parent / "workload.json").read_bytes()).hexdigest(),
        },
    )
    sharing = mode == "simple"
    if sharing:
        from .sql_trace_replay_shared import install_shared_forward, prepare_shared_model

        prepare_shared_model()
    model = HFModelConfig(
        path=settings["model"],
        load_tokenizer=True,
        trust_remote_code=False,
        override_config={"attn_implementation": "sdpa", "use_cache": False, "attention_dropout": 0.0},
        enable_gradient_checkpointing=True,
        use_remove_padding=False,
        use_fused_kernels=False,
    )
    engine = FSDPEngineWithLMHead(
        model_config=model,
        engine_config=FSDPEngineConfig(
            strategy="fsdp",
            dtype="bfloat16",
            model_dtype="float32",
            mixed_precision={"param_dtype": "bfloat16", "reduce_dtype": "float32", "buffer_dtype": "float32"},
            param_offload=settings["offload"],
            optimizer_offload=settings["offload"],
            forward_only=False,
            use_dynamic_bsz=False,
            micro_batch_size_per_gpu=4,
            use_remove_padding=False,
            use_fused_kernels=False,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            seed=settings["seed"],
            fsdp_size=-1,
        ),
        optimizer_config=FSDPOptimizerConfig(
            lr=settings["lr"],
            total_training_steps=settings["steps"],
            lr_warmup_steps=0,
            weight_decay=0.0,
            clip_grad=1.0,
        ),
        checkpoint_config=CheckpointConfig(),
    )
    engine.initialize()
    if any(isinstance(module, torch.nn.Dropout) and module.p for module in engine.module.modules()):
        raise ValueError("Shared training comparison requires dropout=0 in both modes")
    pad = model.hf_config.pad_token_id
    pad = int(pad if pad is not None else model.hf_config.eos_token_id)
    indices = [workload["check_step"]] if phase == "check" else list(range(settings["steps"]))
    batches = {}
    # Both modes compute old-policy probabilities via the unmodified, independent engine forward.
    # No training has taken place yet. This preparation is deliberately outside update timing.
    with engine.eval_mode(), torch.no_grad():
        for step in indices:
            global_groups = workload["groups"][step * world : (step + 1) * world]
            local_group = global_groups[rank]
            denominator = sum(
                len(row["tokens"]) - row["prompt_length"]
                for group in global_groups
                for slot in group["slots"]
                for row in slot["rows"]
                if row
            )
            slot_count = max(len(group["slots"]) for group in global_groups)
            prepared = []
            for slot_index in range(slot_count):
                slot = local_group["slots"][slot_index] if slot_index < len(local_group["slots"]) else None
                data = make_batch(slot, pad, denominator, world)
                _, info = engine.forward_step(data, loss_function=None, forward_only=True)
                data["old_log_probs"] = info["model_output"]["log_probs"].cpu()
                data["clip_ratio"] = torch.full((4,), settings["clip_ratio"])
                prepared.append(data)
            batches[step] = prepared
    if sharing:
        install_shared_forward(engine)
    elif any(
        name == "prefix_grouper" or name.startswith(("prefix_grouper.", "agentlightning.verl.prefix_grouper"))
        for name in sys.modules
    ):
        raise RuntimeError("Baseline imported PrefixGrouper; refusing to publish an independent baseline")
    reference = output.parent / "baseline-check" / f"rank-{rank}" if sharing and phase == "check" else None
    evidence = []
    for step in indices:
        group = workload["groups"][step * world + rank]
        runtime.synchronize()
        dist.barrier()
        runtime.reset_peak_memory_stats()
        started = time.perf_counter()
        forward_seconds = backward_seconds = optimizer_seconds = 0.0
        loss_sum = 0.0
        before = {}
        with engine.train_mode():
            engine.optimizer_zero_grad()
            if phase == "check":
                for index, (name, parameter) in enumerate(engine.module.named_parameters()):
                    before[index] = parameter_sample(parameter, settings["check_samples"])
                    result = verify_evidence(rank_root, reference, f"initial-{index}", before[index], settings)
                    evidence.append({"kind": "initial", "parameter": name, **result})
            for slot_index, data in enumerate(batches[step]):
                runtime.synchronize()
                clock = time.perf_counter()
                loss, info = engine.forward_step(data, loss_function=grpo_loss, forward_only=False)
                runtime.synchronize()
                forward_seconds += time.perf_counter() - clock
                loss_sum += info["loss"]
                if phase == "check":
                    mask = data["loss_mask"].values().bool()
                    old = data["old_log_probs"].values()[mask]
                    result = verify_evidence(rank_root, reference, f"old-logprobs-{slot_index}", old, settings)
                    evidence.append({"kind": "old_response_logprobs", "slot": slot_index, **result})
                    logits = info["model_output"]["log_probs"].values().cpu()[mask]
                    result = verify_evidence(rank_root, reference, f"logprobs-{slot_index}", logits, settings)
                    evidence.append({"kind": "response_logprobs", "slot": slot_index, **result})
                clock = time.perf_counter()
                loss.backward()
                runtime.synchronize()
                backward_seconds += time.perf_counter() - clock
                del loss, info
            if phase == "check":
                for index, (name, parameter) in enumerate(engine.module.named_parameters()):
                    gradient = (
                        torch.zeros_like(before[index])
                        if parameter.grad is None
                        else parameter_sample(parameter.grad, settings["check_samples"])
                    )
                    result = verify_evidence(rank_root, reference, f"gradient-{index}", gradient, settings)
                    evidence.append({"kind": "gradient", "parameter": name, **result})
            clock = time.perf_counter()
            grad_norm = engine.optimizer_step()
            engine.lr_scheduler_step()
            runtime.synchronize()
            optimizer_seconds = time.perf_counter() - clock
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Nonfinite gradient norm: {grad_norm}")
            if phase == "check":
                for index, (name, parameter) in enumerate(engine.module.named_parameters()):
                    delta = parameter_sample(parameter, settings["check_samples"]) - before[index]
                    result = verify_evidence(rank_root, reference, f"update-{index}", delta, settings)
                    evidence.append({"kind": "parameter_update", "parameter": name, **result})
        runtime.synchronize()
        elapsed = time.perf_counter() - started
        if phase == "check":
            result = verify_evidence(rank_root, reference, "loss", torch.tensor([loss_sum]), settings)
            evidence.append({"kind": "loss", **result})
            result = verify_evidence(rank_root, reference, "grad-norm", torch.tensor([grad_norm]), settings)
            evidence.append({"kind": "global_gradient_norm", **result})
            persist(
                rank_root / "checks.jsonl",
                {
                    "step": step,
                    "scope": "full" if not settings["check_samples"] else "sampled_parameters_full_response_logprobs",
                    "results": evidence,
                    "passed": all(row["failed"] == 0 for row in evidence),
                },
            )
        else:
            independent = sum(len(row["tokens"]) for slot in group["slots"] for row in slot["rows"] if row)
            shared = independent - sum(3 * slot["common_prefix"] for slot in group["slots"])
            response_tokens = sum(
                len(row["tokens"]) - row["prompt_length"] for slot in group["slots"] for row in slot["rows"] if row
            )
            persist(
                rank_root / "steps.jsonl",
                {
                    "step": step,
                    "task_id": group["task_id"],
                    "warmup": step < settings["warmup"],
                    "e2e_seconds": elapsed,
                    "forward_seconds": forward_seconds,
                    "backward_sync_seconds": backward_seconds,
                    "optimizer_seconds": optimizer_seconds,
                    "peak_allocated_bytes": runtime.max_memory_allocated(),
                    "loss_dp_scaled": loss_sum,
                    "grad_norm": grad_norm,
                    "response_tokens": response_tokens,
                    "independent_tokens": independent,
                    "simple_tokens": shared,
                },
            )
        if rank == 0:
            print(f"{mode}/{phase}: finished update {step + 1}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
