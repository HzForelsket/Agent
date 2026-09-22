# Copyright (c) Microsoft. All rights reserved.

"""Accelerator-neutral input validation for the SQL fixed-trace training benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

BENCHMARK_ID = "agl-sql-trace-replay"


def write_json(path: Path, value: Any) -> None:
    """Persist a completed metadata record before the next stage starts."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def file_hash(path: Path) -> str:
    """Hash file contents without loading model weights into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_workload(root: Path, *, steps: int, world_size: int) -> dict[str, Any]:
    """Reuse capture validation and retain original SQL rewards and loss boundaries."""
    from analyze_multiturn_sharing import lcp, load_call_sequences, read_records

    records, context = load_call_sequences(root)
    if context["agent"] != "sql" or context["group_size"] != 4:
        raise ValueError("SQL replay requires original SQL capture with exactly four trajectories per question")
    diagnostics: list[str] = []
    rewards = {}
    for event in read_records(root / "events.jsonl", diagnostics):
        if event.get("agent") == "sql" and event.get("event") == "workflow_completed":
            tid = event["trajectory_id"]
            if tid in rewards:
                raise ValueError(f"Duplicate SQL reward: {tid}")
            reward = float(event["reward"])
            if not math.isfinite(reward):
                raise ValueError(f"Nonfinite SQL reward: {tid}")
            rewards[tid] = reward
    calls: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        calls[record["task_id"]][record["sample_index"]].append(record)
    required = steps * world_size
    if len(calls) < required:
        raise ValueError(f"Need {required} complete question groups, found {len(calls)}; reduce steps or devices")
    groups = []
    for task_id, samples in list(calls.items())[:required]:
        trajectories = [sorted(samples[i], key=lambda row: row["turn"]) for i in range(4)]
        scores = [rewards[trajectory[0]["trajectory_id"]] for trajectory in trajectories]
        mean = sum(scores) / 4
        std = math.sqrt(sum((score - mean) ** 2 for score in scores) / 3)
        advantages = [(score - mean) / (std + 1e-6) for score in scores]
        slots = []
        for slot in range(max(map(len, trajectories))):
            rows = []
            for sample, trajectory in enumerate(trajectories):
                if slot >= len(trajectory):
                    rows.append(None)
                    continue
                call = trajectory[slot]
                usage = call["response"]["usage"]
                prompt_length = usage["prompt_tokens"]
                temperature = float(call["request"].get("temperature", 1.0))
                if not math.isfinite(temperature) or temperature <= 0:
                    raise ValueError("Replay requires a positive, finite policy temperature")
                rows.append(
                    {
                        "trajectory_id": call["trajectory_id"],
                        "sample_index": sample,
                        "tokens": call["token_ids"],
                        "prompt_length": prompt_length,
                        "advantage": advantages[sample],
                        "temperature": temperature,
                    }
                )
            sequences = [row["tokens"] if row else [] for row in rows]
            prefix = min(lcp(sequences[0], sequence) for sequence in sequences)
            slots.append({"rows": rows, "common_prefix": prefix})
        groups.append({"task_id": task_id, "rewards": scores, "advantages": advantages, "slots": slots})
    if not any(any(group["advantages"]) for group in groups):
        raise ValueError("All selected groups have zero GRPO advantage; cannot validate a nonzero policy update")
    # The first verification step must contain a real policy gradient without changing the workload order.
    check_step = next(
        i
        for i in range(steps)
        if any(any(group["advantages"]) for group in groups[i * world_size : (i + 1) * world_size])
    )
    files = ("config.json", "selected_tasks.json", "calls.jsonl", "trajectories.jsonl", "events.jsonl")
    return {
        "benchmark_id": BENCHMARK_ID,
        "schema": 1,
        "capture": context,
        "source_sha256": {name: file_hash(root / name) for name in files},
        "steps": steps,
        "world_size": world_size,
        "check_step": check_step,
        "diagnostics": diagnostics,
        "groups": groups,
        "zero_advantage_groups": sum(not any(group["advantages"]) for group in groups),
        "objective": "clipped GRPO, sample-standardized trajectory reward, global response-token mean; KL=0, entropy=0",
        "old_policy": "Recomputed with the initial local checkpoint, fixed across all replay steps; not captured behavior-policy probabilities",
    }
