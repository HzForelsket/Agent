# VERL

!!! tip "Shortcut"

    You can use the shortcut `agl.VERL(...)` to create a VERL instance.

    ```python
    import agentlightning as agl

    agl.VERL(...)
    ```

## Installation

```bash
pip install agentlightning[verl]
```

!!! warning

    To avoid various compatibility issues, follow the steps in the [installation guide](../tutorials/installation.md) to set up VERL and its dependencies. Installing VERL directly with `pip install agentlightning[verl]` can cause issues unless you already have a compatible version of PyTorch installed.

!!! note "Notes for Readers"

    [VERL][agentlightning.algorithm.verl.VERL] in this article refers to a wrapper, provided by Agent-lightning, of the [VERL framework](https://github.com/volcengine/verl). It's a subclass of [agentlightning.Algorithm][]. To differentiate it from the VERL framework, all references to the VERL framework shall use the term "VERL framework", and all references to the Agent-lightning wrapper shall be highlighted with a link.

## Resources

[VERL][agentlightning.algorithm.verl.VERL] expects no initial resources. The first LLM endpoint is directly deployed from the VERL configuration (`.actor_rollout_ref.model.path`). The resource key is always `main_llm`.

[VERL][agentlightning.algorithm.verl.VERL] currently does not support optimizing multiple [LLM][agentlightning.LLM]s together.

!!! note

    The resource type created by [VERL][agentlightning.algorithm.verl.VERL] is actually a [ProxyLLM][agentlightning.ProxyLLM], a subclass of the [LLM][agentlightning.LLM] type. This object contains a **URL template** provided by [VERL][agentlightning.algorithm.verl.VERL], with placeholders for rollout and attempt IDs. When a rollout begins on the agent side, the framework uses the current `rollout_id` and `attempt_id` to format this template, generating a final, unique endpoint URL. This URL points to [VERL][agentlightning.algorithm.verl.VERL]'s internal proxy, allowing it to intercept and log all traffic for that specific attempt, for tracing and load balancing purposes. For agents created with the `@rollout` decorator, this resolution of the template is handled automatically ("auto-stripped"). Class-based agents will need to manually resolve the [`ProxyLLM`][agentlightning.ProxyLLM] using the rollout context.

    ```python
    proxy_llm = resources["main_llm"]
    proxy_llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
    ```

## Customization

Internally, [VERL][agentlightning.algorithm.verl.VERL] decomposes each agent execution into prompt–response pairs via the [Adapter][agentlightning.Adapter] and associates them with their corresponding reward signals as [Triplet][agentlightning.Triplet] objects. The final scalar reward, derived from the last triplet in the trajectory, is propagated to all preceding triplets following the [identical assignment strategy](https://arxiv.org/abs/2508.03680). This ensures that each triplet receives an identical reward signal and can be independently optimized as a valid RLHF trajectory within the VERL framework.

At present, [VERL][agentlightning.algorithm.verl.VERL] does not expose fine-grained control over its reward propagation or credit assignment mechanisms. Users requiring customized reward shaping or trajectory decomposition are advised to clone and modify the [VERL][agentlightning.algorithm.verl.VERL] source implementation directly.

### Shared-prefix training with PrefixGrouper

For text-only GRPO workloads, VERL can use [PrefixGrouper](https://github.com/CASIA-IVA-Lab/PrefixGrouper) to compute an identical prompt once for the responses that share it. Enable it in the Agent Lightning section of the VERL configuration and keep padded inputs enabled:

```python
algorithm = agl.VERL(
    config={
        "agentlightning": {"prefix_grouper": {"enabled": True}},
        "algorithm": {"adv_estimator": "grpo"},
        "actor_rollout_ref": {
            "model": {
                "path": "Qwen/Qwen2.5-0.5B-Instruct",
                "use_remove_padding": False,
            },
            "rollout": {"n": 4, "log_prob_micro_batch_size_per_gpu": 4},
            "actor": {"ppo_micro_batch_size_per_gpu": 4},
            "ref": {"log_prob_micro_batch_size_per_gpu": 4},
        },
    }
)
```

The optimization applies to the actor update and the old/reference-policy log-probability passes; rollout generation remains on the configured vLLM server. Rows are grouped only when their non-padding prompt token IDs match exactly. The integration targets VERL 0.9's `TrainingWorker`/FSDP model-engine stack and uses VERL's native attention patch.

PrefixGrouper currently requires FSDP/FSDP2, `use_remove_padding=False`, fused kernels disabled, Ulysses sequence parallel size 1, and text-only batches. Unsupported multimodal, distillation-top-k, and sum-pi-squared batches fall back to the standard VERL forward.

Agent Lightning calls VERL's device auto-configuration before creating Ray resources. If `torch-npu` and a usable Ascend device are present it selects `trainer.device=npu`; otherwise it selects CUDA. `VERL_PLATFORM=huawei` or `VERL_PLATFORM=nvidia` remains available as an explicit override.

Install the platform-specific stack and compare the standard and shared-prefix paths with one or more Hugging Face model IDs:

```bash
conda run -n agent --no-capture-output python scripts/prefix_grouper_stack.py --backend auto

conda run -n agent --no-capture-output python scripts/benchmark_prefix_grouper.py \
    --models Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-135M-Instruct TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --case 1024:4 \
    --case 1536:8 \
    --batch-size 8 \
    --response-length 64 \
    --output-json prefix-grouper-results.json \
    --output-markdown prefix-grouper-results.md
```

Use `--backend gpu` or `--backend npu` to override installation detection, and `--dry-run` to inspect the commands without changing the environment. The runtime benchmark accepts `--device auto` (the default), `gpu`/`cuda`, `npu`, or an indexed device such as `npu:1`.

The two supported dependency matrices are intentionally separate because the accelerator plugins require different PyTorch versions:

| Backend | PyTorch | vLLM | Accelerator plugin | VERL | CANN |
|---|---:|---:|---:|---:|---:|
| NVIDIA GPU | 2.11.0 | 0.22.1 | CUDA 12.9 wheel | 0.9.0 | N/A |
| Ascend NPU | 2.10.0 | 0.22.1 | vLLM-Ascend 0.22.1rc1 / torch-npu 2.10.0 | 0.9.0 | 9.0.0 |

The NPU installer first installs the CPU PyTorch wheel required by torch-npu, builds the source-only `arctic-inference==0.1.1` package without its obsolete isolated Torch 2.7 build dependency, and builds upstream vLLM with `VLLM_TARGET_DEVICE=empty`. The Ascend plugin then supplies the device kernels. This avoids installing vLLM 0.22.1's CUDA wheel, whose metadata requires PyTorch 2.11. The exact package pins live in `scripts/requirements_prefix_grouper_gpu.txt` and `scripts/requirements_prefix_grouper_npu.txt`; the matrices checked by the benchmark live in `scripts/prefix_grouper_stack.py`.

Each case uses `PROMPT_LENGTH:GROUP_SIZE`. The default `--weights random` mode downloads only model configurations and instantiates the complete architectures, which is sufficient for dense-kernel timing and memory comparisons. Use `--weights pretrained` when learned weights are specifically required. The script checks response log-probability equivalence before reporting median latency, response-token throughput, speedup, peak accelerator memory, and memory reduction.

The GPU requirements select vLLM's CUDA 12.9 build. The NPU requirements follow the vLLM-Ascend 0.22.1rc1 release matrix and require a host with the matching Ascend driver, CANN, NNAL, and device permissions; installing the Python packages alone cannot provide those system components.

## Tutorials Using VERL

- [Train SQL Agent with RL](../how-to/train-sql-agent.md) - A practical example of training a SQL agent using VERL.

## References - Entrypoint

::: agentlightning.algorithm.verl

## References - Implementation

::: agentlightning.verl
