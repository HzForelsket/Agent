# Qwen3-30B-A3B GPU Legacy PrefixGrouper 4 卡 Profile 阶段与算子分析

## 结论摘要

本次采集复现 [2 卡 legacy 分析](profile_qwen3_30b_4096_g8_20260901_gpu0_4_profile_stage_operator_analysis.md) 的配置，只将设备扩展为 4 张 NVIDIA A100-SXM4-80GB。PrefixGrouper 使用优化提交前的高级索引 `UngroupFunction`，不包含 `permute.contiguous`、线性索引或 `LinearUngroupFunction`。

在 4 卡模型级合成负载中，标准重复前缀路径单次前向延迟为 4186.080 ms，legacy PrefixGrouper 为 2806.526 ms，吞吐由 489 token/s 提升至 730 token/s；单次样本对应 1.49x 加速。

Trace 内包含 profiler 开销，因此 trace 外层区间为 4337.346 ms 和 3044.598 ms，对应 1.42x。算子分析使用 trace 内区间，模型级性能使用 profiler 之外的独立计时结果。

主要观察：

- GPU 计算活跃区间由 2730.019 ms 降至 716.381 ms，减少 73.8%。
- NCCL 参数 AllGather 由 2293.253 ms 降至 2188.351 ms，仅减少 4.6%。
- rank 0 的累计 GPU 事件时间中，NCCL 占比由 45.45% 升至 75.07%，计算占比由 54.55% 降至 24.93%。
- 计算与通信重叠由 709.505 ms 降至 24.142 ms，减少 96.6%。
- 48 层 Q/K/V 共执行 144 次 legacy `UngroupFunction`，内部包含 288 次高级索引读取和 288 次高级索引写回。
- legacy ungroup 内部 GPU kernel 累计时间为 26.431 ms，占 rank 0 全部 GPU 累计时间 0.92%，占排除 NCCL 后计算时间 3.70%。
- 与 4 卡 `permute.contiguous` trace 的专项范围相比，ungroup 累计 GPU 时间由 26.431 ms 降至 17.502 ms，减少 8.929 ms（33.8%，1.51x）；但当前单次全模型计时不足以证明模型级加速。

## 实验条件

| 项目 | 配置 |
|---|---|
| 模型 | `/dev/shm/qwen3-30b-a3b-instruct-2507`，pretrained，30.53B 参数，48 层 |
| 硬件 | NVIDIA A100-SXM4-80GB × 4，物理 GPU 0、1、2、3 |
| 分布式 | 单节点，world size 4，VERL RayWorkerGroup + FSDP |
| 精度 | bfloat16 |
| 输入 | prompt length 4096，response length 64，group size 8 |
| Batch | 每 rank 8，全局 32 |
| 模式 | forward-only |
| PrefixGrouper | commit `ee0ac44` 的 legacy 源码；无 `LinearUngroupFunction` |
| 软件栈 | PyTorch 2.11.0+cu129，vLLM 0.22.1+cu129，VERL 0.9.0，Transformers 5.10.4 |
| Profile | 记录 shape，不记录 profile memory |
| 性能采样 | warmup 0，repeat 1；p50、p95、p99 相同 |
| 功耗 | 未采集 |

执行时从优化提交 `296074c` 的父提交导出 `PrefixGrouper/src` 到临时目录，未修改或回退当前工作树：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/tmp/agl-qwen30b-legacy4.jIN57m/source/src:/home/huangzhong/Agent/agent-lightning \
conda run -n agent --no-capture-output \
  python scripts/benchmark_prefix_grouper.py \
  --models /dev/shm/qwen3-30b-a3b-instruct-2507 \
  --case 4096:8 \
  --batch-size-per-rank 8 \
  --response-length 64 \
  --dtype bfloat16 \
  --device gpu \
  --strategy fsdp \
  --n-devices-per-node 4 \
  --warmup 0 \
  --repeats 1 \
  --no-backward \
  --no-gradient-checkpointing \
  --local-files-only \
  --no-power \
  --profile \
  --profile-dir /tmp/agl-qwen30b-legacy4.jIN57m/profiles \
  --output-json /tmp/agl-qwen30b-legacy4.jIN57m/results.json \
  --output-markdown /tmp/agl-qwen30b-legacy4.jIN57m/results.md
```

## 模型级结果

下表来自 profiler 之外的独立设备事件计时；延迟采用 4 个 rank 中的慢端值。

| 路径 | latency | 吞吐 | 最坏 rank 峰值显存 |
|---|---:|---:|---:|
| 标准重复前缀 | 4186.080 ms | 489.2 token/s | 21699.7 MiB |
| Legacy PrefixGrouper | 2806.526 ms | 729.7 token/s | 17778.3 MiB |
| 变化 | 1.49x | +49.2% | -3921.4 MiB（-18.1%） |

4 张设备峰值显存之和由 86676.2 MiB 降至 70994.2 MiB，共减少 15682.0 MiB（18.1%）。

理论工作量中，dense model token 由 133120 降至 18432，减少 86.2%；causal attention pair 由 276956160 降至 42017792，减少 84.8%。因此 1.49x 是完整 PrefixGrouper 路径收益，不是 legacy ungroup 单独带来的收益。

精度检查通过：response log-prob 最大/平均误差为 0.698234/0.058214，logit 最大/平均误差为 2.937500/0.127745，top-1 一致率为 95.02%。该结论仅适用于当前 bfloat16 容差与本次合成输入。

## 阶段耗时

GPU 活跃区间采用时间区间并集，不重复统计同一设备上相互重叠的 kernel。比较值取 4 个 rank 中该指标的较慢值；FSDP pre/post-forward 是 49 个 annotation 的 CPU 累计时间。

| 阶段 | 标准重复前缀 | Legacy PrefixGrouper | 变化 |
|---|---:|---:|---:|
| Trace 外层整步 | 4337.346 ms | 3044.598 ms | -29.8%，1.42x |
| GPU 计算活跃区间 | 2730.019 ms | 716.381 ms | -73.8% |
| NCCL 参数 AllGather 区间 | 2293.253 ms | 2188.351 ms | -4.6% |
| 计算/通信重叠 | 709.505 ms | 24.142 ms | -96.6% |
| FSDP root-pre-forward（CPU） | 2.451 ms | 2.475 ms | +1.0% |
| FSDP pre-forward（CPU） | 31.338 ms | 31.742 ms | +1.3% |
| FSDP post-forward（CPU） | 8.172 ms | 7.935 ms | -2.9% |

### 每个 rank 的阶段数据

| 路径 | Rank | Trace wall | GPU 活跃并集 | 计算并集 | NCCL 并集 | 计算/通信重叠 |
|---|---:|---:|---:|---:|---:|---:|
| 标准重复前缀 | 0 | 4313.199 ms | 4277.734 ms | 2720.396 ms | 2266.233 ms | 708.895 ms |
| 标准重复前缀 | 1 | 4337.346 ms | 4304.112 ms | 2703.753 ms | 2292.063 ms | 691.704 ms |
| 标准重复前缀 | 2 | 4336.641 ms | 4301.601 ms | 2679.388 ms | 2293.253 ms | 671.040 ms |
| 标准重复前缀 | 3 | 4311.548 ms | 4278.869 ms | 2730.019 ms | 2258.356 ms | 709.505 ms |
| Legacy PrefixGrouper | 0 | 3009.385 ms | 2843.789 ms | 713.642 ms | 2149.482 ms | 19.335 ms |
| Legacy PrefixGrouper | 1 | 3044.598 ms | 2881.787 ms | 716.381 ms | 2188.351 ms | 22.945 ms |
| Legacy PrefixGrouper | 2 | 3043.306 ms | 2873.741 ms | 715.384 ms | 2182.499 ms | 24.142 ms |
| Legacy PrefixGrouper | 3 | 2989.395 ms | 2825.130 ms | 712.467 ms | 2135.788 ms | 23.125 ms |

Legacy PrefixGrouper 将计算并集缩短约 2.01 秒，但 NCCL AllGather 基本不变，并减少约 0.69 秒的计算/通信重叠。这解释了为什么 GPU 计算减少 73.8%，而 trace wall 只缩短 29.8%。

## GPU 算子累计时间

GPU kernel 通过 trace 的 `External id` 关联回发起它的 `cpu_op`，再按算子名累计 `dur`。本节统一使用 rank 0；其余 rank 的整体趋势一致。

这些数值是累计 GPU 时间，不是互斥 wall time。不同 stream 上的 NCCL 和计算 kernel 可以重叠，各行不能直接相加作为整步耗时。

占比使用两个口径：

- **总 GPU 占比**：算子累计时间 / rank 0 全部 GPU kernel、memcpy、memset 累计时间。标准路径分母为 4986.629 ms，Legacy PrefixGrouper 分母为 2863.124 ms。
- **计算内占比**：算子累计时间 / 排除 NCCL 后的累计 GPU 计算时间。标准路径分母为 2720.396 ms，Legacy PrefixGrouper 分母为 713.642 ms。

| 路径 | NCCL 累计时间/占比 | 计算累计时间/占比 | 总 GPU 累计时间 |
|---|---:|---:|---:|
| 标准重复前缀 | 2266.233 ms / 45.45% | 2720.396 ms / 54.55% | 4986.629 ms |
| Legacy PrefixGrouper | 2149.482 ms / 75.07% | 713.642 ms / 24.93% | 2863.124 ms |

| GPU 算子/路径 | 标准时间 | 标准总 GPU 占比 | 标准计算内占比 | Legacy 时间 | Legacy 总 GPU 占比 | Legacy 计算内占比 |
|---|---:|---:|---:|---:|---:|---:|
| `record_param_comms` / NCCL AllGather | 2266.233 ms | 45.45% | — | 2149.482 ms | 75.07% | — |
| `aten::mm` | 993.173 ms | 19.92% | 36.51% | 209.444 ms | 7.32% | 29.35% |
| Attention | 308.801 ms | 6.19% | 11.35% | 200.280 ms | 7.00% | 28.06% |
| `aten::mul` | 447.671 ms | 8.98% | 16.46% | 50.482 ms | 1.76% | 7.07% |
| `aten::masked_fill_` | 339.551 ms | 6.81% | 12.48% | 34.356 ms | 1.20% | 4.81% |
| `aten::index` | 141.651 ms | 2.84% | 5.21% | 40.348 ms | 1.41% | 5.65% |
| `aten::copy_` | 122.650 ms | 2.46% | 4.51% | 67.484 ms | 2.36% | 9.46% |
| `aten::_index_put_impl_` | 2.908 ms | 0.06% | 0.11% | 21.636 ms | 0.76% | 3.03% |
| `aten::pow` | 63.306 ms | 1.27% | 2.33% | 8.754 ms | 0.31% | 1.23% |
| `aten::add` | 64.453 ms | 1.29% | 2.37% | 11.276 ms | 0.39% | 1.58% |
| `aten::cat` | 53.504 ms | 1.07% | 1.97% | 11.595 ms | 0.40% | 1.62% |
| `aten::silu` | 48.556 ms | 0.97% | 1.78% | 6.511 ms | 0.23% | 0.91% |
| `aten::mean` | 38.334 ms | 0.77% | 1.41% | 6.589 ms | 0.23% | 0.92% |
| `aten::sum` | 49.085 ms | 0.98% | 1.80% | 9.815 ms | 0.34% | 1.38% |
| `aten::topk` | 19.828 ms | 0.40% | 0.73% | 3.348 ms | 0.12% | 0.47% |
| `aten::neg` | 16.055 ms | 0.32% | 0.59% | 2.457 ms | 0.09% | 0.34% |
| `aten::sort` | 5.807 ms | 0.12% | 0.21% | 4.541 ms | 0.16% | 0.64% |

Attention 的具体实现不同：标准路径使用 `aten::_flash_attention_forward`，累计 308.801 ms；Legacy PrefixGrouper 使用 `aten::_efficient_attention_forward`，累计 200.280 ms。因此该行同时包含 token 工作量变化和 kernel 实现变化。

## Legacy `UngroupFunction` 专项分析

rank 0 trace 中共有 144 次 `UngroupFunction`，对应 48 层 × Q/K/V。下表只统计这些 annotation 内部的算子和其嵌套 GPU kernel，不包含模型其他位置的同名算子。

| 内部步骤 | 调用次数 | 累计 GPU 时间 | 每次平均 | 总 GPU 占比 | 计算内占比 |
|---|---:|---:|---:|---:|---:|
| 高级索引读取 `aten::index` | 288 | 12.688 ms | 44.1 μs | 0.44% | 1.78% |
| 高级索引写回 `aten::_index_put_impl_` | 288 | 12.164 ms | 42.2 μs | 0.42% | 1.70% |
| prefix/suffix 输出零初始化 | 288 | 1.579 ms | 5.5 μs | 0.06% | 0.22% |
| 合计 | — | 26.431 ms | 每个 Q/K/V ungroup 183.6 μs | 0.92% | 3.70% |

完整 Legacy PrefixGrouper trace 中 `aten::index` 为 40.348 ms，`aten::_index_put_impl_` 为 21.636 ms。其中可严格归因于 `UngroupFunction` 的部分分别为 12.688 ms 和 12.164 ms；其余同名算子位于 attention mask、布局构建或其他模型范围，不能全部归因于 ungroup。

## 与 `permute.contiguous` 4 卡结果对照

专项范围使用两个 rank 0 trace 中 `UngroupFunction`/`LinearUngroupFunction` 的完整嵌套 GPU kernel 累计时间：

| 路径 | 读取/写回与布局 | 零初始化 | Ungroup 总计 | 每个 Q/K/V ungroup |
|---|---:|---:|---:|---:|
| Legacy `Index + IndexPut` | 24.853 ms | 1.579 ms | 26.431 ms | 183.6 μs |
| `permute.contiguous + index_select + index_copy_` | 15.926 ms | 1.576 ms | 17.502 ms | 121.5 μs |
| 变化 | -35.9% | 基本不变 | -33.8%，1.51x | -33.8% |

新路径在 ungroup 专项范围内节省 8.929 ms。两次模型级 benchmark 的 PrefixGrouper 独立计时分别为 2806.526 ms（legacy，warmup 0/repeat 1）和 2812.245 ms（新路径，warmup 1/repeat 3），差异仅 +0.20%；trace 慢端分别为 3044.598 ms 和 3101.054 ms，新路径反而高 1.85%。由于采样窗口不同、仅各运行一次完整 profile，而且 NCCL 波动远大于 8.9 ms，本数据只支持“ungroup 局部 GPU 累计时间下降 33.8%”，不支持“30B 全模型延迟已经改善”。

## 原始数据

- [独立计时与显存结果](/tmp/agl-qwen30b-legacy4.jIN57m/results.md)
- [机器可读结果](/tmp/agl-qwen30b-legacy4.jIN57m/results.json)
- [完整 profile 目录](/tmp/agl-qwen30b-legacy4.jIN57m/profiles)
- [标准重复前缀 rank 0 trace](/tmp/agl-qwen30b-legacy4.jIN57m/profiles/01_qwen3-30b-a3b-instruct-2507/prompt_4096_group_8/forward/baseline/192.168.1.30_rank_0.1788244025725532392.pt.trace.json)
- [Legacy PrefixGrouper rank 0 trace](/tmp/agl-qwen30b-legacy4.jIN57m/profiles/01_qwen3-30b-a3b-instruct-2507/prompt_4096_group_8/forward/prefix_grouper/192.168.1.30_rank_0.1788244031123144381.pt.trace.json)
- [`permute.contiguous` 4 卡分析](profile_qwen3_30b_4096_g8_20260901_gpu0_1_2_3_permute_contiguous_stage_operator_analysis.md)

## 口径与限制

1. 本次严格复现参考报告的 warmup 0、repeat 1，只是单次观测；不能据此判断跨运行稳定性或置信区间。
2. Trace 会引入额外开销，所以阶段/算子分析使用 trace 数据，模型级速度使用 profiler 外的独立计时数据。
3. 本次仅采集 forward；不包含 backward、优化器、rollout、vLLM prefill、奖励计算或数据加载。
4. 当前“标准重复前缀”是同一 benchmark 进程中的 `grouped=False` 路径。进程仍加载 PrefixGrouper hook，不是完全不导入 PrefixGrouper 的独立 Agent Lightning 基线；1.49x 仅解释为本次 runner 内路径对比。
5. 1.49x 包含共享前缀减少 token/attention 工作量和 attention kernel 实现变化，不能解释为 legacy ungroup 单独带来 1.49x。
6. legacy 与新 ungroup 的专项 trace 对照来自两次独立运行；硬件、模型、输入和 profile 范围一致，但性能采样 warmup/repeat 不同，完整模型延迟不构成严格 A/B。
7. 算子累计 GPU 时间可能跨 stream 重叠。NCCL、计算及其重叠关系应结合阶段区间并集理解。
8. 原始结果和 trace 当前位于 `/tmp`；系统清理临时目录后链接会失效。
