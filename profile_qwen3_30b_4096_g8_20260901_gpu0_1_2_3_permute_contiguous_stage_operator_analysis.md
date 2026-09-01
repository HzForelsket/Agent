# Qwen3-30B-A3B GPU `permute.contiguous` Profile 阶段与算子分析

## 结论摘要

本次采集比较同一 Agent Lightning + VERL/FSDP 模型级合成负载中的标准重复前缀路径（`grouped=False`）与启用线性 `Ungroup` 的 PrefixGrouper 路径（`grouped=True`）。线性路径将逻辑 BHSD 张量转换为 BSHD，展平 `(batch, sequence)` 后使用 `index_select + index_copy_`，最后转回连续 BHSD。

在 4 张 NVIDIA A100-SXM4-80GB 上，PrefixGrouper 的独立计时前向 p50 延迟由 4192.391 ms 降至 2812.245 ms，吞吐由 489 token/s 提升至 728 token/s，对应 1.49x 加速。

Trace 内包含 profiler 开销，因此 trace 外层区间为 4324.768 ms 和 3101.054 ms，对应 1.39x。算子分析使用 trace 内区间，模型级性能使用 profiler 之外的独立计时结果。

主要观察：

- GPU 计算活跃区间由 2708.341 ms 降至 710.816 ms，减少 73.8%。
- NCCL 参数 AllGather 由 2258.709 ms 降至 2204.453 ms，仅减少 2.4%。
- rank 0 的累计 GPU 事件时间中，NCCL 占比由 45.40% 升至 75.19%，计算占比由 54.60% 降至 24.81%。
- 计算与通信重叠由 691.946 ms 降至 25.534 ms，PrefixGrouper 路径已明显受 FSDP 参数通信限制。
- 48 层 Q/K/V 共调用 144 次 `LinearUngroupFunction`。其内部没有 `aten::index`、`aten::index_put_` 或 `aten::_index_put_impl_`。
- `LinearUngroupFunction` 内部全部 GPU kernel 累计时间为 17.502 ms，占 rank 0 全部 GPU 累计时间 0.61%，占排除 NCCL 后计算时间 2.47%；其中布局转换、读取和写回三项为 15.926 ms，其余 1.576 ms 是输出零初始化。
- 实际 Q/K/V 的 BHSD stride 已对应 BSHD 物理布局，入口 `permute(...).contiguous()` 是零拷贝视图；只有 prefix/suffix 输出转回连续 BHSD 时产生布局拷贝。

## 实验条件

| 项目 | 配置 |
|---|---|
| 模型 | `/dev/shm/qwen3-30b-a3b-instruct-2507`，pretrained，30.53B 参数，48 层 |
| 硬件 | NVIDIA A100-SXM4-80GB × 4，物理 GPU 0、1、2、3 |
| 分布式 | 单节点，world size 4，VERL RayWorkerGroup + FSDP |
| 精度 | bfloat16 |
| 输入 | prompt length 4096，response length 64，group size 8 |
| Batch | 每 rank 8，全局 32 |
| Q/K/V | 每层 Q `[1,32,4608,128]`，K/V `[1,4,4608,128]` |
| 模式 | forward-only |
| 软件栈 | PyTorch 2.11.0+cu129，vLLM 0.22.1+cu129，VERL 0.9.0，Transformers 5.10.4 |
| PrefixGrouper | 包元数据 0.0.1.post1；通过 `PYTHONPATH` 使用本地 `permute.contiguous` 源码 |
| Profile | 记录 shape，不记录 profile memory |
| 性能采样 | warmup 1，repeat 3 |
| 功耗 | 未采集 |

执行命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/home/huangzhong/Agent/PrefixGrouper/src:/home/huangzhong/Agent/agent-lightning \
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
  --warmup 1 \
  --repeats 3 \
  --no-backward \
  --no-gradient-checkpointing \
  --local-files-only \
  --no-power \
  --profile \
  --profile-dir /tmp/agl-qwen30b-permute.hPuZE0/model-profile \
  --output-json /tmp/agl-qwen30b-permute.hPuZE0/model-benchmark.json \
  --output-markdown /tmp/agl-qwen30b-permute.hPuZE0/model-benchmark.md
```

## 模型级结果

下表来自 profiler 之外的独立设备事件计时；延迟采用 4 个 rank 中每次采样的慢端值。

| 路径 | p50 latency | p95 latency | 吞吐 | 最坏 rank 峰值显存 |
|---|---:|---:|---:|---:|
| 标准重复前缀 | 4192.391 ms | 4196.990 ms | 488.5 token/s | 21699.7 MiB |
| PrefixGrouper `permute.contiguous` | 2812.245 ms | 2817.676 ms | 728.2 token/s | 17778.3 MiB |
| 变化 | 1.49x | 1.49x | +49.1% | -3921.4 MiB（-18.1%） |

4 张设备峰值显存之和由 86676.2 MiB 降至 70992.7 MiB，共减少 15683.5 MiB（18.1%）。

理论工作量中，dense model token 由 133120 降至 18432，减少 86.2%；causal attention pair 由 276956160 降至 42017792，减少 84.8%。因此 1.49x 是完整 PrefixGrouper 路径收益，不能全部归因于 `permute.contiguous`。

精度检查通过：response log-prob 最大/平均误差为 0.698234/0.058214，logit 最大/平均误差为 2.937500/0.127745，top-1 一致率为 95.02%。该结论仅适用于当前 bfloat16 容差与本次合成输入。

## 阶段耗时

GPU 活跃区间采用时间区间并集，不重复统计同一设备上相互重叠的 kernel。比较值取 4 个 rank 中该指标的较慢值；FSDP pre/post-forward 是 49 个 annotation 的 CPU 累计时间。

| 阶段 | 标准重复前缀 | PrefixGrouper `permute.contiguous` | 变化 |
|---|---:|---:|---:|
| Trace 外层整步 | 4324.768 ms | 3101.054 ms | -28.3%，1.39x |
| GPU 计算活跃区间 | 2708.341 ms | 710.816 ms | -73.8% |
| NCCL 参数 AllGather 区间 | 2258.709 ms | 2204.453 ms | -2.4% |
| 计算/通信重叠 | 691.946 ms | 25.534 ms | -96.3% |
| FSDP root-pre-forward（CPU） | 2.502 ms | 2.390 ms | -4.5% |
| FSDP pre-forward（CPU） | 46.565 ms | 30.583 ms | -34.3% |
| FSDP post-forward（CPU） | 8.063 ms | 7.805 ms | -3.2% |

### 每个 rank 的阶段数据

| 路径 | Rank | Trace wall | GPU 活跃并集 | 计算并集 | NCCL 并集 | 计算/通信重叠 |
|---|---:|---:|---:|---:|---:|---:|
| 标准重复前缀 | 0 | 4324.518 ms | 4268.445 ms | 2708.341 ms | 2252.050 ms | 691.946 ms |
| 标准重复前缀 | 1 | 4324.371 ms | 4269.765 ms | 2697.184 ms | 2252.006 ms | 679.426 ms |
| 标准重复前缀 | 2 | 4324.233 ms | 4268.953 ms | 2673.371 ms | 2258.709 ms | 663.127 ms |
| 标准重复前缀 | 3 | 4324.768 ms | 4269.894 ms | 2696.340 ms | 2250.552 ms | 676.998 ms |
| PrefixGrouper | 0 | 3044.215 ms | 2832.500 ms | 708.592 ms | 2146.963 ms | 23.056 ms |
| PrefixGrouper | 1 | 3058.307 ms | 2844.366 ms | 710.816 ms | 2159.084 ms | 25.534 ms |
| PrefixGrouper | 2 | 3101.054 ms | 2882.951 ms | 701.495 ms | 2204.453 ms | 22.997 ms |
| PrefixGrouper | 3 | 3068.752 ms | 2859.027 ms | 706.827 ms | 2174.770 ms | 22.570 ms |

PrefixGrouper 将计算并集缩短约 2.00 秒，但 NCCL AllGather 基本不变，并减少约 0.67 秒的计算/通信重叠。这解释了为什么 GPU 计算减少 73.8%，而 trace wall 只缩短 28.3%。

## GPU 算子累计时间

GPU kernel 通过 trace 的 `External id` 关联回发起它的 `cpu_op`，再按算子名累计 `dur`。为保证时间和占比使用同一分母，本节统一使用 rank 0；其余 rank 的整体趋势一致。

这些数值是累计 GPU 时间，不是互斥 wall time。不同 stream 上的 NCCL 和计算 kernel 可以重叠，各行不能直接相加作为整步耗时。

占比使用两个口径：

- **总 GPU 占比**：算子累计时间 / rank 0 全部 GPU kernel、memcpy、memset 累计时间。标准路径分母为 4960.391 ms，PrefixGrouper 分母为 2855.556 ms。
- **计算内占比**：算子累计时间 / 排除 NCCL 后的累计 GPU 计算时间。标准路径分母为 2708.341 ms，PrefixGrouper 分母为 708.592 ms。

| 路径 | NCCL 累计时间/占比 | 计算累计时间/占比 | 总 GPU 累计时间 |
|---|---:|---:|---:|
| 标准重复前缀 | 2252.050 ms / 45.40% | 2708.341 ms / 54.60% | 4960.391 ms |
| PrefixGrouper | 2146.963 ms / 75.19% | 708.592 ms / 24.81% | 2855.556 ms |

| GPU 算子/路径 | 标准时间 | 标准总 GPU 占比 | 标准计算内占比 | PrefixGrouper 时间 | PrefixGrouper 总 GPU 占比 | PrefixGrouper 计算内占比 |
|---|---:|---:|---:|---:|---:|---:|
| `record_param_comms` / NCCL AllGather | 2252.050 ms | 45.40% | — | 2146.963 ms | 75.19% | — |
| `aten::mm` | 988.450 ms | 19.93% | 36.50% | 210.467 ms | 7.37% | 29.70% |
| Attention | 308.417 ms | 6.22% | 11.39% | 200.747 ms | 7.03% | 28.33% |
| `aten::mul` | 445.422 ms | 8.98% | 16.45% | 50.643 ms | 1.77% | 7.15% |
| `aten::masked_fill_` | 337.425 ms | 6.80% | 12.46% | 34.466 ms | 1.21% | 4.86% |
| `aten::index` | 140.180 ms | 2.83% | 5.18% | 29.093 ms | 1.02% | 4.11% |
| `aten::copy_` | 122.572 ms | 2.47% | 4.53% | 73.838 ms | 2.59% | 10.42% |
| `aten::pow` | 63.125 ms | 1.27% | 2.33% | 8.750 ms | 0.31% | 1.23% |
| `aten::add` | 63.950 ms | 1.29% | 2.36% | 10.991 ms | 0.38% | 1.55% |
| `aten::cat` | 53.433 ms | 1.08% | 1.97% | 11.640 ms | 0.41% | 1.64% |
| `aten::silu` | 48.734 ms | 0.98% | 1.80% | 6.536 ms | 0.23% | 0.92% |
| `aten::mean` | 38.391 ms | 0.77% | 1.42% | 6.600 ms | 0.23% | 0.93% |
| `aten::sum` | 48.019 ms | 0.97% | 1.77% | 9.740 ms | 0.34% | 1.37% |
| `aten::topk` | 19.795 ms | 0.40% | 0.73% | 3.360 ms | 0.12% | 0.47% |
| `aten::neg` | 16.045 ms | 0.32% | 0.59% | 2.466 ms | 0.09% | 0.35% |
| `aten::sort` | 5.790 ms | 0.12% | 0.21% | 4.566 ms | 0.16% | 0.64% |

Attention 的具体实现不同：标准路径使用 `aten::_flash_attention_forward`，累计 308.417 ms；PrefixGrouper 路径使用 `aten::_efficient_attention_forward`，累计 200.747 ms。因此该行同时包含 token 工作量变化和 kernel 实现变化。

## `LinearUngroupFunction` 专项分析

rank 0 trace 中共有 144 次 `LinearUngroupFunction`，对应 48 层 × Q/K/V。下表只统计这些 annotation 内部的算子和其嵌套 GPU kernel，不包含模型其他位置的同名算子。

| 内部步骤 | 调用次数 | 累计 GPU 时间 | 每次平均 | 总 GPU 占比 | 计算内占比 |
|---|---:|---:|---:|---:|---:|
| 输出 `permute.contiguous` | 288 | 5.645 ms | 19.6 μs | 0.20% | 0.80% |
| `index_select` | 288 | 3.480 ms | 12.1 μs | 0.12% | 0.49% |
| `index_copy_` | 288 | 6.800 ms | 23.6 μs | 0.24% | 0.96% |
| prefix/suffix 输出零初始化 | 288 | 1.576 ms | 5.5 μs | 0.06% | 0.22% |
| 合计 | — | 17.502 ms | 每个 Q/K/V ungroup 121.5 μs | 0.61% | 2.47% |

每次 ungroup 分别处理 prefix 和 suffix，所以有两次 `index_select`、两次 `index_copy_` 和两次输出布局转换。入口虽然调用了 `x.permute(0, 2, 1, 3).contiguous()`，但 trace 没有第三次 `aten::contiguous`：

- Q 输入逻辑 shape 为 `[1,32,4608,128]`，stride 为 `[18874368,128,4096,1]`。
- K/V 输入逻辑 shape 为 `[1,4,4608,128]`，stride 为 `[2359296,128,512,1]`。
- 这些 BHSD 张量底层已经按 BSHD 排列；`permute(0,2,1,3)` 后 stride 连续，因此随后的 `.contiguous()` 直接返回，不产生 GPU copy。
- 只有 prefix/suffix 输出从 BSHD 转回连续 BHSD 时产生 288 次实际布局拷贝。

专项范围内的 legacy 高级索引调用数为零：

| 算子 | `LinearUngroupFunction` 内调用次数 | 累计 GPU 时间 |
|---|---:|---:|
| `aten::index` | 0 | 0 ms |
| `aten::index_put_` | 0 | 0 ms |
| `aten::_index_put_impl_` | 0 | 0 ms |

完整 PrefixGrouper trace 中仍有 `aten::index` 29.093 ms 和 `aten::_index_put_impl_` 9.588 ms，但 External id 与嵌套区间检查表明它们位于 `LinearUngroupFunction` 之外，不能归因于新的 ungroup 路径。

## 原始数据

- [独立计时与显存结果](/tmp/agl-qwen30b-permute.hPuZE0/model-benchmark.md)
- [机器可读结果](/tmp/agl-qwen30b-permute.hPuZE0/model-benchmark.json)
- [完整 profile 目录](/tmp/agl-qwen30b-permute.hPuZE0/model-profile)
- [标准重复前缀 rank 0 trace](/tmp/agl-qwen30b-permute.hPuZE0/model-profile/01_qwen3-30b-a3b-instruct-2507/prompt_4096_group_8/forward/baseline/192.168.1.30_rank_0.1788242202374433889.pt.trace.json)
- [PrefixGrouper rank 0 trace](/tmp/agl-qwen30b-permute.hPuZE0/model-profile/01_qwen3-30b-a3b-instruct-2507/prompt_4096_group_8/forward/prefix_grouper/192.168.1.30_rank_0.1788242207816388451.pt.trace.json)

## 口径与限制

1. 独立性能计时为 warmup 1、repeat 3，样本量仍较小；p95 是 3 个样本的线性插值，不代表跨运行置信区间。
2. Trace 会引入额外开销，所以阶段/算子分析使用 trace 数据，模型级速度使用 profiler 外的独立计时数据。
3. 本次仅采集 forward；不包含 backward、优化器、rollout、vLLM prefill、奖励计算或数据加载。
4. 当前“标准重复前缀”是同一 benchmark 进程中的 `grouped=False` 路径。进程仍加载 PrefixGrouper hook，不是完全不导入 PrefixGrouper 的独立 Agent Lightning 基线；1.49x 仅解释为本次 runner 内的路径对比。
5. 1.49x 包含共享前缀减少 token/attention 工作量和 attention kernel 实现变化，不能解释为 `permute.contiguous` 单独带来 1.49x。
6. 算子累计 GPU 时间可能跨 stream 重叠。NCCL、计算及其重叠关系应结合阶段区间并集理解。
7. 原始结果和 trace 当前位于 `/tmp`；系统清理临时目录后链接会失效。
