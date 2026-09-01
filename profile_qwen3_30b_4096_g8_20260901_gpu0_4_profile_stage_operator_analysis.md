# Qwen3-30B-A3B GPU Profile 阶段与算子分析

## 结论摘要

本次采集比较同一 Agent Lightning + VERL/FSDP 进程中的标准重复前缀路径（`grouped=False`）与 PrefixGrouper 路径（`grouped=True`）。在 2 张 NVIDIA A100-SXM4-80GB 上，PrefixGrouper 的独立计时前向延迟由 4390.034 ms 降至 3078.091 ms，吞吐由 233 token/s 提升至 333 token/s；单次样本对应 1.43x 加速。

Trace 内包含 profiler 开销，因此 trace 外层区间为 4498.576 ms 和 3284.986 ms，对应 1.37x。算子分析应使用 trace 内区间，端到端性能应使用独立计时结果。

主要观察：

- GPU 计算活跃区间由 2451.999 ms 降至 712.606 ms，减少 70.9%。
- NCCL 参数 AllGather 仍约为 2.4 秒，仅减少 1.5%。
- 在 rank 0 的累计 GPU 事件时间中，NCCL 占比由 49.92% 升至 77.25%，计算占比由 50.08% 降至 22.75%。
- 计算与通信重叠由 425.280 ms 降至 12.151 ms。
- PrefixGrouper 路径已经明显受到 FSDP 参数通信约束；后续收益更依赖通信量或计算/通信重叠，而不是继续压缩注意力计算。

## 实验条件

| 项目 | 配置 |
|---|---|
| 模型 | `.cache/qwen3-30b-a3b-instruct-2507`，pretrained，30.53B 参数 |
| 硬件 | NVIDIA A100-SXM4-80GB × 2，物理 GPU 0、4 |
| 分布式 | 单节点，world size 2，VERL RayWorkerGroup + FSDP |
| 精度 | bfloat16，TF32 开启 |
| 输入 | prompt length 4096，response length 64，group size 8 |
| Batch | 每 rank 8，全局 16 |
| 模式 | forward-only |
| 软件栈 | PyTorch 2.11.0+cu129，vLLM 0.22.1+cu129，VERL 0.9.0 |
| Profile | 记录 shape，不记录 profile memory |
| 性能采样 | warmup 0，repeat 1；因此 p50、p95、p99 相同 |
| 功耗 | 未采集 |

执行命令：

```bash
CUDA_VISIBLE_DEVICES=0,4 conda run -n agent --no-capture-output \
  python scripts/benchmark_prefix_grouper.py \
  --models .cache/qwen3-30b-a3b-instruct-2507 \
  --case 4096:8 \
  --batch-size-per-rank 8 \
  --response-length 64 \
  --device gpu \
  --n-devices-per-node 2 \
  --strategy fsdp \
  --warmup 0 \
  --repeats 1 \
  --no-backward \
  --local-files-only \
  --no-power \
  --profile \
  --profile-dir output/profile_qwen3_30b_4096_g8_20260901_gpu0_4/profiles \
  --output-json output/profile_qwen3_30b_4096_g8_20260901_gpu0_4/results.json \
  --output-markdown output/profile_qwen3_30b_4096_g8_20260901_gpu0_4/results.md
```

## 端到端结果

下表来自 profiler 之外的独立计时 step：

| 路径 | p50 latency | 吞吐 | 最坏 rank 峰值显存 |
|---|---:|---:|---:|
| 标准重复前缀 | 4390.034 ms | 233 token/s | 36218.9 MiB |
| PrefixGrouper | 3078.091 ms | 333 token/s | 32298.2 MiB |
| 变化 | 1.43x | +43.0% | -3920.7 MiB（-10.8%） |

精度检查通过：response log-prob 最大/平均误差为 0.610272/0.057365，logit 最大/平均误差为 2.937500/0.134734，top-1 一致率为 94.63%。该结论仅适用于当前 bfloat16 容差与本次输入。

## 阶段耗时

GPU 活跃区间采用时间区间并集，不会重复统计同一设备上相互重叠的 kernel。表中比较值取两个 rank 中的较慢值；FSDP pre/post-forward 是 49 个 annotation 的 CPU 累计时间。

| 阶段 | 标准重复前缀 | PrefixGrouper | 变化 |
|---|---:|---:|---:|
| Trace 外层整步 | 4498.576 ms | 3284.986 ms | -27.0%，1.37x |
| GPU 计算活跃区间 | 2451.999 ms | 712.606 ms | -70.9% |
| NCCL 参数 AllGather 区间 | 2444.618 ms | 2408.872 ms | -1.5% |
| 计算/通信重叠 | 425.280 ms | 12.151 ms | -97.1% |
| FSDP root-pre-forward（CPU） | 2.637 ms | 2.549 ms | -3.3% |
| FSDP pre-forward（CPU） | 31.356 ms | 40.671 ms | +29.7% |
| FSDP post-forward（CPU） | 8.289 ms | 8.127 ms | -2.0% |

### 每个 rank 的阶段数据

| 路径 | Rank | Trace wall | GPU 活跃并集 | 计算并集 | NCCL 并集 | 计算/通信重叠 |
|---|---:|---:|---:|---:|---:|---:|
| 标准重复前缀 | 0 | 4498.576 ms | 4471.337 ms | 2451.999 ms | 2444.618 ms | 425.280 ms |
| 标准重复前缀 | 1 | 4477.061 ms | 4446.506 ms | 2421.547 ms | 2440.964 ms | 416.005 ms |
| PrefixGrouper | 0 | 3284.686 ms | 3106.343 ms | 709.420 ms | 2408.872 ms | 11.948 ms |
| PrefixGrouper | 1 | 3284.986 ms | 3075.043 ms | 712.606 ms | 2374.587 ms | 12.151 ms |

PrefixGrouper 的计算区间缩短约 1.74 秒，但 NCCL AllGather 基本不变，并且少了约 0.41 秒的通信/计算重叠。这解释了为什么计算量显著下降，而 trace wall 只缩短约 1.21 秒。

## GPU 算子累计时间

GPU kernel 通过 trace 的 `External id` 关联回发起它的 `cpu_op`，再按算子名累计 `dur`。为保证时间和占比使用同一分母，本节统一使用 rank 0；rank 1 的整体趋势与 rank 0 一致。

这些数值是累计 GPU 时间，不是互斥 wall time：不同 stream 上的 NCCL 和计算 kernel 可以重叠，因此各行不能直接相加作为整步耗时。

占比使用两个口径：

- **总 GPU 占比**：算子累计时间 / rank 0 全部 GPU kernel、memcpy、memset 累计时间。标准路径分母为 4896.617 ms，PrefixGrouper 分母为 3118.292 ms。
- **计算内占比**：算子累计时间 / 排除 NCCL 后的累计 GPU 计算时间。标准路径分母为 2451.999 ms，PrefixGrouper 分母为 709.420 ms。

| 路径 | NCCL 累计时间/占比 | 计算累计时间/占比 | 总 GPU 累计时间 |
|---|---:|---:|---:|
| 标准重复前缀 | 2444.618 ms / 49.92% | 2451.999 ms / 50.08% | 4896.617 ms |
| PrefixGrouper | 2408.872 ms / 77.25% | 709.420 ms / 22.75% | 3118.292 ms |

| GPU 算子/路径 | 标准时间 | 标准总 GPU 占比 | 标准计算内占比 | PrefixGrouper 时间 | PrefixGrouper 总 GPU 占比 | PrefixGrouper 计算内占比 |
|---|---:|---:|---:|---:|---:|---:|
| `record_param_comms` / NCCL AllGather | 2444.618 ms | 49.92% | — | 2408.872 ms | 77.25% | — |
| `aten::mm` | 913.052 ms | 18.65% | 37.24% | 211.147 ms | 6.77% | 29.76% |
| Attention | 307.737 ms | 6.28% | 12.55% | 200.818 ms | 6.44% | 28.31% |
| `aten::mul` | 370.821 ms | 7.57% | 15.12% | 50.557 ms | 1.62% | 7.13% |
| `aten::masked_fill_` | 260.268 ms | 5.32% | 10.61% | 34.353 ms | 1.10% | 4.84% |
| `aten::index` | 126.698 ms | 2.59% | 5.17% | 38.352 ms | 1.23% | 5.41% |
| `aten::copy_` | 123.195 ms | 2.52% | 5.02% | 67.701 ms | 2.17% | 9.54% |
| `aten::pow` | 63.436 ms | 1.30% | 2.59% | 8.742 ms | 0.28% | 1.23% |
| `aten::add` | 62.557 ms | 1.28% | 2.55% | 9.352 ms | 0.30% | 1.32% |
| `aten::cat` | 53.355 ms | 1.09% | 2.18% | 12.258 ms | 0.39% | 1.73% |
| `aten::silu` | 48.622 ms | 0.99% | 1.98% | 6.514 ms | 0.21% | 0.92% |
| `aten::mean` | 38.540 ms | 0.79% | 1.57% | 6.592 ms | 0.21% | 0.93% |
| `aten::sum` | 37.562 ms | 0.77% | 1.53% | 6.088 ms | 0.20% | 0.86% |
| `aten::topk` | 19.812 ms | 0.40% | 0.81% | 3.354 ms | 0.11% | 0.47% |
| `aten::neg` | 16.029 ms | 0.33% | 0.65% | 2.464 ms | 0.08% | 0.35% |
| `aten::sort` | 5.796 ms | 0.12% | 0.24% | 4.599 ms | 0.15% | 0.65% |

Attention 的具体实现不同：标准路径使用 `aten::_flash_attention_forward`，累计 307.737 ms；PrefixGrouper 路径使用 `aten::_efficient_attention_forward`/CUTLASS memory-efficient attention，累计 200.818 ms。因此该行既包含 token 工作量变化，也包含 kernel 实现变化。

### PrefixGrouper 新增或更明显的索引算子

PrefixGrouper 为构建共享前缀布局引入了额外索引/掩码处理：

| 算子 | PrefixGrouper 累计 GPU 时间 | 总 GPU 占比 | 计算内占比 |
|---|---:|---:|---:|
| `aten::_index_put_impl_` | 21.712 ms | 0.70% | 3.06% |
| `aten::index_select` | 6.292 ms | 0.20% | 0.89% |
| `aten::bitwise_and` | 5.293 ms | 0.17% | 0.75% |
| `aten::where` | 3.815 ms | 0.12% | 0.54% |
| `aten::le` | 3.586 ms | 0.11% | 0.51% |
| `aten::fill_` | 2.844 ms | 0.09% | 0.40% |

这些额外开销远小于 `aten::mm`、逐元素计算和 attention 节省的时间。

## 原始数据

- [独立计时与显存结果](results.md)
- [机器可读结果](results.json)
- [标准重复前缀 rank 0 trace](profiles/01_qwen3-30b-a3b-instruct-2507/prompt_4096_group_8/forward/baseline/192.168.1.30_rank_0.1788225505770090734.pt.trace.json)
- [标准重复前缀 rank 1 trace](profiles/01_qwen3-30b-a3b-instruct-2507/prompt_4096_group_8/forward/baseline/192.168.1.30_rank_1.1788225505763504541.pt.trace.json)
- [PrefixGrouper rank 0 trace](profiles/01_qwen3-30b-a3b-instruct-2507/prompt_4096_group_8/forward/prefix_grouper/192.168.1.30_rank_0.1788225511457009444.pt.trace.json)
- [PrefixGrouper rank 1 trace](profiles/01_qwen3-30b-a3b-instruct-2507/prompt_4096_group_8/forward/prefix_grouper/192.168.1.30_rank_1.1788225511440071737.pt.trace.json)

## 口径与限制

1. 本次 `warmup=0`、`repeats=1`，只能作为当前 workload 的单次观测；不能据此判断跨运行稳定性或置信区间。
2. Trace 会引入额外开销，所以阶段/算子分析使用 trace 数据，端到端速度使用 profiler 外的独立计时数据。
3. 当前 trace 没有 embedding、每层 attention、MoE、LM head 等显式语义 annotation；这些阶段只能通过算子名推断。若需要严格的逐层语义阶段，应在对应模型路径加入 `torch.profiler.record_function` 后重新采集。
4. 当前“标准重复前缀”是在同一 benchmark 进程中以 `grouped=False` 运行的标准路径。进程仍加载了 PrefixGrouper 代码，因此这不是一个完全不导入、不安装 hook 的独立 Agent Lightning 基线；1.43x 应解释为本次 runner 内的路径对比。
5. 算子累计 GPU 时间可能跨 stream 重叠。NCCL、计算及其重叠关系应结合“阶段耗时”表中的区间并集理解。
