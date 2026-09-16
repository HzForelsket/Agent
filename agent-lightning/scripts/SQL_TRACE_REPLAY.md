# SQL 固定轨迹：独立计算与简单共享的完整训练步对比

正式 ID：`agl-sql-trace-replay`（Agent Lightning SQL 固定轨迹训练回放基准）。
GPU 与 NPU 使用相同入口、输入、分组、损失、优化器与报告逻辑。
只在设备初始化、NCCL/HCCL、已有注意力实现、同步及显存接口处选择设备。
不修改 SQL Agent、采集器、原 2WikiMQA 基准或现有完整 prompt 分组训练入口。

## 输入与训练语义

输入必须是 `collect_traces.py --agent sql` 生成的**原始目录**，包括
`config.json`、`selected_tasks.json`、`calls.jsonl`、`trajectories.jsonl`、`events.jsonl`。
仅复制 `analysis/` 不够：训练需要原始 SQL reward 与 prompt/response 边界。
使用本地对应模型 checkpoint 与 tokenizer，不下载模型，不启动 vLLM，不重新执行 SQL。
模型文件、输入文件、运行源码均记录内容 SHA-256。

- 复用离线分析器对完整题组的校验，同题必须四条轨迹全部有效。
- 同题四个真实 SQL reward 计算 GRPO advantage：`(reward - mean) / (sample_std + 1e-6)`。
  原轨迹各次调用的回复 token 使用该轨迹的 advantage；prompt 不产生 loss。
- baseline 走 Agent Lightning 使用的 VERL 0.9 FSDP 标准前向，不导入、不启用 PrefixGrouper。
- simple 每个调用序号共享**全组四条 `prompt+response` 序列的最长公共前缀**，首次分叉后独立计算。
  公共段若包含回复，仍保留各轨迹自己的 loss/advantage，梯度累加回共享节点。
  不跨题、跨调用序号或在轨迹内部共享，也不对只有部分轨迹相同的分支建树。
- 某条轨迹缺少该次调用时，全组公共前缀为零。
  为保持各 rank 相同的 FSDP collective 次数，缺失调用使用单 token、零 loss 的 padding；
  padding 开销包含在两侧时间中，不计入有效 token 数，也不冒充轨迹数据。
- 每个 rank 每步处理一个完整题组，逐调用序号累积梯度，最后执行一次 AdamW 更新。
  每个 micro-batch 同步梯度，避免 30B 模型长期保留未分片梯度。
  loss 为全局所有原始 response token 的均值；两侧归一化、数据顺序完全相同。
- BF16 前向、FP32 参数/梯度归约，梯度检查点开启；默认参数和优化器 offload。
  `--no-offload` 可在两种模式同时关闭 offload。完整步时间包含所选 offload 开销。
- PPO clip 默认 0.2，KL 与 entropy 系数为零，AdamW weight decay 为零、梯度裁剪为 1。
  old log-prob 由**初始本地 checkpoint 的独立前向**预计算，所有回放步固定使用。
  这不是采集时行为策略 log-prob，也不是在线 on-policy GRPO 或训练收敛实验。
  模型计算使用每次原始请求的 temperature。
- 零 advantage 题组按原顺序保留；若选中组全部为零，则停止，不伪造 reward。

## 运行前检查

训练环境使用 `scripts/prefix_grouper_stack.py` 中对应 GPU/NPU 的固定版本矩阵。
不要在已有 NPU 推理环境直接覆盖安装 GPU 包。
NPU 要求 CANN 9.0.0、torch/torch-npu 2.10.0、VERL 0.9.0；GPU 要求 torch 2.11.0、VERL 0.9.0。
启动器通过包元数据核对完整对应矩阵；CANN 需由 NPU 主机部署确认。
本入口不自动安装包，也不探测另一种设备来决定是否回退。

先只读真实轨迹，不初始化加速器（以下从 `agent-lightning` 目录执行）：

```bash
python scripts/benchmark_sql_trace_replay.py \
  --prepare-only --backend gpu \
  --input /实际路径/sql-30b-run01 \
  --devices 4 --steps 4 --warmup 1 \
  --output /实际路径/sql-replay-input-check
```

`--backend npu` 使用同一个 CPU 数据准备过程。
四卡四步消耗按采集顺序选取的 16 个完整题组（64 条完整轨迹），不循环复制或截断轨迹。
不足时明确报错，需根据真实覆盖率设置设备数与步数。输出目录必须新建。

## GPU

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/benchmark_sql_trace_replay.py \
  --backend gpu --devices 4 \
  --input /实际路径/sql-30b-run01 \
  --model /实际模型路径/Qwen3-30B-A3B-Instruct-2507 \
  --steps 4 --warmup 1 \
  --output /实际路径/sql-replay-gpu
```

## NPU

先加载该 NPU 主机的 CANN 环境，再执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
python scripts/benchmark_sql_trace_replay.py \
  --backend npu --devices 4 \
  --input /实际路径/sql-30b-run01 \
  --model /实际模型路径/Qwen3-30B-A3B-Instruct-2507 \
  --steps 4 --warmup 1 \
  --output /实际路径/sql-replay-npu
```

两条命令都自动顺序启动四组独立 torchrun 进程：baseline 数值检查、simple 数值检查、
baseline 性能测量、simple 性能测量。每次从相同权重及新优化器开始，检查更新不污染测量模型。
检查不通过或未观测到非零梯度/更新时停止，不输出加速结论。工作负载与已完成结果增量保存。
这些命令供目标 GPU/NPU 机器执行；准备入口不表示已在任一设备验证。

## 数值检查范围与报告

检查使用选中数据中首个含非零 advantage 的全局题组 batch，在初始模型上完成一次更新。
回复 log-prob 全量对比；loss、全局梯度范数也对比。
默认每个本地参数分片均匀采样最多 4096 个位置，比较初始参数、梯度与**更新差值**。
初始参数与 old log-prob 要求完全一致，其余用 `--atol` / `--rtol` 的逐元素判据；
参数更新差值的绝对容差另外限制为不超过 `lr * 1e-3`。
这是采样数值检查，不能宣称全部参数和梯度已逐元素验证。
`--check-samples 0` 检查所有本地分片元素；30B 模型会产生很大的 CPU 内存与磁盘需求，
baseline 需保存 FP32 初始参数、梯度和更新差值，合计约 360 GB（十进制），并有检查时的额外 CPU 缓冲。

输出包含：

| 文件 | 内容 |
|---|---|
| `settings.json` / `workload.json` | 固定配置、来源与模型摘要、题组、reward、advantage、共享长度 |
| `*-check/rank-*/checks.jsonl` | 各 rank 的逐项检查结果与检查范围 |
| `baseline-check/rank-*/*.pt` | 基线数值证据 |
| `*-measure/rank-*/steps.jsonl` | 每步完整时间、阶段时间、有效 token、loss、梯度范数、峰值显存 |
| `numerical_checks.json` | 数值检查汇总与通过状态 |
| `report.md` / `report.json` / `comparison.csv` | 两种模式的完整步耗时、阶段耗时、吞吐、显存与加速比 |

E2E 从步前同步之后开始，到参数更新及 offload 完成并同步结束；包含数据搬运、共享打包、
前向/loss、反向/梯度同步、裁剪、优化器和调度器。包含阶段同步计时的固定开销。
不含模型加载、轨迹读取与初始 old-policy 概率准备，也不含 SQL rollout。
默认排除首个完整更新；取每步最慢 rank 时间、全局原始 response token 总数计算吞吐。
GPU 与 NPU 分别生成各自的独立-vs-共享对照，不能把两台机器的结果当成只改变共享策略的受控对比。

重新生成已完成运行的报告无需加速器：

```bash
python scripts/benchmark_sql_trace_replay.py --report-only --output /实际路径/sql-replay-npu
```

## Included Files

| 文件 | 职责 |
|---|---|
| `scripts/benchmark_sql_trace_replay.py` | 公共 CLI、四阶段编排、固定版本检查、报告 |
| `scripts/sql_trace_replay_data.py` | CPU 侧真实轨迹校验、reward/advantage 与共享计划 |
| `agentlightning/verl/sql_trace_replay.py` | 两设备共用的 VERL FSDP 训练步、计时与数值证据 |
| `agentlightning/verl/sql_trace_replay_shared.py` | 仅 simple 进程导入的公共前缀前向及梯度映射 |

当前仅完成本地静态检查与 CLI 检查，尚无本入口的 GPU/NPU 运行正确性或性能证据。
