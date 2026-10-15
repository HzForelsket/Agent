# 固定 token 长度的端到端训练测试

入口：`scripts/test_fixed_length_e2e.py`。

覆盖真实 vLLM rollout → Agent Lightning trace / triplet 转换 → old / reference
log-prob → GRPO advantage → actor 前向、反向和优化器更新。
通过 `--mode no-share|simple` 选择训练路径，默认 `no-share`。

- `no-share`：标准 Agent Lightning / VERL FSDP 路径，不启用 PrefixGrouper。
- `simple`：通过生产 `agentlightning.prefix_grouper.enabled` 接入共享前缀 worker，
  在 old/reference log-prob 与 actor update 中共享同一 prompt 的多条 rollout 前缀，
  response 部分保持独立。rollout 仍使用相同的真实 vLLM 生成流程。

每次训练进程只运行一种模式。两次运行完成后，用独立报告入口比较结果；训练入口不会自动启动另一种模式。

## 与随机 prompt 基准复用的实现

沿用 `scripts/benchmark_prefix_grouper.py`（`pg-verl-ppa`）所用的
`agentlightning/verl/prefix_grouper.py`：E2E 的 `PrefixGrouperActorRolloutRefWorker`
为 actor 和 reference 使用同一个 `PrefixGrouperTrainingWorker`，复用
`build_prefix_grouper` 的打包方式、位置映射和 `apply_prefix_grouper_patch` 的注意力实现。
脚本不另写 simple attention 或修改底层算子。
simple 显式采用该基准默认的 `fusion` 注意力后端，不启用另一套 custom 算子。

两种模式统一使用 SDPA、BF16 参数 / FP32 梯度归约与 buffer，关闭 KV cache、
remove-padding、fused kernels 和 FSDP 引擎的 `torch.compile`，序列并行度为 1。
这些设置与随机输入基准对齐，避免引擎默认值引入额外实现差异。
E2E 的真实 rollout、GRPO loss 和优化器步骤继续使用生产训练路径。
随机输入基准的直接 response-logits 测量并不包含这些阶段，因此不能直接把它的耗时
与本 E2E 耗时比较；底层 kernel 的实际派发仍须在目标设备上通过 profile 确认。

## 长度语义

- `--prompt-tokens P`：合成普通文本经模型 tokenizer 编码后，按 token 重复并截取为
  **精确 P 个 token**。这是受控的合成 workload，不代表自然数据集质量。
- `--response-tokens R`：真实生成 **精确 R 个 token**。
  请求设置 `min_tokens=max_tokens=R`、`ignore_eos=true`，清空停止词与停止 token。
- 使用 vLLM 原生 `/v1/completions` 的 token ID 输入，不应用 chat template，
  不额外添加 BOS/EOS，不执行 decode → encode。`P + R` 超出模型上下文时直接报错。
- 本项目现有 hosted-vLLM LiteLLM 路由会转换 token prompt。因此通过已有
  `daemon_cls` 扩展接口暴露训练进程的 vLLM 服务地址，将真实返回的 token IDs
  写入活动 tracer，再由生产 `LlmProxyTraceToTriplet` 转换。
  此测试覆盖 rollout、存储、trace 转换和训练；不验证 LiteLLM HTTP 路由。
- reward 固定为 `rollout_index / (rollouts_per_sample - 1)`，使同组 reward 不同，
  用于检查 GRPO 与反向传播。它不是对回答质量的评分。若生成轨迹相同导致梯度抵消为零，
  测试会失败，不将零梯度训练计为成功。

## 运行

在具有可用加速器和项目配套 Agent Lightning / VERL / vLLM 依赖的机器上运行。
`--model` 必须是已经下载的完整模型目录，包含权重、config 和 tokenizer。
默认 1 个设备、1 个训练 step、每步 2 个 prompt、每个 prompt 2 条 rollout。
脚本不会自动下载模型。NPU 端到端运行仅可在另一台有真实可用 NPU 的机器进行，
遵循项目固定软件栈；本工作区所在机器不执行 NPU 端到端测试。

```bash
cd /home/huangzhong/Agent/agent-lightning
conda run -n agent --no-capture-output python scripts/test_fixed_length_e2e.py \
  --mode simple \
  --model /absolute/path/to/model \
  --prompt-tokens 128 \
  --response-tokens 32 \
  --steps 1 \
  --train-batch-size 2 \
  --rollouts-per-sample 2 \
  --n-devices 1 \
  --output-dir /absolute/path/to/simple-run
```

运行 no-share 时，将 `--mode` 改为 `no-share`，输出目录改为另一个全新目录
（例如 `/absolute/path/to/no-share-run`），其余参数保持一致。两个模式必须独立启动，
避免 PrefixGrouper 的进程内 patch 影响 no-share。结果与阶段记录均包含 `mode`。

可用 `--n-runners` 调整并发，`--tensor-parallel-size` 调整 rollout TP，
`--micro-batch-size` 调整每设备训练与 log-prob 微批大小；两种模式均默认等于
`--rollouts-per-sample`（默认 2），让同组 rollout 能进入同一微批。
设备数必须能被 TP 整除；`train-batch-size * rollouts-per-sample` 必须能被
`n-devices * micro-batch-size` 整除。simple 还要求 `train-batch-size` 能被设备数整除，
微批大小是 `rollouts-per-sample` 的整数倍，以免设备切分或微批切分拆开共享组。
两种模式均使用固定微批并关闭 actor 内部 shuffle。
`--seed` 控制请求 seed、rollout 和数据采样配置。
不保证不同硬件或软件版本下的逐 token 重现。

只准备输入和配置时，在上述命令末尾增加 `--prepare-only`，并使用另一个全新输出目录。
它只读取本地 config / tokenizer，不加载模型权重、不启动 Ray、不进行 rollout 或训练。
`status=prepared` 不代表 E2E 通过。

## 成功与输出

成功必须同时满足：

- 每次请求的返回 prompt IDs 与输入逐个一致，response IDs 与 usage 的长度均正确，
  结束原因为 `length`。
- trace 转换后以及 actor 更新前，训练批次的行数、prompt/response 张量形状、
  attention mask 和每个 GRPO 组的 rollout 数均符合配置，无 padding 或丢弃的 prompt。
- simple 模式还检查前缀分组确实启用、每组只包含同题且 prompt 完全相同的 sibling
  rollout，并检查 actor 微批切分不会拆散这些组。
- old/ref log-prob、advantage 和 return 均有限，advantage 非零，完整 response 参与训练。
- 各训练阶段完成，actor loss 和学习率有限，梯度范数为有限正数。
  这验证真实更新路径已执行，不构成模型质量改善或逐参数更新量的证明。
- 完成指定数量的训练 step 和互不重复的 rollout。

全部通过才写入 `result.json` 的 `status=passed` 并以 0 退出。异常以非零状态退出。

输出目录必须尚不存在，以免覆盖已有结果：

| 文件 | 内容 |
| --- | --- |
| `run.json` | 原始命令、参数、Python 环境和依赖版本 |
| `dataset.json` | 实际使用的固定长度 prompt token IDs |
| `config.json` / `resolved_config.json` | 配置覆盖项 / 训练时合并后的完整配置 |
| `device.json` | 运行设备信息 |
| `events.jsonl` | 逐 rollout 的真实 token IDs、reward、逐阶段检查与生产 trainer 指标 |
| `result.json` | `prepared` 或完整 E2E 的 `passed` 结果 |
| `failure.json` | 已有完成阶段时记录后续失败与堆栈 |

每个完成阶段立即追加并刷新记录。若尚未完成任何 rollout / 训练阶段就初始化失败，
打印异常并删除本次输出目录；已有阶段证据则保留为部分结果。

## 生成对比报告

报告入口为 `scripts/report_fixed_length_e2e.py`，仅使用 Python 标准库，读取已有结果，
不会加载模型、启动 rollout 或重新运行训练。

```bash
conda run -n agent --no-capture-output python \
  /home/huangzhong/Agent/agent-lightning/scripts/report_fixed_length_e2e.py \
  --no-share-dir /absolute/path/to/no-share-run \
  --simple-dir /absolute/path/to/simple-run \
  --output-dir /absolute/path/to/new-comparison
```

输出目录必须尚不存在。生成两个文件：

- `report.md`：中文对比表，包含完整 step 与各阶段耗时、response 吞吐、生产 trainer
  已记录的显存和其他数值指标，以及降低比例和耗时加速比。
- `report.json`：相同对比的结构化数据，包含逐指标均值、中位数、标准差、最小/最大值、
  原始样本和完整逐 step 指标。

同时报告全部 steps 和固定排除 step 1 的窗口。默认只有 1 step 时，后者明确标为无数据。
response 吞吐按窗口内真实 response token 总数除以完整 step 耗时总和计算；
耗时加速比为 `no-share / simple`，降低比例为 `(no-share - simple) / no-share * 100%`。
显存只比较原始指标，不把显存比例称为加速比；缺失或非有限的可选指标会列明而不补零。
step 时间包含 rollout 与训练，不包含模型加载和 Ray 初始化；嵌套阶段不能相加作为总耗时。

报告只接受两种模式均 `status=passed` 的完整运行。会重新核对 rollout 数量、实际长度、
step 连续性、分组、输入 token IDs、采样 seed/reward、设备、软件版本，以及原始和合并配置。
比较配置时仅移除模式对应的 PrefixGrouper 开关、simple 的 fusion 后端选项和输出路径；
其余配置不同、结果不完整、仅 `prepared`、存在失败记录或必需耗时缺失时，以非零状态退出，
不生成对比报告。报告仍不构成逐元素数值等价、实际 kernel 一致或权重内容一致的证明。
