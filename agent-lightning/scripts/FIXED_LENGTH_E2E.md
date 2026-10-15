# 固定 token 长度的端到端训练测试

入口：`scripts/test_fixed_length_e2e.py`。

覆盖真实 vLLM rollout → Agent Lightning trace / triplet 转换 → old / reference
log-prob → GRPO advantage → actor 前向、反向和优化器更新。使用标准 VERL FSDP
路径，不启用 PrefixGrouper；这是功能测试，不进行性能比较。

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
  --model /absolute/path/to/model \
  --prompt-tokens 128 \
  --response-tokens 32 \
  --steps 1 \
  --train-batch-size 2 \
  --rollouts-per-sample 2 \
  --n-devices 1 \
  --output-dir /absolute/path/to/new-run
```

可用 `--n-runners` 调整并发，`--tensor-parallel-size` 调整 rollout TP，
`--micro-batch-size` 调整每设备训练与 log-prob 微批大小。
设备数必须能被 TP 整除；`train-batch-size * rollouts-per-sample` 必须能被
`n-devices * micro-batch-size` 整除。`--seed` 控制请求 seed、rollout 和数据采样配置。
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
