# 固定长度数据集的端到端训练

入口：`scripts/test_fixed_length_e2e.py`。

先生成固定输入长度的合成问答数据集，再复用
`benchmark_prefix_grouper_2wikimqa_e2e.py` 的配置、`wiki_agent`、
`BenchmarkTrainer` 和 `run_benchmark`。执行相同的 chat completions / LiteLLM
路由、真实 vLLM rollout、trace 转换、old/reference log-prob、GRPO 与 actor 更新。

`--mode no-share|simple` 分别映射到共同流程中的 `baseline|prefix_grouper`。
两种模式在独立进程、独立输出目录运行；不会自动运行另一种模式。

## 与 Wiki 一致的流程

- SDPA、BF16 参数 / FP32 梯度归约与 buffer，关闭 remove-padding 和 fused kernels。
- 开启 gradient checkpointing 和 activation offload。
- 加载相同的 `benchmark_fsdp_sync` hook，每个 micro-batch 同步梯度。
- actor 的编译开关显式关闭，ref 引擎的编译开关沿用 Wiki 默认配置。
- 共用 Wiki 的 trainer，不在 actor 计时内增加形状、分组、有限值或梯度检查。
- reward 使用相同的 answer token-overlap F1；数据中的答案来自合成事实。
  不再使用 rollout index reward，不要求每个 batch 的 advantage 或梯度非零。
  这不是自然数据集质量评测，也不证明每步发生非零参数更新。
- 默认每步 8 个 prompt、每题 4 条 rollout、micro-batch 为 2、16 个 runner，
  seed 为 20260827。不传 `--steps` 时运行数据集完整一轮。
- 设备数默认取可见设备数，rollout TP 默认等于设备数。资源切分沿用 Wiki 规则；
  micro-batch 可以只容纳同题的部分 rollout。实际共享范围取决于微批内的分组。

## 长度与数据准备

- `--prompt-tokens P`：**包括 system prompt、chat template 和 generation prompt**
  的精确输入 token 数。生成带已知答案的合成问答，并在准备数据集时填充到 P。
  若 P 小于必要的消息模板长度，或 tokenizer 无法构造该长度，则报错。
- `--response-tokens R`：**精确生成 R 个 token**。沿用 Wiki 的 chat 请求，附加
  `min_tokens=max_tokens=R`、`ignore_eos=true`、空停止词和停止 token 列表。
  这是用户指定的输出约束；Wiki 原入口仍使用原来的 EOS / 输出上限语义。
- P + R 必须不超过模型上下文，超出时直接报错，不缩短输出预算。
- `--dataset-size` 默认 64；不指定 steps 时，数据行数必须能被 train batch size 整除。
  因此默认运行 8 steps，报告的排除首步窗口有 7 个样本。
- 训练结束后再检查真实 serving prompt IDs 与准备的数据逐个一致、输出长度精确为 R、
  结束原因为 length、step 和 rollout 记录完整。检查不计入训练阶段或运行 wall。

## 使用

模型必须是本地已下载的完整模型目录。在具备可用加速器的机器上运行：

```bash
cd /home/huangzhong/Agent/agent-lightning
conda run -n agent --no-capture-output python scripts/test_fixed_length_e2e.py \
  --mode simple \
  --model /absolute/path/to/model \
  --prompt-tokens 2048 \
  --response-tokens 64 \
  --dataset-size 64 \
  --train-batch-size 8 \
  --rollouts-per-sample 4 \
  --micro-batch-size 4 \
  --n-devices 4 \
  --tensor-parallel-size 4 \
  --device gpu \
  --output-dir /absolute/path/to/simple-run
```

运行 no-share 时只修改 `--mode no-share` 和输出目录，其余参数保持一致。
使用同一 tokenizer 时，两次准备的数据内容相同，报告按实际文件 SHA-256 比较。
可用 `--steps` 限制训练步数。NPU 训练只可在另一台有真实可用 NPU 的机器执行，
沿用 Wiki 的固定软件栈；本工作区不运行 NPU 端到端训练。

只准备数据集与配置时增加 `--prepare-only`。此时必须明确给出 `--device gpu|npu`
和 `--n-devices`，与 Wiki 的 dry-run 一样不探测设备、不加载权重、不启动 Ray。

输出目录必须尚不存在：

| 文件 | 内容 |
| --- | --- |
| `run.json` | 原始命令、环境、依赖版本与参数 |
| `dataset.jsonl` | Wiki 格式的合成问答、精确 chat token IDs |
| `config.json` / `resolved_config.json` | 共用配置覆盖项与合并配置 |
| `metrics.jsonl` | Wiki 格式的逐 step 原始指标和完整 run 元数据 |
| `responses.jsonl` | 真实 serving token IDs、长度、seed、F1 与结束原因 |
| `result.json` | prepared，或完成训练且长度/完整性检查通过的 passed |
| `failure.json` | 已有完成阶段后发生的错误；此类运行不能比较 |

完成的 rollout 和 step 会逐条写入文件。若尚无 rollout/step 完成就失败，删除本次
输出目录；否则保留部分结果和错误记录。`status=prepared` 不代表训练已执行。

## 报告

```bash
conda run -n agent --no-capture-output python \
  /home/huangzhong/Agent/agent-lightning/scripts/report_fixed_length_e2e.py \
  --no-share-dir /absolute/path/to/no-share-run \
  --simple-dir /absolute/path/to/simple-run \
  --output-dir /absolute/path/to/new-comparison
```

报告复用 Wiki 的比较、统计和 Markdown 渲染，输出 `report.json`、`report.md`：

- 整次运行 wall，从训练器初始化到 fit 完成，排除前置数据准备和后置长度检查。
- 全 steps 与固定排除 step 1 的窗口，保留 mean、median、stddev、min/max、p90/p95。
- 相同的训练阶段耗时、每 token 耗时、吞吐、MFU、内存、F1 和输出长度分布。
- 阶段加速比 = baseline 平均时间 / PrefixGrouper 平均时间；wall 加速比单独计算。
  排除首步不保证已经达到稳态，嵌套阶段不能直接相加。

原始记录使用共同流程的 `baseline|prefix_grouper` 模式名，数据标识为 `FixedLength`，
不会被标成 2WikiMQA。只接受当前 Wiki 格式的完整结果；不读取旧的 events.jsonl 格式。
准备数据内容、长度、模型、软件栈、设备、采样与资源设置不同则拒绝比较。
