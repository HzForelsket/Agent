# 多任务多轮在线端到端训练

正式入口为 `scripts/benchmark_multiturn_online_e2e.py`，统一支持：

- `--task sql`：Spider SQL 的生成、执行、检查和改写流程；
- `--task q20`：20 Questions 的 CrewAI 多轮流程，只训练 player，固定 answerer 不进入训练轨迹；
- `--task web`：带 Wikipedia MCP 检索工具的多轮 Web RAG 流程。

每次运行只选择一个任务和一个模式。`baseline` 使用标准 VERL 前向；`simple`
启用 PrefixGrouper，只共享同题 4 条 rollout 完全相同的初始 prompt，后续多轮 suffix
保持独立。两个模式都使用轨迹级聚合，覆盖实时 rollout、GRPO、old/reference
log-prob 和 actor update。

## 四卡最小流程验证：八条轨迹、一个训练步

GRPO 需要同题至少两条 rollout，当前 FSDP 路径要求 prompt batch 能被设备数整除。
四卡最小配置为四个问题、每题两条轨迹。仅检查流程时显式使用下面的小规模配置，
不要直接采用后面的 10-step 对比参数：

```bash
python scripts/benchmark_multiturn_online_e2e.py \
  --task q20 --mode simple --device gpu --model Qwen3-8B \
  --tasks 4 --train-batch-size 4 --rollouts-per-sample 2 --steps 1 \
  --n-runners 8 --n-devices-per-node 4 --micro-batch-size-per-device 2
```

NPU 使用同一组参数，仅替换 `--device npu`。这只验证一个完整训练步，
不能据此给出性能对比结论；若同题的两条轨迹奖励相同，GRPO advantage 为零，
也不能据此声称策略获得了有效学习更新。

Q20 每轮会重建提示，因此一个 rollout 可能拆成多个无法按 token 前缀合并的训练片段。
trainer 在 old/reference log-prob 前按设备数与静态 micro-batch 的公倍数补齐，
在 GRPO advantage 前移除补齐项；actor update 使用完整的
`ppo_mini_batch_size × rollouts-per-sample` 批次。`training/n_logprob_padding`
记录推理补齐数量，`training/n_triplets_dropped_remainder` 记录训练尾批丢弃数量。
长 prompt 超出 prompt 区的部分保留在 suffix 中作为上下文，其 loss mask 为零；
模型回复按实际回复长度标记，避免把 padding 或历史上下文计入策略损失。

## 10 step 对比

以下命令使用 Qwen3-8B、4 张卡、32 个确定性抽样任务、每题 4 条 rollout 和 10 个
训练 step。baseline 与 simple 必须使用不同的新输出目录。

```bash
cd /home/huangzhong/Agent/agent-lightning

python scripts/benchmark_multiturn_online_e2e.py \
  --task sql --mode baseline --device npu \
  --model Qwen/Qwen3-8B --n-devices-per-node 4 \
  --rollouts-per-sample 4 --steps 10 --tasks 32 \
  --output-dir /runs/sql-baseline

python scripts/benchmark_multiturn_online_e2e.py \
  --task sql --mode simple --device npu \
  --model Qwen/Qwen3-8B --n-devices-per-node 4 \
  --rollouts-per-sample 4 --steps 10 --tasks 32 \
  --output-dir /runs/sql-simple
```

把 `--task sql` 分别替换为 `q20` 和 `web` 即可复用同一训练入口。Q20 的 player、
answerer 和可选 search 全部复用训练进程启动的本地 Qwen3-8B 服务，不需要任何
OpenAI Key。只有 player 请求经过 Agent Lightning trace proxy 并进入训练；answerer/search
通过内部直连资源调用同一本地模型，不进入 policy 轨迹。纯本地 Q20 要求 tensor parallel
等于设备数，以保证只有一个共享模型服务。Web 默认自动启动仓库内的 CPU MCP 检索服务，也可通过
`--web-mcp-url http://host:port/sse` 复用已有服务。

纯本地、禁止下载的 Q20 命令如下；baseline 只需更换 `--mode` 和输出目录：

```bash
python scripts/benchmark_multiturn_online_e2e.py \
  --task q20 --mode simple --device npu \
  --model /models/Qwen3-8B --model-name Qwen3-8B \
  --n-devices-per-node 4 --tensor-model-parallel-size 4 \
  --rollouts-per-sample 4 --steps 10 --tasks 32 \
  --local-files-only --output-dir /runs/q20-simple
```

实现中的 `api_key="dummy"` 只是本地 OpenAI 兼容客户端的必填占位值，不是凭证，
不会读取 `OPENAI_API_KEY`，也不会连接 OpenAI 服务。

Q20 的 player、answerer 和 search 单次模型请求默认超时 120 秒，可通过
`--q20-request-timeout` 调整；该参数不改变
整条 rollout attempt 默认 1200 秒的总时限。请求超时仍需结合本地模型服务日志排查。

训练轨迹长度与在线请求预算分别配置，SQL、Q20、Web 使用同一套规则：

- `--max-prompt-length` 和 `--max-response-length` 只决定训练轨迹的 prompt/suffix 容量。
  suffix 可以包含历史回复和工具上下文，并非单次生成长度。
- `--rollout-max-model-len` 决定 vLLM 服务的总上下文容量，默认读取模型配置。
  对默认 Qwen3-8B 为 40960；将训练 suffix 设为 8192 不再把服务上限降到 12288。
- `--rollout-max-tokens` 决定每次在线请求的最大输出长度，默认 2048。
  Q20 的 player、answerer 和可选 search 均显式传入该上限；服务端也会为其他客户端
  补齐缺省预算，并将更大的请求预算限制在该上限内。客户端更小的预算仍然有效。
- 所有在线 chat 请求都在服务端预留输出空间，再由 vLLM 对套用聊天模板后的 token
  序列执行长度限制。超长时默认从左侧截断，保留末尾上下文；这可能丢失早期规则、
  系统提示或任务信息，影响任务质量。显式指定的更短输入上限和截断方向仍被尊重。
  训练 trace 使用服务返回的真实 token IDs，不会把截断前的文本当成模型输入。

例如保留训练 prompt 4096、suffix 8192，可在原命令中使用
`--max-prompt-length 4096 --max-response-length 8192 --rollout-max-tokens 2048`。
如果还需要降低推理上下文容量，可独立设置 `--rollout-max-model-len 12288`；
此时默认每次输入最多保留 10240 tokens，为输出预留 2048 tokens。
服务预算不能保证整段多轮轨迹都落入训练容量；训练转换仍按训练长度截断。

结果 schema 为 3，记录并比较 `rollout_max_model_len`、`rollout_max_tokens` 和 profile
采集配置。baseline/simple 必须使用相同预算与采集配置；报告只接受当前 schema。

带 rollout 标识的 proxy 请求会等待该请求的 trace 写入 Store 后才返回成功，避免
任务已结束但异步 trace 尚未入库的竞态。导出等待默认上限为 30 秒
（`LLMProxy.trace_export_timeout`），失败会返回明确的 trace export 错误。
请求通过上下文显式传递 rollout/attempt/sequence 标识；响应 span 可独立于外部父 span
入库，不依赖 LiteLLM 的请求头序列化格式。若导出超时，`Last trace stage` 会指出
是尚未收到响应回调，还是已进入 Store 写入阶段。
空轨迹诊断会输出 rollout 状态、span 名称及缺失或无效的 token 字段；
`completed rollouts` 包含失败和取消任务，不表示全部生成成功。

若旧版本在首个训练 step 报出 `no trainable transitions` 和
`0 rollouts contained token-bearing triplets`，说明 rollout 已完成，但 LiteLLM trace
没有被转换出非空的 prompt/response token IDs。当前入口会在客户端和本地 vLLM 路由两层
显式请求 `return_token_ids`，并兼容 LiteLLM 将生成 token IDs 放在
`choices[0].provider_specific_fields.token_ids` 的响应形态。更新代码后应使用新的
`--output-dir` 重新运行；失败目录不会被续跑。

GPU 运行只需把 `--device npu` 改成 `--device gpu`。无硬件检查配置时显式使用
`--device gpu|npu --dry-run`。NPU 正式运行要求项目固定的 CANN 9.0.0、
torch/torch-npu 2.10.0、vLLM 0.22.1、vllm-ascend 0.22.1rc1 和 VERL 0.9.0
软件栈。

`--output-dir` 省略时，入口会在仓库 `.cache` 下创建带时间和随机标识的新目录，
并在设备初始化前打印绝对路径、写入 `invocation.json`。`--model Qwen3-8B`
优先指向当前目录中的同名模型目录；不存在时解析为 `Qwen/Qwen3-8B`。
GPU/NPU 都在读取模型配置前通过同一个 `materialize_model` 准备本地权重，默认缓存位于
`.cache/multiturn-artifacts/models`，可用 `--download-dir` 更改；逻辑服务名与权重路径分离。
两侧共用 Q20 Agent、环境模型服务、trace 转换、轨迹聚合、GRPO 和 PrefixGrouper FSDP worker，
设备差异保留在运行时、通信和底层 attention 算子。

## 算子 Profile

在现有训练命令后追加 `--profile`，默认采集第 1 个完整训练步、所有 rank，输出到
`<output-dir>/profile`。例如追加：

```text
--profile --profile-steps 2 3 --profile-memory
```

这会采集第 2、3 步，因此 `--steps` 至少为 3。`--profile-dir /absolute/path`
可指定输出目录；baseline/simple 应分别使用独立目录。`--profile-record-shapes`
默认开启，可用 `--no-profile-record-shapes` 关闭；`--profile-memory` 默认关闭。
这些开关沿用 `benchmark_prefix_grouper.py` 的命名，但这里采集真实在线训练步。
2WikiMQA E2E 脚本目前没有独立的 profile CLI 开关。

采集复用生产 `AgentLightningTrainer` 的 VERL profiler 生命周期：worker 在选定步
开始前启动、actor update 后停止，覆盖 old/reference log-prob 与 actor 前向、反向和更新；
独立的 vLLM 服务在该步 rollout 期间启动／停止采集。验证与 checkpoint 保存不在范围内。
Agent runner 和外部工具进程不会因此生成完整 CPU 算子轨迹；其端到端耗时仍由
`metrics.jsonl` 的阶段指标记录。Q20 的环境模型与 player 共用服务，因此 rollout
profile 也包含 answerer/search 的模型计算。

GPU 使用 torch profiler，导出可供 Perfetto 等工具查看的 Chrome trace；NPU 使用固定栈的
原生 profiler，训练 worker 为 Level1。与 `benchmark_prefix_grouper.py` 一样，NPU
训练 worker 先保存原始数据，训练完成后复用 `AcceleratorRuntime.analyse_profiles`
统一导出 Text 和 Db；解析不计入 `wall_seconds`。actor、ref、rollout 的输出分别位于
profile 根目录下的同名子目录。这里由 VERL 管理分布式 profiler，文件命名和低层
采集选项遵循 VERL；没有复制微基准的手动 profiler 上下文或额外重跑一个训练 step。
`launch.json` 保存具体 profiler 配置，最终 run 记录保存采集选项和绝对输出路径。
采集会引入额外开销，正式速度测量应关闭 profile；报告拒绝比较采集配置不同的两次运行。

每个输出目录包含：

- `launch.json`：完整合并前的训练配置、软件栈和命令；
- `selected_tasks.json`：本次确定性选中的任务；
- `dataset_metadata.json`：数据文件、校验和及 SQL 数据库校验和；
- `metrics.jsonl`：每个训练 step 的阶段耗时、吞吐、显存和 reward，以及最终 run 记录；
- `profile/`：启用 `--profile` 且未指定其他目录时的各 worker 算子轨迹；
- `web_mcp.log`：仅自动启动 Web MCP 时生成。

两种模式完成后，使用统一报告入口生成严格可比的 JSON 和 Markdown：

```bash
python scripts/report_multiturn_online_e2e.py \
  --baseline-dir /runs/sql-baseline \
  --simple-dir /runs/sql-simple \
  --output-dir /runs/sql-report
```

报告会先检查任务、数据校验和、模型、软件栈、seed、batch、rollout、step 和设备形状
完全一致；任一受控字段不一致都会拒绝计算 speedup。steady-state 统计固定排除 step 1。

Qwen3-8B 是 dense 模型；该入口不会向 vLLM 注入 expert parallel，因此不会触发
“Number of experts in the model must be greater than 0”这一 MoE 专用配置错误。
