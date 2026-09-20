# 多任务多轮在线端到端训练

正式入口为 `scripts/benchmark_multiturn_online_e2e.py`，统一支持：

- `--task sql`：Spider SQL 的生成、执行、检查和改写流程；
- `--task q20`：20 Questions 的 CrewAI 多轮流程，只训练 player，固定 answerer 不进入训练轨迹；
- `--task web`：带 Wikipedia MCP 检索工具的多轮 Web RAG 流程。

每次运行只选择一个任务和一个模式。`baseline` 使用标准 VERL 前向；`simple`
启用 PrefixGrouper，只共享同题 4 条 rollout 完全相同的初始 prompt，后续多轮 suffix
保持独立。两个模式都使用轨迹级聚合，覆盖实时 rollout、GRPO、old/reference
log-prob 和 actor update。

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

把 `--task sql` 分别替换为 `q20` 和 `web` 即可复用同一训练入口。Q20 的固定
answerer 使用 `OPENAI_API_KEY`，并可通过 `--q20-answerer-base-url` 与
`--q20-answerer-model` 指向任意 OpenAI 兼容服务；密钥不接受 CLI 参数，避免写入
`launch.json`。Web 默认自动启动仓库内的 CPU MCP 检索服务，也可通过
`--web-mcp-url http://host:port/sse` 复用已有服务。

GPU 运行只需把 `--device npu` 改成 `--device gpu`。无硬件检查配置时显式使用
`--device gpu|npu --dry-run`。NPU 正式运行要求项目固定的 CANN 9.0.0、
torch/torch-npu 2.10.0、vLLM 0.22.1、vllm-ascend 0.22.1rc1 和 VERL 0.9.0
软件栈。

每个输出目录包含：

- `launch.json`：完整合并前的训练配置、软件栈和命令；
- `selected_tasks.json`：本次确定性选中的任务；
- `dataset_metadata.json`：数据文件、校验和及 SQL 数据库校验和；
- `metrics.jsonl`：每个训练 step 的阶段耗时、吞吐、显存和 reward，以及最终 run 记录；
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
