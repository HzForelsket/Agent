# SQL / 20 Questions 原流程轨迹采集

本入口直接调用仓库现有 `spider.LitSQLAgent` 与 `tinker.TwentyQuestionsFlow`。
不修改原有源码、提示词、历史重建、LangGraph/CrewAI 节点、SQL 检查改写流程或 Q20 的 20 轮终止规则。
`trace_workflows.py` 只在外围配置模型地址、安排原流程运行线程并保存最终状态。
模型请求经过记录代理，增加真实 token ID 返回和采样 seed；不会拼接或改写 messages。
SQL 仍使用临时数据库副本和原 Spider evaluator，默认最多 3 次 SQL 生成/改写。
Q20 不将游戏轮数改成采集器的 max-model-calls；后者只是外层失败保护，默认每个角色 128 次调用。

## 安装

在 NPU 主机运行采集器的客户端 Python 环境，进入 `agent-lightning/examples/rag`：

```bash
python -m pip install -r requirements-workflow-traces.txt \
  --index-url https://pypi.org/simple \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

原 SQL 示例使用 LangChain 0.3 / LangGraph 0.6 接口；Q20 使用仓库 lock 中的 CrewAI 1.2.0。
不用安装 Tinker 服务端或启动训练。LiteLLM 只用于模型客户端，不安装其 proxy extra；
代理由本项目 FastAPI/uvicorn 实现，避免 LiteLLM proxy 与 OpenAI Agents SDK 的 websockets 依赖冲突。
以上客户端依赖应与 NPU vLLM 服务环境分开；不替换 NPU torch/CANN。
服务环境仍是 CANN 9.0.0、vLLM 0.22.1、vllm-ascend 0.22.1rc1。

## NPU 采集命令

替换下面的本地 30B BF16 模型路径与芯片 ID。若服务使用独立解释器，在每条采集命令追加
`--vllm-python /实际NPU环境/bin/python`。先加载该机器部署所需的 CANN 环境变量。

```bash
python collect_traces.py \
  --agent sql \
  --model-path /实际模型路径/Qwen3-30B-A3B-Instruct-2507 \
  --npu-devices 0,1,2,3 \
  --tasks 32 --rollouts-per-task 4 --concurrency 4 \
  --trajectory-timeout 3600 \
  --insecure-download \
  --output traces/sql-30b-run01

python collect_traces.py \
  --agent q20 \
  --model-path /实际模型路径/Qwen3-30B-A3B-Instruct-2507 \
  --npu-devices 0,1,2,3 \
  --tasks 32 --rollouts-per-task 4 --concurrency 4 \
  --trajectory-timeout 3600 \
  --output traces/q20-30b-run01
```

每条命令自动启动 vLLM、等待就绪、采集并关闭本次服务，然后生成各自的 `analysis/report.md`、`benefit.csv`。
SQL/Q20 不需要 Wikipedia 检索，所以不启动 RAG MCP 或加载 BGE。
原流程含同步模型调用；它们在线程中执行，让同一 worker 内的 HTTP 记录代理继续处理请求。
每个 worker 仍为独立进程，服务生命周期及日志清理沿用现有采集器。
采集器启动的子进程使用 `/dev/null` 作为标准输入，不接受终端交互。
CrewAI 1.2.0 的首次 trace 查看提示即使关闭 tracing 仍可能启动读取 stdin 的后台线程；
让该提示读到 EOF，避免线程一直阻塞到 Python 退出时触发 `_enter_buffered_busy` / SIGABRT。
SQL 固定 max_tokens=2048；`--temperature` 用于 SQL/RAG。Q20 保留原 CrewLLM 默认采样配置，实际参数见 calls.jsonl。
Q20 的 Player、Answerer、可选 Search 均配置为本次本地 30B 服务；这是本次模型配置，
不表示与原示例默认的云端 Answerer 模型有相同质量。Answerer 保留原结构化输出和 reasoning_effort 配置。
`--q20-search` 打开原示例已有的可选模拟搜索工具，默认关闭；其输出仍由模型生成，并非真实联网搜索。

### 轨迹超时排查

`--trajectory-timeout` 默认 900 秒，限制的是**单条完整轨迹**，包含所有模型请求、SQL 执行和最终评分，
不是单次请求，也不是整次采集的时长。上面的 SQL/Q20 命令显式使用 3600 秒；这只是放宽预算，
不能解决 SQL 或评分卡住的问题。

若日志出现 `asyncio.to_thread` 的 `CancelledError`，接着是 `wait_for` 的 `TimeoutError`，
表示采集器等到轨迹期限后取消了等待。`calls=6` 是收到的 policy 请求数，不保证六次都有有效回复；
SQL 默认三轮生成/改写各带一次检查，正常情况下也可能有六次模型调用。

超时记录会在 `trajectories.jsonl` 保存耗时、配置上限及有效回复数，worker 日志还会打印所有线程栈。
结合该轨迹在 `calls.jsonl` 的 `started_at/finished_at`、`http_status`、`error` 判断：

- 请求耗时已接近总预算：查看 `vllm.log` 的排队和推理情况，再考虑增加轨迹上限或降低并发。
- 模型调用早已结束：查看线程栈是否停在 SQL 执行、结果读取或 Spider evaluator；单纯增大上限可能只会延后失败。
- 已收到请求却没有对应的落盘调用记录：可能仍在请求中，记录是在请求结束时写入的。

取消协程不能终止 SQL/Q20 的同步工作线程，因此超时后 worker 会先保存失败轨迹和继续位置，再以专用退出码退出。
父进程只清理该 worker 的进程组，重建 worker 后从它负责的下一条轨迹继续；其他 worker 和模型服务继续运行。
超时轨迹不自动重试，已完成轨迹不重复采集。每次替换记录在 `worker_restarts.jsonl`，原 worker 日志追加保留。
`worker-N-resume.json` 是本次运行内部重建 worker 使用的检查点，不是跨运行恢复入口；`--output` 仍须为新目录。

全部剩余轨迹处理后照常分析，`completion.json` 保存超时次数、轨迹状态计数，并用 `partial` 标识存在未完成或截断轨迹。
超时轨迹仍为失败，其所属题组不会进入完整轨迹统计。模型服务退出、worker 非超时崩溃或继续位置无效，
仍会终止采集；已有有效调用会保留，并写入 `failure.json`。已有轨迹可用下文的离线分析命令重算，
只有同题所有 rollout 齐全且有效的组会纳入统计。

## 自动数据准备与缓存

- SQL 默认下载原 SQL 文档指定的 Spider 数据包，使用 `train_spider.parquet` 和 `database/`，不混入 `test_database/`。
  缓存在仓库 `data/cache/sql/`；下载只使用标准库，支持 `--insecure-download`，检查 ZIP 格式和 CRC，记录实际 SHA-256。
  下载源没有预先约定的可信 SHA-256，记录的摘要用于标识本次数据，不冒充上游签名验证。
- Q20 使用仓库 `examples/tinker/q20_nouns.csv`（200 行），复制并校验到 `data/cache/q20/`，不需要联网。
- 保存实际抽样题目、种子、题目文件摘要及所选 SQL 数据库摘要。所有缓存被 Git 忽略。
- 自定义 SQL 数据使用 `--dataset /路径/train_spider.parquet --sql-database-dir /路径/spider`，
  后者应包含 `database/DB_ID/DB_ID.sqlite`，也可直接指向 `database/`。
  parquet 必须包含 `question/query/db_id`；可带 `id`，否则使用源行号生成稳定 ID。
- 自定义 Q20 数据使用 `--dataset /路径/nouns.csv`，必须有 `answer/category`，可带 `id`。

只准备数据，不加载模型或运行 Agent：

```bash
python trace_tasks.py --agent sql --tasks 32 --insecure-download --output ../../../output/sql-data-preparation
python trace_tasks.py --agent q20 --tasks 32 --output ../../../output/q20-data-preparation
```

准备输出目录必须新建。已有 ZIP 与数据可复用；下载或数据库定位失败会直接报告错误，不替换为虚构数据。

## 保存与统计

`calls.jsonl` 保存 policy（SQL 的模型、Q20 的 Player）实际请求/回复与真实 token ID。
`environment_calls.jsonl` 另存 Q20 Answerer/Search 调用；`events.jsonl` 保存 SQL reward 或 Q20 完整游戏状态。
`trajectories.jsonl` 保存完整轨迹状态及各角色调用次数，worker 日志保留原流程输出。

一条完整轨迹包含多次独立模型上下文，不强行拼接成一条从未被模型处理过的长对话。
只有同题 G 条轨迹全部完成、调用完整且没有长度截断的组进入对照；答案错误但正常结束的轨迹仍保留。
三种方案使用完全相同的题组：

1. 独立基线：每条轨迹内每次真实调用的 `prompt_token_ids + response_token_ids` 长度求和。
2. 简单共享：同题 policy 的第 k 次调用组成一个位置，只共享全组 G 条序列的最长公共前缀。
3. 前缀树：仅在上述同一位置的不同轨迹之间建树，共享包括部分轨迹形成的分支。

每个位置每条轨迹至多贡献一条序列，因此不存在轨迹内部去重，也不跨位置共享。
某条轨迹已经结束、没有第 k 次调用时，按空序列处理：该位置的全组公共前缀为零；树仍可共享其余轨迹的前缀。
调用序号不是 SQL 语义阶段或 Q20 游戏轮次；不会通过改流程强制对齐，亦不声称达到任意配对策略下的最大收益。
causal attention pairs 按每次调用的完整三角注意力结构计算，再汇总；不是实测推理或训练耗时。
环境模型调用单独报告调用数与 token 数，不纳入 policy 共享收益。各轨迹 loss/advantage 权重不能合并。

## 已有轨迹重算与跨任务对比

所有多轮分析使用 [统一入口](../../scripts/MULTITURN_SHARING.md)。下面的 `--view calls` 保留逐调用统计单位；
训练分段及 micro-batch 收益使用默认的 `training` 视图。`--output-dir` 必须指定尚无报告的目录。

```bash
python ../../scripts/analyze_multiturn_sharing.py --view calls --input traces/sql-30b-run01 --output-dir traces/sql-30b-run01/analysis
python ../../scripts/analyze_multiturn_sharing.py --view calls --input traces/q20-30b-run01 --output-dir traces/q20-30b-run01/analysis

python compare_trace_reports.py \
  --inputs traces/sql-30b-run01/analysis traces/q20-30b-run01/analysis \
  --output traces/sql-q20-comparison
```

若旧版本在最后一条轨迹显示 `completed` 后因 stdin 后台线程退出报错，
可直接使用以上分析命令读取保留的原始目录，无需先重新采集。
单个 worker 完成不代表所有 worker 都完成；分析只纳入同题 G 条轨迹齐全且通过校验的组，
缺失或无效组会在报告中排除。保留 `failure.json`，不要把进程退出失败改写为整次采集成功。

汇总输出为 `report.md`、`comparison.csv`、`comparison.json`，列出每个任务的覆盖率、独立基线、简单共享、树共享及额外收益。
也支持只拷贝多上下文分析目录后重算：

```bash
python ../../scripts/analyze_multiturn_sharing.py --view calls --input /已有分析目录 --output-dir /新的分析目录
```

该目录须保留 `summary.json` 与 `call_sequences.jsonl`。离线分析无需模型、NPU、CrewAI、LangChain 或 OpenAI SDK。
导出目录复用原始调用已经通过的校验，不能补回排除的组或重新验证缺失的原始请求。
这些报告的单位与 RAG 的“单完整序列”不同，汇总入口拒绝混合两种单位；不同 Agent 的比例也不构成性能排名。

## 验证范围

已采集的 SQL 原始目录可交给 [GPU/NPU 固定轨迹训练回放入口](../../scripts/SQL_TRACE_REPLAY.md)，
实测独立计算与全组公共前缀简单共享的前向、反向和参数更新。该入口不重新运行 SQL Agent，
不改变本采集流程；需要原始 `events.jsonl` 中的 SQL reward，仅有分析目录不足以训练。

本地 `agent` 环境已完成依赖解析与安装，并通过两个原流程的导入检查（SQL: LangChain 0.3.27、LangGraph 0.6.11、langchain-openai 0.3.35；Q20: CrewAI 1.2.0）。
正式数据准备入口分别抽取 2 个真实任务，验证了 Spider 7,000 行题目及所选数据库路径/摘要、Q20 200 行词表缓存。
CLI 帮助、语法、格式和原流程源码未修改的检查通过。没有可用 NPU，尚未运行 SQL/Q20 的 NPU 模型调用，
也没有新任务的实际收益数字。运行上述 NPU 命令后，报告才会包含本次真实结果。
原 SQL 与 Q20 源码保持不变；具体提交版本及源码摘要随采集目录保存。
