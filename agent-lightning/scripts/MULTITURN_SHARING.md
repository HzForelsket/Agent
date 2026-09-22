# 统一多轮共享分析

唯一入口是 `scripts/analyze_multiturn_sharing.py`。默认分析 PrefixGrouper 训练分段和
micro-batch 共享潜力；其他统计单位通过 `--view` 明确选择。分析仅使用 Python 标准库，
不加载模型、不连接服务、不运行训练。

## 训练分析（默认）

从仓库根目录运行：

```bash
conda run -n agent --no-capture-output python scripts/analyze_multiturn_sharing.py \
  --input sql=/runs/sql/calls.jsonl \
  --input q20=/runs/q20/calls.jsonl \
  --micro-batch-sizes 1,2,4,8,16,32,64 \
  --output-dir /runs/multiturn-sharing
```

单个输入可以省略标签，也可传包含 `calls.jsonl` 的目录。多个输入的标签必须唯一；
每个输入独立统计，不跨工作负载共享或混合百分比。

每条记录需要：

- `trajectory_id` 或 `rollout_id`，标识一条轨迹。
- `data_id` 或 `task_id`，标识同题采样组。`--group-key auto` 优先使用 `data_id`；
  也可以显式指定 `--group-key task_id` 或 `data_id`。
- 实际 prompt/response token ID，支持 `prompt_token_ids` / `response_token_ids`、
  `prompt_ids` / `response_ids`、`prompt.token_ids` / `response.token_ids`。
- 可选 `turn` / `turn_index`；缺省按文件顺序排列。同一轨迹不能有重复轮次。
- 可选 `role`。默认保留 policy 以及未标注角色的记录；`--role` 可指定其他角色。

训练视图纳入输入中的 token 调用，**不验证轨迹完成状态、finish reason 或采样组覆盖率**。
需要完整组对照时使用 calls/trajectory 视图及采集元数据，或先提供已经筛选的 token 调用。

## 指标和输出

训练重建以相邻完整 token 上下文的前缀连续性为准；前缀中断则拆为独立 segment。
每段的第一轮 prompt 是训练 prompt，之后的模型输出和工具/环境新增 token 都属于 response suffix。
不模拟训练长度截断，也不把没有 policy loss 的上下文 token 从工作量中删除。

| 字段 | 含义 |
|---|---|
| `token_reduction` | 可消除的重复 prompt token / 独立训练总 token，衡量省掉的工作量 |
| `weighted_token_reduction` | 同一工作负载、同一 micro-batch 大小下，所有 task 的可省 token 总和 / 独立 token 总和 |
| `shared_prompt_fraction` | 每条逻辑轨迹可共享的初始 prompt 长度 / 最终轨迹长度；是占比，不是节省率 |
| `training_shared_prompt_fraction` | 可共享的训练 prompt 长度 / 所有训练 segments 的总长度 |
| `shareable_prompt_fraction` | 属于重复 prompt 组的 prompt token 出现次数 / 所有 prompt token 出现次数 |
| `prompt_deduplication_ratio` | 可省 prompt token / 所有 prompt token |
| `token_work_ratio` | 独立 token 工作量 / 去重后的 token 工作量，不是实测加速比 |
| `prefix_breaks` / `training_segments` | 轨迹的前缀中断次数 / 训练分段数 |

同 task 内相同 prompt 出现 n 次、长度 P、micro-batch 大小 B 时，可省 token 上界为：

`(n - ceil(n / B)) * P`

该公式让每个相同 prompt 组独立对齐 batch 边界；**实际装箱偏移、DP rank 拆分和截断可能降低收益**。
共享仅限相同 task 内完全相同的训练 prompt，不对生成后缀建树。`summary.json` 中
每个工作负载的 `prefix_sharing` 是不受 batch 大小约束的潜力，`per_task` 和
`task_distribution` 才是指定 micro-batch 大小下的上界。

输出目录包含：

- `report.md`：各 task、各 micro-batch 的收益、分布和轨迹构成。
- `summary.json`：所有工作负载的 API/训练 token 比例、共享潜力、逐轨迹指标、逐 task 指标及定义。
- `per_trajectory.csv`：轮数、初始 prompt、最终轨迹长度、训练分段及 prompt 占比。
- `per_task_microbatch.csv`：逐 task、逐 micro-batch 的 token 节省和长度统计。
- `task_distribution.csv`：task 分布的均值、分位数、零共享数和 token 加权比例。

## 其他统计视图

以下视图接受一个原始采集目录或本视图导出的分析目录，要求同题采样组完整；
原始输入验证调用连续性、token ID、完成状态和无长度截断。
导出输入复用之前的筛选结果，不能恢复被排除的题组或重新验证缺失的原始请求。

```bash
# 原工作流逐调用：同题不同轨迹的第 k 次 policy 调用对齐。
conda run -n agent --no-capture-output python scripts/analyze_multiturn_sharing.py \
  --view calls --input /runs/sql --output-dir /runs/sql/analysis

# 完整历史轨迹：最终 prompt + 最终 response，每条轨迹一条序列。
conda run -n agent --no-capture-output python scripts/analyze_multiturn_sharing.py \
  --view trajectory --input /runs/rag --output-dir /runs/rag/analysis

# 从已有完整序列导出重算到新目录。
conda run -n agent --no-capture-output python scripts/analyze_multiturn_sharing.py \
  --view trajectory --input /runs/rag/analysis --output-dir /runs/rag/reanalysis
```

calls 视图只在相同调用序号之间共享，不跨序号或在轨迹内去重；缺少某次调用的轨迹作为空序列，
使全组公共前缀为零。环境模型调用另计。trajectory 视图要求逐轮保留原有消息和模型动作，
使用最后一次完整历史 prompt 加最终输出，不累加每轮重复历史。

这两个视图共用公共前缀、前缀树和因果注意力对计算，比较独立执行、全组公共前缀、前缀树三种方案。
`tree_over_simple` 是树相对于简单共享的额外节省，分母是简单共享工作量。
它们不是当前训练 exact-prompt 分组和实际 micro-batch 行为的模拟。

输出统一为 `summary.json`、`per_task.csv`、`benefit.csv`、`report.md`，以及对应视图的
`call_sequences.jsonl` 或 `trajectory_sequences.jsonl`。
SQL/Q20 报告仍可由 `examples/rag/compare_trace_reports.py` 汇总。

所有视图必须显式提供 `--output-dir`，不能覆盖原始采集目录或已有报告。
calls/trajectory 视图不接受训练专用的 `--role`、`--group-key`、`--micro-batch-sizes`。
这些指标是结构工作量估算，不是训练耗时、显存或数值等价性的实测结果。
