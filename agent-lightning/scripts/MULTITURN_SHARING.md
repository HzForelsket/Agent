# 统一多轮共享分析

唯一入口是 `scripts/analyze_multiturn_sharing.py`。默认视图（`--view training`）按
**采集 rollout 数量**比较轨迹长度和初始 prompt 共享潜力，不按训练 micro-batch 大小统计。
其他统计单位通过 `--view` 明确选择。分析仅使用 Python 标准库，不加载模型或运行训练。

## 按 rollout 数量分析（默认）

从仓库根目录运行：

```bash
conda run -n agent --no-capture-output python scripts/analyze_multiturn_sharing.py \
  --input sql=/runs/sql/calls.jsonl \
  --input q20=/runs/q20/calls.jsonl \
  --rollout-counts 1,2,4,8,16,32,64 \
  --output-dir /runs/multiturn-sharing
```

例如每个 task 采集了 64 条轨迹，七档分别分析前 1、2、4、8、16、32、64 条。
同一个 task 只确定一次顺序，大档包含小档所选的全部轨迹；不会每档重新随机抽样。
编号完整且唯一时按 `sample_index`（或 `rollout_index`）升序；否则按轨迹在输入文件中
首次出现的顺序。输出保存实际选择顺序和每档 rollout ID，可复核选样。

每一档的所有统计都针对该档实际选中的 N 条轨迹重新计算：

`独立 token = sum(这 N 条 rollout 的最终轨迹长度) = N × 最终轨迹长度均值`

均值取未四舍五入的值。每条轨迹的最终长度是最后一次调用的 prompt + response token 数。
**不累计该轨迹所有调用的上下文长度，也不累计全部训练分段。**
如果这 N 条轨迹的均值都是 1396 token，则 N=1 时独立 token 是 1396，N=4 时是 5584。
真实数据的均值可以随选中的轨迹改变。

不足 N 条的 task 跳过该档并记录在 `skipped_cohorts` 中，不复制样本补齐，不用较少样本冒充 N 条。
没有任何可用 task 的档位显示共享率 N/A；报告同时显示各档纳入和数量不足的 task 数。
多 task 数据的档位覆盖率可能不同，不能只比较汇总百分比。

单个输入可省略标签，也可传包含 `calls.jsonl` 的目录。多个输入标签必须唯一，独立统计。
每条记录需要轨迹 ID（`trajectory_id` / `rollout_id`）、题组 ID（`data_id` / `task_id`），
以及实际 prompt/response token ID。支持 `prompt_token_ids` / `response_token_ids`、
`prompt_ids` / `response_ids`、`prompt.token_ids` / `response.token_ids` 三种 token 字段形式。
`--group-key auto` 优先使用 `data_id`，也可指定 `task_id` 或 `data_id`。
`turn` / `turn_index` 可选，缺省按文件顺序；同一轨迹的轮次不能重复。
默认保留 policy 和未标注角色的调用；可用 `--role` 指定角色。

默认视图纳入提供的 token 调用，不验证轨迹正常完成、finish reason 或完整采样组覆盖率。
缺失的早期历史不能从最后一次上下文恢复；有前缀中断的轨迹会在输出中记录。

## 共享口径和输出

在所选 N 条 rollout 中，对完全相同的初始 prompt 分组，每组只保留一份 prompt：

`可省 token = sum((组内 rollout 数 - 1) × 相同初始 prompt 长度)`

只有最终上下文仍以该初始 prompt 开头的 rollout 才参与此共享；历史重写后已不存在的
prompt 不能从最终轨迹总长度中扣除。后缀独立，不对生成输出建树，也不跨 task 共享。
这是对所选轨迹的结构共享潜力估算，不模拟实际训练分段、micro-batch、DP rank 或截断。

| 字段 | 含义 |
|---|---|
| `rollout_count` | 当前 task 当前档位实际选择的 rollout 数量 N |
| `available_rollouts` | 当前 task 采集到的 rollout 总数 |
| `selected_rollout_ids` | 当前档位所选轨迹的 ID，按实际选择顺序列出 |
| `independent_total_tokens` | 所选 N 条轨迹的最终长度之和 |
| `reducible_duplicate_prompt_tokens` | 所选轨迹中可消除的重复初始 prompt token |
| `grouped_total_tokens` | 独立 token 减去可省 token |
| `token_reduction` | 可省 token / 独立 token |
| `weighted_token_reduction` | 相同 N 下所有纳入 task 的可省 token 总和 / 独立 token 总和 |
| `initial_prompt_preserved_rollouts` | 最终上下文仍保留初始 prompt 前缀的 rollout 数 |
| `prefix_breaks` | 所选轨迹的相邻调用发生 token 前缀中断的总次数 |
| `token_work_ratio` | 独立 token / 共享后 token，不是实测加速比 |

输出包括：

- `report.md`：每 task、每 rollout 数量档位的长度、共享率和覆盖率。
- `summary.json`：完整计算结果、选样 ID、跳过的档位及定义。
- `per_trajectory.csv`：所有已采集轨迹的编号、采集顺序、轮数、初始 prompt 和最终长度。
- `per_task_rollout_counts.csv`：每个 task 在各 rollout 数量下的统计。
- `task_distribution.csv`：各档位的 task 分布、覆盖率和 token 加权共享率。

原有 API token 比例、训练分段重建和分段共享潜力保留在
`workloads.<工作负载>.training_segment_diagnostics`，仅供解释训练数据结构。
该诊断针对全部输入，**不是 rollout 数量表的统计分母**。

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
calls/trajectory 视图不接受默认视图专用的 `--role`、`--group-key`、`--rollout-counts`。
这些指标是结构工作量估算，不是训练耗时、显存或数值等价性的实测结果。
