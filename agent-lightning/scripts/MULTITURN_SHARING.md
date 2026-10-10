# 统一多轮共享分析

唯一入口是 `scripts/analyze_multiturn_sharing.py`。默认视图（`--view training`）按
**采集 rollout 数量**比较最终轨迹长度和最终轨迹的前缀树共享率，不按训练 micro-batch 大小统计。
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
`--group-key auto` 依次使用 `data_id`、`task_id`、`instance_id`，也可显式指定其中一个。
`turn` / `turn_index` 可选，缺省按文件顺序；同一轨迹的轮次不能重复。
默认保留 policy 和未标注角色的调用；可用 `--role` 指定角色。

默认视图纳入提供的 token 调用，不验证轨迹正常完成、finish reason 或完整采样组覆盖率。
缺失的早期历史不能从最后一次上下文恢复；有前缀中断的轨迹会在输出中记录。

## 共享口径和输出

在所选 N 条 rollout 中，每条序列取最后一次调用的完整 prompt + response token IDs，
对这些最终轨迹构建前缀树，每个树节点只计一次：

`独立 token = sum(所选 N 条最终轨迹长度)`

`共享后 token = 最终轨迹前缀树的节点数（不含根节点）`

`共享率 = (独立 token - 共享后 token) / 独立 token`

不要求初始 prompt 完全相同，也不要求最终轨迹以最初 prompt 开头。共享范围可以包含
工具结果和模型输出；部分 rollout 共同拥有的前缀也会计入。分叉后再次出现的相同片段
不合并，不跨 task 共享。这是最终轨迹的结构共享潜力，不模拟实际训练分段、micro-batch、
DP rank 或截断。`summary.json` 用 `sharing_method=final_trajectory_prefix_tree` 标明口径。

例如最终轨迹为 `[1,2,3,4]`、`[1,2,3,5]`、`[1,2,6]`，独立长度为 11，
前缀树节点数为 6，可省 5，默认共享率为 5/11。若只共享三条共同的 `[1,2]`，
则可省 4，另以 `simple_token_reduction=4/11` 记录，不混入默认共享率。

| 字段 | 含义 |
|---|---|
| `rollout_count` | 当前 task 当前档位实际选择的 rollout 数量 N |
| `available_rollouts` | 当前 task 采集到的 rollout 总数 |
| `selected_rollout_ids` | 当前档位所选轨迹的 ID，按实际选择顺序列出 |
| `independent_total_tokens` | 所选 N 条轨迹的最终长度之和 |
| `shared_prefix_saved_tokens` | 最终轨迹前缀树共享可消除的重复 token 数 |
| `initial_prompt_saved_tokens` | 所有出现位置均位于各自初始 prompt 内的可省 token |
| `other_saved_tokens` | 所有出现位置均位于各自初始 prompt 之后的可省 token |
| `mixed_saved_tokens` | 同一共享节点在不同轨迹中分别属于初始 prompt 和后续部分的可省 token |
| `unattributed_saved_tokens` | 至少一条参与轨迹无法确认初始 prompt 边界的可省 token |
| `grouped_total_tokens` | 独立 token 减去可省 token |
| `token_reduction` | 可省 token / 独立 token |
| `weighted_token_reduction` | 相同 N 下所有纳入 task 的可省 token 总和 / 独立 token 总和 |
| `common_prefix_tokens` | 所选 N 条最终轨迹的全组最长公共前缀长度 |
| `simple_grouped_total_tokens` | 仅将全组最长公共前缀保留一次后的 token 数 |
| `simple_token_reduction` | 仅共享全组最长公共前缀时的节省比例 |
| `sharing_status` | `single_rollout`：只选中一条；`shared`：最终轨迹有可省 token；`no_shared_final_prefix`：最终轨迹之间无共享前缀 |
| `prefix_breaks` | 所选轨迹的相邻调用发生 token 前缀中断的总次数 |
| `token_work_ratio` | 独立 token / 共享后 token，不是实测加速比 |

输出包括：

- `report.md`：每 task、每 rollout 数量档位的长度、共享率和覆盖率。
- `summary.json`：完整计算结果、选样 ID、跳过的档位及定义。
- `per_trajectory.csv`：所有已采集轨迹的编号、采集顺序、轮数、初始 prompt 和最终长度。
- `per_task_rollout_counts.csv`：每个 task 在各 rollout 数量下的统计。
- `task_distribution.csv`：各档位的 task 分布、覆盖率和 token 加权共享率。
- `per_task_round.csv`：每个 task、每档 rollout 数量在每轮交互后的轨迹长度和共享率。
- `round_distribution.csv`：每个 rollout 档位的逐轮汇总、实际参与数量和 token 加权共享率。

报告同时展示每 task 和每 rollout 档位的共享来源。归属统计以**省下的重复 token**为单位：
一个被 k 条轨迹共用的前缀树节点贡献 k−1，不是把 k 个出现位置全部算作节省。
仅当最终轨迹以完整初始 prompt 开头时，才确认该轨迹的初始 prompt 边界。
对于同一共享节点，全部出现位置在边界内归入初始 prompt，全部在边界后归入后续部分；
出现位置归属不一致归入 mixed，任一边界未确认归入 unattributed。四类互斥且合计
等于 `shared_prefix_saved_tokens`；统计不依赖轨迹输入顺序，也不会改变总共享率。
“后续部分”包括后续模型回复、工具结果和消息模板 token，当前不再细分来源。

原有 API token 比例、训练分段重建和分段共享潜力保留在
`workloads.<工作负载>.training_segment_diagnostics`，仅供解释训练数据结构。
该诊断针对全部输入，**不是 rollout 数量表的统计分母**。

汇总表的 task 共享率均值、P50、P95 统计的是各 task 的 `token_reduction`，不是轨迹长度。
N=1 时没有第二条轨迹可以共享，因此这些比例必为零；N>1 但所有 task 的可省 token
均为零时也会全零。查看每 task 的“共享状态”区分原因。没有足够 rollout 的档位显示
N/A，不应解释成共享率为零。比例按两位百分数显示，极小的非零值也可能显示 0.00%；
此时以 `summary.json` 的原始比例和 `shared_prefix_saved_tokens` 为准。

## 每轮交互后的共享率和轨迹均值

默认视图自动输出逐轮统计，无需增加命令行参数。`report.md` 包含逐轮汇总和每 task
明细；JSON 对应 `per_task_round` 和 `round_distribution`，并提供上述两个 CSV。

第 k 轮（从 1 开始）指去重、角色筛选后，按原始轮次排序的第 k 次有效模型调用。
它不是原始 `sequence_id` 数值，也不保证对应一次工具执行或用户对话。
原始序号可能有间隔，缺失的调用无法恢复；CSV/JSON 的 `active_rollout_ids` 与
`source_turn_indices` 按位置对应，保留实际参与轨迹和原始序号以便核查。

每个 task、每个 N 档位沿用最终轨迹统计选中的前 N 条 rollout。第 k 轮只纳入其中
有第 k 次有效调用的轨迹，不重新选样、不复制样本，也不把已停止调用的轨迹末尾填入后续轮次。
没有该轮调用可能是结束，也可能是采集不完整，不能据此认定正常完成。

- 当轮轨迹 = 第 k 次调用的 prompt token IDs + response token IDs。
- 当轮轨迹均值 = 参与轨迹的当轮长度之和 / `active_rollouts`，单位 **token**；不累计历次调用长度。
- 当轮共享率沿用最终轨迹的前缀树算法：`shared_prefix_saved_tokens / independent_total_tokens`。
- 仅剩一条参与轨迹时共享率为零；没有参与轨迹的 task 不生成该轮明细，也不计入该轮 task 均值。
- 共享来源的四类可省 token 同样写入逐轮 CSV/JSON；以当轮上下文是否保留初始 prompt 判断边界。

`rollout_count` 是最初选样档位 N，`active_rollouts` 才是该轮的实际分母数量；
`inactive_rollouts` 表示入选但没有该轮调用的数量。汇总同时提供 `eligible_tasks`、
`active_tasks`、`inactive_tasks` 和 `tasks_with_multiple_rollouts`，后者表示仍有至少两条
轨迹参与、可能产生跨轨迹共享的 task 数。数量不足 N 的 task 仍在 `skipped_cohorts` 中，
不会通过逐轮统计重新纳入。

汇总轨迹均值按全部实际参与 rollout 加权，不直接平均各 task 的长度均值；
汇总共享率按各 task 的可省 token 总和 / 独立 token 总和计算，不跨 task 合并前缀。
`mean`、`p50`、`p95` 则仍是 task 共享率分布，其中包含仅剩一条参与轨迹的零共享率。
各轮参与样本可能不同，比较趋势时必须同时查看覆盖率；若轨迹轮数不同，最后一轮的参与集合
也不同于“每条 rollout 各取最终上下文”的最终轨迹表，两者不要求一致。

## SWE-bench 原始采集结果

接入依据是当前 `examples/claude_code/claude_code_agent.py` 的
`run_instance_async`：它先将 `store.query_spans(rollout.rollout_id)` 返回的 Span
逐行写入 `<output-dir>/stream_<instance_id>.json`。虽然扩展名是 `.json`，内容是
JSONL，不是一个 JSON 数组。分析直接读取这些原始文件，不依赖派生的 HuggingFace
`dataset-<instance_id>/`，也不读取题目输入 `swebench_samples.jsonl` 或 Docker 日志。

在保存采集结果的机器上，从仓库根目录运行（替换实际路径）：

```bash
conda run -n agent --no-capture-output python scripts/analyze_multiturn_sharing.py \
  --input swebench=/runs/swebench-collected \
  --group-key instance_id --rollout-counts 1,2,4,8,16,32,64 \
  --output-dir /runs/swebench-sharing
```

输入可以是单个 `stream_<instance_id>.json`、采集目录，或包含多次独立采集目录的父目录；
目录递归查找 `stream_*.json`，按路径排序。若目录直接包含 `calls.jsonl`，则继续使用
原有 calls 格式，不将两种表示重复计入。同题的多次采集必须放在不同目录：当前采集器的
stream 文件名不含 rollout ID，同目录重跑同题会覆盖旧文件，分析器无法恢复已覆盖轨迹。

| 分析字段 | 当前采集文件中的来源 |
|---|---|
| 题组 | 文件名中的完整 `instance_id` |
| 轨迹 ID | Span 的 `rollout_id`，不从目录名虚构 |
| 轮次顺序 | Span 的 `sequence_id`；同序号按 `start_time` 排序 |
| prompt tokens | `attributes.prompt_token_ids` 或 `attributes.llm.hosted_vllm.prompt_token_ids` |
| response tokens | `attributes.response_token_ids` 或 `attributes.llm.hosted_vllm.response_token_ids`；也支持 raw `choices[0].token_ids` / `provider_specific_fields.token_ids` |
| 请求去重 | 首先按 `(rollout_id, attempt_id, sequence_id)`，同时识别 `gen_ai.response.id` 或 `llm.hosted_vllm.id` |

token 提取规则参照当前 `ExtendedLlmProxyTraceToTriplet`。当前代理为一次请求的整棵
Span 子树写入同一个 `sequence_id`，主 Span 与 raw Span 都可能携带 token，因此同序号
且 token 一致的记录只计一次，不依赖 response ID 是否存在。相同请求身份的 token 内容
不一致时，分析器报出冲突序号及两条记录的位置，不静默选择或改写轮次。序列化的 token
数组会解码，但不会重新分词文本或用 usage token 数替代 token IDs。非 LLM span 不计入；
缺少有效 token 的 LLM span 和重复请求分别计数，写入 `summary.json` 的
`input_diagnostics` 及 `report.md`。某个 stream 完全没有有效 token 调用时明确报错。
同一 rollout 有多个带 token 的 attempt 时拒绝混合，不能把重试当成额外采样。

当前采集入口每次对每个输入问题执行一次 rollout，没有写入 sample index；因此按目录路径
和 Span 顺序确定嵌套选样顺序，并在结果中保留实际 rollout ID。单次采集通常只有 N=1
档位可用；更高档位需要已保存的独立同题采样，不会复制轨迹补齐。

此接入只用于默认 `--view training`，复用现有最终上下文长度、最终轨迹前缀树共享和
训练分段诊断算法。原始 Span 不提供现有 calls/trajectory 视图要求的完整采样清单与
完成状态合同，因此不冒充这两个视图的完整输入。未标注角色的 token 调用会全部纳入，
不能据此认定它们都是 policy 调用；高低价模型应使用相同的实际模型和 tokenizer，
不同模型的 token IDs 不能放在同一个共享比较中。输入缺失调用、历史压缩或重写时，
报告仅反映可见 token 序列，不证明完整轨迹、任务成功率或实测训练收益。

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
