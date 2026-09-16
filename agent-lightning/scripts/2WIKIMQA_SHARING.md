# 2WikiMQA 单轮问答：重新采集与共享统计

继续使用正式入口 `benchmark_prefix_grouper_2wikimqa_e2e.py`，沿用已选定的
模型、数据、设备、步数及每题 rollout 数，指定新的输出目录。每条
`responses.jsonl` 现在直接记录服务端的 `prompt_token_ids`、
`response_token_ids`、实际 token 数和 `finish_reason`。
采集仍是一次模型调用完成一条单轮问答，不改变训练或采样逻辑。

统计脚本只需 Python 标准库，不启动模型。下面的 `G` 必须换成采集时
`--rollouts-per-sample` 的值；输入选择同一次运行的 responses 文件。
独立基线与共享估计使用完全相同的样本，无需再生成一批共享样本。

```bash
conda run -n agent --no-capture-output python scripts/analyze_2wikimqa_sharing.py \
  --input /path/to/run/baseline/responses.jsonl --group-size G \
  --sharing prompt --output /path/to/new-prompt-analysis

conda run -n agent --no-capture-output python scripts/analyze_2wikimqa_sharing.py \
  --input /path/to/run/baseline/responses.jsonl --group-size G \
  --sharing tree --output /path/to/new-tree-analysis
```

- `prompt`：同题完全一致的输入仅算一次，各输出独立计算。
- `tree`：完整输入和输出按精确 token 前缀建树，共享分叉前的节点。
- 基线：各序列 `L = prompt + response`，token 位置为 `ΣL`，
  causal attention pairs 为 `Σ L(L+1)/2`，包含自身注意力。
- 每个共享树节点在深度 d（从 1 开始）贡献 d 个 pairs；不同分支不可互相注意。
- 只纳入同题 G 条齐全的组，排除信息保存在 JSON；重复编号、非法数据直接报错。
  不合并不同运行或重复 epoch 的同题组。
- `finish_reason=length` 的样本仍计入实际工作量，报告单列数量。
  若要观察自然回答长度，采集时不要沿用历史的 64 token 输出上限。
- 不含 padding，不乘模型层数或头数；这些是结构估计，不是训练加速实测值。
  `tree` 结果也不表示当前 PrefixGrouper 已执行完整树共享。

输出 `report.md`、`summary.json`、`per_task.csv`，包含基线和共享数量、
减少比例、平均单条长度、覆盖率、输入摘要及截断情况。输出目录必须不存在。

只有长度而没有 IDs 的记录，只能在用户明确确认同题实际 prompt 完全一致时，
通过 `--sharing prompt --assume-identical-prompts` 分析。必须同时存在准确的
`prompt_tokens`、`response_tokens` 和 `finish_reason`；旧文件缺失这些字段时，
需要重新采集，不能从输出文本或统一输出上限推断。
