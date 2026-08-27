# 共享前缀当前设计报告

##  设计概述

当前方案在 Agent Lightning + VERL 的训练前向中接入 PrefixGrouper，目标是消除同一个 micro-batch 内重复 prompt 的前向计算。

原始 batch 按行交错排列：

```text
row 0: [P1, R1-1]    row 1: [P2, R2-1]    row 2:  [P3, R3-1]
row 3: [P1, R1-2]    row 4: [P2, R2-2]    row 5:  [P3, R3-2]
row 6: [P1, R1-3]    row 7: [P2, R2-3]    row 8:  [P3, R3-3]
row 9: [P1, R1-4]    row 10: [P2, R2-4]   row 11: [P3, R3-4]
```

其中，`R1-2` 表示 prompt `P1` 的第 2 条 response。去重和重排后的物理输入为：

```text
grouped row 0: [P1, R1-1, R1-2, R1-3, R1-4]
grouped row 1: [P2, R2-1, R2-2, R2-3, R2-4]
grouped row 2: [P3, R3-1, R3-2, R3-3, R3-4]
```

但模型保持的逻辑输入仍然是 12 条彼此独立的 decoder 序列：

```text
[P1, R1-1]  [P1, R1-2]  [P1, R1-3]  [P1, R1-4]
[P2, R2-1]  [P2, R2-2]  [P2, R2-3]  [P2, R2-4]
[P3, R3-1]  [P3, R3-2]  [P3, R3-3]  [P3, R3-4]
```

方案只共享完全相同的 prompt。它采用扁平的“一份 prefix + 多份 suffix”分组，不构建前缀树。

##  设计目标

1. 同一 prompt 只计算一次 prefix hidden state。
2. 每条 response 保持原始 decoder causal attention 语义。
3. 保持 response token 的位置编码、log-prob 和 loss 对齐方式不变。
4. 保持多条 response 对共享 prefix 的梯度累积。
5. 上层 PPO/GRPO 继续使用 VERL 原有输出格式，无需感知底层重排。


##  暂不考虑

- 不复用仅部分相同的 prompt。
- 不在不同 micro-batch、不同 data-parallel rank 之间共享 prefix。
- 不构建多级分支树，也不继续去重 response 之间的共同片段。

## 主要修改内容

| 代码位置 | 变更类型 | 主要内容 |
| --- | --- | --- |
| `agent-lightning/pyproject.toml` | 修改 | 增加 `prefix_grouper>=0.0.1.post1` 依赖。 |
| `agent-lightning/agentlightning/verl/config.yaml` | 修改 | 增加总开关，并将开关映射到 VERL 的保组配置。 |
| `agent-lightning/agentlightning/verl/trainer.py` | 修改 | 在训练 batch 切分前按完整 prompt 重排，尽量保留同组 response。 |
| `agent-lightning/agentlightning/verl/entrypoint.py` | 修改 | 开启功能时，将标准 Actor/Reference Worker 切换为 PrefixGrouper Worker。 |
| `agent-lightning/agentlightning/verl/prefix_grouper.py` | 新增 | 接入 prompt 去重、grouped input、position ID、causal attention、输出还原、FSDP `forward_step` 。 |
| `agent-lightning/tests/verl/test_prefix_grouper_verl090.py` | 新增 | 对比标准路径与共享路径的 log-prob、entropy 和参数梯度。 |

<!-- `PrefixGrouper/src/prefix_grouper/` 是未经修改的上游 `0.0.1.post1` 源码，本次只复用其 attention 分解、mask/索引和 autograd 能力，不属于上述修改。 -->

这些修改作用于 rollout 完成后的训练计算；vLLM rollout 生成链路未接入 PrefixGrouper。


<br>
<br>
<br>
<br>
<br>
<br>

##  核心数据结构

###  分组

每个有效 prompt token 序列被转换成 tuple，作为 `OrderedDict` 的 key。key 完全相同的行归入同一组，并保留稳定顺序。

```text
P1 -> rows [0, 3, 6, 9]
P2 -> rows [1, 4, 7, 10]
P3 -> rows [2, 5, 8, 11]
```

每组选第一行的 prompt 作为 prefix representative，同时保留组内 4 行的全部 response。

###  GroupInfo

在统一示例中，三组分别记录：

```text
P1 -> [P1_len, R1-1_len, R1-2_len, R1-3_len, R1-4_len]
P2 -> [P2_len, R2-1_len, R2-2_len, R2-3_len, R2-4_len]
P3 -> [P3_len, R3-1_len, R3-2_len, R3-3_len, R3-4_len]
```

PrefixGrouper 根据这些长度预计算：

- prefix/suffix padding mask；
- prefix/suffix attention mask；
- group/ungroup 索引；
- grouped 和 ungrouped tensor shape。

这些是扁平的张量布局元数据，不是树节点。

##  前向数据流

###  查找重复 prompt

在单个 micro-batch 内按完整 prompt token 去重。如果所有 prompt 都只出现一次，立即返回 VERL 标准前向路径。

###  构造 grouped input

对每组抽取一份 prefix，并将属于同一 prompt 的 response 连续排列。统一示例重排后为：

```text
prefix_ids       = [P1, P2, P3]
ordered_response = [R1-1, R1-2, R1-3, R1-4,
                    R2-1, R2-2, R2-3, R2-4,
                    R3-1, R3-2, R3-3, R3-4]
group_sizes      = [4, 4, 4]
```

这里：

- `P1`、`P2`、`P3` 是去重后保留的三份 prompt；
- `R1-1` 表示 prompt `P1` 的第 1 条 response，`R3-3` 表示 prompt `P3` 的第 3 条 response；
- `group_sizes = [4, 4, 4]` 表示三个 prompt 各自对应 4 条 response。

随后 `concat_input()` 生成每组一行的紧凑物理输入。

###  重建 position ID

每条 response 的 position ID 都从对应的 `prefix_len` 重新开始。设 `P1`、`P2`、`P3` 的有效长度分别为 `p1`、`p2`、`p3`：

```text
P1  : 0, 1, ..., p1-1
R1-1: p1, p1+1, ...    R1-2: p1, p1+1, ...
R1-3: p1, p1+1, ...    R1-4: p1, p1+1, ...

P2  : 0, 1, ..., p2-1
R2-1: p2, p2+1, ...    R2-2: p2, p2+1, ...
R2-3: p2, p2+1, ...    R2-4: p2, p2+1, ...

P3  : 0, 1, ..., p3-1
R3-1: p3, p3+1, ...    R3-2: p3, p3+1, ...
R3-3: p3, p3+1, ...    R3-4: p3, p3+1, ...
```

因此，虽然 `R1-2` 在物理布局中位于 `R1-1` 后面，它的 RoPE 位置仍与独立执行 `[P1, R1-2]` 时相同。

##  Decoder attention 语义

原始路径对统一示例中的 12 条 response 分别计算：

```text
Decoder(P1 + R1-1) ... Decoder(P1 + R1-4)
Decoder(P2 + R2-1) ... Decoder(P2 + R2-4)
Decoder(P3 + R3-1) ... Decoder(P3 + R3-4)
```

共享路径在每个 Transformer 层将 attention 分成两部分。

###  Prefix attention

```text
O_P1 = Attention(Q_P1, K_P1, V_P1, prefix_causal_mask_P1)
O_P2 = Attention(Q_P2, K_P2, V_P2, prefix_causal_mask_P2)
O_P3 = Attention(Q_P3, K_P3, V_P3, prefix_causal_mask_P3)
```

prefix token 只能看到 prefix 中不晚于自身的位置。由于 decoder prefix 本来不能看到后续 response，同一 prompt 的 prefix hidden state 可以只计算一次。

###  Suffix attention

以 `P1` 的第 1 条 response `R1-1` 为例，它独立计算：

```text
O_R1-1 = Attention(
    Q_R1-1,
    [K_P1, K_R1-1],
    [V_P1, V_R1-1],
    suffix_causal_mask_R1-1,
)
```

`R1-2` 至 `R1-4` 分别使用 `P1` 和自身的 K/V；`R2-1` 至 `R2-4` 分别使用 `P2` 和自身的 K/V；`R3-1` 至 `R3-4` 分别使用 `P3` 和自身的 K/V。

mask 保证：

- `R1-1` 可以看到完整的 `P1`；
- `R1-1[j]` 只能看到 `R1-1[0:j]`；
- `R1-1` 不能看到 `R1-2` 至 `R1-4`，也不能看到 `P2`、`P3` 组的任何 token；
- padding token 不参与 attention。

其他 11 条 response 使用相同规则。因此，多个 response 虽然物理上按组拼接，逻辑可见范围仍分别等价于开头列出的 12 条独立 decoder 序列。

###  其他 Transformer 运算

RMSNorm/LayerNorm、MLP、残差连接等操作都按 token 独立执行，不会在 response 分支之间交换信息。唯一会跨 token 混合信息的 attention 已由上述 mask 隔离。

##  输出与 loss 对齐

模型输出 grouped logits 后，`split_output()` 将其拆回逐 response 布局。

自回归模型中，response 第一个 token 由 prefix 最后一个 token 的 logits 预测。因此调用：

```text
split_output(logits, include_prefix_last=1)
```

在统一示例中，`P1` 最后一个位置的 logits 被复制到 `R1-1` 至 `R1-4` 的开头，`P2` 最后一个位置的 logits 被复制到 `R2-1` 至 `R2-4` 的开头，`P3` 最后一个位置的 logits 被复制到 `R3-1` 至 `R3-4` 的开头，再与各自的 completion token 对齐计算 log-prob。最后使用 `inverse_order` 恢复为原始的 row 0 至 row 11 顺序，并转换回 VERL 期望的 nested tensor 布局。

##  反向传播

Group/Ungroup 操作实现了自定义 autograd，prefix K/V 复制使用 PyTorch 可微的 `repeat_interleave`。

因此，每条 response loss 对共享 prefix 的梯度会汇总到同一份 prefix 计算图：

```text
dL/dP1 = dL_R1-1/dP1 + dL_R1-2/dP1 + dL_R1-3/dP1 + dL_R1-4/dP1
dL/dP2 = dL_R2-1/dP2 + dL_R2-2/dP2 + dL_R2-3/dP2 + dL_R2-4/dP2
dL/dP3 = dL_R3-1/dP3 + dL_R3-2/dP3 + dL_R3-3/dP3 + dL_R3-4/dP3
```

在确定性前向条件下，这与原始路径分别计算多份相同 prefix、再由总 loss 汇总梯度的代数结果一致。
