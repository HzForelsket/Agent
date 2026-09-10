# Agent Lightning 接入 PrefixGrouper 的 GRPO 训练效率提升报告

## 1. GRPO 中的重复前缀计算

当前项目使用 Agent Lightning + VERL 执行 GRPO 训练：同一问题采样多个回答，根据组内奖励计算优势，再更新策略。多个回答共享相同的 prompt，但常规训练路径仍将每条 `prompt + response` 作为独立序列处理，重复计算相同前缀。

PrefixGrouper 利用这一重复结构，在同一 micro-batch 内将完全相同的 prompt 合并为一份，各 response 继续独立计算。优化覆盖 rollout 之后的 **old log-prob、reference log-prob 计算和 Actor 更新**，保留 GRPO 原有的奖励、优势和损失公式。当前 vLLM rollout 生成链路未接入 PrefixGrouper。

## 2. PrefixGrouper 的效率提升来源

| 当前实现 | 减少的开销或发挥的作用 |
|---|---|
| 一份 prompt 对应多条 response 的紧凑输入 | 前缀的投影、MLP、归一化等按 token 运算只处理一份，减少重复计算及相关激活存储需求 |
| 前缀与回答分开计算 Attention | 前缀自身的因果 Attention 只计算一次；每条回答仍访问完整前缀和自身历史 token |
| 共享前缀计算图 | Actor 反向传播时，各回答对前缀的梯度汇总到同一计算图，减少重复前缀的反向计算 |
| 按 prompt 重排 batch，并结合保组切分 | 尽量让同组回答进入同一 micro-batch，使前缀共享实际生效 |

设一个 micro-batch 内同一 prompt 对应 `G` 条回答，prompt 长度为 `P`，每条回答长度为 `R`。忽略 padding 时，逻辑上需处理的非重复 token 位置从 `G × (P + R)` 减少到 `P + G × R`，省去 `(G − 1) × P` 个重复前缀位置。

前缀越长、同一 micro-batch 内可共享的回答越多，可消除的重复工作越多。但上述工作量减少不等于实际加速比或显存降幅：分组、张量重排、Attention 拆分及中间张量仍有开销，端到端收益还受 rollout 等其他阶段耗时占比影响。

## 3. 具体提升报告

**对比对象：**同一版本 Agent Lightning + VERL，基线使用未接入 PrefixGrouper 的标准路径，优化组使用当前 PrefixGrouper 接入路径。

**实验配置（待填）：**【模型及精度】、【硬件及卡数】、【软件版本及代码提交】、【prompt/response 实际长度】、【组大小 G】、【micro-batch 大小】。两组保持相同数据、采样配置、训练批量和计时范围。

| 指标 | 未接入 PrefixGrouper | 接入 PrefixGrouper | 变化 |
|---|---:|---:|---:|
| Old log-prob 耗时（s） | 待填 | 待填 | 【】倍加速 |
| Reference log-prob 耗时（s，如启用） | 待填 | 待填 | 【】倍加速 |
| Actor 更新耗时（s，含前向与反向） | 待填 | 待填 | 【】倍加速 |
| 完整训练轮次耗时（s，含 rollout） | 待填 | 待填 | 【】倍加速 |
| 端到端吞吐（回答数/s） | 待填 | 待填 | 【】% |
| 单卡峰值显存（GB，同一测量范围） | 待填 | 待填 | 【】% |

加速比 = 未接入耗时 / 接入后耗时；吞吐按相同回答工作量除以完整训练轮次的墙钟时间计算。仅有局部前向或反向数据时，只报告对应阶段收益。数值与训练效果验证结果：【待填】。

**实测结论（待数据补全）：** 在【配置】下，接入 PrefixGrouper 后，Actor 更新加速比为【】倍，完整训练轮次加速比为【】倍，峰值显存变化为【】%。分阶段数据表明，主要收益来自【阶段及证据】。

实现依据：[PrefixGrouper 接入代码](/home/huangzhong/Agent/agent-lightning/agentlightning/verl/prefix_grouper.py)、[训练流程](/home/huangzhong/Agent/agent-lightning/agentlightning/verl/trainer.py)。当前为基于源码整理的报告框架，实测提升待补充。
