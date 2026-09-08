# 前向 Host 耗时定位计划与回传清单

目标：定位 `pg-ascend-shared-prefix-attention`（Ascend 共享前缀注意力算子微基准）
中 custom 与 K/V 展开后官方 fusion attention 的主机耗时差异。
当前只分析前向，保持原有数学计算、核间分配、输入和异步提交方式。

## 已知数据与未知项

用户最近提供的数据（μs）：prefix=16、suffixes=[8,8]、Hq=12、Hkv=2、D=128，warmup=5。
下面是完成 scale 缓存和 Pack/LSE 合并后的基线；本轮入口优化尚未实机复测：

| 事件 | custom | fusion/permute | 口径 |
|---|---:|---:|---|
| 未开启 profiler 的 median | 555.645 | 312.305 | 正常前向耗时 |
| 外层 forward | 1768.72 | 1380.79 | Host Total，含最后同步 |
| 最后同步 | 217.05 | 172.84 | Host Total |
| python_validate | 207.68 | 不适用 | Host Total |
| python_scale | 49.81 | 不适用 | Host Total |
| load_extension | 40.73 | 不适用 | 旧入口的加载状态检查 |
| autograd_apply | 974.91 | 不适用 | 旧入口 Total，Self=159.09 |
| PyTorch C++ 算子事件 | 616.33 | 未提供 | Host Total |
| PyTorch C++ 算子 Self | 26.68 | 未提供 | Host Self |
| allocate | 181.13 | 无独立标记 | Host Total |
| attention_bridge | 403.48 | 376.39 | Host Total，标记处层级不同 |
| attention_bridge Self | 385.02 | 151.07 | Host Self，子事件覆盖不同 |
| gather_k + gather_v | 不适用 | 574.19 | 按外层 Total/Self、同步及桥接推算 |

Launch 总时间接近，不能证明参数准备、缓存行为、执行队列或设备计算耗时相同。
旧入口中 autograd_apply 内、C++ 算子外的区间为 974.91−616.33=358.58 μs。
本轮使用 torch.library.register_autograd 注册梯度，前向直接调用缓存的算子 overload。
PyTorch 在 no-grad 或没有输入需要梯度时跳过 Function.apply 和 setup_context；
需要梯度时保存同样的状态，继续调用原有反向算子。没有为 benchmark 单独绕过梯度的实现。
输出仍是两个独立张量，改用 torch-npu 的 apply_tensor_without_format 分配；
未开启探针时复用无状态 nullcontext，减少 Python 临时对象。
正常计时的收益必须实机重测，不能把上述 profile 阶段耗时直接当成可节省的时间。

## 分析顺序

1. 确认构建完成、Python 环境与 wheel 路径正确、已安装二进制哈希已记录。
2. 同输入先看未开启 profiler 的 mean/median，确认问题在正常计时中存在。
3. 看下面互不重叠的 C++ 阶段 Total，先找主要贡献者。
4. Attention 桥接若占主导，在 trace 中关联 CANN API 和工作线程，检查
   workspace/executor 准备、tiling、任务提交或等待；不只看 Launch。
5. 对照官方 fusion 桥接，以及基线单独的 gather，避免用 custom 整条前向对比 fusion 单行。
6. 只优化实测占主导的阶段；复测时保持输入、预热、采样和采集配置一致。

## 已实现的探针

以下名称在 `trace_view.json` 和 `operator_details.csv` 中搜索。

| 探针名称 | 范围 | 重点看 |
|---|---|---|
| `pg_host/custom/python_validate` | NPU dispatch、Q rank、plan token 数 | Total |
| `pg_host/custom/python_scale` | 默认 FP32 scale 的缓存查询／显式标量转换 | Total |
| `pg_host/custom/dispatch` | 缓存入口查询与已注册算子调用 | Self 与 Total，不能加到子阶段上 |
| `pg_host/custom/cpp_validate` | C++ 参数检查 | Total |
| `pg_host/custom/allocate` | 通过 torch-npu 原生接口分配最终 out、lse | Total 与内部子事件 |
| `pg_host/custom/attention_bridge` | 整次 EXEC_NPU_CMD_EXT 前向调用 | Total 与对应 CANN/工作线程 |
| `pg_host/fusion/gather_k` | K index_select | Total 与设备任务 |
| `pg_host/fusion/gather_v` | V index_select | Total 与设备任务 |
| `pg_host/fusion/attention_bridge` | 官方 npu_fusion_attention 调用 | Total 与其 C++ Self |

Python 探针仅在 profile capture 的上下文中启用；C++ 使用 PyTorch RecordFunction
用户范围，未开启 profiler 时不导出事件。探针没有新增 NPU 同步、设备拷贝或数据读取。
普通计时仍有进入这些轻量范围的检查成本；带 profiler 的探针记录也会扰动 Host 耗时，
不能把 profile 数值当成正常执行性能。

桥接探针测的是调用线程区间。torch-npu 的执行队列可将参数转换、缓存查询、
workspace 准备和执行放到工作线程，具体取决于配置。不能把这些成本一律归到主线程探针，
需要查看关联线程。此轮没有复制/替换 torch-npu 调度宏，也未修改 CANN 或官方算子。
缓存命中次数和 tiling 次数不由这些粗粒度探针直接推断。

探针版本为 3。load_extension/autograd_apply 被 dispatch 替代，
前向 pack_bridge/lse_compact 仍不存在，报告不要求这些旧事件。
dispatch 包含首次入口解析；预热后的入口缓存命中，不再执行 load_extension。
父 Self 变小本身不代表优化成功。报告中的 `cpp_outside_stages_us` 是 C++ Total
减去三个直接阶段 Total；`before_final_sync_us` 是外层 Total 减最后同步 Total。
`dispatch_outside_cpp_us` 是 dispatch Total 减 C++ 算子 Total，直接观察入口包装的剩余范围。
这些数值仍是墙钟区间，可能包含等待。

## 实机操作

固定 CANN 9.0.0、PyTorch/torch-npu 2.10.0、Python 3.10，在同一实机环境构建。
构建脚本每次删除自身架构目录内的 OPP build，重新生成 schema、元数据、内核和 wheel。
构建失败时不要继续测旧包；下面用 `&&` 串联成功条件。

```bash
cd /home/huangzhong/Agent/PrefixGrouper/npu_ops
set -o pipefail
bash scripts/build_wheel.sh 2>&1 | tee build_host_probes.log && \
bash scripts/run_910b_benchmark.sh /ABS/PATH/new_host_profile_run \
  --prefix 16 --suffixes 8 8 --hq 12 --hkv 2 \
  --warmup 5 --iterations 30 --no-backward \
  --trace-dir /ABS/PATH/new_host_profile_run/profiles \
  --profile-steps 3 --profile-aic-metrics None
```

将 `/ABS/PATH/...` 替换为新的绝对结果路径。输入参数必须替换成产生问题的原始输入；
上面使用最近提供的输入。iterations 和 profile 配置也应与优化前一致。
此示例关闭额外 AI Core 指标，聚焦 Host。默认 shape/memory/stack 采集均关闭；
需要复现之前的采集配置时添加 `--profile-record-shapes --profile-memory`。
两种配置保留同样的普通计时，只改变后续诊断采集。不要把默认轻量采集与上表
旧 shape/memory 采集的 Host 时间直接比较，也不要把关闭采集选项的变化算作算子提速。
`run_910b_benchmark.sh` 目前会先运行原有正确性门禁，其中包含梯度检查；
`--no-backward` 限制的是测速与 profile，不会跳过该门禁。
三个 profile capture 都在预热后单独采集，两条路径按 AB/BA 交替顺序采集，
不是冷启动与热启动对照。
每次 capture 仅一条路径的一次调用，但内部同名 API 可调用多次。

## 自动输出与需要告诉我的指标

新增输出：`profiles/host_profile.md`。它在采集后自动读取 CSV，逐次记录 Self、Total、
Count，汇总同名范围在多个 capture 中的 Total 中位数/最小/最大值，并附 CANN API 表。
同时在 benchmark.json 中保存解析后的 host_metrics、原始 API 汇总与已安装文件哈希。
每完成一个 capture 就更新报告；缺失必需探针会报告失败，而不是填 0。
未导出 api_statistic.csv 时保留“未知”，不推断缓存命中或某阶段没有开销。

机器隔离，不需要上传文件、路径或环境细节。文件留在实机，手工告知以下关键数值即可。
每个 capture 分开填，单位 μs；没有事件填“未观测”。

| 指标 | custom | fusion/permute |
|---|---|---|
| 未开启 profiler 的 mean / median | 待填 | 待填 |
| 外层 forward Total / 最后同步 Total | 待填 | 待填 |
| Python validate / scale Total | 待填 | 不适用 |
| 自定义 dispatch Self / Total | 待填 | 不适用 |
| C++ 算子 Self / Total | 待填 | 待填 |
| cpp_validate / allocate Total | 待填 | 无同名独立探针 |
| attention_bridge Self / Total | 待填 | 待填 |
| 前向 Pack / LSE copy 设备任务是否仍出现 | 应不再出现，填实际观察 | 不适用 |
| gather_k / gather_v Total | 不适用 | 待填 |
| GetWorkspaceSize 完整名称 / 线程 / Count / Time / Avg | 待填 | 待填 |
| Tiling 完整名称 / 线程 / Count / Time / Avg | 待填 | 待填 |
| Launch 完整名称 / Level / Count / Time / Avg | 待填 | 待填 |
| 缓存命中/未命中及直接证据 | 未知 | 未知 |
| Attention 设备任务名称 / 次数 / 耗时 | 待填 | 待填 |
| 其他设备任务与间隙 | 待填 | 待填 |

`api_statistic.csv` 按 API Name 和 Level 汇总整个 capture，可能包含同步或采集管理事件。
不能把它们全加起来与父事件 Self 对账；必须在 trace 中核对时间范围、线程与关联关系。
同一 API 的 Count 大于 1 不代表 attention 调用了多次；Variance 为 0 且 Count 为 1
也不能证明稳定。若内部区间未导出，先依据桥接探针定位主要调用，再决定第二轮精细打点。

## 本地验证边界

报告聚合逻辑使用无设备的定向检查；扩展在固定 Ubuntu 22.04 proot 中编译。
本机没有可用 NPU，不运行本 benchmark，不据此宣称实机探针导出成功、缓存命中或性能改善。
