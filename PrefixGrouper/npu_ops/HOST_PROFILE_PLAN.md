# 前向 Host 耗时定位计划与回传清单

目标：定位 `pg-ascend-shared-prefix-attention`（Ascend 共享前缀注意力算子微基准）
中 custom 与 K/V 展开后官方 fusion attention 的主机耗时差异。
当前只分析前向，保持原有数学计算、核间分配、输入和异步提交方式。

## 已知数据与未知项

用户已有数据（μs），尚未随本次源码在实机复测：

| 事件 | custom | fusion/permute | 口径 |
|---|---:|---:|---|
| 外层 forward | 1750.77 | 964.11 | Host Total，含最后同步 |
| 最后同步 | 343.75 | 276.62 | Host Total |
| 同步前区间 | 1407.02 | 687.49 | 外层减最后同步，含潜在内部等待 |
| `_SharedPrefixAttention` | 788.51 | 不适用 | Host Total |
| `_SharedPrefixAttention` | 78.77 | 不适用 | Host Self |
| PyTorch C++ 算子事件 | 709.74 | 180.28 | Host Total，内部工作不同 |
| custom C++ 算子事件 | 512.58 | 未提供 | Host Self |

Launch 总时间接近，不能证明参数准备、缓存行为、执行队列或设备计算耗时相同。
512.58 是扣除已记录子事件后的剩余时间，不是一个已识别的内部阶段。
目前不能认定参数数量、缓存未命中或 tiling 是原因。

## 分析顺序

1. 确认构建完成、Python 环境与 wheel 路径正确、已安装二进制哈希已记录。
2. 同输入先看未开启 profiler 的 mean/median，确认问题在正常计时中存在。
3. 看下面互不重叠的 C++ 阶段 Total，先找主要贡献者。
4. Attention/Pack 桥接若占主导，在 trace 中关联 CANN API 和工作线程，检查
   workspace/executor 准备、tiling、任务提交或等待；不只看 Launch。
5. 对照官方 fusion 桥接，以及基线单独的 gather，避免用 custom 整条前向对比 fusion 单行。
6. 只优化实测占主导的阶段；复测时保持输入、预热、采样和采集配置一致。

## 已实现的探针

以下名称在 `trace_view.json` 和 `operator_details.csv` 中搜索。

| 探针名称 | 范围 | 重点看 |
|---|---|---|
| `pg_host/custom/python_validate` | Python 输入和 plan 检查 | Total |
| `pg_host/custom/python_scale` | CPU scale 计算、有限值检查和 item | Total |
| `pg_host/custom/load_extension` | 扩展加载检查；预热后通常已加载 | Total |
| `pg_host/custom/autograd_apply` | autograd 包装与 C++ 算子 | Self 与 Total，不能加到子阶段上 |
| `pg_host/custom/cpp_validate` | C++ 参数检查 | Total |
| `pg_host/custom/allocate` | acc、lse_rows、out 分配 | Total 与内部 aten 子事件 |
| `pg_host/custom/attention_bridge` | 整次 EXEC_NPU_CMD_EXT 前向调用 | Total 与对应 CANN/工作线程 |
| `pg_host/custom/pack_bridge` | 整次 EXEC_NPU_CMD_EXT Pack 调用 | Total 与对应 CANN/工作线程 |
| `pg_host/custom/lse_compact` | select + contiguous | Total 与 copy 子事件 |
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

新增子探针后，原来 512.58 μs 的父事件 Self 会被重新分配到子事件。
父 Self 变小本身不代表优化成功。报告中的 `cpp_outside_stages_us` 是 C++ Total
减去五个直接阶段 Total；`before_final_sync_us` 是外层 Total 减最后同步 Total。
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
  --prefix 1024 --suffixes 64 65 63 1 --hq 6 --hkv 2 \
  --warmup 10 --iterations 30 --no-backward \
  --trace-dir /ABS/PATH/new_host_profile_run/profiles \
  --profile-steps 3 --profile-aic-metrics None
```

将 `/ABS/PATH/...` 替换为新的绝对结果路径。输入参数必须替换成产生问题的原始输入；
上面仅展示脚本默认输入。此轮关闭额外 AI Core 指标，聚焦 Host，shape/memory 采集仍开启。
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

优先直接回传：

1. `profiles/host_profile.md`、`benchmark.json`、`benchmark.md`、`build_host_probes.log`。
2. custom 和 fusion 各次 capture 的 `operator_details.csv`、`kernel_details.csv`，
   以及 `api_statistic.csv`（若存在）。
3. 两条路径至少各一个 `trace_view.json`，优先选择外层耗时接近中位数的 capture。

若只能手工提供数值，填写以下表格。每个 capture 分开填，单位 μs；没有事件填“未观测”。

| 指标 | custom | fusion/permute |
|---|---|---|
| 未开启 profiler 的 mean / median | 待填 | 待填 |
| 外层 forward Total / 最后同步 Total | 待填 | 待填 |
| Python validate / scale / load Total | 待填 | 不适用 |
| 自定义 autograd_apply Self / Total | 待填 | 不适用 |
| C++ 算子 Self / Total | 待填 | 待填 |
| cpp_validate / allocate Total | 待填 | 无同名独立探针 |
| attention_bridge Self / Total | 待填 | 待填 |
| pack_bridge / lse_compact Total | 待填 | 不适用 |
| gather_k / gather_v Total | 不适用 | 待填 |
| GetWorkspaceSize 完整名称 / 线程 / Count / Time / Avg | 待填 | 待填 |
| Tiling 完整名称 / 线程 / Count / Time / Avg | 待填 | 待填 |
| Launch 完整名称 / Level / Count / Time / Avg | 待填 | 待填 |
| 缓存命中/未命中及直接证据 | 未知 | 未知 |
| Attention 设备任务名称 / 次数 / 耗时 | 待填 | 待填 |
| Pack、LSE、gather 等其他设备任务与间隙 | 待填 | 待填 |

`api_statistic.csv` 按 API Name 和 Level 汇总整个 capture，可能包含同步或采集管理事件。
不能把它们全加起来与 512.58 μs 对账；必须在 trace 中核对时间范围、线程与关联关系。
同一 API 的 Count 大于 1 不代表 attention 调用了多次；Variance 为 0 且 Count 为 1
也不能证明稳定。若内部区间未导出，先依据桥接探针定位主要调用，再决定第二轮精细打点。

## 本地验证边界

报告聚合逻辑使用无设备的定向检查；扩展在固定 Ubuntu 22.04 proot 中编译。
本机没有可用 NPU，不运行本 benchmark，不据此宣称实机探针导出成功、缓存命中或性能改善。
