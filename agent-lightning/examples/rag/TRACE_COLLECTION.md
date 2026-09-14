# 在 NPU 服务上采集 30B 多轮 RAG 轨迹

本入口调用仓库现有 `RAGAgent`，默认采样 32 题，每题独立运行 4 条完整轨迹，最多 8 次模型调用，
每次最多生成 2048 token，temperature=0.7、seed=20260914、并发 4。
目标模型是 `Qwen/Qwen3-30B-A3B-Instruct-2507`（总参数 30B，激活参数约 3B）。
采集结束自动生成收益表；不启动训练，也不要求安装 PrefixGrouper 或 verl。

收益表只比较**完整轨迹分别表示**与**同题 4 条完整轨迹合并前缀树**，不把每次模型调用当成独立训练样本。
它估计 token 位置和 causal attention pair 的减少，不测量 NPU 训练加速。

## 1. 环境和模型服务

模型服务运行在已经配置好的 NPU 环境中，项目指定版本为 CANN 9.0.0、vLLM 0.22.1、
vllm-ascend 0.22.1rc1（训练栈的 verl 0.9.0 不参与采集）。
本机没有 NPU，当前入口尚未在 NPU 上实跑；这里准备的是连接该服务的客户端和离线统计工具。

采集客户端与 CPU 检索服务可以运行在 NPU 主机的独立 `agent` conda 环境，也可运行在另一台能访问模型 API 的主机。
不要将客户端依赖安装命令用于替换现有 NPU serving 环境的 torch/CANN/vLLM。
在仓库的 `agent-lightning` 目录，使用客户端环境安装：

```bash
conda activate agent
# 客户端只需要 CPU torch；已有可用 torch 时跳过这一行。
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . -r examples/rag/requirements-traces.txt
cd examples/rag
```

`openai` 和 `openai-agents` 都是 pip 包，负责发 HTTP 请求和执行 Agent 循环；
它们无需 NPU 专用版本。实际模型计算由 NPU 上的 vLLM 服务执行。
应用依赖清单是独立采集入口的安装清单，不修改主项目的 `uv.lock`。

在已有可运行该模型的 NPU 部署命令中，设置以下 vLLM API 参数：

```text
--served-model-name Qwen3-30B-A3B-Instruct-2507
--host 127.0.0.1 --port 18030
--enable-auto-tool-choice --tool-call-parser hermes
--max-model-len 32768
```

模型权重路径、NPU 设备选择、tensor parallel 大小沿用该 NPU 机器上可运行的模型部署配置。
上述是 vLLM API 参数片段，不是完整的 NPU 部署命令。
跨主机访问时使用服务端实际监听地址和可达的 API URL。
`--model` 传入 `--served-model-name` 的逻辑名，不传权重目录。
服务必须支持 `return_token_ids=true`，返回真实的 `prompt_token_ids` 和 `choices[0].token_ids`；
采集器会检查它们与 usage 长度一致，缺失时停止该轨迹，不通过本地重新 tokenize 猜测。

可将实际服务信息保存为 `data/server_metadata.json`，使用 `--server-metadata` 随轨迹保存。
内容应包含实际 NPU 型号/数量、模型路径与 revision、CANN/vLLM/vllm-ascend 版本、
完整启动命令、tensor parallel、dtype、chat template，以及检索模型路径和 revision。
客户端自动记录的包版本只代表客户端，不能替代远程服务端版本。

## 2. 准备示例数据和检索服务

以下是本 RAG 示例原有的 MuSiQue tiny 数据和 Wikipedia 检索语料。
可以在能联网的机器下载后整体拷贝 `data/` 到 NPU 主机；无需上传到 Git。
已有这三个文件时跳过下载。
先进入仓库的 `agent-lightning/examples/rag` 目录。`data/` 已被 Git 忽略，
`git pull` 不会带来这些文件，采集入口也不会自动下载它们。

```bash
mkdir -p data
gdown 1Pq4Ag8zVoN8gUtLu0LcBfY35Dm5zL0hq -O data/dataset_tiny.parquet
gdown 1REXCpRLbeZu1KfWWKhIGEQe_WNHUOBkS -O data/chunks_candidate_tiny.pkl
gdown 1f6P-h_8KSRhe5pqDHWbRQWvUhTygfZ-c -O data/index_hnsw_faiss_n32e40_tiny.index
ls -lh data/dataset_tiny.parquet data/chunks_candidate_tiny.pkl data/index_hnsw_faiss_n32e40_tiny.index
python wiki_retriever_mcp.py --data-dir data --device cpu \
  --embedding-model BAAI/bge-large-en-v1.5
```

离线机器将 `--embedding-model` 替换为已下载的 BGE 模型目录。检索索引与该 embedding 模型配套，
保持现有工具每次返回 top-1 文档的行为。该终端保持运行，默认 MCP 地址是 `http://127.0.0.1:8099/sse`。

如果报 `dataset_tiny.parquet` 不存在，先完成以上下载，再重新运行采集。
已有数据放在其他目录时，用 `--dataset /绝对路径/dataset_tiny.parquet` 指定题目文件，
并用检索服务的 `--data-dir /绝对路径/语料目录` 指定索引和文本位置。
若 NPU 主机无法访问 Google Drive，在可联网机器下载这三个文件后拷贝过去。

## 3. 采集并自动生成收益表

另开客户端终端，进入同一 `examples/rag` 目录：

```bash
conda activate agent
python collect_traces.py \
  --endpoint http://127.0.0.1:18030/v1 \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --dataset data/dataset_tiny.parquet \
  --mcp-url http://127.0.0.1:8099/sse \
  --tasks 32 --rollouts-per-task 4 --concurrency 4 \
  --max-model-calls 8 --max-tokens-per-call 2048 \
  --temperature 0.7 --seed 20260914 \
  --output traces/npu-qwen30b-run01
```

有服务端元数据时附加 `--server-metadata data/server_metadata.json`。
API 启用鉴权时，通过 `VLLM_API_KEY` 环境变量提供密钥；不要将密钥写入 URL、元数据或运行命令文件。
每次采集必须用新目录。`--proxy-port` 默认 18031，会占用从该端口起连续 `--concurrency` 个本机端口。
一个 worker 对应一个独立进程，避免 Lightning tracer 在同一线程中并发运行轨迹发生冲突。

主进程首先打印输出目录。可在第三个终端查看增量进度：

```bash
tail -f traces/npu-qwen30b-run01/worker-*.log
```

每次模型调用、工具事件和完成轨迹逐条 flush/fsync 保存。中断时已经保存的记录仍可用于离线分析；
没有完整保存的轨迹不会进入主收益表。若初始化失败且没有任何有效模型调用，会打印失败日志并删除空跑目录。
不支持在原目录续写，以免重复样本混入同一组。

## 4. 保存文件与计算口径

| 文件 | 内容 |
|---|---|
| `config.json`、`environment.json` | 采样配置、客户端包版本、Git revision、入口源码摘要 |
| `selected_tasks.json`、`dataset_metadata.json` | 实际题目、数据文件 SHA-256 |
| `server_metadata.json` | 传入的服务端部署信息（提供参数时保存） |
| `calls.jsonl` | 完整请求/回复、真实 prompt/response token ID、顺序和时间 |
| `events.jsonl` | 工具执行事件、工具结果、最终答案；工具参数也在原始模型消息中 |
| `trajectories.jsonl` | 一条记录对应一条完整或中断轨迹，含题号、组内编号、状态 |
| `spans.jsonl` | Lightning spans 和 reward 记录 |
| `worker-*.log`、`completion.json`、`failure.json` | 进度、进程结束或失败状态；进程退出不代表所有轨迹正常完成 |
| `analysis/report.md` | 可直接阅读的收益表与覆盖率 |
| `analysis/benefit.csv` | 可用表格软件打开的汇总收益表 |
| `analysis/per_task.csv` | 每题收益，CSV 中比例以 0–1 小数保存 |
| `analysis/summary.json` | 精确计数、排除原因、非严格追加转移数量和指标定义 |

重算或分析中断采集的已保存数据，不需要模型、NPU 或 SDK，只需 Python 标准库：

```bash
python analyze_traces.py --input traces/npu-qwen30b-run01
```

设每条完整轨迹实际调用序列的 token 前缀并集为 `T_i`，同题的 4 条轨迹为一个组：

- 基线工作量：`sum_i |T_i|`；每条轨迹内部已有的历史共享不再次计入跨轨迹收益。
- 共享后工作量：`|union_prefix(T_1, T_2, T_3, T_4)|`。
- token 减少比例：`1 - 共享后 / 基线`；工作量缩减倍数：`基线 / 共享后`。
- attention pair 计数：每个唯一节点的祖先数加自身，假定完整 causal attention，不代表 kernel 实际执行量。

严格历史追加时，`T_i` 就是一条完整 token 序列。工具调用重序列化可能改变前轮结尾，
此时保留该轨迹各次调用的真实上下文；只取最后一次调用会丢失部分实际生成 token 的条件上下文。
主表只纳入组内全部完成、无长度截断、调用连续、真实 token ID 完整的组，并显示排除组数。
不同题目之间不合树；同题分叉后重复出现的文本也不当作共享前缀。
完整轨迹是样本单位，optimizer.step 通常聚合一个 batch，二者并非一一对应。

共享激活不能合并不同轨迹的 advantage、loss 权重或 clipping 项，必须保留其贡献并正确累加梯度。
工具输出只有上下文作用，不直接作为模型动作计算 policy loss。
表中比例不包含反向传播、通信、packing、显存或 kernel 调度成本，不能当作训练加速比。

## 已完成的验证

在本地 `agent` 环境通过 CLI 帮助、语法和格式检查，并使用此前真实采集的
32 题 × 4 条完整轨迹（408 次调用）运行正式离线入口：基线 104,998 token 位置，
合并后 54,621，减少 47.98%；attention pairs 减少 29.21%。这些数据来自此前 GPU 采集，
用于核对统计口径，**不是本次 NPU 实测结果**。NPU 侧采集结果以用户运行后生成的文件为准。
