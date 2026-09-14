# 在 NPU 服务上采集 30B 多轮 RAG 轨迹

本入口调用仓库现有 `RAGAgent`，默认采样 32 题，每题独立运行 4 条完整轨迹，最多 8 次模型调用，
每次最多生成 2048 token，temperature=0.7、seed=20260914、并发 4。
目标模型是 `Qwen/Qwen3-30B-A3B-Instruct-2507`（总参数 30B，激活参数约 3B）。
采集结束自动生成收益表；不启动训练，也不要求安装 PrefixGrouper 或 verl。

统计单位统一为**一条完整轨迹、一条 token 序列**，收益表比较**每条完整轨迹独立计算**与**同题 4 条完整轨迹合并前缀树**。
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

服务器没有 CA 证书时，在运行采集与检索服务的各终端设置：

```bash
export RAG_DOWNLOAD_INSECURE=1
```

也可为 `collect_traces.py`、`wiki_retriever_mcp.py` 或 `rag_data.py` 单独传入 `--insecure-download`。
这会跳过示例数据和检索模型下载的 TLS 证书验证，不修改系统证书或模型 API 请求配置。
数据下载只使用 Python 标准库，不需要 `gdown`；示例文件下载后仍必须通过固定 SHA-256 校验。
检索模型下载使用 Hugging Face Hub 1.x 的 HTTP 客户端配置，并禁用独立 TLS 通道的 Xet 下载。
本地 embedding 模型目录同样可用。

若安装 pip 依赖也遇到证书错误，可为上面的安装命令添加所用下载域名的 `--trusted-host`：
PyPI 使用 `--trusted-host pypi.org --trusted-host files.pythonhosted.org`；CPU torch 索引使用
`--trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org`。

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

可将实际服务信息保存为 `../../../data/cache/rag/server_metadata.json`，使用 `--server-metadata` 随轨迹保存。
内容应包含实际 NPU 型号/数量、模型路径与 revision、CANN/vLLM/vllm-ascend 版本、
完整启动命令、tensor parallel、dtype、chat template，以及检索模型路径和 revision。
客户端自动记录的包版本只代表客户端，不能替代远程服务端版本。

## 2. 准备示例数据和检索服务

采集入口和检索服务会自动下载缺失的 MuSiQue tiny 题目、Wikipedia 文本和 FAISS 索引，
统一保存到**仓库根目录的 `data/cache/rag/`**。路径根据脚本位置定位，不依赖启动时的工作目录。
已有非空文件直接复用；新下载文件校验 SHA-256 后原子写入，失败不会留下可被误用的半成品。
两个入口同时启动时使用文件锁，避免重复下载。缓存已被 Git 忽略。

在 `agent-lightning/examples/rag` 目录直接启动检索服务即可，无需手动下载：

```bash
python wiki_retriever_mcp.py --device cpu --embedding-model BAAI/bge-large-en-v1.5
```

该终端保持运行，默认 MCP 地址是 `http://127.0.0.1:8099/sse`。
下一节的采集命令也会自动补齐缓存。若只想提前准备数据，可运行 `python rag_data.py`。
离线机器将 `--embedding-model` 替换为已下载的 BGE 模型目录；检索索引与该 embedding 模型配套，
保持现有工具每次返回 top-1 文档的行为。

若 NPU 主机无法访问 Google Drive，可在联网机器运行 `python rag_data.py`，
再把 `data/cache/rag/` 整体拷贝到 NPU 机器的同一仓库相对路径。
下载失败会显示失败文件和原因；网络恢复后重跑，已经下载完成的文件直接复用。
自定义题目可用 `--dataset /绝对路径/题目.parquet` 指定；缺失的自定义文件不会被示例数据替换。
使用 `dataset_tiny.parquet` 文件名时会自动在其所在目录补齐示例文件。
检索服务可通过 `--data-dir /绝对路径/语料目录` 指定缓存目录。

## 3. 采集并自动生成收益表

另开客户端终端，进入同一 `examples/rag` 目录：

```bash
conda activate agent
python collect_traces.py \
  --endpoint http://127.0.0.1:18030/v1 \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --mcp-url http://127.0.0.1:8099/sse \
  --tasks 32 --rollouts-per-task 4 --concurrency 4 \
  --max-model-calls 8 --max-tokens-per-call 2048 \
  --temperature 0.7 --seed 20260914 \
  --output traces/npu-qwen30b-run01
```

有服务端元数据时附加 `--server-metadata ../../../data/cache/rag/server_metadata.json`。
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
| `analysis/summary.json` | 精确计数、排除原因、序列表示方式和指标定义 |
| `analysis/trajectory_sequences.jsonl` | 实际纳入统计的完整轨迹，每条包含一条 token 序列及完整消息历史 |

重算或分析中断采集的已保存数据，不需要模型、NPU 或 SDK，只需 Python 标准库：

```bash
python analyze_traces.py --input traces/npu-qwen30b-run01
```

每条完整轨迹的序列 `S_i` 包含初始问题、全部模型动作、工具结果和最终回复。
使用最后一次请求的完整历史 `prompt_token_ids` 加最终 `response_token_ids` 构造一条序列；
逐轮检查原有消息和模型动作均保留在后续历史中，否则排除该题组。
同题 4 条完整轨迹为一个组，统一采用以下统计方式：

- 基线 token 工作量：`|S_1| + |S_2| + |S_3| + |S_4|`。
- 共享后 token 工作量：`Trie(S_1, S_2, S_3, S_4)` 的节点数（不计空根节点）。
- token 减少比例：`1 - 共享后 / 基线`；工作量缩减倍数：`基线 / 共享后`。
- 基线 attention pairs：`sum_i |S_i| * (|S_i| + 1) / 2`。
- 共享后 attention pairs：树中每个 token 节点可见祖先数加自身，假定完整 causal attention。

主表只纳入组内全部完成、无长度截断、调用连续、真实 token ID 完整的组，并显示排除组数。
不同题目之间不合树；同题分叉后重复出现的文本也不当作共享前缀。
汇总比例按所有题组的工作量总和计算。

共享激活不能合并不同轨迹的 advantage、loss 权重或 clipping 项，必须保留其贡献并正确累加梯度。
工具输出只有上下文作用，不直接作为模型动作计算 policy loss。
统计使用最终完整历史的序列表示；接入训练时须保证 tokenizer、loss mask 与 rollout log-prob 的位置映射一致。
表中比例不包含反向传播、通信、packing、显存或 kernel 调度成本，不能当作训练加速比。

## 已完成的验证

在本地 `agent` 环境通过 CLI 帮助、语法和格式检查，并使用此前真实采集的
32 题 × 4 条完整轨迹（408 次调用）运行正式离线入口，全部通过消息历史和模型动作保留检查：
基线 101,927 token 位置，合并后 51,656，减少 49.32%；attention pairs 从 49,310,766
降到 34,283,950，减少 30.47%。这些数据来自此前 GPU 采集，
用于核对统计口径，**不是本次 NPU 实测结果**。NPU 侧采集结果以用户运行后生成的文件为准。
