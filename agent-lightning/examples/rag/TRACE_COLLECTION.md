# 在 NPU 服务上采集 30B 多轮 RAG 轨迹

本入口调用仓库现有 `RAGAgent`，默认采样 32 题，每题独立运行 4 条完整轨迹，最多 8 次模型调用，
每次最多生成 2048 token，temperature=0.7、seed=20260914、并发 4。
目标模型是 `Qwen/Qwen3-30B-A3B-Instruct-2507`（总参数 30B，激活参数约 3B）。
采集结束自动生成收益表；不启动训练，也不要求安装 PrefixGrouper 或 verl。

统计单位统一为**一条完整轨迹、一条 token 序列**，收益表比较**每条完整轨迹独立计算**与**同题 4 条完整轨迹合并前缀树**。
它估计 token 位置和 causal attention pair 的减少，不测量 NPU 训练加速。

## 1. NPU 主机环境

现在 `collect_traces.py` 自动启动本机的 **NPU vLLM + CPU MCP**，等待两者就绪后采集，
结束后关闭本次启动的服务并生成收益表。无需另开终端手动启动服务。
采集器和 vLLM 必须处于同一主机/容器、能访问相同模型路径和本机端口。

NPU 环境按项目固定栈准备：CANN 9.0.0、vLLM 0.22.1、vllm-ascend 0.22.1rc1。
采集不需要 verl。客户端及 CPU 检索依赖可安装在独立环境，避免修改已有 NPU torch/CANN 环境。
在仓库的 `agent-lightning` 目录，使用客户端环境安装：

```bash
conda activate agent
# 客户端已有可用 torch 时跳过这一行。
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . -r examples/rag/requirements-traces.txt
cd examples/rag
```

`openai`、`openai-agents` 是普通 pip 包，负责 HTTP 请求与 Agent 循环，不需要 NPU 专用版本。
安装 pip 包若遇证书错误，按下载源添加 `--trusted-host pypi.org --trusted-host files.pythonhosted.org`；
CPU torch 源使用 `--trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org`。
应用依赖清单不修改主项目的 `uv.lock`。

若报 `No module named 'key_value.aio.stores.filetree'`，先在运行采集器的 Python 环境中对齐 MCP 依赖：

```bash
python -m pip install --upgrade \
  'fastmcp==2.13.1' \
  'py-key-value-aio[disk,keyring,memory]==0.2.8' \
  'py-key-value-shared==0.2.8' \
  --index-url https://pypi.org/simple \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

这组版本对应本地已成功运行的 CPU MCP 环境。FastMCP 2.13.1 使用 `DiskStore`，不导入 `filetree`；
出现该导入通常意味着实际加载的 FastMCP 与项目版本不一致，或安装文件混杂。
上述命令用于采集器/MCP 环境，`--vllm-python` 指定的独立服务环境无需修改。
若对齐后仍报相同错误，需检查完整 traceback 的导入来源与 `python -m pip show fastmcp py-key-value-aio py-key-value-shared`。

## 2. 一条命令采集

在 NPU 机器的 `agent-lightning/examples/rag` 目录运行，替换模型路径与实际分配的芯片 ID：

```bash
python collect_traces.py \
  --model-path /实际模型路径/Qwen3-30B-A3B-Instruct-2507 \
  --npu-devices 0,1,2,3 \
  --insecure-download \
  --tasks 32 --rollouts-per-task 4 --concurrency 4 \
  --output traces/npu-qwen30b-run01
```

若 vLLM 安装在另一个环境，追加 `--vllm-python /实际NPU环境/bin/python`；默认使用采集器自身的 Python。
该路径必须是同一主机/容器内已配置好 vLLM Ascend 的解释器。CANN 所需环境变量应在运行命令前加载，子进程会继承。
A3 部署如需 AIV，在运行前设置 `export HCCL_OP_EXPANSION_MODE=AIV`；A2 不需要此设置。
设备编号是 `ASCEND_RT_VISIBLE_DEVICES` 使用的芯片编号，TP 自动取编号个数；也可直接沿用已设置的这个环境变量。

`--model-path` 是**已下载的完整 BF16 30B 模型目录**；服务使用本地权重并禁用 Hugging Face 联网。
`--model` 是逻辑服务名，默认 `Qwen3-30B-A3B-Instruct-2507`，与本地目录名分开记录。
自动启动参数包括 BF16、expert parallel、mp、Hermes 工具解析、eager，以及关闭推理前缀缓存。
默认 `--max-model-len 32768`、`--gpu-memory-utilization 0.85`，后一个是 vLLM 在 NPU 上沿用的参数名。
部署参数参考固定版本的 [vllm-ascend Qwen3-30B 文档](https://github.com/vllm-project/vllm-ascend/blob/v0.22.1rc1/docs/source/tutorials/models/Qwen3-30B-A3B.md)。

入口依次执行：

1. 创建新的输出目录，保存配置、环境与题目，自动补齐缺失的示例数据。
2. 启动 CPU MCP 和 NPU vLLM，分别写入 `mcp.log`、`vllm.log`。
3. 检查 vLLM `/health` 和 `/v1/models`，并连接 MCP 确认 `retrieve` 工具存在。
4. 两者就绪后，创建独立 worker 进程，每题采集 4 条完整轨迹。
5. 关闭本次启动的进程组，自动生成 `analysis/report.md` 和 `analysis/benefit.csv`。

`--startup-timeout` 默认 1800 秒，包含服务内的 BGE 下载和模型加载，首次下载慢时可增大。
等待期间每 30 秒打印进度。默认 vLLM 端口 18030、MCP 端口 8099、代理端口 18031–18034。
可用 `--vllm-port`、`--mcp-port`、`--proxy-port` 修改；端口冲突会直接报错，不会接管或关闭其他进程。
若之前手动启动过服务，先在对应终端退出，或为本次采集选择其他端口。
原来的 `--endpoint` 和 `--mcp-url` 参数已移除，地址由自动启动的本机端口生成。

通过 `VLLM_API_KEY` 环境变量配置模型 API 密钥，采集器和服务共同使用；不写入配置文件。
可用 `--server-metadata /路径/server_metadata.json` 补充实际 NPU 型号、CANN/服务包版本和权重 revision。
`services.json` 自动记录启动命令与部分环境变量，其中 `reference_stack` 是目标版本，不是服务实际版本检测结果。

查看实时日志：

```bash
tail -f traces/npu-qwen30b-run01/mcp.log traces/npu-qwen30b-run01/vllm.log
tail -f traces/npu-qwen30b-run01/worker-*.log
```

采集期间服务退出或 worker 失败会停止其他进程；Ctrl+C / SIGTERM 同样触发清理，
先发送 SIGTERM，超过 15 秒则强制关闭本次进程组，包括 vLLM 子进程。
每次模型调用和轨迹逐条 flush/fsync 保存；已有有效模型调用时保留部分数据、服务日志和 `failure.json`。
如果尚无有效模型调用，会先在终端打印错误和日志末尾，再删除失败的空跑目录。
每次必须使用新输出目录，不在原目录续写，以免混入重复样本。

## 3. 数据与检索模型缓存

缺失的 MuSiQue tiny 题目、Wikipedia 文本及 FAISS 索引自动下载到仓库根目录 `data/cache/rag/`。
BGE 检索模型默认从 ModelScope 下载到 `data/cache/rag/embedding-models/BAAI/bge-large-en-v1.5/`。
已有数据和完整模型缓存会复用；这些缓存以及轨迹目录已被 Git 忽略。

`--insecure-download` 会同时传给 MCP，跳过数据和 BGE 下载中所有请求及跳转的 TLS 证书验证。
也可设置 `RAG_DOWNLOAD_INSECURE=1`。下载只使用标准库，不用 gdown 或 ModelScope SDK。
数据核验固定 SHA-256；模型按 ModelScope manifest 固定 revision，逐文件核验大小与 SHA-256，
中断的模型 `.part` 文件支持续传。SentenceTransformer 严格从本地文件加载，不连接 Hugging Face。

若已有离线 BGE 模型，在采集命令追加：

```text
--embedding-model /实际路径/bge-large-en-v1.5 --local-files-only
```

`--embedding-cache` 可修改模型缓存根目录；`--local-files-only` 只约束检索模型，示例数据仍须存在或能下载。
可用 `--retrieval-data-dir` 指定语料/索引目录，`--dataset` 单独指定题目 parquet。
自定义题目必须包含 id/question/answer 列；缺失的自定义文件不会被示例数据替换。
使用 `dataset_tiny.parquet` 文件名时会在其所在目录自动补齐示例文件。

只想提前下载、不启动服务时，在 `examples/rag` 运行：

```bash
python rag_data.py --insecure-download
python embedding_download.py --insecure-download
```

若不能访问 Google Drive，可在联网机器下载数据后拷贝 `data/cache/rag/`。
数据下载失败会显示文件名、预期/实际 SHA-256、字节数、HTTP 状态、响应类型及内容开头。
示例索引 `index_hnsw_faiss_n32e40_tiny.index` 的已核验大小为 8,735,522 字节。

## 4. 保存文件与计算口径

| 文件 | 内容 |
|---|---|
| `config.json`、`environment.json` | 采样配置、客户端包版本、Git revision、入口源码摘要 |
| `selected_tasks.json`、`dataset_metadata.json` | 实际题目、数据文件 SHA-256 |
| `services.json`、`services_ready.json` | 自动启动命令、环境、就绪时间和进程 ID |
| `mcp.log`、`vllm.log`、`analysis.log` | 检索服务、模型服务和离线统计日志 |
| `server_metadata.json` | 补充的实际服务端部署信息（提供参数时保存） |
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

本次自动服务编排完成 CLI 帮助、Python 语法及格式静态检查；本机无可用 NPU，尚未验证 NPU 启动、就绪探测和进程清理的完整运行链路。

在本地 `agent` 环境通过 CLI 帮助、语法和格式检查，并使用此前真实采集的
32 题 × 4 条完整轨迹（408 次调用）运行正式离线入口，全部通过消息历史和模型动作保留检查：
基线 101,927 token 位置，合并后 51,656，减少 49.32%；attention pairs 从 49,310,766
降到 34,283,950，减少 30.47%。这些数据来自此前 GPU 采集，
用于核对统计口径，**不是本次 NPU 实测结果**。NPU 侧采集结果以用户运行后生成的文件为准。

本地 `agent` 环境使用标准库从 ModelScope revision `bb9873b4c485b66b143ff6d8313e447b30f41c34`
实际下载了完整 BGE 模型，包含 1,340,616,616 字节的 safetensors 权重，全部文件通过大小及 SHA-256 校验。
随后使用默认模型 ID、`--device cpu --insecure-download` 启动 MCP，复用缓存并成功调用 `retrieve`，
返回 chunk 1205。该验证覆盖模型下载、缓存复用和 CPU 检索服务，不涉及 NPU 模型推理；续传分支尚未实际中断验证。
