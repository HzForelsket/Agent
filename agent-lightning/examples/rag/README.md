# RAG Agent Example

[![rag workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-rag.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-rag.yml)

This example demonstrates training a Retrieval-Augmented Generation (RAG) agent using Agent-Lightning with retrieval capabilities. The agent answers multi-hop questions from a tiny MuSiQue dataset by retrieving and reasoning over Wikipedia passages.

For 30B multi-turn trace collection through an NPU model service and a complete-trajectory prefix-sharing benefit table,
see [NPU 轨迹采集与收益表](TRACE_COLLECTION.md).
For unchanged SQL and 20 Questions workflows, see [原流程采集与跨任务报告](WORKFLOW_TRACE_COLLECTION.md).

## Overview

This example can run on a single GPU for demonstration purposes.

**Step 1:** Set up the environment. It is recommended to setup with uv and activate the virtual environment with:

```bash
uv sync --frozen --extra apo --group agents --group torch-gpu-stable --extra verl --group rag
source .venv/bin/activate
```

**Step 2:** Enter the example directory. Missing example files are downloaded automatically by the retriever,
collector and training entrypoints into the repository's `data/cache/rag/` directory. Existing files are reused.

```bash
cd examples/rag
# Optional: prepare the cache without starting a service.
python rag_data.py
```

The data downloader uses Python's standard library. On a host without CA certificates, pass
`--insecure-download` to `rag_data.py`, `wiki_retriever_mcp.py` and `collect_traces.py`, or set
`RAG_DOWNLOAD_INSECURE=1` for all entrypoints. Newly downloaded example files still require matching SHA-256 hashes.

**Step 3:** Start the MCP server. It downloads the BGE model from ModelScope into
`data/cache/rag/embedding-models/`, verifies each file, and loads the model locally. Open a terminal and run:

```bash
python wiki_retriever_mcp.py
```

**Step 4:** Start training. Open another terminal and run:

```bash
python train_rag.py
```

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `rag_agent.py` | RAG agent example using the OpenAI Agents SDK, with debugging utils |
| `train_rag.py` | Initiates the GRPO training process |
| `metric_utils.py` | Scoring utilities for exact match, F1 score, and response parsing |
| `rag_data.py` | Shared automatic data download, checksum verification and cache preparation |
| `embedding_download.py` | ModelScope embedding model downloads with resume, checksums and optional TLS verification |
| `wiki_retriever_mcp.py` | MCP server for Wikipedia retrieval |
| `collect_traces.py` | Automatically start NPU vLLM and CPU MCP, collect complete trajectories, and generate a benefit table |
| `trace_tasks.py` | Cached real RAG/Spider/Q20 inputs, database paths and data-only preparation CLI |
| `trace_workflows.py` | Outer adapters calling unchanged SQL and CrewAI Q20 workflows |
| `analyze_call_traces.py` | Independent-context call-slot sharing analysis without within-trajectory deduplication |
| `compare_trace_reports.py` | Cross-workload summary from SQL/Q20 analysis directories |
| `requirements-workflow-traces.txt` | Additional SQL and CrewAI client dependencies |
| `WORKFLOW_TRACE_COLLECTION.md` | Original-workflow NPU collection commands, metadata and statistical scope |
| `trace_services.py` | Owned service process groups, port checks, readiness, failure monitoring and shutdown |
| `analyze_traces.py` | Reanalyze raw traces or saved analysis directories; independent, single-prefix and trie sharing comparisons |
| `requirements-traces.txt` | Application dependencies for the separate trace collection client |
| `TRACE_COLLECTION.md` | Chinese instructions for NPU serving integration, saved traces and metric definitions |

## How to Prepare the Retrieval Corpus Yourself

To enable semantic retrieval with this MCP server, you need two files:

1. **FAISS index file** (`.index`)
2. **Chunk list file** (`.pkl`)

These two files work together: the FAISS index stores the vector embeddings and their mapping to integer IDs, while the pickle file stores the actual text chunks. The integer IDs in the index correspond exactly to the positions in the chunk list.

### Step 1: Collecting Text Chunks

First, you need a collection of text passages (chunks). For example, you can download a Wikipedia-based dataset such as `wiki18_100w.zip` from the [FlashRAG_dataset](https://huggingface.co/datasets/FlashRAG) or use other pre-split corpora.

### Step 2: Creating the FAISS Index (`nq_hnsw_faiss_n32e40.index`)

- Use a sentence embedding model (e.g., `BAAI/bge-large-en-v1.5`) to encode each chunk into a vector.
- Build a FAISS index from these vectors.
- In this example, we use an **HNSW index** (Hierarchical Navigable Small World graph), which supports efficient approximate nearest-neighbor search.
- The index stores only embeddings and integer IDs (no raw text).

### Step 3: Creating the Chunk List (`nq_list.pkl`)

- Store the raw text chunks in a Python list.
- Save this list with `pickle`.
- The index ID returned by FAISS corresponds to the list index in this file. For example, if FAISS search returns `I[0][i] = 12345`, then the corresponding text chunk is `chunks[12345]`.

### Example Schema

- **`nq_hnsw_faiss_n32e40.index`**
  - Type: FAISS HNSW index
  - Contains:
    - Vector embeddings
    - Graph structure for fast search
    - Integer IDs mapping to chunk positions

- **`nq_list.pkl`**
  - Type: Pickled Python list
  - Element type: string (or dict with text + metadata, depending on preprocessing)
  - Example:
    ```python
    [
        "The Eiffel Tower is located in Paris, France.",
        "Albert Einstein developed the theory of relativity.",
        ...
    ]
    ```

### Step 4: Code Example - Building Index and Chunk List

**Warning:** The following example demonstrates a small-scale workflow only. In practice, for large datasets, you should encode the text in batches and incrementally add them to the index.

```python
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# 1. Prepare your text chunks (list of strings)
chunk_texts = [
    "The Eiffel Tower is located in Paris, France.",
    "Albert Einstein developed the theory of relativity.",
    "Python is a popular programming language.",
    # ... more chunks
]

# 2. Load embedding model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# 3. Encode text chunks into embeddings
embeddings = model.encode(chunk_texts, normalize_embeddings=True)

# 4. Build FAISS HNSW index
dim = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dim, 32)   # 32 neighbors by default
index.hnsw.efConstruction = 40         # efConstruction parameter
index.add(embeddings)

# 5. Save FAISS index
faiss.write_index(index, "nq_hnsw_faiss_n32e40.index")

# 6. Save chunk list
with open("nq_list.pkl", "wb") as f:
    pickle.dump(chunk_texts, f)

print("Index and chunk list saved successfully.")
```
