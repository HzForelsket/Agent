# Copyright (c) Microsoft. All rights reserved.

"""Download missing example data and serve retrieval: python wiki_retriever_mcp.py."""

import argparse
import pickle
from pathlib import Path
from typing import Any

from rag_data import DEFAULT_DATA_DIR, ensure_example_data


def main() -> None:
    """Load the retrieval corpus and serve its original top-one retrieval tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device", default="cpu", help="Embedding device; CPU avoids using the serving accelerator.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()

    ensure_example_data(args.data_dir)

    import faiss
    from fastmcp import FastMCP
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(args.data_dir / "index_hnsw_faiss_n32e40_tiny.index"))
    with (args.data_dir / "chunks_candidate_tiny.pkl").open("rb") as handle:
        chunks = pickle.load(handle)
    model = SentenceTransformer(args.embedding_model, device=args.device)
    if index.ntotal != len(chunks) or model.get_sentence_embedding_dimension() != index.d:
        raise ValueError("Corpus, index and embedding dimensions do not match")
    mcp = FastMCP(name="wiki retrieval mcp")

    @mcp.tool(name="retrieve", description="retrieve relevant chunks from the wikipedia")
    def retrieve(query: str) -> list[dict[str, Any]]:
        """Retrieve the most relevant Wikipedia chunk for the query."""
        embedding = model.encode([query], normalize_embeddings=True)
        distances, indices = index.search(embedding, 1)
        return [
            {"chunk": chunks[idx], "chunk_id": int(idx), "distance": float(distance)}
            for idx, distance in zip(indices[0], distances[0])
            if idx != -1
        ]

    print(f"Loaded {len(chunks)} chunks; embedding device: {args.device}", flush=True)
    mcp.run(transport="sse", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
