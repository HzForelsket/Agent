# Copyright (c) Microsoft. All rights reserved.

"""Download missing example data and serve retrieval: python wiki_retriever_mcp.py."""

import argparse
import os
import pickle
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from rag_data import DEFAULT_DATA_DIR, add_download_argument, ensure_example_data


def main() -> None:
    """Load the retrieval corpus and serve its original top-one retrieval tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    add_download_argument(parser)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_DATA_DIR / "embedding-models")
    parser.add_argument(
        "--local-files-only", action="store_true", help="Load cached/local model files without network access."
    )
    parser.add_argument("--device", default="cpu", help="Embedding device; CPU avoids using the serving accelerator.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    endpoint = urlsplit(args.hf_endpoint)
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.netloc
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        parser.error("hf-endpoint must be an HTTP(S) URL without credentials, query or fragment")

    # The Hub reads these settings during import, before SentenceTransformer is loaded.
    os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
    if args.insecure_download or args.hf_endpoint.rstrip("/") != "https://huggingface.co":
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    ensure_example_data(args.data_dir, insecure=args.insecure_download)

    if args.insecure_download:
        # Xet uses a separate TLS stack; route Hub downloads through the configured HTTP client.
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        import httpx
        from huggingface_hub import set_client_factory

        set_client_factory(lambda: httpx.Client(verify=False, follow_redirects=True, timeout=60))

    import faiss
    from fastmcp import FastMCP
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(args.data_dir / "index_hnsw_faiss_n32e40_tiny.index"))
    with (args.data_dir / "chunks_candidate_tiny.pkl").open("rb") as handle:
        chunks = pickle.load(handle)
    local_only = args.local_files_only or Path(args.embedding_model).is_dir()
    args.embedding_cache.mkdir(parents=True, exist_ok=True)
    print(
        f"Loading embedding model: {args.embedding_model}; endpoint={args.hf_endpoint}; "
        f"cache={args.embedding_cache.resolve()}; local_files_only={local_only}",
        flush=True,
    )
    try:
        model = SentenceTransformer(
            args.embedding_model,
            device=args.device,
            cache_folder=str(args.embedding_cache.resolve()),
            local_files_only=local_only,
        )
    except Exception as error:
        raise RuntimeError(
            f"Embedding model loading failed ({args.embedding_model}, endpoint={args.hf_endpoint}): {error}. "
            "For an unreachable Hub, set --hf-endpoint to a reachable mirror, or pass "
            "--embedding-model /path/to/complete/local/model --local-files-only. "
            "The MCP server has not started."
        ) from error
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
