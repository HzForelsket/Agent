# Copyright (c) Microsoft. All rights reserved.

"""Prepare missing RAG example data in the repository's data/cache/rag directory."""

import argparse
import fcntl
import hashlib
import os
import ssl
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import HTTPSHandler, Request, build_opener

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "rag"
EXAMPLE_FILES = {
    "dataset_tiny.parquet": (
        "1Pq4Ag8zVoN8gUtLu0LcBfY35Dm5zL0hq",
        "f06d0ddda5657937c022a3b2f0599ad7b3a759d4077bd9c7296b3458cbc050ea",
    ),
    "chunks_candidate_tiny.pkl": (
        "1REXCpRLbeZu1KfWWKhIGEQe_WNHUOBkS",
        "f9f93413cbbdba05b92f06589bb5a411bf4b760ea94b50e05b34342693b2334c",
    ),
    "index_hnsw_faiss_n32e40_tiny.index": (
        "1f6P-h_8KSRhe5pqDHWbRQWvUhTygfZ-c",
        "8eee199118ff5b2c4217f8f7330c97f240b3ec9d7107feef63398adff06aa1a3",
    ),
}


def insecure_downloads_enabled() -> bool:
    """Read the shared opt-in for downloads on hosts without CA certificates."""
    return os.environ.get("RAG_DOWNLOAD_INSECURE", "0") == "1"


def add_download_argument(parser: argparse.ArgumentParser) -> None:
    """Expose the same certificate-verification switch on each download entrypoint."""
    parser.add_argument(
        "--insecure-download",
        action="store_true",
        default=insecure_downloads_enabled(),
        help="Skip download TLS certificate verification; also enabled by RAG_DOWNLOAD_INSECURE=1.",
    )


def ensure_example_data(data_dir: Path, *, insecure: bool | None = None) -> None:
    """Download missing example files atomically, preserving existing local data.

    Args:
        data_dir: Directory shared by the example dataset and retrieval corpus.
        insecure: Skip TLS certificate checks; defaults to RAG_DOWNLOAD_INSECURE.
    """
    data_dir = data_dir.resolve()
    insecure = insecure_downloads_enabled() if insecure is None else insecure
    if insecure:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    else:
        context = ssl.create_default_context()
    opener = build_opener(HTTPSHandler(context=context))
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, (file_id, checksum) in EXAMPLE_FILES.items():
            destination = data_dir / name
            if destination.is_file() and destination.stat().st_size:
                print(f"Using existing data: {destination}", flush=True)
                continue
            if destination.exists():
                raise ValueError(f"Data path is empty or is not a file: {destination}; remove or replace it")
            print(f"Downloading example data: {destination}", flush=True)
            try:
                with TemporaryDirectory(prefix=".download-", dir=data_dir) as temporary:
                    downloaded = Path(temporary) / name
                    url = "https://drive.usercontent.google.com/download?" + urlencode(
                        {"id": file_id, "export": "download", "confirm": "t"}
                    )
                    request = Request(
                        url, headers={"User-Agent": "RAG-data-collector/1.0", "Accept-Encoding": "identity"}
                    )
                    with opener.open(request, timeout=60) as response, downloaded.open("wb") as output:
                        status = response.status
                        content_type = response.headers.get("Content-Type", "unknown")
                        content_length = response.headers.get("Content-Length", "unknown")
                        encoding = response.headers.get("Content-Encoding", "identity")
                        digest = hashlib.sha256()
                        received = 0
                        prefix = b""
                        while chunk := response.read(1024 * 1024):
                            if not prefix:
                                prefix = chunk[:160]
                            output.write(chunk)
                            digest.update(chunk)
                            received += len(chunk)
                    actual_checksum = digest.hexdigest()
                    if actual_checksum != checksum:
                        preview = repr(prefix.decode("utf-8", errors="replace"))
                        raise ValueError(
                            f"SHA-256 mismatch for {name}: expected={checksum}, actual={actual_checksum}; "
                            f"received_bytes={received}, HTTP={status}, Content-Type={content_type}, "
                            f"Content-Length={content_length}, Content-Encoding={encoding}; "
                            f"response_prefix={preview}. The response is not the pinned example file; "
                            "it has not been added to the cache."
                        )
                    downloaded.replace(destination)
            except Exception as error:
                raise RuntimeError(
                    f"Automatic download failed for {destination}: {error}. "
                    "Rerun rag_data.py separately to diagnose the download without starting model collection. "
                    "For certificate verification errors use --insecure-download or RAG_DOWNLOAD_INSECURE=1; "
                    "this does not bypass file-content verification. Completed files will be reused."
                ) from error
            print(f"Saved and verified: {destination}", flush=True)


def main() -> None:
    """Prepare data without starting a model or retrieval service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    add_download_argument(parser)
    args = parser.parse_args()
    ensure_example_data(args.data_dir, insecure=args.insecure_download)


if __name__ == "__main__":
    main()
