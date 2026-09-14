# Copyright (c) Microsoft. All rights reserved.

"""Download a safetensors embedding model from ModelScope for strictly local loading."""

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import OpenerDirector, Request

from rag_data import DEFAULT_DATA_DIR, add_download_argument, create_download_opener


def file_hash(path: Path) -> str:
    """Hash a model file without loading its weights into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(opener: OpenerDirector, model_id: str, root: Path, entry: dict[str, Any]) -> None:
    """Resume one file, verify its bytes, and atomically publish it in the local model directory."""
    target = root / entry["Path"]
    expected_size, checksum = entry["Size"], entry["Sha256"]
    if target.is_file():
        if target.stat().st_size != expected_size or file_hash(target) != checksum:
            raise ValueError(f"Invalid cached model file: {target}; remove this file and rerun")
        print(f"Using verified model file: {target}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= expected_size:
        if offset == expected_size and file_hash(partial) == checksum:
            partial.replace(target)
            return
        partial.unlink()
        offset = 0
    url = (
        f"https://modelscope.cn/models/{quote(model_id, safe='/')}/resolve/"
        f"{quote(entry['Revision'], safe='')}/{quote(entry['Path'], safe='/')}"
    )
    headers = {"User-Agent": "RAG-embedding-downloader/1.0", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    print(f"Downloading {target.name}: {offset:,}/{expected_size:,} bytes", flush=True)
    with opener.open(Request(url, headers=headers), timeout=60) as response:
        if response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{expected_size}"):
                raise ValueError(f"Unexpected Content-Range for {target.name}: {content_range}")
        elif response.status == 200:
            offset = 0
        else:
            raise ValueError(f"Unexpected HTTP {response.status} for {target.name}")
        if response.headers.get_content_type() == "text/html":
            raise ValueError(f"Received an HTML page instead of model file {target.name}")
        with partial.open("ab" if offset else "wb") as output:
            received = offset
            last_update = time.monotonic()
            try:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    received += len(chunk)
                    if time.monotonic() - last_update >= 10:
                        print(f"{target.name}: {received:,}/{expected_size:,} bytes", flush=True)
                        last_update = time.monotonic()
            finally:
                output.flush()
                os.fsync(output.fileno())
    actual = file_hash(partial)
    if partial.stat().st_size != expected_size or actual != checksum:
        raise ValueError(
            f"Model file verification failed for {target.name}: bytes={partial.stat().st_size}, "
            f"expected_bytes={expected_size}, SHA-256={actual}, expected_SHA-256={checksum}; rerun to resume"
        )
    partial.replace(target)
    print(f"Saved and verified: {target}", flush=True)


def prepare_model(model_id: str, cache_dir: Path, *, insecure: bool = False, local_only: bool = False) -> Path:
    """Prepare ModelScope model files needed by the safetensors/CPU retrieval path.

    Args:
        model_id: ModelScope repository in namespace/name form.
        cache_dir: Root cache containing model namespace directories.
        insecure: Skip TLS certificate checks on API and file requests.
        local_only: Require a complete cached manifest and model without network access.
    """
    parts = model_id.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} or "\\" in part for part in parts):
        raise ValueError("embedding-model must be a local directory or a ModelScope namespace/model ID")
    root = cache_dir.resolve().joinpath(*parts)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Embedding model cache: {root}", flush=True)
    with (root / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = root / ".modelscope-manifest.json"
        opener = None if local_only else create_download_opener(insecure)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            if local_only:
                raise FileNotFoundError(f"No cached ModelScope manifest: {manifest_path}")
            url = f"https://modelscope.cn/api/v1/models/{quote(model_id, safe='/')}/repo/files?" + urlencode(
                {"Revision": "master", "Recursive": "true"}
            )
            assert opener is not None
            with opener.open(url, timeout=60) as response:
                metadata = json.load(response)
            if metadata.get("Code") != 200:
                raise ValueError(f"ModelScope file listing failed: {metadata.get('Message')}")
            # Download configs, tokenizer assets and safetensors; omit duplicate .bin and ONNX weights.
            files = [
                entry
                for entry in metadata["Data"]["Files"]
                if entry["Type"] == "blob"
                and not entry["Path"].startswith("onnx/")
                and PurePosixPath(entry["Path"]).suffix in {".json", ".txt", ".model", ".safetensors", ".md"}
            ]
            manifest = {"source": "modelscope", "model_id": model_id, "files": files}
        if manifest.get("source") != "modelscope" or manifest.get("model_id") != model_id:
            raise ValueError(f"ModelScope cache identity mismatch: {manifest_path}")
        files = manifest["files"]
        if not any(entry["Path"].endswith(".safetensors") for entry in files):
            raise ValueError("This downloader requires safetensors model weights")
        for entry in files:
            path = PurePosixPath(entry["Path"])
            if path.is_absolute() or ".." in path.parts or "\\" in entry["Path"]:
                raise ValueError(f"Invalid model file path: {entry['Path']}")
            checksum = entry["Sha256"]
            if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum) or entry["Size"] < 0:
                raise ValueError(f"Invalid model file metadata: {entry['Path']}")
        if not manifest_path.exists():
            pending = manifest_path.with_suffix(".tmp")
            with pending.open("w") as output:
                json.dump(manifest, output, indent=2)
                output.flush()
                os.fsync(output.fileno())
            pending.replace(manifest_path)
        for entry in files:
            if local_only:
                target = root / entry["Path"]
                if (
                    not target.is_file()
                    or target.stat().st_size != entry["Size"]
                    or file_hash(target) != entry["Sha256"]
                ):
                    raise ValueError(f"Missing or invalid local model file: {target}")
            else:
                assert opener is not None
                download_file(opener, model_id, root, entry)
    return root


def main() -> None:
    """Download the embedding model without starting the MCP server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_DATA_DIR / "embedding-models")
    parser.add_argument("--local-files-only", action="store_true")
    add_download_argument(parser)
    args = parser.parse_args()
    print(prepare_model(args.model, args.cache_dir, insecure=args.insecure_download, local_only=args.local_files_only))


if __name__ == "__main__":
    main()
