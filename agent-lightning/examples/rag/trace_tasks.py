# Copyright (c) Microsoft. All rights reserved.

"""Prepare real RAG, Spider SQL and 20 Questions inputs for collect_traces.py."""

import fcntl
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import Request

from rag_data import DEFAULT_DATA_DIR, create_download_opener, ensure_example_data

CACHE = DEFAULT_DATA_DIR.parent
EXAMPLES = Path(__file__).resolve().parent.parent
SPIDER_URL = (
    "https://drive.usercontent.google.com/download?id=1oi9J1jZP9TyM35L85CL3qeGWl2jqlnL6&export=download&confirm=t"
)


def sha256(path: Path) -> str:
    """Hash a file without loading database archives into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_spider(directory: Path, insecure: bool) -> Path:
    """Download the repository's documented Spider data archive, without gdown."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        matches = list(directory.rglob("train_spider.parquet"))
        if not matches:
            archive = directory / "spider-data.zip"
            if not archive.exists():
                temporary = directory / "spider-data.zip.part"
                print(f"Downloading SQL data: {SPIDER_URL} -> {archive}", flush=True)
                try:
                    with create_download_opener(insecure).open(Request(SPIDER_URL), timeout=120) as response:
                        with temporary.open("wb") as handle:
                            shutil.copyfileobj(response, handle, length=1024 * 1024)
                    if not zipfile.is_zipfile(temporary):
                        raise ValueError("Spider download is not a ZIP archive (possibly a Google Drive error page)")
                    temporary.replace(archive)
                finally:
                    temporary.unlink(missing_ok=True)
            with zipfile.ZipFile(archive) as bundle:
                for entry in bundle.infolist():
                    target = (directory / entry.filename).resolve()
                    if (
                        not target.is_relative_to(directory.resolve())
                        or (entry.external_attr >> 16) & 0o170000 == 0o120000
                    ):
                        raise ValueError(f"Unsafe Spider archive entry: {entry.filename}")
                corrupt = bundle.testzip()
                if corrupt:
                    raise ValueError(f"Corrupt Spider ZIP member: {corrupt}")
                bundle.extractall(directory)
            (directory / "download.json").write_text(json.dumps({"url": SPIDER_URL, "sha256": sha256(archive)}))
            matches = list(directory.rglob("train_spider.parquet"))
        if len(matches) != 1:
            raise ValueError(f"Expected one train_spider.parquet under {directory}, found {len(matches)}")
        return matches[0]


def prepare_tasks(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select task groups deterministically and resolve every selected SQLite database before serving."""
    import pandas as pd

    kind = config["agent"]
    custom = config.get("dataset")
    if kind == "rag":
        dataset = Path(custom) if custom else DEFAULT_DATA_DIR / "dataset_tiny.parquet"
        if dataset.name == "dataset_tiny.parquet":
            ensure_example_data(dataset.parent, insecure=config["insecure_download"])
        frame = pd.read_parquet(dataset)
        required = {"id", "question", "answer"}
    elif kind == "sql":
        dataset = Path(custom) if custom else prepare_spider(CACHE / "sql", config["insecure_download"])
        frame = pd.read_parquet(dataset)
        required = {"question", "query", "db_id"}
    else:
        dataset = Path(custom) if custom else CACHE / "q20" / "q20_nouns.csv"
        if not custom:
            dataset.parent.mkdir(parents=True, exist_ok=True)
            source = EXAMPLES / "tinker" / "q20_nouns.csv"
            if not dataset.exists():
                shutil.copyfile(source, dataset)
            elif sha256(source) != sha256(dataset):
                raise ValueError("Cached q20_nouns.csv differs from repository source; specify --dataset explicitly")
        frame = pd.read_csv(dataset)
        required = {"answer", "category"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{kind} dataset requires columns {sorted(required)}")
    if frame[list(required)].isnull().any().any():
        raise ValueError("Dataset has null values in required fields")
    if "id" not in frame:
        frame["id"] = [f"{kind}-{index}" for index in range(len(frame))]
    if len(frame) < config["tasks"]:
        raise ValueError(f"Dataset has {len(frame)} rows, fewer than --tasks {config['tasks']}")
    selected = frame.sample(n=config["tasks"], random_state=config["seed"])
    rows = selected.to_dict("records")
    database_metadata = {}
    tasks = []
    for row in rows:
        task = {key: str(row[key]) for key in required | {"id"}}
        if kind == "sql":
            db_id = task["db_id"]
            if Path(db_id).name != db_id or db_id in {".", ".."}:
                raise ValueError(f"Invalid db_id: {db_id}")
            database_root = Path(config["sql_database_dir"]) if config.get("sql_database_dir") else dataset.parent
            spider_root = database_root.parent if database_root.name == "database" else database_root
            database = (spider_root / "database" / db_id / f"{db_id}.sqlite").resolve()
            if not database.is_file():
                raise ValueError(f"Missing original training database: {database}; use --sql-database-dir")
            task.update(database=str(database), spider_dir=str(spider_root.resolve()), answer=task["query"])
            if str(database) not in database_metadata:
                database_metadata[str(database)] = sha256(database)
        elif kind == "q20":
            task["question"] = f"Identify the hidden entity in category: {task['category']}."
        tasks.append(task)
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("Selected task IDs must be unique")
    return tasks, {
        "path": str(dataset.resolve()),
        "sha256": sha256(dataset),
        "rows": len(frame),
        "databases": database_metadata,
    }


if __name__ == "__main__":
    import argparse

    from rag_data import add_download_argument

    parser = argparse.ArgumentParser(description="Prepare real task data without starting a model or Agent.")
    parser.add_argument("--agent", choices=("rag", "sql", "q20"), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--sql-database-dir")
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", type=Path, required=True, help="Fresh preparation metadata directory.")
    add_download_argument(parser)
    args = parser.parse_args()
    if args.tasks < 1:
        parser.error("tasks must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"Preparation output: {args.output.resolve()}", flush=True)
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2))
    try:
        tasks, metadata = prepare_tasks(configuration)
        (args.output / "selected_tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2))
        (args.output / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2))
    except BaseException:
        shutil.rmtree(args.output)
        raise
