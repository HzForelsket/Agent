from __future__ import annotations

import ctypes
import os
from pathlib import Path
from threading import Lock

import torch


_LOCK = Lock()
_LOADED = False


def load_extension() -> None:
    global _LOADED
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return
        package_dir = Path(__file__).resolve().parent
        vendor_root = package_dir / "_opp" / "vendors"
        vendor = vendor_root / "prefix_grouper_npu"
        op_api = vendor / "op_api" / "lib" / "libcust_opapi.so"
        extension = next(iter(sorted(package_dir.glob("_C*.so"))), None)
        if not op_api.is_file() or extension is None:
            raise RuntimeError(
                "prefix-grouper-npu native artifacts are missing; build and install the wheel "
                "with scripts/build_wheel.sh inside the project Ubuntu 22.04 proot"
            )
        current = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
        entries = [entry for entry in current.split(":") if entry]
        if str(vendor) not in entries:
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([str(vendor), *entries])
        ctypes.CDLL(str(op_api), mode=ctypes.RTLD_GLOBAL)
        torch.ops.load_library(str(extension))
        _LOADED = True
