#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cann_env.sh"

PREFIX_GROUPER_NPU_VENDOR_ROOT="$(python - <<'PY'
from pathlib import Path
import prefix_grouper_npu

vendor = Path(prefix_grouper_npu.__file__).resolve().parent / "_opp" / "vendors" / "prefix_grouper_npu"
print(vendor)
PY
)"
export ASCEND_CUSTOM_OPP_PATH="${PREFIX_GROUPER_NPU_VENDOR_ROOT}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
