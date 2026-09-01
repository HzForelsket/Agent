#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" || "${VIRTUAL_ENV}" != "/opt/agent-npu-cpu-dev" ]]; then
    source /opt/agent-npu-cpu-dev/bin/activate
fi
set +u
source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash
set -u
export SOC_VERSION=ascend910b1

PREFIX_GROUPER_NPU_VENDOR_ROOT="$(python - <<'PY'
from pathlib import Path
import prefix_grouper_npu

vendor = Path(prefix_grouper_npu.__file__).resolve().parent / "_opp" / "vendors" / "prefix_grouper_npu"
print(vendor)
PY
)"
export ASCEND_CUSTOM_OPP_PATH="${PREFIX_GROUPER_NPU_VENDOR_ROOT}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
