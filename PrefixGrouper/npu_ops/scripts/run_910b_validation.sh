#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${1:?usage: run_910b_validation.sh RESULT_DIR}"
mkdir -p "${RESULT_DIR}"
RESULT_DIR="$(cd "${RESULT_DIR}" && pwd -P)"
echo "Validation output: ${RESULT_DIR}"

source "${ROOT_DIR}/scripts/activate.sh"

{
    echo "command: $0 ${RESULT_DIR}"
    date --iso-8601=seconds
    echo "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
    python - <<'PY'
import torch
import torch_npu
import prefix_grouper_npu
import sys

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("prefix_grouper_npu:", prefix_grouper_npu.__version__)
print("npu_available:", torch.npu.is_available())
if not torch.npu.is_available():
    raise RuntimeError("A real Ascend NPU is required; use run_cpu_dev.sh check for device-free checks.")
print("device:", torch.npu.get_device_name(0))
PY
} | tee "${RESULT_DIR}/environment.log"

cd "${RESULT_DIR}"
python -m pytest -vv -s -o cache_dir="${RESULT_DIR}/pytest-cache" \
    "${ROOT_DIR}/tests/test_npu_correctness.py" \
    2>&1 | tee "${RESULT_DIR}/correctness.log"
