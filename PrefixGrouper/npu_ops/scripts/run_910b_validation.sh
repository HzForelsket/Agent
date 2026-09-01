#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${1:?usage: run_910b_validation.sh RESULT_DIR}"
mkdir -p "${RESULT_DIR}"

source /opt/agent-npu-cpu-dev/bin/activate
set +u
source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash
set -u
export SOC_VERSION=ascend910b1

{
    echo "command: $0 ${RESULT_DIR}"
    date --iso-8601=seconds
    python - <<'PY'
import torch
import torch_npu
import prefix_grouper_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("prefix_grouper_npu:", prefix_grouper_npu.__version__)
print("npu_available:", torch.npu.is_available())
if torch.npu.is_available():
    print("device:", torch.npu.get_device_name(0))
PY
} | tee "${RESULT_DIR}/environment.log"

python -m pytest -vv -s "${ROOT_DIR}/tests/test_npu_correctness.py" \
    2>&1 | tee "${RESULT_DIR}/correctness.log"

python "${ROOT_DIR}/benchmarks/benchmark_shared_prefix_attention.py" \
    --output "${RESULT_DIR}/benchmark.json" \
    --trace-dir "${RESULT_DIR}/profiler" \
    2>&1 | tee "${RESULT_DIR}/benchmark.log"
