#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${1:?usage: run_910b_benchmark.sh NEW_RESULT_DIR [benchmark arguments]}"
shift
mkdir -p "$(dirname "${RESULT_DIR}")"
mkdir "${RESULT_DIR}"
RESULT_DIR="$(cd "${RESULT_DIR}" && pwd -P)"
echo "Benchmark output: ${RESULT_DIR}"

# Run the public-API correctness gate in a fresh process before any timing.
bash "${ROOT_DIR}/scripts/run_910b_validation.sh" "${RESULT_DIR}/validation"
source "${ROOT_DIR}/scripts/activate.sh"
cd "${RESULT_DIR}"
python -u "${ROOT_DIR}/benchmarks/benchmark_shared_prefix_attention.py" \
    "$@" --output "${RESULT_DIR}/benchmark.json" \
    2>&1 | tee "${RESULT_DIR}/benchmark.log"
