#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/opp/project"
BUILD_DIR="${PROJECT_DIR}/build_out"
STAGE_DIR="${BUILD_DIR}/wheel_stage"
source "${ROOT_DIR}/scripts/cann_env.sh"
if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
    echo "Python 3.10 is required" >&2
    exit 2
fi

cmake --preset default -S "${PROJECT_DIR}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
    -DASCEND_PYTHON_EXECUTABLE="$(command -v python)"
cmake --build "${BUILD_DIR}" --target binary -j"${BUILD_JOBS:-2}"
cmake --build "${BUILD_DIR}" --target package -j"${BUILD_JOBS:-2}"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"
"${BUILD_DIR}/custom_opp_ubuntu_x86_64.run" --quiet --install-path="${STAGE_DIR}"

rm -rf "${ROOT_DIR}/build" "${ROOT_DIR}/prefix_grouper_npu.egg-info"
PREFIX_GROUPER_NPU_OPP_ROOT="${STAGE_DIR}" \
    python -m pip wheel --no-deps --no-build-isolation --wheel-dir "${ROOT_DIR}/dist" "${ROOT_DIR}"
