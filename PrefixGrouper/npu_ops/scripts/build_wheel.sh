#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/opp/project"
BUILD_DIR="${PROJECT_DIR}/build_out"
STAGE_DIR="${BUILD_DIR}/wheel_stage"
CANN_ROOT="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"

if [[ "${CANN_ROOT}" != "/usr/local/Ascend/cann-9.0.0" ]]; then
    echo "CANN 9.0.0 is required; got ASCEND_HOME_PATH=${CANN_ROOT}" >&2
    exit 2
fi
if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
    echo "Python 3.10 is required" >&2
    exit 2
fi

set +u
source "${CANN_ROOT}/bin/setenv.bash"
set -u
cmake --preset default -S "${PROJECT_DIR}"
cmake --build "${BUILD_DIR}" --target binary -j"${BUILD_JOBS:-2}"
cmake --build "${BUILD_DIR}" --target package -j"${BUILD_JOBS:-2}"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"
"${BUILD_DIR}/custom_opp_ubuntu_x86_64.run" --quiet --install-path="${STAGE_DIR}"

rm -rf "${ROOT_DIR}/build" "${ROOT_DIR}/prefix_grouper_npu.egg-info"
PREFIX_GROUPER_NPU_OPP_ROOT="${STAGE_DIR}" \
    python -m pip wheel --no-deps --no-build-isolation --wheel-dir "${ROOT_DIR}/dist" "${ROOT_DIR}"
