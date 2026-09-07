#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/opp/project"
OUTPUT_DIR="${PREFIX_GROUPER_NPU_BUILD_DIR:-${ROOT_DIR}/build/native}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd -P)"
echo "Build output: ${OUTPUT_DIR}"
BUILD_DIR="${OUTPUT_DIR}/opp"
STAGE_DIR="${BUILD_DIR}/wheel_stage"
SOURCE_DIR="${OUTPUT_DIR}/source"
source "${ROOT_DIR}/scripts/cann_env.sh"
if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
    echo "Python 3.10 is required" >&2
    exit 2
fi

cmake --preset default -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
    -DASCEND_PYTHON_EXECUTABLE="$(command -v python)"
cmake --build "${BUILD_DIR}" --target binary -j"${BUILD_JOBS:-2}"
cmake --build "${BUILD_DIR}" --target package -j"${BUILD_JOBS:-2}"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"
"${BUILD_DIR}/custom_opp_ubuntu_x86_64.run" --quiet --install-path="${STAGE_DIR}"

# Stage only build inputs so setuptools cannot reuse another environment's artifacts.
mkdir -p "${SOURCE_DIR}/prefix_grouper_npu" "${SOURCE_DIR}/csrc"
cp "${ROOT_DIR}/setup.py" "${ROOT_DIR}/pyproject.toml" \
    "${ROOT_DIR}/MANIFEST.in" "${ROOT_DIR}/README.md" "${SOURCE_DIR}/"
cp "${ROOT_DIR}/prefix_grouper_npu/"*.py "${SOURCE_DIR}/prefix_grouper_npu/"
cp "${ROOT_DIR}/csrc/"*.cpp "${SOURCE_DIR}/csrc/"
PREFIX_GROUPER_NPU_OPP_ROOT="${STAGE_DIR}" \
    python -m pip wheel --no-deps --no-build-isolation \
        --wheel-dir "${OUTPUT_DIR}/dist" "${SOURCE_DIR}"
