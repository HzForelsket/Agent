#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/opp/project"
BUILD_ARCH="$(uname -m)"
case "${BUILD_ARCH}" in
    x86_64|aarch64) ;;
    *) echo "Unsupported native build architecture: ${BUILD_ARCH}" >&2; exit 2 ;;
esac
OUTPUT_DIR="${PREFIX_GROUPER_NPU_BUILD_DIR:-${ROOT_DIR}/build/native}/${BUILD_ARCH}"
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

# Regenerate OPP schemas, metadata and kernels together. CANN's generated
# ops-info JSON build rule does not track changes to its source INI.
if [[ -L "${BUILD_DIR}" ]]; then
    echo "Refusing to clean a symlinked OPP build directory: ${BUILD_DIR}" >&2
    exit 2
fi
if [[ -d "${BUILD_DIR}" ]]; then
    chmod -R u+w "${BUILD_DIR}"
    rm -rf -- "${BUILD_DIR}"
fi
cmake --preset default -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
    -DASCEND_PYTHON_EXECUTABLE="$(command -v python)"
cmake --build "${BUILD_DIR}" --target binary -j"${BUILD_JOBS:-2}"
cmake --build "${BUILD_DIR}" --target package -j"${BUILD_JOBS:-2}"

# The OPP installer leaves read-only directories in the staging tree.
if [[ -d "${STAGE_DIR}" ]]; then
    chmod -R u+w "${STAGE_DIR}"
fi
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"
shopt -s nullglob
OPP_INSTALLERS=("${BUILD_DIR}"/custom_opp_*_"${BUILD_ARCH}".run)
shopt -u nullglob
if [[ ${#OPP_INSTALLERS[@]} -ne 1 ]]; then
    echo "Expected exactly one ${BUILD_ARCH} OPP installer in ${BUILD_DIR}; found ${#OPP_INSTALLERS[@]}." >&2
    exit 2
fi
"${OPP_INSTALLERS[0]}" --quiet --install-path="${STAGE_DIR}"

# Stage only build inputs so setuptools cannot reuse another environment's artifacts.
# copytree preserves OPP permissions in setuptools' build directory as well.
if [[ -d "${SOURCE_DIR}" ]]; then
    chmod -R u+w "${SOURCE_DIR}"
fi
rm -rf "${SOURCE_DIR}"
mkdir -p "${SOURCE_DIR}/prefix_grouper_npu" "${SOURCE_DIR}/csrc"
cp "${ROOT_DIR}/setup.py" "${ROOT_DIR}/pyproject.toml" \
    "${ROOT_DIR}/MANIFEST.in" "${ROOT_DIR}/README.md" "${SOURCE_DIR}/"
cp "${ROOT_DIR}/prefix_grouper_npu/"*.py "${SOURCE_DIR}/prefix_grouper_npu/"
cp "${ROOT_DIR}/csrc/"*.cpp "${SOURCE_DIR}/csrc/"
WHEEL_DIR="$(mktemp -d "${OUTPUT_DIR}/wheel.XXXXXX")"
PREFIX_GROUPER_NPU_OPP_ROOT="${STAGE_DIR}" \
    python -m pip wheel --no-deps --no-build-isolation \
        --wheel-dir "${WHEEL_DIR}" "${SOURCE_DIR}"
shopt -s nullglob
WHEELS=("${WHEEL_DIR}"/prefix_grouper_npu-*.whl)
shopt -u nullglob
if [[ ${#WHEELS[@]} -ne 1 ]]; then
    echo "Expected exactly one newly built wheel in ${WHEEL_DIR}; found ${#WHEELS[@]}." >&2
    exit 2
fi
mkdir -p "${OUTPUT_DIR}/dist"
WHEEL_PATH="${OUTPUT_DIR}/dist/$(basename "${WHEELS[0]}")"
mv -f "${WHEELS[0]}" "${WHEEL_PATH}"
rmdir "${WHEEL_DIR}"
echo "Installing into Python environment: $(python -c 'import sys; print(sys.executable)')"
python -m pip install --no-deps --force-reinstall "${WHEEL_PATH}"
source "${ROOT_DIR}/scripts/activate.sh"
echo "Configured installed custom OPP: ${PREFIX_GROUPER_NPU_VENDOR_ROOT}"
