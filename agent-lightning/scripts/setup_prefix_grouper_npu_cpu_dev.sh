#!/usr/bin/env bash
set -euo pipefail

CANN_VERSION="9.0.0"
SOC_VERSION="ascend910b1"
VENV_DIR="/opt/agent-npu-cpu-dev"
CACHE_DIR="/var/cache/agent-npu-cpu-dev"
STATE_DIR="/var/lib/agent-npu-cpu-dev"
ASCEND_ROOT="/usr/local/Ascend"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DRY_RUN=0

usage() {
    echo "Usage: $0 [--dry-run]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unsupported argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

print_command() {
    printf "+"
    printf " %q" "$@"
    printf "\n"
}

run() {
    print_command "$@"
    if [[ ${DRY_RUN} -eq 0 ]]; then
        "$@"
    fi
}

fail() {
    echo "$*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || fail "Required file is missing after installation: $1"
}

source_vendor_environment() {
    set +u
    # shellcheck disable=SC1090
    source "$1"
    set -u
}

if [[ "$(id -u)" -ne 0 ]]; then
    fail "Run this script as root inside the isolated Ubuntu 22.04 proot."
fi

if [[ ! -r /etc/os-release ]]; then
    fail "Cannot identify the operating system: /etc/os-release is missing."
fi

# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]] || \
    fail "This setup requires Ubuntu 22.04; found ${ID:-unknown} ${VERSION_ID:-unknown}."
[[ "$(uname -m)" == "x86_64" ]] || fail "This setup is pinned to Atlas A2 development on x86_64."
[[ -f /usr/bin/python3.10 ]] || fail "The proot must provide /usr/bin/python3.10."
if [[ ! -r /usr/bin/python3.10 || ! -w /usr/local || ! -x /usr/local ]]; then
    fail "The proot cannot translate access checks; use PRoot 5.4.0 or newer (faccessat2 support)."
fi

if [[ -d "${ASCEND_ROOT}" ]]; then
    while IFS= read -r installed_dir; do
        installed_version="$(basename -- "${installed_dir}")"
        installed_version="${installed_version#cann-}"
        if [[ "${installed_version}" != "${CANN_VERSION}" ]]; then
            fail "Found unsupported CANN toolkit ${installed_version}; expected exactly ${CANN_VERSION}."
        fi
    done < <(find "${ASCEND_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'cann-*' -print)
fi

if [[ -d "${ASCEND_ROOT}/ascend-toolkit" ]]; then
    while IFS= read -r installed_dir; do
        installed_version="$(basename -- "${installed_dir}")"
        if [[ "${installed_version}" != "${CANN_VERSION}" ]]; then
            fail "Found unsupported CANN toolkit ${installed_version}; expected exactly ${CANN_VERSION}."
        fi
    done < <(
        find "${ASCEND_ROOT}/ascend-toolkit" -mindepth 1 -maxdepth 1 -type d \
            ! -name latest ! -name latest-* -print
    )
fi

run apt-get update
run apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    jq \
    libnuma-dev \
    python3.10-dev \
    python3.10-venv \
    wget
run mkdir -p "${CACHE_DIR}" "${STATE_DIR}"

download_installer() {
    local filename="$1"
    local url="$2"
    local target="${CACHE_DIR}/${filename}"

    if [[ -s "${target}" ]]; then
        echo "Using cached installer: ${target}"
        return
    fi
    run wget \
        --header="Referer: https://www.hiascend.com/" \
        --continue \
        --progress=dot:giga \
        --output-document "${target}" \
        "${url}"
}

install_component() {
    local component="$1"
    local filename="$2"
    local url="$3"
    shift 3
    local marker="${STATE_DIR}/${component}-${CANN_VERSION}"
    local installer="${CACHE_DIR}/${filename}"

    if [[ -f "${marker}" ]]; then
        echo "CANN ${component} ${CANN_VERSION} is already configured."
        return
    fi
    download_installer "${filename}" "${url}"
    run chmod 0755 "${installer}"
    run "${installer}" "$@"
    run touch "${marker}"
}

CANN_BASE_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0"
# Huawei's installer defines --quiet as non-interactive EULA acceptance.
install_component \
    toolkit \
    "Ascend-cann-toolkit_9.0.0_linux-x86_64.run" \
    "${CANN_BASE_URL}/Ascend-cann-toolkit_9.0.0_linux-x86_64.run" \
    --quiet \
    --force \
    --full

if [[ ${DRY_RUN} -eq 0 ]]; then
    require_file "${ASCEND_ROOT}/ascend-toolkit/set_env.sh"
    source_vendor_environment "${ASCEND_ROOT}/ascend-toolkit/set_env.sh"
fi

install_component \
    910b-ops \
    "Ascend-cann-910b-ops_9.0.0_linux-x86_64.run" \
    "${CANN_BASE_URL}/Ascend-cann-910b-ops_9.0.0_linux-x86_64.run" \
    --quiet \
    --force \
    --install
install_component \
    nnal \
    "Ascend-cann-nnal_9.0.0_linux-x86_64.run" \
    "${CANN_BASE_URL}/Ascend-cann-nnal_9.0.0_linux-x86_64.run" \
    --quiet \
    --force \
    --install

if [[ ${DRY_RUN} -eq 0 ]]; then
    require_file "${ASCEND_ROOT}/nnal/atb/set_env.sh"
    source_vendor_environment "${ASCEND_ROOT}/nnal/atb/set_env.sh"
fi
export SOC_VERSION

if [[ ! -f "${VENV_DIR}/bin/python" ]]; then
    run /usr/bin/python3.10 -m venv "${VENV_DIR}"
fi
run "${VENV_DIR}/bin/python" -m pip install --upgrade "pip>=25.1,<26"
stack_python="${VENV_DIR}/bin/python"
if [[ ${DRY_RUN} -eq 1 && ! -x "${stack_python}" ]]; then
    stack_python="/usr/bin/python3.10"
fi
stack_args=(
    "${stack_python}"
    "${REPO_ROOT}/scripts/prefix_grouper_stack.py"
    --backend npu
    --cpu-dev
)
if [[ ${DRY_RUN} -eq 1 ]]; then
    stack_args+=(--dry-run)
    print_command "${stack_args[@]}"
    "${stack_args[@]}"
else
    run "${stack_args[@]}"
fi

echo
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "NPU CPU development environment dry-run completed."
else
    echo "NPU CPU development environment is configured at ${VENV_DIR}."
    echo "Activate it inside proot with:"
    echo "  source ${REPO_ROOT}/scripts/activate_prefix_grouper_npu_cpu_dev.sh"
fi
