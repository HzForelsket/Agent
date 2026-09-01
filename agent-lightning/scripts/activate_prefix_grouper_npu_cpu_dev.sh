#!/usr/bin/env bash

_NPU_CPU_DEV_VENV="/opt/agent-npu-cpu-dev"
_NPU_CPU_DEV_ASCEND_ROOT="/usr/local/Ascend"

if [[ ! -f "${_NPU_CPU_DEV_VENV}/bin/activate" ]]; then
    echo "NPU CPU development environment is not installed at ${_NPU_CPU_DEV_VENV}." >&2
    return 1
fi
if [[ ! -f "${_NPU_CPU_DEV_ASCEND_ROOT}/ascend-toolkit/set_env.sh" ]]; then
    echo "CANN toolkit environment is missing under ${_NPU_CPU_DEV_ASCEND_ROOT}." >&2
    return 1
fi
if [[ ! -f "${_NPU_CPU_DEV_ASCEND_ROOT}/nnal/atb/set_env.sh" ]]; then
    echo "CANN NNAL environment is missing under ${_NPU_CPU_DEV_ASCEND_ROOT}." >&2
    return 1
fi

# shellcheck disable=SC1091
source "${_NPU_CPU_DEV_VENV}/bin/activate"
# shellcheck disable=SC1091
source "${_NPU_CPU_DEV_ASCEND_ROOT}/ascend-toolkit/set_env.sh"
# shellcheck disable=SC1091
source "${_NPU_CPU_DEV_ASCEND_ROOT}/nnal/atb/set_env.sh"
export SOC_VERSION="ascend910b1"

unset _NPU_CPU_DEV_VENV _NPU_CPU_DEV_ASCEND_ROOT

