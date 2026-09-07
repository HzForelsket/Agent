#!/usr/bin/env bash

_prefix_grouper_load_cann() {
    local root candidate env_script version had_nounset=0 status=0
    if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
        root="${ASCEND_HOME_PATH}"
    else
        for candidate in \
            "$HOME/Ascend/cann-9.0.0" \
            "$HOME/Ascend/ascend-toolkit/9.0.0" \
            "$HOME/Ascend/ascend-toolkit/latest" \
            "$HOME/Ascend"; do
            if [[ -f "${candidate}/compiler/version.info" ]]; then
                root="${candidate}"
                break
            fi
        done
    fi
    if [[ -z "${root:-}" || ! -f "${root}/compiler/version.info" ]]; then
        echo "Set ASCEND_HOME_PATH to the CANN 9.0.0 toolkit directory (containing compiler/version.info)." >&2
        return 2
    fi
    root="$(cd "${root}" && pwd -P)" || return
    version="$(sed -n 's/^Version=//p' "${root}/compiler/version.info" | tr -d '\r')"
    if [[ "${version}" != "9.0.0" ]]; then
        echo "CANN 9.0.0 is required; found ${version:-unknown} in ${root}." >&2
        return 2
    fi
    if [[ -f "${root}/bin/setenv.bash" ]]; then
        env_script="${root}/bin/setenv.bash"
    elif [[ -f "${root}/set_env.sh" ]]; then
        env_script="${root}/set_env.sh"
    else
        echo "CANN environment script is missing in ${root}." >&2
        return 2
    fi
    # CANN's environment script reads optional, potentially unset variables.
    [[ $- == *u* ]] && had_nounset=1
    set +u
    source "${env_script}" || status=$?
    if (( had_nounset )); then
        set -u
    fi
    (( status == 0 )) || return "${status}"
    export ASCEND_HOME_PATH="${root}"
    export SOC_VERSION=ascend910b1
}

_prefix_grouper_load_cann
