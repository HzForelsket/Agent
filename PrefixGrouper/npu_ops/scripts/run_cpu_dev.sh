#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-}"
case "${ACTION}" in
    build|check) ;;
    *) echo "usage: bash scripts/run_cpu_dev.sh {build|check}" >&2; exit 2 ;;
esac

exec "$HOME/.codex/skills/proot-ubuntu2204/scripts/proot_ubuntu2204.sh" -- \
    /usr/bin/env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C.UTF-8 /bin/bash -c '
        set -euo pipefail
        root_dir="$1"
        action="$2"
        source /opt/agent-npu-cpu-dev/bin/activate
        export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
        export PREFIX_GROUPER_NPU_BUILD_DIR="${root_dir}/build/proot"
        cd /tmp
        if [[ "${action}" == build ]]; then
            bash "${root_dir}/scripts/build_wheel.sh"
            python -m pip install --no-deps --force-reinstall \
                "${PREFIX_GROUPER_NPU_BUILD_DIR}/dist/"prefix_grouper_npu-*.whl
        else
            source "${root_dir}/scripts/activate.sh"
            python -m pytest -q -o cache_dir="${root_dir}/build/proot/pytest-cache" \
                "${root_dir}/tests/test_plan.py" "${root_dir}/tests/test_schema.py"
        fi
    ' bash "${ROOT_DIR}" "${ACTION}"
