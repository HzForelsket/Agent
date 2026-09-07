#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cann_env.sh"

PREFIX_GROUPER_NPU_VENDOR_ROOT="$(python - <<'PY'
from importlib.metadata import distribution

vendor = distribution("prefix-grouper-npu").locate_file(
    "prefix_grouper_npu/_opp/vendors/prefix_grouper_npu"
).resolve()
if not (vendor / "op_api" / "lib" / "libcust_opapi.so").is_file():
    raise RuntimeError(f"Installed custom OPP library is missing under {vendor}; rebuild and install the wheel.")
print(vendor)
PY
)"
# The generated set_env.bash embeds the staging path; use the installed location.
case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
    *":${PREFIX_GROUPER_NPU_VENDOR_ROOT}:"*) ;;
    *) export ASCEND_CUSTOM_OPP_PATH="${PREFIX_GROUPER_NPU_VENDOR_ROOT}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}" ;;
esac
case ":${LD_LIBRARY_PATH:-}:" in
    *":${PREFIX_GROUPER_NPU_VENDOR_ROOT}/op_api/lib:"*) ;;
    *) export LD_LIBRARY_PATH="${PREFIX_GROUPER_NPU_VENDOR_ROOT}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
