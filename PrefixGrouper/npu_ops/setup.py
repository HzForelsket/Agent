from __future__ import annotations

import os
import pathlib
import re
import shutil

import torch
import torch_npu
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension


ROOT = pathlib.Path(__file__).resolve().parent
OPP_ROOT = pathlib.Path(os.environ.get("PREFIX_GROUPER_NPU_OPP_ROOT", ""))
VENDOR = OPP_ROOT / "vendors" / "prefix_grouper_npu"
if not VENDOR.is_dir():
    raise RuntimeError(
        "PREFIX_GROUPER_NPU_OPP_ROOT must point to the staged OPP root; "
        "run scripts/build_wheel.sh instead of invoking setup.py directly"
    )

match = re.match(r"(\d+)\.(\d+)", torch.__version__)
if not match:
    raise RuntimeError(f"cannot parse PyTorch version: {torch.__version__}")
version_macro = f"-DCURRENT_VERSION=V{match.group(1)}R{match.group(2)}"
torch_npu_root = pathlib.Path(torch_npu.__file__).resolve().parent
op_api = VENDOR / "op_api"
cann_root = pathlib.Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0"))


class build_py(_build_py):
    def run(self) -> None:
        super().run()
        target = pathlib.Path(self.build_lib) / "prefix_grouper_npu" / "_opp" / "vendors"
        shutil.copytree(OPP_ROOT / "vendors", target, dirs_exist_ok=True)


extension = NpuExtension(
    name="prefix_grouper_npu._C",
    sources=["csrc/shared_prefix_attention.cpp"],
    include_dirs=[
        str(op_api / "include"),
        str(cann_root / "x86_64-linux" / "include"),
        str(torch_npu_root / "include" / "third_party" / "acl" / "inc"),
        str(torch_npu_root / "include" / "third_party" / "op-plugin"),
        str(torch_npu_root / "include" / "third_party" / "op-plugin" / "op_plugin" / "include"),
    ],
    library_dirs=[str(op_api / "lib")],
    libraries=["cust_opapi"],
    extra_compile_args=[version_macro, "-O2"],
    extra_link_args=["-Wl,-rpath,$ORIGIN/_opp/vendors/prefix_grouper_npu/op_api/lib"],
)

setup(
    name="prefix-grouper-npu",
    version="0.1.0",
    description="Shared-prefix AscendC attention for PrefixGrouper",
    python_requires=">=3.10,<3.11",
    install_requires=["torch==2.10.0", "torch-npu==2.10.0"],
    packages=find_packages(),
    ext_modules=[extension],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=os.getenv("USE_NINJA") == "1"),
        "build_py": build_py,
    },
    include_package_data=True,
    zip_safe=False,
)
