# prefix-grouper-npu

`prefix-grouper-npu` 0.1.1 is an optional AscendC extension for compact shared-prefix
attention on Atlas A2 / Ascend 910B. It targets CANN 9.0.0, PyTorch 2.10.0 and
torch-npu 2.10.0, supports BF16 TND tensors with head dimension 128, and does
not provide a CPU fallback.

The package stores one prefix K/V per group. Each response suffix attends to
the shared prefix and its own causal suffix range. The backward kernel writes
each compact K/V gradient once and accumulates all response contributions to a
shared prefix.

## Native NPU build

On the NPU server, activate its Python 3.10 environment with PyTorch 2.10.0
and torch-npu 2.10.0, then build and install locally:

```bash
cd /home/huangzhong/Agent/PrefixGrouper/npu_ops
export ASCEND_HOME_PATH="$HOME/Ascend/cann-9.0.0"
bash scripts/build_wheel.sh
```

Import, schema discovery and Meta shape inference can be checked without an
NPU. Numerical correctness and performance results require a real 910B with a
matching driver and are never inferred from device-free checks.

`build_wheel.sh` automatically installs the newly built wheel using the active
Python's `python -m pip install --no-deps --force-reinstall`. It prints the target
interpreter and keeps the wheel in the build output's `dist` directory.
After installation it configures the installed custom OPP in the build process.
The validation entrypoints configure it again before starting Python, so no
manual sourcing of a staging directory's `set_env.bash` is required.

The scripts use the active Python environment; activate the environment with
PyTorch 2.10.0 and torch-npu 2.10.0 before invoking them. CANN is selected by
`ASCEND_HOME_PATH`, or discovered under `~/Ascend`. Native scripts never activate
the CPU development environment or enter proot. The compiler's `version.info` must
report exactly 9.0.0. For an explicit installation path, use:

```bash
export ASCEND_HOME_PATH="$HOME/Ascend/cann-9.0.0"
source scripts/activate.sh
```

Set the path to the actual toolkit directory containing `compiler/version.info`.
`activate.sh` requires the installed operator wheel; `build_wheel.sh` loads CANN
without importing the operator package. Native builds select the current host
architecture (`aarch64` or `x86_64`) for CANN headers and the OPP installer.
Cross-compilation is disabled; an aarch64 NPU server builds its own aarch64 wheel.

`activate.sh` locates the wheel through the current Python's package metadata,
without importing torch or initializing the NPU runtime. It sets both
`ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH` to the installed vendor and its
`op_api/lib` directory. The generated OPP environment script embeds the build
staging path, so it is not sourced from the installed wheel.
For Python commands launched directly from your shell, first run
`source scripts/activate.sh`. Running a build via `bash` cannot modify the parent
shell's environment. The native and CPU proot validation entrypoints already
source this script in their own environments.

## CPU proot development

Only this entrypoint selects the local Ubuntu 22.04 proot, its
`/opt/agent-npu-cpu-dev` Python environment and its CANN installation:

```bash
cd /home/huangzhong/Agent/PrefixGrouper/npu_ops
bash scripts/run_cpu_dev.sh build
bash scripts/run_cpu_dev.sh check
```

`build` builds and installs the wheel inside proot. `check` runs only the existing
plan and schema/Meta tests. It never runs hardware correctness or benchmarks.
The fixed project proot wrapper, its rootfs and the project path inside that
rootfs must already be available. It does not install development dependencies.

Native build outputs are under `build/native/<architecture>`; proot outputs are
under `build/proot/<architecture>`. Each has separate CMake, staged Python sources, setuptools and
wheel directories. Do not install the proot wheel on the NPU server; build there
using its own environment. No hardware validation is performed on this CPU host.

## Interface

```python
from prefix_grouper_npu import build_shared_prefix_plan, shared_prefix_attention

plan = build_shared_prefix_plan(
    prefix_lens=[128], suffix_lens=[64, 65], group_sizes=[2], device=q.device
)
out = shared_prefix_attention(q, k, v, plan)
```

`q` has shape `[T, Hq, 128]`; `k` and `v` have shape `[T, Hkv, 128]`.
All tensors must be contiguous BF16 tensors on one NPU, and `Hq % Hkv == 0`.
Scale computation and validation, softmax, and kernel accumulation use FP32.
PyTorch and the generated CANN ACLNN scalar interfaces require a host `double`
parameter; it only transports the FP32 scale and does not introduce FP64 tensor
computation. BF16 inputs, outputs and gradients are retained.
The compact token order for each group is one prefix followed by every suffix.
Prefix queries use a causal prefix slice. A suffix query uses a full shared
prefix slice plus a causal slice over only its own suffix.

There is no CPU implementation, FP16 mode, dropout, determinism guarantee or
distributed communication. The Meta implementation only provides schema/shape
inference and does not execute attention.

## Validation

On a matching Atlas A2 / 910B, after the native build and installation, capture
the environment log and correctness results with:

```bash
bash /home/huangzhong/Agent/PrefixGrouper/npu_ops/scripts/run_910b_validation.sh \
  /path/to/result-directory
```

This entrypoint requires a usable NPU and fails if none is available. It runs
only `test_npu_correctness.py`, with logs and pytest cache in the result directory.
It does not run benchmarks or profiler collection automatically.

The correctness test uses an FP32 CPU reference that physically concatenates
the prefix into every suffix K/V sequence. Autograd therefore sums each copied
prefix contribution back into the compact reference gradient. Completion
requires cosine similarity at least 0.999, output max absolute error at most
0.05, and gradient max absolute error at most 0.1 for every case.
