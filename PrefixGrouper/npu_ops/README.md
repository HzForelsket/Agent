# prefix-grouper-npu

`prefix-grouper-npu` 0.2.0 is an optional AscendC extension for compact shared-prefix
attention on Atlas A2 / Ascend 910B. It targets CANN 9.0.0, PyTorch 2.10.0 and
torch-npu 2.10.0, supports BF16 TND tensors with a positive dynamic head dimension, and does
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

`build` builds and installs the wheel inside proot. `check` runs the plan,
schema/Meta tests and one small analytical reference check. It never runs
hardware correctness or benchmarks.
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

`q` has shape `[T, Hq, D]`; `k` and `v` have shape `[T, Hkv, D]`.
All tensors must be contiguous BF16 tensors on one NPU, and `Hq % Hkv == 0`.
Heads and D must be positive. D need not be aligned and has no fixed 256 limit;
the default scale is computed from the actual D. Token offsets must fit int32,
and padded tensor/workspace sizes must fit the checked int64 address range.
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

## Kernel design

The forward and backward operators use CANN's MIX_AIC_1_2 Matmul service:
Cube computes matrix products and AIV performs FP32 softmax and accumulation.
Host tiling selects an aligned square block from input size and actual UB,
L1, L0 and core resources. Q, KV and D are split into blocks, including tails.
The budget includes all live local tensors, queues and the three registered
Matmul objects; the launch is no longer capped at 20 cores.

Two GM operand/result slots per AIV separate producer and consumer lifetimes.
The next operand block is copied while the current asynchronous Matmul runs;
the next product is then launched before Vector consumes the previous result.
This pipeline is used by D reductions and output-D block updates. Single-block
workloads only execute startup/drain and have no steady-state D overlap.
Input and result queues have two slots; dependency-specific events and Vector
barriers replace whole-pipeline barriers. Scalar work remains for row statistics,
causal tails and pack gather indices; the implementation is not fully scalar-free.

Forward maintains online FP32 max/sum statistics and a padded FP32 output
accumulator, converting local probabilities to BF16 for PV. Backward computes
delta once with a Vector operator. Separate Q and KV block owners recompute
local probabilities and reduce their entire gradients without global atomics.
Sequence/group end metadata limits KV-gradient queries to the contributing
sequence or group. Prefix K/V storage and GQA heads are never materialized.

FP32 accumulators have D rounded to 16, so each row starts on a 64-byte boundary.
A separate Vector pack operator assigns aligned compact output ranges to cores,
gathers padded rows and writes BF16 tails safely. LSE uses independent padded
FP32 rows, followed by the native strided-to-contiguous copy. Intermediate
score/probability storage is bounded by block size, not sequence length squared.
The linear FP32 accumulators and GM staging introduce extra memory/traffic;
dynamic-D support does not imply a measured speed or memory improvement.

The internal ACLNN forward/backward outputs are padded FP32 accumulators;
the public PyTorch outputs remain compact BF16 with compact FP32 LSE.
Plans add `sequence_end` and `group_end` int32 tensors. Rebuild/install the 0.2.0
wheel and OPP together; there is no old-schema adapter or serial-kernel switch.
The rewritten kernels require fresh hardware validation. Device-free compilation
and schema checks cannot establish their numerical accuracy or pipeline overlap.

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

The hardware suite retains the eight numerical cases and input-contract checks,
and covers:

- Three LSE boundary cases with `T * Hq` equal to 15, 16 and 17. The test captures
  the actual FP32 LSE saved by the public autograd function, checks it against an
  independent dense masked CPU oracle (`rtol=1e-5`, `atol=1e-6`), and checks the
  BF16 output and all three gradients using the original thresholds.
- One same-process A/B/A case. It explicitly overwrites the same Q/K/V buffers,
  resets leaf gradients between calls, switches input values and plan metadata
  for B, and returns to A's cached plan. Each call independently checks output,
  LSE and gradients against the CPU reference. A1/A2 differences are recorded;
  bitwise determinism is not required.
- One public-API autograd integration case: Q/K/V projections, GQA attention,
  output projection, residual and FP32 loss. It uses two prefix groups and checks
  hidden-state gradients and all four projection-weight gradients. Cosine must
  be at least 0.999, output absolute error at most 0.02, gradient absolute error
  at most 0.01, and loss must satisfy `rtol=0.02`, `atol=0.001`.
- Dynamic D values 1, 15, 16, 17, 127, 128, 129, 256, 257 and 513 with two groups,
  GQA, actual saved LSE and all gradients; D=17 also uses an explicit scale.
- Pipeline sequence boundaries 63, 64, 65 and 129 with unaligned D=17.
  The A/B/A lifecycle case uses D=129 to cover padded rows and D tails.

Attention inputs and projection weights originate as BF16 values; loss targets
and CPU reference tensor computation are FP32. Metrics include the worst
token/head/component index and its actual and
expected values. The analytical CPU reference check verifies uniform attention,
LSE and shared-prefix gradient accumulation; it is not NPU execution evidence.

The integration case covers composition with PyTorch autograd, not a model
training step. Agent Lightning/VERL can select this package explicitly as
described below; its model-level correctness still requires a separate run on
real NPU hardware.

## Optional Agent Lightning / VERL Backend

For VERL 0.9.0 and Transformers 5.5.4, configure the model attention backend:

```yaml
agentlightning:
  prefix_grouper:
    enabled: true
actor_rollout_ref:
  model:
    use_remove_padding: false
    use_fused_kernels: false
    override_config:
      attn_implementation: sdpa
      prefix_grouper_npu_backend: custom
```

`prefix_grouper_npu_backend` accepts `fusion` (the default) or `custom`.
`fusion` keeps the existing duplicated-prefix BNSD `npu_fusion_attention` path.
`custom` packs valid grouped tokens into TND before any prefix duplication or
GQA head expansion, calls this package once per attention layer, and restores
the padded model layout. Packing/restoration remain in the autograd graph;
padding gradients cannot reach the compact tokens. Actor and reference workers
receive the same model setting. Calls without a PrefixGrouper still use the
original attention implementation, including rollout and ungrouped batches.

Custom mode requires BF16, equal positive Q/K/V head dimensions, positive prefix/suffix lengths,
Hq divisible by Hkv, zero attention dropout, and full causal attention without
sliding windows, softcap or KV-cache decoding. Unsupported custom calls fail;
there is no automatic fallback to fusion. Use FSDP/FSDP2 with Ulysses size 1.

Install the architecture-native operator wheel in every worker environment,
then source `scripts/activate.sh` before starting the Ray processes. Fusion mode
does not require the operator package. Neither mode changes the global attention
backend of a separate baseline process.

Both existing model benchmark entrypoints accept an explicit comparison flag:

- `agent-lightning/scripts/benchmark_prefix_grouper.py` (`pg-verl-ppa`):
  `--device npu --npu-attention-backend custom` or `--npu-attention-backend fusion`.
- `agent-lightning/scripts/benchmark_prefix_grouper_2wikimqa_e2e.py`
  (`pg-2wikimqa-e2e`): `--device npu --mode prefix_grouper --npu-attention-backend custom`.
  A separate `--mode baseline` process must not select `custom`; its config does
  not enable PrefixGrouper or inject the backend override.

The selected backend is recorded in result metadata. Keep all other workload
and environment settings identical when comparing fusion/custom. These flags
do not authorize a model benchmark run on this device-free development machine.

### PyTorch JIT Deprecation

With the pinned torch/torch-npu 2.10.0 stack, importing `torch_npu` can load
Inductor, then `torch.fx.experimental.optimization`, then `torch.utils.mkldnn`.
The latter uses `torch.jit.script_method` and emits its deprecation warning.
This happens before the custom operator is loaded and is not an operator
correctness failure. No project code uses `script_method`; the warning does
not require replacing this package's autograd function with `torch.compile`.
Do not change the pinned dependencies or globally suppress warnings to hide it.

## Small Native Performance Run

`pg-ascend-shared-prefix-attention` (Ascend shared-prefix attention operator
microbenchmark) runs without loading a model. By default it measures forward
only; add `--backward` to also measure backward and forward+backward. On a real
910B, use a fresh output directory and this small, explicit workload:

```bash
bash scripts/run_910b_benchmark.sh \
  build/native/aarch64/performance-0.2.0 \
  --prefix 1 --suffixes 1 63 --hq 2 --hkv 2 --warmup 2 --iterations 10 \
  --backward
```

The wrapper configures the installed OPP automatically and runs the hardware
tests in a separate Python process first. Any failure prevents timing. It then
runs the existing benchmark in the active Python environment, without proot.
Both paths start from the same compact BF16 Q/K/V and produce compact outputs
and gradients. Fusion gathers repeated prefix K/V rows using a precomputed index
inside each forward call; autograd accumulates those copies back into compact
K/V gradients on the NPU inside backward. Q is already in sequence order and is
not copied. Both paths reuse metadata (custom plan, fusion indices, sequence
lengths and causal mask), whose construction is outside timing.

Before timing, the benchmark compares outputs (cosine at least 0.999, max
absolute error at most 0.05). With `--backward`, it compares compact dQ/dK/dV
using the same random upstream gradient (cosine at least 0.999, max absolute
error at most 0.1). Any failed check prevents timing; gradient metrics are saved
under `gradient_correctness`.
The directory must not already exist. `validation/` contains environment and
correctness logs; `benchmark.log` captures errors; `benchmark.json` records
configuration before input allocation and is updated after each paired round.
The same saves generate `benchmark.md`, with per-mode median, P25/P75/P95,
sample counts, incremental allocation peaks, correctness and measurement scope.
Report writing is outside each pair of samples. Incomplete runs are explicitly
marked; measurement failures save the error and any samples already collected.
When invoking the Python benchmark directly, `--output results.json` also writes
`results.md`; use `--output-markdown PATH` to choose a different Markdown path.
Both output paths must be new and differ from each other.
Only `status=complete` indicates that all requested stages finished.

Warmup and measurement alternate custom/fusion and fusion/custom order each
round to reduce systematic ordering bias. Each path gets the requested number
of warmups and samples. Samples use synchronized host wall-clock latency,
including Python dispatch and synchronization overhead; they are not device-only
kernel times or steady-state throughput. Outputs/gradients stay alive until the
timer stops, then are released before the other path runs. Raw samples are kept
without outlier removal; short runs provide only descriptive percentiles.

The result uses `schema_version=2` with records under `timings.<mode>` for
`shared_prefix_attention` and `npu_fusion_attention_compact`. Each record contains
`samples_ms`, median/mean/min/max, P25/P75/P95, `memory_samples`,
`peak_increment_bytes` and `status`. The mode is `forward`, and with `--backward`
also `backward` and `forward_backward`. The former pre-materialized baseline and
its old result keys are removed because their timing boundary differs.

Backward-only timing prepares and synchronizes a fresh forward graph before
starting the clock. Forward+backward includes fresh forward and gradient
computation together. Both use `torch.autograd.grad` without retained graphs or
leaf `.grad` accumulation. Fusion K/V materialization is included in forward;
its gradient reduction is included in backward. Profiling starts only after
all measurements and covers the same call boundaries in every selected mode.

Memory peaks are reset per sample before graph preparation and reported relative
to the allocation already resident at that point. Backward memory therefore
includes saved tensors from its untimed forward. Both paths' static metadata
remain resident and their sizes are recorded separately. These are incremental
allocated-memory measurements, not isolated process totals or allocator-reserved
memory; no allocator-cache flush is performed between paths.

This is an attention-path microbenchmark on prebuilt compact input, not an
Agent Lightning without PrefixGrouper baseline. It does not measure model
packing/restoration, training or end-to-end speedup. No benchmark is run by
`run_cpu_dev.sh` or by the correctness-only validation entrypoint.

### NPU profiling

Add `--trace-dir NEW_DIRECTORY` to collect CPU/NPU profiles after all ordinary
timing samples. This uses the torch-npu 2.10.0 profiler with CANN 9.0.0 on the
real NPU server. For example, from `npu_ops` in the activated NPU environment:

```bash
source scripts/activate.sh
python benchmarks/benchmark_shared_prefix_attention.py \
  --prefix 1 --suffixes 1 63 --hq 2 --hkv 2 --warmup 2 --iterations 10 \
  --backward --trace-dir build/native/profile-attention \
  --profile-steps 1 --profile-aic-metrics PipeUtilization
```

The same flags can be passed through `run_910b_benchmark.sh`; relative trace
paths then resolve inside its result directory. The trace directory must not
exist. If `--output` is omitted, JSON and Markdown reports are created as
`benchmark.json` and `benchmark.md` inside the trace directory. Existing
correctness gates still run before any timing or profiling.

Profiling records CPU and NPU activities, input shapes and memory allocations
at Level1, requesting `PipeUtilization` hardware counters by default.
`--profile-aic-metrics` selects one group: `PipeUtilization`, `Memory`,
`MemoryL0`, `MemoryUB`, `ResourceConflictRatio`, or `None` for no AI Core
counters. These groups are alternatives, not an automatic sweep. Actual
available counters are determined by the real device and exported CSV columns.
Add `--profile-with-stack` to collect Python stacks. Profiling perturbs execution;
use the unprofiled timing records for latency comparisons.

`--profile-steps` defaults to one isolated capture per operator and selected
mode. Each capture repeats the requested `--warmup` outside the profiler, then
collects one call in a fresh context. Backward-only forward construction and
synchronization happen before that context; forward+backward captures both.
No saved graph is reused. Named `pg_attention/<operator>/<mode>/step_NNN`
ranges identify the call and a nested range identifies final synchronization.
Native framework events expose fusion `index_select` and gradient reduction,
custom Attention/Delta/Pack launches, and their device tasks.

Artifacts are organized under
`<trace-dir>/<mode>/<operator>/step_NNN/<profiler-run>/ASCEND_PROFILER_OUTPUT/`:

- `trace_view.json`: CPU/NPU timeline for dispatch gaps, launches and overlap.
- `kernel_details.csv`: device task durations and available AI Core counters.
- `operator_details.csv`: framework operator timings and associated device work.

Other native memory reports and raw collection data are retained by the
profiler. Allocation records describe tensor memory, not internal GM/UB traffic;
for backward-only profiles, forward allocations predate the capture.

The benchmark's `profiling` object records configuration, per-capture status,
artifact paths, device-task counts and exported kernel column names. Markdown
contains the same capture list with links. The script path and SHA-256 are
recorded to distinguish results generated by different source revisions.
Export is synchronous and completion requires all three files plus nonempty
kernel records. A collection/export failure marks the run `failed_profiling`,
preserving the completed unprofiled timings and earlier captures. Profiles are
saved after every capture; they are never added to `samples_ms`.
