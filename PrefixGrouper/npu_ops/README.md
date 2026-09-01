# prefix-grouper-npu

`prefix-grouper-npu` is an optional AscendC extension for compact shared-prefix
attention on Atlas A2 / Ascend 910B. It targets CANN 9.0.0, PyTorch 2.10.0 and
torch-npu 2.10.0, supports BF16 TND tensors with head dimension 128, and does
not provide a CPU fallback.

The package stores one prefix K/V per group. Each response suffix attends to
the shared prefix and its own causal suffix range. The backward kernel writes
each compact K/V gradient once and accumulates all response contributions to a
shared prefix.

Build only inside the project Ubuntu 22.04 proot:

```bash
source /opt/agent-npu-cpu-dev/bin/activate
cd /home/huangzhong/Agent/PrefixGrouper/npu_ops
./scripts/build_wheel.sh
python -m pip install --no-deps --force-reinstall dist/prefix_grouper_npu-*.whl
```

Import, schema discovery and Meta shape inference can be checked without an
NPU. Numerical correctness and performance results require a real 910B with a
matching driver and are never inferred from device-free checks.

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
The compact token order for each group is one prefix followed by every suffix.
Prefix queries use a causal prefix slice. A suffix query uses a full shared
prefix slice plus a causal slice over only its own suffix.

There is no CPU implementation, FP16 mode, dropout, determinism guarantee or
distributed communication. The Meta implementation only provides schema/shape
inference and does not execute attention.

## Validation

Device-free checks:

```bash
cd /root
python -m pytest -q \
  /home/huangzhong/Agent/PrefixGrouper/npu_ops/tests/test_plan.py \
  /home/huangzhong/Agent/PrefixGrouper/npu_ops/tests/test_schema.py
```

On a matching Atlas A2 / 910B, capture the version log, full correctness matrix,
benchmark measurements and profiler traces with:

```bash
/home/huangzhong/Agent/PrefixGrouper/npu_ops/scripts/run_910b_validation.sh \
  /path/to/result-directory
```

The correctness test uses an FP32 CPU reference that physically concatenates
the prefix into every suffix K/V sequence. Autograd therefore sums each copied
prefix contribution back into the compact reference gradient. Completion
requires cosine similarity at least 0.999, output max absolute error at most
0.05, and gradient max absolute error at most 0.1 for every case.
