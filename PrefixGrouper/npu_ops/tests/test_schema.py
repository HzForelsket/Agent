import pytest
import torch

from prefix_grouper_npu._extension import load_extension


@pytest.mark.parametrize("dim", [1, 15, 16, 17, 127, 128, 129, 256, 257, 513])
def test_schema_and_meta_shape_inference(dim) -> None:
    load_extension()
    q = torch.empty((17, 6, dim), device="meta", dtype=torch.bfloat16)
    k = torch.empty((17, 2, dim), device="meta", dtype=torch.bfloat16)
    metadata = torch.empty((17,), device="meta", dtype=torch.int32)
    scale = torch.tensor(dim, dtype=torch.float32, device="cpu").rsqrt().item()
    out, lse = torch.ops.prefix_grouper_npu.shared_prefix_attention_forward(
        q, k, k, metadata, metadata, metadata, metadata, metadata, scale, [8], [9], [1]
    )
    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16
    assert lse.shape == (17, 6)
    assert lse.dtype == torch.float32

    dq, dk, dv = torch.ops.prefix_grouper_npu.shared_prefix_attention_backward(
        out, q, k, k, out, lse, metadata, metadata, metadata, metadata, metadata, scale
    )
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == k.shape
