import torch

from reference import dense_lse_reference, materialized_reference


def test_small_reference_matches_uniform_attention_and_shared_gradients():
    metadata = ((1, 2), (1, 2, 1), (1, 2))
    q = torch.zeros(7, 2, 128, dtype=torch.float32, requires_grad=True)
    k = torch.zeros(7, 1, 128, dtype=torch.float32, requires_grad=True)
    v = torch.ones(7, 1, 128, dtype=torch.float32, requires_grad=True)
    counts = torch.tensor([1, 2, 1, 2, 3, 4, 3], dtype=torch.float32)
    lse = dense_lse_reference(q, k, *metadata)
    torch.testing.assert_close(lse, counts.log()[:, None].expand(7, 2))
    out = materialized_reference(q, k, v, *metadata)
    torch.testing.assert_close(out, torch.ones_like(out))
    out.sum().backward()
    expected_dv = torch.tensor([3, 1, 29, 17, 7, 1, 2], dtype=torch.float32) / torch.tensor(
        [1, 1, 6, 6, 6, 2, 3], dtype=torch.float32
    )
    torch.testing.assert_close(v.grad, expected_dv[:, None, None].expand_as(v))
    torch.testing.assert_close(q.grad, torch.zeros_like(q))
    torch.testing.assert_close(k.grad, torch.zeros_like(k))
