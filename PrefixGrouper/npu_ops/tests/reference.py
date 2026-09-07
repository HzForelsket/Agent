from __future__ import annotations

import torch


def dense_lse_reference(q, k, prefix_lens, suffix_lens, group_sizes, scale=None):
    """Independent compact mask oracle for the small LSE/lifecycle cases."""
    total = q.shape[0]
    allowed = torch.zeros((total, total), dtype=torch.bool)
    offset = suffix_index = 0
    for prefix_len, group_size in zip(prefix_lens, group_sizes, strict=True):
        prefix_start = offset
        prefix_end = offset + prefix_len
        allowed[offset:prefix_end, offset:prefix_end] = torch.ones(
            (prefix_len, prefix_len), dtype=torch.bool
        ).tril()
        offset = prefix_end
        for _ in range(group_size):
            length = suffix_lens[suffix_index]
            suffix_index += 1
            end = offset + length
            allowed[offset:end, prefix_start:prefix_end] = True
            allowed[offset:end, offset:end] = torch.ones((length, length), dtype=torch.bool).tril()
            offset = end
    assert offset == total and suffix_index == len(suffix_lens)
    keys = k.float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    scale = (torch.tensor(q.shape[-1], dtype=torch.float32).rsqrt() if scale is None
             else torch.tensor(scale, dtype=torch.float32))
    scores = torch.einsum("thd,shd->hts", q.float(), keys) * scale
    return scores.masked_fill(~allowed.unsqueeze(0), float("-inf")).logsumexp(-1).T.contiguous()


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
    ratio = q.shape[1] // k.shape[1]
    k_heads = k.repeat_interleave(ratio, dim=1)
    v_heads = v.repeat_interleave(ratio, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k_heads) * scale
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probabilities, v_heads)


def materialized_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefix_lens: tuple[int, ...],
    suffix_lens: tuple[int, ...],
    group_sizes: tuple[int, ...],
    scale: float | None = None,
) -> torch.Tensor:
    scale = (
        torch.tensor(q.shape[-1], dtype=torch.float32, device="cpu").rsqrt().item()
        if scale is None
        else torch.tensor(scale, dtype=torch.float32, device="cpu").item()
    )
    outputs: list[torch.Tensor] = []
    token_offset = 0
    suffix_index = 0
    for prefix_len, group_size in zip(prefix_lens, group_sizes, strict=True):
        prefix_slice = slice(token_offset, token_offset + prefix_len)
        prefix_q = q[prefix_slice]
        prefix_k = k[prefix_slice]
        prefix_v = v[prefix_slice]
        causal = torch.triu(torch.ones(prefix_len, prefix_len, dtype=torch.bool), diagonal=1)
        outputs.append(_attention(prefix_q, prefix_k, prefix_v, causal, scale))
        token_offset += prefix_len

        for _ in range(group_size):
            suffix_len = suffix_lens[suffix_index]
            suffix_index += 1
            suffix_slice = slice(token_offset, token_offset + suffix_len)
            # cat intentionally materializes the traditional duplicated-prefix baseline.
            expanded_k = torch.cat((prefix_k, k[suffix_slice]), dim=0)
            expanded_v = torch.cat((prefix_v, v[suffix_slice]), dim=0)
            mask = torch.zeros(suffix_len, prefix_len + suffix_len, dtype=torch.bool)
            mask[:, prefix_len:] = torch.triu(
                torch.ones(suffix_len, suffix_len, dtype=torch.bool), diagonal=1
            )
            outputs.append(_attention(q[suffix_slice], expanded_k, expanded_v, mask, scale))
            token_offset += suffix_len
    return torch.cat(outputs, dim=0)
