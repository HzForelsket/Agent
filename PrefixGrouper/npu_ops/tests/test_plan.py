import pytest
import torch

from prefix_grouper_npu import build_shared_prefix_plan, shared_prefix_attention


def test_plan_encodes_shared_prefix_and_suffix_ranges() -> None:
    plan = build_shared_prefix_plan([2], [2, 1], [2], device="cpu")
    assert plan.total_tokens == 5
    assert plan.prefix_start.tolist() == [0, 0, 0, 0, 0]
    assert plan.prefix_end.tolist() == [2, 2, 2, 2, 2]
    assert plan.sequence_start.tolist() == [0, 0, 2, 2, 4]
    assert build_shared_prefix_plan([2], [2, 1], [2], device="cpu") is plan


def test_plan_is_group_isolated() -> None:
    plan = build_shared_prefix_plan([1, 2], [1, 2, 1], [1, 2], device="cpu")
    assert plan.total_tokens == 7
    assert plan.prefix_start.tolist() == [0, 0, 2, 2, 2, 2, 2]
    assert plan.prefix_end.tolist() == [1, 1, 4, 4, 4, 4, 4]
    assert plan.sequence_start.tolist() == [0, 1, 2, 2, 4, 4, 6]


@pytest.mark.parametrize(
    ("prefix_lens", "suffix_lens", "group_sizes", "match"),
    [
        ([], [1], [1], "positive"),
        ([1], [0], [1], "positive"),
        ([1], [1], [0], "positive"),
        ([1, 1], [1], [1], "one entry per group"),
        ([1], [1, 1], [1], r"sum\(group_sizes\)"),
        ([1.5], [1], [1], "integer sequence"),
    ],
)
def test_invalid_plan_metadata_is_rejected(prefix_lens, suffix_lens, group_sizes, match) -> None:
    with pytest.raises(ValueError, match=match):
        build_shared_prefix_plan(prefix_lens, suffix_lens, group_sizes, device="cpu")


def test_attention_has_no_cpu_fallback() -> None:
    plan = build_shared_prefix_plan([1], [1, 1], [2], device="cpu")
    q = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    k = torch.empty((3, 1, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="NPU-only"):
        shared_prefix_attention(q, k, k, plan)
