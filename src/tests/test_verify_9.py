import torch

from src.temp.verify_9 import (
    _spatial_token_metrics,
    downstream_embedding,
    shuffled_content_target,
)


def test_temporal_drop_full_displacement_is_not_forced_to_zero() -> None:
    clean = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]])
    keep = torch.tensor([[True, True, False, False]])
    degraded = clean * keep.unsqueeze(-1)
    full_clean = downstream_embedding(clean)
    degraded_embedding = downstream_embedding(degraded, keep)
    retained_clean = downstream_embedding(clean, keep)
    assert not torch.allclose(full_clean, degraded_embedding)
    torch.testing.assert_close(retained_clean, degraded_embedding)


def test_content_shuffle_preserves_masks_and_valid_token_counts() -> None:
    batch = {
        "s2": torch.arange(4.0)[:, None, None, None, None],
        "s1": torch.arange(4.0)[:, None, None, None, None] + 10.0,
        "s2_mask": torch.arange(4.0)[:, None, None, None],
        "s1_mask": torch.arange(4.0)[:, None, None, None] + 10.0,
        "s2_available": torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
        "s1_available": torch.tensor([[1.0], [1.0], [0.0], [1.0]]),
        "doy": torch.arange(4.0)[:, None],
    }
    shuffled = shuffled_content_target(batch, torch.Generator(device="cpu").manual_seed(42))
    assert not torch.equal(shuffled["s2"], batch["s2"])
    torch.testing.assert_close(shuffled["s2_mask"], batch["s2_mask"])
    torch.testing.assert_close(shuffled["s1_mask"], batch["s1_mask"])
    torch.testing.assert_close(shuffled["s2_available"], batch["s2_available"])
    torch.testing.assert_close(shuffled["s1_available"], batch["s1_available"])
    torch.testing.assert_close(shuffled["doy"], batch["doy"])
    valid_before = (batch["s2_available"] + batch["s1_available"]) > 0
    valid_after = (shuffled["s2_available"] + shuffled["s1_available"]) > 0
    torch.testing.assert_close(valid_after, valid_before)


def test_spatial_token_diagnostics_detect_position_only_collapse() -> None:
    position = torch.randn(1, 2, 2, 3, 8).expand(6, -1, -1, -1, -1).clone()
    available = torch.ones(6, 2, 2, 3, dtype=torch.bool)
    collapsed, collapsed_residual = _spatial_token_metrics(position, position, available)
    content = position + torch.randn_like(position)
    sensitive, sensitive_residual = _spatial_token_metrics(content, position, available)
    assert collapsed["spatial_same_slot_inter_sample_cosine"] > 0.999
    assert collapsed["spatial_clean_zero_token_cosine"] > 0.999
    assert collapsed["spatial_residual_variance"] == 0.0
    assert collapsed["spatial_residual_effective_rank"] == 0.0
    assert torch.count_nonzero(collapsed_residual) == 0
    assert sensitive["spatial_same_slot_inter_sample_cosine"] < collapsed["spatial_same_slot_inter_sample_cosine"]
    assert sensitive["spatial_clean_zero_token_cosine"] < collapsed["spatial_clean_zero_token_cosine"]
    assert sensitive["spatial_residual_variance"] > 0.0
    assert sensitive["spatial_residual_effective_rank"] > 1.0
    assert torch.count_nonzero(sensitive_residual) > 0
