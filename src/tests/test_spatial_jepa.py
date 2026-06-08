import copy

import torch

from src.core.jepa import JepaBatchMasks
from src.core.spatial_jepa import PooledTemporalJepaModel, RadarOpticalPooledEncoder, SpatialPatchTokenizer, SpatialTokenEncoder, SpatialTokenJepaModel


def _batch() -> dict[str, torch.Tensor]:
    doy = torch.tensor([[20.0, 110.0, 200.0, 290.0]])
    return {
        "s2": torch.randn(1, 4, 11, 16, 16),
        "s1": torch.randn(1, 4, 2, 16, 16),
        "climate": torch.empty(1, 4, 0, 16, 16),
        "doy": doy,
        "s2_doy": doy,
        "s1_doy": doy + 1,
        "s2_elapsed_days": doy - doy[:, :1],
        "s1_elapsed_days": doy - doy[:, :1] + 1,
        "s2_available": torch.ones(1, 4),
        "s1_available": torch.ones(1, 4),
        "climate_available": torch.zeros(1, 4),
        "s2_mask": torch.ones(1, 4, 16, 16),
        "s1_mask": torch.ones(1, 4, 16, 16),
    }


def _spatial_model() -> SpatialTokenJepaModel:
    model = SpatialTokenJepaModel(
        s2_channels=11,
        s1_channels=2,
        model_dim=32,
        num_layers=1,
        num_heads=4,
        predictor_dim=32,
        predictor_layers=1,
        dropout=0.0,
    )
    return model.eval()


def test_masked_tokens_receive_distinct_position_identity() -> None:
    model = _spatial_model()
    identity = model.context_encoder.position_identity(**_batch())
    assert not torch.allclose(identity[:, 1, :, 0], identity[:, 1, :, 1])
    assert not torch.allclose(identity[:, 1, :, 0], identity[:, 2, :, 0])


def test_cloud_mask_blocks_masked_pixel_values_from_embeddings() -> None:
    model = _spatial_model()
    batch = _batch()
    batch["s1"].zero_()
    batch["s1_mask"].zero_()
    batch["s1_available"].zero_()
    batch["s2_mask"][:, :, :4, :4] = 0
    changed = copy.deepcopy(batch)
    changed["s2"][:, :, :, :4, :4] = 10000.0
    with torch.no_grad():
        original_embedding = model.encode(**batch)
        changed_embedding = model.encode(**changed)
    torch.testing.assert_close(original_embedding, changed_embedding)


def test_unavailable_target_tokens_are_excluded_from_loss_mask() -> None:
    model = _spatial_model()
    batch = _batch()
    batch["s2_mask"][:, :, :4, :4] = 0
    batch["s1_mask"][:, :, :4, :4] = 0
    target_mask = torch.zeros(1, 4, 16, dtype=torch.bool)
    target_mask[:, 1, 0] = True
    output = model(target_mask=target_mask, **batch)
    assert not bool(output.local_mask[:, 1, :, 0].any())


def test_token_clear_fraction_threshold_rejects_mostly_cloudy_tokens() -> None:
    tokenizer = SpatialPatchTokenizer(in_channels=1, model_dim=4, min_clear_fraction=0.50)
    values = torch.ones(1, 1, 1, 16, 16)
    pixel_mask = torch.ones(1, 1, 16, 16)
    pixel_mask[:, :, :4, :4] = 0
    pixel_mask[:, :, 0, 0] = 1
    pixel_mask[:, :, :4, 4:8] = 0
    pixel_mask[:, :, :2, 4:8] = 1
    _, available, clear_fraction = tokenizer(values, pixel_mask)
    assert float(clear_fraction[0, 0, 0]) == 1.0 / 16.0
    assert not bool(available[0, 0, 0])
    assert float(clear_fraction[0, 0, 1]) == 0.50
    assert bool(available[0, 0, 1])


def test_spatial_temporal_drop_isolation() -> None:
    model = _spatial_model()
    batch = _batch()
    changed = copy.deepcopy(batch)
    changed["s2"][:, 2] = 10000.0
    changed["s1"][:, 2] = -10000.0
    masks = JepaBatchMasks(time_keep=torch.tensor([[1.0, 1.0, 0.0, 1.0]]))
    with torch.no_grad():
        original_embedding = model.encode(**batch, masks=masks)
        changed_embedding = model.encode(**changed, masks=masks)
    torch.testing.assert_close(original_embedding, changed_embedding)


def test_pooled_temporal_drop_isolation() -> None:
    model = PooledTemporalJepaModel(
        s2_channels=11,
        s1_channels=2,
        model_dim=32,
        num_layers=1,
        num_heads=4,
        predictor_layers=1,
        dropout=0.0,
    ).eval()
    batch = _batch()
    changed = copy.deepcopy(batch)
    changed["s2"][:, 2] = 10000.0
    changed["s1"][:, 2] = -10000.0
    masks = JepaBatchMasks(time_keep=torch.tensor([[1.0, 1.0, 0.0, 1.0]]))
    with torch.no_grad():
        original_embedding = model.encode(**batch, masks=masks)
        changed_embedding = model.encode(**changed, masks=masks)
    torch.testing.assert_close(original_embedding, changed_embedding)


def test_pooled_forward_views_accepts_degraded_context_clean_target() -> None:
    model = PooledTemporalJepaModel(
        s2_channels=11,
        s1_channels=2,
        model_dim=32,
        num_layers=1,
        num_heads=4,
        predictor_layers=1,
        dropout=0.0,
    )
    clean = _batch()
    degraded = copy.deepcopy(clean)
    degraded["s2"].zero_()
    degraded["s2_available"].zero_()
    degraded["s2_mask"].zero_()
    target_mask = torch.tensor([[False, True, True, False]])
    output = model.forward_views(target_mask=target_mask, context=degraded, target=clean)
    loss = (output.local_pred * output.local_mask.unsqueeze(-1)).sum()
    loss.backward()
    assert output.local_pred.shape == output.local_target.shape
    assert torch.all(output.local_mask == target_mask)


def test_pooled_predictor_ignores_invalid_masked_timesteps() -> None:
    model = PooledTemporalJepaModel(
        s2_channels=11,
        s1_channels=2,
        model_dim=32,
        num_layers=1,
        num_heads=4,
        predictor_layers=1,
        dropout=0.0,
    ).eval()
    batch = _batch()
    batch["s2_available"][:, 1] = 0.0
    batch["s1_available"][:, 1] = 0.0
    batch["s2_mask"][:, 1] = 0.0
    batch["s1_mask"][:, 1] = 0.0
    mask_with_invalid = torch.tensor([[False, True, True, False]])
    mask_without_invalid = torch.tensor([[False, False, True, False]])
    with torch.no_grad():
        out1 = model(target_mask=mask_with_invalid, **batch)
        out2 = model(target_mask=mask_without_invalid, **batch)
    torch.testing.assert_close(
        out1.local_pred[:, 2],
        out2.local_pred[:, 2],
        rtol=1e-5,
        atol=1e-5,
        msg="Invalid pooled mask token leaked into predictor attention",
    )


def test_equal_observed_reflectance_with_different_clear_fractions_produces_equal_activations() -> None:
    tokenizer = SpatialPatchTokenizer(in_channels=1, model_dim=16)
    base = torch.ones(1, 1, 1, 16, 16) * 0.5
    clear = torch.ones(1, 1, 16, 16)
    half = torch.ones(1, 1, 16, 16)
    half[:, :, :8, :] = 0
    with torch.no_grad():
        base_tokens, base_avail, _ = tokenizer(base, clear)
        half_tokens, half_avail, _ = tokenizer(base, half)
    assert bool(base_avail[0, 0, 0])
    assert not bool(half_avail[0, 0, 0])
    assert not bool(half_avail[0, 0, 4])
    assert bool(half_avail[0, 0, 8])
    assert bool(half_avail[0, 0, 12])
    torch.testing.assert_close(base_tokens[0, 0, 0], half_tokens[0, 0, 8], rtol=0.01, atol=0.01)
    torch.testing.assert_close(base_tokens[0, 0, 1], half_tokens[0, 0, 9], rtol=0.01, atol=0.01)


def test_predictor_ignores_cloud_invalid_masked_tokens() -> None:
    """Cloud-invalid S2 queries masked by target_mask must not change valid predictions.

    The cloudy timestep itself is a masked target.  Its tokens are naturally
    unavailable (context_available=False, target_available=False) so
    local_mask=False there, and no mask token is injected.  The predictor
    receives the (zeroed) context token with predictor_valid=False, which the
    src_key_padding_mask excludes.  Changing the raw pixel values at that
    timestep must therefore not affect predictions for other valid targets.
    """
    model = SpatialTokenJepaModel(s2_channels=11, s1_channels=2, model_dim=32, num_layers=1, num_heads=4, predictor_dim=16, predictor_layers=1, dropout=0.0).eval()
    batch = _batch()
    # S2 entirely cloudy at timestep 1 (index 1) — S1 stays clear
    s2_cloudy = batch["s2_mask"].clone()
    s2_cloudy[:, 1] = 0.0
    # Make S2 at timestep 1 a masked target, plus some S1 tokens at timestep 2
    target_mask = torch.zeros(1, 4, 16, dtype=torch.bool)
    target_mask[:, 1, :4] = True   # S2 tokens here — naturally unavailable
    target_mask[:, 2, 8:12] = True # valid S1+S2 target — must stay invariant
    out1 = model(target_mask=target_mask, s2_mask=s2_cloudy, **{k: v for k, v in batch.items() if k != "s2_mask"})
    batch2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch2["s2"][:, 1] = 9999.0  # garbage values at cloudy timestep
    out2 = model(target_mask=target_mask, s2_mask=s2_cloudy, **{k: v for k, v in batch2.items() if k != "s2_mask"})
    # Valid predictions (timestep 2 S1 tokens) must be identical
    mask = out1.local_mask
    torch.testing.assert_close(out1.local_pred[mask], out2.local_pred[mask], rtol=1e-5, atol=1e-5, msg="Cloud-invalid masked tokens leaked into predictor attention")


def test_pooled_and_spatial_tokenizers_share_identical_cloud_validity() -> None:
    s2 = torch.randn(1, 4, 11, 16, 16)
    s2_mask = torch.ones(1, 4, 16, 16)
    s2_mask[:, :, 8:, :] = 0
    pooled = RadarOpticalPooledEncoder(s2_channels=11, s1_channels=2, model_dim=32)
    spatial = SpatialTokenEncoder(s2_channels=11, s1_channels=2, model_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    with torch.no_grad():
        pooled_avail = pooled.s2_tokenizer(s2, s2_mask)[1]
        spatial_avail = spatial.s2_tokenizer(s2, s2_mask)[1]
    torch.testing.assert_close(pooled_avail, spatial_avail)
