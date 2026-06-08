import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch


def _runner():
    path = Path(__file__).resolve().parents[2] / "runners/[9].py"
    spec = spec_from_file_location("runner_9_tests", path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _batch(batch_size: int = 3, timesteps: int = 4) -> dict[str, torch.Tensor]:
    return {
        "s2": torch.ones(batch_size, timesteps, 11, 16, 16),
        "s1": torch.ones(batch_size, timesteps, 2, 16, 16),
        "climate": torch.empty(batch_size, timesteps, 0, 16, 16),
        "doy": torch.tensor([[30.0, 120.0, 210.0, 300.0]]).repeat(batch_size, 1),
        "s2_available": torch.ones(batch_size, timesteps),
        "s1_available": torch.ones(batch_size, timesteps),
        "climate_available": torch.zeros(batch_size, timesteps),
        "s2_mask": torch.ones(batch_size, timesteps, 16, 16),
        "s1_mask": torch.ones(batch_size, timesteps, 16, 16),
    }


def test_checkpoint_policy_is_fixed_epoch_sweep() -> None:
    runner = _runner()
    assert runner.CHECKPOINT_EPOCHS == [1, 2, 4, 8, 12]
    assert runner.EPOCHS == 12


def test_default_arms_match_viewdrop_screen() -> None:
    runner = _runner()
    assert runner.DEFAULT_ARMS == [
        "A_lr_control_generic",
        "B_generic_viewdrop",
        "C_mixed_viewdrop",
        "D_mixed_viewdrop_consistency",
        "E_spatial_viewdrop_consistency",
    ]
    assert runner.ARM_SPECS["A_lr_control_generic"].viewdrop is False
    assert runner.ARM_SPECS["D_mixed_viewdrop_consistency"].consistency is True
    assert runner.ARM_SPECS["E_spatial_viewdrop_consistency"].spatial_tokens is True


def test_sensor_off_s2_corruption_does_not_mutate_clean_batch() -> None:
    runner = _runner()
    clean = _batch()
    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(42)
    degraded, time_keep = runner._corrupt_batch(clean, "sensor_off_s2", generator)
    assert torch.count_nonzero(degraded["s2"]) == 0
    assert torch.count_nonzero(degraded["s2_mask"]) == 0
    assert torch.count_nonzero(degraded["s2_available"]) == 0
    assert torch.count_nonzero(clean["s2"]) == clean["s2"].numel()
    assert torch.count_nonzero(clean["s2_mask"]) == clean["s2_mask"].numel()
    assert torch.count_nonzero(clean["s2_available"]) == clean["s2_available"].numel()
    assert time_keep.all()


def test_temporal_drop_keeps_at_least_two_timesteps() -> None:
    runner = _runner()
    clean = _batch(batch_size=64)
    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(7)
    degraded, time_keep = runner._corrupt_batch(clean, "temporal_drop_50", generator)
    assert torch.all(degraded["s1_available"].sum(dim=1) >= 2)
    assert torch.all(degraded["s2_available"].sum(dim=1) >= 2)
    assert torch.all(degraded["s1_available"][:, 0] == 1)
    assert torch.all(degraded["s2_available"][:, 0] == 1)
    assert time_keep.shape == (64, 4)
    assert torch.all(time_keep[:, 0])
    assert torch.all(time_keep.sum(dim=1) >= 2)


def test_combined_corruption_keeps_no_s2_and_drops_time_for_s1() -> None:
    runner = _runner()
    clean = _batch(batch_size=64)
    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(11)
    degraded, time_keep = runner._corrupt_batch(clean, "s2_off_tdrop50", generator)
    assert torch.count_nonzero(degraded["s2_available"]) == 0
    assert torch.all(degraded["s1_available"].sum(dim=1) >= 2)
    assert torch.any(degraded["s1_available"][:, 1:] == 0)
    assert time_keep.shape == (64, 4)
    assert not time_keep.all()


def test_mixed_pretrain_dataset_rejects_shape_mismatch() -> None:
    runner = _runner()
    ds_a = type(
        "D",
        (),
        {"shapes": (4, 11, 2), "root": object(), "__len__": lambda self: 1, "__getitem__": lambda self, idx: {}},
    )()
    ds_b = type(
        "D",
        (),
        {"shapes": (4, 10, 2), "root": object(), "__len__": lambda self: 1, "__getitem__": lambda self, idx: {}},
    )()
    with pytest.raises(ValueError, match="identical tensor shapes"):
        runner.MixedPretrainDataset([ds_a, ds_b])


def test_batch_loss_supports_degraded_context_clean_target() -> None:
    runner = _runner()
    device = torch.device("cpu")
    spec = runner.ARM_SPECS["B_generic_viewdrop"]
    model = runner._make_model(
        spec, type("Shapes", (), {"shapes": type("S", (), {"s2_channels": 11, "s1_channels": 2})()})()
    )
    batch = _batch(batch_size=2)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    mask = runner._target_mask(spec, 2, 4, device, generator)
    loss, diagnostics = runner._batch_loss(model, batch, spec, mask, "sensor_off_s2", generator)
    assert torch.isfinite(loss)
    assert diagnostics["jepa_loss"] > 0
    loss.backward()


def test_time_keep_returned_from_corrupt_batch_for_clean() -> None:
    runner = _runner()
    clean = _batch(batch_size=3)
    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(0)
    _, time_keep = runner._corrupt_batch(clean, "clean", generator)
    assert time_keep.dtype == torch.bool
    assert time_keep.all()


def test_time_keep_forwarded_into_consistency_loss() -> None:
    runner = _runner()
    device = torch.device("cpu")
    spec = runner.ARM_SPECS["D_mixed_viewdrop_consistency"]
    model = runner._make_model(
        spec, type("Shapes", (), {"shapes": type("S", (), {"s2_channels": 11, "s1_channels": 2})()})()
    )
    batch = _batch(batch_size=2)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    mask = runner._target_mask(spec, 2, 4, device, generator)
    mode = "temporal_drop_50"
    clean = runner._model_batch(batch)
    corrupted, time_keep = runner._corrupt_batch(batch, mode, generator)
    degraded = runner._model_batch(corrupted)
    output = model.forward_views(
        target_mask=mask, context=(degraded if spec.viewdrop else clean), target=clean, context_time_keep=time_keep
    )
    assert time_keep is not None
    assert not time_keep.all()
    assert torch.isfinite(runner.spatial_jepa_loss(output))


def test_consistency_loss_excludes_dropped_timesteps() -> None:
    runner = _runner()
    device = torch.device("cpu")
    spec = runner.ARM_SPECS["D_mixed_viewdrop_consistency"]
    model = runner._make_model(
        spec, type("Shapes", (), {"shapes": type("S", (), {"s2_channels": 11, "s1_channels": 2})()})()
    )
    model.eval()
    batch = _batch(batch_size=2, timesteps=4)
    clean = runner._model_batch(batch)
    degraded = runner._clone_view(batch)
    degraded["s2"][:, 2:] = 0
    degraded["s1"][:, 2:] = 0
    degraded["s2_available"][:, 2:] = 0
    degraded["s1_available"][:, 2:] = 0
    time_keep = torch.ones((2, 4), dtype=torch.bool)
    time_keep[:, 2:] = False
    loss_masked = runner._embedding_consistency_loss(model, clean, degraded, time_keep=time_keep)
    assert loss_masked < 0.01, f"Masked loss should be ~0 (identical data on kept timesteps), got {loss_masked}"
    loss_no_mask = runner._embedding_consistency_loss(model, clean, degraded, time_keep=None)
    assert loss_no_mask > loss_masked + 0.1, f"Unmasked loss ({loss_no_mask}) should exceed masked ({loss_masked})"


def test_viewdrop_mode_includes_clean() -> None:
    runner = _runner()
    spec = runner.ARM_SPECS["B_generic_viewdrop"]
    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(42)
    seen: set[str] = set()
    for step in range(200):
        seen.add(runner._viewdrop_mode(step, spec, generator))
    assert "clean" in seen
    assert "sensor_off_s2" in seen
    assert "temporal_drop_50" in seen
    assert "s2_off_tdrop50" in seen
