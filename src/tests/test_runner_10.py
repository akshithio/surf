import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _runner():
    path = Path(__file__).resolve().parents[2] / "runners/[10].py"
    spec = importlib.util.spec_from_file_location("runner_10_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _batch() -> dict[str, torch.Tensor]:
    return {
        "s2": torch.arange(4.0)[:, None, None, None, None].repeat(1, 1, 1, 2, 2),
        "s1": torch.arange(4.0)[:, None, None, None, None].repeat(1, 1, 1, 2, 2) + 10.0,
        "climate": torch.empty(4, 1, 0, 2, 2),
        "s2_mask": torch.arange(4.0)[:, None, None, None].repeat(1, 1, 2, 2),
        "s1_mask": torch.arange(4.0)[:, None, None, None].repeat(1, 1, 2, 2) + 10.0,
        "climate_mask": torch.zeros(4, 1, 2, 2),
        "s2_available": torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
        "s1_available": torch.tensor([[1.0], [1.0], [0.0], [1.0]]),
        "climate_available": torch.zeros(4, 1),
        "doy": torch.arange(4.0)[:, None],
    }


def test_revised_runner_10_arm_contract() -> None:
    runner = _runner()
    assert runner.DEFAULT_ARMS == [
        "A_mixed_viewdrop_reference",
        "B_mixed_viewdrop_sensor_dropout",
        "C_generic_step_matched",
        "D_mixed_size_matched",
        "E_spatial_no_consistency",
        "F_spatial_content_regularized",
    ]
    assert runner.SEEDS == [42, 43]
    assert runner.FIXED_UPDATES == 16_600
    assert runner.CHECKPOINT_STEPS == [1_000, 3_000, 10_000, 16_600]
    assert runner.GPU_ASSIGNMENTS[1][-1] == "F_spatial_content_regularized"
    assert "s1_off_tdrop50" in runner.DEFAULT_CONDITIONS
    assert runner.ARM_SPECS["F_spatial_content_regularized"].content_regularized


def test_spatial_mask_exact_budget_allows_timestep_zero_targets() -> None:
    runner = _runner()
    spec = runner.ARM_SPECS["E_spatial_no_consistency"]
    generator = torch.Generator(device="cpu").manual_seed(9)
    mask = runner._target_mask(spec, 128, 4, torch.device("cpu"), generator, spatial_tokens=16)
    flat = mask.reshape(128, 4, 16)
    assert float(mask.float().mean()) == runner.TARGET_MASK_FRACTION
    assert bool(flat[:, 0].any())
    assert bool((~flat).any(dim=2).any(dim=1).all())
    assert set(flat.sum(dim=(1, 2)).tolist()) == {32}


def test_content_shuffle_preserves_missingness_and_missingness_shuffle_preserves_content() -> None:
    runner = _runner()
    batch = _batch()
    generator = torch.Generator(device="cpu").manual_seed(42)
    content = runner._shuffle_content(batch, generator)
    missingness = runner._shuffle_missingness(batch, generator)
    assert not torch.equal(content["s2"], batch["s2"])
    torch.testing.assert_close(content["s2_mask"], batch["s2_mask"])
    torch.testing.assert_close(content["s2_available"], batch["s2_available"])
    torch.testing.assert_close(missingness["s2"], batch["s2"])
    assert not torch.equal(missingness["s2_mask"], batch["s2_mask"])


def test_run_schema_version_is_defined() -> None:
    runner = _runner()
    assert isinstance(runner.RUN_SCHEMA_VERSION, str)
    assert len(runner.RUN_SCHEMA_VERSION) > 0


def test_existing_checkpoint_helpers_select_latest_checkpoint_for_resume(tmp_path: Path) -> None:
    runner = _runner()
    for step in [1000, 3000, 6000, 10000, 16600]:
        (tmp_path / f"checkpoint_step_{step}.pt").write_bytes(b"x")
    checkpoints = runner._existing_checkpoints(tmp_path)
    assert sorted(checkpoints) == ["step_1000", "step_10000", "step_16600", "step_3000", "step_6000"]
    cfg = runner.RunConfig(arm="A_mixed_viewdrop_reference", seed=42, output_dir=tmp_path)
    scheduled = runner._scheduled_checkpoints(checkpoints, cfg)
    assert sorted(scheduled) == ["step_1000", "step_10000", "step_16600", "step_3000"]
    latest = runner._latest_checkpoint(tmp_path, fixed_updates=16600)
    assert latest is not None
    assert latest[0] == 16600
    assert latest[1].name == "checkpoint_step_16600.pt"


def test_eval_metadata_rejects_stale_outputs(tmp_path: Path) -> None:
    runner = _runner()
    cfg = runner.RunConfig(arm="A_mixed_viewdrop_reference", seed=42, output_dir=tmp_path)
    spec = runner.ARM_SPECS[cfg.arm]
    runner.write_json(tmp_path / "eval_metadata.json", runner._eval_metadata(cfg, spec))
    assert runner._eval_metadata_matches(tmp_path / "eval_metadata.json", cfg, spec)
    stale = runner._eval_metadata(cfg, spec)
    stale["checkpoint_steps"] = [1000]
    runner.write_json(tmp_path / "eval_metadata.json", stale)
    assert not runner._eval_metadata_matches(tmp_path / "eval_metadata.json", cfg, spec)


def test_checkpoint_chunks_are_reused_and_aggregated(tmp_path: Path) -> None:
    runner = _runner()
    cfg = runner.RunConfig(arm="A_mixed_viewdrop_reference", seed=42, output_dir=tmp_path)
    rows_a = [
        {
            "arm": cfg.arm,
            "benchmark": "cropharvest",
            "model": "embedding",
            "holdout": "rwanda-ceo",
            "condition": "clean",
            "label_budget": 1.0,
            "robustness_protocol": "clean_train_degraded_test",
            "checkpoint_role": "step_16600",
            "seed": cfg.seed,
            "f1": 0.5,
            "auc": 0.6,
            "balanced_accuracy": 0.7,
            "calibrated_f1": 0.5,
            "calibrated_balanced_accuracy": 0.7,
        }
    ]
    rows_b = [
        {
            "arm": cfg.arm,
            "benchmark": "cropharvest",
            "model": "embedding",
            "holdout": "ethiopia",
            "condition": "clean",
            "label_budget": 1.0,
            "robustness_protocol": "clean_train_degraded_test",
            "checkpoint_role": "step_16600",
            "seed": cfg.seed,
            "f1": 0.4,
            "auc": 0.5,
            "balanced_accuracy": 0.6,
            "calibrated_f1": 0.4,
            "calibrated_balanced_accuracy": 0.6,
        }
    ]
    runner._write_checkpoint_chunk(cfg, "step_16600", "cropharvest_rwanda-ceo", rows_a)
    runner._write_checkpoint_chunk(cfg, "step_16600", "cropharvest_ethiopia", rows_b)

    restored = runner._read_checkpoint_chunks(
        cfg,
        "step_16600",
        ["cropharvest_rwanda-ceo", "cropharvest_ethiopia"],
    )
    assert restored is not None
    assert [row["holdout"] for row in restored] == ["rwanda-ceo", "ethiopia"]
    runner._write_checkpoint_outputs(cfg, "step_16600", restored)
    assert (tmp_path / "probe_results_step_16600.csv").exists()
    assert (tmp_path / "probe_summary_step_16600.csv").exists()


def test_partial_training_checkpoint_fails_loudly(tmp_path: Path, monkeypatch) -> None:
    runner = _runner()
    cfg = runner.RunConfig(arm="A_mixed_viewdrop_reference", seed=42, output_dir=tmp_path, fixed_updates=16_600)
    spec = runner.ARM_SPECS[cfg.arm]
    (tmp_path / "checkpoint_step_10000.pt").write_bytes(b"partial")

    monkeypatch.setattr(runner, "_make_model", lambda spec, dataset: torch.nn.Linear(1, 1))

    with pytest.raises(RuntimeError, match="Partial non-resumable run"):
        runner._train_one(
            cfg,
            spec,
            dataset=[],
            generic=None,
            agro=None,
            generic_heldout=np.array([], dtype=np.int64),
            agro_heldout=np.array([], dtype=np.int64),
            device=torch.device("cpu"),
        )


def test_few_shot_sample_effective_budget() -> None:
    runner = _runner()
    y = np.asarray([0] * 95 + [1] * 5, dtype=np.int64)
    for budget in [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]:
        indices, eff = runner._few_shot_sample(y, budget, 42)
        assert len(indices) <= len(y)
        assert eff >= budget * 0.5, f"budget={budget} eff={eff} is too far below target"
        assert len(np.unique(y[indices])) == 2, f"Not all classes sampled at budget={budget}"
        if budget < 1.0:
            assert eff < 1.0, f"budget={budget} eff={eff} should be < 1.0"
        else:
            assert eff == 1.0


def test_few_shot_sample_many_classes_unseen_floor() -> None:
    runner = _runner()
    rng = np.random.default_rng(42)
    n_classes = 50
    y = np.arange(n_classes, dtype=np.int64).repeat(2)
    indices, eff = runner._few_shot_sample(y, 0.01, 42)
    assert len(np.unique(y[indices])) == n_classes


def test_safe_probability_matrix_repairs_nan_and_zero_rows() -> None:
    runner = _runner()
    prob, repairs = runner._safe_probability_matrix(
        np.asarray(
            [
                [np.nan, np.nan, 0.0],
                [1.0, np.nan, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    assert repairs >= 4
    assert np.isfinite(prob).all()
    np.testing.assert_allclose(prob.sum(axis=1), np.ones(3), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(prob[2], np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_run_multiclass_rows_records_unseen_class_metrics() -> None:
    runner = _runner()
    rows: list[dict] = []
    rng = np.random.default_rng(99)
    y_train = np.asarray([0] * 40 + [1] * 40, dtype=np.int64)
    y_test = np.asarray([0] * 15 + [1] * 15 + [2] * 10, dtype=np.int64)
    x_train = rng.normal(size=(80, 4)).astype(np.float32)
    x_test = rng.normal(size=(40, 4)).astype(np.float32)
    base = {"benchmark": "eurocropsml", "arm": "test", "seed": 42}
    runner._run_multiclass_rows(rows, "test_model", x_train, x_test, y_train, y_test, base, 99)
    seen_effective_budgets = set()
    for row in rows:
        assert row["n_test_original"] == 40
        assert row["n_test_unseen_dropped"] == 10
        assert row["n_classes_test_unseen"] == 1
        assert "effective_label_budget" in row
        seen_effective_budgets.add(row.get("effective_label_budget"))
    assert len(seen_effective_budgets) >= 4


def test_run_metadata_has_schema_version(tmp_path: Path) -> None:
    runner = _runner()
    meta = {"schema_version": runner.RUN_SCHEMA_VERSION, "arm": "test", "seed": 42}
    runner.write_json(tmp_path / "run_metadata.json", meta)
    loaded = json.loads((tmp_path / "run_metadata.json").read_text())
    assert loaded["schema_version"] == runner.RUN_SCHEMA_VERSION


def test_base_keep_combined_keep_logic() -> None:
    b, t = 3, 6
    s2_avail = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    s1_avail = torch.zeros(b, t, dtype=torch.float32)
    climate_avail = torch.zeros(b, t, dtype=torch.float32)
    base_keep = (s2_avail + s1_avail + climate_avail) > 0
    assert base_keep[0, 5].item() is False
    assert base_keep[1].sum().item() == 0.0
    assert base_keep[2].sum().item() == float(t)
    z = torch.randn(b, t, 2, 1, 1)
    w = base_keep.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    emb = (z * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
    ref = torch.stack([z[0, :5].mean(dim=0), torch.zeros(2, 1, 1), z[2].mean(dim=0)])
    torch.testing.assert_close(emb[:, :, 0, 0], ref[:, :, 0, 0])


def test_extract_raw_stats_masked_reduction() -> None:
    from src.evals.evaluation import extract_raw_stats

    class FakeRawDataset:
        def __getitem__(self, idx):
            b = {
                "s2": torch.arange(44.0).reshape(4, 11, 1, 1),
                "s1": torch.zeros(4, 2, 1, 1),
                "climate": torch.empty(4, 0, 1, 1),
                "s2_mask": torch.ones(4, 1, 1),
                "s1_mask": torch.zeros(4, 1, 1),
                "climate_mask": torch.zeros(4, 1, 1),
                "s2_available": torch.tensor([1.0, 1.0, 1.0, 0.0]),
                "s1_available": torch.zeros(4),
                "climate_available": torch.zeros(4),
                "doy": torch.arange(4.0).unsqueeze(1),
            }
            return {k: v.clone() for k, v in b.items()}

        def __len__(self):
            return 1

    dataset = FakeRawDataset()
    indices = np.array([0])
    result = extract_raw_stats(dataset, indices, batch_size=1, num_workers=0,
                               sensor_off="none", temporal_drop_fraction=0.0, seed=42)
    assert result.shape[1] == (7 + 1 + 2) * 5, f"Expected 50 cols, got {result.shape[1]}"
    assert not np.isnan(result).any()
    b5_mean = result[0, 0]
    assert abs(b5_mean - (3 + 14 + 25) / 3.0) < 1e-5, f"B5 mean {b5_mean} != 14.0"


def test_extract_raw_stats_temporal_drop_uses_same_mask() -> None:
    from src.evals.evaluation import extract_raw_stats
    from src.evals.evaluation import apply_numpy_condition

    class FakeRawDataset:
        def __getitem__(self, idx):
            b = {
                "s2": torch.arange(44.0).reshape(4, 11, 1, 1),
                "s1": torch.zeros(4, 2, 1, 1),
                "climate": torch.empty(4, 0, 1, 1),
                "s2_mask": torch.ones(4, 1, 1),
                "s1_mask": torch.zeros(4, 1, 1),
                "climate_mask": torch.zeros(4, 1, 1),
                "s2_available": torch.tensor([1.0, 1.0, 1.0, 0.0]),
                "s1_available": torch.zeros(4),
                "climate_available": torch.zeros(4),
                "doy": torch.arange(4.0).unsqueeze(1),
            }
            return {k: v.clone() for k, v in b.items()}

        def __len__(self):
            return 1

    dataset = FakeRawDataset()
    indices = np.array([0])
    result = extract_raw_stats(dataset, indices, batch_size=1, num_workers=0,
                               sensor_off="none", temporal_drop_fraction=1.0, seed=42)
    b5_mean = result[0, 0]
    assert abs(b5_mean - (3 + 14) / 2.0) < 1e-5, (
        f"B5 mean {b5_mean} != 8.5 (temporal_drop=1.0 should keep only 2 timesteps, "
        f"not include zeroed timesteps)"
    )


def test_extract_raw_stats_sensor_off_updates_availability() -> None:
    from src.evals.evaluation import apply_numpy_condition
    from src.evals.evaluation import extract_raw_stats

    class FakeRawDataset:
        def __getitem__(self, idx):
            b = {
                "s2": torch.arange(44.0).reshape(4, 11, 1, 1),
                "s1": torch.tensor(
                    [
                        [[0.0], [0.0]],
                        [[10.0], [100.0]],
                        [[20.0], [200.0]],
                        [[30.0], [300.0]],
                    ]
                ).reshape(4, 2, 1, 1),
                "climate": torch.empty(4, 0, 1, 1),
                "s2_mask": torch.ones(4, 1, 1),
                "s1_mask": torch.ones(4, 1, 1),
                "climate_mask": torch.zeros(4, 1, 1),
                "s2_available": torch.tensor([1.0, 0.0, 1.0, 0.0]),
                "s1_available": torch.tensor([0.0, 1.0, 1.0, 0.0]),
                "climate_available": torch.zeros(4),
                "doy": torch.arange(4.0).unsqueeze(1),
            }
            return {k: v.clone() for k, v in b.items()}

        def __len__(self):
            return 1

    batch = FakeRawDataset()[0]
    apply_numpy_condition(batch, "s2", 0.0, np.random.default_rng(42))
    assert float(batch["s2_available"].sum()) == 0.0
    assert float(batch["s2_mask"].sum()) == 0.0

    result = extract_raw_stats(
        FakeRawDataset(),
        np.array([0]),
        batch_size=1,
        num_workers=0,
        sensor_off="s2",
        temporal_drop_fraction=0.0,
        seed=42,
    )
    s1_vv_mean = result[0, 8]
    assert abs(s1_vv_mean - 15.0) < 1e-5, f"S1 VV mean {s1_vv_mean} should exclude S2-only timestep"


def test_build_eurocropsml_zarr_synthetic(tmp_path: Path, monkeypatch) -> None:
    runner = _runner()
    from src.datasets.eurocropsml import build_eurocropsml_zarr

    preprocess = tmp_path / "preprocess"
    preprocess.mkdir()
    rng = np.random.default_rng(42)
    prefixes = ["EE", "LV", "PT", "EE", "LV"]
    for i, prefix in enumerate(prefixes):
        t = 96 + i
        data = rng.integers(0, 10000, size=(t, 13), dtype=np.int64)
        dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2025-01-01"), np.timedelta64(1, "D"))[:t]
        np.savez(preprocess / f"{prefix}_sample_{i}_HCAT{i}.npz", data=data, dates=dates)

    output = tmp_path / "out.zarr"
    labels = tmp_path / "labels.csv"
    summary_path = tmp_path / "summary.json"
    build_eurocropsml_zarr(
        preprocess_dir=preprocess,
        output_zarr=output,
        output_labels=labels,
        output_summary=summary_path,
        max_samples=5,
    )
    assert output.exists()
    assert labels.exists()
    assert summary_path.exists()

    import zarr
    z = zarr.open(str(output), mode="r")
    assert "s2" in z
    assert z["s2"].shape == (5, 96, 11, 1, 1)
    assert z["s2"].dtype == np.float32

    import csv
    with labels.open("r") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert rows[0]["label"] == "HCAT0"

    with summary_path.open("r") as f:
        summary = json.load(f)
    assert summary["n_samples"] == 5
    assert summary["fixed_timesteps"] == 96
    assert summary["timesteps"]["max"] == 100
    assert summary["timesteps"]["min"] == 96
    assert summary["eval_reflectance_divisor"] == 10000.0
    assert summary["stored_reflectance_divisor"] is None

    monkeypatch.setattr(runner, "EUROCROPS_ZARR", output)
    monkeypatch.setattr(runner, "EUROCROPS_LABELS_CSV", labels)
    monkeypatch.setattr(runner, "EUROCROPS_SUMMARY", summary_path)
    runner._validate_eurocrops_store()


def test_eurocropsml_linspace_downsample_preserves_first_last_dates(tmp_path: Path) -> None:
    from src.datasets.eurocropsml import build_eurocropsml_zarr

    preprocess = tmp_path / "preprocess"
    preprocess.mkdir()
    rng = np.random.default_rng(42)
    n_t = 216
    data = rng.integers(0, 10000, size=(n_t, 13), dtype=np.int64)
    dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-12-31"), np.timedelta64(1, "D"))[:n_t]
    np.savez(preprocess / "EE_long_HCAT99.npz", data=data, dates=dates)

    output = tmp_path / "out.zarr"
    labels = tmp_path / "labels.csv"
    summary_path = tmp_path / "summary.json"
    build_eurocropsml_zarr(
        preprocess_dir=preprocess,
        output_zarr=output,
        output_labels=labels,
        output_summary=summary_path,
        fixed_timesteps=96,
        max_samples=1,
    )
    assert output.exists()

    import zarr
    z = zarr.open(str(output), mode="r")
    assert z["s2"].shape == (1, 96, 11, 1, 1)

    s2_ns = z["s2_time_ns"][0]
    assert len(s2_ns) == 96
    expected_first_ns = dates[0].astype("datetime64[ns]").astype(np.int64)
    expected_last_ns = dates[-1].astype("datetime64[ns]").astype(np.int64)
    assert int(s2_ns[0]) == int(expected_first_ns), f"First ns {s2_ns[0]} != {expected_first_ns}"
    assert int(s2_ns[-1]) == int(expected_last_ns), f"Last ns {s2_ns[-1]} != {expected_last_ns}"

    with summary_path.open("r") as f:
        summary = json.load(f)
    assert summary["n_downsampled"] == 1
    assert summary["downsampled_fraction"] == 1.0
    assert summary["sequence_policy"] == "fixed_cap_span_downsample"
    assert summary["fixed_timesteps"] == 96
    assert summary["max_observed_timesteps"] == 216
