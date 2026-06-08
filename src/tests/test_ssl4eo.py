from pathlib import Path

import numpy as np
import pytest

import src.datasets.ssl4eo as ssl4eo
from src.datasets.ssl4eo import (
    POOL_AGRO,
    POOL_GENERIC,
    SAMPLE_ID_WIDTH,
    SSL4EO_WDS_REVISION,
    _candidate_composite_bins,
    _candidate_geo_bins,
    _download_archive,
    _select_candidates,
    _state_config,
    _store_arrays,
    _write_summary,
    probe_aligned_cloud_parity,
    probe_lulc_archive,
    TOKEN_PATCH_SIZE,
)


def _candidate(index: int, stratum: str) -> dict[str, object]:
    return {
        "sample": f"ssl4eos12_train_seasonal_data_{index:07d}",
        "source_shard": index // 10 + 1,
        "stratum": stratum,
        "cropland_fraction": 0.9 if stratum == "cropland_dominant" else 0.3,
        "s2_clear_fraction": 0.8,
        "s2_token_available_fraction": 0.75,
        "center_lat": 20.0 + index / 100.0,
        "center_lon": 30.0 + index / 100.0,
    }


def test_candidate_selection_is_deterministic_and_quota_balanced() -> None:
    candidates = []
    for index in range(40):
        candidates.append(_candidate(index, "cropland_dominant"))
    for index in range(40, 70):
        candidates.append(_candidate(index, "field_boundary"))
    for index in range(70, 100):
        candidates.append(_candidate(index, "nearby_non_crop"))
    selected_a = _select_candidates(candidates, 20)
    selected_b = _select_candidates(list(reversed(candidates)), 20)
    assert selected_a == selected_b
    assert len(selected_a[POOL_GENERIC]) == 20
    agro = selected_a[POOL_AGRO]
    assert sum(item["stratum"] == "cropland_dominant" for item in agro) == 13
    assert sum(item["stratum"] == "field_boundary" for item in agro) == 5
    assert sum(item["stratum"] == "nearby_non_crop" for item in agro) == 2
    generic_ids = {item["sample"] for item in selected_a[POOL_GENERIC]}
    agro_ids = {item["sample"] for item in agro}
    assert not generic_ids & agro_ids
    assert _candidate_geo_bins(selected_a[POOL_GENERIC]) == _candidate_geo_bins(agro)
    assert _candidate_composite_bins(selected_a[POOL_GENERIC]) == _candidate_composite_bins(agro)


def test_candidate_selection_preserves_disjoint_control_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = []
    for index, (stratum, lat) in enumerate(
        [
            ("cropland_dominant", 0.0),
            ("cropland_dominant", 0.0),
            ("cropland_dominant", 10.0),
            ("cropland_dominant", 10.0),
            ("cropland_dominant", 20.0),
            ("cropland_dominant", 20.0),
            ("cropland_dominant", 30.0),
            ("cropland_dominant", 30.0),
            ("field_boundary", 40.0),
            ("field_boundary", 40.0),
            ("nearby_non_crop", 60.0),
            ("nearby_non_crop", 60.0),
        ]
    ):
        candidate = _candidate(index, stratum)
        candidate["center_lat"] = lat
        candidate["center_lon"] = 0.0
        candidates.append(candidate)
    monkeypatch.setattr(ssl4eo, "_stable_priority", lambda pool, key: key)
    selected = _select_candidates(candidates, 4)
    generic = selected[POOL_GENERIC]
    agro = selected[POOL_AGRO]
    assert len(generic) == len(agro) == 4
    assert _candidate_geo_bins(generic) == _candidate_geo_bins(agro)
    assert not {item["sample"] for item in generic} & {item["sample"] for item in agro}


def test_max_flow_allocator_preserves_bins_needed_by_scarce_strata(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = []
    for index, (stratum, lat) in enumerate(
        [
            ("cropland_dominant", 0.0),
            ("cropland_dominant", 0.0),
            ("cropland_dominant", 0.0),
            ("cropland_dominant", 0.0),
            ("cropland_dominant", 0.0),
            ("cropland_dominant", 0.0),
            ("cropland_dominant", 10.0),
            ("cropland_dominant", 10.0),
            ("cropland_dominant", 20.0),
            ("cropland_dominant", 20.0),
            ("field_boundary", 20.0),
            ("field_boundary", 20.0),
            ("field_boundary", 40.0),
            ("field_boundary", 40.0),
            ("nearby_non_crop", 60.0),
            ("nearby_non_crop", 60.0),
            ("other", 20.0),
            ("other", 0.0),
        ]
    ):
        candidate = _candidate(index, stratum)
        candidate["center_lat"] = lat
        candidate["center_lon"] = 0.0
        candidates.append(candidate)
    monkeypatch.setattr(ssl4eo, "_stable_priority", lambda pool, key: key)
    selected = _select_candidates(candidates, 8)
    generic = selected[POOL_GENERIC]
    agro = selected[POOL_AGRO]
    assert len(generic) == len(agro) == 8
    assert _candidate_composite_bins(generic) == _candidate_composite_bins(agro)
    assert sum(item["stratum"] == "field_boundary" for item in agro) == 2
    assert sum(item["stratum"] == "nearby_non_crop" for item in agro) == 1


def test_store_preserves_full_sample_ids(tmp_path: Path) -> None:
    path = tmp_path / "fixed.zarr"
    _, arrays = _store_arrays(path, max_samples=2, patch_size=16, pool=POOL_GENERIC)
    sample_ids = [
        "ssl4eos12_train_seasonal_data_0000001",
        "ssl4eos12_train_seasonal_data_0000002",
    ]
    arrays["sample"][:] = np.asarray(sample_ids, dtype=f"S{SAMPLE_ID_WIDTH}")
    arrays["sampling_stratum"][:] = np.asarray(["generic", "generic"], dtype="S32")
    arrays["source_shard"][:] = 1
    summary = _write_summary(path, POOL_GENERIC, arrays, sample_count=2, last_shard=1)
    assert arrays["sample"].dtype == np.dtype(f"S{SAMPLE_ID_WIDTH}")
    assert summary["unique_sample_ids"] == 2


def test_state_config_changes_when_content_contract_changes(tmp_path: Path) -> None:
    generic = tmp_path / "generic.zarr"
    agro = tmp_path / "agro.zarr"
    first = _state_config(generic, agro, max_samples=10, patch_size=16, start_shard=1)
    second = _state_config(generic, agro, max_samples=11, patch_size=16, start_shard=1)
    assert first != second
    assert first["source_revision"] == SSL4EO_WDS_REVISION


def test_download_archive_pins_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_download(repo_id, filename, **kwargs):
        captured.update({"repo_id": repo_id, "filename": filename, **kwargs})
        return str(tmp_path / "archive.tar")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    _download_archive("LULC", 1, tmp_path)
    assert captured["revision"] == SSL4EO_WDS_REVISION


def test_real_lulc_shard_uses_static_esri_crop_labels() -> None:
    matches = list(
        Path("/private/tmp/surf_ssl4eo_probe").rglob("train/LULC/ssl4eos12_shard_000001.tar")
    )
    matches = [path for path in matches if path.exists()]
    if not matches:
        pytest.skip("cached official LULC probe shard is unavailable")
    summary = probe_lulc_archive(matches[0])
    assert summary["samples"] == 512
    assert max(summary["remapped_classes"]) <= 9
    assert summary["clear_esri_crop_pixels"] > 0
    assert (
        summary["clear_esri_crop_pixels"]
        == summary["clear_esri_crop_pixels_mapped_to_class_4"]
    )


def test_materialization_rejects_masks_with_equal_aggregate_but_different_pixels() -> None:
    """Two cloud masks with identical clear_fraction but different pixel layouts must fail."""
    from src.datasets.ssl4eo import _assert_center_patch_cloud_parity
    h = 16
    lulc_mask = np.zeros((4, h, h), dtype=np.uint8)
    s2_mask = np.zeros((4, h, h), dtype=np.uint8)
    lulc_mask[:, :, :8] = 1
    s2_mask[:, :8, :] = 1
    assert abs(float(np.mean(lulc_mask == 0)) - float(np.mean(s2_mask == 0))) < 1e-10
    with pytest.raises(ValueError, match="pixels differ"):
        _assert_center_patch_cloud_parity(lulc_mask, s2_mask, "test-key", 16)


def test_store_arrays_rejects_bad_patch_size() -> None:
    with pytest.raises(ValueError, match="TOKEN_PATCH_SIZE"):
        _store_arrays(Path("/tmp/nonexistent"), max_samples=1, patch_size=3, pool=POOL_GENERIC)


def test_state_config_includes_token_params() -> None:
    config = _state_config(Path("/tmp/generic.zarr"), Path("/tmp/agro.zarr"), 100, 16, 1)
    assert config["token_patch_size"] == TOKEN_PATCH_SIZE
    assert config["min_token_clear_fraction"] == 0.50


def _make_cloud_mask(h: int = 16, clear: bool = True) -> np.ndarray:
    return (np.ones((4, h, h), dtype=np.uint8) if clear else np.zeros((4, h, h), dtype=np.uint8))


class _FakeZarrRoot:
    def __init__(self, cloud_mask: np.ndarray) -> None:
        self.cloud_mask = cloud_mask

    def __getitem__(self, key: str) -> np.ndarray:
        if key == "cloud_mask":
            return self.cloud_mask
        raise KeyError(key)


class _FakeArchive:
    def close(self) -> None:
        pass


def test_probe_parity_exact_match_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mask = _make_cloud_mask()
    monkeypatch.setattr(ssl4eo, "_download_archive", lambda *a: "/fake.tar")
    monkeypatch.setattr(ssl4eo, "_tar_members", lambda p: (_FakeArchive(), {f"s{i}": object() for i in range(10)}))
    monkeypatch.setattr(ssl4eo, "_read_zarr", lambda *a: _FakeZarrRoot(mask))
    result = probe_aligned_cloud_parity(1, tmp_path, num_samples=5, patch_size=16)
    assert result["samples_checked"] == 5
    assert result["total_pixel_diff"] == 0
    assert result["patch_size"] == 16
    assert result["requested_samples"] == 5


def test_probe_parity_one_pixel_mismatch_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mask_lulc = _make_cloud_mask()
    mask_s2 = _make_cloud_mask()
    mask_s2[0, 0, 0] = 0
    call_count = 0

    def fake_read(archive, member):
        nonlocal call_count
        call_count += 1
        return _FakeZarrRoot(mask_lulc if call_count % 2 == 1 else mask_s2)

    monkeypatch.setattr(ssl4eo, "_download_archive", lambda *a: "/fake.tar")
    monkeypatch.setattr(ssl4eo, "_tar_members", lambda p: (_FakeArchive(), {"s0": object()}))
    monkeypatch.setattr(ssl4eo, "_read_zarr", fake_read)
    with pytest.raises(ValueError, match="pixels differ"):
        probe_aligned_cloud_parity(1, tmp_path, num_samples=1, patch_size=16)


def test_probe_parity_insufficient_samples_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ssl4eo, "_download_archive", lambda *a: "/fake.tar")
    monkeypatch.setattr(ssl4eo, "_tar_members", lambda p: (_FakeArchive(), {"s0": object()}))
    monkeypatch.setattr(ssl4eo, "_read_zarr", lambda *a: _FakeZarrRoot(_make_cloud_mask()))
    with pytest.raises(ValueError, match="checked only"):
        probe_aligned_cloud_parity(1, tmp_path, num_samples=5, patch_size=16)
