"""Build local SSL4EO-S12 v1.1 patch zarr stores for architecture pretraining."""

from collections import Counter, deque
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time
from typing import Any

import fsspec
import numpy as np
from numcodecs import Blosc
import zarr
from zarr.storage import ZipStore

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
S1_BANDS = ["VV", "VH"]
CLIMATE_BANDS = ["temperature", "precipitation", "elevation"]
S2L2A_SOURCE_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S2L2A_KEEP = [S2L2A_SOURCE_BANDS.index(name) for name in ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]]
SAMPLES_PER_SHARD = 64
SSL4EO_ZARR_REPO = "embed2scale/SSL4EO-S12-v1.1-Zarr"
HF_DOWNLOAD_RETRIES = 8
BUILD_STATE_NAME = "build_state.json"


def _encode_text(values: list[str], width: int) -> np.ndarray:
    return np.asarray([v.encode("utf-8")[:width] for v in values], dtype=f"S{width}")


def _center_slice(size: int, patch_size: int) -> slice:
    if patch_size > size:
        raise ValueError(f"patch_size={patch_size} is larger than source size={size}")
    start = (size - patch_size) // 2
    return slice(start, start + patch_size)


def _open_zip_group(path: str | Path):
    store = ZipStore(str(path), mode="r")
    return store, zarr.open_group(store, mode="r")


def _open_output_group(path: Path, mode: str = "w"):
    try:
        return zarr.open_group(str(path), mode=mode, zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode=mode, zarr_version=2)


def _create_array(root, name: str, **kwargs):
    if hasattr(root, "create_dataset"):
        return root.create_dataset(name, **kwargs)
    return root.create_array(name, **kwargs)


def _replace_array(root, name: str, **kwargs):
    if name in root:
        del root[name]
    return _create_array(root, name, **kwargs)


def _ensure_array(root, name: str, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: str, compressor):
    if name not in root:
        return _create_array(root, name, shape=shape, chunks=chunks, dtype=dtype, compressor=compressor)
    arr = root[name]
    if len(arr.shape) != len(shape) or tuple(arr.shape[1:]) != tuple(shape[1:]):
        raise ValueError(f"Existing {name} has incompatible shape {arr.shape}; expected tail {shape[1:]}")
    if int(arr.shape[0]) < int(shape[0]):
        arr.resize(shape)
    if int(arr.shape[0]) > int(shape[0]):
        raise ValueError(f"Existing {name} has {arr.shape[0]} rows, requested max_samples={shape[0]}")
    return arr


def _download_ssl4eo_zip(modality: str, shard: int, cache_dir: str | Path | None) -> str:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    filename = f"train/{modality}/ssl4eos12_train_seasonal_data_{shard:06d}.zarr.zip"
    last_error: Exception | None = None
    for attempt in range(HF_DOWNLOAD_RETRIES):
        try:
            return hf_hub_download(
                SSL4EO_ZARR_REPO,
                filename,
                repo_type="dataset",
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        except Exception as exc:
            last_error = exc
            sleep_seconds = min(300, 5 * (attempt + 1))
            print(
                f"download retry {attempt + 1}/{HF_DOWNLOAD_RETRIES} for {filename}: "
                f"{type(exc).__name__}: {exc}"
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"failed to download {filename} after {HF_DOWNLOAD_RETRIES} attempts") from last_error


def _disk_usage_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            stat = item.lstat()
        except FileNotFoundError:
            continue
        blocks = getattr(stat, "st_blocks", 0)
        if blocks:
            total += int(blocks) * 512
        elif item.is_file():
            total += int(stat.st_size)
    return total


def _assert_cache_budget(cache_path: Path | None, max_cache_gib: float | None) -> None:
    if cache_path is None or max_cache_gib is None:
        return
    used_gib = _disk_usage_bytes(cache_path) / (1024**3)
    if used_gib > max_cache_gib:
        raise ValueError(
            f"{cache_path} uses {used_gib:.2f} GiB, above the {max_cache_gib:.2f} GiB SSL4EO cache budget"
        )


def _evict_cached_file(path: str | Path, cache_path: Path) -> None:
    cached_path = Path(path)
    canonical_cached_path = cached_path.parent.resolve() / cached_path.name
    resolved_path = cached_path.resolve()
    cache_root = cache_path.resolve()
    try:
        canonical_cached_path.relative_to(cache_root)
        resolved_path.relative_to(cache_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to evict cached file outside {cache_root}: {cached_path}") from exc

    cached_path.unlink(missing_ok=True)
    if resolved_path != canonical_cached_path:
        resolved_path.unlink(missing_ok=True)


def _evict_cached_files(paths: list[str | Path], cache_path: Path) -> None:
    for path in paths:
        _evict_cached_file(path, cache_path)


def _write_band_names(root) -> None:
    _replace_array(root, "s2_bands", shape=(len(S2_BANDS),), dtype="S32", chunks=(len(S2_BANDS),))[:] = _encode_text(S2_BANDS, 32)
    _replace_array(root, "s1_bands", shape=(len(S1_BANDS),), dtype="S32", chunks=(len(S1_BANDS),))[:] = _encode_text(S1_BANDS, 32)
    _replace_array(root, "climate_bands", shape=(len(CLIMATE_BANDS),), dtype="S32", chunks=(len(CLIMATE_BANDS),))[:] = _encode_text(CLIMATE_BANDS, 32)


def _state_path(output_zarr: Path) -> Path:
    return output_zarr / BUILD_STATE_NAME


def _load_state(output_zarr: Path) -> dict | None:
    path = _state_path(output_zarr)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_state(output_zarr: Path, state: dict) -> None:
    path = _state_path(output_zarr)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2))
    tmp_path.replace(path)


def _infer_write_idx(sample_id) -> int:
    total = int(sample_id.shape[0])
    step = 4096
    for start in range(0, total, step):
        stop = min(total, start + step)
        values = np.asarray(sample_id[start:stop])
        written = values != b""
        if not bool(written.all()):
            return start + int(np.argmax(~written))
    return total


def _state_matches(state: dict | None, max_samples: int, patch_size: int, start_shard: int, ndvi_mean_min: float | None) -> bool:
    if not state:
        return False
    return (
        int(state.get("max_samples", -1)) == int(max_samples)
        and int(state.get("patch_size", -1)) == int(patch_size)
        and int(state.get("start_shard", -1)) == int(start_shard)
        and state.get("ndvi_mean_min") == ndvi_mean_min
    )


def build_ssl4eo_s12_zarr(
    output_zarr: str | Path,
    max_samples: int = 4096,
    patch_size: int = 16,
    start_shard: int = 1,
    ndvi_mean_min: float | None = None,
    cache_dir: str | Path | None = None,
    max_cache_gib: float | None = None,
    evict_cached_shards: bool = False,
) -> dict[str, int | float | str | None]:
    """
    Stream SSL4EO-S12 v1.1 shards into the local SURF zarr layout.

    SSL4EO-S12 v1.1 provides S1/S2/NDVI as zarr zip shards. It does not provide
    matched temperature or precipitation in those zarr shards, so the local
    context tensor is present for channel-contract compatibility but marked
    unavailable via `climate_mask=0`.
    """

    if max_samples <= 0:
        raise ValueError("max_samples must be positive for streamed SSL4EO builds.")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if max_cache_gib is not None and max_cache_gib <= 0:
        raise ValueError("max_cache_gib must be positive when provided.")
    if evict_cached_shards and cache_dir is None:
        raise ValueError("cache_dir is required when evict_cached_shards is enabled.")

    output_zarr = Path(output_zarr).expanduser().resolve()
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir).expanduser().resolve() if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
    _assert_cache_budget(cache_path, max_cache_gib)

    timesteps = 4
    h = patch_size
    w = patch_size
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    output_exists = (output_zarr / ".zgroup").exists() or (output_zarr / "zarr.json").exists()
    root = _open_output_group(output_zarr, mode="a" if output_exists else "w")
    chunks_n = min(256, max_samples)

    s2_arr = _ensure_array(root, "s2", (max_samples, timesteps, len(S2_BANDS), h, w), (chunks_n, timesteps, len(S2_BANDS), h, w), "float32", compressor)
    s1_arr = _ensure_array(root, "s1", (max_samples, timesteps, len(S1_BANDS), h, w), (chunks_n, timesteps, len(S1_BANDS), h, w), "float32", compressor)
    climate_arr = _ensure_array(root, "climate", (max_samples, timesteps, len(CLIMATE_BANDS), h, w), (chunks_n, timesteps, len(CLIMATE_BANDS), h, w), "float32", compressor)
    s2_mask = _ensure_array(root, "s2_mask", (max_samples, timesteps, h, w), (chunks_n, timesteps, h, w), "uint8", compressor)
    s1_mask = _ensure_array(root, "s1_mask", (max_samples, timesteps, h, w), (chunks_n, timesteps, h, w), "uint8", compressor)
    climate_mask = _ensure_array(root, "climate_mask", (max_samples, timesteps, h, w), (chunks_n, timesteps, h, w), "uint8", compressor)
    patch_xy = _ensure_array(root, "patch_xy", (max_samples, 2), (chunks_n, 2), "int32", compressor)
    center_lat = _ensure_array(root, "center_lat", (max_samples,), (chunks_n,), "float64", compressor)
    center_lon = _ensure_array(root, "center_lon", (max_samples,), (chunks_n,), "float64", compressor)
    sample_id = _ensure_array(root, "sample", (max_samples,), (chunks_n,), "S16", compressor)

    inferred_write_idx = _infer_write_idx(sample_id)
    state = _load_state(output_zarr)
    if ndvi_mean_min is not None and inferred_write_idx > 0 and not _state_matches(state, max_samples, patch_size, start_shard, ndvi_mean_min):
        raise ValueError("Cannot safely resume a filtered SSL4EO build without a matching build state file.")

    if _state_matches(state, max_samples, patch_size, start_shard, ndvi_mean_min) and int(state.get("write_idx", -1)) == inferred_write_idx:
        write_idx = inferred_write_idx
        seen = int(state.get("source_samples_seen", write_idx))
        shard = int(state.get("next_shard", start_shard + write_idx // SAMPLES_PER_SHARD))
        local_start = int(state.get("next_local_idx", write_idx % SAMPLES_PER_SHARD))
    else:
        write_idx = inferred_write_idx
        seen = inferred_write_idx
        shard = start_shard + write_idx // SAMPLES_PER_SHARD
        local_start = write_idx % SAMPLES_PER_SHARD

    if write_idx >= max_samples:
        _write_band_names(root)
        if "time" not in root:
            _create_array(root, "time", shape=(timesteps,), dtype="S16", chunks=(timesteps,))[:] = _encode_text(["2000-03-15", "2000-06-15", "2000-09-15", "2000-12-15"], 16)
        summary = {
            "output_zarr": str(output_zarr),
            "samples": int(write_idx),
            "patch_size": int(patch_size),
            "start_shard": int(start_shard),
            "last_shard_read": int(shard - 1),
            "source_samples_seen": int(seen),
            "ndvi_mean_min": ndvi_mean_min,
            "resumed": bool(output_exists),
            "status": "complete",
        }
        (output_zarr / "build_summary.json").write_text(json.dumps(summary, indent=2))
        _write_state(output_zarr, {**summary, "max_samples": int(max_samples), "next_shard": int(shard), "next_local_idx": 0})
        return summary

    center = None
    while write_idx < max_samples:
        s2_path = _download_ssl4eo_zip("S2L2A", shard, cache_path)
        _assert_cache_budget(cache_path, max_cache_gib)
        s1_path = _download_ssl4eo_zip("S1GRD", shard, cache_path)
        _assert_cache_budget(cache_path, max_cache_gib)
        ndvi_path = _download_ssl4eo_zip("NDVI", shard, cache_path)
        _assert_cache_budget(cache_path, max_cache_gib)
        stores = []
        try:
            s2_store, s2_root = _open_zip_group(s2_path)
            s1_store, s1_root = _open_zip_group(s1_path)
            ndvi_store, ndvi_root = _open_zip_group(ndvi_path)
            stores.extend([s2_store, s1_store, ndvi_store])
            if center is None:
                center = _center_slice(int(s2_root["bands"].shape[-1]), patch_size)
            s2_source = s2_root["bands"]
            s1_source = s1_root["bands"]
            ndvi_source = ndvi_root["bands"]
            cloud_source = s2_root["cloud_mask"]
            shard_n = int(s2_source.shape[0])
            for local_idx in range(local_start, shard_n):
                seen += 1
                ndvi = np.asarray(ndvi_source[local_idx, :, 0, center, center], dtype=np.float32)
                if ndvi_mean_min is not None and float(np.nanmean(ndvi)) < ndvi_mean_min:
                    continue
                s2 = np.asarray(s2_source[local_idx, :, S2L2A_KEEP, center, center], dtype=np.float32) / 10000.0
                s1 = np.asarray(s1_source[local_idx, :, :, center, center], dtype=np.float32)
                s2_combined = np.concatenate([s2, ndvi[:, None, :, :]], axis=1)
                clouds = np.asarray(cloud_source[local_idx, :, center, center], dtype=np.uint8)

                s2_arr[write_idx] = np.nan_to_num(s2_combined, nan=0.0, posinf=0.0, neginf=0.0)
                s1_arr[write_idx] = np.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)
                climate_arr[write_idx] = 0.0
                s2_mask[write_idx] = (clouds == 0).astype(np.uint8)
                s1_mask[write_idx] = 1
                climate_mask[write_idx] = 0
                patch_xy[write_idx] = 0
                center_lat[write_idx] = float(s2_root["center_lat"][local_idx])
                center_lon[write_idx] = float(s2_root["center_lon"][local_idx])
                sample_id[write_idx] = str(s2_root["sample"][local_idx]).encode("utf-8")[:16]
                write_idx += 1
                next_local_idx = local_idx + 1
                next_shard = shard
                if next_local_idx >= shard_n:
                    next_shard = shard + 1
                    next_local_idx = 0
                _write_state(
                    output_zarr,
                    {
                        "status": "running",
                        "max_samples": int(max_samples),
                        "patch_size": int(patch_size),
                        "start_shard": int(start_shard),
                        "write_idx": int(write_idx),
                        "source_samples_seen": int(seen),
                        "next_shard": int(next_shard),
                        "next_local_idx": int(next_local_idx),
                        "ndvi_mean_min": ndvi_mean_min,
                    },
                )
                if write_idx >= max_samples:
                    break
        finally:
            for store in stores:
                store.close()
            if evict_cached_shards and cache_path:
                _evict_cached_files([s2_path, s1_path, ndvi_path], cache_path)
                _assert_cache_budget(cache_path, max_cache_gib)
        shard += 1
        local_start = 0

    if write_idx < max_samples:
        s2_arr.resize((write_idx, timesteps, len(S2_BANDS), h, w))
        s1_arr.resize((write_idx, timesteps, len(S1_BANDS), h, w))
        climate_arr.resize((write_idx, timesteps, len(CLIMATE_BANDS), h, w))
        s2_mask.resize((write_idx, timesteps, h, w))
        s1_mask.resize((write_idx, timesteps, h, w))
        climate_mask.resize((write_idx, timesteps, h, w))
        patch_xy.resize((write_idx, 2))
        center_lat.resize((write_idx,))
        center_lon.resize((write_idx,))
        sample_id.resize((write_idx,))

    _replace_array(root, "time", shape=(timesteps,), dtype="S16", chunks=(timesteps,))[:] = _encode_text(["2000-03-15", "2000-06-15", "2000-09-15", "2000-12-15"], 16)
    _write_band_names(root)
    root.attrs["source"] = SSL4EO_ZARR_REPO
    root.attrs["context_note"] = "temperature, precipitation, and elevation slots are present; context is unavailable for SSL4EO pretraining in this build."
    summary = {
        "output_zarr": str(output_zarr),
        "samples": int(write_idx),
        "patch_size": int(patch_size),
        "start_shard": int(start_shard),
        "last_shard_read": int(shard - 1),
        "source_samples_seen": int(seen),
        "ndvi_mean_min": ndvi_mean_min,
        "resumed": bool(output_exists),
        "status": "complete",
    }
    (output_zarr / "build_summary.json").write_text(json.dumps(summary, indent=2))
    _write_state(output_zarr, {**summary, "max_samples": int(max_samples), "next_shard": int(shard), "next_local_idx": 0})
    return summary

SSL4EO_WDS_REPO = "embed2scale/SSL4EO-S12-v1.1"
SSL4EO_WDS_REVISION = "c437e551ee03379fa3d3ab079dbf6827a635270f"
MODALITIES = ["S2L2A", "S1GRD", "NDVI", "DEM", "LULC"]
MAX_TRAIN_SHARD = 477
HF_DOWNLOAD_RETRIES = 8
ESRI_CROPLAND_CLASS = 5
POOL_GENERIC = "generic_fixed"
POOL_AGRO = "agro_fixed"
AGRO_STRATUM_FRACTIONS = {
    "cropland_dominant": 0.625,
    "field_boundary": 0.25,
    "nearby_non_crop": 0.125,
}
TOKEN_PATCH_SIZE = 4
MIN_TOKEN_CLEAR_FRACTION = 0.50
BUILD_SCHEMA_VERSION = 8
SAMPLING_SEED = 42
SAMPLE_ID_WIDTH = 64
CLEAR_FRACTION_BUCKET_WIDTH = 0.10
CLOUD_PARITY_NUM_SAMPLES = 100
NORMALIZATION_CONTRACT = {
    "s2": "SSL4EO-S12 v1.1 L2A reflectance divided by 10000; NDVI appended unchanged",
    "s1": "SSL4EO-S12 v1.1 S1GRD values preserved unchanged",
    "dem": "SSL4EO-S12 v1.1 DEM values preserved unchanged",
    "lulc": "SSL4EO-S12 v1.1 static esri_original labels preserved for sampling metadata only; ESRI class 5 drives cropland sampling",
}

# Conservative bounding boxes deliberately over-exclude border regions.
EXCLUDED_REGIONS = {
    "rwanda": [28.80, 30.90, -2.90, -1.00],
    "togo": [-0.30, 1.90, 6.00, 11.20],
    "ethiopia": [32.80, 48.20, 3.00, 15.10],
    "brazil": [-74.00, -34.50, -34.00, 5.50],
}


def _download_modalities(shard: int, cache_dir: Path) -> dict[str, str]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {modality: executor.submit(_download_archive, modality, shard, cache_dir) for modality in MODALITIES}
        return {modality: future.result() for modality, future in futures.items()}


def _download_archive(modality: str, shard: int, cache_dir: Path) -> str:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    filename = f"train/{modality}/ssl4eos12_shard_{shard:06d}.tar"
    last_error: Exception | None = None
    for attempt in range(HF_DOWNLOAD_RETRIES):
        try:
            return hf_hub_download(
                SSL4EO_WDS_REPO,
                filename,
                repo_type="dataset",
                revision=SSL4EO_WDS_REVISION,
                cache_dir=str(cache_dir),
            )
        except Exception as exc:
            last_error = exc
            sleep_seconds = min(300, 5 * (attempt + 1))
            print(
                f"download retry {attempt + 1}/{HF_DOWNLOAD_RETRIES} for {filename}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"failed to download {filename} after {HF_DOWNLOAD_RETRIES} attempts") from last_error


def _member_key(name: str) -> str:
    base = Path(name).name
    return base[: -len(".zarr.zip")] if base.endswith(".zarr.zip") else base


def _tar_members(path: str | Path) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]:
    archive = tarfile.open(path, mode="r")
    members = {
        _member_key(member.name): member
        for member in archive.getmembers()
        if member.isfile() and member.name.endswith(".zarr.zip")
    }
    if not members:
        archive.close()
        raise ValueError(f"No .zarr.zip samples found in {path}")
    return archive, members


def _read_zarr(archive: tarfile.TarFile, member: tarfile.TarInfo):
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"Could not extract {member.name}")
    mapper = fsspec.filesystem("zip", fo=io.BytesIO(extracted.read()), block_size=None).get_mapper("")
    try:
        return zarr.open_consolidated(mapper, mode="r", zarr_format=2)
    except TypeError:
        return zarr.open_consolidated(mapper, mode="r")


def _center_patch(values: np.ndarray, center: slice) -> np.ndarray:
    return np.asarray(values[..., center, center])


def _time_ns(values: Any) -> np.ndarray:
    return np.asarray(values).astype("datetime64[ns]").astype(np.int64)


def _scalar(root, name: str) -> float:
    return float(np.asarray(root[name][...]).reshape(-1)[0])


def _excluded_region(lat: float, lon: float) -> str | None:
    for name, (min_lon, max_lon, min_lat, max_lat) in EXCLUDED_REGIONS.items():
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return name
    return None


def _lulc_stratum(esri_lulc: np.ndarray) -> tuple[str | None, float]:
    cropland_fraction = float(np.mean(esri_lulc == ESRI_CROPLAND_CLASS))
    if cropland_fraction >= 0.80:
        return "cropland_dominant", cropland_fraction
    if cropland_fraction >= 0.20:
        return "field_boundary", cropland_fraction
    if cropland_fraction >= 0.01:
        return "nearby_non_crop", cropland_fraction
    return None, cropland_fraction


def _quota_counts(max_samples: int) -> dict[str, int]:
    out = {
        name: int(round(max_samples * fraction))
        for name, fraction in AGRO_STRATUM_FRACTIONS.items()
    }
    out["cropland_dominant"] += max_samples - sum(out.values())
    return out


def _geo_bin(lat: float, lon: float) -> str:
    return f"{int(np.floor(lat / 10.0) * 10):+03d}:{int(np.floor(lon / 10.0) * 10):+04d}"


def _candidate_geo_bins(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                _geo_bin(float(candidate["center_lat"]), float(candidate["center_lon"]))
                for candidate in candidates
            ).items()
        )
    )


def _clear_fraction_bucket(clear_fraction: float) -> str:
    if not np.isfinite(clear_fraction) or not 0.0 <= clear_fraction <= 1.0:
        raise ValueError(f"Clear fraction must be within [0, 1], got {clear_fraction}")
    bucket = min(9, int(np.floor(clear_fraction / CLEAR_FRACTION_BUCKET_WIDTH)))
    return f"{bucket * 10:02d}-{(bucket + 1) * 10:03d}%"


def _composite_match_bin(lat: float, lon: float, clear_fraction: float, token_available_fraction: float) -> str:
    return f"{_geo_bin(lat, lon)}|clear={_clear_fraction_bucket(clear_fraction)}|token={_clear_fraction_bucket(token_available_fraction)}"


def _candidate_composite_bins(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                _composite_match_bin(
                    float(candidate["center_lat"]),
                    float(candidate["center_lon"]),
                    float(candidate["s2_clear_fraction"]),
                    float(candidate["s2_token_available_fraction"]),
                )
                for candidate in candidates
            ).items()
        )
    )


def _decode_text(value: Any) -> str:
    return value.decode("utf-8").rstrip("\x00") if isinstance(value, bytes) else str(value)


def _store_exists(path: Path) -> bool:
    return (path / ".zgroup").exists() or (path / "zarr.json").exists()


def _store_arrays(path: Path, max_samples: int, patch_size: int, pool: str):
    if patch_size % TOKEN_PATCH_SIZE != 0:
        raise ValueError(f"patch_size={patch_size} must be a multiple of TOKEN_PATCH_SIZE={TOKEN_PATCH_SIZE}")
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = _store_exists(path)
    root = _open_output_group(path, mode="a" if exists else "w")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    n_chunk = min(128, max_samples)
    t = 4
    h = w = patch_size
    arrays = {
        "s2": _ensure_array(root, "s2", (max_samples, t, len(S2_BANDS), h, w), (n_chunk, t, len(S2_BANDS), h, w), "float32", compressor),
        "s1": _ensure_array(root, "s1", (max_samples, t, len(S1_BANDS), h, w), (n_chunk, t, len(S1_BANDS), h, w), "float32", compressor),
        "s2_mask": _ensure_array(root, "s2_mask", (max_samples, t, h, w), (n_chunk, t, h, w), "uint8", compressor),
        "s1_mask": _ensure_array(root, "s1_mask", (max_samples, t, h, w), (n_chunk, t, h, w), "uint8", compressor),
        "dem": _ensure_array(root, "dem", (max_samples, h, w), (n_chunk, h, w), "float32", compressor),
        "lulc": _ensure_array(root, "lulc", (max_samples, h, w), (n_chunk, h, w), "uint8", compressor),
        "s2_time_ns": _ensure_array(root, "s2_time_ns", (max_samples, t), (n_chunk, t), "int64", compressor),
        "s1_time_ns": _ensure_array(root, "s1_time_ns", (max_samples, t), (n_chunk, t), "int64", compressor),
        "center_lat": _ensure_array(root, "center_lat", (max_samples,), (n_chunk,), "float64", compressor),
        "center_lon": _ensure_array(root, "center_lon", (max_samples,), (n_chunk,), "float64", compressor),
        "elevation": _ensure_array(root, "elevation", (max_samples,), (n_chunk,), "float32", compressor),
        "lulc_cropland_fraction": _ensure_array(root, "lulc_cropland_fraction", (max_samples,), (n_chunk,), "float32", compressor),
        "s2_clear_fraction": _ensure_array(root, "s2_clear_fraction", (max_samples,), (n_chunk,), "float32", compressor),
        "s2_token_available_fraction": _ensure_array(root, "s2_token_available_fraction", (max_samples,), (n_chunk,), "float32", compressor),
        "source_shard": _ensure_array(root, "source_shard", (max_samples,), (n_chunk,), "int32", compressor),
        "sample": _ensure_array(root, "sample", (max_samples,), (n_chunk,), f"S{SAMPLE_ID_WIDTH}", compressor),
        "sampling_stratum": _ensure_array(root, "sampling_stratum", (max_samples,), (n_chunk,), "S32", compressor),
        "patch_xy": _ensure_array(root, "patch_xy", (max_samples, 2), (n_chunk, 2), "int32", compressor),
    }
    if "s2_bands" not in root:
        _replace_array(root, "s2_bands", shape=(len(S2_BANDS),), dtype="S32", chunks=(len(S2_BANDS),))[:] = _encode_text(S2_BANDS, 32)
    if "s1_bands" not in root:
        _replace_array(root, "s1_bands", shape=(len(S1_BANDS),), dtype="S32", chunks=(len(S1_BANDS),))[:] = _encode_text(S1_BANDS, 32)
    root.attrs.update(
        {
            "source": SSL4EO_WDS_REPO,
            "source_revision": SSL4EO_WDS_REVISION,
            "pool": pool,
            "climate": "omitted",
            "lulc_usage": "static_esri_original_sampling_metadata_only",
            "clear_fraction_bucket_width": CLEAR_FRACTION_BUCKET_WIDTH,
            "token_patch_size": TOKEN_PATCH_SIZE,
            "min_token_clear_fraction": MIN_TOKEN_CLEAR_FRACTION,
            "geographic_exclusions": json.dumps(EXCLUDED_REGIONS, sort_keys=True),
            "normalization_contract": json.dumps(NORMALIZATION_CONTRACT, sort_keys=True),
            "build_schema_version": BUILD_SCHEMA_VERSION,
        }
    )
    return root, arrays


def _write_sample(
    arrays: dict[str, Any],
    write_idx: int,
    key: str,
    stratum: str,
    cropland_fraction: float,
    roots: dict[str, Any],
    center: slice,
    source_shard: int,
) -> None:
    if len(key.encode("utf-8")) > SAMPLE_ID_WIDTH:
        raise ValueError(f"Sample ID exceeds {SAMPLE_ID_WIDTH} bytes: {key}")
    s2_root = roots["S2L2A"]
    s1_root = roots["S1GRD"]
    ndvi_root = roots["NDVI"]
    dem_root = roots["DEM"]
    lulc_root = roots["LULC"]
    s2 = _center_patch(np.asarray(s2_root["bands"][...])[:, S2L2A_KEEP], center).astype(np.float32) / 10000.0
    s1 = _center_patch(np.asarray(s1_root["bands"][...]), center).astype(np.float32)
    ndvi = _center_patch(np.asarray(ndvi_root["bands"][...]), center).astype(np.float32)
    dem = _center_patch(np.asarray(dem_root["bands"][...]), center).astype(np.float32)[0, 0]
    lulc = _center_patch(np.asarray(lulc_root["esri_original"][...]), center).astype(np.uint8)[0]
    clouds = _center_patch(np.asarray(s2_root["cloud_mask"][...]), center).astype(np.uint8)
    arrays["s2"][write_idx] = np.nan_to_num(np.concatenate([s2, ndvi], axis=1))
    arrays["s1"][write_idx] = np.nan_to_num(s1)
    arrays["s2_mask"][write_idx] = (clouds == 0).astype(np.uint8)
    arrays["s1_mask"][write_idx] = 1
    arrays["dem"][write_idx] = np.nan_to_num(dem)
    arrays["lulc"][write_idx] = lulc
    s2_time = _time_ns(s2_root["time"][...])
    s1_time = _time_ns(s1_root["time"][...])
    if np.any(s2_time == 0) and np.any(s1_time != 0):
        s2_time = np.where(s2_time == 0, s1_time, s2_time)
    arrays["s2_time_ns"][write_idx] = s2_time
    arrays["s1_time_ns"][write_idx] = s1_time
    arrays["center_lat"][write_idx] = _scalar(s2_root, "center_lat")
    arrays["center_lon"][write_idx] = _scalar(s2_root, "center_lon")
    arrays["elevation"][write_idx] = float(np.nanmean(dem))
    arrays["lulc_cropland_fraction"][write_idx] = cropland_fraction
    arrays["s2_clear_fraction"][write_idx] = float(np.mean(clouds == 0))
    clear_mask = (clouds == 0).astype(np.float32)
    _t, _h, _w = clear_mask.shape
    _ps = _h // TOKEN_PATCH_SIZE
    arrays["s2_token_available_fraction"][write_idx] = float(
        (clear_mask.reshape(_t, _ps, TOKEN_PATCH_SIZE, _ps, TOKEN_PATCH_SIZE).mean(axis=(2, 4)) >= MIN_TOKEN_CLEAR_FRACTION).mean()
    )
    arrays["source_shard"][write_idx] = source_shard
    arrays["sampling_stratum"][write_idx] = stratum.encode("utf-8")[:32]
    arrays["patch_xy"][write_idx] = 0
    arrays["sample"][write_idx] = key.encode("utf-8")


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _geo_bins(lat: np.ndarray, lon: np.ndarray) -> dict[str, int]:
    bins = Counter(_geo_bin(float(sample_lat), float(sample_lon)) for sample_lat, sample_lon in zip(lat, lon))
    return dict(sorted(bins.items()))


def _joint_composite_bins(lat: np.ndarray, lon: np.ndarray, clear_fraction: np.ndarray, token_available_fraction: np.ndarray) -> dict[str, int]:
    bins = Counter(
        _composite_match_bin(float(sample_lat), float(sample_lon), float(sample_clear), float(sample_token))
        for sample_lat, sample_lon, sample_clear, sample_token in zip(lat, lon, clear_fraction, token_available_fraction)
    )
    return dict(sorted(bins.items()))


def _write_summary(path: Path, pool: str, arrays: dict[str, Any], sample_count: int, last_shard: int) -> dict[str, Any]:
    sample_ids = [_decode_text(value) for value in np.asarray(arrays["sample"][:sample_count])]
    if any(not value for value in sample_ids):
        raise ValueError(f"{path} contains empty sample IDs")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path} contains duplicate sample IDs")
    strata = Counter(_decode_text(value) for value in np.asarray(arrays["sampling_stratum"][:sample_count]))
    lat = np.asarray(arrays["center_lat"][:sample_count])
    lon = np.asarray(arrays["center_lon"][:sample_count])
    clear_fraction = np.asarray(arrays["s2_clear_fraction"][:sample_count])
    token_available_fraction = np.asarray(arrays["s2_token_available_fraction"][:sample_count])
    summary = {
        "output_zarr": str(path),
        "pool": pool,
        "samples": int(sample_count),
        "unique_sample_ids": len(set(sample_ids)),
        "build_schema_version": BUILD_SCHEMA_VERSION,
        "source_revision": SSL4EO_WDS_REVISION,
        "last_shard_read": int(last_shard),
        "sampling_strata": dict(sorted(strata.items())),
        "source_shards": dict(sorted(Counter(int(value) for value in np.asarray(arrays["source_shard"][:sample_count])).items())),
        "lulc_cropland_fraction": _numeric_summary(np.asarray(arrays["lulc_cropland_fraction"][:sample_count])),
        "s2_clear_fraction": _numeric_summary(clear_fraction),
        "s2_token_available_fraction": _numeric_summary(token_available_fraction),
        "center_lat": _numeric_summary(lat),
        "center_lon": _numeric_summary(lon),
        "geography_bins_10deg": _geo_bins(lat, lon),
        "geography_composite_bins": _joint_composite_bins(lat, lon, clear_fraction, token_available_fraction),
        "normalization_contract": NORMALIZATION_CONTRACT,
        "status": "complete",
    }
    (path / "build_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp_path.replace(path)


def _state_config(
    generic_path: Path,
    agro_path: Path,
    max_samples: int,
    patch_size: int,
    start_shard: int,
) -> dict[str, Any]:
    return {
        "schema_version": BUILD_SCHEMA_VERSION,
        "source_revision": SSL4EO_WDS_REVISION,
        "generic_output_zarr": str(generic_path),
        "agro_output_zarr": str(agro_path),
        "max_samples": int(max_samples),
        "patch_size": int(patch_size),
        "start_shard": int(start_shard),
        "max_train_shard": MAX_TRAIN_SHARD,
        "sampling_seed": SAMPLING_SEED,
        "agro_stratum_fractions": AGRO_STRATUM_FRACTIONS,
        "excluded_regions": EXCLUDED_REGIONS,
        "modalities": MODALITIES,
        "sample_id_width": SAMPLE_ID_WIDTH,
        "clear_fraction_bucket_width": CLEAR_FRACTION_BUCKET_WIDTH,
        "token_patch_size": TOKEN_PATCH_SIZE,
        "min_token_clear_fraction": MIN_TOKEN_CLEAR_FRACTION,
        "normalization_contract": NORMALIZATION_CONTRACT,
    }


def _stable_priority(pool: str, key: str) -> str:
    return hashlib.sha256(f"{SAMPLING_SEED}:{pool}:{key}".encode("utf-8")).hexdigest()


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    by_key: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            candidate = json.loads(line)
            by_key[str(candidate["sample"])] = candidate
    return list(by_key.values())


def _append_candidates(path: Path, candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, sort_keys=True) + "\n")


def _max_flow_allocations(candidates: list[dict[str, Any]], quotas: dict[str, int]) -> dict[tuple[str, str], int]:
    source = ("source", "")
    sink = ("sink", "")
    residual: dict[tuple[str, str], dict[tuple[str, str], int]] = {}

    def add_edge(start: tuple[str, str], end: tuple[str, str], capacity: int) -> None:
        residual.setdefault(start, {})
        residual.setdefault(end, {})
        residual[start][end] = capacity
        residual[end].setdefault(start, 0)

    edge_counts = Counter(
        (str(candidate["stratum"]), _composite_match_bin(
            float(candidate["center_lat"]),
            float(candidate["center_lon"]),
            float(candidate["s2_clear_fraction"]),
            float(candidate["s2_token_available_fraction"]),
        ))
        for candidate in candidates
        if str(candidate["stratum"]) in quotas
    )
    match_counts = Counter(
        _composite_match_bin(
            float(candidate["center_lat"]),
            float(candidate["center_lon"]),
            float(candidate["s2_clear_fraction"]),
            float(candidate["s2_token_available_fraction"]),
        )
        for candidate in candidates
    )
    for stratum, quota in sorted(quotas.items()):
        add_edge(source, ("stratum", stratum), quota)
    for (stratum, match_bin), count in sorted(edge_counts.items()):
        add_edge(("stratum", stratum), ("match", match_bin), count)
    for match_bin, count in sorted(match_counts.items()):
        add_edge(("match", match_bin), sink, count // 2)

    flow = 0
    while True:
        previous = {source: None}
        queue = deque([source])
        while queue and sink not in previous:
            current = queue.popleft()
            for neighbor, capacity in sorted(residual[current].items()):
                if capacity > 0 and neighbor not in previous:
                    previous[neighbor] = current
                    queue.append(neighbor)
        if sink not in previous:
            break
        capacity = sum(quotas.values())
        current = sink
        while previous[current] is not None:
            capacity = min(capacity, residual[previous[current]][current])
            current = previous[current]
        current = sink
        while previous[current] is not None:
            residual[previous[current]][current] -= capacity
            residual[current][previous[current]] += capacity
            current = previous[current]
        flow += capacity
    required = sum(quotas.values())
    if flow != required:
        fulfilled = {
            stratum: quota - residual[source][("stratum", stratum)]
            for stratum, quota in sorted(quotas.items())
        }
        raise ValueError(f"Could allocate only {flow}/{required} geography-and-cloud-matchable agriculture samples: {fulfilled}")
    return {
        (stratum, match_bin): count - residual[("stratum", stratum)][("match", match_bin)]
        for (stratum, match_bin), count in edge_counts.items()
        if count - residual[("stratum", stratum)][("match", match_bin)] > 0
    }


def _select_candidates(candidates: list[dict[str, Any]], max_samples: int) -> dict[str, list[dict[str, Any]]]:
    eligible = list(candidates)
    quotas = _quota_counts(max_samples)
    stratum_bin_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    bin_index: dict[str, list[dict[str, Any]]] = {}
    for item in eligible:
        stratum = str(item["stratum"])
        composite_bin = _composite_match_bin(
            float(item["center_lat"]),
            float(item["center_lon"]),
            float(item["s2_clear_fraction"]),
            float(item["s2_token_available_fraction"]),
        )
        if stratum in quotas:
            stratum_bin_index.setdefault((stratum, composite_bin), []).append(item)
        bin_index.setdefault(composite_bin, []).append(item)

    agro: list[dict[str, Any]] = []
    allocations = _max_flow_allocations(eligible, quotas)
    for (stratum, match_bin), count in sorted(allocations.items()):
        matching = sorted(
            stratum_bin_index.get((stratum, match_bin), []),
            key=lambda item: _stable_priority(f"{POOL_AGRO}:{stratum}:{match_bin}", str(item["sample"]))
        )[:count]
        agro.extend(matching)
    agro_ids = {str(item["sample"]) for item in agro}
    agro_composite_bins = _candidate_composite_bins(agro)
    generic: list[dict[str, Any]] = []
    for match_bin, count in sorted(agro_composite_bins.items()):
        matching = [
            item
            for item in bin_index.get(match_bin, [])
            if str(item["sample"]) not in agro_ids
        ]
        if len(matching) < count:
            raise ValueError(f"Only {len(matching)} disjoint generic samples in matching bin {match_bin}; need {count}")
        generic.extend(sorted(matching, key=lambda item: _stable_priority(f"{POOL_GENERIC}:{match_bin}", str(item["sample"])))[:count])
    generic_ids = {str(item["sample"]) for item in generic}
    if agro_ids & generic_ids:
        raise ValueError("Generic and agriculture pools overlap")
    if _candidate_geo_bins(generic) != _candidate_geo_bins(agro):
        raise ValueError("Generic and agriculture pool geographic histograms do not match")
    if _candidate_composite_bins(generic) != _candidate_composite_bins(agro):
        raise ValueError("Generic and agriculture pool geography-and-cloud histograms do not match")
    return {POOL_GENERIC: generic, POOL_AGRO: agro}


def _candidate_summary(candidates: list[dict[str, Any]], excluded_counts: dict[str, int]) -> dict[str, Any]:
    if not candidates:
        return {"eligible_samples": 0, "excluded_regions": dict(sorted(excluded_counts.items()))}
    token_avail = np.asarray([item["s2_token_available_fraction"] for item in candidates])
    return {
        "eligible_samples": len(candidates),
        "excluded_regions": dict(sorted(excluded_counts.items())),
        "sampling_strata": dict(sorted(Counter(str(item["stratum"]) for item in candidates).items())),
        "lulc_cropland_fraction": _numeric_summary(np.asarray([item["cropland_fraction"] for item in candidates])),
        "s2_clear_fraction": _numeric_summary(np.asarray([item["s2_clear_fraction"] for item in candidates])),
        "s2_token_available_fraction": _numeric_summary(token_avail),
        "geography_bins_10deg": _geo_bins(
            np.asarray([item["center_lat"] for item in candidates]),
            np.asarray([item["center_lon"] for item in candidates]),
        ),
        "geography_composite_bins": _candidate_composite_bins(candidates),
    }


def probe_lulc_archive(path: str | Path, patch_size: int = 16) -> dict[str, Any]:
    """Verify static ESRI crop labels and remapped clear-pixel labels in one LULC shard."""

    archive, members = _tar_members(path)
    esri_classes: set[int] = set()
    remapped_classes: set[int] = set()
    crop_clear_pixels = 0
    crop_remapped_pixels = 0
    strata: Counter[str] = Counter()
    try:
        for member in members.values():
            root = _read_zarr(archive, member)
            if "esri_original" not in root:
                raise ValueError("LULC sample is missing static esri_original labels")
            esri = np.asarray(root["esri_original"][...], dtype=np.uint8)
            remapped = np.asarray(root["bands"][...], dtype=np.uint8)[:, 0]
            clouds = np.asarray(root["cloud_mask"][...], dtype=np.uint8)
            esri_classes.update(int(value) for value in np.unique(esri))
            remapped_classes.update(int(value) for value in np.unique(remapped))
            clear_crop = (esri == ESRI_CROPLAND_CLASS) & (clouds == 0)
            crop_clear_pixels += int(clear_crop.sum())
            crop_remapped_pixels += int(((remapped == 4) & clear_crop).sum())
            center = _center_slice(int(esri.shape[-1]), patch_size)
            stratum, _ = _lulc_stratum(_center_patch(esri, center)[0])
            strata[stratum or "other"] += 1
    finally:
        archive.close()
    if max(remapped_classes) > 9:
        raise ValueError(f"Expected remapped LULC labels within [0, 9], got {sorted(remapped_classes)}")
    if crop_clear_pixels <= 0:
        raise ValueError("One-shard probe found no clear ESRI cropland pixels")
    if crop_clear_pixels != crop_remapped_pixels:
        raise ValueError(
            f"Static ESRI crop mapping mismatch: {crop_remapped_pixels}/{crop_clear_pixels} "
            "clear crop pixels map to remapped class 4"
        )
    return {
        "archive": str(path),
        "samples": len(members),
        "esri_classes": sorted(esri_classes),
        "remapped_classes": sorted(remapped_classes),
        "clear_esri_crop_pixels": crop_clear_pixels,
        "clear_esri_crop_pixels_mapped_to_class_4": crop_remapped_pixels,
        "sampling_strata": dict(sorted(strata.items())),
    }


def _assert_center_patch_cloud_parity(lulc_clouds: np.ndarray, s2_clouds: np.ndarray, key: str, patch_size: int) -> None:
    """Raise ValueError if LULC and S2L2A center-patch cloud masks differ."""
    if lulc_clouds.shape != s2_clouds.shape:
        raise ValueError(f"Shape mismatch for {key}: LULC {lulc_clouds.shape} vs S2L2A {s2_clouds.shape}")
    if not np.array_equal(lulc_clouds, s2_clouds):
        raise ValueError(
            f"LULC/S2L2A center-patch cloud mask mismatch for {key}: "
            f"{int((lulc_clouds != s2_clouds).sum())} pixels differ"
        )


def probe_aligned_cloud_parity(shard: int, cache_dir: str | Path, num_samples: int = CLOUD_PARITY_NUM_SAMPLES, patch_size: int = 16) -> dict[str, Any]:
    """Compare LULC and S2L2A center-patch cloud masks on the same shard.

    Raises ValueError if any pixel differs (exact equality required) or if fewer
    than ``num_samples`` aligned samples exist in the shard.
    """
    cache_path = Path(cache_dir).expanduser().resolve()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        lulc_future = executor.submit(_download_archive, "LULC", shard, cache_path)
        s2_future = executor.submit(_download_archive, "S2L2A", shard, cache_path)
        lulc_path = lulc_future.result()
        s2_path = s2_future.result()
    lulc_archive, lulc_members = _tar_members(lulc_path)
    s2_archive, s2_members = _tar_members(s2_path)
    checked = 0
    total_diff = 0
    try:
        for key in sorted(set(lulc_members) & set(s2_members)):
            if checked >= num_samples:
                break
            lulc_root = _read_zarr(lulc_archive, lulc_members[key])
            s2_root = _read_zarr(s2_archive, s2_members[key])
            lulc_full = np.asarray(lulc_root["cloud_mask"][...], dtype=np.uint8)
            s2_full = np.asarray(s2_root["cloud_mask"][...], dtype=np.uint8)
            center = _center_slice(lulc_full.shape[-1], patch_size)
            lulc_center = lulc_full[..., center, center]
            s2_center = s2_full[..., center, center]
            _assert_center_patch_cloud_parity(lulc_center, s2_center, key, patch_size)
            total_diff += int((lulc_center != s2_center).sum())
            checked += 1
    finally:
        lulc_archive.close()
        s2_archive.close()
    if checked < num_samples:
        raise ValueError(f"Aligned cloud parity probe checked only {checked}/{num_samples} samples")
    return {
        "shard": shard,
        "samples_checked": checked,
        "total_pixel_diff": total_diff,
        "requested_samples": num_samples,
        "patch_size": patch_size,
    }


def build_ssl4eo_v11_fixed_pools(
    generic_output_zarr: str | Path,
    agro_output_zarr: str | Path,
    cache_dir: str | Path,
    max_samples: int = 49152,
    patch_size: int = 16,
    start_shard: int = 1,
    max_cache_gib: float = 60.0,
    evict_cached_shards: bool = True,
) -> dict[str, Any]:
    """Build deterministic equal-sized generic and agriculture-focused stores."""

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if start_shard < 1 or start_shard > MAX_TRAIN_SHARD:
        raise ValueError(f"start_shard must be within [1, {MAX_TRAIN_SHARD}]")
    if max_cache_gib <= 0:
        raise ValueError("max_cache_gib must be positive")
    generic_path = Path(generic_output_zarr).expanduser().resolve()
    agro_path = Path(agro_output_zarr).expanduser().resolve()
    cache_path = Path(cache_dir).expanduser().resolve()
    state_path = agro_path.parent / "preprocess_8_build_state.json"
    candidates_path = agro_path.parent / "preprocess_8_candidates.jsonl"
    selected_path = agro_path.parent / "preprocess_8_selected.json"
    config = _state_config(generic_path, agro_path, max_samples, patch_size, start_shard)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("config") != config:
            raise ValueError(
                f"{state_path} does not match the current [8] build schema. "
                "Remove the stale [8] stores and preprocess_8 state files before rebuilding."
            )
    else:
        if _store_exists(generic_path) or _store_exists(agro_path) or candidates_path.exists() or selected_path.exists():
            raise ValueError(
                "Found [8] preprocessing files without a matching state file. "
                "Remove the stale [8] stores and preprocess_8 state files before rebuilding."
            )
        state = {
            "status": "running",
            "phase": "scan_candidates",
            "next_scan_shard": start_shard,
            "excluded_regions": {},
            "config": config,
        }
        _write_json(state_path, state)
    cache_path.mkdir(parents=True, exist_ok=True)
    _assert_cache_budget(cache_path, max_cache_gib)
    candidates = _load_candidates(candidates_path)
    known_candidates = {str(item["sample"]) for item in candidates}
    excluded_counts = Counter({key: int(value) for key, value in state.get("excluded_regions", {}).items()})
    if state["phase"] == "scan_candidates":
        for shard in range(int(state["next_scan_shard"]), MAX_TRAIN_SHARD + 1):
            archive_path = _download_archive("LULC", shard, cache_path)
            _assert_cache_budget(cache_path, max_cache_gib)
            archive, members = _tar_members(archive_path)
            shard_candidates: list[dict[str, Any]] = []
            try:
                for key, member in sorted(members.items()):
                    if key in known_candidates:
                        continue
                    root = _read_zarr(archive, member)
                    center = _center_slice(int(root["bands"].shape[-1]), patch_size)
                    esri_lulc = _center_patch(np.asarray(root["esri_original"][...]), center).astype(np.uint8)[0]
                    stratum, cropland_fraction = _lulc_stratum(esri_lulc)
                    lat = _scalar(root, "center_lat")
                    lon = _scalar(root, "center_lon")
                    excluded = _excluded_region(lat, lon)
                    if excluded is not None:
                        excluded_counts[excluded] += 1
                        continue
                    clouds = _center_patch(np.asarray(root["cloud_mask"][...]), center)
                    clear_mask = (clouds == 0).astype(np.float32)
                    clear_frac = float(np.mean(clear_mask))
                    _t, _h, _w = clear_mask.shape
                    _ps = _h // TOKEN_PATCH_SIZE
                    token_available_frac = float(
                        (clear_mask.reshape(_t, _ps, TOKEN_PATCH_SIZE, _ps, TOKEN_PATCH_SIZE).mean(axis=(2, 4)) >= MIN_TOKEN_CLEAR_FRACTION).mean()
                    )
                    candidate = {
                        "sample": key,
                        "source_shard": shard,
                        "stratum": stratum or "other",
                        "cropland_fraction": cropland_fraction,
                        "s2_clear_fraction": clear_frac,
                        "s2_token_available_fraction": token_available_frac,
                        "center_lat": lat,
                        "center_lon": lon,
                    }
                    shard_candidates.append(candidate)
                    known_candidates.add(key)
            finally:
                archive.close()
                if evict_cached_shards:
                    _evict_cached_files([archive_path], cache_path)
                    _assert_cache_budget(cache_path, max_cache_gib)
            _append_candidates(candidates_path, shard_candidates)
            candidates.extend(shard_candidates)
            state.update(
                {
                    "next_scan_shard": shard + 1,
                    "eligible_candidates": len(candidates),
                    "excluded_regions": dict(sorted(excluded_counts.items())),
                }
            )
            _write_json(state_path, state)
            print(json.dumps({"phase": "scan_candidates", "shard": shard, "eligible_candidates": len(candidates)}), flush=True)
        selected = _select_candidates(candidates, max_samples)
        _write_json(selected_path, selected)
        state.update(
            {
                "phase": "materialize_selected",
                "selected_generic": len(selected[POOL_GENERIC]),
                "selected_agro": len(selected[POOL_AGRO]),
                "next_materialize_index": 0,
            }
        )
        _write_json(state_path, state)
    else:
        if not selected_path.exists():
            raise ValueError(f"Missing selected-sample manifest: {selected_path}")
        selected = json.loads(selected_path.read_text())

    generic_root, generic = _store_arrays(generic_path, max_samples, patch_size, POOL_GENERIC)
    agro_root, agro = _store_arrays(agro_path, max_samples, patch_size, POOL_AGRO)
    generic_root.attrs["build_complete"] = False
    agro_root.attrs["build_complete"] = False
    assignments: dict[str, list[tuple[dict[str, Any], int, dict[str, Any]]]] = {}
    assignments_by_shard: dict[int, set[str]] = {}
    for pool, arrays in [(POOL_GENERIC, generic), (POOL_AGRO, agro)]:
        for write_idx, candidate in enumerate(selected[pool]):
            key = str(candidate["sample"])
            shard = int(candidate["source_shard"])
            assignments.setdefault(key, []).append((arrays, write_idx, candidate))
            assignments_by_shard.setdefault(shard, set()).add(key)
    selected_shards = sorted({int(candidate["source_shard"]) for values in selected.values() for candidate in values})
    for shard_index in range(int(state.get("next_materialize_index", 0)), len(selected_shards)):
        shard = selected_shards[shard_index]
        archive_paths = _download_modalities(shard, cache_path)
        _assert_cache_budget(cache_path, max_cache_gib)
        opened = {modality: _tar_members(path) for modality, path in archive_paths.items()}
        try:
            member_maps = {modality: members for modality, (_, members) in opened.items()}
            keys = sorted(assignments_by_shard[shard])
            for key in keys:
                roots = {modality: _read_zarr(opened[modality][0], member_maps[modality][key]) for modality in MODALITIES}
                center = _center_slice(int(roots["LULC"]["bands"].shape[-1]), patch_size)
                lulc_center = _center_patch(np.asarray(roots["LULC"]["cloud_mask"][...]), center).astype(np.uint8)
                s2_center = _center_patch(np.asarray(roots["S2L2A"]["cloud_mask"][...]), center).astype(np.uint8)
                _assert_center_patch_cloud_parity(lulc_center, s2_center, key, patch_size)
                for arrays, write_idx, candidate in assignments[key]:
                    existing = _decode_text(arrays["sample"][write_idx])
                    if existing:
                        if existing != key:
                            raise ValueError(f"Slot {write_idx} contains {existing}; expected {key}")
                        continue
                    stratum = "generic" if arrays is generic else str(candidate["stratum"])
                    _write_sample(
                        arrays,
                        write_idx,
                        key,
                        stratum,
                        float(candidate["cropland_fraction"]),
                        roots,
                        center,
                        shard,
                    )
        finally:
            for archive, _ in opened.values():
                archive.close()
            if evict_cached_shards:
                _evict_cached_files(list(archive_paths.values()), cache_path)
                _assert_cache_budget(cache_path, max_cache_gib)
        state["next_materialize_index"] = shard_index + 1
        _write_json(state_path, state)
        print(json.dumps({"phase": "materialize_selected", "shard": shard, "completed_shards": shard_index + 1, "total_shards": len(selected_shards)}), flush=True)

    generic_summary = _write_summary(generic_path, POOL_GENERIC, generic, max_samples, selected_shards[-1])
    agro_summary = _write_summary(agro_path, POOL_AGRO, agro, max_samples, selected_shards[-1])
    generic_ids = {_decode_text(value) for value in np.asarray(generic["sample"][:])}
    agro_ids = {_decode_text(value) for value in np.asarray(agro["sample"][:])}
    generic_geo_bins = generic_summary["geography_bins_10deg"]
    agro_geo_bins = agro_summary["geography_bins_10deg"]
    generic_composite_bins = generic_summary["geography_composite_bins"]
    agro_composite_bins = agro_summary["geography_composite_bins"]
    if generic_ids & agro_ids:
        raise ValueError("Generic and agriculture stores overlap")
    if generic_geo_bins != agro_geo_bins:
        raise ValueError("Generic and agriculture store geographic histograms do not match")
    if generic_composite_bins != agro_composite_bins:
        raise ValueError("Generic and agriculture store composite matching histograms do not match")
    state.update(
        {
            "status": "complete",
            "phase": "complete",
            "generic_samples": max_samples,
            "agro_samples": max_samples,
            "generic_agro_overlap": len(generic_ids & agro_ids),
            "geography_histograms_match": generic_geo_bins == agro_geo_bins,
            "geography_composite_bins_match": generic_composite_bins == agro_composite_bins,
        }
    )
    _write_json(state_path, state)
    generic_root.attrs["build_complete"] = True
    agro_root.attrs["build_complete"] = True
    return {
        "generic": generic_summary,
        "agro": agro_summary,
        "candidate_scan": _candidate_summary(candidates, dict(excluded_counts)),
        "generic_agro_overlap": len(generic_ids & agro_ids),
        "geography_histograms_match": generic_geo_bins == agro_geo_bins,
        "geography_composite_bins_match": generic_composite_bins == agro_composite_bins,
        "state": state,
    }
