"""Convert CropHarvest HDF5 arrays into the local multimodal Zarr layout."""

import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import zarr
from numcodecs import Blosc

# CropHarvest band ordering from nasaharvest/cropharvest/cropharvest/bands.py
# BANDS = [S1 (VV, VH)] + [S2 without B1,B10 -> 11 bands] + [ERA5 (2)] + [SRTM (2)] + [NDVI]
S1_IDXS = [0, 1]
S2_IDXS = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 17]
CLIMATE_IDXS = [13, 14, 15]
CROPHARVEST_MIN_CHANNELS = max(S1_IDXS + S2_IDXS + CLIMATE_IDXS) + 1

S1_BANDS = ["VV", "VH"]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
CLIMATE_BANDS = ["temperature", "precipitation", "elevation"]


def _encode_text(values: list[str], width: int) -> np.ndarray:
    return np.asarray([v.encode("utf-8")[:width] for v in values], dtype=f"S{width}")


def _open_output_group(path: Path):
    try:
        return zarr.open_group(str(path), mode="w", zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode="w", zarr_version=2)


def _create_array(root, name: str, **kwargs):
    if hasattr(root, "create_dataset"):
        return root.create_dataset(name, **kwargs)
    return root.create_array(name, **kwargs)


def _iter_h5_files(arrays_dir: Path) -> Iterable[Path]:
    for file_path in sorted(arrays_dir.glob("*.h5")):
        if file_path.is_file():
            yield file_path


def _load_cropharvest_array(file_path: Path) -> np.ndarray | None:
    try:
        with h5py.File(file_path, "r") as f:
            if "array" not in f:
                return None
            return np.asarray(f["array"], dtype=np.float32)
    except (OSError, KeyError):
        return None


def build_cropharvest_zarr(
    arrays_dir: str | Path,
    output_zarr: str | Path,
    max_samples: int | None = None,
) -> dict[str, int]:
    arrays_dir = Path(arrays_dir).expanduser().resolve()
    output_zarr = Path(output_zarr).expanduser().resolve()
    if not arrays_dir.exists():
        raise FileNotFoundError(f"CropHarvest arrays directory not found: {arrays_dir}")
    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    files = list(_iter_h5_files(arrays_dir))
    if not files:
        raise FileNotFoundError(f"No .h5 arrays found in {arrays_dir}")
    if max_samples is not None and max_samples > 0:
        files = files[:max_samples]

    # Inspect first valid sample for timestep count and sanity checks.
    arr0 = None
    for candidate in files:
        arr0 = _load_cropharvest_array(candidate)
        if arr0 is not None:
            break
    if arr0 is None:
        raise ValueError("No valid CropHarvest files with 'array' dataset were found.")
    if arr0.ndim != 2 or arr0.shape[1] < CROPHARVEST_MIN_CHANNELS:
        raise ValueError(f"Unexpected CropHarvest array shape: {arr0.shape}")
    t = int(arr0.shape[0])
    n = len(files)
    h = 1
    w = 1

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    root = _open_output_group(output_zarr)
    s2_arr = _create_array(
        root,
        "s2",
        shape=(n, t, len(S2_IDXS), h, w),
        chunks=(min(1024, n), t, len(S2_IDXS), h, w),
        dtype="float32",
        compressor=compressor,
    )
    s1_arr = _create_array(
        root,
        "s1",
        shape=(n, t, len(S1_IDXS), h, w),
        chunks=(min(1024, n), t, len(S1_IDXS), h, w),
        dtype="float32",
        compressor=compressor,
    )
    climate_arr = _create_array(
        root,
        "climate",
        shape=(n, t, len(CLIMATE_BANDS), h, w),
        chunks=(min(1024, n), t, len(CLIMATE_BANDS), h, w),
        dtype="float32",
        compressor=compressor,
    )

    s2_mask = _create_array(
        root,
        "s2_mask",
        shape=(n, t, h, w),
        chunks=(min(1024, n), t, h, w),
        dtype="uint8",
        compressor=compressor,
    )
    s1_mask = _create_array(
        root,
        "s1_mask",
        shape=(n, t, h, w),
        chunks=(min(1024, n), t, h, w),
        dtype="uint8",
        compressor=compressor,
    )
    climate_mask = _create_array(
        root,
        "climate_mask",
        shape=(n, t, h, w),
        chunks=(min(1024, n), t, h, w),
        dtype="uint8",
        compressor=compressor,
    )
    patch_xy = _create_array(
        root,
        "patch_xy",
        shape=(n, 2),
        chunks=(min(1024, n), 2),
        dtype="int32",
        compressor=compressor,
    )

    s2_mask[:] = 1
    s1_mask[:] = 1
    climate_mask[:] = 1
    patch_xy[:] = 0

    write_idx = 0
    skipped = 0
    for file_path in files:
        arr = _load_cropharvest_array(file_path)
        if arr is None:
            skipped += 1
            continue
        if arr.shape[0] != t:
            skipped += 1
            continue
        s1_arr[write_idx, :, :, 0, 0] = np.nan_to_num(arr[:, S1_IDXS], nan=0.0)
        s2_arr[write_idx, :, :, 0, 0] = np.nan_to_num(arr[:, S2_IDXS], nan=0.0)
        climate_arr[write_idx, :, :, 0, 0] = np.nan_to_num(arr[:, CLIMATE_IDXS], nan=0.0)
        write_idx += 1

    if write_idx == 0:
        raise ValueError("No valid CropHarvest arrays were converted.")

    if write_idx < n:
        s2_arr.resize((write_idx, t, len(S2_IDXS), h, w))
        s1_arr.resize((write_idx, t, len(S1_IDXS), h, w))
        climate_arr.resize((write_idx, t, len(CLIMATE_BANDS), h, w))
        s2_mask.resize((write_idx, t, h, w))
        s1_mask.resize((write_idx, t, h, w))
        climate_mask.resize((write_idx, t, h, w))
        patch_xy.resize((write_idx, 2))

    # CropHarvest arrays are monthly/regularized; we use a fixed synthetic year for DOY encoding.
    months = [f"2000-{m:02d}-15" for m in range(1, t + 1)]
    _create_array(
        root,
        "time",
        shape=(t,),
        dtype="S16",
        chunks=(t,),
        compressor=compressor,
    )[:] = _encode_text(months, width=16)
    _create_array(
        root,
        "s2_bands",
        shape=(len(S2_BANDS),),
        dtype="S32",
        chunks=(len(S2_BANDS),),
        compressor=compressor,
    )[:] = _encode_text(S2_BANDS, width=32)
    _create_array(
        root,
        "s1_bands",
        shape=(len(S1_BANDS),),
        dtype="S32",
        chunks=(len(S1_BANDS),),
        compressor=compressor,
    )[:] = _encode_text(S1_BANDS, width=32)
    _create_array(
        root,
        "climate_bands",
        shape=(len(CLIMATE_BANDS),),
        dtype="S32",
        chunks=(len(CLIMATE_BANDS),),
        compressor=compressor,
    )[:] = _encode_text(CLIMATE_BANDS, width=32)

    root.attrs["metadata"] = json.dumps(
        {
            "source": "CropHarvest",
            "arrays_dir": str(arrays_dir),
            "n_samples": write_idx,
            "timesteps": t,
            "num_skipped": skipped,
            "notes": "Converted point-level monthly features to 1x1 patch tensors for quick SSL prototyping.",
        }
    )

    return {
        "num_patches": write_idx,
        "num_skipped": skipped,
        "num_timesteps": t,
        "s2_channels": len(S2_BANDS),
        "s1_channels": len(S1_BANDS),
        "climate_channels": len(CLIMATE_BANDS),
    }
