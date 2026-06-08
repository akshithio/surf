"""Convert EuroCropsML NPZ files into the SURF multimodal Zarr layout."""

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr
from numcodecs import Blosc

from src.datasets.cropharvest import _create_array, _encode_text, _open_output_group

EUROCROPS_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]

# EuroCropsML 13-band ordering: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, SCL
# We keep indices 1..8 (B2-B8A), 10 (B11), 11 (B12) and compute NDVI from B8 (idx 7) and B4 (idx 3).
EUROCROPS_S2_IDXS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
EUROCROPS_B8_IDX = 7
EUROCROPS_B4_IDX = 3

COUNTRY_PREFIX = {"EE": "Estonia", "LV": "Latvia", "PT": "Portugal"}


def _hcat_from_filename(stem: str) -> str:
    return stem.split("_")[-1]


def _country_from_filename(stem: str) -> str:
    prefix = stem[:2]
    return COUNTRY_PREFIX.get(prefix, prefix)


def _iter_npz_files(preprocess_dir: Path, max_samples: int | None = None) -> Iterable[Path]:
    for file_path in sorted(preprocess_dir.glob("*.npz")):
        if file_path.is_file():
            yield file_path


def _scan_max_timesteps(
    files: list[Path],
) -> tuple[int, list[int]]:
    """Scan all NPZ files for their timestep counts and return (max, all_counts).
    
    Validates NPZ contract (keys exist, shapes match) on every file.
    Raises ValueError with the offending filename on the first malformed file.
    """
    counts: list[int] = []
    for f in files:
        with np.load(str(f)) as data:
            if "data" not in data or "dates" not in data:
                raise ValueError(f"{f.name}: missing 'data' or 'dates' key")
            raw = data["data"]
            dates = data["dates"]
        if raw.ndim != 2:
            raise ValueError(f"{f.name}: data.ndim={raw.ndim}, expected 2")
        if raw.shape[1] < 13:
            raise ValueError(f"{f.name}: data.shape[1]={raw.shape[1]}, expected >= 13 (13 EuroCropsML bands)")
        n_t = len(dates)
        if n_t != raw.shape[0]:
            raise ValueError(f"{f.name}: len(dates)={n_t} != data.shape[0]={raw.shape[0]}")
        if n_t < 1:
            raise ValueError(f"{f.name}: zero timesteps")
        counts.append(n_t)
    return max(counts), counts


def build_eurocropsml_zarr(
    preprocess_dir: str | Path = "data/eurocropsml/preprocess",
    output_zarr: str | Path = "data/eurocropsml/processed/v1.zarr",
    output_labels: str | Path = "data/eurocropsml/processed/labels.csv",
    output_summary: str | Path = "data/eurocropsml/processed/summary.json",
    fixed_timesteps: int = 96,
    max_samples: int | None = None,
    chunk_size: int = 1024,
) -> None:
    preprocess_dir = Path(preprocess_dir).expanduser().resolve()
    output_zarr = Path(output_zarr).expanduser().resolve()
    output_labels = Path(output_labels).expanduser().resolve()
    output_summary = Path(output_summary).expanduser().resolve()

    if not preprocess_dir.exists():
        raise FileNotFoundError(f"EuroCropsML preprocess directory not found: {preprocess_dir}")

    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    output_labels.parent.mkdir(parents=True, exist_ok=True)

    files = list(_iter_npz_files(preprocess_dir))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {preprocess_dir}")
    n_total = len(files)
    if max_samples is not None and max_samples > 0:
        files = files[:max_samples]
    n = len(files)
    h = 1
    w = 1
    n_s2 = len(EUROCROPS_S2_IDXS) + 1

    print(f"Scanning {n} NPZ files for timestep distribution ...")
    max_t, all_counts = _scan_max_timesteps(files)
    all_counts_sorted = sorted(all_counts)
    p50 = all_counts_sorted[n // 2] if n > 0 else 0
    p99 = all_counts_sorted[int(n * 0.99)] if n > 0 else 0
    print(f"  Timesteps: min={all_counts_sorted[0] if n else 0}, max={max_t}, "
          f"p50={p50}, p99={p99}, mean={np.mean(all_counts):.1f}")

    t = min(fixed_timesteps, max_t)
    n_downsampled = sum(1 for c in all_counts if c > t) if t < max_t else 0

    if n_downsampled:
        print(f"  Capping at t={t} (p99={p99}, max={max_t}); "
              f"{n_downsampled}/{n} samples ({100 * n_downsampled / max(1, n):.2f}%) "
              f"will be deterministically downsampled via span-preserving linspace")
    elif t == fixed_timesteps:
        print(f"  All {n} samples fit within t={t} timesteps")
    else:
        print(f"  Using fixed_timesteps={t} (dataset max={max_t})")

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    root = _open_output_group(output_zarr)

    s2_arr = _create_array(
        root,
        "s2",
        shape=(n, t, n_s2, h, w),
        chunks=(min(chunk_size, n), t, n_s2, h, w),
        dtype="float32",
        compressor=compressor,
    )
    s2_arr.attrs["_ARRAY_DIMENSIONS"] = ["sample", "time", "band", "y", "x"]

    s1_arr = _create_array(
        root,
        "s1",
        shape=(n, t, 2, h, w),
        chunks=(min(chunk_size, n), t, 2, h, w),
        dtype="float32",
        compressor=compressor,
        fill_value=0.0,
    )
    s1_arr.attrs["_ARRAY_DIMENSIONS"] = ["sample", "time", "band", "y", "x"]

    climate_arr = _create_array(
        root,
        "climate",
        shape=(n, t, 0, h, w),
        chunks=(min(chunk_size, n), t, 1, h, w),
        dtype="float32",
        compressor=compressor,
    )
    climate_arr.attrs["_ARRAY_DIMENSIONS"] = ["sample", "time", "band", "y", "x"]

    s2_mask_arr = _create_array(
        root,
        "s2_mask",
        shape=(n, t, h, w),
        chunks=(min(chunk_size, n), t, h, w),
        dtype="float32",
        compressor=compressor,
        fill_value=1.0,
    )
    s2_mask_arr.attrs["_ARRAY_DIMENSIONS"] = ["sample", "time", "y", "x"]

    s1_mask_arr = _create_array(
        root,
        "s1_mask",
        shape=(n, t, h, w),
        chunks=(min(chunk_size, n), t, h, w),
        dtype="float32",
        compressor=compressor,
        fill_value=0.0,
    )
    s1_mask_arr.attrs["_ARRAY_DIMENSIONS"] = ["sample", "time", "y", "x"]

    climate_mask_arr = _create_array(
        root,
        "climate_mask",
        shape=(n, t, h, w),
        chunks=(min(chunk_size, n), t, h, w),
        dtype="float32",
        compressor=compressor,
        fill_value=0.0,
    )
    climate_mask_arr.attrs["_ARRAY_DIMENSIONS"] = ["sample", "time", "y", "x"]

    time_arr = _create_array(
        root,
        "s2_time_ns",
        shape=(n, t),
        chunks=(min(chunk_size, n), t),
        dtype="int64",
        compressor=compressor,
    )
    time_arr.attrs["_ARRAY_DIMENSIONS"] = ["sample", "time"]

    band_names = _encode_text(EUROCROPS_S2_BANDS, 16)
    root["s2_bands"] = band_names

    s1_band_names = _encode_text(["VV", "VH"], 16)
    root["s1_bands"] = s1_band_names

    labels_rows: list[dict[str, str]] = []
    written = 0

    for idx, file_path in enumerate(files):
        with np.load(str(file_path)) as data:
            raw = data["data"].astype(np.float32)
            dates = data["dates"]
        n_t = raw.shape[0]
        stem = file_path.stem

        s2_sel = raw[:, EUROCROPS_S2_IDXS]
        b4 = raw[:, EUROCROPS_B4_IDX]
        b8 = raw[:, EUROCROPS_B8_IDX]
        ndvi = np.divide(
            b8 - b4, b8 + b4, out=np.zeros(n_t, dtype=np.float32), where=(b8 + b4) > 0,
        )
        s2 = np.concatenate([s2_sel, ndvi[:, None]], axis=1)

        if n_t > t:
            take = np.linspace(0, n_t - 1, t).round().astype(np.int64)
            s2 = s2[take]
            raw_dates = dates[take]
            s2_mask = np.ones((t, h, w), dtype=np.float32)
        else:
            padded = np.zeros((t, n_s2), dtype=np.float32)
            padded[:n_t] = s2
            s2 = padded
            s2_mask = np.zeros((t, h, w), dtype=np.float32)
            s2_mask[:n_t] = 1.0
            raw_dates = dates
        # Convert dates: fill padded slots with the last valid date
        s2_ns_full = np.zeros(t, dtype=np.int64)
        valid_ns = raw_dates.astype("datetime64[ns]").astype(np.int64)
        s2_ns_full[:len(valid_ns)] = valid_ns
        if n_t < t:
            s2_ns_full[n_t:] = int(valid_ns[-1])
        s2_ns = s2_ns_full

        s1 = np.zeros((t, 2, h, w), dtype=np.float32)
        climate = np.zeros((t, 0, h, w), dtype=np.float32)
        s1_mask = np.zeros((t, h, w), dtype=np.float32)

        s2_arr[idx] = s2.reshape(t, n_s2, h, w)
        s1_arr[idx] = s1
        climate_arr[idx] = climate
        s2_mask_arr[idx] = s2_mask
        s1_mask_arr[idx] = s1_mask
        climate_mask_arr[idx] = np.zeros((t, h, w), dtype=np.float32)
        time_arr[idx] = s2_ns

        hcat = _hcat_from_filename(stem)
        country = _country_from_filename(stem)
        labels_rows.append({"label": hcat, "country": country, "filename": stem + ".npz"})

        written += 1
        if (idx + 1) % max(1, n // 20) == 0:
            print(f"  Processed {idx + 1}/{n} samples", flush=True)

    with open(output_labels, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "country"])
        writer.writeheader()
        writer.writerows(labels_rows)

    from collections import Counter
    label_counts = dict(Counter(r["label"] for r in labels_rows).most_common())
    country_counts = dict(Counter(r["country"] for r in labels_rows).most_common())

    summary = {
        "n_samples": written,
        "n_total_available": n_total,
        "n_downsampled": int(n_downsampled),
        "downsampled_fraction": round(n_downsampled / max(1, n), 6),
        "sequence_policy": "fixed_cap_span_downsample",
        "fixed_timesteps": t,
        "max_observed_timesteps": int(max_t),
        "p99_timesteps": int(p99),
        "timesteps": {
            "max": int(max_t),
            "min": int(all_counts_sorted[0]) if n > 0 else 0,
            "p50": int(p50),
            "p99": int(p99),
            "mean": float(np.mean(all_counts).item()) if n > 0 else 0.0,
            "std": float(np.std(all_counts).item()) if n > 0 else 0.0,
        },
        "s2_bands": EUROCROPS_S2_BANDS,
        "n_s2_bands": n_s2,
        "n_s1_bands": 2,
        "n_climate_bands": 0,
        "label_counts": label_counts,
        "country_counts": country_counts,
        "stored_reflectance_divisor": None,
        "eval_reflectance_divisor": 10000.0,
        "created_from_preprocess_dir": str(preprocess_dir),
    }

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Written {written} samples to {output_zarr}")
    print(f"Labels CSV: {output_labels} ({len(labels_rows)} rows)")
    print(f"Summary: {output_summary}")
