"""Run agriculture-aligned spatial-token JEPA bridge experiments."""

import json
import math
import multiprocessing as mp
import queue
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Subset

from src.datasets.dataset import MultimodalPatchDataset
from src.evals.evaluation import (
    CONDITIONS,
    extract_embeddings,
    extract_raw_stats,
    load_labels,
    make_strict_holdout_splits,
    run_probes,
    valid_cropharvest_files,
)
from src.utils.io_utils import load_env_file, summarize_rows, write_csv
from src.core.jepa import cosine_ema_momentum
from src.core.spatial_jepa import PooledTemporalJepaModel, SpatialTokenJepaModel, spatial_jepa_loss
from src.datasets.ssl4eo import (
    AGRO_STRATUM_FRACTIONS,
    BUILD_SCHEMA_VERSION,
    CLOUD_PARITY_NUM_SAMPLES,
    EXCLUDED_REGIONS,
    MIN_TOKEN_CLEAR_FRACTION,
    NORMALIZATION_CONTRACT,
    POOL_AGRO,
    POOL_GENERIC,
    S2_BANDS,
    S1_BANDS,
    SAMPLE_ID_WIDTH,
    SSL4EO_WDS_REVISION,
    TOKEN_PATCH_SIZE,
    _joint_composite_bins,
    build_ssl4eo_v11_fixed_pools,
    probe_aligned_cloud_parity,
)

PREPROCESS_START_SHARD = 1
PREPROCESS_PATCH_SIZE = 16
PREPROCESS_MAX_SAMPLES = 49152
PREPROCESS_CLEAR_INCOMPLETE = True
PREPROCESS_MAX_CACHE_GIB = 60.0
DEFAULT_ARMS = ["A_pool_generic_fixed", "B_pool_agro_fixed", "C_spatial_temporal", "D_spatiotemporal"]
DEFAULT_HOLDOUTS = ["rwanda-ceo", "togo", "togo-eval", "ethiopia", "lem-brazil"]
DEFAULT_CONDITIONS = ["clean", "sensor_off_s2", "temporal_drop_50", "s2_off_tdrop50"]
SELECTED_ARMS: list[str] = []
SELECTED_HOLDOUTS: list[str] = []
SELECTED_CONDITIONS: list[str] = []
SELECTED_SEEDS = [42]
EPOCHS = 12
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
NUM_WORKERS = 4
MODEL_DIM = 384
NUM_LAYERS = 4
NUM_HEADS = 8
PREDICTOR_DIM = 192
PREDICTOR_LAYERS = 2
TARGET_MASK_FRACTION = 0.50
MIN_SPATIAL_TEMPORAL_BLOCKS = 2
LR = 3e-5
MIN_LR = 5e-6
GLOBAL_LOSS_WEIGHT = 0.25
DEVICE = "cuda"
_MP_CONTEXT = "spawn"
GPU_ASSIGNMENTS: dict[int, list[str]] = {
    0: ["A_pool_generic_fixed", "D_spatiotemporal"],
    1: ["B_pool_agro_fixed", "C_spatial_temporal"],
}
LIMIT_TRAIN_SAMPLES = 0
LIMIT_PROBE_TRAIN_SAMPLES = 0
LIMIT_TEST_SAMPLES = 0
DRY_RUN = False
SMOKE_MODEL = False
METRICS = ["f1", "auc", "balanced_accuracy", "calibrated_f1", "calibrated_balanced_accuracy"]
S2_CHANNELS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
S1_CHANNELS = ["VV", "VH"]
CROPHARVEST_S2_REFLECTANCE_DIVISOR = 10000.0
MAX_DISTRIBUTION_SCALE_RATIO = 25.0


@dataclass(frozen=True)
class ArmSpec:
    name: str
    pretrain_zarr: Path
    spatial_tokens: bool
    spatial_temporal_masking: bool
    climate: bool = False


@dataclass
class RunConfig:
    arm: str
    output_dir: Path
    seed: int
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    num_workers: int = NUM_WORKERS
    lr: float = LR
    min_lr: float = MIN_LR
    device: str = DEVICE
    limit_train_samples: int = LIMIT_TRAIN_SAMPLES
    limit_probe_train_samples: int = LIMIT_PROBE_TRAIN_SAMPLES
    limit_test_samples: int = LIMIT_TEST_SAMPLES


ARM_SPECS = {
    "A_pool_generic_fixed": ArmSpec("A_pool_generic_fixed", "data/processed/ssl4eo_s12_v11_generic_fixed_48k.zarr", False, False),
    "B_pool_agro_fixed": ArmSpec("B_pool_agro_fixed", "data/processed/ssl4eo_s12_v11_agro_fixed_48k.zarr", False, False),
    "C_spatial_temporal": ArmSpec("C_spatial_temporal", "data/processed/ssl4eo_s12_v11_agro_fixed_48k.zarr", True, False),
    "D_spatiotemporal": ArmSpec("D_spatiotemporal", "data/processed/ssl4eo_s12_v11_agro_fixed_48k.zarr", True, True),
    "E_spatiotemporal_climate": ArmSpec("E_spatiotemporal_climate", "data/processed/ssl4eo_s12_v11_agro_fixed_48k.zarr", True, True, True),
}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8").rstrip("\x00") if isinstance(value, bytes) else str(value) for value in values]


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _excluded_region(lat: float, lon: float) -> str | None:
    for name, (min_lon, max_lon, min_lat, max_lat) in EXCLUDED_REGIONS.items():
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return name
    return None


def _clear_incomplete_downloads(cache_dir: Path) -> int:
    removed = 0
    if not cache_dir.exists():
        return removed
    for p in cache_dir.rglob("*.incomplete"):
        p.unlink(missing_ok=True)
        removed += 1
    return removed


def _sampled_band_distributions(path: Path, s2_reflectance_divisor: float | None = None) -> dict[str, Any]:
    root = zarr.open_group(str(path), mode="r")
    output: dict[str, Any] = {}
    for array_name, band_name in [("s2", "s2_bands"), ("s1", "s1_bands")]:
        array = root[array_name]
        row_count = min(1024, int(array.shape[0]))
        indices = np.linspace(0, int(array.shape[0]) - 1, row_count, dtype=np.int64)
        values = np.asarray(array.get_orthogonal_selection((indices, slice(None), slice(None), slice(None), slice(None))))
        bands = _decode(np.asarray(root[band_name][:]))
        if array_name == "s2" and s2_reflectance_divisor is not None:
            values = values.copy()
            reflectance = [index for index, band in enumerate(bands) if band != "NDVI"]
            values[:, :, reflectance] /= s2_reflectance_divisor
        output[array_name] = {
            band: _numeric_summary(values[:, :, band_idx].reshape(-1))
            for band_idx, band in enumerate(bands)
        }
    return output


def _validate_store(path: Path, pool: str) -> dict[str, Any]:
    root = zarr.open_group(str(path), mode="r")
    dataset = MultimodalPatchDataset(path)
    if len(dataset) != PREPROCESS_MAX_SAMPLES:
        raise ValueError(f"{path} has {len(dataset)} rows; expected {PREPROCESS_MAX_SAMPLES}")
    required = [
        "s2", "s1", "s2_mask", "s1_mask", "dem", "lulc",
        "s2_time_ns", "s1_time_ns", "center_lat", "center_lon",
        "elevation", "sample", "sampling_stratum", "lulc_cropland_fraction",
        "s2_clear_fraction", "s2_token_available_fraction", "source_shard",
    ]
    missing = [key for key in required if key not in root]
    if missing:
        raise ValueError(f"{path} is missing arrays: {missing}")
    if "climate" in root or "climate_mask" in root:
        raise ValueError(f"{path} must omit climate until an actual climate join is implemented")
    if np.dtype(root["sample"].dtype) != np.dtype(f"S{SAMPLE_ID_WIDTH}"):
        raise ValueError(f"{path} sample IDs use {root['sample'].dtype}; expected S{SAMPLE_ID_WIDTH}")
    if int(root.attrs.get("build_schema_version", -1)) != BUILD_SCHEMA_VERSION:
        raise ValueError(f"{path} has stale build schema {root.attrs.get('build_schema_version')}")
    if str(root.attrs.get("source_revision", "")) != SSL4EO_WDS_REVISION:
        raise ValueError(f"{path} has source revision {root.attrs.get('source_revision')}; expected {SSL4EO_WDS_REVISION}")
    if root.attrs.get("build_complete") is not True:
        raise ValueError(f"{path} is incomplete")
    contract = json.loads(str(root.attrs.get("normalization_contract", "{}")))
    if contract != NORMALIZATION_CONTRACT:
        raise ValueError(f"{path} normalization contract does not match the current builder")
    if int(root.attrs.get("token_patch_size", -1)) != TOKEN_PATCH_SIZE:
        raise ValueError(f"{path} token_patch_size mismatch: expected {TOKEN_PATCH_SIZE}, got {root.attrs.get('token_patch_size')}")
    if abs(float(root.attrs.get("min_token_clear_fraction", -1.0)) - MIN_TOKEN_CLEAR_FRACTION) > 1e-6:
        raise ValueError(f"{path} min_token_clear_fraction mismatch: expected {MIN_TOKEN_CLEAR_FRACTION}, got {root.attrs.get('min_token_clear_fraction')}")
    if dataset.shapes.patch_h != PREPROCESS_PATCH_SIZE or dataset.shapes.patch_w != PREPROCESS_PATCH_SIZE:
        raise ValueError(f"{path} has patch shape {dataset.shapes.patch_h}x{dataset.shapes.patch_w}")
    if not bool(np.all(np.asarray(root["s2_time_ns"][:]) != 0)):
        raise ValueError(f"{path} contains missing S2 timestamps")
    if not bool(np.all(np.asarray(root["s1_time_ns"][:]) != 0)):
        raise ValueError(f"{path} contains missing S1 timestamps")
    if not bool(np.isfinite(np.asarray(root["elevation"][:])).all()):
        raise ValueError(f"{path} contains non-finite elevation")
    clear_frac = np.asarray(root["s2_clear_fraction"][:])
    if not bool(np.isfinite(clear_frac).all()) or not bool((clear_frac >= 0.0).all()) or not bool((clear_frac <= 1.0).all()):
        raise ValueError(f"{path} contains out-of-range s2_clear_fraction values")
    token_frac = np.asarray(root["s2_token_available_fraction"][:])
    if not bool(np.isfinite(token_frac).all()) or not bool((token_frac >= 0.0).all()) or not bool((token_frac <= 1.0).all()):
        raise ValueError(f"{path} contains out-of-range s2_token_available_fraction values")
    lat = np.asarray(root["center_lat"][:])
    lon = np.asarray(root["center_lon"][:])
    leaked = Counter(
        region
        for sample_lat, sample_lon in zip(lat, lon)
        if (region := _excluded_region(float(sample_lat), float(sample_lon))) is not None
    )
    if leaked:
        raise ValueError(f"{path} contains geographically excluded samples: {dict(leaked)}")
    strata = Counter(_decode(np.asarray(root["sampling_stratum"][:])))
    sample_ids = _decode(np.asarray(root["sample"][:]))
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError(f"{path} contains empty sample IDs")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path} contains duplicate sample IDs")
    if pool == POOL_GENERIC:
        if strata != {"generic": PREPROCESS_MAX_SAMPLES}:
            raise ValueError(f"{path} generic strata mismatch: {dict(strata)}")
    elif pool == POOL_AGRO:
        expected = {name: int(round(PREPROCESS_MAX_SAMPLES * fraction)) for name, fraction in AGRO_STRATUM_FRACTIONS.items()}
        expected["cropland_dominant"] += PREPROCESS_MAX_SAMPLES - sum(expected.values())
        if strata != Counter(expected):
            raise ValueError(f"{path} agriculture strata mismatch: expected={expected}, got={dict(strata)}")
    else:
        raise ValueError(f"Unknown pool={pool}")
    sample = dataset[0]
    for name in ["s2", "s1", "s2_mask", "s1_mask", "doy", "s2_doy", "s1_doy", "s2_elapsed_days", "s1_elapsed_days"]:
        if not bool(np.isfinite(sample[name].numpy()).all()):
            raise ValueError(f"{path} sample contains non-finite values in {name}")
    return {
        "path": str(path),
        "pool": pool,
        "build_complete": True,
        "build_schema_version": BUILD_SCHEMA_VERSION,
        "source_revision": SSL4EO_WDS_REVISION,
        "num_samples": len(dataset),
        "patch_size": PREPROCESS_PATCH_SIZE,
        "s2_shape": list(root["s2"].shape),
        "s1_shape": list(root["s1"].shape),
        "dem_shape": list(root["dem"].shape),
        "lulc_shape": list(root["lulc"].shape),
        "sampling_strata": dict(sorted(strata.items())),
        "unique_sample_ids": len(set(sample_ids)),
        "excluded_regions": dict(sorted(leaked.items())),
        "climate": "omitted",
        "timestamps": "per_sample_per_modality",
        "lulc_cropland_fraction": _numeric_summary(np.asarray(root["lulc_cropland_fraction"][:])),
        "s2_clear_fraction": _numeric_summary(np.asarray(root["s2_clear_fraction"][:])),
        "s2_token_available_fraction": _numeric_summary(np.asarray(root["s2_token_available_fraction"][:])),
        "source_shards": dict(sorted(Counter(int(value) for value in np.asarray(root["source_shard"][:])).items())),
        "geography_bins_10deg": dict(
            sorted(
                Counter(
                    f"{int(np.floor(float(sample_lat) / 10.0) * 10):+03d}:"
                    f"{int(np.floor(float(sample_lon) / 10.0) * 10):+04d}"
                    for sample_lat, sample_lon in zip(lat, lon)
                ).items()
            )
        ),
        "geography_composite_bins": _joint_composite_bins(
            lat, lon,
            np.asarray(root["s2_clear_fraction"][:]),
            np.asarray(root["s2_token_available_fraction"][:]),
        ),
        "normalization_contract": contract,
        "sampled_band_distributions": _sampled_band_distributions(path),
    }


def _ensure_data() -> None:
    Path("data/cache/ssl4eo_v11_wds_48k").mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    if PREPROCESS_CLEAR_INCOMPLETE:
        summary["incomplete_downloads_removed"] = _clear_incomplete_downloads(Path("data/cache/ssl4eo_v11_wds_48k"))
    summary["cloud_parity_probe"] = probe_aligned_cloud_parity(
        PREPROCESS_START_SHARD, Path("data/cache/ssl4eo_v11_wds_48k"), patch_size=PREPROCESS_PATCH_SIZE,
    )
    summary["build"] = build_ssl4eo_v11_fixed_pools(
        generic_output_zarr="data/processed/ssl4eo_s12_v11_generic_fixed_48k.zarr",
        agro_output_zarr="data/processed/ssl4eo_s12_v11_agro_fixed_48k.zarr",
        cache_dir=Path("data/cache/ssl4eo_v11_wds_48k"),
        max_samples=PREPROCESS_MAX_SAMPLES,
        patch_size=PREPROCESS_PATCH_SIZE,
        start_shard=PREPROCESS_START_SHARD,
        max_cache_gib=PREPROCESS_MAX_CACHE_GIB,
        evict_cached_shards=True,
    )
    summary["generic_validation"] = _validate_store("data/processed/ssl4eo_s12_v11_generic_fixed_48k.zarr", POOL_GENERIC)
    summary["agro_validation"] = _validate_store("data/processed/ssl4eo_s12_v11_agro_fixed_48k.zarr", POOL_AGRO)
    generic_ids = set(_decode(np.asarray(zarr.open_group(str("data/processed/ssl4eo_s12_v11_generic_fixed_48k.zarr"), mode="r")["sample"][:])))
    agro_ids = set(_decode(np.asarray(zarr.open_group(str("data/processed/ssl4eo_s12_v11_agro_fixed_48k.zarr"), mode="r")["sample"][:])))
    summary["generic_agro_overlap"] = len(generic_ids & agro_ids)
    if summary["generic_agro_overlap"]:
        raise ValueError(f"Generic and agriculture pools overlap by {summary['generic_agro_overlap']} samples")
    summary["geography_histograms_match"] = (
        summary["generic_validation"]["geography_bins_10deg"] == summary["agro_validation"]["geography_bins_10deg"]
    )
    if not summary["geography_histograms_match"]:
        raise ValueError("Generic and agriculture pools do not have matching geographic histograms")
    summary["geography_composite_bins_match"] = (
        summary["generic_validation"]["geography_composite_bins"]
        == summary["agro_validation"]["geography_composite_bins"]
    )
    if not summary["geography_composite_bins_match"]:
        raise ValueError("Generic and agriculture pools do not have matching composite matching histograms")
    if "data/cropharvest/processed/v2.zarr".exists():
        summary["cropharvest_model_input_contract"] = {
            "s2": f"CropHarvest reflectance bands divided by {CROPHARVEST_S2_REFLECTANCE_DIVISOR:g} at load time; NDVI unchanged",
            "s1": "CropHarvest S1 values preserved unchanged",
        }
        summary["cropharvest_model_input_sampled_band_distributions"] = _sampled_band_distributions(
            "data/cropharvest/processed/v2.zarr", s2_reflectance_divisor=CROPHARVEST_S2_REFLECTANCE_DIVISOR,
        )
    Path("data/processed/preprocess_8_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Wrote data/processed/preprocess_8_summary.json")


def _assert_store_contract(dataset: MultimodalPatchDataset, label: str) -> None:
    root = dataset.root
    if _decode(np.asarray(root["s2_bands"][:])) != S2_CHANNELS:
        raise ValueError(f"{label} has an incompatible S2 channel contract")
    if _decode(np.asarray(root["s1_bands"][:])) != S1_CHANNELS:
        raise ValueError(f"{label} has an incompatible S1 channel contract")


def _assert_pool_matching(generic: dict[str, Any], agro: dict[str, Any]) -> None:
    if "geography_bins_10deg" not in generic or "geography_bins_10deg" not in agro:
        raise ValueError("Generic and agriculture stores are missing geography_bins_10deg keys")
    if "geography_composite_bins" not in generic or "geography_composite_bins" not in agro:
        raise ValueError("Generic and agriculture stores are missing geography_composite_bins keys")
    if generic["geography_bins_10deg"] != agro["geography_bins_10deg"]:
        raise ValueError("Generic and agriculture stores must have matching geographic histograms")
    if generic["geography_composite_bins"] != agro["geography_composite_bins"]:
        raise ValueError("Generic and agriculture stores must have matching composite matching histograms")


def _mean_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    return summarize_rows(rows, keys)


def _assert_preprocess_qa() -> dict[str, Any]:
    if not Path("data/processed/preprocess_8_summary.json").exists():
        raise FileNotFoundError("Run [8] to build the data first; missing data/processed/preprocess_8_summary.json")
    summary = json.loads(Path("data/processed/preprocess_8_summary.json").read_text())
    state = summary.get("build", {}).get("state", {})
    if state.get("status") != "complete" or state.get("phase") != "complete":
        raise ValueError("Preprocessing build state is incomplete")
    for label in ["generic_validation", "agro_validation"]:
        validation = summary.get(label, {})
        if validation.get("build_complete") is not True:
            raise ValueError(f"{label} is incomplete")
        if int(validation.get("build_schema_version", -1)) != BUILD_SCHEMA_VERSION:
            raise ValueError(f"{label} has a stale build schema")
        if validation.get("source_revision") != SSL4EO_WDS_REVISION:
            raise ValueError(f"{label} has an unexpected source revision")
        if int(validation.get("unique_sample_ids", -1)) != len(MultimodalPatchDataset(validation["path"])):
            raise ValueError(f"{label} failed sample-ID uniqueness QA")
        if "sampled_band_distributions" not in validation:
            raise ValueError(f"{label} is missing sampled band-distribution QA")
    parity = summary.get("cloud_parity_probe", {})
    if not isinstance(parity, dict):
        raise ValueError("Preprocessing QA is missing a cloud_parity_probe record")
    if int(parity.get("shard", -1)) != PREPROCESS_START_SHARD:
        raise ValueError(
            f"Cloud parity probe shard {parity.get('shard')} does not match START_SHARD={PREPROCESS_START_SHARD}"
        )
    if int(parity.get("samples_checked", 0)) != CLOUD_PARITY_NUM_SAMPLES:
        raise ValueError(
            f"Cloud parity probe checked {parity.get('samples_checked')} samples; expected {CLOUD_PARITY_NUM_SAMPLES}"
        )
    if int(parity.get("requested_samples", 0)) != CLOUD_PARITY_NUM_SAMPLES:
        raise ValueError(
            f"Cloud parity probe requested {parity.get('requested_samples')} samples; "
            f"expected {CLOUD_PARITY_NUM_SAMPLES}"
        )
    if int(parity.get("total_pixel_diff", -1)) != 0:
        raise ValueError(f"Cloud parity probe found {parity.get('total_pixel_diff')} mismatched pixels; expected 0")
    if int(parity.get("patch_size", -1)) != PREPROCESS_PATCH_SIZE:
        raise ValueError(
            f"Cloud parity probe patch_size {parity.get('patch_size')} does not match PATCH_SIZE={PREPROCESS_PATCH_SIZE}"
        )
    if "cropharvest_model_input_sampled_band_distributions" not in summary:
        raise ValueError("Preprocessing QA is missing normalized CropHarvest model-input distributions")
    if int(summary.get("generic_agro_overlap", -1)) != 0:
        raise ValueError("Generic and agriculture stores must be disjoint")
    generic = summary["generic_validation"]
    agro = summary["agro_validation"]
    _assert_pool_matching(generic, agro)
    _assert_distributions_compatible(
        generic["sampled_band_distributions"],
        agro["sampled_band_distributions"],
        "agriculture SSL4EO",
    )
    _assert_distributions_compatible(
        generic["sampled_band_distributions"],
        summary["cropharvest_model_input_sampled_band_distributions"],
        "CropHarvest model input",
    )
    return summary


def _assert_distributions_compatible(reference: dict[str, Any], candidate: dict[str, Any], label: str) -> None:
    for modality in ["s2", "s1"]:
        if set(reference.get(modality, {})) != set(candidate.get(modality, {})):
            raise ValueError(f"{label} has an incompatible {modality} band set")
        for band, reference_stats in reference[modality].items():
            candidate_stats = candidate[modality][band]
            values = np.asarray(
                [
                    reference_stats["min"],
                    reference_stats["max"],
                    reference_stats["std"],
                    reference_stats["p10"],
                    reference_stats["p90"],
                    candidate_stats["min"],
                    candidate_stats["max"],
                    candidate_stats["std"],
                    candidate_stats["p10"],
                    candidate_stats["p90"],
                ],
                dtype=np.float64,
            )
            if not bool(np.isfinite(values).all()):
                raise ValueError(f"{label} has non-finite distribution QA for {modality}/{band}")
            if band == "NDVI":
                if reference_stats["min"] < -1.5 or reference_stats["max"] > 1.5:
                    raise ValueError("Reference SSL4EO NDVI distribution is outside the expected range")
                if candidate_stats["min"] < -1.5 or candidate_stats["max"] > 1.5:
                    raise ValueError(f"{label} NDVI distribution is outside the expected range")
            reference_scale = max(
                abs(float(reference_stats["p10"])),
                abs(float(reference_stats["p90"])),
                float(reference_stats["std"]),
                1e-6,
            )
            candidate_scale = max(
                abs(float(candidate_stats["p10"])),
                abs(float(candidate_stats["p90"])),
                float(candidate_stats["std"]),
                1e-6,
            )
            ratio = max(reference_scale, candidate_scale) / min(reference_scale, candidate_scale)
            if ratio > MAX_DISTRIBUTION_SCALE_RATIO:
                raise ValueError(
                    f"{label} distribution scale for {modality}/{band} differs by {ratio:.2f}x; "
                    f"limit is {MAX_DISTRIBUTION_SCALE_RATIO:.2f}x"
                )


def _limit(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def _pretrain_split(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(size * 0.10)))
    return np.sort(indices[val_size:]), np.sort(indices[:val_size])


def _condition_map(selected: list[str]) -> list[tuple[str, str, float]]:
    available = {name: (sensor, temporal_drop) for name, sensor, temporal_drop in CONDITIONS}
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    return [(name, *available[name]) for name in selected]


def _lr_lambda(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(float(step + 1) / float(warmup_steps), min_lr_ratio)
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _make_model(spec: ArmSpec, dataset: MultimodalPatchDataset) -> torch.nn.Module:
    if spec.climate:
        raise ValueError("E_spatiotemporal_climate is blocked until a validated AgERA5 join is implemented")
    if spec.spatial_tokens:
        return SpatialTokenJepaModel(
            s2_channels=dataset.shapes.s2_channels,
            s1_channels=dataset.shapes.s1_channels,
            model_dim=MODEL_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            predictor_dim=PREDICTOR_DIM,
            predictor_layers=PREDICTOR_LAYERS,
        )
    return PooledTemporalJepaModel(
        s2_channels=dataset.shapes.s2_channels,
        s1_channels=dataset.shapes.s1_channels,
        model_dim=MODEL_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        predictor_layers=PREDICTOR_LAYERS,
    )


def _temporal_target_mask(
    batch_size: int, timesteps: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    if timesteps < 2:
        raise ValueError("JEPA masking requires at least two timesteps")
    mask = torch.zeros((batch_size, timesteps), dtype=torch.bool, device=device)
    width = min(timesteps - 1, max(1, int(round(timesteps * TARGET_MASK_FRACTION))))
    for row in range(batch_size):
        start = int(torch.randint(1, max(2, timesteps - width + 1), (1,), generator=generator, device=device).item())
        mask[row, start : min(timesteps, start + width)] = True
    return mask


def _spatiotemporal_slots(timesteps: int) -> list[tuple[int, int, int, int, int]]:
    """Build non-overlapping spatiotemporal mask slots.

    Each slot is (t_start, y, x, time_width, cell_count).  time_width=2 slots
    cover non-adjacent pairs of timesteps to prevent quadrant overlap.  Any
    remaining available timesteps use time_width=1 (spatial-only) slots.
    """
    quadrants = [(0, 0), (0, 2), (2, 0), (2, 2)]
    slots: list[tuple[int, int, int, int, int]] = []
    covered = set()
    for t in range(1, timesteps - 1, 2):
        for y, x in quadrants:
            slots.append((t, y, x, 2, 8))
            covered.add(t)
            covered.add(t + 1)
    for t in range(1, timesteps):
        if t not in covered:
            for y, x in quadrants:
                slots.append((t, y, x, 1, 4))
    return slots


def _spatial_target_mask(
    batch_size: int,
    timesteps: int,
    spatial_temporal: bool,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    spatial = 16
    if not spatial_temporal:
        return _temporal_target_mask(batch_size, timesteps, device, generator)[:, :, None].expand(
            batch_size, timesteps, spatial
        )
    mask = torch.zeros((batch_size, timesteps, 4, 4), dtype=torch.bool, device=device)
    target_count = int(round(timesteps * spatial * TARGET_MASK_FRACTION))
    max_target_count = (timesteps - 1) * spatial
    if target_count > max_target_count:
        raise ValueError(f"Target mask budget {target_count} exceeds the visible-context limit {max_target_count}")
    slots = _spatiotemporal_slots(timesteps)
    if target_count % 4 != 0:
        raise ValueError(f"Target budget {target_count} is not a multiple of the minimum block size 4")
    if len(slots) < MIN_SPATIAL_TEMPORAL_BLOCKS:
        raise ValueError(
            f"Only {len(slots)} non-overlapping slots available; need at least {MIN_SPATIAL_TEMPORAL_BLOCKS}"
        )
    for row in range(batch_size):
        used = 0
        for idx in torch.randperm(len(slots), generator=generator, device=device):
            t, y, x, tw, cells = slots[idx]
            if used + cells > target_count:
                continue
            mask[row, t : t + tw, y : y + 2, x : x + 2] = True
            used += cells
            if used == target_count:
                break
        if used != target_count:
            raise ValueError(f"Row {row}: budget {target_count}, got {used}")
    return mask.flatten(start_dim=2)


def _target_mask(
    spec: ArmSpec, batch_size: int, timesteps: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    if spec.spatial_tokens:
        return _spatial_target_mask(batch_size, timesteps, spec.spatial_temporal_masking, device, generator)
    return _temporal_target_mask(batch_size, timesteps, device, generator)


def _model_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key]
        for key in [
            "s2",
            "s1",
            "climate",
            "doy",
            "s2_doy",
            "s1_doy",
            "s2_elapsed_days",
            "s1_elapsed_days",
            "s2_available",
            "s1_available",
            "climate_available",
            "s2_mask",
            "s1_mask",
        ]
        if key in batch
    }


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    spec: ArmSpec,
    cfg: RunConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    global_step: int,
    total_steps: int,
) -> tuple[float, int, float]:
    training = optimizer is not None
    model.train(training)
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + (global_step if training else 99173))
    total_loss = 0.0
    batches = 0
    mask_fraction = 0.0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        mask = _target_mask(spec, batch["s2"].shape[0], batch["s2"].shape[1], device, generator)
        mask_fraction += float(mask.float().mean().item())
        output = model(target_mask=mask, **_model_batch(batch))
        loss = spatial_jepa_loss(output, global_weight=GLOBAL_LOSS_WEIGHT)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite JEPA loss")
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            model.update_target_encoder(cosine_ema_momentum(global_step, total_steps))
            global_step += 1
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(1, batches), global_step, mask_fraction / max(1, batches)


def _train_one(cfg: RunConfig, spec: ArmSpec, dataset: MultimodalPatchDataset, device: torch.device) -> torch.nn.Module:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train_indices, val_indices = _pretrain_split(len(dataset), cfg.seed)
    train_indices = _limit(train_indices, cfg.limit_train_samples, cfg.seed + 101)
    if cfg.limit_train_samples > 0:
        val_indices = _limit(val_indices, max(1, cfg.limit_train_samples // 10), cfg.seed + 102)
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = _make_model(spec, dataset).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
    total_steps = cfg.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(step, total_steps, 2 * max(1, len(train_loader)), cfg.min_lr / cfg.lr),
    )
    best_val = float("inf")
    global_step = 0
    history = []
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_loss, global_step, train_mask_fraction = _train_epoch(
            model, train_loader, spec, cfg, device, optimizer, scheduler, global_step, total_steps
        )
        with torch.no_grad():
            val_loss, global_step, val_mask_fraction = _train_epoch(
                model, val_loader, spec, cfg, device, None, None, global_step, total_steps
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mask_fraction": train_mask_fraction,
            "val_mask_fraction": val_mask_fraction,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps({"arm": cfg.arm, **row}), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "val_loss": val_loss},
                cfg.output_dir / "best_checkpoint.pt",
            )
    torch.save(
        {"model_state_dict": model.state_dict(), "config": asdict(cfg), "epoch": cfg.epochs, "val_loss": val_loss},
        cfg.output_dir / "final_checkpoint.pt",
    )
    (cfg.output_dir / "train_history.json").write_text(
        json.dumps(
            {"best_val_loss": best_val, "train_seconds": time.time() - started, "history": history},
            indent=2,
            default=str,
        )
    )
    return cfg.output_dir / "best_checkpoint.pt", cfg.output_dir / "final_checkpoint.pt"


def _evaluate_holdout(
    cfg: RunConfig,
    model: torch.nn.Module,
    dataset: MultimodalPatchDataset,
    y: np.ndarray,
    groups: np.ndarray,
    holdout: str,
    conditions: list[tuple[str, str, float]],
    device: torch.device,
    checkpoint_role: str = "best",
) -> list[dict[str, Any]]:
    _, _, test_idx, train_idx = make_strict_holdout_splits(y, groups, holdout, cfg.seed)
    train_idx = _limit(train_idx, cfg.limit_probe_train_samples, cfg.seed + 202)
    test_idx = _limit(test_idx, cfg.limit_test_samples, cfg.seed + 303)
    rows: list[dict[str, Any]] = []
    clean_x_train = extract_embeddings(
        model, dataset, train_idx, device, cfg.eval_batch_size, cfg.num_workers, "none", 0.0, cfg.seed
    )
    clean_stats_train = extract_raw_stats(
        dataset, train_idx, cfg.eval_batch_size, cfg.num_workers, "none", 0.0, cfg.seed
    )
    base = {
        "experiment": "agriculture_aligned_spatial_jepa_bridge",
        "arm": cfg.arm,
        "holdout": holdout,
        "seed": cfg.seed,
        "checkpoint_role": checkpoint_role,
    }
    for condition, sensor_off, temporal_drop in conditions:
        x_test = extract_embeddings(
            model,
            dataset,
            test_idx,
            device,
            cfg.eval_batch_size,
            cfg.num_workers,
            sensor_off,
            temporal_drop,
            cfg.seed + 999,
        )
        stats_test = extract_raw_stats(
            dataset, test_idx, cfg.eval_batch_size, cfg.num_workers, sensor_off, temporal_drop, cfg.seed + 999
        )
        for model_name, left, right in [
            ("embedding", clean_x_train, x_test),
            ("raw_stats", clean_stats_train, stats_test),
            (
                "embedding_plus_raw_stats",
                np.concatenate([clean_x_train, clean_stats_train], axis=1),
                np.concatenate([x_test, stats_test], axis=1),
            ),
        ]:
            before = len(rows)
            run_probes(rows, model_name, left, right, y[train_idx], y[test_idx], condition, cfg.seed)
            for row in rows[before:]:
                row.update({**base, "robustness_protocol": "clean_train_degraded_test"})
        if condition == "clean":
            continue
        matched_x_train = extract_embeddings(
            model, dataset, train_idx, device, cfg.eval_batch_size, cfg.num_workers, sensor_off, temporal_drop, cfg.seed
        )
        matched_stats_train = extract_raw_stats(
            dataset, train_idx, cfg.eval_batch_size, cfg.num_workers, sensor_off, temporal_drop, cfg.seed
        )
        for model_name, left, right in [
            ("embedding", matched_x_train, x_test),
            ("raw_stats", matched_stats_train, stats_test),
            (
                "embedding_plus_raw_stats",
                np.concatenate([matched_x_train, matched_stats_train], axis=1),
                np.concatenate([x_test, stats_test], axis=1),
            ),
        ]:
            before = len(rows)
            run_probes(rows, model_name, left, right, y[train_idx], y[test_idx], condition, cfg.seed)
            for row in rows[before:]:
                row.update({**base, "robustness_protocol": "condition_matched_retrained"})
    return rows


def _smoke_model() -> None:
    device = torch.device("cpu")
    batch = {
        "s2": torch.randn(2, 4, 11, 16, 16),
        "s1": torch.randn(2, 4, 2, 16, 16),
        "climate": torch.empty(2, 4, 0, 16, 16),
        "doy": torch.tensor([[30.0, 120.0, 210.0, 300.0]]).repeat(2, 1),
        "s2_available": torch.ones(2, 4),
        "s1_available": torch.ones(2, 4),
        "climate_available": torch.zeros(2, 4),
        "s2_mask": torch.ones(2, 4, 16, 16),
        "s1_mask": torch.ones(2, 4, 16, 16),
    }
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    rows = []
    for name in DEFAULT_ARMS:
        spec = ARM_SPECS[name]
        model = _make_model(
            spec, type("Shapes", (), {"shapes": type("S", (), {"s2_channels": 11, "s1_channels": 2})()})()
        )
        mask = _target_mask(spec, 2, 4, device, generator)
        loss = spatial_jepa_loss(model(target_mask=mask, **batch), GLOBAL_LOSS_WEIGHT)
        loss.backward()
        rows.append({"arm": name, "loss": float(loss.item())})
    Path("artifacts/[8]").mkdir(parents=True, exist_ok=True)
    write_csv(Path("artifacts/[8]") / "smoke_model.csv", rows)


def _run_worker(gpu_id: int, worker_arms: list[str], queue: "mp.Queue", preprocess_qa: dict[str, Any]) -> None:
    """Run assigned arms on a specific GPU device."""
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        eval_dataset = MultimodalPatchDataset("data/cropharvest/processed/v2.zarr", s2_reflectance_divisor=CROPHARVEST_S2_REFLECTANCE_DIVISOR)
        _assert_store_contract(eval_dataset, "evaluation")
        valid_files = valid_cropharvest_files("data/cropharvest/raw/features/arrays", eval_dataset.shapes.timesteps)
        if len(valid_files) != len(eval_dataset):
            raise ValueError(f"Valid H5 count {len(valid_files)} does not match eval Zarr length {len(eval_dataset)}")
        y, groups = load_labels(valid_files, "data/cropharvest/raw/labels.geojson")
        holdouts = SELECTED_HOLDOUTS or DEFAULT_HOLDOUTS
        conditions = _condition_map(SELECTED_CONDITIONS or DEFAULT_CONDITIONS)
        all_rows: list[dict[str, Any]] = []
        for arm_name in worker_arms:
            spec = ARM_SPECS[arm_name]
            pretrain_dataset = MultimodalPatchDataset(spec.pretrain_zarr)
            _assert_store_contract(pretrain_dataset, arm_name)
            for seed in SELECTED_SEEDS:
                cfg = RunConfig(arm=arm_name, output_dir=Path("artifacts/[8]") / f"{arm_name}_seed{seed}", seed=seed)
                _set_seed(seed)
                best_ckpt, final_ckpt = _train_one(cfg, spec, pretrain_dataset, device)
                model = _make_model(spec, pretrain_dataset).to(device)
                for checkpoint_role, ckpt_path in [("best", best_ckpt), ("final", final_ckpt)]:
                    state = torch.load(ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(state["model_state_dict"])
                    model.eval()
                    run_rows = []
                    for holdout in holdouts:
                        run_rows.extend(
                            _evaluate_holdout(
                                cfg, model, eval_dataset, y, groups, holdout, conditions, device, checkpoint_role
                            )
                        )
                    all_rows.extend(run_rows)
                    write_csv(cfg.output_dir / f"probe_results_{checkpoint_role}.csv", run_rows)
                    write_csv(
                        cfg.output_dir / f"probe_summary_{checkpoint_role}.csv",
                        summarize_rows(
                            run_rows,
                            [
                                "arm",
                                "model",
                                "holdout",
                                "condition",
                                "label_budget",
                                "robustness_protocol",
                                "checkpoint_role",
                            ],
                        ),
                    )
        result_path = Path("artifacts/[8]") / f"worker_{gpu_id}_results.json"
        result_path.write_text(json.dumps(all_rows, indent=2, default=str))
        queue.put({"gpu_id": gpu_id, "success": True, "result_path": str(result_path)})
    except Exception:
        queue.put({"gpu_id": gpu_id, "success": False, "error": traceback.format_exc()})
        raise


def _validate_gpu_setup(arms: list[str]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; [8] requires at least one NVIDIA GPU")
    if torch.cuda.device_count() <= max(GPU_ASSIGNMENTS):
        raise RuntimeError(
            f"GPU count {torch.cuda.device_count()} is insufficient; "
            f"need at least {max(GPU_ASSIGNMENTS) + 1} devices for GPU_ASSIGNMENTS {dict(GPU_ASSIGNMENTS)}"
        )
    assigned = [a for worker_list in GPU_ASSIGNMENTS.values() for a in worker_list]
    counts = Counter(assigned)
    dupes = sorted(arm for arm, count in counts.items() if count > 1)
    if dupes:
        raise ValueError(f"Duplicate arm assignments in GPU_ASSIGNMENTS: {dupes}")
    missing = sorted(set(arms) - set(assigned))
    extra = sorted(set(assigned) - set(arms))
    if missing:
        raise ValueError(f"GPU_ASSIGNMENTS is missing active arms: {missing}")
    if extra:
        raise ValueError(f"GPU_ASSIGNMENTS contains extra arms not in active set: {extra}")


def _run_workers(arms: list[str], preprocess_qa: dict[str, Any]) -> dict[int, Path]:
    ctx = mp.get_context(_MP_CONTEXT)
    q: mp.Queue = ctx.Queue()
    gpu_workers: dict[int, mp.Process] = {}
    gpu_active: dict[int, list[str]] = {}
    for gpu_id, worker_arms in GPU_ASSIGNMENTS.items():
        active = [a for a in worker_arms if a in arms]
        if not active:
            continue
        p = ctx.Process(target=_run_worker, args=(gpu_id, active, q, preprocess_qa))
        p.start()
        gpu_workers[gpu_id] = p
        gpu_active[gpu_id] = active
    if not gpu_workers:
        raise ValueError("No GPU workers launched — check GPU_ASSIGNMENTS vs selected arms")
    results: dict[int, Path] = {}
    finished: set[int] = set()

    def _kill_all(keep_alive: int | None = None) -> None:
        for oid, op in gpu_workers.items():
            if oid == keep_alive:
                continue
            op.terminate()
            op.join(10)
            if op.is_alive():
                op.kill()
                op.join(5)

    def _abort(failed_gpu: int | None = None, msg: dict | None = None) -> RuntimeError:
        _kill_all(keep_alive=failed_gpu)
        if failed_gpu is not None:
            return RuntimeError(
                f"GPU worker {failed_gpu} exited with code {gpu_workers[failed_gpu].exitcode} without reporting success"
            )
        if msg is not None:
            return RuntimeError(f"GPU worker {msg['gpu_id']} failed:\n{msg['error']}")
        return RuntimeError("unreachable")

    while len(finished) < len(gpu_workers):
        try:
            msg = q.get(timeout=30)
        except queue.Empty:
            dead = [g for g in gpu_workers if g not in finished and not gpu_workers[g].is_alive()]
            if dead:
                primary = dead[0]
                gpu_workers[primary].join(5)
                raise _abort(failed_gpu=primary)
            continue
        if not msg["success"]:
            raise _abort(msg=msg)
        finished.add(msg["gpu_id"])
        results[msg["gpu_id"]] = Path(msg["result_path"])

    errors: list[str] = []
    for gpu_id, p in gpu_workers.items():
        p.join(30)
        if p.is_alive():
            p.terminate()
            p.join(10)
            if p.is_alive():
                p.kill()
                p.join(5)
            errors.append(f"GPU worker {gpu_id} (PID {p.pid}) did not terminate after reporting success")
        elif p.exitcode != 0:
            errors.append(f"GPU worker {gpu_id} (PID {p.pid}) exited with code {p.exitcode} after reporting success")
    if errors:
        raise RuntimeError("\n".join(errors))

    return results


def main() -> None:
    load_env_file(Path(".env"))
    if not "data/processed/ssl4eo_s12_v11_generic_fixed_48k.zarr".exists() or not "data/processed/ssl4eo_s12_v11_agro_fixed_48k.zarr".exists():
        _ensure_data()
    if SMOKE_MODEL:
        _smoke_model()
        return
    arms = SELECTED_ARMS or DEFAULT_ARMS
    unknown = sorted(set(arms) - set(ARM_SPECS))
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}")
    if "E_spatiotemporal_climate" in arms:
        raise ValueError("E_spatiotemporal_climate remains disabled until the climate join is implemented")
    holdouts = SELECTED_HOLDOUTS or DEFAULT_HOLDOUTS
    conditions = _condition_map(SELECTED_CONDITIONS or DEFAULT_CONDITIONS)
    Path("artifacts/[8]").mkdir(parents=True, exist_ok=True)
    manifest = [
        {"arm": arm, "seed": seed, "run_dir": str(Path("artifacts/[8]") / f"{arm}_seed{seed}")}
        for arm in arms
        for seed in SELECTED_SEEDS
    ]
    write_csv(Path("artifacts/[8]") / "run_manifest.csv", manifest)
    if DRY_RUN:
        return
    _validate_gpu_setup(arms)
    preprocess_qa = _assert_preprocess_qa()
    worker_results = _run_workers(arms, preprocess_qa)
    all_rows: list[dict[str, Any]] = []
    for result_path in worker_results.values():
        all_rows.extend(json.loads(result_path.read_text()))
    final_rows = [r for r in all_rows if r.get("checkpoint_role") == "final"]
    write_csv(Path("artifacts/[8]") / "probe_results.csv", final_rows)
    write_csv(
        Path("artifacts/[8]") / "probe_summary.csv",
        summarize_rows(
            final_rows,
            ["arm", "model", "holdout", "condition", "label_budget", "robustness_protocol", "checkpoint_role"],
        ),
    )
    best_diag = [r for r in all_rows if r.get("checkpoint_role") == "best"]
    write_csv(
        Path("artifacts/[8]") / "probe_summary_best_checkpoint.csv",
        summarize_rows(
            best_diag,
            ["arm", "model", "holdout", "condition", "label_budget", "robustness_protocol", "checkpoint_role"],
        ),
    )
    priority = [
        r
        for r in final_rows
        if r["condition"] == "clean"
        and float(r["label_budget"]) == 1.0
        and r["robustness_protocol"] == "clean_train_degraded_test"
    ]
    write_csv(Path("artifacts/[8]") / "priority_summary.csv", summarize_rows(priority, ["arm", "model", "checkpoint_role"]))
    write_csv(
        Path("artifacts/[8]") / "per_holdout_priority_summary.csv",
        summarize_rows(priority, ["arm", "model", "holdout", "checkpoint_role"]),
    )
    (Path("artifacts/[8]") / "metadata.json").write_text(
        json.dumps(
            {
                "experiment": "[8] Agriculture-Aligned Spatial-Token JEPA Bridge",
                "arms": arms,
                "holdouts": holdouts,
                "seeds": SELECTED_SEEDS,
                "conditions": [name for name, _, _ in conditions],
                "num_runs": len(manifest),
                "num_rows": len(all_rows),
                "target_mask_fraction": TARGET_MASK_FRACTION,
                "minimum_spatial_temporal_blocks": MIN_SPATIAL_TEMPORAL_BLOCKS,
                "robustness_protocols": ["clean_train_degraded_test", "condition_matched_retrained"],
                "cropharvest_s2_reflectance_divisor": CROPHARVEST_S2_REFLECTANCE_DIVISOR,
                "preprocess_summary": str(Path("data/processed/preprocess_8_summary.json")),
                "generic_agro_overlap": preprocess_qa["generic_agro_overlap"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
