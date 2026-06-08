"""Matched-control content-sensitive JEPA with external transfer checks."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import resource
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

for _thread_var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler, Subset

from src.core.jepa import JepaBatchMasks, cosine_ema_momentum
from src.core.spatial_jepa import PooledTemporalJepaModel, SpatialJepaOutput, SpatialTokenJepaModel
from src.datasets.dataset import MultimodalPatchDataset
from src.datasets.eurocropsml import build_eurocropsml_zarr
from src.evals.evaluation import (
    CONDITIONS,
    extract_embeddings,
    extract_raw_stats,
    load_labels,
    make_strict_holdout_splits,
    run_probes,
    valid_cropharvest_files,
)
from src.utils.io_utils import load_env_file, summarize_rows, write_csv, write_json

RUN_SCHEMA_VERSION = "v10.4"
EVAL_SCHEMA_VERSION = "eval_v10.2"
MIN_OPEN_FILES = 8192
MULTICLASS_PROBE_MAX_ITER = 1_000
MULTICLASS_PROBE_TOL = 1e-3

OUTPUT_DIR = Path("artifacts/[10]")
GENERIC_ZARR = Path("data/processed/ssl4eo_s12_v11_generic_fixed_48k.zarr")
AGRO_ZARR = Path("data/processed/ssl4eo_s12_v11_agro_fixed_48k.zarr")
CROPHARVEST_ZARR = Path("data/cropharvest/processed")
CROPHARVEST_ARRAYS = Path("data/cropharvest/raw/features/arrays")
CROPHARVEST_LABELS = Path("data/cropharvest/raw/labels.geojson")
EUROCROPS_ZARR = Path("data/eurocropsml/processed/v1.zarr")
EUROCROPS_LABELS_CSV = Path("data/eurocropsml/processed/labels.csv")
EUROCROPS_SUMMARY = Path("data/eurocropsml/processed/summary.json")

SEEDS = [42, 43]
DEFAULT_HOLDOUTS = ["rwanda-ceo", "togo", "togo-eval", "ethiopia", "lem-brazil"]
DEFAULT_CONDITIONS = [
    "clean",
    "sensor_off_s2",
    "sensor_off_s1",
    "temporal_drop_50",
    "s2_off_tdrop50",
    "s1_off_tdrop50",
]
EUROCROPS_CONDITIONS = ["clean", "temporal_drop_25", "temporal_drop_50"]
SUMMARY_KEYS = ["arm", "benchmark", "model", "holdout", "condition", "label_budget", "robustness_protocol", "checkpoint_role"]

BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
EUROCROPS_EVAL_BATCH_SIZE_POOLED = 256
EUROCROPS_EVAL_BATCH_SIZE_SPATIAL = 16
DIAGNOSTIC_BATCH_SIZE = 64
TRAIN_NUM_WORKERS = 8
EVAL_NUM_WORKERS = 4
DIAGNOSTIC_NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 4
MODEL_DIM = 384
NUM_LAYERS = 4
NUM_HEADS = 8
PREDICTOR_DIM = 192
PREDICTOR_LAYERS = 2
TARGET_MASK_FRACTION = 0.50
FIXED_UPDATES = 16_600
CHECKPOINT_STEPS = [1_000, 3_000, 10_000, 16_600]
LR = 1.5e-5
MIN_LR = 2.5e-6
GLOBAL_LOSS_WEIGHT = 0.25
CONTENT_REG_MULTIPLIER = 0.05
VARIANCE_FLOOR_WEIGHT = 1.0
COVARIANCE_WEIGHT = 0.04
VARIANCE_FLOOR = 0.04
MIXED_SIZE_MATCHED_PER_POOL = 24_576
DEVICE = "cuda"
_MP_CONTEXT = "spawn"
GPU_ASSIGNMENTS: dict[int, list[str]] = {
    0: ["A_mixed_viewdrop_reference", "C_generic_step_matched", "E_spatial_no_consistency"],
    1: ["B_mixed_viewdrop_sensor_dropout", "D_mixed_size_matched", "F_spatial_content_regularized"],
}
DRY_RUN = False

HELDOUT_SAMPLES = 2048
_GENERIC_HELDOUT_SEED = 20260605
_AGRO_HELDOUT_SEED = 20260606

S2_CHANNELS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
S1_CHANNELS = ["VV", "VH"]
CROPHARVEST_S2_REFLECTANCE_DIVISOR = 10000.0
EUROCROPSML_S2_REFLECTANCE_DIVISOR = 10000.0
VIEWDROP_REFERENCE_MODES = ["clean", "sensor_off_s2", "temporal_drop_50", "s2_off_tdrop50"]
VIEWDROP_SENSOR_DROPOUT_MODES = [
    "clean",
    "sensor_off_s2",
    "sensor_off_s1",
    "temporal_drop_50",
    "s2_off_tdrop50",
    "s1_off_tdrop50",
]


def _configure_runtime_limits() -> None:
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(max(soft, MIN_OPEN_FILES), hard)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (OSError, ValueError):
        pass
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


_configure_runtime_limits()


@dataclass(frozen=True)
class ArmSpec:
    data_source: str
    encoder: str
    viewdrop_modes: tuple[str, ...]
    content_regularized: bool
    purpose: str


ARM_SPECS: dict[str, ArmSpec] = {
    "A_mixed_viewdrop_reference": ArmSpec(
        data_source="mixed_full",
        encoder="pooled",
        viewdrop_modes=tuple(VIEWDROP_REFERENCE_MODES),
        content_regularized=False,
        purpose="Reference [9] C under fixed-update multi-seed conditions",
    ),
    "B_mixed_viewdrop_sensor_dropout": ArmSpec(
        data_source="mixed_full",
        encoder="pooled",
        viewdrop_modes=tuple(VIEWDROP_SENSOR_DROPOUT_MODES),
        content_regularized=False,
        purpose="Adds S1-off training modes to the pooled mixed-pool objective",
    ),
    "C_generic_step_matched": ArmSpec(
        data_source="generic",
        encoder="pooled",
        viewdrop_modes=tuple(VIEWDROP_SENSOR_DROPOUT_MODES),
        content_regularized=False,
        purpose="Generic-only control with the same optimizer update budget",
    ),
    "D_mixed_size_matched": ArmSpec(
        data_source="mixed_size_matched",
        encoder="pooled",
        viewdrop_modes=tuple(VIEWDROP_SENSOR_DROPOUT_MODES),
        content_regularized=False,
        purpose="48k mixed-pool control at fixed sample count and update count",
    ),
    "E_spatial_no_consistency": ArmSpec(
        data_source="mixed_full",
        encoder="spatial",
        viewdrop_modes=tuple(VIEWDROP_SENSOR_DROPOUT_MODES),
        content_regularized=False,
        purpose="Spatial-token JEPA without the failing global consistency regularizer",
    ),
    "F_spatial_content_regularized": ArmSpec(
        data_source="mixed_full",
        encoder="spatial",
        viewdrop_modes=tuple(VIEWDROP_SENSOR_DROPOUT_MODES),
        content_regularized=True,
        purpose="Spatial-token JEPA with variance/covariance anti-collapse pressure",
    ),
}
DEFAULT_ARMS = [
    "A_mixed_viewdrop_reference",
    "B_mixed_viewdrop_sensor_dropout",
    "C_generic_step_matched",
    "D_mixed_size_matched",
    "E_spatial_no_consistency",
    "F_spatial_content_regularized",
]


@dataclass
class RunConfig:
    arm: str
    seed: int
    output_dir: Path
    fixed_updates: int = FIXED_UPDATES
    checkpoint_steps: tuple[int, ...] = tuple(CHECKPOINT_STEPS)
    batch_size: int = BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    diagnostic_batch_size: int = DIAGNOSTIC_BATCH_SIZE
    train_num_workers: int = TRAIN_NUM_WORKERS
    eval_num_workers: int = EVAL_NUM_WORKERS
    diagnostic_num_workers: int = DIAGNOSTIC_NUM_WORKERS
    pin_memory: bool = PIN_MEMORY
    prefetch_factor: int = PREFETCH_FACTOR
    lr: float = LR
    min_lr: float = MIN_LR
    device: str = DEVICE


class NamedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: np.ndarray, root: Any | None = None, shapes: Any | None = None) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.root = root if root is not None else getattr(dataset, "root", None)
        self.shapes = shapes if shapes is not None else getattr(dataset, "shapes", None)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.dataset[int(self.indices[idx])]


class MixedPretrainDataset(ConcatDataset):
    def __init__(self, datasets: list[Dataset]) -> None:
        if len(datasets) < 2:
            raise ValueError("MixedPretrainDataset requires at least two datasets")
        first = datasets[0]
        first_shapes = getattr(first, "shapes")
        for other in datasets[1:]:
            if getattr(other, "shapes") != first_shapes:
                raise ValueError("Mixed pretrain datasets must have identical tensor shapes")
        super().__init__(datasets)
        self.shapes = first_shapes
        self.root = getattr(first, "root", None)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8").rstrip("\x00") if isinstance(value, bytes) else str(value) for value in values]


def _validate_ssl4eo_store(dataset: MultimodalPatchDataset, name: str) -> None:
    s2_bands = _decode(np.asarray(dataset.root["s2_bands"][:]))
    s1_bands = _decode(np.asarray(dataset.root["s1_bands"][:]))
    if s2_bands != S2_CHANNELS:
        raise ValueError(f"{name} S2 bands mismatch: {s2_bands}")
    if s1_bands != S1_CHANNELS:
        raise ValueError(f"{name} S1 bands mismatch: {s1_bands}")


def _base_datasets() -> tuple[MultimodalPatchDataset, MultimodalPatchDataset]:
    generic = MultimodalPatchDataset(GENERIC_ZARR)
    agro = MultimodalPatchDataset(AGRO_ZARR)
    _validate_ssl4eo_store(generic, "generic")
    _validate_ssl4eo_store(agro, "agriculture")
    if generic.shapes != agro.shapes:
        raise ValueError("Generic and agriculture stores must have identical tensor shapes")
    return generic, agro


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _checkpoint_chunk_dir(cfg: RunConfig, checkpoint_role: str) -> Path:
    return cfg.output_dir / "_eval_chunks" / checkpoint_role


def _checkpoint_chunk_path(cfg: RunConfig, checkpoint_role: str, chunk_name: str) -> Path:
    safe_name = chunk_name.replace("/", "_").replace(" ", "_")
    return _checkpoint_chunk_dir(cfg, checkpoint_role) / f"{safe_name}.csv"


def _checkpoint_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return summarize_rows([row for row in rows if "f1" in row], SUMMARY_KEYS)


def _write_checkpoint_outputs(cfg: RunConfig, checkpoint_role: str, rows: list[dict[str, Any]]) -> None:
    write_csv(cfg.output_dir / f"probe_results_{checkpoint_role}.csv", rows)
    write_csv(cfg.output_dir / f"probe_summary_{checkpoint_role}.csv", _checkpoint_summary(rows))


def _read_checkpoint_chunks(cfg: RunConfig, checkpoint_role: str, chunk_names: list[str]) -> list[dict[str, str]] | None:
    rows: list[dict[str, str]] = []
    for chunk_name in chunk_names:
        path = _checkpoint_chunk_path(cfg, checkpoint_role, chunk_name)
        if not path.exists():
            return None
        rows.extend(_read_csv_rows(path))
    return rows


def _write_checkpoint_chunk(cfg: RunConfig, checkpoint_role: str, chunk_name: str, rows: list[dict[str, Any]]) -> None:
    path = _checkpoint_chunk_path(cfg, checkpoint_role, chunk_name)
    write_csv(path, rows)


def _validate_eurocrops_store() -> None:
    missing = [str(path) for path in [EUROCROPS_ZARR, EUROCROPS_LABELS_CSV, EUROCROPS_SUMMARY] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"EuroCropsML processed outputs missing: {missing}")

    import zarr

    try:
        summary = json.loads(EUROCROPS_SUMMARY.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Could not read EuroCropsML summary: {EUROCROPS_SUMMARY}") from exc

    expected_summary = {
        "sequence_policy": "fixed_cap_span_downsample",
        "fixed_timesteps": 96,
        "s2_bands": S2_CHANNELS,
        "n_s2_bands": len(S2_CHANNELS),
        "n_s1_bands": 2,
        "n_climate_bands": 0,
        "stored_reflectance_divisor": None,
        "eval_reflectance_divisor": EUROCROPSML_S2_REFLECTANCE_DIVISOR,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"EuroCropsML summary {key}={summary.get(key)!r}; expected {expected!r}")

    root = zarr.open(str(EUROCROPS_ZARR), mode="r")
    n_samples = int(summary.get("n_samples", -1))
    if int(root["s2"].shape[0]) != n_samples:
        raise ValueError(f"EuroCropsML zarr samples {root['s2'].shape[0]} != summary n_samples {n_samples}")
    if tuple(root["s2"].shape[1:]) != (96, len(S2_CHANNELS), 1, 1):
        raise ValueError(f"EuroCropsML s2 shape mismatch: {root['s2'].shape}")
    if _decode(np.asarray(root["s2_bands"][:])) != S2_CHANNELS:
        raise ValueError("EuroCropsML S2 band order mismatch")
    if _decode(np.asarray(root["s1_bands"][:])) != S1_CHANNELS:
        raise ValueError("EuroCropsML S1 band order mismatch")

    labels = _read_csv_rows(EUROCROPS_LABELS_CSV)
    if len(labels) != n_samples:
        raise ValueError(f"EuroCropsML labels rows {len(labels)} != summary n_samples {n_samples}")
    country_counts = Counter(str(row.get("country", "")) for row in labels)
    required_countries = {"Latvia", "Portugal", "Estonia"}
    missing_countries = sorted(country for country in required_countries if country_counts[country] <= 0)
    if missing_countries:
        raise ValueError(f"EuroCropsML labels missing required countries: {missing_countries}")
    if dict(country_counts) != summary.get("country_counts"):
        raise ValueError("EuroCropsML labels country counts do not match summary")


def _heldout_indices(generic: MultimodalPatchDataset, agro: MultimodalPatchDataset) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic held-out validation indices for diagnostics.
    
    These indices are excluded from every pretraining arm so that SSL diagnostics
    measure held-out generalization rather than memorisation.
    """
    rng = np.random.default_rng(_GENERIC_HELDOUT_SEED)
    generic_heldout = np.sort(rng.choice(len(generic), size=min(HELDOUT_SAMPLES, len(generic)), replace=False))
    rng = np.random.default_rng(_AGRO_HELDOUT_SEED)
    agro_heldout = np.sort(rng.choice(len(agro), size=min(HELDOUT_SAMPLES, len(agro)), replace=False))
    return generic_heldout, agro_heldout


def _stable_indices(size: int, count: int, seed: int) -> np.ndarray:
    if count > size:
        raise ValueError(f"Requested {count} samples from a dataset with {size} samples")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(size), size=count, replace=False))


def _pretrain_dataset(
    spec: ArmSpec,
    seed: int,
    generic: MultimodalPatchDataset,
    agro: MultimodalPatchDataset,
    generic_heldout: np.ndarray,
    agro_heldout: np.ndarray,
) -> Dataset:
    generic_train_idx = np.setdiff1d(np.arange(len(generic)), generic_heldout)
    agro_train_idx = np.setdiff1d(np.arange(len(agro)), agro_heldout)
    if spec.data_source == "generic":
        return NamedSubset(generic, generic_train_idx, root=generic.root, shapes=generic.shapes)
    if spec.data_source == "mixed_full":
        return MixedPretrainDataset(
            [
                NamedSubset(generic, generic_train_idx, root=generic.root, shapes=generic.shapes),
                NamedSubset(agro, agro_train_idx, root=agro.root, shapes=agro.shapes),
            ]
        )
    if spec.data_source == "mixed_size_matched":
        generic_idx = _stable_indices(len(generic_train_idx), MIXED_SIZE_MATCHED_PER_POOL, seed + 1001)
        agro_idx = _stable_indices(len(agro_train_idx), MIXED_SIZE_MATCHED_PER_POOL, seed + 2001)
        return MixedPretrainDataset(
            [
                NamedSubset(generic, generic_train_idx[generic_idx], root=generic.root, shapes=generic.shapes),
                NamedSubset(agro, agro_train_idx[agro_idx], root=agro.root, shapes=agro.shapes),
            ]
        )
    raise ValueError(f"Unknown data_source={spec.data_source}")


def _make_model(spec: ArmSpec, dataset: Dataset) -> torch.nn.Module:
    shapes = getattr(dataset, "shapes")
    if spec.encoder == "pooled":
        return PooledTemporalJepaModel(
            s2_channels=shapes.s2_channels,
            s1_channels=shapes.s1_channels,
            model_dim=MODEL_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            predictor_layers=PREDICTOR_LAYERS,
        )
    if spec.encoder == "spatial":
        return SpatialTokenJepaModel(
            s2_channels=shapes.s2_channels,
            s1_channels=shapes.s1_channels,
            model_dim=MODEL_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            predictor_dim=PREDICTOR_DIM,
            predictor_layers=PREDICTOR_LAYERS,
        )
    raise ValueError(f"Unknown encoder={spec.encoder}")


def _model_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = [
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
    return {key: batch[key] for key in keys if key in batch}


def _clone_view(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def _drop_timesteps(view: dict[str, torch.Tensor], keep: torch.Tensor) -> None:
    drop = ~keep
    for key in ["s2", "s1", "climate"]:
        if key in view and view[key].ndim >= 2:
            view[key][drop] = 0
    for key in ["s2_mask", "s1_mask", "climate_mask"]:
        if key in view:
            view[key][drop] = 0
    for key in ["s2_available", "s1_available", "climate_available"]:
        if key in view:
            view[key][drop] = 0


def _corrupt_batch(
    batch: dict[str, torch.Tensor],
    mode: str,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if mode not in set(VIEWDROP_SENSOR_DROPOUT_MODES):
        raise ValueError(f"Unknown view-drop mode: {mode}")
    view = _clone_view(batch)
    b, t = view["s2"].shape[:2]
    time_keep = torch.ones((b, t), dtype=torch.bool, device=view["s2"].device)
    if mode in {"sensor_off_s2", "s2_off_tdrop50"}:
        view["s2"].zero_()
        view["s2_available"].zero_()
        if "s2_mask" in view:
            view["s2_mask"].zero_()
    if mode in {"sensor_off_s1", "s1_off_tdrop50"}:
        view["s1"].zero_()
        view["s1_available"].zero_()
        if "s1_mask" in view:
            view["s1_mask"].zero_()
    if mode in {"temporal_drop_50", "s2_off_tdrop50", "s1_off_tdrop50"}:
        if t > 2:
            time_keep[:, 1:] = torch.rand((b, t - 1), generator=generator, device=view["s2"].device) >= 0.50
            low = time_keep.sum(dim=1) < 2
            if low.any():
                time_keep[low, 1] = True
        _drop_timesteps(view, time_keep)
    return view, time_keep


def _sample_viewdrop_mode(modes: tuple[str, ...], generator: torch.Generator) -> str:
    idx = int(torch.randint(0, len(modes), (1,), generator=generator, device=generator.device).item())
    return modes[idx]


def _pooled_target_mask(
    batch_size: int,
    timesteps: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if timesteps < 2:
        raise ValueError("JEPA masking requires at least two timesteps")
    mask = torch.zeros((batch_size, timesteps), dtype=torch.bool, device=device)
    width = min(timesteps - 1, max(1, int(round(timesteps * TARGET_MASK_FRACTION))))
    for row in range(batch_size):
        start = int(torch.randint(1, max(2, timesteps - width + 1), (1,), generator=generator, device=device).item())
        mask[row, start : min(timesteps, start + width)] = True
    return mask


def _spatial_target_mask(
    batch_size: int,
    timesteps: int,
    spatial_tokens: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    grid = int(round(math.sqrt(spatial_tokens)))
    if grid * grid != spatial_tokens or grid < 2:
        raise ValueError(f"Spatial masks require square token grids, got {spatial_tokens}")
    budget = int(round(timesteps * spatial_tokens * TARGET_MASK_FRACTION))
    if budget <= 0 or budget >= timesteps * spatial_tokens:
        raise ValueError("Spatial target budget leaves no context")
    block = 2
    block_cells = block * block
    if budget % block_cells != 0:
        raise ValueError("Spatial target budget must be divisible by 2x2 block size")
    mask = torch.zeros((batch_size, timesteps, grid, grid), dtype=torch.bool, device=device)
    starts = [(y, x) for y in range(0, grid, block) for x in range(0, grid, block)]
    for row in range(batch_size):
        reserved_visible_t = int(torch.randint(0, timesteps, (1,), generator=generator, device=device).item())
        options = [(t, y, x) for t in range(timesteps) if t != reserved_visible_t for y, x in starts]
        order = torch.randperm(len(options), generator=generator, device=device).tolist()
        remaining = budget
        for index in order:
            t, y, x = options[index]
            current = mask[row, t, y : y + block, x : x + block]
            if bool(current.any()):
                continue
            current[:] = True
            remaining -= block_cells
            if remaining == 0:
                break
        if remaining != 0:
            raise RuntimeError("Could not construct exact spatial target mask")
        if bool(mask[row].all(dim=(1, 2)).all()):
            raise RuntimeError("Spatial target mask left no visible timestep")
    return mask.reshape(batch_size, timesteps, spatial_tokens)


def _target_mask(
    spec: ArmSpec,
    batch_size: int,
    timesteps: int,
    device: torch.device,
    generator: torch.Generator,
    spatial_tokens: int = 16,
) -> torch.Tensor:
    if spec.encoder == "pooled":
        return _pooled_target_mask(batch_size, timesteps, device, generator)
    return _spatial_target_mask(batch_size, timesteps, spatial_tokens, device, generator)


def _jepa_loss_components(output: SpatialJepaOutput) -> tuple[torch.Tensor, dict[str, float]]:
    local_pred = F.normalize(output.local_pred, dim=-1)
    local_target = F.normalize(output.local_target.detach(), dim=-1)
    local_distance = 1.0 - (local_pred * local_target).sum(dim=-1)
    local_weights = output.local_mask.float()
    local_loss = (local_distance * local_weights).sum() / local_weights.sum().clamp_min(1.0)
    global_pred = F.normalize(output.global_pred, dim=-1)
    global_target = F.normalize(output.global_target.detach(), dim=-1)
    global_loss = 1.0 - (global_pred * global_target).sum(dim=-1).mean()
    combined = local_loss + GLOBAL_LOSS_WEIGHT * global_loss
    return combined, {
        "local_loss": float(local_loss.detach().item()),
        "global_loss": float(global_loss.detach().item()),
        "combined_loss": float(combined.detach().item()),
        "valid_target_tokens": float(local_weights.sum().detach().item()),
    }


def _sequence_embedding(features: torch.Tensor, time_keep: torch.Tensor | None = None) -> torch.Tensor:
    if time_keep is None:
        return features.mean(dim=1)
    weights = time_keep.float().unsqueeze(-1)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _content_regularizer(embeddings: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    if embeddings.shape[0] < 2:
        zero = embeddings.sum() * 0.0
        return zero, {"variance_floor_loss": 0.0, "covariance_loss": 0.0}
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = F.relu(VARIANCE_FLOOR - std).mean()
    cov = centered.T @ centered / float(max(1, embeddings.shape[0] - 1))
    cov = cov / torch.sqrt(torch.diag(cov).clamp_min(1e-6)[:, None] * torch.diag(cov).clamp_min(1e-6)[None, :])
    off_diag = cov - torch.diag(torch.diag(cov))
    covariance_loss = off_diag.square().mean()
    total = VARIANCE_FLOOR_WEIGHT * variance_loss + COVARIANCE_WEIGHT * covariance_loss
    return total, {
        "variance_floor_loss": float(variance_loss.detach().item()),
        "covariance_loss": float(covariance_loss.detach().item()),
    }


def _training_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    spec: ArmSpec,
    target_mask: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    clean = _model_batch(batch)
    mode = _sample_viewdrop_mode(spec.viewdrop_modes, generator)
    corrupted, context_time_keep = _corrupt_batch(batch, mode, generator)
    degraded = _model_batch(corrupted)
    output = model.forward_views(
        target_mask=target_mask,
        context=degraded,
        target=clean,
        context_time_keep=context_time_keep,
    )
    loss, metrics = _jepa_loss_components(output)
    metrics.update({"mode": mode})
    if spec.content_regularized:
        clean_z = model.encode(**clean)
        degraded_z = model.encode(**degraded, masks=JepaBatchMasks(time_keep=context_time_keep))
        clean_reg, clean_metrics = _content_regularizer(_sequence_embedding(clean_z))
        degraded_reg, degraded_metrics = _content_regularizer(_sequence_embedding(degraded_z, context_time_keep))
        reg_loss = clean_reg + degraded_reg
        loss = loss + CONTENT_REG_MULTIPLIER * reg_loss
        metrics.update(
            {
                "content_regularizer": float(reg_loss.detach().item()),
                "clean_variance_floor_loss": clean_metrics["variance_floor_loss"],
                "clean_covariance_loss": clean_metrics["covariance_loss"],
                "degraded_variance_floor_loss": degraded_metrics["variance_floor_loss"],
                "degraded_covariance_loss": degraded_metrics["covariance_loss"],
            }
        )
    else:
        metrics.update(
            {
                "content_regularizer": 0.0,
                "clean_variance_floor_loss": 0.0,
                "clean_covariance_loss": 0.0,
                "degraded_variance_floor_loss": 0.0,
                "degraded_covariance_loss": 0.0,
            }
        )
    return loss, metrics


def _lr_lambda(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(float(step + 1) / float(warmup_steps), min_lr_ratio)
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _existing_checkpoints(run_dir: Path) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for path in sorted(run_dir.glob("checkpoint_step_*.pt")):
        try:
            step = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        checkpoints[f"step_{step}"] = path
    return checkpoints


def _latest_checkpoint(run_dir: Path, fixed_updates: int) -> tuple[int, Path] | None:
    latest: tuple[int, Path] | None = None
    for path in run_dir.glob("checkpoint_step_*.pt"):
        try:
            step = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if step > fixed_updates:
            continue
        if latest is None or step > latest[0]:
            latest = (step, path)
    return latest


def _scheduled_checkpoints(checkpoints: dict[str, Path], cfg: RunConfig) -> dict[str, Path]:
    scheduled = set(cfg.checkpoint_steps)
    out: dict[str, Path] = {}
    for role, path in checkpoints.items():
        try:
            step = int(role.split("_")[1])
        except (IndexError, ValueError):
            continue
        if step in scheduled and step <= cfg.fixed_updates:
            out[role] = path
    return out


def _has_all_scheduled_checkpoints(checkpoints: dict[str, Path], cfg: RunConfig) -> bool:
    expected = {f"step_{step}" for step in cfg.checkpoint_steps}
    return expected.issubset(set(checkpoints))


def _write_training_metadata(cfg: RunConfig, spec: ArmSpec) -> None:
    write_json(
        cfg.output_dir / "run_metadata.json",
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "arm": cfg.arm,
            "seed": cfg.seed,
            "fixed_updates": cfg.fixed_updates,
            "checkpoint_steps": list(cfg.checkpoint_steps),
            "arm_spec": asdict(spec),
        },
    )


def _eval_metadata(cfg: RunConfig, spec: ArmSpec) -> dict[str, Any]:
    arm_spec = asdict(spec)
    arm_spec["viewdrop_modes"] = list(arm_spec["viewdrop_modes"])
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "eval_schema_version": EVAL_SCHEMA_VERSION,
        "arm": cfg.arm,
        "seed": cfg.seed,
        "fixed_updates": cfg.fixed_updates,
        "checkpoint_steps": list(cfg.checkpoint_steps),
        "arm_spec": arm_spec,
        "cropharvest_conditions": list(DEFAULT_CONDITIONS),
        "cropharvest_holdouts": list(DEFAULT_HOLDOUTS),
        "eurocrops_conditions": list(EUROCROPS_CONDITIONS),
        "eurocrops_protocol": "latvia_portugal_to_estonia",
        "final_checkpoint_protocol": "full_cropharvest_and_eurocrops",
        "intermediate_checkpoint_protocol": "rwanda_ethiopia_clean_s1_off_tdrop50_embedding_only",
    }


def _eval_metadata_matches(path: Path, cfg: RunConfig, spec: ArmSpec) -> bool:
    if not path.exists():
        return False
    try:
        observed = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return observed == _eval_metadata(cfg, spec)


def _loader_options(num_workers: int, pin_memory: bool, prefetch_factor: int) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = prefetch_factor
    return options


def _infinite_loader(dataset: Dataset, cfg: RunConfig, device: torch.device, seed: int) -> Iterable[dict[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True, num_samples=cfg.fixed_updates * cfg.batch_size, generator=generator)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        **_loader_options(cfg.train_num_workers, cfg.pin_memory, cfg.prefetch_factor),
        drop_last=True,
    )
    for batch in loader:
        yield {key: value.to(device, non_blocking=cfg.pin_memory) for key, value in batch.items()}


def _safe_covariance(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    return centered.T @ centered / float(values.shape[0] - 1)


def _effective_rank_from_features(values: np.ndarray) -> float:
    cov = _safe_covariance(values)
    eig = np.linalg.eigvalsh(cov)
    eig = eig[eig > np.finfo(np.float64).eps]
    if len(eig) == 0:
        return 0.0
    prob = eig / eig.sum()
    return float(np.exp(-(prob * np.log(prob)).sum()))


def _offdiag_covariance_mean(values: np.ndarray) -> float:
    cov = _safe_covariance(values)
    if cov.size == 0:
        return 0.0
    diag = np.sqrt(np.clip(np.diag(cov), 1e-8, None))
    corr = cov / np.clip(diag[:, None] * diag[None, :], 1e-8, None)
    off = corr - np.diag(np.diag(corr))
    return float(np.mean(off**2))


def _shuffle_content(batch: dict[str, torch.Tensor], generator: torch.Generator) -> dict[str, torch.Tensor]:
    out = _clone_view(batch)
    b = int(batch["s2"].shape[0])
    if b < 2:
        return out
    offset = int(torch.randint(1, b, (1,), generator=generator, device=batch["s2"].device).item())
    perm = (torch.arange(b, device=batch["s2"].device) + offset) % b
    for key in ["s2", "s1", "climate"]:
        if key in out:
            out[key] = out[key][perm]
    return out


def _shuffle_missingness(batch: dict[str, torch.Tensor], generator: torch.Generator) -> dict[str, torch.Tensor]:
    out = _clone_view(batch)
    b = int(batch["s2"].shape[0])
    if b < 2:
        return out
    offset = int(torch.randint(1, b, (1,), generator=generator, device=batch["s2"].device).item())
    perm = (torch.arange(b, device=batch["s2"].device) + offset) % b
    for key in ["s2_mask", "s1_mask", "climate_mask", "s2_available", "s1_available", "climate_available"]:
        if key in out:
            out[key] = out[key][perm]
    return out


def _same_time_inter_sample_cosine(features: np.ndarray) -> float:
    if features.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    normalized = features / np.clip(norms, 1e-8, None)
    shifted = np.roll(normalized, shift=1, axis=0)
    return float(np.mean(np.sum(normalized * shifted, axis=-1)))


def _diagnostic_loaders(
    generic: MultimodalPatchDataset,
    agro: MultimodalPatchDataset,
    generic_heldout: np.ndarray,
    agro_heldout: np.ndarray,
    cfg: RunConfig,
) -> dict[str, DataLoader]:
    return {
        "generic_heldout": DataLoader(
            Subset(generic, generic_heldout.tolist()),
            batch_size=cfg.diagnostic_batch_size,
            shuffle=False,
            **_loader_options(cfg.diagnostic_num_workers, cfg.pin_memory, cfg.prefetch_factor),
        ),
        "agriculture_heldout": DataLoader(
            Subset(agro, agro_heldout.tolist()),
            batch_size=cfg.diagnostic_batch_size,
            shuffle=False,
            **_loader_options(cfg.diagnostic_num_workers, cfg.pin_memory, cfg.prefetch_factor),
        ),
    }


def _diagnose_checkpoint(
    model: torch.nn.Module,
    spec: ArmSpec,
    loaders: dict[str, DataLoader],
    cfg: RunConfig,
    step: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    checkpoint_rows: list[dict[str, Any]] = []
    corruption_rows: list[dict[str, Any]] = []
    spectra: dict[str, Any] = {}
    with torch.no_grad():
        for pool, loader in loaders.items():
            normal_losses: list[float] = []
            normal_local_losses: list[float] = []
            normal_global_losses: list[float] = []
            shuffled_losses: list[float] = []
            missingness_losses: list[float] = []
            clean_embeddings: list[np.ndarray] = []
            zero_embeddings: list[np.ndarray] = []
            sequence_embeddings: list[np.ndarray] = []
            target_embeddings: list[np.ndarray] = []
            local_target_tokens: list[np.ndarray] = []
            corruption_accumulators: dict[str, list[dict[str, float]]] = {mode: [] for mode in VIEWDROP_SENSOR_DROPOUT_MODES if mode != "clean"}
            generator = torch.Generator(device=device)
            generator.manual_seed(cfg.seed + step + 707)
            for batch in loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                mask = _target_mask(
                    spec,
                    batch["s2"].shape[0],
                    batch["s2"].shape[1],
                    device,
                    generator,
                    spatial_tokens=getattr(getattr(model, "context_encoder", None), "num_spatial_tokens", 16),
                )
                clean = _model_batch(batch)
                normal = model.forward_views(target_mask=mask, context=clean, target=clean)
                normal_loss, normal_meta = _jepa_loss_components(normal)
                shuffled = _model_batch(_shuffle_content(batch, generator))
                shuffled_out = model.forward_views(target_mask=mask, context=clean, target=shuffled)
                shuffled_loss, _ = _jepa_loss_components(shuffled_out)
                missing_context = _model_batch(_shuffle_missingness(batch, generator))
                missing_out = model.forward_views(target_mask=mask, context=missing_context, target=clean)
                missing_loss, _ = _jepa_loss_components(missing_out)
                normal_losses.append(float(normal_loss.item()))
                normal_local_losses.append(float(normal_meta["local_loss"]))
                normal_global_losses.append(float(normal_meta["global_loss"]))
                shuffled_losses.append(float(shuffled_loss.item()))
                missingness_losses.append(float(missing_loss.item()))
                target_embeddings.append(normal.global_target.detach().cpu().numpy())
                if hasattr(normal, "local_target") and normal.local_target is not None and hasattr(normal, "local_mask"):
                    lt = normal.local_target.detach().cpu().numpy()
                    lm = normal.local_mask.detach().cpu().numpy().astype(bool)
                    if lt.ndim == 4 and lt.shape[2] > 1:
                        for ti in range(lt.shape[1]):
                            for si in range(lt.shape[2]):
                                slot_embs = lt[:, ti, si, :][lm[:, ti, si]]
                                if len(slot_embs) >= 2:
                                    local_target_tokens.append(slot_embs)
                clean_z = model.encode(**clean)
                sequence_embeddings.append(clean_z.detach().cpu().numpy())
                clean_embedding = _sequence_embedding(clean_z)
                clean_embeddings.append(clean_embedding.detach().cpu().numpy())
                zero_batch = _clone_view(batch)
                zero_batch["s2"].zero_()
                zero_batch["s1"].zero_()
                zero_z = model.encode(**_model_batch(zero_batch))
                zero_embeddings.append(_sequence_embedding(zero_z).detach().cpu().numpy())
                for mode in corruption_accumulators:
                    corrupted, keep = _corrupt_batch(batch, mode, generator)
                    degraded = _model_batch(corrupted)
                    degraded_z = model.encode(**degraded, masks=JepaBatchMasks(time_keep=keep))
                    full_clean = _sequence_embedding(clean_z)
                    retained_clean = _sequence_embedding(clean_z, keep)
                    degraded_embedding = _sequence_embedding(degraded_z, keep)
                    corruption_accumulators[mode].append(
                        {
                            "full_clean_displacement": float(
                                (1.0 - F.cosine_similarity(full_clean, degraded_embedding, dim=-1)).mean().item()
                            ),
                            "retained_displacement": float(
                                (1.0 - F.cosine_similarity(retained_clean, degraded_embedding, dim=-1)).mean().item()
                            ),
                            "degraded_variance": float(degraded_embedding.var(dim=0, unbiased=False).mean().item()),
                        }
                    )
            clean_np = np.concatenate(clean_embeddings, axis=0)
            zero_np = np.concatenate(zero_embeddings, axis=0)
            sequence_np = np.concatenate(sequence_embeddings, axis=0)
            target_np = np.concatenate(target_embeddings, axis=0)
            residual = clean_np - zero_np
            checkpoint_rows.append(
                {
                    "arm": cfg.arm,
                    "seed": cfg.seed,
                    "step": step,
                    "pool": pool,
                    "normal_loss": float(np.mean(normal_losses)),
                    "content_shuffle_loss": float(np.mean(shuffled_losses)),
                    "content_shuffle_gap": float(np.mean(shuffled_losses) - np.mean(normal_losses)),
                    "missingness_shuffle_loss": float(np.mean(missingness_losses)),
                    "missingness_shuffle_gap": float(np.mean(missingness_losses) - np.mean(normal_losses)),
                    "embedding_variance": float(clean_np.var(axis=0).mean()),
                    "embedding_effective_rank": _effective_rank_from_features(clean_np),
                    "target_effective_rank": _effective_rank_from_features(target_np),
                    "same_time_inter_sample_cosine": _same_time_inter_sample_cosine(sequence_np),
                    "clean_zero_cosine": float(np.mean(np.sum(clean_np * zero_np, axis=1) / (np.linalg.norm(clean_np, axis=1) * np.linalg.norm(zero_np, axis=1) + 1e-8))),
                    "residual_variance": float(residual.var(axis=0).mean()),
                    "residual_effective_rank": _effective_rank_from_features(residual),
                    "offdiag_covariance": _offdiag_covariance_mean(clean_np),
                    "normal_local_loss": float(np.mean(normal_local_losses)),
                    "normal_global_loss": float(np.mean(normal_global_losses)),
                    "spatial_token_target_variance": float(
                        np.mean([t.var(axis=0).mean() for t in local_target_tokens])
                    ) if local_target_tokens else 0.0,
                    "spatial_token_target_effective_rank": float(
                        np.mean([_effective_rank_from_features(t) for t in local_target_tokens])
                    ) if local_target_tokens else 0.0,
                    "spatial_token_same_slot_cosine": float(
                        np.mean([_same_time_inter_sample_cosine(t) for t in local_target_tokens])
                    ) if local_target_tokens else 0.0,
                }
            )
            spectra[f"{cfg.arm}/seed_{cfg.seed}/step_{step}/{pool}"] = {
                "context_eigenvalues": np.linalg.eigvalsh(_safe_covariance(clean_np)).astype(float).tolist(),
                "target_eigenvalues": np.linalg.eigvalsh(_safe_covariance(target_np)).astype(float).tolist(),
                "residual_eigenvalues": np.linalg.eigvalsh(_safe_covariance(residual)).astype(float).tolist(),
            }
            for mode, vals in corruption_accumulators.items():
                corruption_rows.append(
                    {
                        "arm": cfg.arm,
                        "seed": cfg.seed,
                        "step": step,
                        "pool": pool,
                        "condition": mode,
                        "full_clean_displacement": float(np.mean([v["full_clean_displacement"] for v in vals])),
                        "retained_displacement": float(np.mean([v["retained_displacement"] for v in vals])),
                        "degraded_variance": float(np.mean([v["degraded_variance"] for v in vals])),
                    }
                )
    model.train()
    return checkpoint_rows, corruption_rows, spectra


def _train_one(
    cfg: RunConfig,
    spec: ArmSpec,
    dataset: Dataset,
    generic: MultimodalPatchDataset,
    agro: MultimodalPatchDataset,
    generic_heldout: np.ndarray,
    agro_heldout: np.ndarray,
    device: torch.device,
) -> dict[str, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model = _make_model(spec, dataset).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
    checkpoints: dict[str, Path] = _existing_checkpoints(cfg.output_dir)
    start_step = 1
    mode_counts: dict[str, int] = {mode: 0 for mode in spec.viewdrop_modes}
    latest = _latest_checkpoint(cfg.output_dir, cfg.fixed_updates)
    _write_training_metadata(cfg, spec)
    if latest is not None:
        resume_step, resume_path = latest
        if resume_step < cfg.fixed_updates:
            raise RuntimeError(
                f"Partial non-resumable run found for {cfg.arm} seed={cfg.seed}: "
                f"{resume_path}. Delete {cfg.output_dir} before relaunching."
            )
        if not _has_all_scheduled_checkpoints(checkpoints, cfg):
            raise RuntimeError(
                f"Final checkpoint exists for {cfg.arm} seed={cfg.seed}, but scheduled "
                f"checkpoints are incomplete. Delete {cfg.output_dir} before relaunching."
            )
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        start_step = resume_step + 1
        saved_counts = state.get("mode_counts", {})
        mode_counts.update({str(k): int(v) for k, v in saved_counts.items() if str(k) in mode_counts})
        print(
            f"  [{cfg.arm} seed={cfg.seed}] resuming from {resume_path.name} at step {start_step}",
            flush=True,
        )
    stream = _infinite_loader(dataset, cfg, device, cfg.seed + 3109)
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + 911)
    history: list[dict[str, Any]] = []
    recent_losses: list[float] = []
    target_count_sum: np.ndarray | None = None
    target_count_batches = 0
    started = time.time()
    if start_step > cfg.fixed_updates:
        print(
            f"  [{cfg.arm} seed={cfg.seed}] training already reached step {cfg.fixed_updates}",
            flush=True,
        )
        if not (cfg.output_dir / "train_history.json").exists():
            write_json(
                cfg.output_dir / "train_history.json",
                {
                    "train_seconds": 0.0,
                    "history": [],
                    "fixed_updates": cfg.fixed_updates,
                    "checkpoint_steps": list(cfg.checkpoint_steps),
                    "arm_spec": asdict(spec),
                    "recovered_from_checkpoints": True,
                },
            )
        return _scheduled_checkpoints(checkpoints, cfg)
    for step in range(start_step, cfg.fixed_updates + 1):
        model.train()
        batch = next(stream)
        lr = cfg.lr * _lr_lambda(step, cfg.fixed_updates, 2_000, cfg.min_lr / cfg.lr)
        _set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        mask = _target_mask(
            spec,
            batch["s2"].shape[0],
            batch["s2"].shape[1],
            device,
            generator,
            spatial_tokens=getattr(getattr(model, "context_encoder", None), "num_spatial_tokens", 16),
        )
        per_timestep_counts = mask.reshape(mask.shape[0], mask.shape[1], -1).sum(dim=(0, 2)).detach().cpu().numpy()
        target_count_sum = per_timestep_counts if target_count_sum is None else target_count_sum + per_timestep_counts
        target_count_batches += int(mask.shape[0])
        loss, metrics = _training_loss(model, batch, spec, mask, generator)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss for {cfg.arm} seed {cfg.seed} step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.update_target_encoder(cosine_ema_momentum(step, cfg.fixed_updates))
        recent_losses.append(float(loss.item()))
        mode_counts[str(metrics["mode"])] = mode_counts.get(str(metrics["mode"]), 0) + 1
        if step in cfg.checkpoint_steps:
            checkpoint_path = cfg.output_dir / f"checkpoint_step_{step}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "arm_spec": asdict(spec),
                    "step": step,
                    "mode_counts": mode_counts,
                },
                checkpoint_path,
            )
            checkpoints[f"step_{step}"] = checkpoint_path
            row = {
                "arm": cfg.arm,
                "seed": cfg.seed,
                "step": step,
                "train_loss": float(np.mean(recent_losses)),
                "lr": float(lr),
                "mode_counts": dict(mode_counts),
                "target_counts_per_sample_by_timestep": (
                    (target_count_sum / float(max(1, target_count_batches))).astype(float).tolist()
                    if target_count_sum is not None
                    else []
                ),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            recent_losses = []
            target_count_sum = None
            target_count_batches = 0
    write_json(
        cfg.output_dir / "train_history.json",
        {
            "train_seconds": time.time() - started,
            "history": history,
            "fixed_updates": cfg.fixed_updates,
            "checkpoint_steps": list(cfg.checkpoint_steps),
            "arm_spec": asdict(spec),
        },
    )
    _write_training_metadata(cfg, spec)
    return _scheduled_checkpoints(checkpoints, cfg)


def _condition_map(selected: list[str]) -> list[tuple[str, str, float]]:
    available = {name: (sensor, temporal_drop) for name, sensor, temporal_drop in CONDITIONS}
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    return [(name, *available[name]) for name in selected]


def _limit(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def _evaluate_cropharvest_holdout(
    cfg: RunConfig,
    model: torch.nn.Module,
    dataset: MultimodalPatchDataset,
    y: np.ndarray,
    groups: np.ndarray,
    holdout: str,
    conditions: list[tuple[str, str, float]],
    device: torch.device,
    checkpoint_role: str,
    matched_retrained_models: list[str] | None = None,
) -> list[dict[str, Any]]:
    _, _, test_idx, train_idx = make_strict_holdout_splits(y, groups, holdout, cfg.seed)
    rows: list[dict[str, Any]] = []
    clean_x_train = extract_embeddings(model, dataset, train_idx, device, cfg.eval_batch_size, cfg.eval_num_workers, "none", 0.0, cfg.seed)
    clean_stats_train = extract_raw_stats(dataset, train_idx, cfg.eval_batch_size, cfg.eval_num_workers, "none", 0.0, cfg.seed)
    base = {
        "experiment": "[10]",
        "benchmark": "cropharvest",
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
            cfg.eval_num_workers,
            sensor_off,
            temporal_drop,
            cfg.seed + 999,
        )
        stats_test = extract_raw_stats(dataset, test_idx, cfg.eval_batch_size, cfg.eval_num_workers, sensor_off, temporal_drop, cfg.seed + 999)
        for model_name, left, right in [
            ("embedding", clean_x_train, x_test),
            ("raw_stats", clean_stats_train, stats_test),
            ("embedding_plus_raw_stats", np.concatenate([clean_x_train, clean_stats_train], axis=1), np.concatenate([x_test, stats_test], axis=1)),
        ]:
            before = len(rows)
            run_probes(rows, model_name, left, right, y[train_idx], y[test_idx], condition, cfg.seed)
            for row in rows[before:]:
                row.update({**base, "robustness_protocol": "clean_train_degraded_test"})
        if condition == "clean":
            continue
        if matched_retrained_models is not None and len(matched_retrained_models) == 0:
            continue
        run_matched = matched_retrained_models or ["embedding", "raw_stats", "embedding_plus_raw_stats"]
        matched_x = extract_embeddings(model, dataset, train_idx, device, cfg.eval_batch_size, cfg.eval_num_workers, sensor_off, temporal_drop, cfg.seed)
        matched_stats = extract_raw_stats(dataset, train_idx, cfg.eval_batch_size, cfg.eval_num_workers, sensor_off, temporal_drop, cfg.seed)
        paired: list[tuple[str, np.ndarray, np.ndarray]] = []
        if "embedding" in run_matched:
            paired.append(("embedding", matched_x, x_test))
        if "raw_stats" in run_matched:
            paired.append(("raw_stats", matched_stats, stats_test))
        if "embedding_plus_raw_stats" in run_matched:
            paired.append(("embedding_plus_raw_stats", np.concatenate([matched_x, matched_stats], axis=1), np.concatenate([x_test, stats_test], axis=1)))
        for model_name, left, right in paired:
            before = len(rows)
            run_probes(rows, model_name, left, right, y[train_idx], y[test_idx], condition, cfg.seed)
            for row in rows[before:]:
                row.update({**base, "robustness_protocol": "condition_matched_retrained"})
    return rows


def _load_eurocrops_labels() -> tuple[np.ndarray, np.ndarray]:
    if not EUROCROPS_LABELS_CSV.exists():
        raise FileNotFoundError(f"EuroCropsML labels CSV not found: {EUROCROPS_LABELS_CSV}")
    labels: list[str] = []
    countries: list[str] = []
    with EUROCROPS_LABELS_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "label" not in fieldnames:
            raise ValueError(
                f"EuroCrops labels CSV missing required column 'label'. "
                f"Found columns: {fieldnames}"
            )
        if "country" not in fieldnames:
            raise ValueError(
                f"EuroCrops labels CSV missing required column 'country'. "
                f"Found columns: {fieldnames}"
            )
        for row in reader:
            labels.append(str(row["label"]))
            countries.append(str(row["country"]))
    countries_arr = np.asarray(countries, dtype=object)
    found = sorted(np.unique(countries_arr).tolist())
    required = {"Latvia", "Portugal", "Estonia"}
    missing = required - set(found)
    if missing:
        raise ValueError(
            f"EuroCropsML requires train countries Latvia+Portugal and test country Estonia. "
            f"Found countries: {found}. Missing: {sorted(missing)}"
        )
    classes = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return np.asarray([classes[label] for label in labels], dtype=np.int64), countries_arr


def _fit_multiclass_probe(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Any:
    clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            max_iter=MULTICLASS_PROBE_MAX_ITER,
            tol=MULTICLASS_PROBE_TOL,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
    )
    return clf.fit(x_train, y_train)


def _safe_probability_matrix(prob: np.ndarray) -> tuple[np.ndarray, int]:
    prob = np.asarray(prob, dtype=np.float64)
    repaired = int((~np.isfinite(prob)).sum())
    if repaired:
        prob = np.where(np.isfinite(prob), prob, 0.0)
    prob = np.clip(prob, 0.0, None)
    row_sum = prob.sum(axis=1, keepdims=True)
    zero_rows = (row_sum[:, 0] <= 0.0) | (~np.isfinite(row_sum[:, 0]))
    if np.any(zero_rows):
        repaired += int(np.sum(zero_rows))
        prob[zero_rows] = 1.0 / max(1, prob.shape[1])
        row_sum = prob.sum(axis=1, keepdims=True)
    prob = prob / np.clip(row_sum, 1e-12, None)
    return prob.astype(np.float32, copy=False), repaired


def _few_shot_sample(
    y: np.ndarray, budget: float, seed: int
) -> tuple[np.ndarray, float]:
    """Per-class few-shot sampler.
    
    For each class, samples at least 1 sample and at most n_class samples.
    Returns (indices, effective_budget) where effective_budget is the
    true fraction of the training set that was sampled.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    total = len(y)
    indices: list[int] = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        n = len(c_idx)
        desired = max(1, min(n, int(np.ceil(budget * n))))
        picked = rng.choice(c_idx.tolist(), size=min(desired, n), replace=False)
        indices.extend(picked)
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    effective_budget = len(indices) / max(1, total)
    return indices, effective_budget


def _run_multiclass_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    base: dict[str, Any],
    seed: int,
) -> None:
    n_test_original = len(y_test)
    test_only = np.setdiff1d(np.unique(y_test), np.unique(y_train))
    n_test_unseen_classes = len(test_only)
    shared = np.sort(np.intersect1d(np.unique(y_train), np.unique(y_test)))
    train_mask = np.isin(y_train, shared)
    test_mask = np.isin(y_test, shared)
    y_train_f = y_train[train_mask]
    y_test_f = y_test[test_mask]
    x_train_f = x_train[train_mask]
    x_test_f = x_test[test_mask]
    n_test_dropped = n_test_original - len(y_test_f)
    class_map = {c: i for i, c in enumerate(sorted(shared))}
    y_train_mapped = np.asarray([class_map[c] for c in y_train_f], dtype=np.int64)
    y_test_mapped = np.asarray([class_map[c] for c in y_test_f], dtype=np.int64)
    for budget in [0.01, 0.05, 0.10, 0.25, 1.0]:
        if budget < 1.0:
            try:
                sub, _ = train_test_split(
                    np.arange(len(y_train_mapped)), train_size=budget,
                    random_state=seed + int(budget * 1000), stratify=y_train_mapped,
                )
                effective_budget = len(sub) / max(1, len(y_train_mapped))
            except ValueError:
                sub, effective_budget = _few_shot_sample(
                    y_train_mapped, budget, seed + int(budget * 1000),
                )
        else:
            sub = np.arange(len(y_train_mapped))
            effective_budget = 1.0
        print(
            f"  [{base.get('arm')} seed={base.get('seed')}] multiclass probe "
            f"{base.get('benchmark')} {base.get('condition')} {model_name} budget={budget} "
            f"n_train={len(sub)} n_classes={len(np.unique(y_train_mapped[sub]))}",
            flush=True,
        )
        clf = _fit_multiclass_probe(x_train_f[sub], y_train_mapped[sub], seed + int(budget * 1000))
        pred = clf.predict(x_test_f)
        prob, probability_repairs = _safe_probability_matrix(clf.predict_proba(x_test_f))
        n_seen_train = len(np.unique(y_train_mapped[sub]))
        n_full_train = len(np.unique(y_train_mapped))
        row = {
            **base,
            "model": model_name,
            "label_budget": budget,
            "effective_label_budget": round(effective_budget, 6),
            "n_train_sub": int(len(sub)),
            "n_train_available": int(len(y_train_f)),
            "n_test": int(len(y_test_f)),
            "n_test_original": n_test_original,
            "n_test_unseen_dropped": n_test_dropped,
            "n_classes_test_unseen": n_test_unseen_classes,
            "n_classes_full": n_full_train,
            "n_classes_sub": n_seen_train,
            "probe_estimator": "sgd_log_loss",
            "probe_max_iter": MULTICLASS_PROBE_MAX_ITER,
            "probe_tol": MULTICLASS_PROBE_TOL,
            "probe_probability_repairs": probability_repairs,
            "macro_f1": float(f1_score(y_test_mapped, pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test_mapped, pred)),
        }
        if prob.shape[1] >= 3:
            row["top_3_accuracy"] = float(top_k_accuracy_score(y_test_mapped, prob, k=min(3, prob.shape[1]), labels=np.arange(prob.shape[1])))
        rows.append(row)


def _evaluate_eurocropsml(
    cfg: RunConfig,
    model: torch.nn.Module,
    device: torch.device,
    checkpoint_role: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    y, countries = _load_eurocrops_labels()
    dataset = MultimodalPatchDataset(EUROCROPS_ZARR, s2_reflectance_divisor=EUROCROPSML_S2_REFLECTANCE_DIVISOR)
    if len(y) != len(dataset):
        raise ValueError(f"EuroCrops labels length {len(y)} != dataset length {len(dataset)}")
    train_idx = np.where((countries == "Latvia") | (countries == "Portugal"))[0]
    test_idx = np.where(countries == "Estonia")[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("EuroCropsML requires Latvia/Portugal train and Estonia test samples")
    rows: list[dict[str, Any]] = []
    clean_x_train = extract_embeddings(model, dataset, train_idx, device, batch_size, cfg.eval_num_workers, "none", 0.0, cfg.seed)
    clean_stats_train = extract_raw_stats(dataset, train_idx, batch_size, cfg.eval_num_workers, "none", 0.0, cfg.seed)
    for condition, temporal_drop in [("clean", 0.0), ("temporal_drop_25", 0.25), ("temporal_drop_50", 0.50)]:
        print(
            f"  [{cfg.arm} seed={cfg.seed}] EuroCrops {checkpoint_role} condition={condition}",
            flush=True,
        )
        x_test = extract_embeddings(model, dataset, test_idx, device, batch_size, cfg.eval_num_workers, "none", temporal_drop, cfg.seed + 515)
        stats_test = extract_raw_stats(dataset, test_idx, batch_size, cfg.eval_num_workers, "none", temporal_drop, cfg.seed + 515)
        base = {
            "experiment": "[10]",
            "benchmark": "eurocropsml",
            "arm": cfg.arm,
            "holdout": "estonia",
            "condition": condition,
            "seed": cfg.seed,
            "checkpoint_role": checkpoint_role,
            "robustness_protocol": "latvia_portugal_to_estonia",
        }
        _run_multiclass_rows(rows, "embedding", clean_x_train, x_test, y[train_idx], y[test_idx], base, cfg.seed)
        _run_multiclass_rows(rows, "raw_stats", clean_stats_train, stats_test, y[train_idx], y[test_idx], base, cfg.seed)
        _run_multiclass_rows(
            rows,
            "embedding_plus_raw_stats",
            np.concatenate([clean_x_train, clean_stats_train], axis=1),
            np.concatenate([x_test, stats_test], axis=1),
            y[train_idx],
            y[test_idx],
            base,
            cfg.seed,
        )
    return rows


def _evaluate_checkpoints(
    cfg: RunConfig,
    spec: ArmSpec,
    dataset: Dataset,
    checkpoints: dict[str, Path],
    device: torch.device,
) -> list[dict[str, Any]]:
    eval_dataset = MultimodalPatchDataset(CROPHARVEST_ZARR, s2_reflectance_divisor=CROPHARVEST_S2_REFLECTANCE_DIVISOR)
    valid_files = valid_cropharvest_files(CROPHARVEST_ARRAYS, eval_dataset.shapes.timesteps)
    if len(valid_files) != len(eval_dataset):
        raise ValueError(f"Valid H5 count {len(valid_files)} != eval Zarr length {len(eval_dataset)}")
    y, groups = load_labels(valid_files, CROPHARVEST_LABELS)
    conditions = _condition_map(DEFAULT_CONDITIONS)
    sentinel_conditions = [c for c in conditions if c[0] in {"clean", "s1_off_tdrop50"}]
    rows: list[dict[str, Any]] = []
    model = _make_model(spec, dataset).to(device)
    for checkpoint_role, path in sorted(checkpoints.items(), key=lambda item: int(item[0].split("_")[1])):
        checkpoint_probe_path = cfg.output_dir / f"probe_results_{checkpoint_role}.csv"
        checkpoint_summary_path = cfg.output_dir / f"probe_summary_{checkpoint_role}.csv"
        if checkpoint_probe_path.exists() and checkpoint_summary_path.exists():
            rows.extend(_read_csv_rows(checkpoint_probe_path))
            continue
        step_num = int(checkpoint_role.split("_")[1])
        is_final = (step_num == FIXED_UPDATES)
        chunk_names = (
            [f"cropharvest_{holdout}" for holdout in DEFAULT_HOLDOUTS] + ["eurocropsml"]
            if is_final
            else [f"cropharvest_{holdout}" for holdout in ["rwanda-ceo", "ethiopia"]]
        )
        chunk_rows = _read_checkpoint_chunks(cfg, checkpoint_role, chunk_names)
        if chunk_rows is not None:
            rows.extend(chunk_rows)
            _write_checkpoint_outputs(cfg, checkpoint_role, chunk_rows)
            continue
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        checkpoint_rows: list[dict[str, Any]] = []
        if is_final:
            for holdout in DEFAULT_HOLDOUTS:
                chunk_name = f"cropharvest_{holdout}"
                chunk_path = _checkpoint_chunk_path(cfg, checkpoint_role, chunk_name)
                if chunk_path.exists():
                    checkpoint_rows.extend(_read_csv_rows(chunk_path))
                    continue
                print(
                    f"  [{cfg.arm} seed={cfg.seed}] CropHarvest {checkpoint_role} holdout={holdout}",
                    flush=True,
                )
                holdout_rows = _evaluate_cropharvest_holdout(
                    cfg, model, eval_dataset, y, groups, holdout, conditions, device, checkpoint_role,
                    matched_retrained_models=["embedding", "embedding_plus_raw_stats"],
                )
                checkpoint_rows.extend(holdout_rows)
                _write_checkpoint_chunk(cfg, checkpoint_role, chunk_name, holdout_rows)
            arm_spec = ARM_SPECS[cfg.arm]
            eurocrops_batch_size = (
                EUROCROPS_EVAL_BATCH_SIZE_POOLED if arm_spec.encoder == "pooled" else EUROCROPS_EVAL_BATCH_SIZE_SPATIAL
            )
            eurocrops_chunk = "eurocropsml"
            eurocrops_path = _checkpoint_chunk_path(cfg, checkpoint_role, eurocrops_chunk)
            if eurocrops_path.exists():
                checkpoint_rows.extend(_read_csv_rows(eurocrops_path))
            else:
                eurocrops_rows = _evaluate_eurocropsml(cfg, model, device, checkpoint_role, eurocrops_batch_size)
                checkpoint_rows.extend(eurocrops_rows)
                _write_checkpoint_chunk(cfg, checkpoint_role, eurocrops_chunk, eurocrops_rows)
        else:
            for holdout in ["rwanda-ceo", "ethiopia"]:
                chunk_name = f"cropharvest_{holdout}"
                chunk_path = _checkpoint_chunk_path(cfg, checkpoint_role, chunk_name)
                if chunk_path.exists():
                    checkpoint_rows.extend(_read_csv_rows(chunk_path))
                    continue
                _, _, test_idx, train_idx = make_strict_holdout_splits(y, groups, holdout, cfg.seed)
                x_train = extract_embeddings(model, eval_dataset, train_idx, device, cfg.eval_batch_size, cfg.eval_num_workers, "none", 0.0, cfg.seed)
                for condition, sensor_off, temporal_drop in sentinel_conditions:
                    x_test = extract_embeddings(model, eval_dataset, test_idx, device, cfg.eval_batch_size, cfg.eval_num_workers, sensor_off, temporal_drop, cfg.seed + 999)
                    base = {
                        "experiment": "[10]",
                        "benchmark": "cropharvest",
                        "arm": cfg.arm,
                        "holdout": holdout,
                        "seed": cfg.seed,
                        "checkpoint_role": checkpoint_role,
                        "robustness_protocol": "clean_train_degraded_test",
                    }
                    before = len(checkpoint_rows)
                    run_probes(checkpoint_rows, "embedding", x_train, x_test, y[train_idx], y[test_idx], condition, cfg.seed)
                    for row in checkpoint_rows[before:]:
                        row.update(base)
                _write_checkpoint_chunk(
                    cfg,
                    checkpoint_role,
                    chunk_name,
                    [
                        row
                        for row in checkpoint_rows
                        if row.get("checkpoint_role") == checkpoint_role and row.get("holdout") == holdout
                    ],
                )
        rows.extend(checkpoint_rows)
        _write_checkpoint_outputs(cfg, checkpoint_role, checkpoint_rows)
    return rows


def _evaluate_one_run(
    cfg: RunConfig,
    spec: ArmSpec,
    pretrain: Dataset,
    checkpoints: dict[str, Path],
    generic: MultimodalPatchDataset,
    agro: MultimodalPatchDataset,
    generic_heldout: np.ndarray,
    agro_heldout: np.ndarray,
    device: torch.device,
) -> list[dict[str, Any]]:
    checkpoints = _scheduled_checkpoints(checkpoints, cfg)
    expected_roles = {f"step_{step}" for step in cfg.checkpoint_steps}
    missing = sorted(expected_roles - set(checkpoints))
    if missing:
        raise FileNotFoundError(f"{cfg.arm} seed={cfg.seed} missing checkpoints for evaluation: {missing}")

    model = _make_model(spec, pretrain).to(device)
    diagnostic_outputs = [
        cfg.output_dir / "checkpoint_diagnostics.csv",
        cfg.output_dir / "corruption_diagnostics.csv",
        cfg.output_dir / "covariance_spectra.json",
    ]
    if not all(path.exists() for path in diagnostic_outputs):
        loaders = _diagnostic_loaders(generic, agro, generic_heldout, agro_heldout, cfg)
        checkpoint_rows: list[dict[str, Any]] = []
        corruption_rows: list[dict[str, Any]] = []
        spectra: dict[str, Any] = {}
        for checkpoint_role, path in sorted(checkpoints.items(), key=lambda item: int(item[0].split("_")[1])):
            step_num = int(checkpoint_role.split("_")[1])
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            ckpt_rows, corr_rows, ckpt_spectra = _diagnose_checkpoint(model, spec, loaders, cfg, step_num, device)
            checkpoint_rows.extend(ckpt_rows)
            corruption_rows.extend(corr_rows)
            spectra.update(ckpt_spectra)

        write_csv(cfg.output_dir / "checkpoint_diagnostics.csv", checkpoint_rows)
        write_csv(cfg.output_dir / "corruption_diagnostics.csv", corruption_rows)
        write_json(cfg.output_dir / "covariance_spectra.json", spectra)
    probe_rows = _evaluate_checkpoints(cfg, spec, pretrain, checkpoints, device)
    write_csv(cfg.output_dir / "probe_results.csv", probe_rows)
    write_json(cfg.output_dir / "eval_metadata.json", _eval_metadata(cfg, spec))
    return probe_rows


def _run_training_worker(gpu_id: int, worker_arms: list[str], q: "mp.Queue") -> None:
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        generic, agro = _base_datasets()
        generic_heldout, agro_heldout = _heldout_indices(generic, agro)
        for arm_name in worker_arms:
            spec = ARM_SPECS[arm_name]
            for seed in SEEDS:
                run_dir = OUTPUT_DIR / f"{arm_name}_seed{seed}"
                if _is_training_complete(arm_name, seed):
                    print(f"  [{arm_name} seed={seed}] training already complete, skipping", flush=True)
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                _set_seed(seed)
                print(f"\n=== GPU{gpu_id} [10] train {arm_name} seed={seed} ===", flush=True)
                cfg = RunConfig(arm=arm_name, seed=seed, output_dir=run_dir)
                pretrain = _pretrain_dataset(spec, seed, generic, agro, generic_heldout, agro_heldout)
                _train_one(
                    cfg, spec, pretrain, generic, agro, generic_heldout, agro_heldout, device,
                )
        q.put({"gpu_id": gpu_id, "success": True})
    except Exception:
        q.put({"gpu_id": gpu_id, "success": False, "error": traceback.format_exc()})
        raise


def _run_evaluation_worker(gpu_id: int, worker_arms: list[str], q: "mp.Queue") -> None:
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        generic, agro = _base_datasets()
        generic_heldout, agro_heldout = _heldout_indices(generic, agro)
        all_rows: list[dict[str, Any]] = []
        for arm_name in worker_arms:
            spec = ARM_SPECS[arm_name]
            for seed in SEEDS:
                run_dir = OUTPUT_DIR / f"{arm_name}_seed{seed}"
                if not _is_training_complete(arm_name, seed):
                    raise RuntimeError(f"{arm_name} seed={seed} is not fully trained")
                if _is_run_complete(arm_name, seed):
                    print(f"  [{arm_name} seed={seed}] evaluation already complete, skipping", flush=True)
                    probe_path = run_dir / "probe_results.csv"
                    if probe_path.exists():
                        with probe_path.open("r", newline="", encoding="utf-8") as handle:
                            all_rows.extend(list(csv.DictReader(handle)))
                    continue
                _set_seed(seed)
                print(f"\n=== GPU{gpu_id} [10] eval {arm_name} seed={seed} ===", flush=True)
                cfg = RunConfig(arm=arm_name, seed=seed, output_dir=run_dir)
                pretrain = _pretrain_dataset(spec, seed, generic, agro, generic_heldout, agro_heldout)
                checkpoints = _existing_checkpoints(run_dir)
                probe_rows = _evaluate_one_run(
                    cfg, spec, pretrain, checkpoints, generic, agro, generic_heldout, agro_heldout, device,
                )
                all_rows.extend(probe_rows)
        result_path = OUTPUT_DIR / f"_worker_{gpu_id}_results.json"
        result_path.write_text(json.dumps(all_rows, indent=2, default=str))
        q.put({"gpu_id": gpu_id, "success": True, "result_path": str(result_path)})
    except Exception:
        q.put({"gpu_id": gpu_id, "success": False, "error": traceback.format_exc()})
        raise


def _priority_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("benchmark") == "cropharvest"
        and row.get("condition") in {"clean", "sensor_off_s2", "sensor_off_s1", "temporal_drop_50", "s2_off_tdrop50", "s1_off_tdrop50"}
        and float(row.get("label_budget", -1.0)) == 1.0
        and row.get("robustness_protocol") == "clean_train_degraded_test"
    ]


def _validate_gpu_setup(arms: list[str]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; [10] requires at least one NVIDIA GPU")
    if torch.cuda.device_count() <= max(GPU_ASSIGNMENTS):
        raise RuntimeError(
            f"GPU count {torch.cuda.device_count()} is insufficient; "
            f"need at least {max(GPU_ASSIGNMENTS) + 1} devices for "
            f"GPU_ASSIGNMENTS {dict(GPU_ASSIGNMENTS)}"
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


def _run_workers(arms: list[str], phase: str) -> None:
    ctx = mp.get_context(_MP_CONTEXT)
    q: mp.Queue = ctx.Queue()
    gpu_workers: dict[int, mp.Process] = {}
    if phase == "train":
        target = _run_training_worker
    elif phase == "eval":
        target = _run_evaluation_worker
    else:
        raise ValueError(f"Unknown worker phase: {phase}")
    for gpu_id, worker_arms in GPU_ASSIGNMENTS.items():
        active = [a for a in worker_arms if a in arms]
        if not active:
            continue
        p = ctx.Process(target=target, args=(gpu_id, active, q))
        p.start()
        gpu_workers[gpu_id] = p
    if not gpu_workers:
        raise ValueError("No GPU workers launched; check GPU_ASSIGNMENTS vs selected arms")

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
                f"GPU worker {failed_gpu} exited with code "
                f"{gpu_workers[failed_gpu].exitcode} without reporting success"
            )
        if msg is not None:
            return RuntimeError(f"GPU worker {msg['gpu_id']} failed:\n{msg['error']}")
        return RuntimeError("unreachable")

    finished: set[int] = set()
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

    errors: list[str] = []
    for gpu_id, p in gpu_workers.items():
        p.join(30)
        if p.is_alive():
            p.terminate()
            p.join(10)
            if p.is_alive():
                p.kill()
                p.join(5)
            errors.append(
                f"GPU worker {gpu_id} (PID {p.pid}) did not terminate after reporting success"
            )
        elif p.exitcode != 0:
            errors.append(
                f"GPU worker {gpu_id} (PID {p.pid}) exited with code "
                f"{p.exitcode} after reporting success"
            )
    if errors:
        raise RuntimeError("\n".join(errors))


def _is_run_complete(arm: str, seed: int) -> bool:
    run_dir = OUTPUT_DIR / f"{arm}_seed{seed}"
    if not _is_training_complete(arm, seed):
        return False
    cfg = RunConfig(arm=arm, seed=seed, output_dir=run_dir)
    spec = ARM_SPECS[arm]
    if not _eval_metadata_matches(run_dir / "eval_metadata.json", cfg, spec):
        return False
    required_outputs = [
        "probe_results.csv",
        "checkpoint_diagnostics.csv",
        "corruption_diagnostics.csv",
        "covariance_spectra.json",
    ]
    for name in required_outputs:
        if not (run_dir / name).exists():
            return False
    return True


def _is_training_complete(arm: str, seed: int) -> bool:
    run_dir = OUTPUT_DIR / f"{arm}_seed{seed}"
    if not run_dir.exists():
        return False
    run_meta_path = run_dir / "run_metadata.json"
    if not run_meta_path.exists():
        return False
    if not (run_dir / "train_history.json").exists():
        return False
    for step in CHECKPOINT_STEPS:
        if not (run_dir / f"checkpoint_step_{step}.pt").exists():
            return False
    try:
        run_meta = json.loads(run_meta_path.read_text())
        if run_meta.get("schema_version") != RUN_SCHEMA_VERSION:
            print(
                f"  Training schema mismatch: {arm} seed={seed} has schema "
                f"{run_meta.get('schema_version')}, current={RUN_SCHEMA_VERSION}",
                flush=True,
            )
            return False
        if int(run_meta.get("fixed_updates", -1)) != FIXED_UPDATES:
            return False
        if list(run_meta.get("checkpoint_steps", [])) != CHECKPOINT_STEPS:
            return False
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False
    return True


def main() -> None:
    load_env_file(Path(".env"))
    if not torch.cuda.is_available():
        raise RuntimeError("[10] requires CUDA")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DRY_RUN:
        if not EUROCROPS_ZARR.exists() or not EUROCROPS_LABELS_CSV.exists() or not EUROCROPS_SUMMARY.exists():
            print("EuroCropsML processed store not found. Building from preprocess files...", flush=True)
            build_eurocropsml_zarr(
                preprocess_dir=Path("data/eurocropsml/preprocess"),
                output_zarr=EUROCROPS_ZARR,
                output_labels=EUROCROPS_LABELS_CSV,
                output_summary=EUROCROPS_SUMMARY,
            )
            print("EuroCropsML processed store built.", flush=True)
        _validate_eurocrops_store()

    manifest = [
        {
            "arm": arm,
            "seed": seed,
            "data_source": ARM_SPECS[arm].data_source,
            "encoder": ARM_SPECS[arm].encoder,
            "fixed_updates": FIXED_UPDATES,
            "checkpoint_steps": json.dumps(CHECKPOINT_STEPS),
            "run_dir": str(OUTPUT_DIR / f"{arm}_seed{seed}"),
            "status": "planned",
        }
        for arm in DEFAULT_ARMS
        for seed in SEEDS
    ]
    write_csv(OUTPUT_DIR / "run_manifest.csv", manifest)
    write_json(
        OUTPUT_DIR / "metadata.json",
        {
            "experiment": "[10] Matched-Control Content-Sensitive JEPA With External Transfer",
            "arms": DEFAULT_ARMS,
            "arm_specs": {name: asdict(spec) for name, spec in ARM_SPECS.items()},
            "seeds": SEEDS,
            "fixed_updates": FIXED_UPDATES,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "cropharvest_conditions": DEFAULT_CONDITIONS,
            "eurocrops_conditions": EUROCROPS_CONDITIONS,
            "eurocrops_status": "configured" if EUROCROPS_ZARR.exists() and EUROCROPS_LABELS_CSV.exists() else "skipped_missing_processed_store",
            "schema_version": RUN_SCHEMA_VERSION,
            "gpu_assignments": GPU_ASSIGNMENTS,
        },
    )
    if DRY_RUN:
        return

    _validate_gpu_setup(DEFAULT_ARMS)
    print("\n=== [10] training phase ===", flush=True)
    _run_workers(DEFAULT_ARMS, "train")
    print("\n=== [10] evaluation phase ===", flush=True)
    _run_workers(DEFAULT_ARMS, "eval")

    all_probe_rows: list[dict[str, Any]] = []
    for arm in DEFAULT_ARMS:
        for seed in SEEDS:
            run_dir = OUTPUT_DIR / f"{arm}_seed{seed}"
            probe_path = run_dir / "probe_results.csv"
            if probe_path.exists():
                with probe_path.open("r", newline="", encoding="utf-8") as handle:
                    all_probe_rows.extend(list(csv.DictReader(handle)))

    write_csv(OUTPUT_DIR / "probe_results.csv", all_probe_rows)
    crop_rows = [row for row in all_probe_rows if "f1" in row]
    euro_rows = [row for row in all_probe_rows if "macro_f1" in row]
    write_csv(
        OUTPUT_DIR / "probe_summary.csv",
        summarize_rows(
            crop_rows,
            ["arm", "benchmark", "model", "holdout", "condition", "label_budget", "robustness_protocol", "checkpoint_role"],
        ),
    )
    if euro_rows:
        write_csv(
            OUTPUT_DIR / "eurocrops_probe_summary.csv",
            summarize_rows(
                euro_rows,
                ["arm", "benchmark", "model", "holdout", "condition", "label_budget", "robustness_protocol", "checkpoint_role"],
                metrics=[
                    "macro_f1", "balanced_accuracy",
                    "effective_label_budget", "n_test_unseen_dropped",
                    "n_classes_test_unseen", "n_train_sub",
                ],
            ),
        )
    priority = _priority_rows(all_probe_rows)
    write_csv(
        OUTPUT_DIR / "priority_summary.csv",
        summarize_rows(priority, ["arm", "model", "condition", "checkpoint_role"]),
    )
    write_csv(
        OUTPUT_DIR / "per_holdout_priority_summary.csv",
        summarize_rows(priority, ["arm", "model", "holdout", "condition", "checkpoint_role"]),
    )
    print(f"\nDone. Results in {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
