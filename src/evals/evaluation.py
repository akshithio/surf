"""Shared CropHarvest split, feature extraction, and probe utilities."""

import inspect
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _thread_var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_thread_var, "1")

import h5py
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from src.core.jepa import JepaBatchMasks, TemporalJepaModel
from src.datasets.dataset import MultimodalPatchDataset


@dataclass
class ExperimentConfig:
    zarr_path: Path
    arrays_dir: Path
    labels_geojson: Path
    output_dir: Path
    seed: int = 42
    batch_size: int = 2048
    num_workers: int = 4
    epochs: int = 100
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    model_dim: int = 256
    encoder_hidden: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    use_doy: bool = True
    modality_dropout_p: float = 0.25
    s2_blackout_max_p: float = 0.30
    sample_s2_dropout_p: float = 0.0
    temporal_drop_max_fraction: float = 0.50
    ema_base: float = 0.996
    ema_final: float = 0.9995
    early_stopping_patience: int = 0
    device: str = "cuda"


BUDGETS = [0.01, 0.05, 0.10, 0.25, 1.00]
CONDITIONS = [
    ("clean", "none", 0.0),
    ("sensor_off_s2", "s2", 0.0),
    ("sensor_off_s1", "s1", 0.0),
    ("sensor_off_climate", "climate", 0.0),
    ("temporal_drop_30", "none", 0.3),
    ("temporal_drop_50", "none", 0.5),
    ("temporal_drop_70", "none", 0.7),
    ("s2_off_tdrop50", "s2", 0.5),
    ("s1_off_tdrop50", "s1", 0.5),
]
RAW_STAT_S2_IDXS = [3, 4, 5, 6, 7, 8, 9]
PROBE_SOLVER = "liblinear"
PROBE_MAX_ITER = 20000
PROBE_TOL = 1e-5


def _loader_options(
    num_workers: int,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    persistent_workers: bool = False,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        options["prefetch_factor"] = prefetch_factor
        options["persistent_workers"] = persistent_workers
    return options


def load_array_shape(path: Path) -> tuple[int, int] | None:
    try:
        with h5py.File(path, "r") as f:
            if "array" not in f:
                return None
            arr = f["array"]
            if len(arr.shape) != 2:
                return None
            return int(arr.shape[0]), int(arr.shape[1])
    except OSError:
        return None


def valid_cropharvest_files(arrays_dir: Path, expected_timesteps: int) -> list[Path]:
    files = sorted(p for p in arrays_dir.glob("*.h5") if p.is_file())
    valid: list[Path] = []
    for path in files:
        shape = load_array_shape(path)
        if shape is None:
            continue
        timesteps, channels = shape
        if timesteps == expected_timesteps and channels >= 15:
            valid.append(path)
    return valid


def load_labels(valid_files: list[Path], labels_geojson: Path) -> tuple[np.ndarray, np.ndarray]:
    geo = json.loads(labels_geojson.read_text())
    label_map: dict[tuple[int, str], int] = {}
    group_map: dict[tuple[int, str], str] = {}
    for feat in geo["features"]:
        prop = feat["properties"]
        key = (int(prop["index"]), str(prop["dataset"]))
        label_map[key] = int(prop["is_crop"])
        group_map[key] = str(prop["dataset"])

    labels: list[int] = []
    groups: list[str] = []
    for path in valid_files:
        idx_str, dataset = path.stem.split("_", 1)
        key = (int(idx_str), dataset)
        labels.append(label_map[key])
        groups.append(group_map[key])
    return np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=object)


def make_splits(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_val, test = train_test_split(
        idx,
        test_size=0.10,
        random_state=seed,
        stratify=y,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.1111111111,
        random_state=seed + 1,
        stratify=y[train_val],
    )
    return np.sort(train), np.sort(val), np.sort(test)


def make_strict_holdout_splits(
    y: np.ndarray,
    groups: np.ndarray,
    heldout_group: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    test = idx[groups == heldout_group]
    train_val = idx[groups != heldout_group]
    if len(test) == 0:
        raise ValueError(f"No samples found for strict holdout group: {heldout_group}")
    if len(np.unique(y[test])) < 2:
        raise ValueError(f"Strict holdout group is one-class: {heldout_group}")
    if len(np.unique(y[train_val])) < 2:
        raise ValueError(f"Strict holdout training pool is one-class after excluding: {heldout_group}")
    train, val = train_test_split(
        train_val,
        test_size=0.10,
        random_state=seed,
        stratify=y[train_val],
    )
    return np.sort(train), np.sort(val), np.sort(test), np.sort(train_val)


def prepare_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def condition_time_keep(
    batch_size: int,
    timesteps: int,
    drop_fraction: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor | None:
    if drop_fraction <= 0.0:
        return None
    keep_prob = max(0.0, min(1.0, 1.0 - drop_fraction))
    keep = torch.bernoulli(
        torch.full((batch_size, timesteps), keep_prob, dtype=torch.float32, device=device),
        generator=generator,
    )
    keep[:, 0] = 1.0
    low = keep.sum(dim=1) < 2.0
    if low.any():
        keep[low, 1] = 1.0
    return keep


def apply_sensor_off(batch: dict[str, torch.Tensor], sensor_off: str) -> None:
    if sensor_off == "none":
        return
    if sensor_off == "s2":
        batch["s2"].zero_()
        batch["s2_mask"].zero_()
        batch["s2_available"].zero_()
    elif sensor_off == "s1":
        batch["s1"].zero_()
        batch["s1_mask"].zero_()
        batch["s1_available"].zero_()
    elif sensor_off == "climate":
        batch["climate"].zero_()
        batch["climate_mask"].zero_()
        batch["climate_available"].zero_()
    else:
        raise ValueError(f"Unknown sensor_off={sensor_off}")


def supported_encode_inputs(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    signature = inspect.signature(model.encode)
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    allowed = {
        "s2",
        "s1",
        "climate",
        "doy",
        "s2_available",
        "s1_available",
        "climate_available",
        "s2_mask",
        "s1_mask",
        "climate_mask",
        "s2_doy",
        "s1_doy",
        "s2_elapsed_days",
        "s1_elapsed_days",
    }
    if not accepts_kwargs:
        allowed &= set(signature.parameters)
    return {key: batch[key] for key in allowed if key in batch}


def extract_embeddings(
    model: TemporalJepaModel,
    dataset: MultimodalPatchDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    sensor_off: str,
    temporal_drop_fraction: float,
    seed: int,
) -> np.ndarray:
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        **_loader_options(num_workers, pin_memory=(device.type == "cuda")),
    )
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = prepare_batch(batch, device)
            apply_sensor_off(batch, sensor_off)
            base_keep = (batch["s2_available"] + batch["s1_available"] + batch["climate_available"]) > 0
            keep = condition_time_keep(
                batch_size=batch["s2"].shape[0],
                timesteps=batch["s2"].shape[1],
                drop_fraction=temporal_drop_fraction,
                device=device,
                generator=rng,
            )
            if keep is not None:
                keep = keep * base_keep.float()
            else:
                keep = base_keep.float()
            z = model.encode(
                **supported_encode_inputs(model, batch),
                masks=JepaBatchMasks(time_keep=keep),
            )
            w = keep.unsqueeze(-1)
            emb = (z * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
            chunks.append(emb.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def apply_numpy_condition(
    batch: dict[str, torch.Tensor],
    sensor_off: str,
    temporal_drop_fraction: float,
    rng: np.random.Generator,
) -> torch.Tensor | None:
    if sensor_off == "s2":
        batch["s2"].zero_()
        batch["s2_mask"].zero_()
        batch["s2_available"].zero_()
    elif sensor_off == "s1":
        batch["s1"].zero_()
        batch["s1_mask"].zero_()
        batch["s1_available"].zero_()
    elif sensor_off == "climate":
        batch["climate"].zero_()
        batch["climate_mask"].zero_()
        batch["climate_available"].zero_()
    elif sensor_off != "none":
        raise ValueError(f"Unknown sensor_off={sensor_off}")
    if temporal_drop_fraction > 0.0:
        b, t = batch["s2"].shape[:2]
        keep = rng.binomial(1, 1.0 - temporal_drop_fraction, size=(b, t)).astype(np.float32)
        keep[:, 0] = 1.0
        low = keep.sum(axis=1) < 2
        keep[low, 1] = 1.0
        keep_t = torch.from_numpy(keep).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        batch["s2"] *= keep_t
        batch["s1"] *= keep_t
        batch["climate"] *= keep_t
        return torch.from_numpy(keep)  # (B, T)
    return None


def extract_raw_features(
    dataset: MultimodalPatchDataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    sensor_off: str,
    temporal_drop_fraction: float,
    seed: int,
) -> np.ndarray:
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        **_loader_options(num_workers),
    )
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for batch in loader:
        apply_numpy_condition(batch, sensor_off, temporal_drop_fraction, rng)
        x = torch.cat(
            [
                batch["s2"].flatten(start_dim=1),
                batch["s1"].flatten(start_dim=1),
                batch["climate"].flatten(start_dim=1),
            ],
            dim=1,
        )
        chunks.append(x.numpy())
    return np.concatenate(chunks, axis=0)


def extract_raw_stats(
    dataset: MultimodalPatchDataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    sensor_off: str,
    temporal_drop_fraction: float,
    seed: int,
) -> np.ndarray:
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        **_loader_options(num_workers),
    )
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for batch in loader:
        temporal_keep = apply_numpy_condition(batch, sensor_off, temporal_drop_fraction, rng)
        s2_keep = batch["s2_available"].bool()
        s1_keep = batch["s1_available"].bool()
        if temporal_keep is not None:
            keep = temporal_keep.bool()
            s2_keep &= keep
            s1_keep &= keep
        s2 = batch["s2"].squeeze(-1).squeeze(-1)
        s1 = batch["s1"].squeeze(-1).squeeze(-1)
        s2_masked = torch.where(s2_keep.unsqueeze(-1), s2, torch.nan)
        s1_masked = torch.where(s1_keep.unsqueeze(-1), s1, torch.nan)
        red = s2_masked[:, :, 2]
        nir = s2_masked[:, :, 6]
        denom = nir + red
        ndvi = torch.where(
            denom.abs() > 1e-6,
            (nir - red) / denom,
            torch.full_like(nir, torch.nan),
        )
        values = torch.cat([s2_masked[:, :, RAW_STAT_S2_IDXS], ndvi.unsqueeze(-1), s1_masked], dim=-1)
        stats = torch.cat(
            [
                torch.nanmean(values, dim=1),
                torch.where(torch.isfinite(values), values, torch.full_like(values, float("inf"))).amin(dim=1),
                torch.where(torch.isfinite(values), values, torch.full_like(values, float("-inf"))).amax(dim=1),
                torch.nanquantile(values, 0.10, dim=1),
                torch.nanquantile(values, 0.90, dim=1),
            ],
            dim=1,
        )
        stats = torch.where(torch.isfinite(stats), stats, torch.zeros_like(stats))
        chunks.append(stats.numpy())
    return np.concatenate(chunks, axis=0)


def subset_indices(y: np.ndarray, budget: float, seed: int) -> np.ndarray:
    idx = np.arange(len(y))
    if budget >= 1.0:
        return idx
    sub, _ = train_test_split(
        idx,
        train_size=budget,
        random_state=seed,
        stratify=y,
    )
    return np.sort(sub)


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    candidates = np.unique(prob)
    if len(candidates) > 256:
        candidates = np.quantile(prob, np.linspace(0.01, 0.99, 199))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        pred = (prob >= threshold).astype(np.int64)
        score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def fit_probe_with_calibration(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[Any, float, int, int, dict[str, Any]]:
    idx = np.arange(len(y_train))
    if len(idx) >= 20 and len(np.unique(y_train)) == 2 and min(np.bincount(y_train)) >= 4:
        fit_idx, cal_idx = train_test_split(
            idx,
            test_size=0.20,
            random_state=seed,
            stratify=y_train,
        )
    else:
        fit_idx = idx
        cal_idx = idx
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=PROBE_MAX_ITER,
            class_weight="balanced",
            solver=PROBE_SOLVER,
            tol=PROBE_TOL,
            random_state=seed,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x_train[fit_idx], y_train[fit_idx])
    convergence_warnings = [warning for warning in caught if issubclass(warning.category, ConvergenceWarning)]
    logistic = clf.named_steps["logisticregression"]
    n_iter = int(np.max(logistic.n_iter_)) if hasattr(logistic, "n_iter_") else -1
    probe_meta = {
        "probe_solver": PROBE_SOLVER,
        "probe_max_iter": PROBE_MAX_ITER,
        "probe_tol": PROBE_TOL,
        "probe_n_iter": n_iter,
        "probe_converged": int(len(convergence_warnings) == 0),
        "probe_convergence_warnings": len(convergence_warnings),
        "probe_warning_message": str(convergence_warnings[0].message) if convergence_warnings else "",
    }
    cal_prob = clf.predict_proba(x_train[cal_idx])[:, 1]
    threshold = best_f1_threshold(y_train[cal_idx], cal_prob)
    return clf, threshold, int(len(fit_idx)), int(len(cal_idx)), probe_meta


def run_probes(
    rows: list[dict[str, Any]],
    model_name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    condition: str,
    seed: int,
) -> None:
    for budget in BUDGETS:
        sub = subset_indices(y_train, budget, seed + int(round(budget * 1000)))
        clf, threshold, n_probe_fit, n_probe_calibration, probe_meta = fit_probe_with_calibration(
            x_train[sub],
            y_train[sub],
            seed + int(round(budget * 1000)) + 17,
        )
        pred = clf.predict(x_test)
        prob = clf.predict_proba(x_test)[:, 1]
        calibrated_pred = (prob >= threshold).astype(np.int64)
        rows.append(
            {
                "model": model_name,
                "condition": condition,
                "label_budget": budget,
                "n_train_sub": int(len(sub)),
                "n_probe_fit": n_probe_fit,
                "n_probe_calibration": n_probe_calibration,
                "n_test": int(len(y_test)),
                "threshold_source": "source_validation",
                "threshold": threshold,
                **probe_meta,
                "f1": float(f1_score(y_test, pred)),
                "auc": float(roc_auc_score(y_test, prob)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "calibrated_f1": float(f1_score(y_test, calibrated_pred, zero_division=0)),
                "calibrated_balanced_accuracy": float(balanced_accuracy_score(y_test, calibrated_pred)),
            }
        )
