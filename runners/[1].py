"""Run full CropHarvest temporal JEPA pretraining and frozen-probe evaluation."""

import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.dataset import MultimodalPatchDataset
from evals.evaluation import (
    CONDITIONS,
    ExperimentConfig,
    extract_embeddings,
    extract_raw_features,
    extract_raw_stats,
    load_labels,
    make_splits,
    make_strict_holdout_splits,
    prepare_batch,
    run_probes,
    valid_cropharvest_files,
)
from core.jepa import (
    JepaBatchMasks,
    TemporalJepaModel,
    cosine_ema_momentum,
    jepa_cosine_loss,
)

SEED = 42
BATCH_SIZE = 2048
NUM_WORKERS = 4
EPOCHS = 100
LR = 3e-4
MIN_LR = 1e-5
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 0.05
MODEL_DIM = 256
ENCODER_HIDDEN = 128
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.1
USE_DOY = True
MODALITY_DROPOUT_P = 0.25
S2_BLACKOUT_MAX_P = 0.30
SAMPLE_S2_DROPOUT_P = 0.0
TEMPORAL_DROP_MAX_FRACTION = 0.50
EMA_BASE = 0.996
EMA_FINAL = 0.9995
EARLY_STOPPING_PATIENCE = 0
DEVICE = "cuda"
SKIP_RAW_BASELINE = False
INCLUDE_RAW_STATS_BASELINE = False
STRICT_HOLDOUT_GROUP = None
SELECTED_CONDITIONS: list[str] = []


def _jsonable_config(cfg: ExperimentConfig) -> dict[str, Any]:
    out = asdict(cfg)
    return {k: str(v) if isinstance(v, Path) else v for k, v in out.items()}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_modality_keep(
    batch: dict[str, torch.Tensor],
    modality_dropout_p: float,
    s2_blackout_p: float,
    sample_s2_dropout_p: float,
) -> torch.Tensor | None:
    if modality_dropout_p <= 0.0 and s2_blackout_p <= 0.0 and sample_s2_dropout_p <= 0.0:
        return None
    availability = torch.stack(
        [batch["s2_available"], batch["s1_available"], batch["climate_available"]],
        dim=-1,
    )
    keep = torch.ones_like(availability)
    if modality_dropout_p > 0.0:
        keep = torch.bernoulli(torch.full_like(availability, 1.0 - modality_dropout_p))
    if s2_blackout_p > 0.0:
        other_available = (availability[..., 1] + availability[..., 2]) > 0
        s2_drop = torch.bernoulli(torch.full_like(batch["s2_available"], s2_blackout_p)) > 0
        keep[..., 0] = torch.where(other_available & s2_drop, torch.zeros_like(keep[..., 0]), keep[..., 0])
    if sample_s2_dropout_p > 0.0:
        sample_has_backup = ((availability[..., 1] + availability[..., 2]) > 0).any(dim=1)
        sample_drop = (
            torch.bernoulli(
                torch.full(
                    (availability.shape[0],),
                    sample_s2_dropout_p,
                    dtype=availability.dtype,
                    device=availability.device,
                )
            )
            > 0
        )
        sample_drop = sample_drop & sample_has_backup
        keep[..., 0] = torch.where(
            sample_drop.unsqueeze(1),
            torch.zeros_like(keep[..., 0]),
            keep[..., 0],
        )

    no_modality = (keep * availability).sum(dim=-1, keepdim=True) <= 0
    keep = torch.where(no_modality, torch.ones_like(keep), keep)
    return keep


def _build_time_keep(
    batch_size: int,
    timesteps: int,
    max_drop_fraction: float,
    device: torch.device,
) -> torch.Tensor | None:
    if max_drop_fraction <= 0.0:
        return None
    sampled_drop = float(torch.rand((), device=device).item()) * max_drop_fraction
    keep_prob = max(0.0, min(1.0, 1.0 - sampled_drop))
    keep = torch.bernoulli(torch.full((batch_size, timesteps), keep_prob, dtype=torch.float32, device=device))
    keep[:, 0] = 1.0
    low = keep.sum(dim=1) < 2.0
    if low.any():
        keep[low, 1] = 1.0
    return keep


def lr_lambda_for_step(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(float(step + 1) / float(warmup_steps), min_lr_ratio)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def run_epoch(
    model: TemporalJepaModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    device: torch.device,
    cfg: ExperimentConfig,
    global_step: int,
    total_steps: int,
) -> tuple[float, int]:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    count = 0
    for batch in loader:
        batch = prepare_batch(batch, device)
        masks = None
        if train:
            optimizer.zero_grad(set_to_none=True)
            masks = JepaBatchMasks(
                modality_keep=_build_modality_keep(
                    batch=batch,
                    modality_dropout_p=cfg.modality_dropout_p,
                    s2_blackout_p=cfg.s2_blackout_max_p,
                    sample_s2_dropout_p=cfg.sample_s2_dropout_p,
                ),
                time_keep=_build_time_keep(
                    batch_size=batch["s2"].shape[0],
                    timesteps=batch["s2"].shape[1],
                    max_drop_fraction=cfg.temporal_drop_max_fraction,
                    device=device,
                ),
            )

        pred, target = model(
            s2=batch["s2"],
            s1=batch["s1"],
            climate=batch["climate"],
            doy=batch["doy"],
            s2_available=batch["s2_available"],
            s1_available=batch["s1_available"],
            climate_available=batch["climate_available"],
            masks=masks,
        )
        loss = jepa_cosine_loss(pred, target)
        if torch.isnan(loss):
            raise FloatingPointError("NaN loss")
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            model.update_target_encoder(cosine_ema_momentum(global_step, total_steps, cfg.ema_base, cfg.ema_final))
            global_step += 1

        value = float(loss.item())
        total += value
        count += 1
    return total / max(count, 1), global_step


def train_jepa(
    cfg: ExperimentConfig,
    dataset: MultimodalPatchDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
) -> tuple[TemporalJepaModel, dict[str, Any]]:
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    model = TemporalJepaModel(
        s2_channels=dataset.shapes.s2_channels,
        s1_channels=dataset.shapes.s1_channels,
        climate_channels=dataset.shapes.climate_channels,
        model_dim=cfg.model_dim,
        encoder_hidden=cfg.encoder_hidden,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        use_doy=cfg.use_doy,
        ema_momentum=cfg.ema_base,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.epochs * max(len(train_loader), 1)
    warmup_steps = cfg.warmup_epochs * max(len(train_loader), 1)
    min_lr_ratio = cfg.min_lr / cfg.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda_for_step(step, total_steps, warmup_steps, min_lr_ratio),
    )

    best_val = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    global_step = 0
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_loss, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            global_step=global_step,
            total_steps=total_steps,
        )
        with torch.no_grad():
            val_loss, global_step = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                scheduler=None,
                device=device,
                cfg=cfg,
                global_step=global_step,
                total_steps=total_steps,
            )
        lr = float(optimizer.param_groups[0]["lr"])
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": _jsonable_config(cfg),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                cfg.output_dir / "best_checkpoint.pt",
            )
        else:
            epochs_without_improvement += 1
            if cfg.early_stopping_patience > 0 and epochs_without_improvement >= cfg.early_stopping_patience:
                print(
                    json.dumps(
                        {
                            "early_stop": True,
                            "epoch": epoch,
                            "best_val_loss": best_val,
                            "patience": cfg.early_stopping_patience,
                        }
                    ),
                    flush=True,
                )
                break

    summary = {
        "best_val_loss": best_val,
        "train_seconds": time.time() - start,
        "history": history,
    }
    (cfg.output_dir / "train_history.json").write_text(json.dumps(summary, indent=2))
    ckpt = torch.load(cfg.output_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, summary


def main() -> None:
    cfg = ExperimentConfig(
        zarr_path="data/cropharvest/processed",
        arrays_dir="data/cropharvest/raw/features/arrays",
        labels_geojson="data/cropharvest/raw/labels.geojson",
        output_dir=Path("artifacts/[1]"),
        seed=SEED,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        epochs=EPOCHS,
        lr=LR,
        min_lr=MIN_LR,
        warmup_epochs=WARMUP_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        model_dim=MODEL_DIM,
        encoder_hidden=ENCODER_HIDDEN,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        use_doy=USE_DOY,
        modality_dropout_p=MODALITY_DROPOUT_P,
        s2_blackout_max_p=S2_BLACKOUT_MAX_P,
        sample_s2_dropout_p=SAMPLE_S2_DROPOUT_P,
        temporal_drop_max_fraction=TEMPORAL_DROP_MAX_FRACTION,
        ema_base=EMA_BASE,
        ema_final=EMA_FINAL,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        device=DEVICE,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    dataset = MultimodalPatchDataset(cfg.zarr_path, device="cpu")
    valid_files = valid_cropharvest_files(cfg.arrays_dir, dataset.shapes.timesteps)
    if len(valid_files) != len(dataset):
        raise ValueError(f"Valid H5 count {len(valid_files)} does not match Zarr length {len(dataset)}")
    y, groups = load_labels(valid_files, cfg.labels_geojson)
    strict_holdout = STRICT_HOLDOUT_GROUP
    if strict_holdout:
        train_idx, val_idx, test_idx, probe_train_idx = make_strict_holdout_splits(
            y=y,
            groups=groups,
            heldout_group=strict_holdout,
            seed=cfg.seed,
        )
    else:
        train_idx, val_idx, test_idx = make_splits(y, cfg.seed)
        probe_train_idx = train_idx
    np.savez(
        cfg.output_dir / "splits.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        probe_train_idx=probe_train_idx,
        y=y,
        groups=groups,
    )
    metadata = {
        "config": _jsonable_config(cfg),
        "num_samples": int(len(dataset)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "num_probe_train": int(len(probe_train_idx)),
        "strict_holdout_group": strict_holdout,
        "class_balance": {"crop": int(y.sum()), "non_crop": int((1 - y).sum())},
        "test_class_balance": {
            "crop": int(y[test_idx].sum()),
            "non_crop": int((1 - y[test_idx]).sum()),
        },
        "groups": {str(k): int(v) for k, v in zip(*np.unique(groups, return_counts=True))},
    }
    (cfg.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)

    model, train_summary = train_jepa(cfg, dataset, train_idx, val_idx, device)
    (cfg.output_dir / "train_summary.json").write_text(json.dumps(train_summary, indent=2))

    y_train = y[probe_train_idx]
    y_test = y[test_idx]
    rows: list[dict[str, Any]] = []
    condition_map = {name: (name, sensor, drop) for name, sensor, drop in CONDITIONS}
    selected_conditions = [condition_map[name] for name in (SELECTED_CONDITIONS or list(condition_map))]
    for condition_name, sensor_off, temporal_drop in selected_conditions:
        x_train = extract_embeddings(
            model,
            dataset,
            probe_train_idx,
            device,
            cfg.batch_size,
            cfg.num_workers,
            sensor_off,
            temporal_drop,
            cfg.seed,
        )
        x_test = extract_embeddings(
            model,
            dataset,
            test_idx,
            device,
            cfg.batch_size,
            cfg.num_workers,
            sensor_off,
            temporal_drop,
            cfg.seed + 999,
        )
        run_probes(rows, "surf_jepa_v0", x_train, x_test, y_train, y_test, condition_name, cfg.seed)

        if INCLUDE_RAW_STATS_BASELINE:
            stats_train = extract_raw_stats(
                dataset,
                probe_train_idx,
                cfg.batch_size,
                cfg.num_workers,
                sensor_off,
                temporal_drop,
                cfg.seed,
            )
            stats_test = extract_raw_stats(
                dataset,
                test_idx,
                cfg.batch_size,
                cfg.num_workers,
                sensor_off,
                temporal_drop,
                cfg.seed + 999,
            )
            run_probes(rows, "raw_stats", stats_train, stats_test, y_train, y_test, condition_name, cfg.seed)
            run_probes(
                rows,
                "surf_jepa_v0_plus_raw_stats",
                np.concatenate([x_train, stats_train], axis=1),
                np.concatenate([x_test, stats_test], axis=1),
                y_train,
                y_test,
                condition_name,
                cfg.seed,
            )

        if not SKIP_RAW_BASELINE:
            raw_train = extract_raw_features(
                dataset,
                probe_train_idx,
                cfg.batch_size,
                cfg.num_workers,
                sensor_off,
                temporal_drop,
                cfg.seed,
            )
            raw_test = extract_raw_features(
                dataset,
                test_idx,
                cfg.batch_size,
                cfg.num_workers,
                sensor_off,
                temporal_drop,
                cfg.seed + 999,
            )
            run_probes(rows, "raw_flattened", raw_train, raw_test, y_train, y_test, condition_name, cfg.seed)

        with (cfg.output_dir / "probe_results_partial.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    results_csv = cfg.output_dir / "probe_results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        summary.setdefault((str(row["model"]), str(row["condition"])), []).append(row)
    summary_rows = []
    for (model_name, condition), vals in sorted(summary.items()):
        row = {
            "model": model_name,
            "condition": condition,
            "mean_f1": float(np.mean([float(v["f1"]) for v in vals])),
            "mean_auc": float(np.mean([float(v["auc"]) for v in vals])),
            "mean_balanced_accuracy": float(np.mean([float(v["balanced_accuracy"]) for v in vals])),
            "n_budgets": len(vals),
        }
        if "calibrated_f1" in vals[0]:
            row["mean_calibrated_f1"] = float(np.mean([float(v["calibrated_f1"]) for v in vals]))
        if "calibrated_balanced_accuracy" in vals[0]:
            row["mean_calibrated_balanced_accuracy"] = float(
                np.mean([float(v["calibrated_balanced_accuracy"]) for v in vals])
            )
        if "probe_converged" in vals[0]:
            row["all_probes_converged"] = int(all(int(v["probe_converged"]) == 1 for v in vals))
        if "probe_convergence_warnings" in vals[0]:
            row["total_probe_convergence_warnings"] = int(sum(int(v["probe_convergence_warnings"]) for v in vals))
        if "probe_n_iter" in vals[0]:
            row["max_probe_n_iter"] = int(max(int(v["probe_n_iter"]) for v in vals))
        summary_rows.append(row)
    with (cfg.output_dir / "probe_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"results_csv": str(results_csv), "summary_rows": summary_rows}, indent=2), flush=True)


main()
