"""Evaluate trained CropHarvest JEPA checkpoints on dataset-held-out probes."""

import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets.dataset import MultimodalPatchDataset
from evals.evaluation import (
    CONDITIONS,
    ExperimentConfig,
    extract_embeddings,
    extract_raw_features,
    extract_raw_stats,
    load_labels,
    run_probes,
    valid_cropharvest_files,
)
from utils.io_utils import write_csv, summarize_rows
from core.jepa import TemporalJepaModel


DEFAULT_HOLDOUTS = ["rwanda-ceo", "togo", "ethiopia", "lem-brazil", "togo-eval"]
RUN_DIRS: list[Path] = []
SELECTED_HOLDOUTS: list[str] = []
SELECTED_CONDITIONS = ["clean"]
BATCH_SIZE: int | None = None
NUM_WORKERS = 4
DEVICE = "cuda"
INCLUDE_RAW_BASELINE = False
INCLUDE_RAW_STATS_BASELINE = False


def _load_run_config(run_dir: Path, data_args: Any) -> ExperimentConfig:
    metadata = json.loads((run_dir / "metadata.json").read_text())
    raw = metadata["config"]
    raw.update(
        {
            "zarr_path": data_args.zarr_path,
            "arrays_dir": data_args.arrays_dir,
            "labels_geojson": data_args.labels_geojson,
            "output_dir": run_dir,
            "device": data_args.device,
        }
    )
    names = {f.name for f in fields(ExperimentConfig)}
    filtered = {k: v for k, v in raw.items() if k in names}
    return ExperimentConfig(**filtered)


def _load_model(run_dir: Path, cfg: ExperimentConfig, dataset: MultimodalPatchDataset, device: torch.device) -> TemporalJepaModel:
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
    ckpt = torch.load(run_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _parse_run_identity(run_dir: Path) -> tuple[str, int]:
    match = re.match(r"(.+)_seed(\d+)$", run_dir.name)
    if match is None:
        return run_dir.name, -1
    return match.group(1), int(match.group(2))





def evaluate_run(
    run_dir: Path,
    dataset: MultimodalPatchDataset,
    y: np.ndarray,
    groups: np.ndarray,
    holdouts: list[str],
    conditions: list[tuple[str, str, float]],
    args: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = _load_run_config(run_dir, args)
    cfg.batch_size = args.batch_size or cfg.batch_size
    cfg.num_workers = args.num_workers
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = _load_model(run_dir, cfg, dataset, device)
    config_name, seed = _parse_run_identity(run_dir)

    rows: list[dict[str, Any]] = []
    for holdout in holdouts:
        test_idx = np.where(groups == holdout)[0]
        train_idx = np.where(groups != holdout)[0]
        if len(test_idx) == 0:
            continue
        if len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
            continue

        y_train = y[train_idx]
        y_test = y[test_idx]
        for condition_name, sensor_off, temporal_drop in conditions:
            x_train = extract_embeddings(
                model=model,
                dataset=dataset,
                indices=train_idx,
                device=device,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                sensor_off=sensor_off,
                temporal_drop_fraction=temporal_drop,
                seed=seed if seed >= 0 else cfg.seed,
            )
            x_test = extract_embeddings(
                model=model,
                dataset=dataset,
                indices=test_idx,
                device=device,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                sensor_off=sensor_off,
                temporal_drop_fraction=temporal_drop,
                seed=(seed if seed >= 0 else cfg.seed) + 999,
            )
            before = len(rows)
            run_probes(rows, "surf_jepa_v0", x_train, x_test, y_train, y_test, condition_name, cfg.seed)
            for row in rows[before:]:
                row["config"] = config_name
                row["seed"] = seed
                row["holdout"] = holdout

            if args.include_raw_stats_baseline:
                stats_train = extract_raw_stats(
                    dataset=dataset,
                    indices=train_idx,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    sensor_off=sensor_off,
                    temporal_drop_fraction=temporal_drop,
                    seed=cfg.seed,
                )
                stats_test = extract_raw_stats(
                    dataset=dataset,
                    indices=test_idx,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    sensor_off=sensor_off,
                    temporal_drop_fraction=temporal_drop,
                    seed=cfg.seed + 999,
                )
                before = len(rows)
                run_probes(rows, "raw_stats", stats_train, stats_test, y_train, y_test, condition_name, cfg.seed)
                for row in rows[before:]:
                    row["config"] = config_name
                    row["seed"] = seed
                    row["holdout"] = holdout
                before = len(rows)
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
                for row in rows[before:]:
                    row["config"] = config_name
                    row["seed"] = seed
                    row["holdout"] = holdout

            if args.include_raw_baseline:
                raw_train = extract_raw_features(
                    dataset=dataset,
                    indices=train_idx,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    sensor_off=sensor_off,
                    temporal_drop_fraction=temporal_drop,
                    seed=cfg.seed,
                )
                raw_test = extract_raw_features(
                    dataset=dataset,
                    indices=test_idx,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    sensor_off=sensor_off,
                    temporal_drop_fraction=temporal_drop,
                    seed=cfg.seed + 999,
                )
                before = len(rows)
                run_probes(rows, "raw_flattened", raw_train, raw_test, y_train, y_test, condition_name, cfg.seed)
                for row in rows[before:]:
                    row["config"] = config_name
                    row["seed"] = seed
                    row["holdout"] = holdout

    return rows, summarize_rows(rows, ["config", "model", "holdout", "condition"])


def main() -> None:
    args = type(
        "EmbeddedArgs",
        (),
        {
            "zarr_path": "data/cropharvest/processed",
            "arrays_dir": "data/cropharvest/raw/features/arrays",
            "labels_geojson": "data/cropharvest/raw/labels.geojson",
            "runs_root": Path("artifacts/[2]/cropharvest_jepa_v2_confirm_generalization"),
            "run_dir": RUN_DIRS,
            "output_dir": Path("artifacts/[2]"),
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "device": DEVICE,
            "include_raw_baseline": INCLUDE_RAW_BASELINE,
            "include_raw_stats_baseline": INCLUDE_RAW_STATS_BASELINE,
        },
    )()

    holdouts = SELECTED_HOLDOUTS or DEFAULT_HOLDOUTS
    requested_conditions = SELECTED_CONDITIONS or ["clean"]
    condition_map = {name: (name, sensor, drop) for name, sensor, drop in CONDITIONS}
    conditions = [condition_map[name] for name in requested_conditions]

    run_dirs = list(args.run_dir)
    if args.runs_root is not None:
        run_dirs.extend(sorted(p for p in args.runs_root.glob("*_seed*") if (p / "best_checkpoint.pt").exists()))
    if not run_dirs:
        raise ValueError('Set RUN_DIRS or Path("artifacts/[2]/cropharvest_jepa_v2_confirm_generalization") inside this file.')

    dataset = MultimodalPatchDataset(args.zarr_path, device="cpu")
    valid_files = valid_cropharvest_files(args.arrays_dir, dataset.shapes.timesteps)
    if len(valid_files) != len(dataset):
        raise ValueError(f"Valid H5 count {len(valid_files)} does not match Zarr length {len(dataset)}")
    y, groups = load_labels(valid_files, args.labels_geojson)

    all_rows: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows, summary = evaluate_run(
            run_dir=run_dir,
            dataset=dataset,
            y=y,
            groups=groups,
            holdouts=holdouts,
            conditions=conditions,
            args=args,
        )
        write_csv(run_dir / "grouped_holdout_probe_results.csv", rows)
        write_csv(run_dir / "grouped_holdout_probe_summary.csv", summary)
        all_rows.extend(rows)
        all_summary.extend(summary)
        print(json.dumps({"run_dir": str(run_dir), "rows": len(rows), "summary_rows": len(summary)}), flush=True)

    write_csv(args.output_dir / "grouped_holdout_probe_results.csv", all_rows)
    write_csv(args.output_dir / "grouped_holdout_probe_summary.csv", all_summary)
    print(
        json.dumps(
            {
                "runs": len(run_dirs),
                "rows": len(all_rows),
                "summary_rows": len(all_summary),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


main()
