"""Run strict heldout hybrid probes over existing JEPA checkpoints."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets.dataset import MultimodalPatchDataset
from evals.evaluation import (
    CONDITIONS,
    PROBE_MAX_ITER,
    PROBE_SOLVER,
    PROBE_TOL,
    extract_embeddings,
    extract_raw_features,
    extract_raw_stats,
    load_labels,
    make_strict_holdout_splits,
    run_probes,
    valid_cropharvest_files,
)
from utils.io_utils import read_json, write_csv, summarize_rows
from core.jepa import TemporalJepaModel

DEFAULT_CONFIGS = ["large_default", "large_dual_s2"]
DEFAULT_HOLDOUTS = ["rwanda-ceo", "togo", "togo-eval", "ethiopia", "lem-brazil"]
DEFAULT_SEEDS = [7, 11, 42]
DEFAULT_CONDITIONS = ["clean", "sensor_off_s2", "temporal_drop_50", "temporal_drop_70", "s2_off_tdrop50"]
SELECTED_CONFIGS: list[str] = []
SELECTED_HOLDOUTS: list[str] = []
SELECTED_SEEDS: list[int] = []
SELECTED_CONDITIONS: list[str] = []
BATCH_SIZE = 512
NUM_WORKERS = 4
DEVICE = "cuda"
SKIP_RAW_FLATTENED = False
DRY_RUN = False
METRICS = ["f1", "auc", "balanced_accuracy", "calibrated_f1", "calibrated_balanced_accuracy"]


@dataclass(frozen=True)
class CheckpointSpec:
    config: str
    holdout: str
    seed: int
    run_dir: Path
    checkpoint_path: Path
    metadata: dict[str, Any]


def _config_from_run_name(name: str, configs: list[str]) -> str | None:
    for config in sorted(configs, key=len, reverse=True):
        if name.startswith(f"{config}_"):
            return config
    return None


def _metadata_seed(metadata: dict[str, Any]) -> int:
    config = metadata.get("config", {})
    return int(config.get("seed", metadata.get("seed", -1)))


def discover_checkpoints(
    checkpoint_root: Path,
    configs: list[str],
    holdouts: list[str],
    seeds: list[int],
) -> tuple[list[CheckpointSpec], list[dict[str, Any]]]:
    specs: list[CheckpointSpec] = []
    manifest: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for checkpoint_path in sorted(checkpoint_root.glob("**/best_checkpoint.pt")):
        run_dir = checkpoint_path.parent
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = read_json(metadata_path)
        config_name = _config_from_run_name(run_dir.name, configs)
        if config_name is None:
            continue
        holdout = metadata.get("strict_holdout_group")
        seed = _metadata_seed(metadata)
        if holdout not in holdouts or seed not in seeds:
            continue
        if metadata.get("strict_holdout_group") != holdout:
            raise ValueError(f"Checkpoint metadata is missing strict holdout for {run_dir}")
        key = (config_name, str(holdout), int(seed))
        if key in seen:
            raise ValueError(f"Duplicate checkpoint for {key}")
        seen.add(key)
        specs.append(
            CheckpointSpec(
                config=config_name,
                holdout=str(holdout),
                seed=int(seed),
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                metadata=metadata,
            )
        )
    for config in configs:
        for holdout in holdouts:
            for seed in seeds:
                status = "found" if (config, holdout, seed) in seen else "missing"
                match = next((s for s in specs if (s.config, s.holdout, s.seed) == (config, holdout, seed)), None)
                manifest.append(
                    {
                        "config": config,
                        "holdout": holdout,
                        "seed": seed,
                        "status": status,
                        "run_dir": str(match.run_dir) if match is not None else "",
                        "checkpoint_path": str(match.checkpoint_path) if match is not None else "",
                    }
                )
    return specs, manifest


def _load_model(spec: CheckpointSpec, dataset: MultimodalPatchDataset, device: torch.device) -> TemporalJepaModel:
    checkpoint = torch.load(spec.checkpoint_path, map_location=device)
    config = checkpoint.get("config", spec.metadata.get("config", {}))
    model = TemporalJepaModel(
        s2_channels=dataset.shapes.s2_channels,
        s1_channels=dataset.shapes.s1_channels,
        climate_channels=dataset.shapes.climate_channels,
        model_dim=int(config.get("model_dim", 768)),
        encoder_hidden=int(config.get("encoder_hidden", 384)),
        num_layers=int(config.get("num_layers", 8)),
        num_heads=int(config.get("num_heads", 12)),
        dropout=float(config.get("dropout", 0.1)),
        use_doy=bool(config.get("use_doy", True)),
        ema_momentum=float(config.get("ema_base", 0.996)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _append_probe_rows(
    rows: list[dict[str, Any]],
    arm: str,
    spec: CheckpointSpec,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    condition: str,
) -> None:
    before = len(rows)
    run_probes(rows, arm, x_train, x_test, y_train, y_test, condition, spec.seed)
    for row in rows[before:]:
        row["experiment"] = "strict_hybrid_raw_cue_probe"
        row["config"] = spec.config
        row["arm"] = arm
        row["holdout"] = spec.holdout
        row["seed"] = spec.seed
        row["run_dir"] = str(spec.run_dir)
        row["checkpoint_path"] = str(spec.checkpoint_path)


def _condition_map(selected: list[str]) -> list[tuple[str, str, float]]:
    available = {name: (sensor, drop) for name, sensor, drop in CONDITIONS}
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Unknown conditions: {missing}")
    return [(name, *available[name]) for name in selected]


def main() -> None:
    configs = SELECTED_CONFIGS or DEFAULT_CONFIGS
    holdouts = SELECTED_HOLDOUTS or DEFAULT_HOLDOUTS
    seeds = SELECTED_SEEDS or DEFAULT_SEEDS
    conditions = SELECTED_CONDITIONS or DEFAULT_CONDITIONS
    Path("artifacts/[6]").mkdir(parents=True, exist_ok=True)

    specs, manifest = discover_checkpoints(Path("artifacts/[4]/strict"), configs, holdouts, seeds)
    write_csv(Path("artifacts/[6]") / "checkpoint_manifest.csv", manifest)
    missing = [row for row in manifest if row["status"] != "found"]
    if missing:
        write_csv(Path("artifacts/[6]") / "missing_checkpoints.csv", missing)
        raise ValueError(f"Missing {len(missing)} strict checkpoints; see missing_checkpoints.csv")
    if DRY_RUN:
        return

    device = torch.device(DEVICE if (DEVICE != "cuda" or torch.cuda.is_available()) else "cpu")
    dataset = MultimodalPatchDataset("data/cropharvest/processed", device="cpu")
    valid_files = valid_cropharvest_files("data/cropharvest/raw/features/arrays", dataset.shapes.timesteps)
    if len(valid_files) != len(dataset):
        raise ValueError(f"Valid H5 count {len(valid_files)} does not match Zarr length {len(dataset)}")
    y, groups = load_labels(valid_files, "data/cropharvest/raw/labels.geojson")
    selected_conditions = _condition_map(conditions)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.metadata.get("strict_holdout_group") != spec.holdout:
            raise ValueError(f"Refusing mismatched checkpoint/eval holdout for {spec.run_dir}")
        _, _, test_idx, probe_train_idx = make_strict_holdout_splits(y, groups, spec.holdout, spec.seed)
        y_train = y[probe_train_idx]
        y_test = y[test_idx]
        model = _load_model(spec, dataset, device)
        for condition_name, sensor_off, temporal_drop in selected_conditions:
            x_train = extract_embeddings(
                model=model,
                dataset=dataset,
                indices=probe_train_idx,
                device=device,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                sensor_off=sensor_off,
                temporal_drop_fraction=temporal_drop,
                seed=spec.seed,
            )
            x_test = extract_embeddings(
                model=model,
                dataset=dataset,
                indices=test_idx,
                device=device,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                sensor_off=sensor_off,
                temporal_drop_fraction=temporal_drop,
                seed=spec.seed + 999,
            )
            stats_train = extract_raw_stats(
                dataset=dataset,
                indices=probe_train_idx,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                sensor_off=sensor_off,
                temporal_drop_fraction=temporal_drop,
                seed=spec.seed,
            )
            stats_test = extract_raw_stats(
                dataset=dataset,
                indices=test_idx,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                sensor_off=sensor_off,
                temporal_drop_fraction=temporal_drop,
                seed=spec.seed + 999,
            )
            _append_probe_rows(rows, "surf_jepa_v0", spec, x_train, x_test, y_train, y_test, condition_name)
            _append_probe_rows(rows, "raw_stats", spec, stats_train, stats_test, y_train, y_test, condition_name)
            _append_probe_rows(
                rows,
                "surf_jepa_v0_plus_raw_stats",
                spec,
                np.concatenate([x_train, stats_train], axis=1),
                np.concatenate([x_test, stats_test], axis=1),
                y_train,
                y_test,
                condition_name,
            )
            if not SKIP_RAW_FLATTENED:
                raw_train = extract_raw_features(
                    dataset=dataset,
                    indices=probe_train_idx,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    sensor_off=sensor_off,
                    temporal_drop_fraction=temporal_drop,
                    seed=spec.seed,
                )
                raw_test = extract_raw_features(
                    dataset=dataset,
                    indices=test_idx,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    sensor_off=sensor_off,
                    temporal_drop_fraction=temporal_drop,
                    seed=spec.seed + 999,
                )
                _append_probe_rows(rows, "raw_flattened", spec, raw_train, raw_test, y_train, y_test, condition_name)
            write_csv(Path("artifacts/[6]") / "probe_results_partial.csv", rows)

    write_csv(Path("artifacts/[6]") / "probe_results.csv", rows)
    write_csv(Path("artifacts/[6]") / "probe_summary.csv", summarize_rows(rows, ["config", "arm", "holdout", "condition"]))
    write_csv(Path("artifacts/[6]") / "priority_summary.csv", summarize_rows(rows, ["config", "arm"]))
    write_csv(Path("artifacts/[6]") / "per_holdout_priority_summary.csv", summarize_rows(rows, ["config", "arm", "holdout"]))
    lem_rows = [row for row in rows if row["holdout"] == "lem-brazil"]
    write_csv(Path("artifacts/[6]") / "lem_brazil_summary.csv", summarize_rows(lem_rows, ["config", "arm", "condition"]))
    write_csv(Path("artifacts/[6]") / "lem_brazil_priority_summary.csv", summarize_rows(lem_rows, ["config", "arm"]))
    write_csv(Path("artifacts/[6]") / "label_budget_curves.csv", summarize_rows(rows, ["config", "arm", "condition", "label_budget"]))
    metadata = {
        "experiment": "[6] Strict Hybrid Raw-Cue Probe",
        "checkpoint_root": str(Path("artifacts/[4]/strict")),
        "configs": configs,
        "holdouts": holdouts,
        "seeds": seeds,
        "conditions": conditions,
        "arms": ["surf_jepa_v0", "raw_stats", "surf_jepa_v0_plus_raw_stats"]
        + ([] if SKIP_RAW_FLATTENED else ["raw_flattened"]),
        "strict_checkpoint_holdout_equals_eval_holdout": True,
        "probe_solver": PROBE_SOLVER,
        "probe_max_iter": PROBE_MAX_ITER,
        "probe_tol": PROBE_TOL,
        "num_checkpoints": len(specs),
        "num_rows": len(rows),
    }
    (Path("artifacts/[6]") / "metadata.json").write_text(json.dumps(metadata, indent=2))


main()
