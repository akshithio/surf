"""Diagnose why LEM Brazil behaves differently under strict heldout evaluation."""

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasets.dataset import MultimodalPatchDataset
from evals.evaluation import PROBE_MAX_ITER, PROBE_SOLVER, PROBE_TOL, load_labels, make_strict_holdout_splits, valid_cropharvest_files
from utils.io_utils import write_csv


HOLDOUT = "lem-brazil"
SEED = 42


BANDS = [
    "VV",
    "VH",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
    "temperature_2m",
    "total_precipitation",
    "elevation",
    "slope",
    "NDVI",
]


def _load_label_props(labels_geojson: Path) -> dict[tuple[int, str], dict[str, Any]]:
    geo = json.loads(labels_geojson.read_text())
    return {
        (int(feat["properties"]["index"]), str(feat["properties"]["dataset"])): feat["properties"]
        for feat in geo["features"]
    }


def _load_arrays(files: list[Path]) -> np.ndarray:
    values = []
    for path in files:
        with h5py.File(path, "r") as f:
            values.append(np.nan_to_num(np.asarray(f["array"], dtype=np.float32), nan=0.0))
    return np.stack(values, axis=0)


def _file_key(path: Path) -> tuple[int, str]:
    idx_str, dataset = path.stem.split("_", 1)
    return int(idx_str), dataset


def _date_year(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value)
    if not text or text == "nan":
        return None
    try:
        return int(text[:4])
    except ValueError:
        return None


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or np.nanstd(score) == 0:
        return float("nan")
    auc = float(roc_auc_score(y, score))
    return max(auc, 1.0 - auc)


def main() -> None:
    Path("artifacts/[4]").mkdir(parents=True, exist_ok=True)
    dataset = MultimodalPatchDataset("data/cropharvest/processed", device="cpu")
    files = valid_cropharvest_files("data/cropharvest/raw/features/arrays", dataset.shapes.timesteps)
    y, groups = load_labels(files, "data/cropharvest/raw/labels.geojson")
    _, _, test_idx, probe_train_idx = make_strict_holdout_splits(y, groups, HOLDOUT, SEED)
    props = _load_label_props("data/cropharvest/raw/labels.geojson")
    arrays = _load_arrays(files)

    dataset_rows = []
    for group in sorted(np.unique(groups)):
        idx = np.where(groups == group)[0]
        group_props = [props[_file_key(files[i])] for i in idx]
        years = [_date_year(p.get("export_end_date")) for p in group_props]
        dataset_rows.append(
            {
                "dataset": str(group),
                "n": int(len(idx)),
                "crop": int(y[idx].sum()),
                "non_crop": int((1 - y[idx]).sum()),
                "crop_fraction": float(y[idx].mean()),
                "export_year_min": min(v for v in years if v is not None) if any(v is not None for v in years) else "",
                "export_year_max": max(v for v in years if v is not None) if any(v is not None for v in years) else "",
                "nan_fraction": float(np.isnan(arrays[idx]).mean()),
                "zero_fraction": float((arrays[idx] == 0).mean()),
            }
        )
    write_csv(Path("artifacts/[4]") / "dataset_summary.csv", dataset_rows)

    holdout = test_idx
    source = probe_train_idx
    separability_rows = []
    for band_idx, band in enumerate(BANDS):
        hold_values = arrays[holdout, :, band_idx]
        source_values = arrays[source, :, band_idx]
        for stat_name, reducer in {
            "mean": lambda x: x.mean(axis=1),
            "max": lambda x: x.max(axis=1),
            "min": lambda x: x.min(axis=1),
            "std": lambda x: x.std(axis=1),
        }.items():
            hold_score = reducer(hold_values)
            source_score = reducer(source_values)
            separability_rows.append(
                {
                    "band": band,
                    "stat": stat_name,
                    "lem_auc_abs": _safe_auc(y[holdout], hold_score),
                    "source_auc_abs": _safe_auc(y[source], source_score),
                    "lem_crop_mean": float(hold_score[y[holdout] == 1].mean()),
                    "lem_non_crop_mean": float(hold_score[y[holdout] == 0].mean()),
                    "source_crop_mean": float(source_score[y[source] == 1].mean()),
                    "source_non_crop_mean": float(source_score[y[source] == 0].mean()),
                }
            )
    separability_rows.sort(key=lambda r: (np.nan_to_num(float(r["lem_auc_abs"]), nan=-1.0)), reverse=True)
    write_csv(Path("artifacts/[4]") / "lem_raw_feature_separability.csv", separability_rows)

    x_train = arrays[source].reshape(len(source), -1)
    x_test = arrays[holdout].reshape(len(holdout), -1)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=PROBE_MAX_ITER,
            class_weight="balanced",
            solver=PROBE_SOLVER,
            tol=PROBE_TOL,
            random_state=SEED,
        ),
    )
    clf.fit(x_train, y[source])
    prob = clf.predict_proba(x_test)[:, 1]
    pred_rows = []
    for local_i, global_i in enumerate(holdout):
        prop = props[_file_key(files[int(global_i)])]
        pred_rows.append(
            {
                "file": files[int(global_i)].name,
                "label": int(y[int(global_i)]),
                "prob_crop_raw": float(prob[local_i]),
                "classification_label": prop.get("classification_label", ""),
                "label_text": prop.get("label", ""),
                "lat": prop.get("lat", ""),
                "lon": prop.get("lon", ""),
                "export_end_date": prop.get("export_end_date", ""),
                "collection_date": prop.get("collection_date", ""),
                "planting_date": prop.get("planting_date", ""),
                "harvest_date": prop.get("harvest_date", ""),
            }
        )
    write_csv(Path("artifacts/[4]") / "lem_raw_predictions.csv", pred_rows)

    strict_results = Path("artifacts/[4]/strict") / "analysis" / "strict_probe_results_all.csv"
    if strict_results.exists():
        rows = list(csv.DictReader(strict_results.open()))
        lem_rows = [r for r in rows if r.get("holdout") == HOLDOUT]
        write_csv(Path("artifacts/[4]") / "lem_existing_strict_results.csv", lem_rows)


main()
