"""Shared IO utilities and result-aggregation helpers for experiment runners."""

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ["f1", "auc", "balanced_accuracy", "calibrated_f1", "calibrated_balanced_accuracy"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(
    rows: list[dict[str, Any]],
    keys: list[str],
    metrics: list[str] = METRICS,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key_values, vals in sorted(grouped.items()):
        row: dict[str, Any] = dict(zip(keys, key_values))
        for metric in metrics:
            row[f"mean_{metric}"] = float(np.mean([float(v[metric]) for v in vals]))
        row["n_rows"] = len(vals)
        row["n_seeds"] = len({int(v["seed"]) for v in vals})
        row["n_holdouts"] = len({str(v["holdout"]) for v in vals})
        if "probe_converged" in vals[0]:
            row["all_probes_converged"] = int(all(int(v["probe_converged"]) == 1 for v in vals))
        if "probe_convergence_warnings" in vals[0]:
            row["total_probe_convergence_warnings"] = int(
                sum(int(v["probe_convergence_warnings"]) for v in vals)
            )
        if "probe_n_iter" in vals[0]:
            row["max_probe_n_iter"] = int(max(int(v["probe_n_iter"]) for v in vals))
        out.append(row)
    return out


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp_path.replace(path)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)
