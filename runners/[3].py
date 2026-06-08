"""Run external frozen-embedding baselines on strict CropHarvest heldouts."""

import csv
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.dataset import MultimodalPatchDataset
from evals.evaluation import CONDITIONS, load_labels, make_strict_holdout_splits, run_probes, valid_cropharvest_files
from utils.io_utils import write_csv, summarize_rows


DEFAULT_HOLDOUTS = ["rwanda-ceo", "togo", "togo-eval", "ethiopia", "lem-brazil"]
DEFAULT_CONDITIONS = ["clean", "sensor_off_s2", "temporal_drop_50", "temporal_drop_70", "s2_off_tdrop50"]
BASELINE = "presto"
SELECTED_HOLDOUTS: list[str] = []
SELECTED_CONDITIONS: list[str] = []
SEED = 42
BATCH_SIZE = 512
NUM_WORKERS = 2
DEVICE = "cuda"
OLMO_MODEL_ID = "OLMOEARTH_V1_BASE"


@dataclass
class LabelTable:
    labels: np.ndarray
    groups: np.ndarray
    lats: np.ndarray
    lons: np.ndarray


class CropHarvestH5Dataset(Dataset):
    def __init__(self, files: list[Path], label_table: LabelTable) -> None:
        self.files = files
        self.label_table = label_table

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        with h5py.File(self.files[idx], "r") as f:
            arr = np.asarray(f["array"], dtype=np.float32)
        return {
            "array": torch.from_numpy(np.nan_to_num(arr, nan=0.0)),
            "latlon": torch.tensor(
                [self.label_table.lats[idx], self.label_table.lons[idx]],
                dtype=torch.float32,
            ),
        }


def _load_label_table(valid_files: list[Path], labels_geojson: Path) -> LabelTable:
    geo = json.loads(labels_geojson.read_text())
    props_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for feat in geo["features"]:
        prop = feat["properties"]
        props_by_key[(int(prop["index"]), str(prop["dataset"]))] = prop

    labels: list[int] = []
    groups: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    for path in valid_files:
        idx_str, dataset = path.stem.split("_", 1)
        prop = props_by_key[(int(idx_str), dataset)]
        labels.append(int(prop["is_crop"]))
        groups.append(str(prop["dataset"]))
        lats.append(float(prop["lat"]))
        lons.append(float(prop["lon"]))
    return LabelTable(
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=object),
        lats=np.asarray(lats, dtype=np.float32),
        lons=np.asarray(lons, dtype=np.float32),
    )


def _condition_mask(
    batch_size: int,
    timesteps: int,
    sensor_off: str,
    temporal_drop_fraction: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    keep = torch.ones((batch_size, timesteps), dtype=torch.float32, device=device)
    if temporal_drop_fraction > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        keep = torch.bernoulli(
            torch.full((batch_size, timesteps), 1.0 - temporal_drop_fraction, device=device),
            generator=generator,
        )
        keep[:, 0] = 1.0
        low = keep.sum(dim=1) < 2.0
        if low.any():
            keep[low, 1] = 1.0
    return keep





def _assert_condition_sensitive(
    baseline: str,
    holdout: str,
    condition: str,
    clean_train: np.ndarray,
    clean_test: np.ndarray,
    stressed_train: np.ndarray,
    stressed_test: np.ndarray,
) -> None:
    if np.allclose(clean_train, stressed_train) and np.allclose(clean_test, stressed_test):
        raise RuntimeError(
            "Stress condition produced embeddings identical to clean embeddings: "
            f"baseline={baseline}, holdout={holdout}, condition={condition}. "
            "Treat this condition as invalid until the extraction path is fixed."
        )


class PrestoExtractor:
    name = "presto"

    def __init__(self, device: torch.device) -> None:
        from presto import Presto
        from presto.dataops.pipelines.dynamicworld import DynamicWorld2020_2021
        from presto.dataops.pipelines.s1_s2_era5_srtm import BANDS_GROUPS_IDX, NORMED_BANDS, S1_S2_ERA5_SRTM

        self.device = device
        self.model = Presto.load_pretrained().encoder.to(device)
        self.model.eval()
        self.unknown_dynamic_world = int(DynamicWorld2020_2021.class_amount)
        self.normalize = S1_S2_ERA5_SRTM.normalize
        self.normed_bands = list(NORMED_BANDS)
        self.s2_indices = [
            i
            for group, idxs in BANDS_GROUPS_IDX.items()
            if group.startswith("S2") or group == "NDVI"
            for i in idxs
        ]
        self.s1_indices = list(BANDS_GROUPS_IDX["S1"])
        self.climate_indices = list(BANDS_GROUPS_IDX["ERA5"])

    def _batch_to_presto(
        self,
        arr: torch.Tensor,
        latlon: torch.Tensor,
        sensor_off: str,
        temporal_drop_fraction: float,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_np = arr.cpu().numpy()
        x_np = self.normalize(x_np)
        x = torch.from_numpy(x_np).to(self.device).float()
        mask = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        if sensor_off == "s2":
            mask[:, :, self.s2_indices] = 1.0
        elif sensor_off == "s1":
            mask[:, :, self.s1_indices] = 1.0
        elif sensor_off == "climate":
            mask[:, :, self.climate_indices] = 1.0
        elif sensor_off != "none":
            raise ValueError(sensor_off)
        keep = _condition_mask(x.shape[0], x.shape[1], sensor_off, temporal_drop_fraction, seed, self.device)
        mask = torch.where(keep[:, :, None] > 0, mask, torch.ones_like(mask))
        dynamic_world = torch.full(
            (x.shape[0], x.shape[1]),
            self.unknown_dynamic_world,
            dtype=torch.long,
            device=self.device,
        )
        return x, dynamic_world, latlon.to(self.device).float(), mask

    @torch.no_grad()
    def encode_loader(
        self,
        loader: DataLoader,
        sensor_off: str,
        temporal_drop_fraction: float,
        seed: int,
    ) -> np.ndarray:
        chunks = []
        for step, batch in enumerate(loader):
            x, dynamic_world, latlon, mask = self._batch_to_presto(
                batch["array"],
                batch["latlon"],
                sensor_off=sensor_off,
                temporal_drop_fraction=temporal_drop_fraction,
                seed=seed + step,
            )
            emb = self.model(
                x,
                dynamic_world=dynamic_world,
                latlons=latlon,
                mask=mask,
                month=1,
                eval_task=True,
            )
            chunks.append(emb.cpu().numpy())
        return np.concatenate(chunks, axis=0)


class OlmoEarthExtractor:
    name = "olmoearth"

    def __init__(self, device: torch.device, model_id: str) -> None:
        import enum
        import typing

        if not hasattr(typing, "TypeAlias"):
            typing.TypeAlias = object
        if not hasattr(enum, "StrEnum"):
            class StrEnum(str, enum.Enum):
                pass

            enum.StrEnum = StrEnum

        try:
            import torch.distributed.tensor as distributed_tensor
        except Exception:
            distributed_tensor = None
        if distributed_tensor is not None and not hasattr(distributed_tensor, "distribute_tensor"):
            distributed_tensor.distribute_tensor = lambda tensor, *_args, **_kwargs: tensor

        from olmoearth_pretrain.data.constants import Modality
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
        from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

        self.device = device
        self.Modality = Modality
        self.MaskedOlmoEarthSample = MaskedOlmoEarthSample
        self.MaskValue = MaskValue
        self.normalizer = Normalizer(Strategy.COMPUTED)
        selected = getattr(ModelID, model_id)
        self.model = load_model_from_id(selected).to(device)
        self.model.eval()

        # CropHarvest array layout:
        # [VV, VH, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, ERA5..., SRTM..., NDVI]
        self.s2_source = {
            "B02": 2,
            "B03": 3,
            "B04": 4,
            "B05": 5,
            "B06": 6,
            "B07": 7,
            "B08": 8,
            "B8A": 9,
            "B09": 10,
            "B11": 11,
            "B12": 12,
        }

    def _make_sample(
        self,
        arr: torch.Tensor,
        sensor_off: str,
        temporal_drop_fraction: float,
        seed: int,
    ):
        np_arr = arr.cpu().numpy()
        b, t, _ = np_arr.shape
        s2_order = self.Modality.SENTINEL2_L2A.band_order
        s2 = np.zeros((b, 1, 1, t, len(s2_order)), dtype=np.float32)
        for out_idx, band in enumerate(s2_order):
            source_idx = self.s2_source.get(band)
            if source_idx is not None:
                s2[:, 0, 0, :, out_idx] = np_arr[:, :, source_idx]
        s1 = np.zeros((b, 1, 1, t, 2), dtype=np.float32)
        s1[:, 0, 0, :, 0] = np_arr[:, :, 0]
        s1[:, 0, 0, :, 1] = np_arr[:, :, 1]
        s2 = self.normalizer.normalize(self.Modality.SENTINEL2_L2A, s2)
        s1 = self.normalizer.normalize(self.Modality.SENTINEL1, s1)

        online = float(self.MaskValue.ONLINE_ENCODER.value)
        missing = float(self.MaskValue.MISSING.value)
        s2_mask = np.full((b, 1, 1, t, self.Modality.SENTINEL2_L2A.num_band_sets), online, dtype=np.float32)
        s1_mask = np.full((b, 1, 1, t, self.Modality.SENTINEL1.num_band_sets), online, dtype=np.float32)
        if sensor_off == "s2":
            s2_mask[:] = missing
        elif sensor_off == "s1":
            s1_mask[:] = missing
        elif sensor_off not in {"none", "climate"}:
            raise ValueError(sensor_off)
        keep = _condition_mask(b, t, sensor_off, temporal_drop_fraction, seed, torch.device("cpu")).numpy()
        s2_mask = np.where(keep[:, None, None, :, None] > 0, s2_mask, missing)
        s1_mask = np.where(keep[:, None, None, :, None] > 0, s1_mask, missing)

        timestamps = np.zeros((b, t, 3), dtype=np.int64)
        timestamps[:, :, 0] = 15
        timestamps[:, :, 1] = np.arange(t, dtype=np.int64)[None, :]
        timestamps[:, :, 2] = 2020
        return self.MaskedOlmoEarthSample(
            sentinel2_l2a=torch.tensor(s2, dtype=torch.float32, device=self.device),
            sentinel2_l2a_mask=torch.tensor(s2_mask, dtype=torch.float32, device=self.device),
            sentinel1=torch.tensor(s1, dtype=torch.float32, device=self.device),
            sentinel1_mask=torch.tensor(s1_mask, dtype=torch.float32, device=self.device),
            timestamps=torch.tensor(timestamps, dtype=torch.long, device=self.device),
        )

    def _pool_modality(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        online = mask == float(self.MaskValue.ONLINE_ENCODER.value)
        while online.ndim > 2 and int(np.prod(online.shape[1:])) != tokens.shape[1]:
            online = online.any(dim=-1)
        online = online.reshape(online.shape[0], -1)
        if online.shape[1] != tokens.shape[1]:
            raise ValueError(f"Token/mask shape mismatch: tokens={tuple(tokens.shape)}, mask={tuple(online.shape)}")
        weighted = tokens * online.unsqueeze(-1).to(tokens.dtype)
        counts = online.sum(dim=1, keepdim=True)
        pooled = weighted.sum(dim=1) / counts.clamp_min(1).to(tokens.dtype)
        return torch.where(counts > 0, pooled, torch.zeros_like(pooled))

    @torch.no_grad()
    def encode_loader(
        self,
        loader: DataLoader,
        sensor_off: str,
        temporal_drop_fraction: float,
        seed: int,
    ) -> np.ndarray:
        chunks = []
        for step, batch in enumerate(loader):
            sample = self._make_sample(
                batch["array"],
                sensor_off=sensor_off,
                temporal_drop_fraction=temporal_drop_fraction,
                seed=seed + step,
            )
            output = self.model.encoder(sample, fast_pass=False, patch_size=1)["tokens_and_masks"]
            pieces = []
            for attr in ["sentinel2_l2a", "sentinel1"]:
                value = getattr(output, attr, None)
                mask = getattr(output, output.get_masked_modality_name(attr), None)
                if value is not None:
                    if mask is None:
                        raise ValueError(f"Missing mask for OlmoEarth modality {attr}")
                    pieces.append(self._pool_modality(value, mask))
            if not pieces:
                raise RuntimeError("OlmoEarth produced no usable modality features")
            chunks.append(torch.cat(pieces, dim=-1).cpu().numpy())
        return np.concatenate(chunks, axis=0)


def _build_extractor(name: str, device: torch.device, olmo_model_id: str):
    if name == "presto":
        return PrestoExtractor(device)
    if name == "olmoearth":
        return OlmoEarthExtractor(device, olmo_model_id)
    raise ValueError(name)


def _evaluate_baseline(
    baseline: str,
    extractor_factory: Callable[[], Any],
    dataset: CropHarvestH5Dataset,
    y: np.ndarray,
    groups: np.ndarray,
    holdouts: list[str],
    conditions: list[str],
    batch_size: int,
    num_workers: int,
    seed: int,
    output_dir: Path,
) -> None:
    condition_map = {name: (sensor, drop) for name, sensor, drop in CONDITIONS}
    rows: list[dict[str, Any]] = []
    extractor = extractor_factory()
    for holdout in holdouts:
        _, _, test_idx, probe_train_idx = make_strict_holdout_splits(y, groups, holdout, seed)
        y_train = y[probe_train_idx]
        y_test = y[test_idx]
        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, probe_train_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            torch.utils.data.Subset(dataset, test_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        clean_train = extractor.encode_loader(train_loader, "none", 0.0, seed)
        clean_test = extractor.encode_loader(test_loader, "none", 0.0, seed + 999)
        for condition in conditions:
            sensor_off, temporal_drop = condition_map[condition]
            if condition == "clean":
                x_train = clean_train
                x_test = clean_test
            else:
                x_train = extractor.encode_loader(train_loader, sensor_off, temporal_drop, seed)
                x_test = extractor.encode_loader(test_loader, sensor_off, temporal_drop, seed + 999)
                _assert_condition_sensitive(
                    baseline=baseline,
                    holdout=holdout,
                    condition=condition,
                    clean_train=clean_train,
                    clean_test=clean_test,
                    stressed_train=x_train,
                    stressed_test=x_test,
                )
            before = len(rows)
            run_probes(rows, baseline, x_train, x_test, y_train, y_test, condition, seed)
            for row in rows[before:]:
                row["baseline"] = baseline
                row["holdout"] = holdout
                row["seed"] = seed
        write_csv(output_dir / baseline / f"{holdout}_partial.csv", rows)
    write_csv(output_dir / baseline / "probe_results.csv", rows)
    write_csv(output_dir / baseline / "probe_summary.csv", summarize_rows(rows, ["baseline", "holdout", "condition"]))


def main() -> None:
    if BASELINE not in {"presto", "olmoearth"}:
        raise ValueError(f"Unknown BASELINE: {BASELINE}")
    Path("artifacts/[5]").mkdir(parents=True, exist_ok=True)
    status_path = Path("artifacts/[5]") / BASELINE / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataset_for_shape = MultimodalPatchDataset("data/cropharvest/processed", device="cpu")
        valid_files = valid_cropharvest_files("data/cropharvest/raw/features/arrays", dataset_for_shape.shapes.timesteps)
        if len(valid_files) != len(dataset_for_shape):
            raise ValueError(f"Valid H5 count {len(valid_files)} does not match Zarr length {len(dataset_for_shape)}")
        label_table = _load_label_table(valid_files, "data/cropharvest/raw/labels.geojson")
        # Keep this sanity call because it is the shared label/split code used by the JEPA runner.
        labels_check, groups_check = load_labels(valid_files, "data/cropharvest/raw/labels.geojson")
        if not (np.array_equal(label_table.labels, labels_check) and np.array_equal(label_table.groups, groups_check)):
            raise ValueError("Label table mismatch")
        dataset = CropHarvestH5Dataset(valid_files, label_table)
        holdouts = SELECTED_HOLDOUTS or DEFAULT_HOLDOUTS
        conditions = SELECTED_CONDITIONS or DEFAULT_CONDITIONS
        device = torch.device(DEVICE if (DEVICE != "cuda" or torch.cuda.is_available()) else "cpu")
        _evaluate_baseline(
            baseline=BASELINE,
            extractor_factory=lambda: _build_extractor(BASELINE, device, OLMO_MODEL_ID),
            dataset=dataset,
            y=label_table.labels,
            groups=label_table.groups,
            holdouts=holdouts,
            conditions=conditions,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            seed=SEED,
            output_dir=Path("artifacts/[5]"),
        )
        status_path.write_text(json.dumps({"status": "completed", "baseline": BASELINE}, indent=2))
    except Exception as exc:
        status_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "baseline": BASELINE,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "python": sys.version,
                },
                indent=2,
            )
        )
        raise


main()
