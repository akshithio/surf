"""Run strict heldout Temporal Block-JEPA architecture ablations."""

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from datasets.cropharvest import build_cropharvest_zarr
from datasets.dataset import MultimodalPatchDataset
from evals.evaluation import (
    CONDITIONS,
    extract_embeddings,
    load_labels,
    make_strict_holdout_splits,
    run_probes,
    valid_cropharvest_files,
)
from utils.io_utils import load_env_file, write_csv, summarize_rows
from core.jepa import (
    JepaBatchMasks,
    TemporalBlockJepaModel,
    TemporalJepaModel,
    cosine_ema_momentum,
    jepa_cosine_loss,
    masked_jepa_cosine_loss,
)
from datasets.ssl4eo import build_ssl4eo_s12_zarr


DEFAULT_HOLDOUTS = ["rwanda-ceo", "togo", "togo-eval", "ethiopia", "lem-brazil"]
DEFAULT_CONDITIONS = ["clean", "sensor_off_s2", "temporal_drop_50", "temporal_drop_70", "s2_off_tdrop50"]
DEFAULT_ARMS = ["A_control", "B_full_target", "C_transformer_predictor", "D_multiblock", "E_cross_modal", "F_rawcue", "G_full"]
S2_CHANNELS_CANONICAL = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
S1_CHANNELS_CANONICAL = ["VV", "VH"]
CONTEXT_CHANNELS_MINIMAL = ["temperature", "precipitation", "elevation"]
RAW_CUE_S2_IDXS = [3, 4, 5, 6, 7, 8, 9, 10]
METRICS = ["f1", "auc", "balanced_accuracy", "calibrated_f1", "calibrated_balanced_accuracy"]
SELECTED_ARMS: list[str] = []
SELECTED_HOLDOUTS: list[str] = []
SELECTED_SEEDS = [42]
SELECTED_CONDITIONS: list[str] = []
EPOCHS = 20
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 512
NUM_WORKERS = 4
MODEL_DIM = 768
ENCODER_HIDDEN = 384
NUM_LAYERS = 8
NUM_HEADS = 12
PREDICTOR_LAYERS = 2
LR = 7e-5
EARLY_STOPPING_PATIENCE = 5
DEVICE = "cuda"
LIMIT_TRAIN_SAMPLES = 0
LIMIT_PROBE_TRAIN_SAMPLES = 0
LIMIT_TEST_SAMPLES = 0
SMOKE_MODEL = False
DRY_RUN = False

# Preprocessing / data build constants
SSL4EO_SAMPLES = 49152
SSL4EO_START_SHARD = 1
SSL4EO_MAX_INPUT_GIB = 60.0
PATCH_SIZE = 16
NDVI_MEAN_MIN = None
OVERWRITE_EXISTING = False
BUILD_CROPHARVEST = True
BUILD_SSL4EO = True
CLEAR_INCOMPLETE_DOWNLOADS = True


@dataclass(frozen=True)
class ArmSpec:
    name: str
    use_block_model: bool
    full_target_encoder: bool
    transformer_predictor: bool
    multiblock_masking: bool
    cross_modal_prediction: bool
    raw_cue_loss_weight: float


@dataclass
class RunConfig:
    pretrain_zarr_path: Path
    eval_zarr_path: Path
    arrays_dir: Path
    labels_geojson: Path
    output_dir: Path
    holdout: str
    arm: str
    seed: int = 42
    epochs: int = 20
    batch_size: int = 512
    eval_batch_size: int = 512
    num_workers: int = 4
    model_dim: int = 768
    encoder_hidden: int = 384
    num_layers: int = 8
    num_heads: int = 12
    predictor_layers: int = 2
    dropout: float = 0.1
    lr: float = 7e-5
    min_lr: float = 1e-5
    warmup_epochs: int = 2
    weight_decay: float = 0.05
    ema_base: float = 0.996
    ema_final: float = 0.9995
    modality_dropout_p: float = 0.25
    s2_blackout_max_p: float = 0.30
    sample_s2_dropout_p: float = 0.30
    temporal_drop_max_fraction: float = 0.50
    target_block_min: int = 2
    target_block_max: int = 4
    target_blocks: int = 2
    early_stopping_patience: int = 5
    device: str = "cuda"
    limit_train_samples: int = 0
    limit_probe_train_samples: int = 0
    limit_test_samples: int = 0


ARM_SPECS = {
    "A_control": ArmSpec("A_control", False, False, False, False, False, 0.0),
    "B_full_target": ArmSpec("B_full_target", True, True, False, False, False, 0.0),
    "C_transformer_predictor": ArmSpec("C_transformer_predictor", True, True, True, False, False, 0.0),
    "D_multiblock": ArmSpec("D_multiblock", True, True, True, True, False, 0.0),
    "E_cross_modal": ArmSpec("E_cross_modal", True, True, True, True, True, 0.0),
    "F_rawcue": ArmSpec("F_rawcue", True, True, True, True, False, 0.05),
    "G_full": ArmSpec("G_full", True, True, True, True, True, 0.05),
}


def _jsonable_config(cfg: RunConfig, arm_spec: ArmSpec) -> dict[str, Any]:
    out = asdict(cfg)
    out.update({"arm_spec": asdict(arm_spec)})
    return {k: str(v) if isinstance(v, Path) else v for k, v in out.items()}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _limit_indices(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def _make_pretrain_splits(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_n = max(1, int(round(0.10 * n)))
    val_idx = np.sort(idx[:val_n])
    train_idx = np.sort(idx[val_n:])
    if len(train_idx) == 0:
        raise ValueError("Pretraining dataset is too small to make a train/val split.")
    return train_idx, val_idx


def _assert_shared_channel_contract(
    pretrain_dataset: MultimodalPatchDataset,
    eval_dataset: MultimodalPatchDataset,
) -> None:
    expected = {
        "s2_bands": S2_CHANNELS_CANONICAL,
        "s1_bands": S1_CHANNELS_CANONICAL,
        "climate_bands": CONTEXT_CHANNELS_MINIMAL,
    }
    mismatches = []
    for key, expected_names in expected.items():
        for label, dataset in (("pretrain", pretrain_dataset), ("eval", eval_dataset)):
            if key not in dataset.root:
                mismatches.append(f"{label} missing {key}")
                continue
            names = _decode_zarr_names(dataset, key)
            if names != expected_names:
                mismatches.append(f"{label} {key}: expected={expected_names}, found={names}")
    if mismatches:
        raise ValueError(
            "Pretrain/eval zarr stores must share the same channel contract. "
            + "; ".join(mismatches)
        )


def _decode_zarr_names(dataset: MultimodalPatchDataset, key: str) -> list[str]:
    values = np.asarray(dataset.root[key][:])
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8").rstrip("\x00"))
        else:
            out.append(str(value))
    return out


def _decode(values: Any) -> list[str]:
    return [
        value.decode("utf-8").rstrip("\x00") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _torch_isfinite(tensor: Any) -> bool:
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    return bool(np.isfinite(arr).all())


def _assert_channel_contract(path: Path) -> dict[str, Any]:
    root = zarr.open_group(str(path), mode="r")
    expected = {
        "s2_bands": S2_CHANNELS_CANONICAL,
        "s1_bands": S1_CHANNELS_CANONICAL,
        "climate_bands": ["temperature", "precipitation", "elevation"],
    }
    summary: dict[str, Any] = {"path": str(path)}
    for array_name in ["s2", "s1", "climate", "s2_mask", "s1_mask", "climate_mask"]:
        summary[array_name] = list(root[array_name].shape)
    for key, expected_names in expected.items():
        got = _decode(np.asarray(root[key][:]))
        if got != expected_names:
            raise ValueError(f"{path} {key} mismatch: expected={expected_names}, got={got}")
    return summary


def _clear_incomplete_downloads(cache_dir: Path) -> int:
    removed = 0
    if not cache_dir.exists():
        return removed
    for p in cache_dir.rglob("*.incomplete"):
        p.unlink(missing_ok=True)
        removed += 1
    return removed


def _disk_usage_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            stat = item.lstat()
        except FileNotFoundError:
            continue
        blocks = getattr(stat, "st_blocks", 0)
        if blocks:
            total += int(blocks) * 512
        elif item.is_file():
            total += int(stat.st_size)
    return total


def _assert_input_budget(path: Path, max_gib: float) -> dict[str, Any]:
    used_bytes = _disk_usage_bytes(path)
    used_gib = used_bytes / (1024**3)
    summary = {"path": str(path), "disk_bytes": int(used_bytes), "disk_gib": used_gib, "max_gib": float(max_gib)}
    if used_gib > max_gib:
        raise ValueError(f"{path} uses {used_gib:.2f} GiB, above the {max_gib:.2f} GiB input budget")
    return summary


def _assert_dataset_samples(path: Path, expected_patch_size: int | None = None) -> dict[str, Any]:
    dataset = MultimodalPatchDataset(path)
    if len(dataset) == 0:
        raise ValueError(f"{path} is empty")
    indices = sorted(set([0, len(dataset) // 2, len(dataset) - 1]))
    for idx in indices:
        sample = dataset[idx]
        for key in ["s2", "s1", "climate", "s2_mask", "s1_mask", "climate_mask", "doy"]:
            tensor = sample[key]
            if not _torch_isfinite(tensor):
                raise ValueError(f"{path} sample {idx} has non-finite values in {key}")
    if expected_patch_size is not None:
        if dataset.shapes.patch_h != expected_patch_size or dataset.shapes.patch_w != expected_patch_size:
            raise ValueError(
                f"{path} patch mismatch: expected {expected_patch_size}x{expected_patch_size}, "
                f"got {dataset.shapes.patch_h}x{dataset.shapes.patch_w}"
            )
    return {
        "num_patches": dataset.shapes.num_patches,
        "timesteps": dataset.shapes.timesteps,
        "s2_channels": dataset.shapes.s2_channels,
        "s1_channels": dataset.shapes.s1_channels,
        "climate_channels": dataset.shapes.climate_channels,
        "patch_h": dataset.shapes.patch_h,
        "patch_w": dataset.shapes.patch_w,
    }


def _ensure_data() -> None:
    if OVERWRITE_EXISTING:
        for path in ["data/cropharvest/processed/v2.zarr", "data/processed/ssl4eo_s12_v11_48k.zarr"]:
            if path.exists():
                shutil.rmtree(path)
    "data/cropharvest/processed/v2.zarr".parent.mkdir(parents=True, exist_ok=True)
    "data/processed/ssl4eo_s12_v11_48k.zarr".parent.mkdir(parents=True, exist_ok=True)
    Path("data/cache/ssl4eo_hf_48k").mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    if CLEAR_INCOMPLETE_DOWNLOADS:
        summaries["incomplete_downloads_removed"] = _clear_incomplete_downloads(Path("data/cache/ssl4eo_hf_48k"))
    if BUILD_CROPHARVEST and not "data/cropharvest/processed/v2.zarr".exists():
        summaries["cropharvest_build"] = build_cropharvest_zarr(
            arrays_dir="data/cropharvest/raw/features/arrays",
            output_zarr="data/cropharvest/processed/v2.zarr",
            max_samples=None,
        )
    if BUILD_SSL4EO and not "data/processed/ssl4eo_s12_v11_48k.zarr".exists():
        summaries["ssl4eo_build"] = build_ssl4eo_s12_zarr(
            output_zarr="data/processed/ssl4eo_s12_v11_48k.zarr",
            max_samples=SSL4EO_SAMPLES,
            patch_size=PATCH_SIZE,
            start_shard=SSL4EO_START_SHARD,
            ndvi_mean_min=NDVI_MEAN_MIN,
            cache_dir=Path("data/cache/ssl4eo_hf_48k"),
            max_cache_gib=SSL4EO_MAX_INPUT_GIB,
            evict_cached_shards=True,
        )
        summaries["ssl4eo_input_budget"] = _assert_input_budget(Path("data/cache/ssl4eo_hf_48k"), SSL4EO_MAX_INPUT_GIB)
    summaries["cropharvest_contract"] = _assert_channel_contract("data/cropharvest/processed/v2.zarr")
    summaries["ssl4eo_contract"] = _assert_channel_contract("data/processed/ssl4eo_s12_v11_48k.zarr")
    summaries["cropharvest_dataset"] = _assert_dataset_samples("data/cropharvest/processed/v2.zarr", expected_patch_size=1)
    summaries["ssl4eo_dataset"] = _assert_dataset_samples("data/processed/ssl4eo_s12_v11_48k.zarr", expected_patch_size=PATCH_SIZE)
    out_path = Path("data/processed/preprocess_7_summary.json")
    out_path.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))
    print(f"Wrote {out_path}")


def _prepare_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _build_time_keep(batch_size: int, timesteps: int, max_drop_fraction: float, device: torch.device) -> torch.Tensor | None:
    if max_drop_fraction <= 0.0:
        return None
    sampled_drop = float(torch.rand((), device=device).item()) * max_drop_fraction
    keep_prob = max(0.0, min(1.0, 1.0 - sampled_drop))
    keep = torch.bernoulli(torch.full((batch_size, timesteps), keep_prob, device=device))
    keep[:, 0] = 1.0
    low = keep.sum(dim=1) < 2.0
    if low.any():
        keep[low, 1] = 1.0
    return keep


def _build_target_mask(
    batch_size: int,
    timesteps: int,
    cfg: RunConfig,
    arm_spec: ArmSpec,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    mask = torch.zeros((batch_size, timesteps), dtype=torch.bool, device=device)
    if timesteps <= 1:
        return mask
    if not arm_spec.multiblock_masking:
        width = min(cfg.target_block_min, timesteps - 1)
        for b in range(batch_size):
            start_high = max(2, timesteps - width + 1)
            start = int(torch.randint(1, start_high, (1,), device=device, generator=generator).item())
            mask[b, start : min(timesteps, start + width)] = True
        return mask
    for b in range(batch_size):
        for _ in range(cfg.target_blocks):
            width = int(
                torch.randint(
                    cfg.target_block_min,
                    cfg.target_block_max + 1,
                    (1,),
                    device=device,
                    generator=generator,
                ).item()
            )
            start_high = max(2, timesteps - width + 1)
            start = int(torch.randint(1, start_high, (1,), device=device, generator=generator).item())
            mask[b, start : min(timesteps, start + width)] = True
    empty = mask.sum(dim=1) == 0
    if empty.any() and timesteps > 1:
        mask[empty, 1] = True
    return mask


def _build_modality_keep(
    batch: dict[str, torch.Tensor],
    cfg: RunConfig,
    arm_spec: ArmSpec,
) -> torch.Tensor | None:
    if (
        cfg.modality_dropout_p <= 0.0
        and cfg.s2_blackout_max_p <= 0.0
        and cfg.sample_s2_dropout_p <= 0.0
        and not arm_spec.cross_modal_prediction
    ):
        return None
    availability = torch.stack(
        [batch["s2_available"], batch["s1_available"], batch["climate_available"]],
        dim=-1,
    )
    keep = torch.ones_like(availability)
    if cfg.modality_dropout_p > 0.0:
        keep = torch.bernoulli(torch.full_like(availability, 1.0 - cfg.modality_dropout_p))
    if cfg.s2_blackout_max_p > 0.0:
        other_available = (availability[..., 1] + availability[..., 2]) > 0
        s2_drop = torch.bernoulli(torch.full_like(batch["s2_available"], cfg.s2_blackout_max_p)) > 0
        keep[..., 0] = torch.where(other_available & s2_drop, torch.zeros_like(keep[..., 0]), keep[..., 0])
    if cfg.sample_s2_dropout_p > 0.0:
        sample_has_backup = ((availability[..., 1] + availability[..., 2]) > 0).any(dim=1)
        sample_drop = torch.bernoulli(
            torch.full((availability.shape[0],), cfg.sample_s2_dropout_p, dtype=availability.dtype, device=availability.device)
        ) > 0
        keep[..., 0] = torch.where(sample_drop.unsqueeze(1) & sample_has_backup.unsqueeze(1), torch.zeros_like(keep[..., 0]), keep[..., 0])
    if arm_spec.cross_modal_prediction:
        choices = torch.randint(0, 3, (availability.shape[0],), device=availability.device)
        for modality in range(3):
            drop = choices == modality
            has_backup = ((availability.sum(dim=-1) - availability[..., modality]) > 0).any(dim=1)
            keep[..., modality] = torch.where(drop.unsqueeze(1) & has_backup.unsqueeze(1), torch.zeros_like(keep[..., modality]), keep[..., modality])
    no_modality = ((keep * availability).sum(dim=-1, keepdim=True) <= 0)
    return torch.where(no_modality, torch.ones_like(keep), keep)


def _raw_cue_target(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    s2 = batch["s2"].mean(dim=(-1, -2))
    s1 = batch["s1"].mean(dim=(-1, -2))
    if s2.shape[-1] <= max(RAW_CUE_S2_IDXS):
        raise ValueError("Expected canonical S2 channels with NDVI for raw-cue targets.")
    values = torch.cat([s2[:, :, RAW_CUE_S2_IDXS], s1], dim=-1)
    return torch.cat(
        [
            values.mean(dim=1),
            values.amin(dim=1),
            values.amax(dim=1),
            values.quantile(0.10, dim=1),
            values.quantile(0.90, dim=1),
        ],
        dim=1,
    )


def _extract_raw_cue_stats(
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
        num_workers=num_workers,
    )
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for batch in loader:
        if sensor_off == "s2":
            batch["s2"].zero_()
        elif sensor_off == "s1":
            batch["s1"].zero_()
        elif sensor_off == "climate":
            batch["climate"].zero_()
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
        chunks.append(_raw_cue_target(batch).numpy())
    return np.concatenate(chunks, axis=0)


def _lr_lambda(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(float(step + 1) / float(warmup_steps), min_lr_ratio)
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _make_model(cfg: RunConfig, arm_spec: ArmSpec, dataset: MultimodalPatchDataset) -> nn.Module:
    kwargs = {
        "s2_channels": dataset.shapes.s2_channels,
        "s1_channels": dataset.shapes.s1_channels,
        "climate_channels": dataset.shapes.climate_channels,
        "model_dim": cfg.model_dim,
        "encoder_hidden": cfg.encoder_hidden,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "dropout": cfg.dropout,
        "use_doy": True,
        "ema_momentum": cfg.ema_base,
    }
    if not arm_spec.use_block_model:
        return TemporalJepaModel(**kwargs)
    return TemporalBlockJepaModel(
        **kwargs,
        predictor_layers=cfg.predictor_layers,
        full_target_encoder=arm_spec.full_target_encoder,
        transformer_predictor=arm_spec.transformer_predictor,
        raw_cue_dim=50,
    )


def _make_model_from_shapes(cfg: RunConfig, arm_spec: ArmSpec, s2_channels: int, s1_channels: int, climate_channels: int) -> nn.Module:
    kwargs = {
        "s2_channels": s2_channels,
        "s1_channels": s1_channels,
        "climate_channels": climate_channels,
        "model_dim": cfg.model_dim,
        "encoder_hidden": cfg.encoder_hidden,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "dropout": cfg.dropout,
        "use_doy": True,
        "ema_momentum": cfg.ema_base,
    }
    if not arm_spec.use_block_model:
        return TemporalJepaModel(**kwargs)
    return TemporalBlockJepaModel(
        **kwargs,
        predictor_layers=cfg.predictor_layers,
        full_target_encoder=arm_spec.full_target_encoder,
        transformer_predictor=arm_spec.transformer_predictor,
        raw_cue_dim=50,
    )


def smoke_model(arms: list[str], args: Any) -> None:
    device = torch.device("cpu")
    batch_size = 2
    timesteps = 4
    cfg = RunConfig(
        pretrain_zarr_path=Path("smoke_pretrain.zarr"),
        eval_zarr_path=Path("smoke_eval.zarr"),
        arrays_dir=Path("arrays"),
        labels_geojson=Path("labels.geojson"),
        output_dir=args.output_dir,
        holdout="smoke",
        arm="smoke",
        seed=1,
        epochs=1,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        model_dim=args.model_dim,
        encoder_hidden=args.encoder_hidden,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        predictor_layers=args.predictor_layers,
        lr=args.lr,
        early_stopping_patience=args.early_stopping_patience,
        device="cpu",
    )
    batch = {
        "s2": torch.randn(batch_size, timesteps, 11, 16, 16),
        "s1": torch.randn(batch_size, timesteps, 2, 16, 16),
        "climate": torch.randn(batch_size, timesteps, 3, 16, 16),
        "doy": torch.linspace(45, 315, timesteps).repeat(batch_size, 1),
        "s2_available": torch.ones(batch_size, timesteps),
        "s1_available": torch.ones(batch_size, timesteps),
        "climate_available": torch.ones(batch_size, timesteps),
    }
    rows = []
    for arm in arms:
        arm_spec = ARM_SPECS[arm]
        model = _make_model_from_shapes(cfg, arm_spec, 11, 2, 3).to(device)
        masks = JepaBatchMasks(
            modality_keep=_build_modality_keep(batch, cfg, arm_spec),
            time_keep=_build_time_keep(batch_size, timesteps, cfg.temporal_drop_max_fraction, device),
        )
        if arm_spec.use_block_model:
            target_mask = _build_target_mask(batch_size, timesteps, cfg, arm_spec, device)
            pred, target, raw_pred = model(**batch, target_mask=target_mask, masks=masks)
            loss = masked_jepa_cosine_loss(pred, target, target_mask)
            if arm_spec.raw_cue_loss_weight > 0:
                loss = loss + arm_spec.raw_cue_loss_weight * F.smooth_l1_loss(raw_pred, _raw_cue_target(batch))
        else:
            pred, target = model(**batch, masks=masks)
            loss = jepa_cosine_loss(pred, target)
        loss.backward()
        rows.append({"arm": arm, "loss": float(loss.detach().item()), "ok": 1})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "smoke_model.csv", rows)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    cfg: RunConfig,
    arm_spec: ArmSpec,
    device: torch.device,
    global_step: int,
    total_steps: int,
) -> tuple[float, int]:
    train = optimizer is not None
    model.train(train)
    mask_generator = None
    if not train:
        mask_generator = torch.Generator(device=device)
        mask_generator.manual_seed(cfg.seed + 9109)
    total = 0.0
    count = 0
    for batch in loader:
        batch = _prepare_batch(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        modality_keep = _build_modality_keep(batch, cfg, arm_spec) if train else None
        time_keep = _build_time_keep(batch["s2"].shape[0], batch["s2"].shape[1], cfg.temporal_drop_max_fraction, device) if train else None
        masks = JepaBatchMasks(modality_keep=modality_keep, time_keep=time_keep)
        if arm_spec.use_block_model:
            target_mask = _build_target_mask(
                batch["s2"].shape[0],
                batch["s2"].shape[1],
                cfg,
                arm_spec,
                device,
                generator=mask_generator,
            )
            pred, target, raw_pred = model(
                s2=batch["s2"],
                s1=batch["s1"],
                climate=batch["climate"],
                doy=batch["doy"],
                s2_available=batch["s2_available"],
                s1_available=batch["s1_available"],
                climate_available=batch["climate_available"],
                target_mask=target_mask,
                masks=masks,
            )
            loss = masked_jepa_cosine_loss(pred, target, target_mask)
            if arm_spec.raw_cue_loss_weight > 0.0:
                raw_target = _raw_cue_target(batch)
                loss = loss + arm_spec.raw_cue_loss_weight * F.smooth_l1_loss(raw_pred, raw_target)
        else:
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
        total += float(loss.item())
        count += 1
    return total / max(count, 1), global_step


def train_one(cfg: RunConfig, arm_spec: ArmSpec, dataset: MultimodalPatchDataset, train_idx: np.ndarray, val_idx: np.ndarray, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = _make_model(cfg, arm_spec, dataset).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(len(train_loader), 1)
    warmup_steps = cfg.warmup_epochs * max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(step, total_steps, warmup_steps, cfg.min_lr / cfg.lr),
    )
    best_val = float("inf")
    stale = 0
    global_step = 0
    history = []
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_loss, global_step = _train_epoch(model, train_loader, optimizer, scheduler, cfg, arm_spec, device, global_step, total_steps)
        with torch.no_grad():
            val_loss, global_step = _train_epoch(model, val_loader, None, None, cfg, arm_spec, device, global_step, total_steps)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": float(optimizer.param_groups[0]["lr"])}
        history.append(row)
        print(json.dumps({"arm": cfg.arm, "holdout": cfg.holdout, "seed": cfg.seed, **row}), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": _jsonable_config(cfg, arm_spec),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                cfg.output_dir / "best_checkpoint.pt",
            )
        else:
            stale += 1
            if cfg.early_stopping_patience > 0 and stale >= cfg.early_stopping_patience:
                break
    summary = {"best_val_loss": best_val, "train_seconds": time.time() - start, "history": history}
    (cfg.output_dir / "train_history.json").write_text(json.dumps(summary, indent=2))
    checkpoint = torch.load(cfg.output_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, summary


def _condition_map(selected: list[str]) -> list[tuple[str, str, float]]:
    available = {name: (sensor, drop) for name, sensor, drop in CONDITIONS}
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Unknown conditions: {missing}")
    return [(name, *available[name]) for name in selected]


def _append_probe_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    cfg: RunConfig,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    condition: str,
) -> None:
    before = len(rows)
    run_probes(rows, model_name, x_train, x_test, y_train, y_test, condition, cfg.seed)
    for row in rows[before:]:
        row["experiment"] = "temporal_block_jepa_v1"
        row["arm"] = cfg.arm
        row["holdout"] = cfg.holdout
        row["seed"] = cfg.seed
        row["run_dir"] = str(cfg.output_dir)


def evaluate_one(
    cfg: RunConfig,
    model: nn.Module,
    dataset: MultimodalPatchDataset,
    probe_train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    conditions: list[tuple[str, str, float]],
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_name, sensor_off, temporal_drop in conditions:
        x_train = extract_embeddings(model, dataset, probe_train_idx, device, cfg.eval_batch_size, cfg.num_workers, sensor_off, temporal_drop, cfg.seed)
        x_test = extract_embeddings(model, dataset, test_idx, device, cfg.eval_batch_size, cfg.num_workers, sensor_off, temporal_drop, cfg.seed + 999)
        stats_train = _extract_raw_cue_stats(dataset, probe_train_idx, cfg.eval_batch_size, cfg.num_workers, sensor_off, temporal_drop, cfg.seed)
        stats_test = _extract_raw_cue_stats(dataset, test_idx, cfg.eval_batch_size, cfg.num_workers, sensor_off, temporal_drop, cfg.seed + 999)
        _append_probe_rows(rows, "embedding", cfg, x_train, x_test, y_train, y_test, condition_name)
        _append_probe_rows(rows, "raw_stats", cfg, stats_train, stats_test, y_train, y_test, condition_name)
        _append_probe_rows(
            rows,
            "embedding_plus_raw_stats",
            cfg,
            np.concatenate([x_train, stats_train], axis=1),
            np.concatenate([x_test, stats_test], axis=1),
            y_train,
            y_test,
            condition_name,
        )
    return rows


def pretrain_one(
    cfg: RunConfig,
    arm_spec: ArmSpec,
    pretrain_dataset: MultimodalPatchDataset,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_idx, pretrain_val_idx = _make_pretrain_splits(len(pretrain_dataset), cfg.seed)
    pretrain_idx = _limit_indices(pretrain_idx, cfg.limit_train_samples, cfg.seed + 101)
    if cfg.limit_train_samples > 0:
        pretrain_val_idx = _limit_indices(
            pretrain_val_idx,
            max(1, cfg.limit_train_samples // 10),
            cfg.seed + 102,
        )
    model, train_summary = train_one(cfg, arm_spec, pretrain_dataset, pretrain_idx, pretrain_val_idx, device)
    metadata = {
        "experiment": "[7] Temporal Block-JEPA v1",
        "config": _jsonable_config(cfg, arm_spec),
        "num_pretrain": int(len(pretrain_idx)),
        "num_pretrain_val": int(len(pretrain_val_idx)),
        "train_summary": train_summary,
        "pretrain_dataset": str(cfg.pretrain_zarr_path),
        "eval_dataset": str(cfg.eval_zarr_path),
    }
    (cfg.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return model, metadata


def evaluate_holdout(
    cfg: RunConfig,
    model: nn.Module,
    eval_dataset: MultimodalPatchDataset,
    y: np.ndarray,
    groups: np.ndarray,
    conditions: list[tuple[str, str, float]],
    device: torch.device,
) -> list[dict[str, Any]]:
    _, _, test_idx, probe_train_idx = make_strict_holdout_splits(y, groups, cfg.holdout, cfg.seed)
    probe_train_idx = _limit_indices(probe_train_idx, cfg.limit_probe_train_samples, cfg.seed + 202)
    test_idx = _limit_indices(test_idx, cfg.limit_test_samples, cfg.seed + 303)
    y_train = y[probe_train_idx]
    y_test = y[test_idx]
    rows = evaluate_one(cfg, model, eval_dataset, probe_train_idx, test_idx, y_train, y_test, conditions, device)
    write_csv(cfg.output_dir / f"probe_results_{cfg.holdout}.csv", rows)
    write_csv(cfg.output_dir / f"probe_summary_{cfg.holdout}.csv", summarize_rows(rows, ["arm", "model", "holdout", "condition"]))
    return rows


def main() -> None:
    load_env_file(Path(".env"))
    _ensure_data()
    args = type(
        "EmbeddedArgs",
        (),
        {
            "pretrain_zarr_path": "data/processed/ssl4eo_s12_v11_48k.zarr",
            "eval_zarr_path": "data/cropharvest/processed/v2.zarr",
            "arrays_dir": "data/cropharvest/raw/features/arrays",
            "labels_geojson": "data/cropharvest/raw/labels.geojson",
            "output_dir": Path("artifacts/[7]"),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "model_dim": MODEL_DIM,
            "encoder_hidden": ENCODER_HIDDEN,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "predictor_layers": PREDICTOR_LAYERS,
            "lr": LR,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "device": DEVICE,
            "limit_train_samples": LIMIT_TRAIN_SAMPLES,
            "limit_probe_train_samples": LIMIT_PROBE_TRAIN_SAMPLES,
            "limit_test_samples": LIMIT_TEST_SAMPLES,
        },
    )()

    arms = SELECTED_ARMS or DEFAULT_ARMS
    unknown_arms = sorted(set(arms) - set(ARM_SPECS))
    if unknown_arms:
        raise ValueError(f"Unknown arms: {unknown_arms}")
    if SMOKE_MODEL:
        smoke_model(arms, args)
        return
    holdouts = SELECTED_HOLDOUTS or DEFAULT_HOLDOUTS
    seeds = SELECTED_SEEDS
    conditions = _condition_map(SELECTED_CONDITIONS or DEFAULT_CONDITIONS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [
        {"arm": arm, "seed": seed, "run_dir": str(args.output_dir / f"{arm}_seed{seed}")}
        for arm in arms
        for seed in seeds
    ]
    write_csv(args.output_dir / "run_manifest.csv", manifest)
    if DRY_RUN:
        return

    _set_seed(min(seeds))
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    pretrain_dataset = MultimodalPatchDataset(args.pretrain_zarr_path, device="cpu")
    eval_dataset = MultimodalPatchDataset(args.eval_zarr_path, device="cpu")
    _assert_shared_channel_contract(pretrain_dataset, eval_dataset)
    valid_files = valid_cropharvest_files(args.arrays_dir, eval_dataset.shapes.timesteps)
    if len(valid_files) != len(eval_dataset):
        raise ValueError(f"Valid H5 count {len(valid_files)} does not match eval Zarr length {len(eval_dataset)}")
    y, groups = load_labels(valid_files, args.labels_geojson)

    all_rows: list[dict[str, Any]] = []
    for row in manifest:
        arm = str(row["arm"])
        cfg = RunConfig(
            pretrain_zarr_path=args.pretrain_zarr_path,
            eval_zarr_path=args.eval_zarr_path,
            arrays_dir=args.arrays_dir,
            labels_geojson=args.labels_geojson,
            output_dir=Path(row["run_dir"]),
            holdout="all",
            arm=arm,
            seed=int(row["seed"]),
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            model_dim=args.model_dim,
            encoder_hidden=args.encoder_hidden,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            predictor_layers=args.predictor_layers,
            lr=args.lr,
            early_stopping_patience=args.early_stopping_patience,
            device=args.device,
            limit_train_samples=args.limit_train_samples,
            limit_probe_train_samples=args.limit_probe_train_samples,
            limit_test_samples=args.limit_test_samples,
        )
        _set_seed(cfg.seed)
        model, _ = pretrain_one(cfg, ARM_SPECS[arm], pretrain_dataset, device)
        run_rows: list[dict[str, Any]] = []
        for holdout in holdouts:
            holdout_cfg = replace(cfg, holdout=holdout)
            rows = evaluate_holdout(holdout_cfg, model, eval_dataset, y, groups, conditions, device)
            run_rows.extend(rows)
            all_rows.extend(rows)
        write_csv(cfg.output_dir / "probe_results.csv", run_rows)
        write_csv(cfg.output_dir / "probe_summary.csv", summarize_rows(run_rows, ["arm", "model", "holdout", "condition"]))
        write_csv(args.output_dir / "probe_results_partial.csv", all_rows)

    write_csv(args.output_dir / "probe_results.csv", all_rows)
    write_csv(args.output_dir / "probe_summary.csv", summarize_rows(all_rows, ["arm", "model", "holdout", "condition"]))
    write_csv(args.output_dir / "priority_summary.csv", summarize_rows(all_rows, ["arm", "model"]))
    write_csv(args.output_dir / "per_holdout_priority_summary.csv", summarize_rows(all_rows, ["arm", "model", "holdout"]))
    lem_rows = [row for row in all_rows if row["holdout"] == "lem-brazil"]
    write_csv(args.output_dir / "lem_brazil_priority_summary.csv", summarize_rows(lem_rows, ["arm", "model"]))
    metadata = {
        "experiment": "[7] Temporal Block-JEPA v1",
        "pretrain_zarr_path": str(args.pretrain_zarr_path),
        "eval_zarr_path": str(args.eval_zarr_path),
        "channel_contract": {
            "s2": S2_CHANNELS_CANONICAL,
            "s1": S1_CHANNELS_CANONICAL,
            "context": CONTEXT_CHANNELS_MINIMAL,
        },
        "arms": arms,
        "holdouts": holdouts,
        "seeds": seeds,
        "conditions": [name for name, _, _ in conditions],
        "num_runs": len(manifest),
        "num_rows": len(all_rows),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


main()
