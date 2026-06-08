"""Run native-SSL4EO content-sensitivity diagnostics on experiment [9] checkpoints."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core.jepa import JepaBatchMasks
from src.core.spatial_jepa import PooledTemporalJepaModel, SpatialTokenJepaModel
from src.datasets.dataset import MultimodalPatchDataset
from src.utils.io_utils import write_csv, write_json

ARTIFACT_DIR = ROOT / "artifacts/[9]"
OUTPUT_DIR = ARTIFACT_DIR / "verification"
GENERIC_ZARR = ROOT / "data/processed/ssl4eo_s12_v11_generic_fixed_48k.zarr"
AGRO_ZARR = ROOT / "data/processed/ssl4eo_s12_v11_agro_fixed_48k.zarr"
CHECKPOINT_EPOCHS = [1, 2, 4, 8, 12]
ARMS = [
    "A_lr_control_generic",
    "B_generic_viewdrop",
    "C_mixed_viewdrop",
    "D_mixed_viewdrop_consistency",
    "E_spatial_viewdrop_consistency",
]
SEED = 42
AGRO_VALIDATION_SAMPLES = 2048
BATCH_SIZE = 32
NUM_WORKERS = 2
DEVICE = torch.device("cuda")
CORRUPTION_MODES = ["sensor_off_s2", "sensor_off_s1", "temporal_drop_50", "s2_off_tdrop50"]
SIGNAL_FIELDS = ["s2", "s1", "climate"]
MISSINGNESS_FIELDS = [
    "s2_mask",
    "s1_mask",
    "climate_mask",
    "s2_available",
    "s1_available",
    "climate_available",
]
SPATIAL_RESIDUAL_TOKENS_PER_BATCH = 64


def _load_runner() -> ModuleType:
    path = ROOT / "runners/[9].py"
    spec = importlib.util.spec_from_file_location("runner_9_verification", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _shuffle_fields(
    batch: dict[str, torch.Tensor],
    fields: list[str],
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    output = {key: value.clone() for key, value in batch.items()}
    batch_size = int(batch["s2"].shape[0])
    if batch_size < 2:
        raise ValueError("Shuffle diagnostics require batches with at least two samples")
    device = batch["s2"].device
    offset = int(torch.randint(1, batch_size, (1,), generator=generator, device=device).item())
    permutation = (torch.arange(batch_size, device=device) + offset) % batch_size
    for key in fields:
        if key in output:
            output[key] = output[key][permutation]
    return output


def shuffled_content_target(
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    return _shuffle_fields(batch, SIGNAL_FIELDS, generator)


def shuffled_missingness_context(
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    return _shuffle_fields(batch, MISSINGNESS_FIELDS, generator)


def zero_content_target(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = {key: value.clone() for key, value in batch.items()}
    for key in SIGNAL_FIELDS:
        if key in output:
            output[key].zero_()
    return output


def sensor_off_s1(batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    output = {key: value.clone() for key, value in batch.items()}
    output["s1"].zero_()
    output["s1_available"].zero_()
    if "s1_mask" in output:
        output["s1_mask"].zero_()
    batch_size, timesteps = output["s2"].shape[:2]
    return output, torch.ones((batch_size, timesteps), dtype=torch.bool, device=DEVICE)


def _masked_time_mean(features: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    weights = available.float().unsqueeze(-1)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_token_mean(features: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    weights = available.float().unsqueeze(-1)
    return (features * weights).sum(dim=(2, 3)) / weights.sum(dim=(2, 3)).clamp_min(1.0)


def _row_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=-1)


def downstream_embedding(features: torch.Tensor, time_keep: torch.Tensor | None = None) -> torch.Tensor:
    if time_keep is None:
        return features.mean(dim=1)
    weights = time_keep.float().unsqueeze(-1)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def effective_rank(eigenvalues: np.ndarray) -> float:
    values = np.asarray(eigenvalues, dtype=np.float64)
    values = values[values > np.finfo(np.float64).eps]
    if not len(values):
        return 0.0
    probabilities = values / values.sum()
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def _covariance_spectrum(features: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    values = features.double().numpy()
    values = values - values.mean(axis=0, keepdims=True)
    covariance = values.T @ values / max(1, values.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1].clip(min=0.0)
    return covariance, eigenvalues


def mean_off_diagonal_cosine(features: torch.Tensor) -> float:
    if features.shape[0] < 2:
        return float("nan")
    normalized = F.normalize(features.float(), dim=-1)
    summed = normalized.sum(dim=0)
    numerator = float((summed @ summed - features.shape[0]).item())
    return numerator / float(features.shape[0] * (features.shape[0] - 1))


def same_time_inter_sample_cosine(features: torch.Tensor, available: torch.Tensor) -> float:
    values = []
    for timestep in range(features.shape[1]):
        selected = features[available[:, timestep].bool(), timestep]
        if selected.shape[0] >= 2:
            values.append(mean_off_diagonal_cosine(selected))
    return float(np.mean(values)) if values else float("nan")


def _feature_metrics(features: torch.Tensor, available: torch.Tensor, prefix: str) -> tuple[dict[str, float], np.ndarray]:
    sequence = _masked_time_mean(features, available)
    valid_time = features[available.bool()]
    _, spectrum = _covariance_spectrum(sequence)
    return {
        f"{prefix}_sequence_feature_variance": float(sequence.var(dim=0, unbiased=True).mean().item()),
        f"{prefix}_timestep_feature_variance": float(valid_time.var(dim=0, unbiased=True).mean().item()),
        f"{prefix}_sequence_feature_norm": float(sequence.norm(dim=-1).mean().item()),
        f"{prefix}_timestep_feature_norm": float(valid_time.norm(dim=-1).mean().item()),
        f"{prefix}_sequence_inter_sample_cosine": mean_off_diagonal_cosine(sequence),
        f"{prefix}_same_time_inter_sample_cosine": same_time_inter_sample_cosine(features, available),
        f"{prefix}_effective_rank": effective_rank(spectrum),
    }, spectrum


def _loss_parts(output: Any) -> dict[str, float]:
    local_values = 1.0 - _row_cosine(output.local_pred, output.local_target.detach())
    local_weights = output.local_mask.float()
    global_values = 1.0 - _row_cosine(output.global_pred, output.global_target.detach())
    return {
        "local_sum": float((local_values * local_weights).sum().item()),
        "local_count": float(local_weights.sum().item()),
        "global_sum": float(global_values.sum().item()),
        "global_count": float(global_values.numel()),
    }


def _add_loss(accumulator: dict[str, float], output: Any) -> None:
    for key, value in _loss_parts(output).items():
        accumulator[key] += value


def _finalize_loss(accumulator: dict[str, float], global_weight: float, prefix: str) -> dict[str, float]:
    local = accumulator["local_sum"] / max(1.0, accumulator["local_count"])
    global_loss = accumulator["global_sum"] / max(1.0, accumulator["global_count"])
    return {
        f"{prefix}_local_loss": local,
        f"{prefix}_global_loss": global_loss,
        f"{prefix}_combined_loss": local + global_weight * global_loss,
        f"{prefix}_valid_masked_tokens": accumulator["local_count"],
    }


def _full_representations(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(model, PooledTemporalJepaModel):
        inputs = model._inputs(**batch)
        context = model.context_encoder(**inputs)
        target = model.target_encoder(**inputs)
        available = (inputs["s2_available"] + inputs["s1_available"]) > 0
        return context, target, available
    context_tokens, context_available = model.context_encoder(**batch)
    target_tokens, target_available = model.target_encoder(**batch)
    return (
        _masked_token_mean(context_tokens, context_available),
        _masked_token_mean(target_tokens, target_available),
        target_available.any(dim=(2, 3)),
    )


def _target_representations(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, PooledTemporalJepaModel):
        inputs = model._inputs(**batch)
        return model.target_encoder(**inputs), (inputs["s2_available"] + inputs["s1_available"]) > 0
    target_tokens, target_available = model.target_encoder(**batch)
    return _masked_token_mean(target_tokens, target_available), target_available.any(dim=(2, 3))


def _spatial_target_tokens(
    model: SpatialTokenJepaModel,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return model.target_encoder(**batch)


def _encoded_representations(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    time_keep: torch.Tensor | None,
) -> torch.Tensor:
    masks = JepaBatchMasks(time_keep=time_keep) if time_keep is not None else None
    return model.encode(masks=masks, **batch)


def _spatial_token_metrics(
    clean: torch.Tensor,
    zero: torch.Tensor,
    available: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    statistics, residual = _spatial_token_batch_statistics(clean, zero, available)
    metrics = _finalize_spatial_token_statistics(statistics)
    _, spectrum = _covariance_spectrum(residual.cpu())
    metrics["spatial_residual_effective_rank"] = effective_rank(spectrum)
    return metrics, residual


def _spatial_token_batch_statistics(
    clean: torch.Tensor,
    zero: torch.Tensor,
    available: torch.Tensor,
) -> tuple[dict[str, torch.Tensor | float], torch.Tensor]:
    batch_size, timesteps, modalities, spatial_slots, dimension = clean.shape
    slots = timesteps * modalities * spatial_slots
    clean_flat = clean.reshape(batch_size, slots, dimension)
    zero_flat = zero.reshape(batch_size, slots, dimension)
    available_flat = available.reshape(batch_size, slots)
    normalized = F.normalize(clean_flat.float(), dim=-1) * available_flat.unsqueeze(-1)
    sums = normalized.sum(dim=0)
    counts = available_flat.sum(dim=0).float()
    valid_clean = clean_flat[available_flat]
    valid_zero = zero_flat[available_flat]
    residual = valid_clean - valid_zero
    return {
        "slot_normalized_sum": sums.cpu(),
        "slot_count": counts.cpu(),
        "clean_zero_cosine_sum": float(_row_cosine(valid_clean, valid_zero).sum().item()),
        "residual_sum": residual.sum(dim=0).cpu(),
        "residual_square_sum": residual.square().sum(dim=0).cpu(),
        "residual_count": float(residual.shape[0]),
    }, residual


def _merge_spatial_token_statistics(
    total: dict[str, torch.Tensor | float],
    batch: dict[str, torch.Tensor | float],
) -> None:
    for key, value in batch.items():
        if key not in total:
            total[key] = value.clone() if isinstance(value, torch.Tensor) else value
        elif isinstance(value, torch.Tensor):
            total[key] = total[key] + value
        else:
            total[key] = float(total[key]) + value


def _finalize_spatial_token_statistics(statistics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    sums = statistics["slot_normalized_sum"]
    counts = statistics["slot_count"]
    pair_counts = counts * (counts - 1.0)
    valid_slots = pair_counts > 0
    pair_numerator = sums.square().sum(dim=-1) - counts
    same_slot_cosine = pair_numerator[valid_slots].sum() / pair_counts[valid_slots].sum().clamp_min(1.0)
    residual_count = float(statistics["residual_count"])
    residual_sum = statistics["residual_sum"]
    residual_square_sum = statistics["residual_square_sum"]
    residual_variance = (
        residual_square_sum - residual_sum.square() / max(1.0, residual_count)
    ) / max(1.0, residual_count - 1.0)
    return {
        "spatial_same_slot_inter_sample_cosine": float(same_slot_cosine.item()),
        "spatial_clean_zero_token_cosine": float(statistics["clean_zero_cosine_sum"]) / max(1.0, residual_count),
        "spatial_residual_variance": float(residual_variance.mean().item()),
    }


def _sample_residual_tokens(residual: torch.Tensor) -> torch.Tensor:
    if residual.shape[0] <= SPATIAL_RESIDUAL_TOKENS_PER_BATCH:
        return residual.cpu()
    indices = torch.linspace(
        0,
        residual.shape[0] - 1,
        SPATIAL_RESIDUAL_TOKENS_PER_BATCH,
        device=residual.device,
    ).long()
    return residual[indices].cpu()


def _corruption_metrics(clean: torch.Tensor, degraded: torch.Tensor) -> dict[str, float]:
    clean_covariance, _ = _covariance_spectrum(clean)
    degraded_covariance, degraded_spectrum = _covariance_spectrum(degraded)
    centroid_shift = degraded.mean(dim=0) - clean.mean(dim=0)
    covariance_shift = np.linalg.norm(degraded_covariance - clean_covariance, ord="fro")
    covariance_scale = max(np.linalg.norm(clean_covariance, ord="fro"), 1e-12)
    return {
        "degraded_feature_variance": float(degraded.var(dim=0, unbiased=True).mean().item()),
        "degraded_effective_rank": effective_rank(degraded_spectrum),
        "centroid_shift_l2": float(centroid_shift.norm().item()),
        "covariance_shift_fro": float(covariance_shift),
        "relative_covariance_shift_fro": float(covariance_shift / covariance_scale),
    }


def _history_row(arm_dir: Path, epoch: int) -> dict[str, float]:
    history = json.loads((arm_dir / "train_history.json").read_text())["history"]
    row = next(item for item in history if int(item["epoch"]) == epoch)
    return {
        "train_loss": float(row["train_loss"]),
        "val_loss": float(row["val_loss"]),
        "train_jepa_loss": float(row["train_jepa_loss"]),
        "val_jepa_loss": float(row["val_jepa_loss"]),
    }


@torch.no_grad()
def _analyze_checkpoint(
    runner: ModuleType,
    model: torch.nn.Module,
    loader: DataLoader,
    spec: Any,
    arm: str,
    epoch: int,
    pool: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[float]]]:
    model.eval()
    generators = {
        name: torch.Generator(device=DEVICE).manual_seed(SEED + offset)
        for name, offset in {
            "mask": 1001,
            "content_shuffle": 2001,
            "missingness_shuffle": 3001,
            "sensor_off_s2": 4001,
            "temporal_drop_50": 4002,
            "s2_off_tdrop50": 4003,
        }.items()
    }
    losses = {
        name: defaultdict(float)
        for name in ["normal", "content_shuffled", "missingness_shuffled", *CORRUPTION_MODES]
    }
    context_features: list[torch.Tensor] = []
    target_features: list[torch.Tensor] = []
    zero_target_features: list[torch.Tensor] = []
    availability: list[torch.Tensor] = []
    corruption: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    spatial_statistics: dict[str, torch.Tensor | float] = {}
    spatial_residual_samples: list[torch.Tensor] = []

    for raw_batch in loader:
        raw_batch = {key: value.to(DEVICE) for key, value in raw_batch.items()}
        clean = runner._model_batch(raw_batch)
        batch_size, timesteps = clean["s2"].shape[:2]
        target_mask = runner._target_mask(spec, batch_size, timesteps, DEVICE, generators["mask"])

        normal_output = model.forward_views(target_mask=target_mask, context=clean, target=clean)
        content_shuffled = shuffled_content_target(clean, generators["content_shuffle"])
        content_shuffled_output = model.forward_views(
            target_mask=target_mask,
            context=clean,
            target=content_shuffled,
        )
        missingness_shuffled = shuffled_missingness_context(clean, generators["missingness_shuffle"])
        missingness_shuffled_output = model.forward_views(
            target_mask=target_mask,
            context=missingness_shuffled,
            target=clean,
        )
        _add_loss(losses["normal"], normal_output)
        _add_loss(losses["content_shuffled"], content_shuffled_output)
        _add_loss(losses["missingness_shuffled"], missingness_shuffled_output)

        context, target, available = _full_representations(model, clean)
        zero_batch = zero_content_target(clean)
        zero_target, _ = _target_representations(model, zero_batch)
        context_features.append(context.cpu())
        target_features.append(target.cpu())
        zero_target_features.append(zero_target.cpu())
        availability.append(available.cpu())

        if isinstance(model, SpatialTokenJepaModel):
            clean_tokens, token_available = _spatial_target_tokens(model, clean)
            zero_tokens, _ = _spatial_target_tokens(model, zero_batch)
            batch_statistics, residual = _spatial_token_batch_statistics(clean_tokens, zero_tokens, token_available)
            _merge_spatial_token_statistics(spatial_statistics, batch_statistics)
            spatial_residual_samples.append(_sample_residual_tokens(residual))

        clean_full_features = _encoded_representations(model, clean, None)
        clean_full_embedding = downstream_embedding(clean_full_features)
        for mode in CORRUPTION_MODES:
            if mode == "sensor_off_s1":
                degraded_raw, time_keep = sensor_off_s1(raw_batch)
            else:
                degraded_raw, time_keep = runner._corrupt_batch(raw_batch, mode, generators[mode])
            degraded = runner._model_batch(degraded_raw)
            degraded_output = model.forward_views(
                target_mask=target_mask,
                context=degraded,
                target=clean,
                context_time_keep=time_keep,
            )
            _add_loss(losses[mode], degraded_output)

            degraded_features = _encoded_representations(model, degraded, time_keep)
            degraded_embedding = downstream_embedding(degraded_features, time_keep)
            clean_retained_features = _encoded_representations(model, clean, time_keep)
            clean_retained_embedding = downstream_embedding(clean_retained_features, time_keep)

            corruption[mode]["clean_full"].append(clean_full_embedding.cpu())
            corruption[mode]["degraded"].append(degraded_embedding.cpu())
            corruption[mode]["full_cosine_displacement"].append(
                (1.0 - _row_cosine(clean_full_embedding, degraded_embedding)).cpu()
            )
            corruption[mode]["full_l2_displacement"].append(
                (clean_full_embedding - degraded_embedding).norm(dim=-1).cpu()
            )
            corruption[mode]["retained_cosine_displacement"].append(
                (1.0 - _row_cosine(clean_retained_embedding, degraded_embedding)).cpu()
            )
            corruption[mode]["retained_l2_displacement"].append(
                (clean_retained_embedding - degraded_embedding).norm(dim=-1).cpu()
            )

    context = torch.cat(context_features)
    target = torch.cat(target_features)
    zero_target = torch.cat(zero_target_features)
    available = torch.cat(availability)
    context_metrics, context_spectrum = _feature_metrics(context, available, "context")
    target_metrics, target_spectrum = _feature_metrics(target, available, "target")
    target_sequence = _masked_time_mean(target, available)
    zero_sequence = _masked_time_mean(zero_target, available)
    valid_target = target[available.bool()]
    valid_zero = zero_target[available.bool()]
    normal = _finalize_loss(losses["normal"], runner.GLOBAL_LOSS_WEIGHT, "normal")
    content_shuffled = _finalize_loss(
        losses["content_shuffled"],
        runner.GLOBAL_LOSS_WEIGHT,
        "content_shuffled",
    )
    missingness_shuffled = _finalize_loss(
        losses["missingness_shuffled"],
        runner.GLOBAL_LOSS_WEIGHT,
        "missingness_shuffled",
    )
    if normal["normal_valid_masked_tokens"] != content_shuffled["content_shuffled_valid_masked_tokens"]:
        raise RuntimeError("Content shuffling changed the valid masked-token count")
    row: dict[str, Any] = {
        "arm": arm,
        "epoch": epoch,
        "pool": pool,
        "seed": SEED,
        "num_samples": int(target.shape[0]),
        **normal,
        **content_shuffled,
        **missingness_shuffled,
        "content_shuffle_local_loss_gap": content_shuffled["content_shuffled_local_loss"] - normal["normal_local_loss"],
        "content_shuffle_global_loss_gap": content_shuffled["content_shuffled_global_loss"] - normal["normal_global_loss"],
        "content_shuffle_combined_loss_gap": content_shuffled["content_shuffled_combined_loss"] - normal["normal_combined_loss"],
        "missingness_shuffle_local_loss_gap": missingness_shuffled["missingness_shuffled_local_loss"] - normal["normal_local_loss"],
        "clean_zero_target_sequence_cosine": float(_row_cosine(target_sequence, zero_sequence).mean().item()),
        "clean_zero_target_timestep_cosine": float(_row_cosine(valid_target, valid_zero).mean().item()),
        **context_metrics,
        **target_metrics,
    }
    spectra = {
        "context": [float(value) for value in context_spectrum],
        "target": [float(value) for value in target_spectrum],
    }
    if spatial_statistics:
        row.update(_finalize_spatial_token_statistics(spatial_statistics))
        residual = torch.cat(spatial_residual_samples)
        _, residual_spectrum = _covariance_spectrum(residual)
        row["spatial_residual_effective_rank"] = effective_rank(residual_spectrum)
        row["spatial_token_content_shuffle_loss_gap"] = row["content_shuffle_local_loss_gap"]
        spectra["spatial_clean_zero_residual"] = [float(value) for value in residual_spectrum]

    corruption_rows = []
    for mode in CORRUPTION_MODES:
        clean_full = torch.cat(corruption[mode]["clean_full"])
        degraded = torch.cat(corruption[mode]["degraded"])
        mode_row: dict[str, Any] = {
            "arm": arm,
            "epoch": epoch,
            "pool": pool,
            "seed": SEED,
            "mode": mode,
            "num_samples": int(degraded.shape[0]),
            **_finalize_loss(losses[mode], runner.GLOBAL_LOSS_WEIGHT, "degraded_context"),
            **_corruption_metrics(clean_full, degraded),
        }
        for metric in [
            "full_cosine_displacement",
            "full_l2_displacement",
            "retained_cosine_displacement",
            "retained_l2_displacement",
        ]:
            mode_row[metric] = float(torch.cat(corruption[mode][metric]).mean().item())
        corruption_rows.append(mode_row)
    return row, corruption_rows, spectra


def _decode_samples(dataset: MultimodalPatchDataset, indices: np.ndarray) -> list[str]:
    if "sample" not in dataset.root:
        return [str(index) for index in indices]
    values = np.asarray(dataset.root["sample"].get_orthogonal_selection((indices,)))
    return [
        value.decode("utf-8").rstrip("\x00") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _dataset_metadata(dataset: MultimodalPatchDataset) -> dict[str, Any]:
    return {
        "path": str(dataset.zarr_path),
        "num_samples": len(dataset),
        "source_revision": dataset.root.attrs.get("source_revision"),
        "build_schema_version": dataset.root.attrs.get("build_schema_version"),
        "build_complete": dataset.root.attrs.get("build_complete"),
    }


def _validation_sets(
    runner: ModuleType,
    generic: MultimodalPatchDataset,
    agro: MultimodalPatchDataset,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _, generic_val = runner._pretrain_split(len(generic), SEED)
    _, mixed_val = runner._pretrain_split(len(generic) + len(agro), SEED)
    generic_common = np.intersect1d(generic_val, mixed_val[mixed_val < len(generic)])
    agro_heldout = mixed_val[mixed_val >= len(generic)] - len(generic)
    rng = np.random.default_rng(SEED)
    agro_heldout = np.sort(rng.choice(agro_heldout, size=AGRO_VALIDATION_SAMPLES, replace=False))
    sets = {
        "generic_common_heldout": generic_common,
        "agriculture_heldout": agro_heldout,
    }
    manifest = {
        "seed": SEED,
        "selection": {
            "generic_common_heldout": "all generic indices held out from both generic-only and mixed [9] training",
            "agriculture_heldout": f"{AGRO_VALIDATION_SAMPLES} fixed agriculture indices held out from mixed [9] training",
        },
        "datasets": {
            "generic": _dataset_metadata(generic),
            "agriculture": _dataset_metadata(agro),
        },
        "sets": {
            name: {
                "indices": [int(index) for index in indices],
                "sample_ids": _decode_samples(generic if name == "generic_common_heldout" else agro, indices),
            }
            for name, indices in sets.items()
        },
    }
    return sets, manifest


def main() -> None:
    runner = _load_runner()
    generic = MultimodalPatchDataset(GENERIC_ZARR)
    agro = MultimodalPatchDataset(AGRO_ZARR)
    validation_sets, manifest = _validation_sets(runner, generic, agro)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "validation_sets.json", manifest)

    checkpoint_rows: list[dict[str, Any]] = []
    corruption_rows: list[dict[str, Any]] = []
    spectra: dict[str, Any] = {}
    for pool, indices in validation_sets.items():
        dataset = generic if pool == "generic_common_heldout" else agro
        loader = DataLoader(
            Subset(dataset, indices.tolist()),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        for arm in ARMS:
            spec = runner.ARM_SPECS[arm]
            arm_dir = ARTIFACT_DIR / f"{arm}_seed{SEED}"
            for epoch in CHECKPOINT_EPOCHS:
                checkpoint = torch.load(
                    arm_dir / f"checkpoint_epoch_{epoch}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                model = runner._make_model(spec, dataset).to(DEVICE)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                row, checkpoint_corruption, checkpoint_spectra = _analyze_checkpoint(
                    runner,
                    model,
                    loader,
                    spec,
                    arm,
                    epoch,
                    pool,
                )
                row.update(_history_row(arm_dir, epoch))
                checkpoint_rows.append(row)
                corruption_rows.extend(checkpoint_corruption)
                spectra[f"{pool}/{arm}/epoch_{epoch}"] = checkpoint_spectra
                write_csv(OUTPUT_DIR / "checkpoint_diagnostics.csv", checkpoint_rows)
                write_csv(OUTPUT_DIR / "corruption_diagnostics.csv", corruption_rows)
                write_json(OUTPUT_DIR / "covariance_spectra.json", spectra)
                print(
                    json.dumps(
                        {
                            "pool": pool,
                            "arm": arm,
                            "epoch": epoch,
                            "normal_local_loss": row["normal_local_loss"],
                            "content_shuffle_local_gap": row["content_shuffle_local_loss_gap"],
                            "target_effective_rank": row["target_effective_rank"],
                        }
                    ),
                    flush=True,
                )
                del model, checkpoint

    print(f"Wrote [9] verification artifacts to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
