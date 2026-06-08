"""Spatial-token JEPA models for agriculture-aligned SSL4EO pretraining."""

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.core.jepa import JepaBatchMasks
from src.core.model import DoyEmbedding
from src.datasets.ssl4eo import MIN_TOKEN_CLEAR_FRACTION, TOKEN_PATCH_SIZE


def _elapsed_embedding(elapsed_days: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
    scaled = elapsed_days.unsqueeze(-1) / 366.0
    return mlp(scaled)


def _masked_cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    loss = 1.0 - (pred * target).sum(dim=-1)
    weights = mask.float()
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


@dataclass
class SpatialJepaOutput:
    local_pred: torch.Tensor
    local_target: torch.Tensor
    local_mask: torch.Tensor
    global_pred: torch.Tensor
    global_target: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    return (values * weights).sum(dim=dims) / weights.sum(dim=dims).clamp_min(1.0)


class RadarOpticalPooledEncoder(nn.Module):
    """Pooled reference encoder sharing the same mask-aware tokenizer as SpatialTokenEncoder."""

    def __init__(
        self,
        s2_channels: int,
        s1_channels: int,
        model_dim: int,
        image_size: int = 16,
        spatial_patch_size: int = TOKEN_PATCH_SIZE,
        min_clear_fraction: float = MIN_TOKEN_CLEAR_FRACTION,
    ) -> None:
        super().__init__()
        self.s2_tokenizer = SpatialPatchTokenizer(
            s2_channels, model_dim, image_size, spatial_patch_size, min_clear_fraction
        )
        self.s1_tokenizer = SpatialPatchTokenizer(
            s1_channels, model_dim, image_size, spatial_patch_size, min_clear_fraction
        )
        self.s2_doy = DoyEmbedding(model_dim)
        self.s1_doy = DoyEmbedding(model_dim)
        self.s2_elapsed = nn.Sequential(nn.Linear(1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.s1_elapsed = nn.Sequential(nn.Linear(1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.modality_gate = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        s2_doy: torch.Tensor,
        s1_doy: torch.Tensor,
        s2_elapsed_days: torch.Tensor,
        s1_elapsed_days: torch.Tensor,
        s2_available: torch.Tensor,
        s1_available: torch.Tensor,
        s2_mask: torch.Tensor | None = None,
        s1_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s2_tokens, s2_token_available, _ = self.s2_tokenizer(s2, s2_mask)
        s1_tokens, s1_token_available, _ = self.s1_tokenizer(s1, s1_mask)
        s2_token_available = s2_token_available & s2_available.bool()[:, :, None]
        s1_token_available = s1_token_available & s1_available.bool()[:, :, None]
        s2_z = _masked_mean(s2_tokens, s2_token_available, dims=(2,))
        s1_z = _masked_mean(s1_tokens, s1_token_available, dims=(2,))
        s2_z = s2_z + self.s2_doy(s2_doy) + _elapsed_embedding(s2_elapsed_days, self.s2_elapsed)
        s1_z = s1_z + self.s1_doy(s1_doy) + _elapsed_embedding(s1_elapsed_days, self.s1_elapsed)
        s2_modality_avail = s2_token_available.any(dim=2) & s2_available.bool()
        s1_modality_avail = s1_token_available.any(dim=2) & s1_available.bool()
        weights = torch.stack([s2_modality_avail, s1_modality_avail], dim=-1).float()
        weights = weights * torch.clamp(self.modality_gate, min=0.05, max=10.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.norm(s2_z * weights[..., 0:1] + s1_z * weights[..., 1:2])

    def position_identity(
        self,
        s2_doy: torch.Tensor,
        s1_doy: torch.Tensor,
        s2_elapsed_days: torch.Tensor,
        s1_elapsed_days: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        s2_time = self.s2_doy(s2_doy) + _elapsed_embedding(s2_elapsed_days, self.s2_elapsed)
        s1_time = self.s1_doy(s1_doy) + _elapsed_embedding(s1_elapsed_days, self.s1_elapsed)
        return 0.5 * (s2_time + s1_time)


class PooledTargetEncoder(nn.Module):
    def __init__(
        self,
        s2_channels: int,
        s1_channels: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.local = RadarOpticalPooledEncoder(s2_channels, s1_channels, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, time_visible: torch.Tensor | None = None, **kwargs: torch.Tensor) -> torch.Tensor:
        z = self.local(**kwargs)
        if time_visible is None:
            return self.norm(self.temporal(z))
        padding = ~time_visible.bool()
        z = z.masked_fill(padding.unsqueeze(-1), 0.0)
        z = self.norm(self.temporal(z, src_key_padding_mask=padding))
        return z.masked_fill(padding.unsqueeze(-1), 0.0)


class PooledTemporalJepaModel(nn.Module):
    """Full-target pooled baseline used to isolate sampling alignment."""

    def __init__(
        self,
        s2_channels: int,
        s1_channels: int,
        model_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 8,
        predictor_layers: int = 2,
        dropout: float = 0.1,
        ema_momentum: float = 0.996,
    ) -> None:
        super().__init__()
        self.context_encoder = PooledTargetEncoder(s2_channels, s1_channels, model_dim, num_layers, num_heads, dropout)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.predictor_encoder = nn.TransformerEncoder(layer, num_layers=predictor_layers)
        self.predictor_norm = nn.LayerNorm(model_dim)
        self.predictor_head = nn.Linear(model_dim, model_dim)
        self.ema_momentum = ema_momentum
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> "PooledTemporalJepaModel":
        super().train(mode)
        self.target_encoder.eval()
        return self

    @staticmethod
    def _inputs(
        s2: torch.Tensor,
        s1: torch.Tensor,
        doy: torch.Tensor,
        s2_available: torch.Tensor,
        s1_available: torch.Tensor,
        s2_doy: torch.Tensor | None = None,
        s1_doy: torch.Tensor | None = None,
        s2_elapsed_days: torch.Tensor | None = None,
        s1_elapsed_days: torch.Tensor | None = None,
        s2_mask: torch.Tensor | None = None,
        s1_mask: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fallback_elapsed = doy - doy[:, :1]
        return {
            "s2": s2,
            "s1": s1,
            "s2_doy": s2_doy if s2_doy is not None else doy,
            "s1_doy": s1_doy if s1_doy is not None else doy,
            "s2_elapsed_days": s2_elapsed_days if s2_elapsed_days is not None else fallback_elapsed,
            "s1_elapsed_days": s1_elapsed_days if s1_elapsed_days is not None else fallback_elapsed,
            "s2_available": s2_available,
            "s1_available": s1_available,
            "s2_mask": s2_mask,
            "s1_mask": s1_mask,
        }

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        m = self.ema_momentum if momentum is None else momentum
        for online, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(m).add_(online.data, alpha=1.0 - m)

    def forward_views(
        self,
        target_mask: torch.Tensor,
        context: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        context_time_keep: torch.Tensor | None = None,
    ) -> SpatialJepaOutput:
        context_inputs = self._inputs(**context)
        target_inputs = self._inputs(**target)
        context_available = (context_inputs["s2_available"] + context_inputs["s1_available"]) > 0
        available = (target_inputs["s2_available"] + target_inputs["s1_available"]) > 0
        time_visible = (~target_mask) & context_available
        if context_time_keep is not None:
            time_visible = time_visible & context_time_keep.bool()
        context = self.context_encoder(time_visible=time_visible, **context_inputs)
        position = self.context_encoder.local.position_identity(**target_inputs)
        masked_tokens = self.mask_token.expand_as(context) + position
        predictor_input = torch.where(target_mask.unsqueeze(-1), masked_tokens, context)
        with torch.no_grad():
            target = self.target_encoder(time_visible=available, **target_inputs)
        predictor_valid = time_visible | (target_mask & available)
        predictor_input = predictor_input.masked_fill(~predictor_valid.unsqueeze(-1), 0.0)
        pred = self.predictor_encoder(predictor_input, src_key_padding_mask=~predictor_valid)
        pred = self.predictor_head(self.predictor_norm(pred))
        pred = pred.masked_fill(~predictor_valid.unsqueeze(-1), 0.0)
        local_mask = target_mask & available
        global_pred = _masked_mean(pred, local_mask, dims=(1,))
        global_target = _masked_mean(target, local_mask, dims=(1,))
        return SpatialJepaOutput(pred, target, local_mask, global_pred, global_target)

    def forward(self, target_mask: torch.Tensor, **kwargs: torch.Tensor) -> SpatialJepaOutput:
        return self.forward_views(target_mask, kwargs, kwargs)

    def encode(self, masks: JepaBatchMasks | None = None, **kwargs: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(**kwargs)
        time_visible = masks.time_keep.bool() if masks is not None and masks.time_keep is not None else None
        return self.context_encoder(time_visible=time_visible, **inputs)


class SpatialPatchTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_dim: int,
        image_size: int = 16,
        patch_size: int = TOKEN_PATCH_SIZE,
        min_clear_fraction: float = MIN_TOKEN_CLEAR_FRACTION,
    ) -> None:
        super().__init__()
        if not 0.0 <= min_clear_fraction <= 1.0:
            raise ValueError("min_clear_fraction must be within [0, 1]")
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.min_clear_fraction = min_clear_fraction
        self.grid_size = image_size // patch_size
        self.num_spatial_tokens = self.grid_size**2
        self.proj = nn.Conv2d(in_channels, model_dim, kernel_size=1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        pixel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, c, h, w = x.shape
        if pixel_mask is None:
            pixel_mask = torch.ones((b, t, h, w), dtype=x.dtype, device=x.device)
        else:
            pixel_mask = pixel_mask.to(dtype=x.dtype)
        x = x * pixel_mask.unsqueeze(2)
        x = x.reshape(b * t, c, h, w)
        mask = pixel_mask.reshape(b * t, 1, h, w)
        if h != self.image_size or w != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="nearest")
            mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode="nearest")
        x = self.proj(x)
        gs = self.grid_size
        ps = self.patch_size
        x = x.reshape(b * t, -1, gs, ps, gs, ps).sum(dim=(3, 5))
        counts = mask.reshape(b * t, 1, gs, ps, gs, ps).sum(dim=(3, 5)).clamp_min(1.0)
        x = x / counts
        clear_fraction = F.avg_pool2d(mask, kernel_size=ps, stride=ps).flatten(1)
        available = clear_fraction >= self.min_clear_fraction
        return (
            x.reshape(b * t, -1, self.num_spatial_tokens).transpose(1, 2).reshape(b, t, self.num_spatial_tokens, -1),
            available.reshape(b, t, self.num_spatial_tokens),
            clear_fraction.reshape(b, t, self.num_spatial_tokens),
        )


class SpatialTokenEncoder(nn.Module):
    """Shared spatiotemporal encoder over aligned radar and optical tokens."""

    def __init__(
        self,
        s2_channels: int,
        s1_channels: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        image_size: int = 16,
        spatial_patch_size: int = TOKEN_PATCH_SIZE,
    ) -> None:
        super().__init__()
        self.s2_tokenizer = SpatialPatchTokenizer(s2_channels, model_dim, image_size, spatial_patch_size)
        self.s1_tokenizer = SpatialPatchTokenizer(s1_channels, model_dim, image_size, spatial_patch_size)
        self.num_spatial_tokens = self.s2_tokenizer.num_spatial_tokens
        self.spatial_embed = nn.Parameter(torch.zeros(1, 1, 1, self.num_spatial_tokens, model_dim))
        self.modality_embed = nn.Parameter(torch.zeros(1, 1, 2, 1, model_dim))
        self.s2_doy = DoyEmbedding(model_dim)
        self.s1_doy = DoyEmbedding(model_dim)
        self.s2_elapsed = nn.Sequential(nn.Linear(1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.s1_elapsed = nn.Sequential(nn.Linear(1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        nn.init.trunc_normal_(self.spatial_embed, std=0.02)
        nn.init.trunc_normal_(self.modality_embed, std=0.02)

    def forward(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        doy: torch.Tensor,
        s2_available: torch.Tensor,
        s1_available: torch.Tensor,
        s2_doy: torch.Tensor | None = None,
        s1_doy: torch.Tensor | None = None,
        s2_elapsed_days: torch.Tensor | None = None,
        s1_elapsed_days: torch.Tensor | None = None,
        s2_mask: torch.Tensor | None = None,
        s1_mask: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fallback_elapsed = doy - doy[:, :1]
        s2_doy = s2_doy if s2_doy is not None else doy
        s1_doy = s1_doy if s1_doy is not None else doy
        s2_elapsed_days = s2_elapsed_days if s2_elapsed_days is not None else fallback_elapsed
        s1_elapsed_days = s1_elapsed_days if s1_elapsed_days is not None else fallback_elapsed
        b, t = s2.shape[:2]
        s = self.num_spatial_tokens
        s2_z, s2_token_available, _ = self.s2_tokenizer(s2, s2_mask)
        s1_z, s1_token_available, _ = self.s1_tokenizer(s1, s1_mask)
        s2_time = self.s2_doy(s2_doy) + _elapsed_embedding(s2_elapsed_days, self.s2_elapsed)
        s1_time = self.s1_doy(s1_doy) + _elapsed_embedding(s1_elapsed_days, self.s1_elapsed)
        z = torch.stack([s2_z + s2_time[:, :, None, :], s1_z + s1_time[:, :, None, :]], dim=2)
        z = z + self.spatial_embed + self.modality_embed
        available = torch.stack([s2_token_available, s1_token_available], dim=2)
        timestep_available = torch.stack([s2_available, s1_available], dim=2)[:, :, :, None].expand(b, t, 2, s).bool()
        available = available & timestep_available
        if visible_mask is not None:
            available = available & visible_mask
        flat = z.reshape(b, t * 2 * s, -1)
        padding = ~available.reshape(b, t * 2 * s)
        encoded = self.norm(self.encoder(flat, src_key_padding_mask=padding))
        encoded = encoded.masked_fill(padding.unsqueeze(-1), 0.0)
        return encoded.reshape(b, t, 2, s, -1), available

    def position_identity(
        self,
        doy: torch.Tensor,
        s2_doy: torch.Tensor | None = None,
        s1_doy: torch.Tensor | None = None,
        s2_elapsed_days: torch.Tensor | None = None,
        s1_elapsed_days: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        fallback_elapsed = doy - doy[:, :1]
        s2_doy = s2_doy if s2_doy is not None else doy
        s1_doy = s1_doy if s1_doy is not None else doy
        s2_elapsed_days = s2_elapsed_days if s2_elapsed_days is not None else fallback_elapsed
        s1_elapsed_days = s1_elapsed_days if s1_elapsed_days is not None else fallback_elapsed
        s2_time = self.s2_doy(s2_doy) + _elapsed_embedding(s2_elapsed_days, self.s2_elapsed)
        s1_time = self.s1_doy(s1_doy) + _elapsed_embedding(s1_elapsed_days, self.s1_elapsed)
        identity = torch.stack([s2_time, s1_time], dim=2)[:, :, :, None, :]
        identity = identity + self.spatial_embed + self.modality_embed
        return identity


class SpatialTokenJepaModel(nn.Module):
    """Masked spatial-temporal latent predictor with an EMA full target encoder."""

    def __init__(
        self,
        s2_channels: int,
        s1_channels: int,
        model_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 8,
        predictor_dim: int = 192,
        predictor_layers: int = 2,
        dropout: float = 0.1,
        ema_momentum: float = 0.996,
    ) -> None:
        super().__init__()
        self.context_encoder = SpatialTokenEncoder(s2_channels, s1_channels, model_dim, num_layers, num_heads, dropout)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, model_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.predictor_in = nn.Linear(model_dim, predictor_dim)
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=max(1, predictor_dim // 48),
            dim_feedforward=predictor_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=predictor_layers)
        self.predictor_out = nn.Sequential(nn.LayerNorm(predictor_dim), nn.Linear(predictor_dim, model_dim))
        self.ema_momentum = ema_momentum
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> "SpatialTokenJepaModel":
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        m = self.ema_momentum if momentum is None else momentum
        for online, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(m).add_(online.data, alpha=1.0 - m)

    def forward_views(
        self,
        target_mask: torch.Tensor,
        context: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        context_time_keep: torch.Tensor | None = None,
    ) -> SpatialJepaOutput:
        b, t, s = target_mask.shape
        expanded_target_mask = target_mask[:, :, None, :].expand(b, t, 2, s)
        visible_mask = ~expanded_target_mask
        if context_time_keep is not None:
            visible_mask = visible_mask & context_time_keep[:, :, None, None]
        context, context_available = self.context_encoder(visible_mask=visible_mask, **context)
        with torch.no_grad():
            target_z, target_available = self.target_encoder(**target)
        local_mask = expanded_target_mask & target_available
        position = self.context_encoder.position_identity(**target)
        masked_tokens = self.mask_token.expand_as(context) + position
        predictor_tokens = torch.where(local_mask.unsqueeze(-1), masked_tokens, context)
        predictor_valid = context_available | local_mask
        flat = predictor_tokens.reshape(b, t * 2 * s, -1)
        flat_pred = self.predictor_out(
            self.predictor(self.predictor_in(flat), src_key_padding_mask=~predictor_valid.reshape(b, t * 2 * s))
        )
        pred = flat_pred.reshape_as(context)
        global_pred = _masked_mean(pred, local_mask, dims=(1, 2, 3))
        global_target = _masked_mean(target_z, local_mask, dims=(1, 2, 3))
        return SpatialJepaOutput(pred, target_z, local_mask, global_pred, global_target)

    def forward(self, target_mask: torch.Tensor, **kwargs: torch.Tensor) -> SpatialJepaOutput:
        return self.forward_views(target_mask, kwargs, kwargs)

    def encode(self, masks: JepaBatchMasks | None = None, **kwargs: torch.Tensor) -> torch.Tensor:
        visible_mask = None
        if masks is not None and masks.time_keep is not None:
            b, t = masks.time_keep.shape
            s = self.context_encoder.num_spatial_tokens
            visible_mask = masks.time_keep[:, :, None, None].expand(b, t, 2, s).bool()
        z, available = self.context_encoder(visible_mask=visible_mask, **kwargs)
        return _masked_mean(z, available, dims=(2, 3))


def spatial_jepa_loss(output: SpatialJepaOutput, global_weight: float = 0.25) -> torch.Tensor:
    local = _masked_cosine_loss(output.local_pred, output.local_target, output.local_mask)
    global_loss = _masked_cosine_loss(
        output.global_pred,
        output.global_target,
        torch.ones(output.global_pred.shape[0], device=output.global_pred.device),
    )
    return local + global_weight * global_loss
