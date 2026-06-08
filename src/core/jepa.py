"""Leak-free temporal JEPA modules for agricultural remote-sensing time series."""

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.core.model import DoyEmbedding, PatchEncoder


@dataclass
class JepaBatchMasks:
    modality_keep: torch.Tensor | None = None
    time_keep: torch.Tensor | None = None


def _causal_mask(timesteps: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.full((timesteps, timesteps), float("-inf"), device=device),
        diagonal=1,
    )


class LocalFusionEncoder(nn.Module):
    """Per-timestep multimodal encoder used as the JEPA target branch."""

    def __init__(
        self,
        s2_channels: int,
        s1_channels: int,
        climate_channels: int,
        model_dim: int,
        encoder_hidden: int,
        use_doy: bool,
    ) -> None:
        super().__init__()
        self.s2_encoder = PatchEncoder(s2_channels, encoder_hidden, model_dim)
        self.s1_encoder = PatchEncoder(s1_channels, encoder_hidden, model_dim)
        self.climate_encoder = PatchEncoder(climate_channels, encoder_hidden, model_dim)
        self.doy_embed = DoyEmbedding(model_dim) if use_doy else None
        self.modality_gate = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        climate: torch.Tensor,
        doy: torch.Tensor,
        s2_available: torch.Tensor,
        s1_available: torch.Tensor,
        climate_available: torch.Tensor,
        modality_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s2_z = self.s2_encoder(s2)
        s1_z = self.s1_encoder(s1)
        climate_z = self.climate_encoder(climate)

        weights = torch.stack([s2_available, s1_available, climate_available], dim=-1)
        if modality_keep is not None:
            weights = weights * modality_keep

        gate = torch.clamp(self.modality_gate, min=0.05, max=10.0)
        weights = weights * gate
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        fused = s2_z * weights[..., 0:1] + s1_z * weights[..., 1:2] + climate_z * weights[..., 2:3]
        if self.doy_embed is not None:
            fused = fused + self.doy_embed(doy)
        return self.norm(fused)


class CausalContextEncoder(nn.Module):
    """Causal temporal encoder; timestep t cannot attend to t+1...T."""

    def __init__(
        self,
        local_encoder: LocalFusionEncoder,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.local_encoder = local_encoder
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

    def forward(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        climate: torch.Tensor,
        doy: torch.Tensor,
        s2_available: torch.Tensor,
        s1_available: torch.Tensor,
        climate_available: torch.Tensor,
        masks: JepaBatchMasks | None = None,
    ) -> torch.Tensor:
        masks = masks or JepaBatchMasks()
        z = self.local_encoder(
            s2=s2,
            s1=s1,
            climate=climate,
            doy=doy,
            s2_available=s2_available,
            s1_available=s1_available,
            climate_available=climate_available,
            modality_keep=masks.modality_keep,
        )
        if masks.time_keep is not None:
            z = z * masks.time_keep.unsqueeze(-1)
        attn_mask = _causal_mask(z.shape[1], z.device)
        z = self.temporal(z, mask=attn_mask)
        return self.norm(z)


class TemporalJepaModel(nn.Module):
    def __init__(
        self,
        s2_channels: int,
        s1_channels: int,
        climate_channels: int,
        model_dim: int = 256,
        encoder_hidden: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_doy: bool = True,
        ema_momentum: float = 0.996,
    ) -> None:
        super().__init__()
        local = LocalFusionEncoder(
            s2_channels=s2_channels,
            s1_channels=s1_channels,
            climate_channels=climate_channels,
            model_dim=model_dim,
            encoder_hidden=encoder_hidden,
            use_doy=use_doy,
        )
        self.context_encoder = CausalContextEncoder(
            local_encoder=local,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.target_encoder = copy.deepcopy(local)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        m = self.ema_momentum if momentum is None else momentum
        online = self.context_encoder.local_encoder.parameters()
        target = self.target_encoder.parameters()
        for p_online, p_target in zip(online, target):
            p_target.data.mul_(m).add_(p_online.data, alpha=1.0 - m)

    def forward(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        climate: torch.Tensor,
        doy: torch.Tensor,
        s2_available: torch.Tensor,
        s1_available: torch.Tensor,
        climate_available: torch.Tensor,
        masks: JepaBatchMasks | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_z = self.context_encoder(
            s2=s2,
            s1=s1,
            climate=climate,
            doy=doy,
            s2_available=s2_available,
            s1_available=s1_available,
            climate_available=climate_available,
            masks=masks,
        )
        with torch.no_grad():
            target_z = self.target_encoder(
                s2=s2,
                s1=s1,
                climate=climate,
                doy=doy,
                s2_available=s2_available,
                s1_available=s1_available,
                climate_available=climate_available,
            )
        pred_next = self.predictor(context_z[:, :-1])
        true_next = target_z[:, 1:]
        return pred_next, true_next

    def encode(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        climate: torch.Tensor,
        doy: torch.Tensor,
        s2_available: torch.Tensor,
        s1_available: torch.Tensor,
        climate_available: torch.Tensor,
        masks: JepaBatchMasks | None = None,
    ) -> torch.Tensor:
        return self.context_encoder(
            s2=s2,
            s1=s1,
            climate=climate,
            doy=doy,
            s2_available=s2_available,
            s1_available=s1_available,
            climate_available=climate_available,
            masks=masks,
        )


class TemporalBlockPredictor(nn.Module):
    """Mask-token predictor over temporal latent sequences."""

    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        causal: bool = True,
    ) -> None:
        super().__init__()
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
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        self.causal = causal

    def forward(self, context_z: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        tokens = torch.where(
            target_mask.unsqueeze(-1).bool(),
            self.mask_token.expand(context_z.shape[0], context_z.shape[1], -1),
            context_z,
        )
        attn_mask = _causal_mask(tokens.shape[1], tokens.device) if self.causal else None
        z = self.temporal(tokens, mask=attn_mask)
        return self.out(self.norm(z))


class TemporalBlockJepaModel(nn.Module):
    """Temporal masked-latent JEPA with optional full target encoder and raw-cue head."""

    def __init__(
        self,
        s2_channels: int,
        s1_channels: int,
        climate_channels: int,
        model_dim: int = 768,
        encoder_hidden: int = 384,
        num_layers: int = 8,
        num_heads: int = 12,
        predictor_layers: int = 2,
        dropout: float = 0.1,
        use_doy: bool = True,
        ema_momentum: float = 0.996,
        full_target_encoder: bool = True,
        transformer_predictor: bool = True,
        raw_cue_dim: int = 50,
    ) -> None:
        super().__init__()
        local = LocalFusionEncoder(
            s2_channels=s2_channels,
            s1_channels=s1_channels,
            climate_channels=climate_channels,
            model_dim=model_dim,
            encoder_hidden=encoder_hidden,
            use_doy=use_doy,
        )
        self.context_encoder = CausalContextEncoder(
            local_encoder=local,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        target_local = copy.deepcopy(local)
        if full_target_encoder:
            target_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=model_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.target_encoder = nn.Sequential(
                target_local,
                nn.TransformerEncoder(target_layer, num_layers=num_layers),
                nn.LayerNorm(model_dim),
            )
        else:
            self.target_encoder = target_local
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        if transformer_predictor:
            self.predictor = TemporalBlockPredictor(
                model_dim=model_dim,
                num_layers=predictor_layers,
                num_heads=num_heads,
                dropout=dropout,
                causal=True,
            )
        else:
            self.predictor = nn.Sequential(
                nn.Linear(model_dim, model_dim * 2),
                nn.GELU(),
                nn.Linear(model_dim * 2, model_dim),
            )
        self.transformer_predictor = transformer_predictor
        self.raw_cue_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, raw_cue_dim),
        )
        self.full_target_encoder = full_target_encoder
        self.ema_momentum = ema_momentum
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> "TemporalBlockJepaModel":
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        m = self.ema_momentum if momentum is None else momentum
        for p_online, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(m).add_(p_online.data, alpha=1.0 - m)

    def _target_forward(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        climate: torch.Tensor,
        doy: torch.Tensor,
        s2_available: torch.Tensor,
        s1_available: torch.Tensor,
        climate_available: torch.Tensor,
    ) -> torch.Tensor:
        target_training = self.target_encoder.training
        self.target_encoder.eval()
        if self.full_target_encoder:
            local = self.target_encoder[0](
                s2=s2,
                s1=s1,
                climate=climate,
                doy=doy,
                s2_available=s2_available,
                s1_available=s1_available,
                climate_available=climate_available,
            )
            out = self.target_encoder[2](self.target_encoder[1](local))
        else:
            out = self.target_encoder(
                s2=s2,
                s1=s1,
                climate=climate,
                doy=doy,
                s2_available=s2_available,
                s1_available=s1_available,
                climate_available=climate_available,
            )
        if target_training:
            self.target_encoder.eval()
        return out

    def forward(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        climate: torch.Tensor,
        doy: torch.Tensor,
        s2_available: torch.Tensor,
        s1_available: torch.Tensor,
        climate_available: torch.Tensor,
        target_mask: torch.Tensor,
        masks: JepaBatchMasks | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = masks or JepaBatchMasks()
        context_time_keep = 1.0 - target_mask.float()
        if masks.time_keep is not None:
            context_time_keep = context_time_keep * masks.time_keep
        context_masks = JepaBatchMasks(
            modality_keep=masks.modality_keep,
            time_keep=context_time_keep,
        )
        context_z = self.context_encoder(
            s2=s2,
            s1=s1,
            climate=climate,
            doy=doy,
            s2_available=s2_available,
            s1_available=s1_available,
            climate_available=climate_available,
            masks=context_masks,
        )
        with torch.no_grad():
            target_z = self._target_forward(
                s2=s2,
                s1=s1,
                climate=climate,
                doy=doy,
                s2_available=s2_available,
                s1_available=s1_available,
                climate_available=climate_available,
            )
        if self.transformer_predictor:
            pred = self.predictor(context_z, target_mask)
        else:
            pred = self.predictor(context_z)
        pooled = context_z.mean(dim=1)
        raw_cue_pred = self.raw_cue_head(pooled)
        return pred, target_z, raw_cue_pred

    def encode(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        climate: torch.Tensor,
        doy: torch.Tensor,
        s2_available: torch.Tensor,
        s1_available: torch.Tensor,
        climate_available: torch.Tensor,
        masks: JepaBatchMasks | None = None,
    ) -> torch.Tensor:
        return self.context_encoder(
            s2=s2,
            s1=s1,
            climate=climate,
            doy=doy,
            s2_available=s2_available,
            s1_available=s1_available,
            climate_available=climate_available,
            masks=masks,
        )


def jepa_cosine_loss(pred_next: torch.Tensor, true_next: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred_next, dim=-1)
    target = F.normalize(true_next.detach(), dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def masked_jepa_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    loss = 1.0 - (pred * target).sum(dim=-1)
    weights = target_mask.float()
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def cosine_ema_momentum(
    step: int,
    total_steps: int,
    base: float = 0.996,
    final: float = 0.9995,
) -> float:
    if total_steps <= 1:
        return final
    progress = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return final - (final - base) * (math.cos(math.pi * progress) + 1.0) / 2.0
