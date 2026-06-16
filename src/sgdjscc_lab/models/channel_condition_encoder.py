"""models/channel_condition_encoder.py – Channel-condition encoder (Phase 5-A).

Turns a :class:`MeasurementBundle` (receiver evidence) into a compact diffusion
condition: a set of scalar channel descriptors plus a small grid of "condition
tokens".  Inspired by DiffCom treating the received signal as a natural
conditioning input — here we expose it as tokens/scalars that a
channel-conditioned decoder can consume.

Two modes (both runnable **without training**):

- ``"stats"``  : tokens are pooled statistics of the received latent and the
                 reliability map (no learnable parameters → deterministic shapes,
                 zero checkpoints needed).
- ``"linear"`` : a single ``nn.Linear`` projects the per-cell statistics to
                 ``token_dim`` — the hook for a future *learned* encoder. The
                 weights are randomly initialised; only the interface matters at
                 this stage.

Output dict:
    ``scalars``        ``[B, K]``  channel descriptor vector
    ``tokens``         ``[B, T, token_dim]``  T = token_grid²
    ``reliability_map````[B, 1, token_grid, token_grid]``
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgdjscc_lab.channels.measurement import MeasurementBundle

# Number of scalar channel descriptors emitted (see _scalars).
NUM_SCALARS = 5


class ChannelConditionEncoder(nn.Module):
    """Encode receiver evidence into condition tokens + scalars.

    Parameters
    ----------
    token_grid:
        Spatial side length of the token grid (T = token_grid²).
    token_dim:
        Dimensionality of each condition token.
    mode:
        ``"stats"`` (training-free) or ``"linear"`` (learnable projection hook).
    snr_norm:
        SNR normalisation divisor used when forming the scalar descriptor.
    """

    def __init__(
        self,
        token_grid: int = 4,
        token_dim: int = 8,
        mode: str = "stats",
        snr_norm: float = 30.0,
    ) -> None:
        super().__init__()
        self.token_grid = int(token_grid)
        self.token_dim = int(token_dim)
        self.mode = mode
        self.snr_norm = float(snr_norm)
        # Per-cell features are [received_mean, reliability] → 2 dims.
        self._in_feats = 2
        if mode == "linear":
            self.proj = nn.Linear(self._in_feats, token_dim)
        else:
            self.proj = None

    # ── Scalar channel descriptors ───────────────────────────────────────────
    def _scalars(self, bundle: MeasurementBundle, device, bsz) -> torch.Tensor:
        def _mean(t, default):
            if t is None:
                return torch.full((bsz,), float(default), device=device)
            return t.float().reshape(bsz, -1).mean(dim=1)

        snr = bundle.snr_db_est if bundle.snr_db_est is not None else bundle.snr_db_true
        snr_val = 0.0 if snr is None else float(snr) / self.snr_norm
        snr_t = torch.full((bsz,), snr_val, device=device)

        gain = _mean(bundle.channel_gain, 1.0)
        noise = _mean(bundle.noise_var, 0.0)
        noise = torch.log1p(noise.clamp(min=0.0))
        rel = _mean(bundle.reliability if bundle.reliability is not None else bundle.mask, 1.0)
        phase = _mean(bundle.phase_est, 0.0)

        return torch.stack([snr_t, gain, noise, rel, phase], dim=1)  # [B, 5]

    # ── Condition tokens ─────────────────────────────────────────────────────
    def _tokens(self, bundle: MeasurementBundle):
        feat = bundle.best_estimate.float()                       # [B,C,H,W]
        bsz = feat.shape[0]
        g = self.token_grid

        recv_map = feat.mean(dim=1, keepdim=True)                 # [B,1,H,W]
        rel = bundle.reliability
        if rel is None:
            rel = bundle.mask
        if rel is None:
            rel = torch.ones_like(recv_map)
        rel = rel.float()
        if rel.shape[-2:] != recv_map.shape[-2:]:
            rel = F.interpolate(rel, size=recv_map.shape[-2:], mode="nearest")

        recv_pool = F.adaptive_avg_pool2d(recv_map, (g, g))       # [B,1,g,g]
        rel_pool = F.adaptive_avg_pool2d(rel, (g, g))             # [B,1,g,g]

        cells = torch.cat([recv_pool, rel_pool], dim=1)           # [B,2,g,g]
        per_cell = cells.permute(0, 2, 3, 1).reshape(bsz, g * g, self._in_feats)

        if self.mode == "linear":
            tokens = self.proj(per_cell)                          # [B,T,token_dim]
        else:
            # Training-free: tile/pad the 2-dim per-cell features to token_dim.
            reps = math.ceil(self.token_dim / self._in_feats)
            tokens = per_cell.repeat(1, 1, reps)[:, :, : self.token_dim]
        return tokens, rel_pool

    def encode(self, bundle: MeasurementBundle) -> Dict:
        """Encode a measurement bundle into ``{scalars, tokens, reliability_map}``."""
        device = bundle.best_estimate.device
        bsz = bundle.best_estimate.shape[0]
        scalars = self._scalars(bundle, device, bsz)
        tokens, rel_map = self._tokens(bundle)
        return {"scalars": scalars, "tokens": tokens, "reliability_map": rel_map}

    # nn.Module.forward delegates to encode for convenience.
    def forward(self, bundle: MeasurementBundle) -> Dict:  # noqa: D401
        return self.encode(bundle)
