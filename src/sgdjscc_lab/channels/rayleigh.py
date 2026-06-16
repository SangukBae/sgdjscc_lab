"""channels/rayleigh.py – Slow (flat) Rayleigh fading channel (Phase 5-A).

API-compatible with ``AWGNChannel``: ``transmit(latent, snr_db) -> Tensor`` so it
can be dropped onto ``JSCCModel.channel_model`` and flow through the unchanged
SGD-JSCC forward pass.  It additionally exposes ``observe()`` returning a
:class:`MeasurementBundle` (DiffCom-style receiver evidence).

Model (per sample, flat fading over the latent block):

    y = g · x + n,    g = |h|,  h ~ CN(0, 1)   (so E[g²] = 1)

with AWGN ``n`` matched to *snr_db*.  With perfect CSI the equalised latent uses
the paper's MMSE + power-normalization form ``y / sqrt(g² + σ²)`` (see
:func:`~sgdjscc_lab.channels.measurement.mmse_equalize`); with no CSI the receiver
only sees ``y`` (``transmit`` then returns the un-equalised latent, exercising the
blind path).  The gain is real (``g = |h|``), so the complex phase rotation of a
true ``h ∈ ℂ`` channel is not modelled (see ``mmse_equalize``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from sgdjscc_lab.channels.measurement import (
    ChannelTape, MeasurementBundle, awgn_noise_like, mmse_equalize,
)


class RayleighChannel(ChannelTape):
    """Slow flat Rayleigh fading.

    Parameters
    ----------
    csi:
        ``"perfect"`` (equalise with the true gain), ``"none"`` (no equalisation;
        blind), or ``"imperfect"`` (equalise with a noisy gain estimate).
    csi_error_std:
        Std of the multiplicative gain-estimate error when ``csi="imperfect"``.
    """

    def __init__(self, csi: str = "perfect", csi_error_std: float = 0.1) -> None:
        self.csi = csi
        self.csi_error_std = csi_error_std
        self._init_tape()   # record/replay support (shared realisation)

    # ── AWGN-compatible entry point ──────────────────────────────────────────
    def transmit(self, latent: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Return the receiver latent used by the forward pass (equalised iff CSI).

        Routed through the tape so a recorded realisation can be replayed during
        channel-conditioned reconstruction (see ``ChannelTape``).
        """
        return self._taped_transmit(latent, snr_db)

    # ── Rich observation ─────────────────────────────────────────────────────
    def observe(self, latent: torch.Tensor, snr_db: float) -> MeasurementBundle:
        bsz, c, h, w = latent.shape
        # Per-sample Rayleigh gain magnitude with E[g²] = 1.
        a = torch.randn(bsz, 1, 1, 1, device=latent.device)
        b = torch.randn(bsz, 1, 1, 1, device=latent.device)
        g = torch.sqrt((a ** 2 + b ** 2) / 2.0)             # [B,1,1,1]

        faded = g * latent
        noise, noise_var = awgn_noise_like(faded, snr_db)
        received = faded + noise

        equalized, g_used = self._equalize(received, g, noise_var)
        # Per-element noise level d_i = σ²/(g²+σ²) (paper eq. 12). Flat fading → the
        # same value for every element (a uniform-d case where water-filling reduces
        # to the standard step-matched sampler). Use the SAME gain estimate that
        # produced `equalized` so (f̃, d) stay a consistent pair (matters for
        # csi="imperfect", where g_used is the noisy estimate, not the true gain).
        g_for_d = g_used if g_used is not None else g
        noise_level = (noise_var / (g_for_d ** 2 + noise_var + 1e-12)).repeat(1, c, h, w)
        reliability = (g / (g + 1.0)).repeat(1, 1, h, w)     # [B,1,H,W] in (0,1)

        return MeasurementBundle(
            received=received,
            equalized=equalized,
            channel_gain=g,
            noise_var=noise_var,
            noise_level=noise_level,
            mask=torch.ones(bsz, 1, h, w, device=latent.device),
            snr_db_true=float(snr_db),
            reliability=reliability,
            meta={"channel": "rayleigh", "csi": self.csi},
        )

    def _equalize(self, received: torch.Tensor, gain: torch.Tensor,
                  noise_var: torch.Tensor):
        """Return ``(equalized, g_hat)`` — the equalized latent and the gain
        ESTIMATE used to produce it (so callers derive a consistent noise level).
        ``(None, None)`` when blind (``csi="none"``)."""
        if self.csi == "none":
            return None, None
        g_hat = gain
        if self.csi == "imperfect":
            g_hat = gain * (1.0 + self.csi_error_std * torch.randn_like(gain))
        # Paper MMSE + power normalization (real gain): y / sqrt(g² + σ²).
        return mmse_equalize(received, g_hat, noise_var), g_hat
