"""channels/fast_fading.py – Block/fast Rayleigh fading channel (Phase 5-A).

Like ``RayleighChannel`` but the gain varies *within* a latent rather than being
constant per sample: the flattened latent is split into blocks of
``block_length`` elements, each drawn its own Rayleigh gain.  ``block_length=1``
gives per-symbol fast fading; a large block length approaches slow fading.

API-compatible with ``AWGNChannel`` (``transmit(latent, snr_db) -> Tensor``) and
exposes ``observe()`` returning a :class:`MeasurementBundle` whose reliability map
reflects the per-block gain (useful as a per-token reliability proxy for the
channel-condition encoder).

Equalization here is the paper's per-element MMSE + power normalization
(``y / sqrt(g² + σ²)``), and ``observe()`` exposes the per-element noise level
``noise_level`` (``d_i = σ²/(g_i²+σ²)``, paper eq. 12).

The paper's fast-fading *training-free water-filling denoising* (Algorithm 4) is
implemented in ``acceleration/water_filling.py`` (it consumes ``equalized`` +
``noise_level`` from this bundle). This module provides the channel model +
equalization + per-element noise levels; the Algorithm-4 denoising LOOP lives
there. Remaining: wiring the real MDTv2 f0-predictor into that loop (the loop
itself takes an injectable f0-predictor and is verified with a synthetic one).
"""

from __future__ import annotations

from typing import Optional

import torch

from sgdjscc_lab.channels.measurement import (
    ChannelTape, MeasurementBundle, awgn_noise_like, mmse_equalize,
)


class FastFadingChannel(ChannelTape):
    """Block-wise Rayleigh fading.

    Parameters
    ----------
    block_length:
        Number of consecutive latent elements sharing one gain (>=1).
    csi:
        ``"perfect"`` | ``"none"`` | ``"imperfect"`` (see ``RayleighChannel``).
    csi_error_std:
        Gain-estimate error std for ``csi="imperfect"``.
    """

    def __init__(self, block_length: int = 64, csi: str = "perfect",
                 csi_error_std: float = 0.1) -> None:
        self.block_length = max(int(block_length), 1)
        self.csi = csi
        self.csi_error_std = csi_error_std
        self._init_tape()

    def transmit(self, latent: torch.Tensor, snr_db: float) -> torch.Tensor:
        return self._taped_transmit(latent, snr_db)

    def observe(self, latent: torch.Tensor, snr_db: float) -> MeasurementBundle:
        bsz, c, h, w = latent.shape
        flat = latent.reshape(bsz, -1)
        n = flat.shape[1]
        n_blocks = (n + self.block_length - 1) // self.block_length

        # Per-block Rayleigh gains expanded to per-element.
        a = torch.randn(bsz, n_blocks, device=latent.device)
        b = torch.randn(bsz, n_blocks, device=latent.device)
        g_block = torch.sqrt((a ** 2 + b ** 2) / 2.0)            # [B, n_blocks]
        g = g_block.repeat_interleave(self.block_length, dim=1)[:, :n]   # [B, n]

        faded = (flat * g).reshape(bsz, c, h, w)
        noise, noise_var = awgn_noise_like(faded, snr_db)
        received = faded + noise

        g_map = g.reshape(bsz, c, h, w)
        equalized, g_used = self._equalize(received, g_map, noise_var)
        # Per-element noise level d_i = σ²/(g_i²+σ²) of the equalized latent
        # (paper eq. 12) — the heterogeneous levels the water-filling denoiser needs.
        # Use the SAME per-element gain estimate that produced `equalized` so the
        # (f̃, d) pair stays consistent under csi="imperfect".
        g_for_d = g_used if g_used is not None else g_map
        noise_level = noise_var / (g_for_d ** 2 + noise_var + 1e-12)
        # Reliability: per-element gain collapsed across channels → [B,1,H,W].
        reliability = (g_map / (g_map + 1.0)).mean(dim=1, keepdim=True)

        return MeasurementBundle(
            received=received,
            equalized=equalized,
            channel_gain=g_block.mean(dim=1).reshape(-1, 1, 1, 1),
            noise_var=noise_var,
            noise_level=noise_level,
            mask=torch.ones(bsz, 1, h, w, device=latent.device),
            snr_db_true=float(snr_db),
            reliability=reliability,
            meta={"channel": "fast_fading", "csi": self.csi,
                  "block_length": self.block_length},
        )

    def _equalize(self, received: torch.Tensor, gain_map: torch.Tensor,
                  noise_var: torch.Tensor):
        """Return ``(equalized, g_hat)`` — the per-element equalized latent and the
        gain ESTIMATE map used (so the noise level is derived consistently).
        ``(None, None)`` when blind (``csi="none"``).

        NOTE: this is per-element MMSE equalization, NOT the fast-fading
        water-filling denoising of Algorithm 4 (that lives in
        ``acceleration/water_filling.py``)."""
        if self.csi == "none":
            return None, None
        g_hat = gain_map
        if self.csi == "imperfect":
            g_hat = gain_map * (1.0 + self.csi_error_std * torch.randn_like(gain_map))
        # Paper MMSE + power normalization (real gain): y / sqrt(g² + σ²).
        return mmse_equalize(received, g_hat, noise_var), g_hat
