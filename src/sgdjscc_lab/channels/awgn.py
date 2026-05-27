"""awgn.py – Additive White Gaussian Noise channel model.

Extracted from the inline _JSCCModel.channel() method in SGDJSCC/inference_one.py.
The computation is identical; extraction enables independent replacement with
Rayleigh or other channel models in Phase 5.
"""

from __future__ import annotations

import torch


class AWGNChannel:
    """AWGN channel model.

    Adds signal-power-normalised Gaussian noise to a latent tensor at the
    requested SNR.

    Interface mirrors the proposed ChannelModel.transmit() from the README:
        def transmit(self, latent, snr_db, **kwargs) -> Tensor

    Future channels (e.g. Rayleigh) should follow the same signature.
    """

    def transmit(self, latent: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Apply AWGN noise.

        Exact replication of SGDJSCC/inference_one.py _JSCCModel.channel().

        sigma² = (‖x‖² / d) / SNR_linear
        y      = x + N(0, sigma²·I)

        where d = latent.numel() / batch_size (per-sample element count).
        """
        bsz, c, h, w = latent.shape
        norm_2 = torch.linalg.norm(latent.reshape([bsz, -1]), ord=2, dim=1)
        noise_sigma = torch.sqrt(
            (norm_2 ** 2 / (latent.numel() / bsz)) / (10 ** (snr_db / 10))
        ).reshape([-1, 1, 1, 1]).repeat([1, c, h, w])
        return latent + torch.randn_like(latent) * noise_sigma
