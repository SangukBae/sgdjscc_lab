"""evaluators/quality.py – Pixel-quality metric wrappers (Phase 3+).

Currently provides thin wrappers over skimage / pytorch-msssim.
Full metric integration (LPIPS, CLIP, object preservation, hallucination)
will be added in Phase 3.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch


def compute_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """Peak Signal-to-Noise Ratio.

    Parameters
    ----------
    original, reconstructed:
        ``[N, 3, H, W]`` or ``[3, H, W]`` float tensors in [0, max_val].

    Returns
    -------
    float
        Average PSNR across the batch (dB).
    """
    mse = torch.mean((original.float() - reconstructed.float()) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse))


def compute_ssim(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """Structural Similarity Index Measure.

    Delegates to pytorch_msssim.ssim when available; falls back to a
    skimage-based computation otherwise.

    Parameters
    ----------
    original, reconstructed:
        ``[N, 3, H, W]`` float tensors in [0, 1].

    Returns
    -------
    float
        SSIM value in [0, 1].

    Note
    ----
    Full SSIM/LPIPS/CLIP integration is planned for Phase 3.
    """
    try:
        from pytorch_msssim import ssim as ms_ssim
        return float(
            ms_ssim(
                original.float().clamp(0, 1),
                reconstructed.float().clamp(0, 1),
                data_range=1.0,
            )
        )
    except ImportError:
        pass

    # Fallback: skimage on CPU
    from skimage.metrics import structural_similarity

    orig_np  = original.float().clamp(0, 1).cpu().numpy()
    recon_np = reconstructed.float().clamp(0, 1).cpu().numpy()

    scores = []
    for o, r in zip(orig_np, recon_np):
        o = np.transpose(o, (1, 2, 0))
        r = np.transpose(r, (1, 2, 0))
        scores.append(
            structural_similarity(o, r, channel_axis=-1, data_range=1.0)
        )
    return float(np.mean(scores))
