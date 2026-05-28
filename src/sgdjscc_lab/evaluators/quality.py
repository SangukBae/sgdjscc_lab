"""evaluators/quality.py – Pixel-quality metric wrappers (Phase 3).

Supported metrics
-----------------
PSNR  : compute_psnr(original, reconstructed, max_val=1.0) -> float
SSIM  : compute_ssim(original, reconstructed) -> float
LPIPS : compute_lpips(original, reconstructed, net='vgg', device=None) -> float

Input convention
----------------
All functions accept ``[N, 3, H, W]`` float tensors in [0, 1].
3-D inputs ``[3, H, W]`` are automatically expanded to batch-1.
Shape mismatch raises ValueError.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────────────────────────────────────

def _validate(original: torch.Tensor, reconstructed: torch.Tensor) -> tuple:
    """Ensure inputs are 4-D, float, and shape-compatible.

    Returns (original, reconstructed) as [N, 3, H, W] tensors.
    """
    if original.ndim == 3:
        original = original.unsqueeze(0)
    if reconstructed.ndim == 3:
        reconstructed = reconstructed.unsqueeze(0)
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original={tuple(original.shape)}, "
            f"reconstructed={tuple(reconstructed.shape)}"
        )
    if original.ndim != 4 or original.shape[1] != 3:
        raise ValueError(
            f"Expected [N, 3, H, W] tensors, got shape {tuple(original.shape)}"
        )
    return original.float(), reconstructed.float()


# ─────────────────────────────────────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """Peak Signal-to-Noise Ratio (batch average).

    Parameters
    ----------
    original, reconstructed:
        ``[N, 3, H, W]`` or ``[3, H, W]`` float tensors in [0, max_val].
    max_val:
        Maximum pixel value (default 1.0 for [0, 1]-normalised inputs).

    Returns
    -------
    float
        Mean PSNR over the batch (dB). Returns ``inf`` when MSE = 0.
    """
    original, reconstructed = _validate(original, reconstructed)
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse))


# ─────────────────────────────────────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """Structural Similarity Index Measure (batch average).

    Parameters
    ----------
    original, reconstructed:
        ``[N, 3, H, W]`` float tensors in [0, 1].

    Returns
    -------
    float
        Mean SSIM in [0, 1]. Uses ``pytorch_msssim`` when available;
        falls back to ``skimage.metrics.structural_similarity``.
    """
    original, reconstructed = _validate(original, reconstructed)
    o = original.clamp(0, 1)
    r = reconstructed.clamp(0, 1)

    try:
        from pytorch_msssim import ssim as ms_ssim
        return float(ms_ssim(o, r, data_range=1.0))
    except ImportError:
        pass

    from skimage.metrics import structural_similarity

    o_np = o.cpu().numpy()
    r_np = r.cpu().numpy()
    scores = []
    for oi, ri in zip(o_np, r_np):
        oi = np.transpose(oi, (1, 2, 0))
        ri = np.transpose(ri, (1, 2, 0))
        scores.append(
            structural_similarity(oi, ri, channel_axis=-1, data_range=1.0)
        )
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# LPIPS
# ─────────────────────────────────────────────────────────────────────────────

def compute_lpips(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    net: str = "vgg",
    device: Optional[torch.device] = None,
) -> float:
    """Learned Perceptual Image Patch Similarity (batch average).

    Requires the ``lpips`` package (``pip install lpips``).

    Parameters
    ----------
    original, reconstructed:
        ``[N, 3, H, W]`` float tensors in [0, 1].
    net:
        Backbone network for LPIPS: ``'vgg'`` (default) or ``'alex'``.
        VGG is closer to perceptual quality; AlexNet is faster.
    device:
        Device to run LPIPS on. Defaults to the device of *original*.

    Returns
    -------
    float
        Mean LPIPS in [0, 1] (lower = more perceptually similar).

    Raises
    ------
    ImportError
        If the ``lpips`` package is not installed.
    """
    try:
        import lpips as lpips_lib
    except ImportError as exc:
        raise ImportError(
            "lpips package not found. Install with: pip install lpips"
        ) from exc

    original, reconstructed = _validate(original, reconstructed)
    if device is None:
        device = original.device

    # lpips expects inputs in [-1, 1]
    o = (original.to(device) * 2 - 1).clamp(-1, 1)
    r = (reconstructed.to(device) * 2 - 1).clamp(-1, 1)

    loss_fn = lpips_lib.LPIPS(net=net).to(device)
    loss_fn.eval()
    with torch.no_grad():
        dist = loss_fn(o, r)      # [N, 1, 1, 1]
    return float(dist.mean().item())


# ─────────────────────────────────────────────────────────────────────────────
# Convenience class
# ─────────────────────────────────────────────────────────────────────────────

class QualityEvaluator:
    """Batch-persistent wrapper that computes PSNR, SSIM, and optionally LPIPS.

    Parameters
    ----------
    use_lpips:
        If True, also compute LPIPS (requires the ``lpips`` package).
    lpips_net:
        Backbone for LPIPS (``'vgg'`` or ``'alex'``).
    device:
        Device for LPIPS computation; defaults to ``'cpu'``.
    """

    def __init__(
        self,
        use_lpips: bool = True,
        lpips_net: str = "vgg",
        device: Optional[torch.device] = None,
    ) -> None:
        self.use_lpips = use_lpips
        self.lpips_net = lpips_net
        self.device = device or torch.device("cpu")
        self._lpips_fn = None

    def _get_lpips(self):
        if self._lpips_fn is None:
            import lpips as lpips_lib
            self._lpips_fn = lpips_lib.LPIPS(net=self.lpips_net).to(self.device)
            self._lpips_fn.eval()
        return self._lpips_fn

    def evaluate(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> dict:
        """Compute all quality metrics.

        Returns
        -------
        dict with keys: ``psnr``, ``ssim``, and (if use_lpips) ``lpips``.
        Values are scalar floats; ``lpips=None`` when disabled.
        """
        results = {
            "psnr": compute_psnr(original, reconstructed),
            "ssim": compute_ssim(original, reconstructed),
            "lpips": None,
        }
        if self.use_lpips:
            try:
                o, r = _validate(original, reconstructed)
                o = (o.to(self.device) * 2 - 1).clamp(-1, 1)
                r = (r.to(self.device) * 2 - 1).clamp(-1, 1)
                fn = self._get_lpips()
                with torch.no_grad():
                    dist = fn(o, r)
                results["lpips"] = float(dist.mean().item())
            except Exception:
                results["lpips"] = None
        return results
