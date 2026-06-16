"""video/motion_residual.py – Lightweight motion / residual estimation (Phase 4-B).

Pixel-level companion to ``semantic_delta.py``: where the semantic delta measures
*what* changed (objects, relations, scene), the motion residual measures *how
much* the pixels moved.  It is a cheap, model-free signal the temporal pipeline
can use to decide how strongly to reuse a keyframe reconstruction for an
inter-frame.

Two estimates are produced:

- **residual energy** – mean absolute frame difference (global change magnitude).
- **block motion**    – per-block mean-abs-diff energy on a coarse grid, whose
                        mean approximates motion intensity and whose max flags a
                        localised moving region.

No optical-flow library is required (kept inside the SGD-JSCC env's dependency
budget); a true flow estimator can replace ``estimate`` later behind the same
return contract.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def residual_energy(prev: torch.Tensor, curr: torch.Tensor) -> float:
    """Mean absolute difference between two ``[*,3,H,W]`` frames in [0, 1]."""
    a = prev.float()
    b = curr.float()
    if a.shape != b.shape:
        b = F.interpolate(b if b.dim() == 4 else b.unsqueeze(0),
                          size=a.shape[-2:], mode="bilinear", align_corners=False)
        if a.dim() == 3:
            b = b[0]
    return float((a - b).abs().mean().item())


def block_motion(prev: torch.Tensor, curr: torch.Tensor, grid: int = 8) -> Dict:
    """Per-block residual energy on a ``grid×grid`` partition.

    Returns ``{"mean", "max", "map"}`` where ``map`` is a ``[grid, grid]`` list of
    block energies.
    """
    a = prev.float()
    b = curr.float()
    if a.dim() == 3:
        a = a.unsqueeze(0)
    if b.dim() == 3:
        b = b.unsqueeze(0)
    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)

    diff = (a - b).abs().mean(dim=1, keepdim=True)        # [N,1,H,W]
    pooled = F.adaptive_avg_pool2d(diff, output_size=(grid, grid))  # [N,1,g,g]
    pooled = pooled[0, 0]
    return {
        "mean": float(pooled.mean().item()),
        "max": float(pooled.max().item()),
        "map": pooled.tolist(),
    }


def estimate(prev: torch.Tensor, curr: torch.Tensor, grid: int = 8) -> Dict:
    """Combined motion/residual estimate between two frames.

    Returns ``{"residual_energy", "block_mean", "block_max", "block_map"}``.
    """
    bm = block_motion(prev, curr, grid=grid)
    return {
        "residual_energy": residual_energy(prev, curr),
        "block_mean": bm["mean"],
        "block_max": bm["max"],
        "block_map": bm["map"],
    }


class MotionResidualEstimator:
    """OO wrapper around :func:`estimate`."""

    def __init__(self, grid: int = 8) -> None:
        self.grid = grid

    def estimate(self, prev: torch.Tensor, curr: torch.Tensor) -> Dict:
        return estimate(prev, curr, grid=self.grid)
