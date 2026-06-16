"""video/scene_change_detector.py – Heuristic scene-change detection (Phase 4-B).

Marks scene boundaries in an ordered frame sequence so the keyframe extractor can
start a new GOP whenever the content changes substantially.  Phase 4-B uses a
practical heuristic (no trained shot detector) combining up to three signals
between consecutive frames:

- **colour histogram delta** – pure-torch, always available, fast.
- **CLIP image-image distance** – ``1 − cosine_similarity`` (optional; needs CLIP).
- **LPIPS** – perceptual distance (optional; needs the ``lpips`` package).

The signals are weighted into a combined score; a frame is a scene boundary when
its score exceeds ``threshold``.  Frame 0 is always a boundary.  The histogram
path keeps the detector usable — and unit-testable — without any model weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def color_histogram(frame: torch.Tensor, bins: int = 16) -> torch.Tensor:
    """Return a normalised per-channel colour histogram for a ``[*,3,H,W]`` frame.

    The result is a flat ``[3*bins]`` tensor summing to 1.
    """
    f = frame.float().clamp(0, 1)
    if f.dim() == 4:
        f = f[0]
    hists = []
    for c in range(f.shape[0]):
        h = torch.histc(f[c], bins=bins, min=0.0, max=1.0)
        hists.append(h)
    hist = torch.cat(hists)
    total = hist.sum()
    return hist / total if total > 0 else hist


def histogram_distance(a: torch.Tensor, b: torch.Tensor, bins: int = 16) -> float:
    """L1 colour-histogram distance in [0, 1] between two frames.

    ``color_histogram`` returns a single distribution summing to 1, so the L1
    distance between two histograms lies in [0, 2]; dividing by 2 maps it to
    [0, 1] (0 = identical colour distribution, 1 = disjoint).
    """
    ha = color_histogram(a, bins)
    hb = color_histogram(b, bins)
    return float((ha - hb).abs().sum().item() / 2.0)


@dataclass
class SceneChangeConfig:
    threshold: float = 0.35
    hist_weight: float = 1.0
    clip_weight: float = 0.0
    lpips_weight: float = 0.0
    hist_bins: int = 16


class SceneChangeDetector:
    """Detect scene boundaries in an ordered frame sequence.

    Parameters
    ----------
    config:
        :class:`SceneChangeConfig` of weights and the decision threshold.
    clip_evaluator:
        Optional ``CLIPScoreEvaluator`` enabling the CLIP image-image signal.
    use_lpips:
        Enable the LPIPS signal (requires the ``lpips`` package).
    device:
        Compute device for LPIPS.
    """

    def __init__(
        self,
        config: Optional[SceneChangeConfig] = None,
        clip_evaluator=None,
        use_lpips: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or SceneChangeConfig()
        self._clip = clip_evaluator
        self.use_lpips = use_lpips
        self._device = device or torch.device("cpu")

    # ── Pairwise distance ────────────────────────────────────────────────────

    def frame_distance(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> Dict:
        """Return per-signal and combined distance between two frames."""
        cfg = self.config
        hist = histogram_distance(frame_a, frame_b, cfg.hist_bins)

        clip_d = None
        if self._clip is not None and cfg.clip_weight > 0:
            try:
                sim = self._clip.image_image_score(frame_a, frame_b)
                clip_d = max(0.0, 1.0 - float(sim))
            except Exception as exc:  # noqa: BLE001
                logger.warning("CLIP scene signal failed: %s", exc)

        lpips_d = None
        if self.use_lpips and cfg.lpips_weight > 0:
            try:
                from sgdjscc_lab.evaluators.quality import compute_lpips
                lpips_d = compute_lpips(frame_a, frame_b, device=self._device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LPIPS scene signal failed: %s", exc)

        # Weighted combination over whichever signals are present.
        num = cfg.hist_weight * hist
        den = cfg.hist_weight
        if clip_d is not None:
            num += cfg.clip_weight * clip_d
            den += cfg.clip_weight
        if lpips_d is not None:
            num += cfg.lpips_weight * lpips_d
            den += cfg.lpips_weight
        combined = float(num / den) if den > 0 else float(hist)

        return {"hist": hist, "clip": clip_d, "lpips": lpips_d, "combined": combined}

    # ── Sequence detection ───────────────────────────────────────────────────

    def detect(self, frames: List[torch.Tensor]) -> Dict:
        """Detect scene boundaries across an ordered list of frames.

        Returns
        -------
        dict with keys:
            ``boundaries`` – list[bool], True where a new scene begins (idx 0 = True).
            ``distances``  – list[float] combined distance vs previous frame
                             (distances[0] = 0.0).
            ``signals``    – list[dict] per-frame raw signal breakdown.
        """
        n = len(frames)
        boundaries = [False] * n
        distances = [0.0] * n
        signals: List[Dict] = [{"hist": 0.0, "clip": None, "lpips": None, "combined": 0.0}]

        if n == 0:
            return {"boundaries": [], "distances": [], "signals": []}
        boundaries[0] = True

        for i in range(1, n):
            d = self.frame_distance(frames[i - 1], frames[i])
            distances[i] = d["combined"]
            signals.append(d)
            if d["combined"] >= self.config.threshold:
                boundaries[i] = True

        return {"boundaries": boundaries, "distances": distances, "signals": signals}
