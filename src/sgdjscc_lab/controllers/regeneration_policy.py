"""controllers/regeneration_policy.py – Error-type-aware regeneration policy (Phase 4-A).

Phase 3's regeneration loop retried with a single scalar tweak (scale up
``guidance_scale`` and add a few ``diffusion_step``s) regardless of *why* the
reconstruction failed.  Phase 4-A replaces that with a policy that inspects the
packet-matcher error report and selects strategies keyed to the detected failure
mode:

================  =========================================================
Failure mode      Corrective strategy
================  =========================================================
missing object    strengthen text + object guidance (raise CFG, force text)
hallucination     weaken text CFG, keep / strengthen edge (ControlNet) guide
structural        raise ControlNet scale and diffusion steps (more structure)
================  =========================================================

Each strategy is a small, declarative parameter adjustment (multipliers / deltas
/ flags) applied on top of the run config.  The *selection* logic is pure and
deterministic so it is straightforward to unit-test branching, while
:func:`apply_strategy` produces the concrete retry config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class RegenerationStrategy:
    """A declarative retry adjustment for one failure mode."""

    name: str
    guidance_scale_mult: float = 1.0
    controlnet_scale_mult: float = 1.0
    diffusion_step_delta: int = 0
    force_use_text: Optional[bool] = None   # None = leave as-is
    reason: str = ""

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "guidance_scale_mult": self.guidance_scale_mult,
            "controlnet_scale_mult": self.controlnet_scale_mult,
            "diffusion_step_delta": self.diffusion_step_delta,
            "force_use_text": self.force_use_text,
            "reason": self.reason,
        }


# Default strategy templates per failure mode.
_STRATEGIES: Dict[str, RegenerationStrategy] = {
    "missing_object": RegenerationStrategy(
        name="strengthen_text",
        guidance_scale_mult=1.5,
        controlnet_scale_mult=1.1,
        diffusion_step_delta=10,
        force_use_text=True,
        reason="objects from the original are missing; strengthen text/object guidance",
    ),
    "hallucination": RegenerationStrategy(
        name="weaken_text_strengthen_edge",
        guidance_scale_mult=0.6,
        controlnet_scale_mult=1.4,
        diffusion_step_delta=0,
        force_use_text=True,
        reason="hallucinated objects added; reduce text CFG, keep edge guidance strong",
    ),
    "structural": RegenerationStrategy(
        name="strengthen_structure",
        guidance_scale_mult=1.0,
        controlnet_scale_mult=1.5,
        diffusion_step_delta=15,
        reason="structural / relational distortion; increase control signal and steps",
    ),
}


class RegenerationPolicy:
    """Select error-type-aware regeneration strategies from an error report.

    Parameters
    ----------
    strategies:
        Optional override map ``failure_mode -> RegenerationStrategy``.
    structural_clip_threshold:
        CLIP image-image similarity below which a structural failure is inferred
        even when object counts look fine.
    """

    def __init__(
        self,
        strategies: Optional[Dict[str, RegenerationStrategy]] = None,
        structural_clip_threshold: float = 0.6,
    ) -> None:
        self.strategies = strategies or dict(_STRATEGIES)
        self.structural_clip_threshold = structural_clip_threshold

    def select(
        self,
        error_report: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
    ) -> List[RegenerationStrategy]:
        """Return an ordered list of strategies for the detected failure modes.

        Parameters
        ----------
        error_report:
            Output of ``evaluators/semantic_packet_matcher.compare`` (optional).
        metrics:
            Per-image metric dict (optional); used as a fallback signal when no
            packet report is available (e.g. low ``clip_image_image`` → structural).

        Returns
        -------
        list of :class:`RegenerationStrategy`, ordered by severity.  Empty when no
        failure mode is detected.
        """
        report = error_report or {}
        metrics = metrics or {}
        modes: List[str] = []

        missing = _count(report, "missing_objects", "missing_object_count")
        additional = _count(report, "additional_objects", "additional_object_count")
        relation_errors = int(report.get("relation_error_count", 0) or 0)
        attribute_errors = int(report.get("attribute_error_count", 0) or 0)
        scene_match = report.get("scene_match", True)

        if missing > 0:
            modes.append("missing_object")
        if additional > 0:
            modes.append("hallucination")
        if relation_errors > 0 or attribute_errors > 0 or scene_match is False:
            modes.append("structural")

        # Fallback: structural failure from low image-image CLIP similarity.
        if not modes:
            clip_img = metrics.get("clip_image_image")
            if clip_img is not None and float(clip_img) < self.structural_clip_threshold:
                modes.append("structural")

        return [self.strategies[m] for m in modes if m in self.strategies]


def _count(report: Dict, list_key: str, count_key: str) -> int:
    """Robustly read a count from either a list field or an explicit count field."""
    if count_key in report and report[count_key] is not None:
        return int(report[count_key])
    val = report.get(list_key)
    if isinstance(val, (list, tuple, set)):
        return len(val)
    return 0


def apply_strategy(cfg: DictConfig, strategy: RegenerationStrategy) -> DictConfig:
    """Return a deep copy of *cfg* with *strategy* adjustments applied.

    Never mutates the input config; ``diffusion_step`` is clamped to ≥ 1.
    """
    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    out.guidance_scale = round(float(cfg.get("guidance_scale", 4.0)) * strategy.guidance_scale_mult, 6)
    out.controlnet_scale = round(float(cfg.get("controlnet_scale", 0.3)) * strategy.controlnet_scale_mult, 6)
    out.diffusion_step = max(int(cfg.get("diffusion_step", 50)) + strategy.diffusion_step_delta, 1)
    if strategy.force_use_text is not None:
        out.use_text = bool(strategy.force_use_text)
    return out
