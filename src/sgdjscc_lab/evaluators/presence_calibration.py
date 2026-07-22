"""evaluators/presence_calibration.py – Presence ensemble / calibration (ETRI 5차, step 8).

Combines one or more :class:`~sgdjscc_lab.evaluators.presence_backends.PresenceBackend`
results into a single calibrated presence decision per object.

Scope note: this is calibration *structure*, not a verified accuracy
improvement. With the default backend set (``clip`` only, or ``mock`` in
tests) the ensemble modes degenerate to whatever that one backend says — they
only start doing real ensembling once a second backend (OWLv2/VQA/GT) is
actually configured and available. See docs/etri_strategy.md 5차 구현 결과 for
what "verified" does and does not mean here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from sgdjscc_lab.evaluators.presence_backends import (
    PresenceBackend, PresenceBackendUnavailableError, PresenceResult,
)

logger = logging.getLogger(__name__)

MODE_CLIP_ONLY = "clip_only"
MODE_OWLV2_ONLY = "owlv2_only"
MODE_VQA_ONLY = "vqa_only"
MODE_GT_ONLY = "gt_only"
MODE_ENSEMBLE_MAJORITY = "ensemble_majority"
MODE_ENSEMBLE_WEIGHTED = "ensemble_weighted"

_SINGLE_BACKEND_MODES = {
    MODE_CLIP_ONLY: "clip",
    MODE_OWLV2_ONLY: "owlv2",
    MODE_VQA_ONLY: "vqa",
    MODE_GT_ONLY: "gt",
}
MODES = (
    MODE_CLIP_ONLY, MODE_OWLV2_ONLY, MODE_VQA_ONLY, MODE_GT_ONLY,
    MODE_ENSEMBLE_MAJORITY, MODE_ENSEMBLE_WEIGHTED,
)


@dataclass
class CalibratedPresence:
    """Result of combining one or more backend :class:`PresenceResult`s."""

    object_name: str
    final_present: bool
    final_confidence: float
    contributing_backends: List[str] = field(default_factory=list)
    per_backend: List[Dict] = field(default_factory=list)
    mode: str = ""

    def to_dict(self) -> Dict:
        return {
            "object_name": self.object_name,
            "final_present": bool(self.final_present),
            "final_confidence": float(self.final_confidence),
            "contributing_backends": list(self.contributing_backends),
            "per_backend": list(self.per_backend),
            "mode": self.mode,
        }


class PresenceCalibrator:
    """Combine configured :class:`PresenceBackend`\\ s into one decision per object.

    Parameters
    ----------
    backends:
        ``{backend_name: PresenceBackend instance}``. Only backends actually
        needed by ``mode`` (or all of them, for ensemble modes) are queried.
    mode:
        One of :data:`MODES`. Default ``"clip_only"`` — same presence judge
        1~4차 already used, just routed through the common interface.
    weights:
        Optional per-backend weights for ``ensemble_weighted`` (default 1.0
        for any backend not listed).
    threshold / uncertain_band:
        Compatible with the existing ``object_presence_threshold`` /
        ``object_presence_uncertain_band`` config keys (see
        ``evaluators/object_preservation.py``). ``threshold`` is the final
        bar ``ensemble_weighted``'s combined confidence must clear to count
        as present. ``uncertain_band`` is not applied inside this class (a
        calibrator judges one image at a time, with no "was it seen before"
        state) — callers that need original→reconstruction hysteresis (e.g.
        ``PacketVerifier``'s missing/additional recheck) read
        ``final_confidence`` and apply the band themselves.
    """

    def __init__(
        self,
        backends: Dict[str, PresenceBackend],
        mode: str = MODE_CLIP_ONLY,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.25,
        uncertain_band: float = 0.0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown PresenceCalibrator mode={mode!r}; expected one of {MODES}.")
        self.backends = dict(backends)
        self.mode = mode
        self.weights = dict(weights or {})
        self.threshold = float(threshold)
        self.uncertain_band = max(float(uncertain_band), 0.0)

    def _backend_order(self) -> List[str]:
        single = _SINGLE_BACKEND_MODES.get(self.mode)
        if single is not None:
            return [single]
        return list(self.backends.keys())   # ensemble modes: everything configured

    def calibrate(
        self,
        object_name: str,
        image: Optional[torch.Tensor] = None,
        packet: Optional[Dict] = None,
        gt_metadata: Optional[Dict] = None,
    ) -> CalibratedPresence:
        """Query the configured backend(s) for *object_name* and combine.

        ``image`` may be ``None`` — this is a normal, supported call (e.g.
        held-out remeasurement from saved packets with no pixels available):
        image-based backends (clip/owlv2/vqa) then raise
        :class:`PresenceBackendUnavailableError` and are skipped, while
        image-free backends (mock/gt) still answer normally. ``gt_metadata``
        is forwarded to every backend's ``check()`` (only ``gt`` uses it).

        Raises :class:`PresenceBackendUnavailableError` when no backend could
        answer (e.g. the single backend required by a ``*_only`` mode is not
        configured/available, or every ensemble backend failed).
        """
        results: List[PresenceResult] = []
        for name in self._backend_order():
            backend = self.backends.get(name)
            if backend is None:
                logger.debug("PresenceCalibrator: backend %r not configured, skipping.", name)
                continue
            try:
                results.append(backend.check(object_name, image=image, packet=packet, gt_metadata=gt_metadata))
            except PresenceBackendUnavailableError as exc:
                logger.debug("PresenceCalibrator: backend %r unavailable for %r: %s", name, object_name, exc)
                continue

        if not results:
            raise PresenceBackendUnavailableError(
                f"No presence backend could evaluate object_name={object_name!r} "
                f"in mode={self.mode!r} (configured backends: {sorted(self.backends)})."
            )

        present, confidence = self._combine(results)
        return CalibratedPresence(
            object_name=object_name,
            final_present=present,
            final_confidence=confidence,
            contributing_backends=[r.backend for r in results],
            per_backend=[r.to_dict() for r in results],
            mode=self.mode,
        )

    def _combine(self, results: List[PresenceResult]):
        if self.mode in _SINGLE_BACKEND_MODES:
            r = results[0]
            return bool(r.present), float(r.confidence)

        if self.mode == MODE_ENSEMBLE_MAJORITY:
            votes = sum(1 for r in results if r.present)
            present = votes * 2 > len(results)          # strict majority; ties → False
            confidence = float(sum(r.confidence for r in results) / len(results))
            return present, confidence

        if self.mode == MODE_ENSEMBLE_WEIGHTED:
            total_w = 0.0
            acc = 0.0
            for r in results:
                w = float(self.weights.get(r.backend, 1.0))
                acc += w * r.confidence
                total_w += w
            confidence = float(acc / total_w) if total_w > 0 else 0.0
            present = confidence >= self.threshold
            return present, confidence

        raise ValueError(f"Unhandled PresenceCalibrator mode={self.mode!r}")  # pragma: no cover


def build_presence_calibrator(cfg) -> Optional[PresenceCalibrator]:
    """Build a :class:`PresenceCalibrator` from ``verifier.*`` cfg keys.

    Returns ``None`` when ``verifier.use_presence_calibration`` is false
    (default) — callers should treat ``None`` exactly like "calibration
    disabled, use the legacy CLIP-derived packet comparison unchanged".
    """
    from omegaconf import OmegaConf
    from sgdjscc_lab.evaluators.presence_backends import build_presence_backend

    if not bool(OmegaConf.select(cfg, "verifier.use_presence_calibration", default=False)):
        return None

    mode = str(OmegaConf.select(cfg, "verifier.presence_mode", default=MODE_CLIP_ONLY))
    backend_names = OmegaConf.select(cfg, "verifier.presence_backends", default=["clip"])
    backend_names = list(backend_names) if backend_names else ["clip"]

    backends: Dict[str, PresenceBackend] = {}
    for name in backend_names:
        try:
            backends[name] = build_presence_backend(name, cfg=OmegaConf.select(cfg, "verifier.presence_backend_cfg", default=None))
        except NotImplementedError as exc:
            logger.warning("Skipping unknown presence backend %r: %s", name, exc)

    threshold = float(OmegaConf.select(cfg, "object_presence_threshold", default=0.25))
    uncertain_band = float(OmegaConf.select(cfg, "object_presence_uncertain_band", default=0.0))
    weights = OmegaConf.select(cfg, "verifier.presence_backend_weights", default=None)
    weights = dict(weights) if weights else None

    return PresenceCalibrator(
        backends=backends, mode=mode, weights=weights,
        threshold=threshold, uncertain_band=uncertain_band,
    )
