"""controllers/verifier_controller.py – Error-type-aware verifier controller (ETRI 2차, step 7).

Turns a ``PacketVerifier`` report (``evaluators/packet_verifier.py`` — a
``semantic_packet_matcher`` comparison plus a ``severity`` scalar) into one of
a small set of controller *decisions*, plus a list of declarative "candidate
actions" (e.g. negative-prompt / prompt-emphasis candidates).

Scope note (2차): this module decides and logs; it does **not** wire
candidate actions into an actual sampler call (no negative-prompt injection,
no prompt-emphasis application, no forced recompute). That wiring — and the
stronger regeneration behaviour it implies — is 3~4차+ follow-up work per
``docs/etri_strategy.md``. Candidate actions are recorded as plain dicts so a
future consumer can act on them without this module needing to change.

Decision set
------------
``accept``                        errors below threshold; keep the reconstruction as-is.
``suppress_extra``                additional/hallucinated objects dominate the report.
``strengthen_missing``            missing objects dominate the report.
``strengthen_structure_guidance`` relation / attribute / scene errors dominate.
``fallback_recompute``            overall severity is high regardless of error type.
``keyframe_fallback``             severity is extreme AND the item is a video
                                   inter-frame (fall back to the keyframe reconstruction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VerifierControllerConfig:
    """Thresholds driving :class:`VerifierController` decisions."""

    accept_severity: float = 0.15
    fallback_severity: float = 0.6
    keyframe_fallback_severity: float = 0.85
    missing_object_threshold: int = 1
    additional_object_threshold: int = 1
    structural_error_threshold: int = 1


@dataclass
class ControllerDecision:
    """One controller decision + its supporting evidence (dict/JSON-serialisable)."""

    decision: str
    severity: float
    triggered_modes: List[str] = field(default_factory=list)
    candidate_actions: List[Dict] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "controller_decision": self.decision,
            "severity": self.severity,
            "triggered_modes": list(self.triggered_modes),
            "candidate_actions": list(self.candidate_actions),
            "reason": self.reason,
        }


def _count(report: Dict, list_key: str, count_key: str) -> int:
    """Robustly read a count from either a list field or an explicit count field."""
    if count_key in report and report[count_key] is not None:
        return int(report[count_key])
    val = report.get(list_key)
    if isinstance(val, (list, tuple, set)):
        return len(val)
    return 0


class VerifierController:
    """Rule-based, error-type-aware decision policy driven off a verifier report.

    Parameters
    ----------
    config:
        Thresholds (see :class:`VerifierControllerConfig`). Defaults are
        conservative placeholders meant to be tuned per dataset/experiment.
    """

    def __init__(self, config: Optional[VerifierControllerConfig] = None) -> None:
        self.config = config or VerifierControllerConfig()

    def decide(self, report: Dict, is_interframe: bool = False) -> ControllerDecision:
        """Return a :class:`ControllerDecision` for one packet-verifier *report*.

        Parameters
        ----------
        report:
            Output of ``PacketVerifier.verify()`` (must contain ``severity``;
            falls back to computing severity as 0.0 if absent).
        is_interframe:
            Whether this item is a video inter-frame (enables
            ``keyframe_fallback`` as a candidate decision). Ignored for
            single-image use (always False there).
        """
        cfg = self.config
        severity = float(report.get("severity", 0.0) or 0.0)

        missing = _count(report, "missing_objects", "missing_object_count")
        additional = _count(report, "additional_objects", "additional_object_count")
        relation_errors = int(report.get("relation_error_count", 0) or 0)
        attribute_errors = int(report.get("attribute_error_count", 0) or 0)
        scene_mismatch = not report.get("scene_match", True)
        structural_errors = relation_errors + attribute_errors

        modes: List[str] = []
        if missing >= cfg.missing_object_threshold:
            modes.append("missing_object")
        if additional >= cfg.additional_object_threshold:
            modes.append("additional_object")
        if structural_errors >= cfg.structural_error_threshold or scene_mismatch:
            modes.append("structural")

        # Severe overall mismatch wins regardless of the per-mode breakdown —
        # checked before mode-specific branches so a catastrophic frame is
        # never quietly "fixed" with a small prompt nudge.
        if is_interframe and severity >= cfg.keyframe_fallback_severity:
            return ControllerDecision(
                decision="keyframe_fallback",
                severity=severity,
                triggered_modes=modes,
                candidate_actions=[{
                    "type": "keyframe_fallback_candidate",
                    "detail": "severity exceeds keyframe_fallback_severity on an "
                              "inter-frame; fall back to the keyframe reconstruction",
                }],
                reason=f"severity {severity:.3f} >= keyframe_fallback_severity "
                       f"{cfg.keyframe_fallback_severity:.3f} (inter-frame)",
            )
        if severity >= cfg.fallback_severity:
            return ControllerDecision(
                decision="fallback_recompute",
                severity=severity,
                triggered_modes=modes,
                candidate_actions=[{
                    "type": "fallback_recompute_candidate",
                    "detail": "severity exceeds severity_threshold; recompute "
                              "from scratch rather than nudging one error type",
                }],
                reason=f"severity {severity:.3f} >= severity_threshold {cfg.fallback_severity:.3f}",
            )

        if not modes or severity <= cfg.accept_severity:
            return ControllerDecision(
                decision="accept",
                severity=severity,
                triggered_modes=modes,
                candidate_actions=[],
                reason=f"severity {severity:.3f} <= accept_severity {cfg.accept_severity:.3f}"
                       if severity <= cfg.accept_severity else "no error mode triggered",
            )

        # Dominant single-mode decision, ordered additional > missing > structural
        # (mirrors controllers/regeneration_policy.py's mode ordering).
        if "additional_object" in modes and additional >= missing and additional >= structural_errors:
            candidate_actions = [
                {"type": "negative_prompt_candidate", "object": obj}
                for obj in (report.get("additional_objects") or [])
            ]
            return ControllerDecision(
                decision="suppress_extra",
                severity=severity,
                triggered_modes=modes,
                candidate_actions=candidate_actions,
                reason=f"{additional} additional object(s) dominate the error report",
            )

        if "missing_object" in modes:
            candidate_actions = [
                {"type": "prompt_emphasis_candidate", "object": obj}
                for obj in (report.get("missing_objects") or [])
            ]
            return ControllerDecision(
                decision="strengthen_missing",
                severity=severity,
                triggered_modes=modes,
                candidate_actions=candidate_actions,
                reason=f"{missing} missing object(s) dominate the error report",
            )

        # structural (relation / attribute / scene)
        candidate_actions = [{
            "type": "structure_guidance_candidate",
            "detail": "raise ControlNet/edge structural guidance strength",
        }]
        return ControllerDecision(
            decision="strengthen_structure_guidance",
            severity=severity,
            triggered_modes=modes,
            candidate_actions=candidate_actions,
            reason="relation/attribute/scene errors dominate the error report",
        )
