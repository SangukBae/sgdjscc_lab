"""controllers/channel_condition_policy.py – Channel-conditioning policy (Phase 5-A).

Decides *how* to condition the generative decoder on the channel observation,
based on the CSI regime and the receiver's confidence.  This is the bridge
between the reliability estimate and the three DiffCom-style conditioning modes.

Decision logic (overridable via the ``channel_condition`` config block):

================  ==============================  ===============================
CSI regime        Confidence                      Mode / behaviour
================  ==============================  ===============================
known             high                            latent_conditioned (light prior)
known             low                             joint_conditioned (strong prior)
unknown / blind   any                             blind_conditioned (conservative,
                                                  more steps, blind SNR estimate)
================  ==============================  ===============================

The explicit ``condition_mode`` config value (if set to anything other than
``"auto"``) overrides this table, so experiments can pin a single mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

VALID_MODES = ("latent_conditioned", "joint_conditioned", "blind_conditioned")


@dataclass
class ChannelConditionDecision:
    mode: str
    guidance_mult: float = 1.0
    controlnet_mult: float = 1.0
    step_delta: int = 0
    blind_snr: bool = False          # force blind SNR estimation (use_gt_csi off)
    reason: str = ""

    def as_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "guidance_mult": self.guidance_mult,
            "controlnet_mult": self.controlnet_mult,
            "step_delta": self.step_delta,
            "blind_snr": self.blind_snr,
            "reason": self.reason,
        }


class ChannelConditionPolicy:
    """Select a conditioning mode + decoder adjustments.

    Parameters
    ----------
    confidence_threshold:
        Confidence below which a known-CSI channel uses the stronger
        joint-conditioned mode.
    overrides:
        Optional dict overriding per-mode multipliers / step deltas.
    """

    def __init__(self, confidence_threshold: float = 0.5,
                 overrides: Optional[Dict] = None) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.overrides = overrides or {}

    def decide(
        self,
        csi: str = "perfect",
        confidence: float = 1.0,
        forced_mode: str = "auto",
    ) -> ChannelConditionDecision:
        """Return a :class:`ChannelConditionDecision`.

        Parameters
        ----------
        csi:
            ``"perfect"`` | ``"imperfect"`` | ``"none"`` | ``"unknown"``.
        confidence:
            Scalar receiver confidence in [0, 1] (from ``ReliabilityHead``).
        forced_mode:
            If not ``"auto"`` and a valid mode name, pins that mode.
        """
        blind = csi in ("none", "unknown", "blind", "imperfect")

        if forced_mode in VALID_MODES:
            mode = forced_mode
        elif blind:
            mode = "blind_conditioned"
        elif confidence >= self.confidence_threshold:
            mode = "latent_conditioned"
        else:
            mode = "joint_conditioned"

        # Per-mode defaults (low confidence / blind → lean harder on the prior).
        table = {
            "latent_conditioned": dict(guidance_mult=0.9, controlnet_mult=1.0,
                                       step_delta=0, blind_snr=False,
                                       reason="reliable channel; trust received latent"),
            "joint_conditioned": dict(guidance_mult=1.3, controlnet_mult=1.3,
                                      step_delta=5, blind_snr=False,
                                      reason="low confidence; strengthen prior + structure"),
            "blind_conditioned": dict(guidance_mult=1.2, controlnet_mult=1.1,
                                      step_delta=10, blind_snr=True,
                                      reason="unknown/imperfect CSI; conservative blind decode"),
        }[mode]
        table.update(self.overrides.get(mode, {}))
        return ChannelConditionDecision(mode=mode, **table)
