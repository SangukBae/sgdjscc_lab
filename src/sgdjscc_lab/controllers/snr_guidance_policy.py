"""controllers/snr_guidance_policy.py – SNR → guidance-parameter policy (Phase 4-A).

Maps the channel SNR onto a set of diffusion-guidance parameters.  The intuition
(shared with FAST-GSC's adaptive transmission and the SGD-JSCC step-matching
idea) is that the *received* latent is reliable at high SNR and unreliable at low
SNR, so the strength of the generative prior should scale inversely with channel
quality:

- **low SNR**  : the latent is badly corrupted → lean hard on the semantic prior
                 (strong text + edge guidance, more denoising steps).
- **mid SNR**  : trust structure but keep moderate guidance.
- **high SNR** : the latent already carries most of the signal → weak guidance,
                 and optionally skip the diffusion prior entirely (it can *hurt*
                 by hallucinating detail the channel already delivered cleanly).

The three regimes are defined by two SNR thresholds and are fully overridable
from config (``adaptive_guidance`` block), so experiments can retune them without
touching code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class GuidanceDecision:
    """Resolved guidance parameters for one (image, SNR) inference."""

    guidance_scale: float
    controlnet_scale: float
    diffusion_step: int
    use_text: bool
    skip_diffusion: bool = False
    regime: str = "moderate"

    def as_dict(self) -> Dict:
        return {
            "guidance_scale": self.guidance_scale,
            "controlnet_scale": self.controlnet_scale,
            "diffusion_step": self.diffusion_step,
            "use_text": self.use_text,
            "skip_diffusion": self.skip_diffusion,
            "regime": self.regime,
        }


# Built-in default regime table.  Each regime is a partial parameter set; values
# left as None inherit from the base config in adaptive_guidance_controller.
_DEFAULT_POLICY: Dict[str, Dict] = {
    "thresholds": {"low": 0.0, "high": 8.0},   # SNR<=low → strong; SNR>=high → weak
    "strong": {  # SNR <= low (e.g. <= 0 dB)
        "guidance_scale_mult": 1.5,
        "controlnet_scale_mult": 1.5,
        "diffusion_step_delta": 10,
        "use_text": True,
        "skip_diffusion": False,
    },
    "moderate": {  # low < SNR < high
        "guidance_scale_mult": 1.0,
        "controlnet_scale_mult": 1.2,
        "diffusion_step_delta": 0,
        "use_text": True,
        "skip_diffusion": False,
    },
    "weak": {  # SNR >= high (e.g. >= 8 dB)
        "guidance_scale_mult": 0.5,
        "controlnet_scale_mult": 0.8,
        "diffusion_step_delta": -10,
        "use_text": False,
        "skip_diffusion": False,   # set True to skip the diffusion prior at high SNR
    },
}


def merge_policy(overrides: Optional[Dict]) -> Dict:
    """Deep-merge user *overrides* onto the default policy table."""
    policy = {
        "thresholds": dict(_DEFAULT_POLICY["thresholds"]),
        "strong": dict(_DEFAULT_POLICY["strong"]),
        "moderate": dict(_DEFAULT_POLICY["moderate"]),
        "weak": dict(_DEFAULT_POLICY["weak"]),
    }
    if not overrides:
        return policy
    for key, val in overrides.items():
        if key in policy and isinstance(val, dict):
            policy[key].update(val)
        else:
            policy[key] = val
    return policy


def classify_regime(snr_db: float, policy: Optional[Dict] = None) -> str:
    """Return ``"strong"`` | ``"moderate"`` | ``"weak"`` for *snr_db*."""
    policy = policy or _DEFAULT_POLICY
    th = policy.get("thresholds", _DEFAULT_POLICY["thresholds"])
    low = float(th.get("low", 0.0))
    high = float(th.get("high", 8.0))
    if snr_db <= low:
        return "strong"
    if snr_db >= high:
        return "weak"
    return "moderate"


def decide(
    snr_db: float,
    base_guidance_scale: float,
    base_controlnet_scale: float,
    base_diffusion_step: int,
    base_use_text: bool,
    overrides: Optional[Dict] = None,
) -> GuidanceDecision:
    """Compute a :class:`GuidanceDecision` from base params and the SNR regime.

    The regime supplies multiplicative / additive adjustments relative to the
    base (config) values, so the policy degrades gracefully if a user tunes the
    base config.  ``diffusion_step`` is clamped to a sane minimum of 1.
    """
    policy = merge_policy(overrides)
    regime = classify_regime(snr_db, policy)
    p = policy[regime]

    g = float(base_guidance_scale) * float(p.get("guidance_scale_mult", 1.0))
    c = float(base_controlnet_scale) * float(p.get("controlnet_scale_mult", 1.0))
    step = int(base_diffusion_step) + int(p.get("diffusion_step_delta", 0))
    step = max(step, 1)
    use_text = bool(p.get("use_text", base_use_text)) and bool(base_use_text)
    skip = bool(p.get("skip_diffusion", False))

    return GuidanceDecision(
        guidance_scale=round(g, 6),
        controlnet_scale=round(c, 6),
        diffusion_step=step,
        use_text=use_text,
        skip_diffusion=skip,
        regime=regime,
    )
