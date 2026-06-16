"""controllers/adaptive_guidance_controller.py – SNR-aware guidance controller (Phase 4-A).

Sits between the run config and the SGD-JSCC forward pass.  Given the current
channel SNR it consults :mod:`snr_guidance_policy` and returns a *modified copy*
of the config whose guidance parameters (``guidance_scale``, ``controlnet_scale``,
``diffusion_step``, ``use_text``) and, optionally, the diffusion-skip path
(``use_semantic``) are set for the SNR regime.

It never mutates the input config and never touches the SGD-JSCC numerics — it
only chooses *which* config the unchanged forward pass runs with, so the
algorithm-preservation invariant is respected.

Activation
----------
Controlled entirely by config so the legacy image path is unaffected by default::

    use_adaptive_guidance: false        # master switch (off by default)
    adaptive_guidance:                  # optional policy overrides
      thresholds: {low: 0.0, high: 8.0}
      weak: {skip_diffusion: true}      # e.g. skip the prior at high SNR
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from sgdjscc_lab.controllers.snr_guidance_policy import GuidanceDecision, decide

logger = logging.getLogger(__name__)


class AdaptiveGuidanceController:
    """Resolve and apply SNR-dependent guidance parameters to a run config.

    Parameters
    ----------
    overrides:
        Optional policy-table overrides (the ``adaptive_guidance`` config block).
    """

    def __init__(self, overrides: Optional[Dict] = None) -> None:
        self.overrides = overrides or {}

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "AdaptiveGuidanceController":
        """Build a controller from a run config's ``adaptive_guidance`` block."""
        raw = OmegaConf.select(cfg, "adaptive_guidance", default=None)
        overrides = OmegaConf.to_container(raw, resolve=True) if raw is not None else None
        return cls(overrides=overrides)

    def decide(self, cfg: DictConfig, snr_db: Optional[float] = None) -> GuidanceDecision:
        """Return the :class:`GuidanceDecision` for *cfg* at *snr_db*.

        ``snr_db`` defaults to ``cfg.snr_db`` (the configured channel SNR).
        """
        snr = float(snr_db if snr_db is not None else cfg.get("snr_db", 0.0))
        return decide(
            snr_db=snr,
            base_guidance_scale=float(cfg.get("guidance_scale", 4.0)),
            base_controlnet_scale=float(cfg.get("controlnet_scale", 0.3)),
            base_diffusion_step=int(cfg.get("diffusion_step", 50)),
            base_use_text=bool(cfg.get("use_text", True)),
            overrides=self.overrides,
        )

    def apply(
        self, cfg: DictConfig, snr_db: Optional[float] = None
    ) -> Tuple[DictConfig, GuidanceDecision]:
        """Return ``(adapted_cfg, decision)``.

        ``adapted_cfg`` is a deep copy of *cfg* with guidance fields overwritten
        per the SNR regime.  When ``skip_diffusion`` is selected, ``use_semantic``
        is turned off so the forward pass falls back to the plain VAE-decode path
        (an "unconditional"/skip reconstruction).
        """
        decision = self.decide(cfg, snr_db)
        adapted = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        adapted.guidance_scale = decision.guidance_scale
        adapted.controlnet_scale = decision.controlnet_scale
        adapted.diffusion_step = decision.diffusion_step
        adapted.use_text = decision.use_text
        if decision.skip_diffusion:
            adapted.use_semantic = False
        logger.info(
            "Adaptive guidance [%s regime @ %.1f dB]: gs=%.2f cn=%.2f steps=%d "
            "use_text=%s skip=%s",
            decision.regime, float(snr_db if snr_db is not None else cfg.get("snr_db", 0.0)),
            decision.guidance_scale, decision.controlnet_scale, decision.diffusion_step,
            decision.use_text, decision.skip_diffusion,
        )
        return adapted, decision


def maybe_apply_adaptive_guidance(
    cfg: DictConfig, snr_db: Optional[float] = None
) -> Tuple[DictConfig, Optional[GuidanceDecision]]:
    """Apply adaptive guidance iff ``cfg.use_adaptive_guidance`` is true.

    Returns ``(cfg_or_adapted, decision_or_None)``.  When the master switch is
    off, the original config is returned unchanged and the decision is ``None`` —
    preserving the legacy inference path exactly.
    """
    from sgdjscc_lab.phase_gates import phase4_enabled
    if not phase4_enabled(cfg) or not bool(cfg.get("use_adaptive_guidance", False)):
        return cfg, None
    controller = AdaptiveGuidanceController.from_config(cfg)
    return controller.apply(cfg, snr_db)
