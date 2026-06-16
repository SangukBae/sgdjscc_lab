"""models/diffusion_wrapper_channel.py – Channel-conditioned diffusion wrapper (Phase 5-A).

Wraps (does not replace) the existing diffusion path so the noisy received signal
actually influences reconstruction.  Because the SGD-JSCC denoiser is frozen
(algorithm-preservation invariant), conditioning is injected at the **config /
adapter level** — the safest of the options suggested in the plan:

- the received/equalised latent already enters the forward pass as the diffusion
  init (``use_jscc_feature``), since the channel runs inside ``JSCCModel.channel``;
- the receiver **confidence** (from ``ReliabilityHead``) scales ``guidance_scale``,
  ``controlnet_scale`` and ``diffusion_step`` (low confidence → stronger prior,
  more steps), per the channel-condition policy;
- the blind mode forces blind SNR estimation (``use_gt_csi=False``);
- the encoded condition tokens are attached to the cfg as
  ``cfg.channel_condition_tokens`` — an **extra-context placeholder** that a
  future condition-aware (FiLM / cross-attention) denoiser can consume. The
  current frozen denoiser does not read them (documented limitation).

The Phase-4 ``cfg.prompt_override`` / ``cfg.staged_prompts`` contract is preserved
untouched: if a ``base_prompt`` is supplied it is passed straight through.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from sgdjscc_lab.channels.measurement import MeasurementBundle
from sgdjscc_lab.controllers.channel_condition_policy import (
    ChannelConditionPolicy, ChannelConditionDecision,
)


class ChannelConditionedDiffusion:
    """Build a channel-conditioned run config from a measurement bundle.

    Parameters
    ----------
    encoder:
        ``ChannelConditionEncoder`` (created lazily if None).
    reliability_head:
        ``ReliabilityHead`` (created lazily if None).
    policy:
        ``ChannelConditionPolicy`` (created lazily if None).
    """

    def __init__(self, encoder=None, reliability_head=None, policy=None) -> None:
        self._encoder = encoder
        self._rel = reliability_head
        self._policy = policy or ChannelConditionPolicy()

    def _get_encoder(self):
        if self._encoder is None:
            from sgdjscc_lab.models.channel_condition_encoder import ChannelConditionEncoder
            self._encoder = ChannelConditionEncoder()
        return self._encoder

    def _get_rel(self):
        if self._rel is None:
            from sgdjscc_lab.models.reliability_head import ReliabilityHead
            self._rel = ReliabilityHead()
        return self._rel

    def build_conditioned_cfg(
        self,
        base_cfg: DictConfig,
        bundle: MeasurementBundle,
        mode: str = "auto",
        csi: str = "perfect",
        base_prompt: Optional[str] = None,
    ) -> Tuple[DictConfig, Dict]:
        """Return ``(conditioned_cfg, info)``.

        ``conditioned_cfg`` is a deep copy of *base_cfg* with channel-driven
        guidance/step adjustments and the condition tokens attached.  ``info``
        carries the decision, confidence and a JSON-friendly condition summary.
        """
        condition = self._get_encoder().encode(bundle)
        rel = self._get_rel().predict(bundle, condition)
        confidence = float(rel["confidence"].mean().item())

        decision: ChannelConditionDecision = self._policy.decide(
            csi=csi, confidence=confidence, forced_mode=mode,
        )

        out = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        out.guidance_scale = round(float(base_cfg.get("guidance_scale", 4.0)) * decision.guidance_mult, 6)
        out.controlnet_scale = round(float(base_cfg.get("controlnet_scale", 0.3)) * decision.controlnet_mult, 6)
        out.diffusion_step = max(int(base_cfg.get("diffusion_step", 50)) + decision.step_delta, 1)
        # latent_conditioned / joint use the received latent as the init.
        out.use_jscc_feature = True
        if decision.blind_snr:
            out.use_gt_csi = False
        # Preserve the Phase-4 prompt contract.
        if base_prompt is not None:
            out.prompt_override = base_prompt
        # Extra-context placeholder for a future condition-aware denoiser.
        out.channel_condition_tokens = condition["tokens"].detach().cpu().tolist()

        info = {
            "mode": decision.mode,
            "confidence": confidence,
            "decision": decision.as_dict(),
            "measurement": bundle.summary(),
        }
        return out, info
