"""pipelines/channel_conditioned_infer.py – DiffCom-inspired channel-conditioned inference (Phase 5-A).

A separate inference entry that (1) transmits over a Phase 5-A channel, (2) builds
a :class:`MeasurementBundle` of receiver evidence, (3) encodes it into a channel
condition, and (4) reconstructs with a channel-conditioned run config — without
disturbing the Phase 4 image/video paths.

It is config-gated (``cfg.use_channel_conditioning``) and fully dependency-
injected (``reconstruct_fn`` / ``measure_fn``), so the orchestration is testable
without checkpoints, mirroring the temporal pipeline's design.

Supported condition modes (DiffCom naming): ``latent_conditioned`` /
``joint_conditioned`` / ``blind_conditioned`` (chosen by
``ChannelConditionPolicy`` or pinned via ``cfg.condition_mode``).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from sgdjscc_lab.channels.measurement import MeasurementBundle, aggregate_bundles
from sgdjscc_lab.models.diffusion_wrapper_channel import ChannelConditionedDiffusion

logger = logging.getLogger(__name__)

_SCALING_FACTOR = 15.45   # mirrors infer_pipeline / inference_one.py


class ChannelConditionedInference:
    """Orchestrate measure → encode → condition → reconstruct.

    Parameters
    ----------
    reconstruct_fn:
        ``(frame, cfg) -> reconstructed_tensor``.
    measure_fn:
        ``(frame, snr_db) -> MeasurementBundle``.
    wrapper:
        ``ChannelConditionedDiffusion`` (created lazily if None).
    base_cfg:
        Base run config (copied + augmented per frame).
    csi:
        CSI regime string passed to the policy.
    condition_mode:
        ``"auto"`` or a pinned mode name.
    """

    def __init__(
        self,
        reconstruct_fn: Callable,
        measure_fn: Callable,
        wrapper: Optional[ChannelConditionedDiffusion] = None,
        base_cfg: Optional[DictConfig] = None,
        csi: str = "perfect",
        condition_mode: str = "auto",
    ) -> None:
        self.reconstruct_fn = reconstruct_fn
        self.measure_fn = measure_fn
        self.wrapper = wrapper or ChannelConditionedDiffusion()
        self.base_cfg = base_cfg
        self.csi = csi
        self.condition_mode = condition_mode

    def run(
        self,
        frame: torch.Tensor,
        snr_db: Optional[float] = None,
        base_prompt: Optional[str] = None,
    ) -> Dict:
        """Run channel-conditioned reconstruction on a single frame.

        Returns ``{reconstruction, measurement, info, cfg}``.
        """
        bundle = self.measure_fn(frame, snr_db)
        cfg_out, info = self.wrapper.build_conditioned_cfg(
            self.base_cfg, bundle, mode=self.condition_mode,
            csi=self.csi, base_prompt=base_prompt,
        )
        recon = self.reconstruct_fn(frame, cfg_out)
        return {"reconstruction": recon, "measurement": bundle, "info": info, "cfg": cfg_out}


# ── One-pass builder backed by real models ────────────────────────────────────

class OnePassChannelConditionedInference:
    """Single-forward channel-conditioned inference over a full (patched) image.

    Unlike the generic :class:`ChannelConditionedInference` (which calls a separate
    ``measure_fn`` then ``reconstruct_fn``), this runs the encode+transmit stage
    once per patch, aggregates the receiver evidence to an image-level measurement,
    decides the conditioned cfg, then runs the diffusion decode reusing the SAME
    received latent — no throwaway measurement forward.
    """

    def __init__(self, models, base_cfg, wrapper, csi="perfect", condition_mode="auto"):
        self.models = models
        self.base_cfg = base_cfg
        self.wrapper = wrapper
        self.csi = csi
        self.condition_mode = condition_mode

    def run(self, frame: torch.Tensor, snr_db: Optional[float] = None,
            base_prompt: Optional[str] = None) -> Dict:
        from sgdjscc_lab.utils.preprocessing import prepare_patches, merge_patches
        from sgdjscc_lab.pipelines.infer_pipeline import run_image_channel_conditioned

        jscc = self.models.jscc_model
        jscc.snr = float(snr_db if snr_db is not None else self.base_cfg.get("snr_db", 10))

        base_cfg = self.base_cfg
        if base_prompt is not None:   # honour the Phase-4 prompt_override contract
            base_cfg = OmegaConf.create(OmegaConf.to_container(self.base_cfg, resolve=True))
            base_cfg.prompt_override = base_prompt

        patches, meta = prepare_patches(frame)
        out_patches, info = run_image_channel_conditioned(
            patches, self.models, base_cfg, self.wrapper,
            csi=self.csi, condition_mode=self.condition_mode,
        )
        recon = merge_patches(out_patches.cpu(), meta)
        # Return the *actually applied* conditioned cfg (not base_cfg) so logging
        # / debugging see the real guidance_scale / diffusion_step / use_gt_csi.
        resolved_cfg = info.get("resolved_cfg", base_cfg)
        return {"reconstruction": recon, "measurement": info.get("measurement"),
                "info": info, "cfg": resolved_cfg}


def build_channel_condition_wrapper(cfg: DictConfig) -> ChannelConditionedDiffusion:
    """Build a config-driven :class:`ChannelConditionedDiffusion` (no dead config)."""
    from sgdjscc_lab.models.channel_condition_encoder import ChannelConditionEncoder
    from sgdjscc_lab.models.reliability_head import ReliabilityHead
    from sgdjscc_lab.controllers.channel_condition_policy import ChannelConditionPolicy

    cc = OmegaConf.select(cfg, "channel_condition", default=None)
    cc = OmegaConf.to_container(cc, resolve=True) if cc is not None else {}
    encoder = ChannelConditionEncoder(
        token_grid=int(cc.get("token_grid", 4)),
        token_dim=int(cc.get("token_dim", 8)),
        mode=str(cc.get("encoder_mode", "stats")),
    )
    policy = ChannelConditionPolicy(
        confidence_threshold=float(cc.get("confidence_threshold", 0.5)),
        overrides=cc.get("policy_overrides"),
    )
    return ChannelConditionedDiffusion(
        encoder=encoder, reliability_head=ReliabilityHead(), policy=policy,
    )


def build_channel_conditioned_inference(models, cfg: DictConfig, channel=None):
    """Construct a one-pass channel-conditioned inference object from models + cfg.

    Installs the chosen channel on ``models.jscc_model.channel_model`` so the
    single forward transmits over it, and builds the condition encoder / policy /
    reliability head from the ``channel_condition`` config block.
    """
    from sgdjscc_lab.channels import build_channel

    channel = channel or build_channel(cfg)
    models.jscc_model.channel_model = channel
    wrapper = build_channel_condition_wrapper(cfg)
    return OnePassChannelConditionedInference(
        models=models, base_cfg=cfg, wrapper=wrapper,
        csi=str(cfg.get("csi", "perfect")),
        condition_mode=str(cfg.get("condition_mode", "auto")),
    )


def maybe_channel_conditioned_reconstruct(frame, models, cfg, snr_db=None, base_prompt=None):
    """Channel-conditioned reconstruct iff ``cfg.use_channel_conditioning`` is set.

    Returns ``(reconstruction, info)``; when disabled, returns
    ``(None, None)`` so callers can fall back to the standard path.
    """
    from sgdjscc_lab.phase_gates import phase5_enabled
    if not phase5_enabled(cfg) or not bool(cfg.get("use_channel_conditioning", False)):
        return None, None
    cci = build_channel_conditioned_inference(models, cfg)
    out = cci.run(frame, snr_db=snr_db, base_prompt=base_prompt)
    return out["reconstruction"], out["info"]
