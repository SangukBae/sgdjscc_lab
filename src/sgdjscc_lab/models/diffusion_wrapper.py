"""diffusion_wrapper.py – Diffusion backbone + ControlNet loader.

Extracted from the semantic pipeline construction block in runtime.build_models()
(originally inference_one.py lines 293–310).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch

from sgdjscc_lab._sgdjscc import ensure_sgdjscc_on_path
from sgdjscc_lab.models.jscc_model import DDCONFIG

logger = logging.getLogger(__name__)


def build_diffusion_pipeline(cfg, device: torch.device, jscc_model):
    """Load MDTv2 (+ optional ControlNet), CLIP, and shared VAE.

    Returns a DiffusionGenerator configured identically to inference_one.py.

    Parameters
    ----------
    cfg:
        Loaded OmegaConf config.  Uses cfg.model_root, cfg.use_controlnet.
    device:
        Compute device.
    jscc_model:
        An already-loaded JSCCModel whose VAE state_dict is shared with the
        diffusion pipeline (mirrors inference_one.py lines 303–307).
    """
    ensure_sgdjscc_on_path()
    from models.test_advanced_network.autoencoderkl import AutoencoderKL
    from models.test_advanced_network.mask_diffusion import MDTv2
    from models.test_advanced_network.mask_diffusion_controlnet import MDTv2_ControlNet
    from models.test_advanced_network.diffusion_element_wise import DiffusionGenerator
    import clip

    model_root = Path(cfg.model_root)

    # ── Base diffusion backbone ───────────────────────────────────────────────
    denoiser = MDTv2(depth=12, hidden_size=512, patch_size=1, num_heads=8)
    backbone_ckpt = torch.load(model_root / "diffusion_backbone.pth", map_location=device)
    denoiser.load_state_dict(backbone_ckpt["model_ema"])
    logger.info("Loaded diffusion_backbone.pth")

    # ── Optional ControlNet ───────────────────────────────────────────────────
    if cfg.use_controlnet:
        denoiser = MDTv2_ControlNet(
            base_model=denoiser, copy_blocks_num=6, hidden_size=512
        )
        ctrl_ckpt = torch.load(
            model_root / "diffusion_controlnet.pth", map_location=device
        )
        denoiser.load_state_dict(ctrl_ckpt["model_ema"])
        logger.info("Loaded diffusion_controlnet.pth")

    denoiser.to(device)

    # ── CLIP text encoder (ViT-L/14 as in inference_one.py line 299) ─────────
    clip_model, _ = clip.load("ViT-L/14", device=str(device))
    clip_model.to(device)

    # ── Shared VAE (inference_one.py lines 303–307) ───────────────────────────
    vae_shared = AutoencoderKL(DDCONFIG, 16)
    vae_shared.load_state_dict(jscc_model.vae.state_dict(), strict=False)
    vae_shared.to(device)

    logger.info("Semantic pipeline ready (use_controlnet=%s)", cfg.use_controlnet)
    return DiffusionGenerator(denoiser, vae_shared, clip_model, device, torch.float32)
