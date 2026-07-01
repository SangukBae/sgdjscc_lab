"""diffusion_wrapper.py – Diffusion backbone + ControlNet loader.

Extracted from the semantic pipeline construction block in runtime.build_models()
(originally inference_one.py lines 293–310).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Union

import torch

from sgdjscc_lab._sgdjscc import ensure_sgdjscc_on_path
from sgdjscc_lab.models.jscc_model import DDCONFIG

logger = logging.getLogger(__name__)

# Methods the interruptible wrapper loop needs from a DiffusionGenerator. The
# SGD-JSCC ``DiffusionGenerator.generate`` (SGDJSCC/models/test_advanced_network/
# diffusion_element_wise.py, lines ~120-160) runs a monolithic DPM-Solver++(2M)
# loop with **no per-step callback hook**, and SGDJSCC/ is read-only — so to stop
# mid-loop we re-drive the same per-step update here using the generator's public
# helpers.  When any helper is missing (or step_style != continuous), we fall back
# to the original ``generate`` (no interrupt).
_INTERRUPTIBLE_REQUIRES = (
    "encode_text", "pred_image", "adjust_cfg_weight", "expand_scalar",
    "sigmoid_schedule", "sigmoid_schedule_inverse", "text_embed", "model", "device",
)


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
    # hidden_size=512 is CONFIRMED from the public SGDJSCC code
    # (SGDJSCC/inference_config.py: MDTv2(depth=12, hidden_size=512, patch_size=1,
    # num_heads=8)). Kept fixed for checkpoint compatibility — DO NOT change.
    # NOTE: the paper table's "embedding size = 256" is NOT evidence for a 256-d
    # backbone. The public code uses frequency_embedding_size=256 for the
    # timestep/noise embedder (mask_diffusion.py: TimestepEmbedder), a SEPARATE
    # quantity from the transformer hidden_size. See docs/paper_training_alignment.md.
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


# ─────────────────────────────────────────────────────────────────────────────
# Intra-sampler early-exit (Phase 5-B): re-driven DPM-Solver++(2M) loop
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_convergence_score_fn():
    """Default score_fn: increasing convergence score = 1 − relative latent update.

    A small update between steps → score near 1 (converged); used by the
    EarlyExitController's diminishing-returns / threshold logic.
    """
    prev = {"x": None}

    def _score(state, i, total):
        x = state["x_t"]
        if prev["x"] is None:
            prev["x"] = x
            return 0.0
        denom = float(torch.linalg.norm(prev["x"].reshape(-1)) + 1e-8)
        rel = float(torch.linalg.norm((x - prev["x"]).reshape(-1)) / denom)
        prev["x"] = x
        return max(0.0, 1.0 - rel)

    return _score


def generate_interruptible(
    pipe,
    *,
    prompt,
    negative_prompt,
    latent,
    curr_step,
    diffusion_step,
    c=None,
    controlnet=False,
    not_control=None,
    class_guidance=4.0,
    cfg_weighting_method="constant",
    mask_token=None,
    mask_step=1,
    step_style="continuous",
    controller=None,
    score_fn=None,
    check_interval=5,
    min_steps=1,
    **fallback_kwargs,
):
    """Run the SGD-JSCC continuous DPM-Solver++(2M) loop with mid-loop early-exit.

    Re-drives the per-step update from ``pipe`` helpers (the original ``generate``
    exposes no callback) so the loop can terminate as soon as the
    ``EarlyExitController`` is satisfied — saving the remaining denoising steps.

    Falls back to ``pipe.generate(..., return_latent=True)`` (no interrupt) when
    ``step_style != "continuous"`` or the pipe lacks the required helpers.

    Returns ``(denoised_latent, info)`` where ``info`` reports
    ``interrupted`` / ``stopped_at`` / ``total_steps`` / ``reason`` / ``fallback``.
    """
    import numpy as np
    from sgdjscc_lab.acceleration.early_exit import run_interruptible_sampling

    supported = step_style == "continuous" and all(hasattr(pipe, m) for m in _INTERRUPTIBLE_REQUIRES)
    if not supported:
        logger.warning(
            "Interruptible sampler unsupported (step_style=%s); using pipe.generate "
            "(no intra-sampler interrupt).", step_style)
        _img, lat = pipe.generate(
            prompt=prompt, negative_prompt=negative_prompt, latent=latent,
            curr_step=curr_step, diffusion_step=diffusion_step, step_style=step_style,
            class_guidance=class_guidance, cfg_weighting_method=cfg_weighting_method,
            c=c, controlnet=controlnet, not_control=not_control, mask_token=mask_token,
            mask_step=mask_step, return_latent=True, **fallback_kwargs)
        return lat, {"interrupted": False, "fallback": True,
                     "stopped_at": diffusion_step, "total_steps": diffusion_step}

    # ── Schedule (mirrors generate() continuous branch) ──────────────────────
    labels = pipe.encode_text(prompt, pipe.text_embed)
    neg = (pipe.encode_text(negative_prompt, pipe.text_embed)
           if negative_prompt is not None else torch.zeros_like(labels))
    curr_timestep = pipe.sigmoid_schedule_inverse(curr_step.cpu().numpy())
    timesteps = np.linspace(0.001, curr_timestep, diffusion_step)[:, :, 0].transpose(1, 0)
    noise_levels = np.sqrt(pipe.sigmoid_schedule(timesteps))
    noise_levels = np.flip(noise_levels, axis=1)
    timesteps = np.flip(timesteps, axis=1)

    lambdas = np.log((1 - noise_levels) / noise_levels)
    hs = np.stack([lambdas[:, i] - lambdas[:, i - 1] for i in range(lambdas.shape[1])], axis=1)
    rs = np.stack([hs[:, i - 1] / hs[:, i] for i in range(hs.shape[1])], axis=1)

    labels = torch.cat([labels, neg])
    pipe.model.eval()
    total = noise_levels.shape[1] - 1

    if not_control is None:
        n = latent.size(0)
        not_control = [1] * n + [1] * n
    if score_fn is None and controller is not None:
        score_fn = _heuristic_convergence_score_fn()

    state = {"x_t": latent, "x0_prev": None, "mask_step": mask_step}

    def step_fn(state, i, total):
        x_t = state["x_t"]
        x0_prev = state["x0_prev"]
        curr_noise, next_noise = noise_levels[:, i], noise_levels[:, i + 1]
        curr_ts = timesteps[:, i]
        cg = pipe.adjust_cfg_weight(class_guidance, curr_ts, cfg_weighting_method)
        x0_pred = pipe.pred_image(
            x_t, labels, curr_noise, cg, c, controlnet,
            mask_token if state["mask_step"] > 0 else None, not_control=not_control)
        state["x0"] = x0_pred   # predicted clean latent (for verified score_fns)
        state["mask_step"] = state["mask_step"] - 1 if state["mask_step"] > 0 else 0
        x_pred_scale = np.sqrt(1 - next_noise ** 2) - next_noise / curr_noise * np.sqrt(1 - curr_noise ** 2)
        x_t_scale = next_noise / curr_noise
        x_pred_scale = pipe.expand_scalar(torch.from_numpy(x_pred_scale).to(pipe.device), x_t.shape)
        x_t_scale = pipe.expand_scalar(torch.from_numpy(x_t_scale).to(pipe.device), x_t.shape)
        if x0_prev is None:
            x_t = x_pred_scale * x0_pred + x_t_scale * x_t
        else:
            rs_i_1 = pipe.expand_scalar(torch.from_numpy(rs[:, i - 1])[:, None].to(pipe.device), x_t.shape)
            D = (1 + 1 / (2 * rs_i_1)) * x0_pred - (1 / (2 * rs_i_1)) * x0_prev
            x_t = x_pred_scale * D + x_t_scale * x_t
        state["x0_prev"] = x0_pred
        state["x_t"] = x_t
        return state

    result = run_interruptible_sampling(
        state, step_fn, total, score_fn=score_fn, controller=controller,
        check_interval=check_interval, min_steps=min_steps,
    )

    # Final x0 prediction (mirrors generate() line ~151) at the reached noise level.
    final_state = result["state"]
    last_idx = min(result["stopped_at"], total)
    final_next = noise_levels[:, last_idx]
    ts_idx = max(last_idx - 1, 0)
    cg = pipe.adjust_cfg_weight(class_guidance, timesteps[:, ts_idx], cfg_weighting_method)
    x0_pred = pipe.pred_image(final_state["x_t"], labels, final_next, cg, c, controlnet,
                              not_control=not_control)
    info = {
        "interrupted": not result["completed"],
        "stopped_at": result["stopped_at"],
        "total_steps": total,
        "reason": result["reason"],
        "history": result["history"],
        "fallback": False,
    }
    return x0_pred, info
