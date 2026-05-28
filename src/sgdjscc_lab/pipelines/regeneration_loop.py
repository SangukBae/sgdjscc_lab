"""pipelines/regeneration_loop.py – Semantic-reliability-triggered regeneration.

Purpose
-------
Generative reconstruction (diffusion) is non-deterministic and occasionally
produces outputs that are semantically inconsistent with the original: objects
appear or disappear, spatial layouts are inverted, or colours shift dramatically.

This module provides a wrapper that:
1. Runs normal inference once.
2. Evaluates the Semantic Reliability Score (SRS).
3. If SRS < threshold, retries inference with adjusted diffusion parameters.
4. Returns the best result (highest SRS across all attempts).

Design constraints
------------------
- Does NOT modify the SGDJSCC VAE encode/decode or AWGN logic.
- Parameters adjusted between attempts: guidance_scale and diffusion_step.
- Max retries = 1 by default (2 total attempts) to avoid 2× compute cost.
- Can be disabled entirely by setting threshold = 0.0.

Phase 3 status: functional prototype.
Full production loop (multi-strategy, beam-like selection) is Phase 5+ work.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def regenerate_if_needed(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    metrics: Dict,
    threshold: float,
    pipeline_fn,
    cfg: DictConfig,
    models,
    max_retries: int = 1,
    retry_guidance_scale_multiplier: float = 1.5,
    retry_diffusion_step_delta: int = 10,
) -> Tuple[torch.Tensor, Dict]:
    """Return the best reconstruction; retry if SRS is below threshold.

    Parameters
    ----------
    original:
        Original image tensor ``[1, 3, H, W]`` in [0, 1] (for metric recomputation).
    reconstructed:
        Initial reconstruction ``[1, 3, H, W]`` in [0, 1].
    metrics:
        Dict from the first inference run (must contain ``semantic_reliability_score``).
    threshold:
        SRS below this value triggers a retry.  Set to 0.0 to disable.
    pipeline_fn:
        Callable ``(patches, models, cfg) → out_patches``.
        Typically ``pipelines.infer_pipeline.run_single_image``.
    cfg:
        OmegaConf config used for the first run (copied and modified per retry).
    models:
        ModelBundle instance.
    max_retries:
        Maximum number of additional attempts (default 1).
    retry_guidance_scale_multiplier:
        Factor to multiply ``guidance_scale`` on each retry.
    retry_diffusion_step_delta:
        Additional diffusion steps added per retry (avoids too-few steps issue).

    Returns
    -------
    (best_reconstructed, best_metrics)
        The reconstruction with the highest SRS found across all attempts.
    """
    srs = metrics.get("semantic_reliability_score")
    if srs is None or threshold <= 0.0 or srs >= threshold:
        # No retry needed
        return reconstructed, metrics

    logger.info(
        "SRS %.4f < threshold %.4f — attempting regeneration (%d retries max)",
        srs, threshold, max_retries,
    )

    best_recon   = reconstructed
    best_metrics = metrics
    best_srs     = float(srs)

    for attempt in range(1, max_retries + 1):
        # Build modified config for this retry
        retry_cfg = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
        retry_cfg.guidance_scale = float(cfg.guidance_scale) * (
            retry_guidance_scale_multiplier ** attempt
        )
        retry_cfg.diffusion_step = int(cfg.diffusion_step) + retry_diffusion_step_delta * attempt

        logger.info(
            "Retry %d: guidance_scale=%.2f  diffusion_step=%d",
            attempt, retry_cfg.guidance_scale, retry_cfg.diffusion_step,
        )

        try:
            from sgdjscc_lab.utils.preprocessing import prepare_patches, merge_patches
            from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator

            patches, meta = prepare_patches(original)
            patches = patches.to(models.device)
            out_patches = pipeline_fn(patches, models, retry_cfg)
            retry_recon = merge_patches(out_patches.cpu(), meta)

            # Recompute SRS for the new reconstruction
            srs_eval = SemanticReliabilityEvaluator()
            retry_metrics = srs_eval.evaluate(original, retry_recon)
            retry_srs = retry_metrics.get("semantic_reliability_score", -999)

            logger.info("Retry %d SRS: %.4f (prev best: %.4f)", attempt, retry_srs, best_srs)

            if retry_srs > best_srs:
                best_recon   = retry_recon
                best_metrics = retry_metrics
                best_srs     = retry_srs

        except Exception as exc:
            logger.warning("Retry %d failed: %s", attempt, exc)

    return best_recon, best_metrics


def build_regeneration_pipeline(cfg: DictConfig):
    """Return a partial that wraps run_single_image for use in regenerate_if_needed.

    Usage
    -----
    >>> pipeline_fn = build_regeneration_pipeline(cfg)
    >>> best_recon, best_metrics = regenerate_if_needed(
    ...     original, reconstructed, metrics, threshold=0.5,
    ...     pipeline_fn=pipeline_fn, cfg=cfg, models=models,
    ... )
    """
    from sgdjscc_lab.pipelines.infer_pipeline import _process_patches

    def _fn(patches, models_, cfg_):
        return _process_patches(patches, models_, cfg_)

    return _fn
