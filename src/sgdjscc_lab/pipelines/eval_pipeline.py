"""pipelines/eval_pipeline.py – SNR-sweep evaluation pipeline (Phase 3).

Orchestrates inference and metric computation over a dataset, writing results to
CSV incrementally.

Entry points
------------
evaluate_dataset(cfg, models, eval_ctx, snr_db, ...)
    Run inference + metrics on all images in cfg.input_path at one SNR value.
    Returns a list of per-image result dicts.

evaluate_single_snr(cfg, models, eval_ctx, snr_db, csv_logger=None)
    Call evaluate_dataset and optionally stream rows to a CSV logger.
    Returns aggregate summary dict.

evaluate_snr_sweep(cfg, models, eval_ctx, snr_list, csv_logger=None)
    Loop over SNR values, calling evaluate_single_snr for each.
    Returns dict keyed by SNR value with per-SNR summaries.

The ``reconstruct_fn`` optional parameter allows callers (and tests) to inject
a custom inference function instead of the default SGDJSCC pipeline.

Algorithm preservation
----------------------
This pipeline does NOT modify the forward-pass logic of infer_pipeline.py.
It wraps it: load → infer → measure → log.

CSV columns (from utils.csv_logger.RESULT_COLUMNS)
--------------------------------------------------
filename, snr_db, psnr, ssim, lpips, clip_image_image, clip_text_image,
object_preservation_rate, missing_object_rate, additional_object_rate,
hallucination_score, semantic_reliability_score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EvalContext – holds all evaluator instances
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalContext:
    """Container for Phase 3 evaluator instances.

    Create once per experiment; all evaluators are lazy-loaded and cache
    their underlying models across images.

    Parameters
    ----------
    quality_evaluator:
        ``QualityEvaluator`` instance.  Created lazily if None.
    clip_evaluator:
        ``CLIPScoreEvaluator`` instance.  Shared across clip / obj_pres / hall.
    srs_evaluator:
        ``SemanticReliabilityEvaluator`` instance.  Created lazily if None.
    enabled_metrics:
        Set of metric names to compute.  All metrics enabled by default.
    clip_model_name:
        CLIP model variant (e.g. ``"ViT-B/32"``).  Passed to CLIPScoreEvaluator.
    srs_weights:
        Dict with keys ``w_img``, ``w_txt``, ``w_pres``, ``w_miss``, ``w_add``.
        Passed to SemanticReliabilityEvaluator; None uses the default weights.
    text_list:
        Optional list of N captions for text-image CLIP scoring.
        Set at call time via evaluate_dataset; ignored here.
    """
    quality_evaluator: Optional[object] = None
    clip_evaluator:    Optional[object] = None
    srs_evaluator:     Optional[object] = None
    enabled_metrics:   Optional[set]    = None
    clip_model_name:   str              = "ViT-B/32"
    srs_weights:       Optional[dict]   = None

    def __post_init__(self):
        if self.enabled_metrics is None:
            self.enabled_metrics = {
                "psnr", "ssim", "lpips",
                "clip_image_image", "clip_text_image",
                "object_preservation_rate", "hallucination_score",
                "semantic_reliability_score",
            }

    def _get_quality(self):
        if self.quality_evaluator is None:
            from sgdjscc_lab.evaluators.quality import QualityEvaluator
            use_lpips = "lpips" in self.enabled_metrics
            self.quality_evaluator = QualityEvaluator(use_lpips=use_lpips)
        return self.quality_evaluator

    def _get_clip(self):
        if self.clip_evaluator is None:
            from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
            self.clip_evaluator = CLIPScoreEvaluator(model_name=self.clip_model_name)
        return self.clip_evaluator

    def _get_srs(self):
        if self.srs_evaluator is None:
            from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
            self.srs_evaluator = SemanticReliabilityEvaluator(
                clip_evaluator=self._get_clip(),
                weights=self.srs_weights,
            )
        return self.srs_evaluator


# ─────────────────────────────────────────────────────────────────────────────
# Default reconstruction function
# ─────────────────────────────────────────────────────────────────────────────

def _default_reconstruct(
    fpath: Path,
    models,
    cfg: DictConfig,
) -> tuple:
    """Load, patch, infer, and merge a single image.

    Returns (original [1,3,H,W], reconstructed [1,3,H,W]) both on CPU.
    """
    from sgdjscc_lab.io import load_image_as_tensor
    from sgdjscc_lab.utils.preprocessing import prepare_patches, merge_patches
    from sgdjscc_lab.pipelines.infer_pipeline import _process_patches

    raw = load_image_as_tensor(fpath)          # [1, 3, H, W]
    patches, meta = prepare_patches(raw)        # [N, 3, 128, 128]
    patches = patches.to(models.device)

    out_patches = _process_patches(patches, models, cfg)
    reconstructed = merge_patches(out_patches.cpu(), meta)   # [1, 3, H, W]
    return raw, reconstructed


# ─────────────────────────────────────────────────────────────────────────────
# Per-image metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    eval_ctx: EvalContext,
    filename: str,
    snr_db: float,
    text_list: Optional[List[str]] = None,
) -> Dict:
    """Compute all enabled metrics for one image pair.

    Returns a flat dict matching RESULT_COLUMNS.
    """
    row: Dict = {
        "filename": filename,
        "snr_db":   snr_db,
        "psnr":     None,
        "ssim":     None,
        "lpips":    None,
        "clip_image_image":         None,
        "clip_text_image":          None,
        "object_preservation_rate": None,
        "missing_object_rate":      None,
        "additional_object_rate":   None,
        "hallucination_score":      None,
        "semantic_reliability_score": None,
    }

    em = eval_ctx.enabled_metrics

    # ── Quality metrics (fast, CPU) ──────────────────────────────────────────
    if em & {"psnr", "ssim", "lpips"}:
        try:
            q = eval_ctx._get_quality().evaluate(original, reconstructed)
            row["psnr"]  = q.get("psnr")
            row["ssim"]  = q.get("ssim")
            row["lpips"] = q.get("lpips")
        except Exception as exc:
            logger.warning("Quality metric error for %s: %s", filename, exc)

    # ── CLIP / SRS metrics (needs model) ────────────────────────────────────
    clip_needed = em & {
        "clip_image_image", "clip_text_image",
        "object_preservation_rate", "hallucination_score",
        "semantic_reliability_score",
    }
    if clip_needed:
        try:
            srs_result = eval_ctx._get_srs().evaluate(
                original, reconstructed, text_list=text_list
            )
            row["clip_image_image"]           = srs_result.get("clip_image_image")
            row["clip_text_image"]            = srs_result.get("clip_text_image")
            row["object_preservation_rate"]   = srs_result.get("object_preservation_rate")
            row["missing_object_rate"]        = srs_result.get("missing_object_rate")
            row["additional_object_rate"]     = srs_result.get("additional_object_rate")
            row["hallucination_score"]        = srs_result.get("hallucination_score")
            row["semantic_reliability_score"] = srs_result.get("semantic_reliability_score")
        except Exception as exc:
            logger.warning("CLIP/SRS metric error for %s: %s", filename, exc)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline functions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    cfg: DictConfig,
    models,
    eval_ctx: EvalContext,
    snr_db: float,
    text_list: Optional[List[str]] = None,
    csv_logger=None,
    reconstruct_fn: Optional[Callable] = None,
) -> List[Dict]:
    """Run inference + metrics on all images in cfg.input_path at one SNR.

    Parameters
    ----------
    cfg:
        Loaded OmegaConf config.  ``cfg.input_path`` is the image source.
    models:
        ModelBundle from runtime.build_models().
    eval_ctx:
        EvalContext with evaluator instances.
    snr_db:
        Channel SNR for this run (sets models.jscc_model.snr).
    text_list:
        Optional per-image captions (same length as image list).
    csv_logger:
        Optional CSVLogger; if provided, each row is written immediately.
    reconstruct_fn:
        Optional callable ``(fpath, models, cfg) -> (original, reconstructed)``.
        Defaults to ``_default_reconstruct``.

    Returns
    -------
    List of per-image result dicts.
    """
    from sgdjscc_lab.io import list_image_files

    if reconstruct_fn is None:
        reconstruct_fn = _default_reconstruct

    # Update SNR on the model (no rebuild required)
    if models is not None and hasattr(models, "jscc_model"):
        models.jscc_model.snr = float(snr_db)

    files = list_image_files(cfg.input_path)
    logger.info("evaluate_dataset: %d images, SNR=%.1f dB", len(files), snr_db)

    # Regeneration loop settings (read from cfg; off by default)
    use_regen       = bool(cfg.get("use_regeneration_loop", False))
    regen_threshold = float(cfg.get("regeneration_threshold", 0.5))
    regen_max_retries = int(cfg.get("regeneration_max_retries", 1))

    rows: List[Dict] = []
    for idx, fpath in enumerate(files, start=1):
        logger.info("[%d/%d] %s", idx, len(files), fpath.name)
        txt = [text_list[idx - 1]] if (text_list and idx <= len(text_list)) else None

        try:
            original, reconstructed = reconstruct_fn(fpath, models, cfg)
            row = _compute_metrics(
                original, reconstructed, eval_ctx,
                filename=fpath.name, snr_db=snr_db, text_list=txt,
            )

            # Regeneration: retry if SRS is below threshold
            if use_regen and row.get("semantic_reliability_score") is not None:
                from sgdjscc_lab.pipelines.regeneration_loop import (
                    regenerate_if_needed, build_regeneration_pipeline,
                )
                pipeline_fn = build_regeneration_pipeline(cfg)
                metrics_for_regen = {"semantic_reliability_score": row["semantic_reliability_score"]}
                new_recon, _ = regenerate_if_needed(
                    original, reconstructed, metrics_for_regen,
                    threshold=regen_threshold,
                    pipeline_fn=pipeline_fn,
                    cfg=cfg,
                    models=models,
                    max_retries=regen_max_retries,
                )
                if new_recon is not reconstructed:
                    logger.info("Regeneration improved SRS for %s; recomputing metrics.", fpath.name)
                    row = _compute_metrics(
                        original, new_recon, eval_ctx,
                        filename=fpath.name, snr_db=snr_db, text_list=txt,
                    )

        except Exception as exc:
            logger.error("Failed on %s: %s", fpath.name, exc, exc_info=True)
            row = {"filename": fpath.name, "snr_db": snr_db}

        rows.append(row)
        if csv_logger is not None:
            csv_logger.write_row(row)

    return rows


def evaluate_single_snr(
    cfg: DictConfig,
    models,
    eval_ctx: EvalContext,
    snr_db: float,
    text_list: Optional[List[str]] = None,
    csv_logger=None,
    reconstruct_fn: Optional[Callable] = None,
) -> Dict:
    """Run evaluate_dataset and return an aggregate summary.

    Returns
    -------
    dict with per-metric mean/std (from utils.metrics_io.summarize_metrics)
    plus ``rows`` (the full per-image list) and ``snr_db``.
    """
    from sgdjscc_lab.utils.metrics_io import summarize_metrics

    rows = evaluate_dataset(
        cfg, models, eval_ctx, snr_db,
        text_list=text_list, csv_logger=csv_logger,
        reconstruct_fn=reconstruct_fn,
    )
    summary = summarize_metrics(rows)
    summary["snr_db"] = snr_db
    summary["rows"]   = rows
    return summary


def evaluate_snr_sweep(
    cfg: DictConfig,
    models,
    eval_ctx: EvalContext,
    snr_list: List[float],
    text_list: Optional[List[str]] = None,
    csv_logger=None,
    reconstruct_fn: Optional[Callable] = None,
) -> Dict[float, Dict]:
    """Evaluate at multiple SNR values.

    Parameters
    ----------
    snr_list:
        Ordered list of SNR values in dB.  E.g. ``[-5, 0, 5, 10, 15, 20, 25]``.

    Returns
    -------
    dict keyed by SNR value, each containing the output of evaluate_single_snr.
    """
    results: Dict[float, Dict] = {}
    for snr_db in snr_list:
        logger.info("=== SNR = %.1f dB ===", snr_db)
        results[snr_db] = evaluate_single_snr(
            cfg, models, eval_ctx, snr_db,
            text_list=text_list, csv_logger=csv_logger,
            reconstruct_fn=reconstruct_fn,
        )
    return results
