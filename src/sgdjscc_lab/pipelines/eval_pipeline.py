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
    # Object presence judge settings (provisional CLIP probe — see
    # object_preservation.py). Wired from cfg keys object_presence_threshold /
    # object_presence_uncertain_band so threshold changes actually reach the
    # ObjectPreservation / Hallucination evaluators (ETRI plan step 0).
    presence_threshold: float           = 0.25
    presence_uncertain_band: float      = 0.0
    # Compute device for the eval-side models (CLIP / packet BLIP2 / VQA). When
    # None they default to CPU, which breaks fp16 (Half) caption/VQA models —
    # set this to the model device (e.g. cuda:0) to keep everything consistent.
    device:            Optional[object] = None
    # ── Phase 4-A packet-aware settings ──────────────────────────────────────
    packet_weights:    Optional[dict]   = None
    packet_blend:      float            = 0.5
    packet_extractor:  Optional[object] = None
    # ── Phase 5-C SRS-v2 settings ────────────────────────────────────────────
    use_srs_v2:        bool             = False
    srs_v2_weights:    Optional[dict]   = None
    use_vqa_hallucination: bool         = False
    vqa_fn:            Optional[object] = None   # injected VQA backend (else CLIP)
    vqa_backend_cfg:   Optional[dict]   = None   # config to lazily build a backend
    srs_v2_evaluator:  Optional[object] = None
    # ── FID (paper §VI, dataset-level) ───────────────────────────────────────
    fid_evaluator:     Optional[object] = None
    fid_feature_fn:    Optional[object] = None   # inject a cheap extractor (tests)

    def __post_init__(self):
        if self.enabled_metrics is None:
            self.enabled_metrics = {
                "psnr", "ssim", "lpips",
                "clip_image_image", "clip_text_image",
                "object_preservation_rate", "hallucination_score",
                "semantic_reliability_score",
            }

    def _get_fid(self):
        """Lazily build the dataset-level FID evaluator (paper §VI).

        A fresh evaluator is returned each call so per-SNR runs do not mix
        feature statistics across SNR conditions.
        """
        from sgdjscc_lab.evaluators.fid import FIDEvaluator
        return FIDEvaluator(feature_fn=self.fid_feature_fn, device=self.device)

    def _get_quality(self):
        if self.quality_evaluator is None:
            from sgdjscc_lab.evaluators.quality import QualityEvaluator
            use_lpips = "lpips" in self.enabled_metrics
            self.quality_evaluator = QualityEvaluator(use_lpips=use_lpips)
        return self.quality_evaluator

    def _get_clip(self):
        if self.clip_evaluator is None:
            from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
            self.clip_evaluator = CLIPScoreEvaluator(
                model_name=self.clip_model_name, device=self.device,
            )
        return self.clip_evaluator

    def _get_srs(self):
        if self.srs_evaluator is None:
            from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
            self.srs_evaluator = SemanticReliabilityEvaluator(
                clip_evaluator=self._get_clip(),
                weights=self.srs_weights,
                packet_weights=self.packet_weights,
                packet_blend=self.packet_blend,
                presence_threshold=self.presence_threshold,
                presence_uncertain_band=self.presence_uncertain_band,
            )
        return self.srs_evaluator

    def _get_packet_extractor(self, models=None, cfg=None):
        """Lazily build a SemanticPacketExtractor for packet-aware evaluation.

        Reuses the shared CLIP evaluator and, when available, the loaded BLIP2 /
        segmentation / depth extractors so packet objects, scene and structural
        summaries stay consistent with the metrics.  The MuGE edge extractor is
        intentionally not wired here because it expects 128×128 patches, whereas
        packets are built from full images.
        """
        if self.packet_extractor is not None:
            return self.packet_extractor
        from sgdjscc_lab.guidance.semantic_packet_extractor import SemanticPacketExtractor

        text_extractor = getattr(models, "text_extractor", None) if models else None
        seg_extractor = None
        depth_extractor = None
        if cfg is not None and bool(cfg.get("packet_use_segmentation", False)):
            from sgdjscc_lab.guidance.segmentation_extractor import SegmentationExtractor
            seg_extractor = SegmentationExtractor(device=self._get_clip().device)
        if cfg is not None and bool(cfg.get("packet_use_depth", False)):
            from sgdjscc_lab.guidance.depth_extractor import DepthExtractor
            depth_extractor = DepthExtractor(device=self._get_clip().device)

        caption_objects = (
            bool(cfg.get("packet_caption_objects", True)) if cfg is not None else True
        )
        self.packet_extractor = SemanticPacketExtractor(
            text_extractor=text_extractor,
            clip_evaluator=self._get_clip(),
            segmentation_extractor=seg_extractor,
            depth_extractor=depth_extractor,
            device=self._get_clip().device,
            caption_objects=caption_objects,
        )
        return self.packet_extractor

    def _get_srs_v2(self):
        """Lazily build the SRS-v2 evaluator (base SRS + packet + temporal + VQA)."""
        if self.srs_v2_evaluator is None:
            from sgdjscc_lab.evaluators.semantic_reliability_v2 import SemanticReliabilityV2Evaluator
            vqa = None
            if self.use_vqa_hallucination:
                from sgdjscc_lab.evaluators.hallucination_vqa import VQAHallucinationEvaluator
                if self.vqa_fn is not None:
                    # Explicitly injected backend (e.g. tests).
                    vqa = VQAHallucinationEvaluator(
                        vqa_fn=self.vqa_fn, clip_evaluator=self._get_clip(),
                    )
                else:
                    # Build a real local backend from config (CLIP fallback if absent).
                    backend_cfg = self.vqa_backend_cfg
                    if backend_cfg is not None and self.device is not None \
                            and not backend_cfg.get("device"):
                        backend_cfg = {**backend_cfg, "device": str(self.device)}
                    vqa = VQAHallucinationEvaluator.from_config(
                        vqa_backend_cfg=backend_cfg,
                        clip_evaluator=self._get_clip(),
                    )
            self.srs_v2_evaluator = SemanticReliabilityV2Evaluator(
                base_evaluator=self._get_srs(), vqa_evaluator=vqa,
                weights=self.srs_v2_weights,
            )
        return self.srs_v2_evaluator


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
    orig_packet: Optional[Dict] = None,
    recon_packet: Optional[Dict] = None,
) -> Dict:
    """Compute all enabled metrics for one image pair.

    Returns a flat dict matching RESULT_COLUMNS (plus PACKET_RESULT_COLUMNS when
    packets are supplied).
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
        "fid":                      None,   # filled in per-SNR after the loop
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
                original, reconstructed, text_list=text_list,
                orig_packet=orig_packet, recon_packet=recon_packet,
            )
            row["clip_image_image"]           = srs_result.get("clip_image_image")
            row["clip_text_image"]            = srs_result.get("clip_text_image")
            row["object_preservation_rate"]   = srs_result.get("object_preservation_rate")
            row["missing_object_rate"]        = srs_result.get("missing_object_rate")
            row["additional_object_rate"]     = srs_result.get("additional_object_rate")
            row["hallucination_score"]        = srs_result.get("hallucination_score")
            row["semantic_reliability_score"] = srs_result.get("semantic_reliability_score")

            # ── Packet-aware columns (present only when packets supplied) ─────
            if orig_packet is not None and recon_packet is not None:
                row["srs_base"]                 = srs_result.get("srs_base")
                row["srs_packet"]               = srs_result.get("srs_packet")
                row["object_match_rate"]        = srs_result.get("object_match_rate")
                row["relation_consistency"]     = srs_result.get("relation_consistency")
                row["attribute_consistency"]    = srs_result.get("attribute_consistency")
                row["segmentation_consistency"] = srs_result.get("segmentation_consistency")
                row["scene_match"]              = srs_result.get("scene_match")
                row["missing_object_count"]     = srs_result.get("missing_object_count")
                row["additional_object_count"]  = srs_result.get("additional_object_count")
                row["relation_error_count"]     = srs_result.get("relation_error_count")
                row["attribute_error_count"]    = srs_result.get("attribute_error_count")
                row["_error_report"]            = srs_result.get("error_report")

            # ── Phase 5-C: SRS-v2 via the full evaluator ──────────────────────
            # Reuses the already-computed base SRS (no CLIP recompute) and adds the
            # packet layer (when packets present), the temporal layer (when
            # supplied) and a stronger VQA hallucination layer (when enabled,
            # CLIP fallback otherwise). Missing layers are renormalised away.
            if eval_ctx.use_srs_v2:
                v2 = eval_ctx._get_srs_v2().evaluate(
                    original, reconstructed, text_list=text_list,
                    orig_packet=orig_packet, recon_packet=recon_packet,
                    base_result=srs_result,
                )
                row["srs_v2"] = v2.get("srs_v2")
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

    caller_provided_reconstruct = reconstruct_fn is not None

    # Update SNR on the model (no rebuild required)
    if models is not None and hasattr(models, "jscc_model"):
        models.jscc_model.snr = float(snr_db)

    files = list_image_files(cfg.input_path)
    logger.info("evaluate_dataset: %d images, SNR=%.1f dB", len(files), snr_db)

    # Phase master switches
    from sgdjscc_lab.phase_gates import effective_flag as _eff, phase4_enabled, phase5_enabled

    # Regeneration loop settings (read from cfg; off by default)
    use_regen       = bool(cfg.get("use_regeneration_loop", False))
    regen_threshold = float(cfg.get("regeneration_threshold", 0.5))
    regen_max_retries = int(cfg.get("regeneration_max_retries", 1))

    # ── Phase 4-A settings (gated by use_phase4) ──────────────────────────────
    use_packet       = _eff(cfg, "use_packet_eval", phase=4)
    use_packet_regen = _eff(cfg, "use_packet_regeneration", phase=4)
    packet_dir       = cfg.get("packet_dir", None) if use_packet else None

    # ── Phase 5 settings (gated by use_phase5) ────────────────────────────────
    use_channel_cond = _eff(cfg, "use_channel_conditioning", phase=5)
    use_regen_search = _eff(cfg, "use_regeneration_search", phase=5)

    # Phase 4-A: adaptive guidance (only when phase4 master switch is on).
    if phase4_enabled(cfg):
        from sgdjscc_lab.controllers.adaptive_guidance_controller import (
            maybe_apply_adaptive_guidance,
        )
        eval_cfg, guidance_decision = maybe_apply_adaptive_guidance(cfg, snr_db)
    else:
        eval_cfg, guidance_decision = cfg, None

    # Phase 5-B: sampler / step-budget override (only when phase5 master switch is on).
    if phase5_enabled(cfg):
        from sgdjscc_lab.acceleration import build_sampler_cfg
        eval_cfg, sampler_spec = build_sampler_cfg(eval_cfg)
        if sampler_spec.sampler_type != "baseline" or "acceleration" in cfg:
            logger.info("Sampler: %s @ %d steps", sampler_spec.sampler_type, sampler_spec.steps)

    # Phase 5-A: route reconstruction through the channel-conditioned path when
    # enabled and the caller did not inject a custom reconstruct_fn.
    if use_channel_cond and models is not None and not caller_provided_reconstruct:
        from sgdjscc_lab.pipelines.channel_conditioned_infer import (
            build_channel_conditioned_inference,
        )
        from sgdjscc_lab.io import load_image_as_tensor as _load_img
        _cci = build_channel_conditioned_inference(models, eval_cfg)
        logger.info("Channel-conditioned reconstruction enabled (mode=%s, csi=%s).",
                    eval_cfg.get("condition_mode", "auto"), eval_cfg.get("csi", "perfect"))

        def _cc_reconstruct(fpath, models_, cfg_):
            raw = _load_img(fpath)
            out = _cci.run(raw, snr_db=snr_db)
            return raw, out["reconstruction"]

        reconstruct_fn = _cc_reconstruct
    elif reconstruct_fn is None:
        reconstruct_fn = _default_reconstruct

    packet_extractor = (
        eval_ctx._get_packet_extractor(models, cfg) if use_packet else None
    )

    # ── FID (paper §VI): dataset-level. Accumulate Inception features across all
    # images, compute one scalar after the loop, then fill it into every row of
    # this SNR group. Because the value is only known post-loop, CSV streaming is
    # deferred to after the loop when FID is enabled (rows are still buffered).
    fid_eval = eval_ctx._get_fid() if "fid" in eval_ctx.enabled_metrics else None
    stream_rows = csv_logger is not None and fid_eval is None

    rows: List[Dict] = []
    for idx, fpath in enumerate(files, start=1):
        logger.info("[%d/%d] %s", idx, len(files), fpath.name)
        txt = [text_list[idx - 1]] if (text_list and idx <= len(text_list)) else None

        try:
            original, reconstructed = reconstruct_fn(fpath, models, eval_cfg)

            orig_packet = recon_packet = None
            if packet_extractor is not None:
                orig_packet = packet_extractor.extract(original, frame_id=f"{fpath.stem}_orig")
                recon_packet = packet_extractor.extract(reconstructed, frame_id=fpath.stem)

            row = _compute_metrics(
                original, reconstructed, eval_ctx,
                filename=fpath.name, snr_db=snr_db, text_list=txt,
                orig_packet=orig_packet, recon_packet=recon_packet,
            )
            if guidance_decision is not None:
                row["guidance_regime"] = guidance_decision.regime

            # ── Regeneration ─────────────────────────────────────────────────
            if use_packet and use_packet_regen and orig_packet is not None and models is not None:
                reconstructed, recon_packet, row = _run_packet_regeneration(
                    original, reconstructed, recon_packet, orig_packet, row,
                    eval_cfg, models, eval_ctx, packet_extractor,
                    filename=fpath.name, snr_db=snr_db, text_list=txt,
                    threshold=regen_threshold, max_retries=regen_max_retries,
                    regime=(guidance_decision.regime if guidance_decision else None),
                )
            elif use_regen and row.get("semantic_reliability_score") is not None:
                from sgdjscc_lab.pipelines.regeneration_loop import (
                    regenerate_if_needed, build_regeneration_pipeline,
                )
                pipeline_fn = build_regeneration_pipeline(eval_cfg)
                metrics_for_regen = {"semantic_reliability_score": row["semantic_reliability_score"]}
                new_recon, _ = regenerate_if_needed(
                    original, reconstructed, metrics_for_regen,
                    threshold=regen_threshold,
                    pipeline_fn=pipeline_fn,
                    cfg=eval_cfg,
                    models=models,
                    max_retries=regen_max_retries,
                )
                if new_recon is not reconstructed:
                    logger.info("Regeneration improved SRS for %s; recomputing metrics.", fpath.name)
                    # Adopt the regenerated reconstruction as THE reconstruction so
                    # every downstream consumer (FID accumulation, packet metrics,
                    # packet save) uses the same tensor the row metrics were computed
                    # on — otherwise FID / packets would be stale (initial recon).
                    reconstructed = new_recon
                    # Re-extract the recon packet from the NEW reconstruction and feed
                    # both packets back in, so the recomputed row's packet columns and
                    # the saved recon_packet match the final image (was: packet args
                    # dropped here → stale packet metrics/artifacts).
                    if packet_extractor is not None:
                        recon_packet = packet_extractor.extract(new_recon, frame_id=fpath.stem)
                    row = _compute_metrics(
                        original, new_recon, eval_ctx,
                        filename=fpath.name, snr_db=snr_db, text_list=txt,
                        orig_packet=orig_packet, recon_packet=recon_packet,
                    )
                    if guidance_decision is not None:
                        row["guidance_regime"] = guidance_decision.regime

            # ── Phase 5-C: multi-strategy regeneration search ────────────────
            if use_regen_search and models is not None:
                reconstructed, recon_packet, row = _run_regeneration_search(
                    original, reconstructed, recon_packet, orig_packet, row,
                    eval_cfg, models, eval_ctx, packet_extractor,
                    filename=fpath.name, snr_db=snr_db, text_list=txt, cfg=cfg,
                    regime=(guidance_decision.regime if guidance_decision else None),
                )

            # ── FID accumulation (final reconstruction for this image) ────────
            if fid_eval is not None:
                fid_eval.add(original, reconstructed)

            # ── Persist packets / error report ───────────────────────────────
            # Namespace by SNR so an SNR sweep does not overwrite earlier results
            # for the same image (reproducibility).
            if packet_dir is not None and orig_packet is not None:
                from pathlib import Path as _Path
                snr_packet_dir = _Path(packet_dir) / f"snr_{snr_db:g}"
                _save_packets(snr_packet_dir, fpath.stem, orig_packet, recon_packet, row)

        except Exception as exc:
            logger.error("Failed on %s: %s", fpath.name, exc, exc_info=True)
            row = {"filename": fpath.name, "snr_db": snr_db}

        rows.append(row)
        if stream_rows:
            csv_logger.write_row(_csv_safe(row))

    # ── Dataset-level FID: compute once, fill every row, then flush to CSV ─────
    if fid_eval is not None:
        fid_value = fid_eval.compute()
        backend = fid_eval.backend_name      # "inception" | "proxy" | "unavailable"
        for r in rows:
            r["fid"] = fid_value
            r["fid_backend"] = backend       # persisted so results stay unambiguous
        logger.info("FID @ SNR=%.1f dB: %s (backend=%s, %d samples)", snr_db,
                    f"{fid_value:.4f}" if fid_value is not None else "None",
                    backend, fid_eval.n_samples())
        if csv_logger is not None:
            for r in rows:
                csv_logger.write_row(_csv_safe(r))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4-A helpers
# ─────────────────────────────────────────────────────────────────────────────

def _csv_safe(row: Dict) -> Dict:
    """Strip private/non-scalar keys (e.g. ``_error_report``) before CSV write."""
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _save_packets(packet_dir, stem, orig_packet, recon_packet, row) -> None:
    """Serialise original / reconstructed packets and the error report."""
    from sgdjscc_lab.utils.packet_io import (
        save_packet, packet_path, orig_packet_path, save_error_report,
    )
    try:
        save_packet(orig_packet, orig_packet_path(packet_dir, stem))
        if recon_packet is not None:
            save_packet(recon_packet, packet_path(packet_dir, stem))
        report = row.get("_error_report")
        if report is not None:
            save_error_report(report, packet_dir, stem)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save packets for %s: %s", stem, exc)


def _reconstruct_with_cfg(original: torch.Tensor, models, cfg: DictConfig) -> torch.Tensor:
    """Re-run the SGD-JSCC forward pass on *original* with a modified config."""
    from sgdjscc_lab.utils.preprocessing import prepare_patches, merge_patches
    from sgdjscc_lab.pipelines.infer_pipeline import _process_patches

    patches, meta = prepare_patches(original)
    patches = patches.to(models.device)
    out_patches = _process_patches(patches, models, cfg)
    return merge_patches(out_patches.cpu(), meta)


def _run_packet_regeneration(
    original, reconstructed, recon_packet, orig_packet, row,
    cfg, models, eval_ctx, packet_extractor,
    filename, snr_db, text_list, threshold, max_retries, regime=None,
):
    """Error-type-aware regeneration keyed on the packet-matcher report.

    Selects strategies via :class:`RegenerationPolicy` (missing-object →
    strengthen text; hallucination → weaken text + strengthen edge; structural →
    raise control / steps), re-runs inference per strategy, and keeps the best
    result by ``srs_packet``.  Returns ``(reconstructed, recon_packet, row)``.
    """
    from sgdjscc_lab.controllers.regeneration_policy import (
        RegenerationPolicy, apply_strategy,
    )

    srs = row.get("srs_packet", row.get("semantic_reliability_score"))
    if srs is None or threshold <= 0.0 or srs >= threshold:
        return reconstructed, recon_packet, row

    report = row.get("_error_report")
    strategies = RegenerationPolicy().select(error_report=report, metrics=row)
    if not strategies:
        return reconstructed, recon_packet, row

    best_recon, best_packet, best_row, best_srs = reconstructed, recon_packet, row, float(srs)
    for strat in strategies[:max_retries]:
        retry_cfg = apply_strategy(cfg, strat)
        logger.info(
            "Packet regen [%s] for %s: gs=%.2f cn=%.2f steps=%d",
            strat.name, filename, retry_cfg.guidance_scale,
            retry_cfg.controlnet_scale, retry_cfg.diffusion_step,
        )
        try:
            new_recon = _reconstruct_with_cfg(original, models, retry_cfg)
            new_packet = packet_extractor.extract(new_recon, frame_id=filename)
            new_row = _compute_metrics(
                original, new_recon, eval_ctx,
                filename=filename, snr_db=snr_db, text_list=text_list,
                orig_packet=orig_packet, recon_packet=new_packet,
            )
            if regime is not None:
                new_row["guidance_regime"] = regime
            new_srs = new_row.get("srs_packet", new_row.get("semantic_reliability_score", -999))
            if new_srs is not None and new_srs > best_srs:
                best_recon, best_packet, best_row, best_srs = new_recon, new_packet, new_row, new_srs
        except Exception as exc:  # noqa: BLE001
            logger.warning("Packet regen strategy %s failed: %s", strat.name, exc)

    return best_recon, best_packet, best_row


def _run_regeneration_search(
    original, reconstructed, recon_packet, orig_packet, row,
    eval_cfg, models, eval_ctx, packet_extractor,
    filename, snr_db, text_list, cfg, regime=None,
):
    """Phase 5-C: multi-strategy regeneration search keeping the best verified SRS.

    Tries strong/weak-text, unconditional and channel-conditioned-retry strategies
    (ordered by ``AdaptiveSearchPolicy``) and replaces the reconstruction with the
    highest verified-SRS candidate.  Returns ``(reconstructed, recon_packet, row)``.
    """
    from sgdjscc_lab.evaluators.regeneration_search import RegenerationSearch

    srs_eval = eval_ctx._get_srs()
    max_strat = int(OmegaConf.select(cfg, "regeneration_search.max_strategies", default=3))
    # Honour the configured verifier: "srs" (base) or "srs_v2" (packet/temporal/VQA).
    metric = str(OmegaConf.select(cfg, "regeneration_search.verify_metric", default="srs"))
    use_v2 = metric == "srs_v2"

    def _reconstruct(cfg_i):
        if bool(cfg_i.get("use_channel_conditioning", False)):
            from sgdjscc_lab.pipelines.channel_conditioned_infer import (
                build_channel_conditioned_inference,
            )
            cci = build_channel_conditioned_inference(models, cfg_i)
            return cci.run(original, snr_db=snr_db)["reconstruction"]
        return _reconstruct_with_cfg(original, models, cfg_i)

    def _verify(recon):
        base = srs_eval.evaluate(original, recon, text_list=text_list)
        if not use_v2:
            return base.get("semantic_reliability_score", -999.0)
        rp = packet_extractor.extract(recon, frame_id=filename) if packet_extractor is not None else None
        # Re-score base with the candidate's packet so srs_packet reflects it.
        base = srs_eval.evaluate(
            original, recon, text_list=text_list,
            orig_packet=orig_packet, recon_packet=rp,
        )
        v2 = eval_ctx._get_srs_v2().evaluate(
            original, recon, text_list=text_list,
            orig_packet=orig_packet, recon_packet=rp, base_result=base,
        )
        score = v2.get("srs_v2")
        return -999.0 if score is None else float(score)

    # Score the current reconstruction with the SAME verifier for a fair baseline.
    try:
        init_score = _verify(reconstructed)
    except Exception:  # noqa: BLE001
        init_score = row.get("srs_v2") if use_v2 else row.get("semantic_reliability_score")

    out = RegenerationSearch(reconstruct_fn=_reconstruct, verify_fn=_verify).search(
        eval_cfg,
        error_report=row.get("_error_report"),
        hallucination_score=row.get("hallucination_score"),
        channel_state={"csi": cfg.get("csi", "perfect"), "snr_db": snr_db},
        initial_recon=reconstructed, initial_score=init_score,
        max_strategies=max_strat,
    )
    best = out["best_recon"]
    if best is None or best is reconstructed:
        row["regeneration_strategy"] = out.get("best_strategy")
        return reconstructed, recon_packet, row

    logger.info("Regeneration search chose '%s' for %s (SRS %.4f).",
                out.get("best_strategy"), filename, out.get("best_score") or -1.0)
    new_packet = (
        packet_extractor.extract(best, frame_id=filename)
        if packet_extractor is not None else None
    )
    new_row = _compute_metrics(
        original, best, eval_ctx, filename=filename, snr_db=snr_db,
        text_list=text_list, orig_packet=orig_packet, recon_packet=new_packet,
    )
    if regime is not None:
        new_row["guidance_regime"] = regime
    new_row["regeneration_strategy"] = out.get("best_strategy")
    return best, new_packet, new_row


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
