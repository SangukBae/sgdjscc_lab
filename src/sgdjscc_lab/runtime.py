"""runtime.py – Model loading assembly point (Phase 2 compatibility shim).

Phase 2 restructuring moved the individual builders to:
  • models/jscc_model.py      → build_jscc_model, JSCCModel
  • models/diffusion_wrapper.py → build_diffusion_pipeline
  • models/model_bundle.py    → ModelBundle
  • guidance/text_extractor.py → build_text_extractor
  • guidance/edge_extractor.py → build_edge_extractor

This file retains the public surface (resolve_device, build_models) so that
existing scripts continue to work without modification.
"""

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig

from sgdjscc_lab.models.jscc_model import build_jscc_model
from sgdjscc_lab.models.diffusion_wrapper import build_diffusion_pipeline
from sgdjscc_lab.models.model_bundle import ModelBundle

logger = logging.getLogger(__name__)


def resolve_device(device_str: str) -> torch.device:
    """Parse a device string; fall back to CPU if CUDA is unavailable."""
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def build_models(cfg: DictConfig, device: torch.device) -> ModelBundle:
    """Build and load all models required for inference.

    Orchestrates the individual builders in models/ and guidance/ to produce
    a ModelBundle ready for pipelines/infer_pipeline.run_batch().
    """
    from pathlib import Path
    from sgdjscc_lab.guidance.text_extractor import build_text_extractor
    from sgdjscc_lab.guidance.edge_extractor import build_edge_extractor

    model_root = Path(cfg.model_root)
    logger.info("Loading models from: %s", model_root)

    # ── 1. JSCC model ─────────────────────────────────────────────────────────
    jscc_model = build_jscc_model(model_root, device)
    jscc_model.snr = float(cfg.snr_db)

    # Optionally replace the (public) blind SNR predictor with one trained by the
    # `csi_estimation` stage — this is what connects that stage's checkpoint to the
    # actual inference blind step-matching path (jscc.snr_prediction_net).
    snr_ckpt = cfg.get("snr_estimator_checkpoint", None)
    if snr_ckpt:
        from sgdjscc_lab.models.csi_estimation import load_snr_estimator_into
        _train = cfg.get("train", None)
        _csi = _train.get("csi_estimation", None) if _train is not None else None
        latent_ch = int(_csi.get("latent_ch", 16)) if _csi is not None else 16
        load_snr_estimator_into(jscc_model, str(snr_ckpt), latent_ch=latent_ch, device=device)

    # ── 2. Semantic pipeline (diffusion + guidance) ───────────────────────────
    text_extractor = None
    edge_extractor = None
    sem_pipeline   = None

    if cfg.use_semantic:
        if cfg.use_text:
            text_extractor = build_text_extractor(device)
        edge_extractor = build_edge_extractor(model_root, device)
        sem_pipeline   = build_diffusion_pipeline(cfg, device, jscc_model)

    return ModelBundle(
        jscc_model=jscc_model,
        sem_pipeline=sem_pipeline,
        text_extractor=text_extractor,
        edge_extractor=edge_extractor,
        device=device,
        offload_caption=False,
        offload_canny=False,
    )
