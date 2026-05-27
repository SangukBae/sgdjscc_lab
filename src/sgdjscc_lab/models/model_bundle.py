"""model_bundle.py – ModelBundle dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ModelBundle:
    """Container for all loaded models passed to the inference pipeline.

    Fields
    ------
    jscc_model
        JSCCModel instance (VAE + SNR predictor + canny TX net).
    sem_pipeline
        DiffusionGenerator instance, or None when use_semantic=False.
    text_extractor
        TextExtractor instance (BLIP2), or None when use_text=False.
    edge_extractor
        EdgeExtractor instance (MuGE), or None when use_semantic=False.
    device
        Primary compute device.
    offload_device
        CPU device used when offloading large models between patches.
    offload_caption
        If True, move caption model to CPU after each extraction call.
    offload_canny
        If True, move canny net to CPU after each extraction call.
    extra
        Reserved for future extensions (e.g. depth estimator, seg model).
    """

    jscc_model: torch.nn.Module
    sem_pipeline: object
    text_extractor: Optional[object]
    edge_extractor: Optional[object]
    device: torch.device
    offload_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    offload_caption: bool = False
    offload_canny: bool = False
    extra: dict = field(default_factory=dict)

    # ── Backward-compatible accessors ────────────────────────────────────────
    @property
    def caption_model(self):
        """Phase-1 compat: return the raw BLIP2 model from the extractor."""
        return self.text_extractor._model if self.text_extractor is not None else None

    @property
    def canny_net(self):
        """Phase-1 compat: return the raw MuGE model from the extractor."""
        return self.edge_extractor._model if self.edge_extractor is not None else None
