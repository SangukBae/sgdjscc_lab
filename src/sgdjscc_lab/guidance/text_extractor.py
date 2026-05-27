"""text_extractor.py – BLIP2-based text caption extractor.

Extracted from _extract_caption() / _build_caption_model() in runtime.py and
pipeline.py.  The preprocessing.py duplicate (extract_caption) is now a thin
re-export of this module.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sgdjscc_lab._sgdjscc import ensure_sgdjscc_on_path
from sgdjscc_lab.utils.memory import release_cuda_memory

logger = logging.getLogger(__name__)


class TextExtractor:
    """Wraps a BLIP2 model to generate per-image text captions.

    Implements the SemanticGuideExtractor.extract() interface from the README.
    """

    def __init__(self, model) -> None:
        self._model = model

    @property
    def model(self):
        return self._model

    def extract(
        self,
        img_tensor: torch.Tensor,
        device: torch.device,
        offload_device: Optional[torch.device] = None,
        offload_after: bool = False,
    ) -> list:
        """Generate BLIP2 captions for a batch of 128×128 image tensors.

        Mirrors image_caption() call in inference_one.py.

        Parameters
        ----------
        img_tensor:
            ``[N, 3, 128, 128]`` float in [0, 1], on any device.
        device:
            Device to move the model to before extraction.
        offload_device:
            If offload_after=True, move the model here after extraction.
        offload_after:
            Move the model to offload_device after extraction to free VRAM.

        Returns
        -------
        list
            List-of-lists, e.g. ``[["a cat on a mat"]]``.
        """
        ensure_sgdjscc_on_path()
        from utils.utils import image_caption

        self._model.to(device)
        try:
            with torch.inference_mode():
                return image_caption(self._model, img_tensor, device)
        finally:
            if offload_after and offload_device is not None:
                self._model.to(offload_device)
                release_cuda_memory()


def build_text_extractor(device: torch.device) -> TextExtractor:
    """Load BLIP2 (Salesforce/blip2-opt-2.7b-coco) and return a TextExtractor.

    Mirrors _build_caption_model() from runtime.py (originally
    inference_one.py lines 276–279).
    """
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    logger.info("Loading BLIP2 (Salesforce/blip2-opt-2.7b-coco)…")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b-coco", torch_dtype=torch.float16
    )
    model.processor = processor
    model.eval()
    model.to(device)
    logger.info("BLIP2 ready.")
    return TextExtractor(model)
