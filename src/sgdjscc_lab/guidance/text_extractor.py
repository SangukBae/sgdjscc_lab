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


def _dtype_for_device(device: torch.device) -> torch.dtype:
    """fp16 on CUDA (fast), fp32 on CPU (no fp16 conv kernel on CPU)."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    return torch.float16 if dev.type == "cuda" else torch.float32


def _align_model_device_dtype(model, device) -> None:
    """Move *model* to *device* and cast to the device-appropriate dtype in place."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    model.to(device=dev, dtype=_dtype_for_device(dev))


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

        # Move to the target device AND re-normalise dtype: a fp16 model moved to
        # CPU would otherwise crash on the half conv kernel. CPU → float32,
        # CUDA → float16 (matches the original fast path).
        _align_model_device_dtype(self._model, device)
        try:
            with torch.inference_mode():
                return image_caption(self._model, img_tensor, device)
        finally:
            if offload_after and offload_device is not None:
                _align_model_device_dtype(self._model, offload_device)
                release_cuda_memory()


def build_text_extractor(device: torch.device) -> TextExtractor:
    """Load BLIP2 (Salesforce/blip2-opt-2.7b-coco) and return a TextExtractor.

    Mirrors _build_caption_model() from runtime.py (originally
    inference_one.py lines 276–279).

    The dtype is chosen by device: ``float16`` on CUDA (fast, matches the original
    path), but ``float32`` on CPU — PyTorch has no fp16 conv kernel on CPU, so a
    half model raises ``"slow_conv2d_cpu" not implemented for 'Half'`` whenever the
    caption model runs on CPU (e.g. ``--device cpu`` or a CPU-side packet extractor).
    """
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    dtype = _dtype_for_device(dev)

    logger.info("Loading BLIP2 (Salesforce/blip2-opt-2.7b-coco) [%s, %s]…", dev, dtype)
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b-coco", torch_dtype=dtype
    )
    model.processor = processor
    model.eval()
    model.to(dev)
    logger.info("BLIP2 ready.")
    return TextExtractor(model)
