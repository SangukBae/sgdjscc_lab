"""guidance/depth_extractor.py – Monocular depth estimation extractor.

Uses the DPT (Dense Prediction Transformer) model from the ``transformers``
library (Intel/dpt-large by default) for per-pixel depth estimation.

Why depth guidance?
-------------------
Depth maps capture 3D structure independent of texture.  When used as
ControlNet conditioning alongside the Canny edge map, depth preserves large-scale
spatial layout (foreground/background separation, object scale relationships)
that Canny edges alone cannot capture.  Phase 5+ can wire this extractor into
the semantic pipeline as an additional guidance channel.

Output specification
--------------------
``extract()`` returns a ``[N, 1, H, W]`` float tensor of raw (metric) depth
values in metres.  Values are *not* normalised so callers can apply their own
normalisation strategy (e.g. min-max per image, or fixed range for ControlNet).

Example
-------
>>> extractor = DepthExtractor(device=torch.device("cuda:0"))
>>> depth = extractor.extract(image)   # image: [1, 3, H, W] in [0, 1]
>>> depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Intel/dpt-large"


class DepthExtractor:
    """Monocular depth estimator based on DPT-Large.

    Parameters
    ----------
    model_name:
        HuggingFace model hub identifier.  Default ``'Intel/dpt-large'``.
        Alternative: ``'Intel/dpt-hybrid-midas'`` (smaller, slightly lower quality).
    device:
        Compute device.  Defaults to CPU.

    Notes
    -----
    The first call downloads the model weights (~1.3 GB for dpt-large) from
    HuggingFace Hub.  Subsequent calls use the local cache.

    Requires: ``transformers >= 4.26``, ``Pillow``, ``torch``.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or torch.device("cpu")
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import DPTForDepthEstimation, DPTImageProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers package required for depth extraction. "
                "Install with: pip install transformers"
            ) from exc

        logger.info("Loading depth model: %s", self.model_name)
        self._processor = DPTImageProcessor.from_pretrained(self.model_name)
        self._model = (
            DPTForDepthEstimation.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )
        logger.info("Depth model loaded on %s", self.device)

    def extract(
        self,
        image: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Estimate per-pixel depth for a batch of images.

        Parameters
        ----------
        image:
            ``[N, 3, H, W]`` float tensor in [0, 1].
        device:
            Override compute device for this call (default: self.device).

        Returns
        -------
        torch.Tensor
            ``[N, 1, H, W]`` float tensor of depth values (in metres, raw from model).
            Spatial resolution matches *image*'s H × W (bilinearly upsampled from
            model output resolution).
        """
        if device is not None:
            self.device = device
        self._load()

        from PIL import Image as PILImage
        import numpy as np

        n, c, h, w = image.shape
        image_cpu = image.float().clamp(0, 1).cpu()

        pil_list = []
        for i in range(n):
            arr = (image_cpu[i].permute(1, 2, 0).numpy() * 255).astype("uint8")
            pil_list.append(PILImage.fromarray(arr))

        inputs = self._processor(images=pil_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # predicted_depth: [N, H', W']
            depth = outputs.predicted_depth

        # Upsample to original resolution
        depth = depth.unsqueeze(1).float()                          # [N, 1, H', W']
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)

        return depth.cpu()
