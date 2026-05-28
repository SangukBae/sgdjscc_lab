"""guidance/segmentation_extractor.py – Semantic segmentation extractor.

Uses the SegFormer model from the ``transformers`` library
(nvidia/segformer-b0-finetuned-ade-512-512 by default) for semantic segmentation.

Why segmentation guidance?
--------------------------
Segmentation maps provide region-level structural information: which parts of
the image are sky, road, vegetation, people, etc.  Unlike Canny edges (which
mark boundaries between regions), segmentation provides category-aware region
masks that can guide reconstruction at the semantic level.

In Phase 3+, the extracted label map can be used for:
  - Region-dropout corruption to simulate partial structural-guide loss
  - Object-region-based preservation scoring
  - ControlNet conditioning with semantic-aware masks

Output specification
--------------------
``extract()`` returns a dict with:
  ``label_map``   : ``[N, H, W]`` int64 tensor of class indices.
  ``num_classes`` : total number of classes (150 for ADE20K default model).
  ``class_names`` : list of class name strings.

Example
-------
>>> extractor = SegmentationExtractor(device=torch.device("cuda:0"))
>>> result = extractor.extract(image)   # image: [1, 3, H, W] in [0, 1]
>>> label_map = result["label_map"]     # [1, H, W] int64
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"


class SegmentationExtractor:
    """Semantic segmentation extractor based on SegFormer-B0 (ADE20K).

    Parameters
    ----------
    model_name:
        HuggingFace model hub identifier.
        Default: ``'nvidia/segformer-b0-finetuned-ade-512-512'`` (150 ADE20K classes).
        Alternative: ``'nvidia/segformer-b2-finetuned-ade-512-512'`` (higher quality).
    device:
        Compute device.  Defaults to CPU.

    Notes
    -----
    The first call downloads model weights (~14 MB for B0) from HuggingFace Hub.

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
        self._id2label: Dict[int, str] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers package required for segmentation extraction. "
                "Install with: pip install transformers"
            ) from exc

        logger.info("Loading segmentation model: %s", self.model_name)
        self._processor = SegformerImageProcessor.from_pretrained(self.model_name)
        self._model = (
            SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )
        self._id2label = self._model.config.id2label
        logger.info(
            "Segmentation model loaded on %s (%d classes)",
            self.device,
            len(self._id2label),
        )

    def extract(
        self,
        image: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Dict:
        """Segment a batch of images into semantic regions.

        Parameters
        ----------
        image:
            ``[N, 3, H, W]`` float tensor in [0, 1].
        device:
            Override compute device for this call (default: self.device).

        Returns
        -------
        dict with keys:
            ``label_map``   : ``[N, H, W]`` int64 tensor of class indices.
                              Shape matches input *image* (bilinearly upsampled).
            ``num_classes`` : number of segmentation classes in the model.
            ``class_names`` : list of class name strings indexed by class id.
        """
        if device is not None:
            self.device = device
        self._load()

        from PIL import Image as PILImage

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
            logits = outputs.logits   # [N, num_classes, H', W']

        # Upsample to original resolution and take argmax
        logits_up = F.interpolate(
            logits.float(), size=(h, w), mode="bilinear", align_corners=False
        )  # [N, num_classes, H, W]
        label_map = logits_up.argmax(dim=1).cpu().to(torch.int64)   # [N, H, W]

        num_classes = len(self._id2label)
        class_names = [self._id2label[i] for i in range(num_classes)]

        return {
            "label_map":   label_map,
            "num_classes": num_classes,
            "class_names": class_names,
        }
