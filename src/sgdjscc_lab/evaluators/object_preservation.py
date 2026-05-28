"""evaluators/object_preservation.py – Object preservation rate evaluator.

Approach (Phase 3 – CLIP text-probing heuristic)
-------------------------------------------------
A vocabulary of common object categories (default: 80 COCO classes) is probed
against the original and reconstructed images using CLIP text-image similarity.

An object is considered "present" in an image when the maximum per-category
CLIP similarity exceeds ``presence_threshold``.

preservation_rate = |objects present in both| / |objects present in original|

Limitations
-----------
- Relies on CLIP's zero-shot text alignment; may misfire on rare objects or
  unusual viewpoints.
- Presence threshold (default 0.25) requires per-dataset calibration for
  precise absolute numbers; relative comparisons (original vs reconstructed)
  are more reliable.
- A full detector-based version (e.g. DETIC, YOLOv8) would be more accurate
  but requires additional heavyweight dependencies not in the Phase 3 scope.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# COCO 80-class vocabulary (used as default object probe set)
_COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class ObjectPreservationEvaluator:
    """Evaluate how many objects present in the original survive in reconstruction.

    Parameters
    ----------
    clip_evaluator:
        A ``CLIPScoreEvaluator`` instance.  If None, a default one is created
        on first use (device defaults to CPU).
    vocabulary:
        List of object category names to probe.  Defaults to COCO 80 classes.
    presence_threshold:
        CLIP similarity threshold above which an object is considered "present".
        Default 0.25; lower values detect more objects but increase false positives.
    device:
        Compute device.  Ignored when *clip_evaluator* is provided.
    """

    def __init__(
        self,
        clip_evaluator=None,
        vocabulary: Optional[List[str]] = None,
        presence_threshold: float = 0.25,
        device: Optional[torch.device] = None,
    ) -> None:
        self._clip = clip_evaluator
        self.vocabulary = vocabulary or _COCO_CLASSES
        self.presence_threshold = presence_threshold
        self._device = device or torch.device("cpu")
        self._text_features: Optional[torch.Tensor] = None

    def _get_clip(self):
        if self._clip is None:
            from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
            self._clip = CLIPScoreEvaluator(device=self._device)
        return self._clip

    def _get_text_features(self) -> torch.Tensor:
        """Encode vocabulary into CLIP text features (cached after first call)."""
        if self._text_features is not None:
            return self._text_features
        clip_eval = self._get_clip()
        clip_eval._load()
        prompts = [f"a photo of a {obj}" for obj in self.vocabulary]
        self._text_features = clip_eval._encode_texts(prompts)  # [V, D]
        return self._text_features

    def _detect_objects(self, image: torch.Tensor) -> List[str]:
        """Return list of vocabulary items detected in *image*.

        Parameters
        ----------
        image:
            ``[1, 3, H, W]`` float tensor in [0, 1].
        """
        clip_eval = self._get_clip()
        img_feat = clip_eval._encode_images(image)          # [1, D]
        txt_feat = self._get_text_features()                # [V, D]
        sims = (img_feat @ txt_feat.T).squeeze(0)           # [V]
        detected = [
            self.vocabulary[i]
            for i, s in enumerate(sims.tolist())
            if s >= self.presence_threshold
        ]
        return detected

    def evaluate(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Compute object preservation metrics.

        Parameters
        ----------
        original, reconstructed:
            ``[N, 3, H, W]`` float tensors in [0, 1].  Each sample is
            evaluated independently; results are averaged over the batch.
        metadata:
            Optional dict (unused in Phase 3; reserved for GT annotations).

        Returns
        -------
        dict with keys:
            ``preservation_rate`` – fraction of original objects found in reconstruction.
            ``matched_objects``   – objects present in both (list from last sample).
            ``missing_objects``   – objects lost in reconstruction (list from last sample).
            ``original_count``    – mean number of detected objects in original.
            ``reconstructed_count`` – mean number in reconstruction.
        """
        n = original.shape[0]
        rates: List[float] = []
        matched_last: List[str] = []
        missing_last: List[str] = []
        orig_counts: List[int] = []
        recon_counts: List[int] = []

        for i in range(n):
            orig_objs  = set(self._detect_objects(original[i:i+1]))
            recon_objs = set(self._detect_objects(reconstructed[i:i+1]))

            matched = orig_objs & recon_objs
            missing = orig_objs - recon_objs

            rate = len(matched) / max(len(orig_objs), 1)
            rates.append(rate)
            orig_counts.append(len(orig_objs))
            recon_counts.append(len(recon_objs))
            matched_last = sorted(matched)
            missing_last = sorted(missing)

        return {
            "preservation_rate":    float(sum(rates) / max(len(rates), 1)),
            "matched_objects":      matched_last,
            "missing_objects":      missing_last,
            "original_count":       float(sum(orig_counts) / max(len(orig_counts), 1)),
            "reconstructed_count":  float(sum(recon_counts) / max(len(recon_counts), 1)),
        }
