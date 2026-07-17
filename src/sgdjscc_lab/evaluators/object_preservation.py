"""evaluators/object_preservation.py – Object preservation rate evaluator.

Approach (Phase 3 – CLIP text-probing heuristic)
-------------------------------------------------
A vocabulary of common object categories (default: 80 COCO classes) is probed
against the original and reconstructed images using CLIP text-image similarity.

An object is considered "present" in an image when the maximum per-category
CLIP similarity exceeds ``presence_threshold``.

preservation_rate = |objects present in both| / |objects present in original|

Uncertain band / hysteresis (optional, default off)
---------------------------------------------------
``uncertain_band`` (default 0.0) opens a hysteresis band of similarity scores
around the presence threshold: an object that was confidently detected in the
*original* (score >= threshold) still counts as *preserved* in the
reconstruction when its reconstruction score stays above
``threshold - uncertain_band``.  This makes borderline CLIP scores less likely
to flip an object between "preserved" and "missing" (the flicker source noted
in docs/etri_strategy.md).  With the default band of 0.0 the behaviour is
bit-identical to the original single-threshold rule.

PROVISIONAL IMPLEMENTATION NOTE (ETRI plan step 0 / 슬라이드 6·7)
----------------------------------------------------------------
The CLIP global text-image probe used here is an *interim* presence judge.
It is threshold-sensitive and known to misfire on rare objects; per the ETRI
strategy (docs/etri_strategy.md, 5차 단계) it will be reinforced/replaced by a
grounded detector (OWLv2) and VQA-based presence verification, after which all
presence-derived metrics (preservation / hallucination / SRS / PTC / SFR / SDI)
must be re-measured.  Keep the constructor interface stable so those backends
can slot in behind the same ``_detect_objects`` contract.

Limitations
-----------
- Relies on CLIP's zero-shot text alignment; may misfire on rare objects or
  unusual viewpoints.
- Presence threshold (default 0.25) requires per-dataset calibration for
  precise absolute numbers; relative comparisons (original vs reconstructed)
  are more reliable.
- A full detector-based version (e.g. OWLv2, DETIC, YOLOv8) would be more
  accurate but requires additional heavyweight dependencies (deferred to the
  OWLv2/VQA reinforcement stage).
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
    uncertain_band:
        Optional hysteresis band width (default 0.0 = off, legacy behaviour).
        When > 0, an object detected in the original is still counted as
        preserved while its reconstruction score is >= ``presence_threshold -
        uncertain_band`` (see module docstring).
    device:
        Compute device.  Ignored when *clip_evaluator* is provided.
    """

    def __init__(
        self,
        clip_evaluator=None,
        vocabulary: Optional[List[str]] = None,
        presence_threshold: float = 0.25,
        uncertain_band: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self._clip = clip_evaluator
        self.vocabulary = vocabulary or _COCO_CLASSES
        self.presence_threshold = presence_threshold
        self.uncertain_band = max(float(uncertain_band), 0.0)
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

    def _object_similarities(self, image: torch.Tensor) -> List[float]:
        """Per-vocabulary CLIP similarity scores for *image* (``[1,3,H,W]``)."""
        clip_eval = self._get_clip()
        img_feat = clip_eval._encode_images(image)          # [1, D]
        txt_feat = self._get_text_features()                # [V, D]
        sims = (img_feat @ txt_feat.T).squeeze(0)           # [V]
        return sims.tolist()

    def _detect_objects(
        self, image: torch.Tensor, threshold: Optional[float] = None,
    ) -> List[str]:
        """Return list of vocabulary items detected in *image*.

        Parameters
        ----------
        image:
            ``[1, 3, H, W]`` float tensor in [0, 1].
        threshold:
            Optional override of ``self.presence_threshold`` (used by the
            hysteresis logic; callers such as ``HallucinationEvaluator`` may
            also pass a stricter threshold).
        """
        thr = self.presence_threshold if threshold is None else float(threshold)
        sims = self._object_similarities(image)
        return [self.vocabulary[i] for i, s in enumerate(sims) if s >= thr]

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

        thr = self.presence_threshold
        band = self.uncertain_band
        vocab = self.vocabulary
        for i in range(n):
            orig_sims  = self._object_similarities(original[i:i+1])
            recon_sims = self._object_similarities(reconstructed[i:i+1])
            orig_objs  = {v for v, s in zip(vocab, orig_sims) if s >= thr}
            recon_objs = {v for v, s in zip(vocab, recon_sims) if s >= thr}

            # Hysteresis: objects confirmed in the original stay "preserved"
            # while their recon score is inside the uncertain band below the
            # threshold.  With band == 0 this is exactly recon_objs (legacy).
            recon_keep = recon_objs if band == 0.0 else {
                v for v, s in zip(vocab, recon_sims) if s >= thr - band
            }

            matched = orig_objs & recon_keep
            missing = orig_objs - recon_keep

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
