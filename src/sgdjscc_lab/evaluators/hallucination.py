"""evaluators/hallucination.py – Hallucination detection evaluator.

Approach (Phase 3 – CLIP text-probing heuristic)
-------------------------------------------------
Hallucination here means objects/attributes that appear in the *reconstructed*
image but were absent in the *original*.  This is the inverse of the
object-preservation problem.

extra_objects = {objects detected in reconstructed} − {objects detected in original}
hallucination_score = |extra_objects| / max(|objects in original|, 1)

The score is normalised by the number of original objects so that a
heavily-cluttered original does not exaggerate a few added artefacts.

Why this matters
----------------
Generative reconstruction (diffusion models) can hallucinate objects that
look plausible but were never in the original scene.  A visually sharp image
with hallucinated content may score well on PSNR/SSIM but fail semantic
reliability.

Limitations
-----------
Same CLIP-probing caveats as ``object_preservation.py``:
- Threshold-sensitive; absolute values require dataset-specific calibration.
- A POPE-style VQA model would provide more rigorous yes/no per-object evidence.
- Phase 3 delivers a research-grade first-pass metric, not a benchmark-certified
  hallucination detector.

PROVISIONAL IMPLEMENTATION NOTE (ETRI plan step 0 / 슬라이드 6·7)
----------------------------------------------------------------
Like ``object_preservation.py``, the CLIP presence probe here is an interim
judge; per docs/etri_strategy.md (5차 단계) it will be reinforced by OWLv2
grounded detection and VQA verification behind the same evaluator interface,
after which hallucination numbers must be re-measured.  The optional
``uncertain_band`` (default 0.0 = legacy behaviour) requires a reconstruction
score of at least ``presence_threshold + uncertain_band`` before flagging an
object as hallucinated, so borderline CLIP scores do not inflate the rate.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class HallucinationEvaluator:
    """Detect objects added by generative reconstruction that were not in the original.

    Parameters
    ----------
    clip_evaluator:
        A ``CLIPScoreEvaluator`` instance shared with other evaluators (saves VRAM).
    vocabulary:
        Object category names to probe.  Defaults to COCO 80 classes.
    presence_threshold:
        CLIP similarity threshold to decide object presence (default 0.25).
    uncertain_band:
        Optional hysteresis band (default 0.0 = legacy).  When > 0, an object
        only counts as hallucinated if its reconstruction score is at least
        ``presence_threshold + uncertain_band`` (see module docstring).
    device:
        Compute device (used only if clip_evaluator is None).
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
        self._vocab = vocabulary
        self.presence_threshold = presence_threshold
        self.uncertain_band = max(float(uncertain_band), 0.0)
        self._device = device or torch.device("cpu")
        self._obj_eval = None

    def _get_obj_eval(self):
        if self._obj_eval is None:
            from sgdjscc_lab.evaluators.object_preservation import ObjectPreservationEvaluator
            self._obj_eval = ObjectPreservationEvaluator(
                clip_evaluator=self._clip,
                vocabulary=self._vocab,
                presence_threshold=self.presence_threshold,
                uncertain_band=self.uncertain_band,
                device=self._device,
            )
        return self._obj_eval

    def evaluate(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Compute hallucination metrics.

        Parameters
        ----------
        original, reconstructed:
            ``[N, 3, H, W]`` float tensors in [0, 1].

        Returns
        -------
        dict with keys:
            ``hallucination_score`` – |extra_objects| / max(|original_objects|, 1),
                                      averaged over the batch (lower is better).
            ``extra_objects``       – objects in reconstructed not in original
                                      (list from the last sample in the batch).
            ``notes``               – human-readable interpretation string.
        """
        obj_eval = self._get_obj_eval()
        n = original.shape[0]
        scores: List[float] = []
        extra_last: List[str] = []

        # Hysteresis: with a band, flagging an object as hallucinated requires a
        # confidently-above-threshold recon score (thr + band); band 0 = legacy.
        extra_thr = self.presence_threshold + self.uncertain_band
        for i in range(n):
            orig_objs  = set(obj_eval._detect_objects(original[i:i+1]))
            recon_objs = set(obj_eval._detect_objects(reconstructed[i:i+1], threshold=extra_thr))

            extra = recon_objs - orig_objs
            score = len(extra) / max(len(orig_objs), 1)
            scores.append(score)
            extra_last = sorted(extra)

        mean_score = float(sum(scores) / max(len(scores), 1))

        if mean_score == 0.0:
            notes = "No hallucinated objects detected."
        elif mean_score < 0.1:
            notes = "Minor hallucination; semantics largely preserved."
        elif mean_score < 0.3:
            notes = "Moderate hallucination; some added objects not in original."
        else:
            notes = "Significant hallucination; reconstructed meaning may differ substantially."

        return {
            "hallucination_score": mean_score,
            "extra_objects":       extra_last,
            "notes":               notes,
        }
