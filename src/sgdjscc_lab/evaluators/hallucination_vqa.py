"""evaluators/hallucination_vqa.py – VQA-style hallucination check (Phase 5-C).

A stronger alternative to the pure CLIP-probing hallucination heuristic: ask a
yes/no visual question ("Is there a {object} in the image?") about the
reconstructed image and the original, and flag objects that the VQA model
confirms in the reconstruction but denies in the original.

The VQA model is **injected** as ``vqa_fn(image, question) -> str`` so the
evaluator is testable with a mock and does not hard-depend on a heavy VQA model.
When no ``vqa_fn`` is supplied it falls back to the Phase-3 CLIP
``HallucinationEvaluator`` and reports ``method="clip_fallback"`` — so the
interface and evaluation path are complete even without VQA weights.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def _is_yes(answer: str) -> bool:
    return str(answer).strip().lower().startswith(("y", "true", "1"))


class VQAHallucinationEvaluator:
    """Yes/no VQA hallucination detector with a CLIP fallback.

    Parameters
    ----------
    vqa_fn:
        ``(image[1,3,H,W], question:str) -> answer:str``.  When None, the CLIP
        heuristic ``HallucinationEvaluator`` is used instead.
    clip_evaluator:
        Shared CLIP evaluator for the fallback / object detection paths.
    question_template:
        Template for the yes/no question (``{obj}`` is substituted).
    """

    def __init__(
        self,
        vqa_fn: Optional[Callable] = None,
        clip_evaluator=None,
        question_template: str = "Is there a {obj} in the image?",
    ) -> None:
        self.vqa_fn = vqa_fn
        self._clip = clip_evaluator
        self.question_template = question_template
        self._fallback = None

    @classmethod
    def from_config(cls, cfg=None, clip_evaluator=None, vqa_backend_cfg=None):
        """Build an evaluator, constructing a real local VQA backend from config.

        ``vqa_backend_cfg`` (or ``cfg.vqa_backend``) selects the backend
        (``mock`` / ``blip2`` / ``llava`` / ``mplug``).  If the backend is
        unavailable, ``vqa_fn`` is None and the evaluator transparently uses the
        CLIP-heuristic fallback.
        """
        from sgdjscc_lab.evaluators.vqa_backend import build_vqa_backend
        backend_cfg = vqa_backend_cfg
        if backend_cfg is None and cfg is not None:
            try:
                backend_cfg = cfg.get("vqa_backend", None)
            except AttributeError:
                backend_cfg = getattr(cfg, "vqa_backend", None)
        vqa_fn = build_vqa_backend(backend_cfg) if backend_cfg is not None else None
        return cls(vqa_fn=vqa_fn, clip_evaluator=clip_evaluator)

    def _get_fallback(self):
        if self._fallback is None:
            from sgdjscc_lab.evaluators.hallucination import HallucinationEvaluator
            self._fallback = HallucinationEvaluator(clip_evaluator=self._clip)
        return self._fallback

    def evaluate(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        objects: Optional[List[str]] = None,
    ) -> Dict:
        """Compute a VQA-based hallucination score.

        Parameters
        ----------
        original, reconstructed:
            ``[1, 3, H, W]`` tensors in [0, 1].
        objects:
            Candidate object names to probe.  Required for the VQA path; for the
            CLIP fallback they are detected automatically.

        Returns
        -------
        dict with ``hallucination_score`` (in [0, 1], lower is better),
        ``hallucinated_objects`` and ``method``.
        """
        if self.vqa_fn is None:
            res = self._get_fallback().evaluate(original, reconstructed)
            res["method"] = "clip_fallback"
            res.setdefault("hallucinated_objects", res.get("extra_objects", []))
            return res

        if not objects:
            return {"hallucination_score": 0.0, "hallucinated_objects": [],
                    "method": "vqa", "notes": "no candidate objects"}

        try:
            hallucinated: List[str] = []
            n_in_original = 0
            for obj in objects:
                q = self.question_template.format(obj=obj)
                in_recon = _is_yes(self.vqa_fn(reconstructed, q))
                in_orig = _is_yes(self.vqa_fn(original, q))
                if in_orig:
                    n_in_original += 1
                if in_recon and not in_orig:
                    hallucinated.append(obj)
        except Exception as exc:  # noqa: BLE001 – backend failed at runtime
            # Disable the backend for the rest of this run so we don't retry the
            # (heavy, failing) load on every object/question. Log the reason once;
            # subsequent calls take the clip_fallback branch above silently.
            logger.warning(
                "VQA backend failed (%s); disabling VQA and using CLIP fallback "
                "for the remainder of this run.", exc)
            self.vqa_fn = None
            res = self._get_fallback().evaluate(original, reconstructed)
            res["method"] = "vqa_error_fallback"
            res.setdefault("hallucinated_objects", res.get("extra_objects", []))
            return res

        score = len(hallucinated) / max(n_in_original, 1)
        return {
            "hallucination_score": float(score),
            "hallucinated_objects": sorted(hallucinated),
            "method": "vqa",
        }
