"""evaluators/semantic_reliability.py – Semantic Reliability Score (SRS).

SRS is the headline research metric for this project.  It aggregates multiple
semantic signals into a single score that reflects how faithfully the
*transmitted intent* survives wireless channel corruption and generative
reconstruction.

Formula
-------
SRS = w_img  * clip_image_image
    + w_txt  * clip_text_image
    + w_pres * object_preservation_rate
    - w_miss * missing_object_rate
    - w_add  * additional_object_rate

Default weights (configurable via constructor or config dict):
  w_img  = 0.30
  w_txt  = 0.25
  w_pres = 0.25
  w_miss = 0.10
  w_add  = 0.10

Range
-----
SRS is in (−∞, 1].  Typical values for high-quality reconstruction: 0.70–0.90.
Heavily corrupted or hallucinated images may produce negative SRS values.

Usage
-----
>>> evaluator = SemanticReliabilityEvaluator(device=torch.device("cuda:0"))
>>> result = evaluator.evaluate(original, reconstructed, text_list=["a cat"])
>>> print(result["semantic_reliability_score"])   # e.g. 0.812
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Default SRS weights
_DEFAULT_WEIGHTS = {
    "w_img":  0.30,
    "w_txt":  0.25,
    "w_pres": 0.25,
    "w_miss": 0.10,
    "w_add":  0.10,
}

# Phase 4-A packet-aware composite weights.  These score the packet-matcher error
# report (see evaluators/semantic_packet_matcher.py); ``segmentation`` falls back
# to 1.0 when no segmentation summary is present in the packets.
_DEFAULT_PACKET_WEIGHTS = {
    "w_obj":  0.40,   # object match rate
    "w_rel":  0.20,   # relation consistency
    "w_attr": 0.20,   # attribute consistency
    "w_seg":  0.10,   # segmentation consistency
    "w_scene": 0.10,  # scene match
    "w_add_penalty": 0.10,   # penalty per (normalised) additional/hallucinated object
}


class SemanticReliabilityEvaluator:
    """Composite semantic reliability evaluator.

    Combines CLIP image-image similarity, CLIP text-image similarity, object
    preservation rate, missing object rate, and additional (hallucinated) object
    rate into the Semantic Reliability Score (SRS).

    Parameters
    ----------
    clip_evaluator:
        Shared ``CLIPScoreEvaluator`` instance.  Created lazily if None.
    obj_pres_evaluator:
        Shared ``ObjectPreservationEvaluator`` instance.  Created lazily if None.
    hallucination_evaluator:
        Shared ``HallucinationEvaluator`` instance.  Created lazily if None.
    weights:
        Dict with keys ``w_img``, ``w_txt``, ``w_pres``, ``w_miss``, ``w_add``.
        Missing keys default to ``_DEFAULT_WEIGHTS``.
    device:
        Compute device (used when sub-evaluators are created internally).
    """

    def __init__(
        self,
        clip_evaluator=None,
        obj_pres_evaluator=None,
        hallucination_evaluator=None,
        weights: Optional[Dict[str, float]] = None,
        packet_weights: Optional[Dict[str, float]] = None,
        packet_blend: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        self._clip = clip_evaluator
        self._obj_pres = obj_pres_evaluator
        self._hall = hallucination_evaluator
        self._device = device or torch.device("cpu")

        w = dict(_DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)
        self.weights = w

        pw = dict(_DEFAULT_PACKET_WEIGHTS)
        if packet_weights:
            pw.update(packet_weights)
        self.packet_weights = pw
        # Fraction of srs_packet contributed by packet terms vs srs_base.
        self.packet_blend = float(packet_blend)

    # ── Lazy sub-evaluator builders ──────────────────────────────────────────

    def _get_clip(self):
        if self._clip is None:
            from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
            self._clip = CLIPScoreEvaluator(device=self._device)
        return self._clip

    def _get_obj_pres(self):
        if self._obj_pres is None:
            from sgdjscc_lab.evaluators.object_preservation import ObjectPreservationEvaluator
            self._obj_pres = ObjectPreservationEvaluator(
                clip_evaluator=self._get_clip(), device=self._device
            )
        return self._obj_pres

    def _get_hall(self):
        if self._hall is None:
            from sgdjscc_lab.evaluators.hallucination import HallucinationEvaluator
            self._hall = HallucinationEvaluator(
                clip_evaluator=self._get_clip(), device=self._device
            )
        return self._hall

    # ── Main evaluation ──────────────────────────────────────────────────────

    def evaluate(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        text_list: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        orig_packet: Optional[Dict] = None,
        recon_packet: Optional[Dict] = None,
    ) -> Dict:
        """Compute the Semantic Reliability Score and sub-metrics.

        Parameters
        ----------
        original, reconstructed:
            ``[N, 3, H, W]`` float tensors in [0, 1].
        text_list:
            Optional list of N text prompts / captions.  When provided,
            ``clip_text_image`` is computed; otherwise it is set to ``None``.
        metadata:
            Reserved for future use (e.g. GT segmentation annotations).

        Returns
        -------
        dict with keys:
            ``semantic_reliability_score`` – SRS scalar float.
            ``clip_image_image``           – CLIP image-image cosine similarity.
            ``clip_text_image``            – CLIP text-image similarity (or None).
            ``object_preservation_rate``   – fraction of original objects preserved.
            ``missing_object_rate``        – fraction of objects lost.
            ``additional_object_rate``     – fraction of hallucinated objects added.
            ``weights``                    – weights used in this computation.
        """
        w = self.weights

        # ── CLIP image-image ────────────────────────────────────────────────
        clip_img = self._get_clip().image_image_score(original, reconstructed)

        # ── CLIP text-image ─────────────────────────────────────────────────
        clip_txt: Optional[float] = None
        if text_list is not None:
            clip_txt = self._get_clip().text_image_score(text_list, reconstructed)

        # ── Object preservation / hallucination ─────────────────────────────
        pres_result = self._get_obj_pres().evaluate(original, reconstructed)
        hall_result = self._get_hall().evaluate(original, reconstructed)

        pres_rate = pres_result["preservation_rate"]
        orig_count = pres_result["original_count"]

        # missing_rate = (1 - preservation_rate) when objects exist
        missing_rate = 1.0 - pres_rate if orig_count > 0 else 0.0

        # additional_rate = hallucination_score (already normalised by orig count)
        add_rate = hall_result["hallucination_score"]

        # ── SRS composite ───────────────────────────────────────────────────
        # When text is not available, redistribute w_txt to w_img
        if clip_txt is None:
            effective_w_img = w["w_img"] + w["w_txt"]
            txt_term = 0.0
        else:
            effective_w_img = w["w_img"]
            txt_term = w["w_txt"] * clip_txt

        srs = (
            effective_w_img * clip_img
            + txt_term
            + w["w_pres"] * pres_rate
            - w["w_miss"] * missing_rate
            - w["w_add"]  * add_rate
        )

        result = {
            "semantic_reliability_score": float(srs),
            # srs_base is an explicit alias for the Phase-3 SRS so packet-aware
            # rows can report base vs packet side by side.
            "srs_base":                  float(srs),
            "clip_image_image":          clip_img,
            "clip_text_image":           clip_txt,
            "object_preservation_rate":  pres_rate,
            "missing_object_rate":       missing_rate,
            "additional_object_rate":    add_rate,
            "hallucination_score":       hall_result["hallucination_score"],
            "weights":                   dict(w),
        }

        # ── Packet-aware extension (Phase 4-A) ───────────────────────────────
        if orig_packet is not None and recon_packet is not None:
            result.update(
                self.score_packet(orig_packet, recon_packet, srs_base=float(srs))
            )

        return result

    # ── Packet-aware scoring (Phase 4-A) ─────────────────────────────────────

    def score_packet(
        self,
        orig_packet: Dict,
        recon_packet: Dict,
        srs_base: Optional[float] = None,
    ) -> Dict:
        """Compute ``srs_packet`` and packet error terms from two packets.

        ``srs_packet`` blends the base SRS with a packet-consistency composite
        (object / relation / attribute / segmentation / scene), so it rewards
        reconstructions that preserve fine-grained semantic structure beyond what
        the CLIP/object heuristics capture.

        Parameters
        ----------
        orig_packet, recon_packet:
            Semantic packet dicts.
        srs_base:
            The base SRS to blend with.  When None it is not recomputed; the
            packet composite is reported on its own and used as ``srs_packet``.

        Returns
        -------
        dict with ``srs_packet`` plus the full packet-matcher error report and the
        three consistency terms.
        """
        from sgdjscc_lab.evaluators.semantic_packet_matcher import compare

        report = compare(orig_packet, recon_packet)
        pw = self.packet_weights

        obj_rate = float(report["object_match_rate"])
        rel_cons = float(report["relation_consistency"])
        attr_cons = float(report["attribute_consistency"])
        seg_cons = report.get("segmentation_consistency")
        seg_term = 1.0 if seg_cons is None else float(seg_cons)
        scene_term = 1.0 if report["scene_match"] else 0.0

        # Normalise additional-object penalty by original object count.
        n_orig = max(len(orig_packet.get("objects") or []), 1)
        add_pen = report["additional_object_count"] / n_orig

        packet_composite = (
            pw["w_obj"] * obj_rate
            + pw["w_rel"] * rel_cons
            + pw["w_attr"] * attr_cons
            + pw["w_seg"] * seg_term
            + pw["w_scene"] * scene_term
            - pw["w_add_penalty"] * add_pen
        )

        if srs_base is None:
            srs_packet = packet_composite
        else:
            b = self.packet_blend
            srs_packet = (1.0 - b) * float(srs_base) + b * packet_composite

        out = {
            "srs_packet":              float(srs_packet),
            "packet_composite":        float(packet_composite),
            "relation_consistency":    rel_cons,
            "attribute_consistency":   attr_cons,
            "segmentation_consistency": seg_cons,
            "object_match_rate":       obj_rate,
            "scene_match":             bool(report["scene_match"]),
            "missing_object_count":    report["missing_object_count"],
            "additional_object_count": report["additional_object_count"],
            "relation_error_count":    report["relation_error_count"],
            "attribute_error_count":   report["attribute_error_count"],
            "error_report":            report,
        }
        return out
