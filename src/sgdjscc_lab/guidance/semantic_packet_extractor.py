"""guidance/semantic_packet_extractor.py – Unified semantic packet builder (Phase 4-A).

A *semantic packet* bundles every channel-independent description of a frame into
one dict that can be serialised to JSON, compared against a reconstructed-frame
packet, and (in later phases) coded for transmission.  Fields:

    caption              str            – BLIP2 caption (coarse intent)
    scene                str            – coarse scene label (indoor/outdoor/…)
    objects              list[str]      – detected object categories
    relations            list[triplet]  – (subject, predicate, object) triplets
    attributes           dict[obj→list] – colour / material / size adjectives
    edge_summary         dict           – soft-edge density / mean / std
    segmentation_summary dict           – per-class pixel fractions + dominant class
    depth_summary        dict           – normalised depth statistics
    importance           dict           – per-object importance + transmission order
    meta                 dict           – version, frame id, source flags

The class is deliberately decoupled from heavy models: every sub-extractor is
injected and optional.  ``build_packet`` is a pure function that assembles a
packet from already-extracted pieces, which keeps the whole packet layer
unit-testable without CLIP / BLIP2 / SegFormer weights.

Phase 4-A only *serialises* the packet (see ``utils/packet_io.py``); it is not
transmitted over the channel yet.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import torch

from sgdjscc_lab.guidance.importance_estimator import ImportanceEstimator
from sgdjscc_lab.guidance.object_extractor import ObjectExtractor
from sgdjscc_lab.guidance.relation_extractor import RelationExtractor
from sgdjscc_lab.utils.packet_io import PACKET_VERSION

logger = logging.getLogger(__name__)

# Coarse scene vocabulary for CLIP scene probing.
_SCENE_LABELS: List[str] = [
    "indoor scene", "outdoor scene", "street scene", "kitchen", "bedroom",
    "living room", "office", "beach", "forest", "mountain", "city",
    "countryside", "sky", "underwater", "sports field",
]

# Small adjective vocabularies used to attach attributes to objects.
_ADJECTIVES: Dict[str, List[str]] = {
    "color": [
        "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown",
        "black", "white", "gray", "grey", "silver", "gold",
    ],
    "material": [
        "wooden", "metal", "metallic", "plastic", "glass", "leather", "stone",
        "concrete", "fabric", "paper",
    ],
    "size": ["small", "large", "big", "tiny", "huge", "tall", "short", "long"],
}


# ── Summary helpers (pure, tensor → dict) ─────────────────────────────────────

def summarize_edge(edge: torch.Tensor, threshold: float = 0.1) -> Dict:
    """Summarise a soft-edge map ``[*, H, W]`` into density / mean / std."""
    e = edge.float()
    return {
        "density": float((e > threshold).float().mean().item()),
        "mean": float(e.mean().item()),
        "std": float(e.std().item()),
    }


def summarize_segmentation(label_map: torch.Tensor, class_names: List[str]) -> Dict:
    """Summarise a segmentation label map ``[*, H, W]`` into per-class fractions."""
    flat = label_map.reshape(-1).long()
    total = max(flat.numel(), 1)
    hist: Dict[str, float] = {}
    ids, counts = torch.unique(flat, return_counts=True)
    for cid, cnt in zip(ids.tolist(), counts.tolist()):
        name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        hist[name] = float(cnt / total)
    dominant = max(hist, key=hist.get) if hist else None
    return {
        "class_histogram": hist,
        "dominant_class": dominant,
        "num_regions": len(hist),
    }


def summarize_depth(depth: torch.Tensor) -> Dict:
    """Summarise a depth map ``[*, 1, H, W]`` into normalised statistics."""
    d = depth.float()
    dmin, dmax = float(d.min().item()), float(d.max().item())
    rng = (dmax - dmin) or 1.0
    norm = (d - dmin) / rng
    return {
        "min": dmin,
        "max": dmax,
        "mean": float(d.mean().item()),
        "std": float(d.std().item()),
        # Fraction of pixels in the nearest depth quartile (foreground share).
        "near_fraction": float((norm > 0.75).float().mean().item()),
    }


def extract_attributes(caption: str, objects: List[str]) -> Dict[str, List[str]]:
    """Attach colour/material/size adjectives to objects mentioned in a caption.

    For each object, adjectives are collected from a short window of words that
    immediately precede the object mention.
    """
    if not caption or not objects:
        return {}
    words = re.findall(r"[a-z]+", caption.lower())
    all_adj = {a for group in _ADJECTIVES.values() for a in group}
    attrs: Dict[str, List[str]] = {}
    for obj in objects:
        head = obj.lower().split()[-1]  # match on the head noun
        if head not in words:
            continue
        idx = words.index(head)
        window = words[max(0, idx - 3):idx]
        found = [w for w in window if w in all_adj]
        if found:
            attrs[obj] = found
    return attrs


def build_packet(
    caption: Optional[str] = None,
    objects: Optional[List[str]] = None,
    scene: Optional[str] = None,
    relations: Optional[List[Dict]] = None,
    attributes: Optional[Dict[str, List[str]]] = None,
    edge_summary: Optional[Dict] = None,
    segmentation_summary: Optional[Dict] = None,
    depth_summary: Optional[Dict] = None,
    frame_id: Optional[str] = None,
    importance_estimator: Optional[ImportanceEstimator] = None,
    meta: Optional[Dict] = None,
) -> Dict:
    """Assemble a semantic packet dict from already-extracted pieces (pure).

    Any field may be omitted; missing fields default to empty / None.  When an
    ``importance_estimator`` is provided (or by default), per-object importance
    and transmission order are computed from the assembled packet.
    """
    packet: Dict = {
        "caption": caption or "",
        "scene": scene,
        "objects": sorted(set(objects or [])),
        "relations": relations or [],
        "attributes": attributes or {},
        "edge_summary": edge_summary,
        "segmentation_summary": segmentation_summary,
        "depth_summary": depth_summary,
        "meta": {
            "version": PACKET_VERSION,
            "frame_id": frame_id,
            **(meta or {}),
        },
    }
    est = importance_estimator or ImportanceEstimator()
    packet["importance"] = est.estimate(packet)
    return packet


class SemanticPacketExtractor:
    """Build semantic packets from images using injected sub-extractors.

    All sub-extractors are optional; whichever are supplied contribute their
    field, and the rest default to empty.  This lets callers build a
    caption-only packet (cheap) or a full caption+object+structure packet.

    Parameters
    ----------
    text_extractor:
        ``TextExtractor`` (BLIP2) producing the caption.
    object_extractor:
        ``ObjectExtractor``.  If None, one is created lazily from
        ``clip_evaluator`` (so object detection still works) — and a caption-only
        fallback is used when no CLIP evaluator is available.
    relation_extractor:
        ``RelationExtractor``.  Defaults to a fresh instance.
    clip_evaluator:
        Shared ``CLIPScoreEvaluator`` used for object detection and scene probing.
    edge_extractor / segmentation_extractor / depth_extractor:
        Optional structural extractors (MuGE / SegFormer / DPT).
    device:
        Compute device for internally created components.
    """

    def __init__(
        self,
        text_extractor=None,
        object_extractor: Optional[ObjectExtractor] = None,
        relation_extractor: Optional[RelationExtractor] = None,
        clip_evaluator=None,
        edge_extractor=None,
        segmentation_extractor=None,
        depth_extractor=None,
        importance_estimator: Optional[ImportanceEstimator] = None,
        device: Optional[torch.device] = None,
        caption_objects: bool = True,
    ) -> None:
        self.text_extractor = text_extractor
        self._clip = clip_evaluator
        self.object_extractor = object_extractor
        self.relation_extractor = relation_extractor or RelationExtractor()
        self.edge_extractor = edge_extractor
        self.segmentation_extractor = segmentation_extractor
        self.depth_extractor = depth_extractor
        self.importance_estimator = importance_estimator or ImportanceEstimator()
        self._device = device or torch.device("cpu")
        # Fold open-vocabulary caption nouns into the object list so packets are
        # not empty for categories outside COCO-80 (e.g. "mushroom").
        self.caption_objects = caption_objects

    def _get_object_extractor(self) -> ObjectExtractor:
        if self.object_extractor is None:
            self.object_extractor = ObjectExtractor(
                clip_evaluator=self._clip, device=self._device
            )
        return self.object_extractor

    def _probe_scene(self, image: torch.Tensor) -> Optional[str]:
        """Pick the best-matching coarse scene label via CLIP (or None)."""
        if self._clip is None:
            return None
        try:
            img_feat = self._clip._encode_images(image)            # [1, D]
            txt_feat = self._clip._encode_texts(_SCENE_LABELS)     # [L, D]
            sims = (img_feat @ txt_feat.T).squeeze(0)              # [L]
            return _SCENE_LABELS[int(sims.argmax().item())]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scene probing failed: %s", exc)
            return None

    def extract(
        self,
        image: torch.Tensor,
        frame_id: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> Dict:
        """Build a semantic packet for *image* (``[1, 3, H, W]`` in [0, 1]).

        Parameters
        ----------
        image:
            Single-image tensor.  Structural extractors are applied directly to it.
        frame_id:
            Optional identifier stored in ``packet["meta"]["frame_id"]``.
        caption:
            Optional pre-computed caption; if None and a text extractor is set, the
            caption is generated.
        """
        # ── Caption ──────────────────────────────────────────────────────────
        if caption is None and self.text_extractor is not None:
            try:
                out = self.text_extractor.extract(image, self._device)
                # TextExtractor returns list-of-lists, e.g. [["a cat"]].
                caption = out[0][0] if out and out[0] else ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("Caption extraction failed: %s", exc)
                caption = ""
        caption = caption or ""

        # ── Objects ──────────────────────────────────────────────────────────
        # Combine CLIP/COCO detections with open-vocabulary caption nouns so the
        # object layer is populated even for categories outside COCO-80 (e.g.
        # "mushroom").  ``include_caption_nouns`` keeps it config-toggleable.
        obj_ext = self._get_object_extractor()
        try:
            objects = obj_ext.extract_objects(
                caption=caption,
                image=image if (self._clip is not None or obj_ext._clip is not None) else None,
                include_caption_nouns=self.caption_objects,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Object extraction failed: %s", exc)
            objects = obj_ext.from_caption(caption)

        # ── Relations & attributes (caption-derived) ─────────────────────────
        relations = self.relation_extractor.extract(caption, objects)
        attributes = extract_attributes(caption, objects)

        # ── Scene ────────────────────────────────────────────────────────────
        scene = self._probe_scene(image)

        # ── Structural summaries (optional) ──────────────────────────────────
        edge_summary = None
        if self.edge_extractor is not None:
            try:
                data, _unc = self.edge_extractor.extract(image, self._device)
                edge_summary = summarize_edge(torch.mean(data.float(), dim=1))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Edge summary failed: %s", exc)

        seg_summary = None
        if self.segmentation_extractor is not None:
            try:
                seg = self.segmentation_extractor.extract(image, self._device)
                seg_summary = summarize_segmentation(seg["label_map"], seg["class_names"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Segmentation summary failed: %s", exc)

        depth_summary = None
        if self.depth_extractor is not None:
            try:
                depth = self.depth_extractor.extract(image, self._device)
                depth_summary = summarize_depth(depth)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Depth summary failed: %s", exc)

        return build_packet(
            caption=caption,
            objects=objects,
            scene=scene,
            relations=relations,
            attributes=attributes,
            edge_summary=edge_summary,
            segmentation_summary=seg_summary,
            depth_summary=depth_summary,
            frame_id=frame_id,
            importance_estimator=self.importance_estimator,
        )
