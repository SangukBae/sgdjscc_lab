"""guidance/object_extractor.py – Object-list extractor for semantic packets (Phase 4-A).

Produces the ``objects`` field of a semantic packet: the set of recognisable
object categories present in an image.  Two detection paths are supported and can
be combined:

1. **CLIP zero-shot probing** – reuses the same COCO-80 text-probing heuristic as
   ``evaluators/object_preservation.py`` (an object is "present" when its CLIP
   text-image similarity exceeds ``presence_threshold``).  This path needs a
   ``CLIPScoreEvaluator``.
2. **Caption keyword matching** – ``from_caption`` scans a BLIP2 caption for
   vocabulary words.  This path is pure-Python (no model), which keeps the packet
   pipeline usable — and unit-testable — without CLIP weights.

The extractor deliberately mirrors ``ObjectPreservationEvaluator`` so the packet
object list and the SRS object-preservation metric stay consistent.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import torch

from sgdjscc_lab.evaluators.object_preservation import _COCO_CLASSES

logger = logging.getLogger(__name__)

# Words filtered out of open-vocabulary caption-noun extraction: function words,
# prepositions, common verbs/be-verbs, photographic boilerplate, and the
# colour/material/size adjectives used elsewhere in the packet.  Anything left is
# treated as a candidate object noun.
_FUNCTION_WORDS = {
    "the", "and", "with", "for", "are", "was", "were", "has", "have", "had",
    "that", "this", "these", "those", "its", "his", "her", "their", "them",
    "they", "she", "him", "you", "your", "our", "out", "into", "onto", "over",
    "under", "near", "next", "front", "back", "side", "from", "off", "but",
    "not", "all", "some", "any", "more", "most", "very", "around", "between",
    "above", "below", "behind", "beside",
}
_BOILERPLATE_WORDS = {
    "close", "shot", "view", "photo", "image", "picture", "background",
    "foreground", "scene", "looking", "showing", "sitting", "standing",
    "lying", "holding", "wearing", "smiling", "facing",
}
_ADJECTIVE_WORDS = {
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "silver", "gold", "wooden", "metal",
    "metallic", "plastic", "glass", "leather", "stone", "concrete", "fabric",
    "paper", "small", "large", "big", "tiny", "huge", "tall", "short", "long",
}
_NON_OBJECT_WORDS = _FUNCTION_WORDS | _BOILERPLATE_WORDS | _ADJECTIVE_WORDS


class ObjectExtractor:
    """Extract a list of object categories from an image or caption.

    Parameters
    ----------
    clip_evaluator:
        Shared ``CLIPScoreEvaluator`` instance.  Required only for the
        image-based :meth:`extract` path; ``from_caption`` works without it.
    vocabulary:
        Candidate object category names.  Defaults to the COCO-80 vocabulary.
    presence_threshold:
        CLIP similarity threshold above which an object is considered present.
    device:
        Compute device used when a CLIP evaluator is created internally.
    """

    def __init__(
        self,
        clip_evaluator=None,
        vocabulary: Optional[List[str]] = None,
        presence_threshold: float = 0.25,
        device: Optional[torch.device] = None,
    ) -> None:
        self._clip = clip_evaluator
        self.vocabulary = vocabulary or list(_COCO_CLASSES)
        self.presence_threshold = presence_threshold
        self._device = device or torch.device("cpu")
        self._obj_pres = None

    def _get_obj_pres(self):
        """Reuse ObjectPreservationEvaluator's detector for CLIP probing."""
        if self._obj_pres is None:
            from sgdjscc_lab.evaluators.object_preservation import ObjectPreservationEvaluator
            self._obj_pres = ObjectPreservationEvaluator(
                clip_evaluator=self._clip,
                vocabulary=self.vocabulary,
                presence_threshold=self.presence_threshold,
                device=self._device,
            )
        return self._obj_pres

    def extract(self, image: torch.Tensor) -> List[str]:
        """Detect objects in *image* (``[1, 3, H, W]`` float in [0, 1]) via CLIP.

        Returns a sorted, de-duplicated list of detected category names.
        """
        detected = self._get_obj_pres()._detect_objects(image)
        return sorted(set(detected))

    def from_caption(self, caption: str) -> List[str]:
        """Detect vocabulary objects mentioned in a caption string (no model).

        Matching is whitespace/word-boundary aware so ``"car"`` does not match
        inside ``"scarf"``.  Multi-word categories (e.g. ``"traffic light"``) are
        matched as substrings.
        """
        if not caption:
            return []
        text = caption.lower()
        found: List[str] = []
        for obj in self.vocabulary:
            if " " in obj:
                if obj in text:
                    found.append(obj)
            elif re.search(rf"\b{re.escape(obj)}s?\b", text):
                found.append(obj)
        return sorted(set(found))

    def nouns_from_caption(self, caption: str) -> List[str]:
        """Extract open-vocabulary candidate object nouns from a caption.

        Unlike :meth:`from_caption` (which is limited to ``self.vocabulary`` and so
        misses words like ``"mushroom"`` that are not in COCO-80), this keeps every
        content word after removing function words, photographic boilerplate,
        common verbs, and the colour/material/size adjectives.  It is a lightweight,
        deterministic, dependency-free noun proxy (no POS tagger), so the original
        and reconstructed packets are extracted identically and their object diff is
        meaningful.  Plurals are singularised crudely (trailing ``s``).
        """
        if not caption:
            return []
        words = re.findall(r"[a-z]+", caption.lower())
        nouns: List[str] = []
        for w in words:
            if len(w) <= 2 or w in _NON_OBJECT_WORDS:
                continue
            # crude singularisation so "cats" and "cat" collapse
            if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
                w = w[:-1]
            nouns.append(w)
        return sorted(set(nouns))

    def extract_objects(self, caption: str = "", image: Optional[torch.Tensor] = None,
                        include_caption_nouns: bool = True) -> List[str]:
        """Combined object list: CLIP/vocab detections ∪ open-vocab caption nouns.

        Uses the CLIP detector when an *image* and a CLIP evaluator are available,
        always folds in the vocabulary words found in the caption, and (by default)
        adds open-vocabulary caption nouns so packets are not empty for objects
        outside COCO-80.
        """
        objs: set = set()
        if image is not None and (self._clip is not None or self.vocabulary):
            try:
                if self._clip is not None:
                    objs |= set(self.extract(image))
            except Exception:  # noqa: BLE001
                pass
        objs |= set(self.from_caption(caption))
        if include_caption_nouns:
            objs |= set(self.nouns_from_caption(caption))
        return sorted(objs)
