"""evaluators/object_vocabulary_filter.py – Object-noun vocabulary filter for
held-out packet-metric remeasurement (ETRI 5차 remeasurement follow-up).

Problem this addresses
-----------------------
``guidance/semantic_packet_extractor.py``'s open-vocabulary caption-noun
folding (``ObjectExtractor.nouns_from_caption``) keeps every caption content
word that survives a short function-word/adjective/boilerplate denylist — it
has no notion of "object noun" vs. count modifier (``"one"``), action/event
word (``"walking"``), or scene/region word (``"sidewalk"``). Those non-object
tokens can end up in a packet's ``objects`` list and get counted as
missing/additional objects by ``evaluators/semantic_packet_matcher.compare()``,
which inflates or deflates ``PacketVerifier``'s ``severity``/
``object_match_rate`` — and the PTC/SFR/SDI temporal metrics derived from it —
without reflecting an actual object-presence error.

Scope: held-out remeasurement only
------------------------------------
This filter is wired into ``pipelines/heldout_remeasurement.py::remeasure()``
only (not into ``PacketVerifier`` or the live loop-internal path used by
``pipelines/packet_verification.py``/``controllers/verifier_controller.py``).
Two reasons:

1. ``remeasure()`` is the one place where the SAME filtered packet feeds both
   the "clip_only" and "calibrated" ``PacketVerifier`` calls *and* the
   ``temporal_consistency.evaluate_sequence()`` PTC/SFR/SDI calculation (which
   reads ``orig_packet``/``recon_packet`` directly, bypassing
   ``PacketVerifier`` entirely) — filtering only inside ``PacketVerifier``
   would leave PTC/SFR/SDI unfiltered, breaking the "fair comparison" this was
   built for.
2. The live loop-internal path is out of this task's scope; leaving it
   untouched avoids any risk of changing the accept/suppress/regenerate
   control-loop's numbers.

Disabled by default (``ObjectVocabularyFilter.enabled=False`` /
``verifier.object_vocabulary_filter.enabled: false``) — existing remeasurement
reports are byte-identical (no new report keys, no CSV column changes) until a
caller opts in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

# Count / quantity modifiers — never object nouns on their own.
_COUNT_TERMS: FrozenSet[str] = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "single", "couple", "few", "several",
    "many", "multiple", "dozen", "pair",
})

# Action / event words (usually gerunds or their bare verb form) — describe
# what's happening, not what object is present.
_ACTION_TERMS: FrozenSet[str] = frozenset({
    "walking", "standing", "running", "holding", "moving", "sitting",
    "jumping", "riding", "driving", "talking", "looking", "wearing",
    "smiling", "facing", "showing", "crossing", "waiting", "playing",
    "working", "eating", "drinking", "carrying", "pushing", "pulling",
    "throwing", "catching", "flying", "swimming", "dancing", "singing",
    "walk", "stand", "run", "hold", "move", "sit", "jump", "ride", "drive",
})

# Scene / region words — describe WHERE, not a discrete transmittable object.
_SCENE_TERMS: FrozenSet[str] = frozenset({
    "sidewalk", "street", "road", "room", "indoor", "outdoor", "background",
    "foreground", "scene", "area", "ground", "floor", "path", "pavement",
    "park", "yard", "field", "sky", "horizon", "city", "town",
    "neighborhood", "environment", "setting", "location", "place", "view",
})

DEFAULT_EXCLUDED_TERMS: FrozenSet[str] = _COUNT_TERMS | _ACTION_TERMS | _SCENE_TERMS

_NUMERIC_RE = re.compile(r"^\d+$")


def _is_excluded_term(term: str, excluded_terms: FrozenSet[str]) -> bool:
    t = str(term).strip().lower()
    if not t:
        return True
    if _NUMERIC_RE.match(t):
        return True
    if t in excluded_terms:
        return True
    # A multi-word phrase (e.g. "person walking") is excluded only if EVERY
    # word in it is itself excluded/numeric — a phrase containing a real
    # object noun ("walking person") must stay.
    words = t.split()
    if len(words) > 1 and all(_NUMERIC_RE.match(w) or w in excluded_terms for w in words):
        return True
    return False


@dataclass
class ObjectVocabularyFilter:
    """Filters non-object tokens (count/action/scene words) out of a packet's
    ``objects`` list before it reaches ``semantic_packet_matcher.compare()``.

    Parameters
    ----------
    enabled:
        Master switch. ``False`` (default) makes :meth:`filter_objects` a
        no-op passthrough (returns the input unchanged, nothing removed) —
        existing callers/configs are unaffected.
    excluded_terms:
        Term set checked case-insensitively against each object string when
        no GT vocabulary applies. Defaults to :data:`DEFAULT_EXCLUDED_TERMS`.
    use_gt_vocabulary:
        When ``True`` (default) and a non-empty *gt_metadata* is passed to
        :meth:`filter_objects`, the GT annotation's object labels (its dict
        keys — see ``pipelines/heldout_remeasurement.py::convert_gt_to_presence``,
        which only ever populates GT labels from segment ``objects[].label``,
        never ``scene``/``events``) become the vocabulary of record: an
        object is kept iff it case-insensitively matches a GT label, full
        stop — the excluded-terms heuristic is not consulted at all for that
        call. This is a deliberately closed-world simplification for a
        curated evaluation video whose GT enumerates every object category
        that can legitimately appear; it is not a general-purpose
        hallucination filter (an object absent from GT is treated as
        vocabulary noise, not flagged as an "additional/hallucinated"
        object, when this mode is active).
    """

    enabled: bool = False
    excluded_terms: FrozenSet[str] = field(default_factory=lambda: DEFAULT_EXCLUDED_TERMS)
    use_gt_vocabulary: bool = True

    def filter_objects(
        self, objects: Optional[List[str]], gt_metadata: Optional[Dict] = None,
    ) -> Tuple[List[str], List[str]]:
        """Return ``(kept, removed)`` — both sorted, de-duplicated lists.

        ``removed`` is exactly the set of objects excluded from the
        ``compare()``-visible object set; callers use it to populate
        ``filtered_objects``/``filtered_missing_objects``/
        ``filtered_additional_objects`` debug fields.
        """
        objs = list(dict.fromkeys(str(o) for o in (objects or [])))
        if not self.enabled or not objs:
            return objs, []

        if self.use_gt_vocabulary and gt_metadata:
            gt_labels = {str(k).strip().lower() for k in gt_metadata.keys()}
            kept = [o for o in objs if o.strip().lower() in gt_labels]
            removed = [o for o in objs if o.strip().lower() not in gt_labels]
            return sorted(set(kept)), sorted(set(removed))

        kept = [o for o in objs if not _is_excluded_term(o, self.excluded_terms)]
        removed = [o for o in objs if _is_excluded_term(o, self.excluded_terms)]
        return sorted(set(kept)), sorted(set(removed))


def build_object_vocabulary_filter(cfg) -> Optional[ObjectVocabularyFilter]:
    """Build an :class:`ObjectVocabularyFilter` from
    ``verifier.object_vocabulary_filter.*`` config keys.

    Returns ``None`` when ``verifier.object_vocabulary_filter.enabled`` is
    false/absent (default) — callers should treat ``None`` exactly like "no
    filter, objects list passed through unchanged" (equivalent to
    ``ObjectVocabularyFilter(enabled=False)``, but saves every caller from
    constructing a no-op instance).

    Config keys (all optional, under ``verifier.object_vocabulary_filter``)
    -------------------------------------------------------------------------
    enabled: bool = false
    use_gt_vocabulary: bool = true
    excluded_terms: list[str] | None      -- full override of the default set
    additional_excluded_terms: list[str]  -- merged on top of the default set
                                              (ignored if excluded_terms is set)
    """
    from omegaconf import OmegaConf

    if not bool(OmegaConf.select(cfg, "verifier.object_vocabulary_filter.enabled", default=False)):
        return None

    override = OmegaConf.select(cfg, "verifier.object_vocabulary_filter.excluded_terms", default=None)
    extra = OmegaConf.select(cfg, "verifier.object_vocabulary_filter.additional_excluded_terms", default=None)
    use_gt = bool(OmegaConf.select(cfg, "verifier.object_vocabulary_filter.use_gt_vocabulary", default=True))

    if override is not None:
        terms = frozenset(str(t).strip().lower() for t in override)
    else:
        terms = DEFAULT_EXCLUDED_TERMS
        if extra:
            terms = terms | frozenset(str(t).strip().lower() for t in extra)

    return ObjectVocabularyFilter(enabled=True, excluded_terms=terms, use_gt_vocabulary=use_gt)
