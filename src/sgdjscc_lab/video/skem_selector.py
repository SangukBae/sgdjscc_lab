"""video/skem_selector.py – PSSS-driven SKEM keyframe selector.

Implements the LGVSC paper's SKEM (semantic-guided keyframe extraction
module, Sec. "SKIM or SKEM-based Semantic Encoder at Transmitter" →
"SKEM"): an autoregressive keyframe selector that inserts a new keyframe
whenever the current frame's semantic description has diverged from the
*latest keyframe's* description by more than a threshold, as measured by
PSSS (``video/psss.py``).

This is a genuinely **independent** selector from the pre-existing
``video/keyframe_extractor.py::KeyframeExtractor`` (scene-change histogram/
CLIP/LPIPS distance + ``max_gop`` cap — the "SKIM-nearest" fixed/adaptive
extractor every 1C mode used before this module existed). Both expose the
same ``.extract(frames) -> Dict`` interface (``keyframes``/``gops``/
``frame_roles``/``boundaries``/``distances``) so ``TemporalPipeline`` accepts
either interchangeably — see ``build_keyframe_extractor()`` in
``keyframe_extractor.py`` for the config-driven factory that picks one.

Algorithm (paper-faithful core + two ETRI additions, both clearly marked)
---------------------------------------------------------------------------
Per the paper: ``K_1 = 1`` (first frame is always a keyframe). For each
subsequent frame ``i``, compare its description to the *latest* keyframe's
description via PSSS's ``S_rel`` (Eq. 2); if ``S_rel > eta_th`` insert frame
``i`` as a new keyframe and advance the "latest keyframe" pointer, else leave
the current segment growing. ``eta_th`` (``threshold``) defaults to 0.35,
matching the paper's SKEM+DSA experimental setting.

Two additions beyond the paper (both are ETRI additions for practical
robustness — the paper does not specify a lower bound, and the upper bound
in the paper is stated only in terms of a downstream CBR calibration
procedure, not enforced inside the selector loop itself):

- ``min_segment_length``: frames within this many positions of the latest
  keyframe are never evaluated (and never inserted as a keyframe) — bounds
  the number of (real-MLLM-cost) PSSS calls and prevents pathologically
  short segments.
- ``max_segment_length``: a keyframe is forced once a segment reaches this
  length, regardless of the PSSS score — bounds worst-case segment length
  for CBR/latency and mirrors the safety role
  ``keyframe.max_gop`` plays for the existing fixed/scene-change extractor.

Both are recorded as such (not attributed to PSSS) in each keyframe's
``reason`` string and in ``psss_scores``/``keyframe_reasons``.

Variable-length segments
--------------------------
Because keyframe spacing depends on the (data-dependent) PSSS decisions
above, the GOPs this selector produces are genuinely variable-length — this
is the concrete implementation of the "DSA needs a variable-length-aware
decoder" argument in the paper's Appendix C ("SKEM+SFA is architecturally
incompatible"). Nothing else needs to change for a variable-length segment
to flow through: ``TemporalPipeline``/``video/segment.py``/
``video/video_generator.py``'s segment-level contract (ETRI 1A) was already
segment-length-agnostic (a GOP's length was already whatever the extractor
produced) — this selector simply makes the *existing* fixed/scene-change
extractor's implicit "whatever length scene-change detection settles on"
behaviour into an explicit, semantically-motivated PSSS decision.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from sgdjscc_lab.video.psss import DEFAULT_SEMANTIC_FOCUS, PsssBackend

logger = logging.getLogger(__name__)


class PsssKeyframeSelector:
    """SKEM keyframe selector: PSSS-thresholded autoregressive keyframe
    insertion producing variable-length GOPs.

    Parameters
    ----------
    caption_fn:
        ``(frame_tensor, frame_index) -> str`` — the "convert to text first"
        step the paper's SKEM prescribes (Sec. subsec:skim, "using
        pre-trained multimodal language models ... to generate textual
        descriptions"). Dependency-injected so this module never itself
        depends on a specific captioner; ``keyframe_extractor.py``'s factory
        wires it to whichever caption source a run has available (BLIP2/
        Qwen captioner, a ``--captions`` file, or a test/mock function).
        Memoised internally per call to :meth:`extract` (each frame's
        caption is computed at most once even though it may be compared
        against multiple later candidates).
    psss_backend:
        A :class:`~sgdjscc_lab.video.psss.PsssBackend` (mock/proxy/real).
    threshold:
        ``eta_th`` — the semantic divergence threshold (paper default 0.35).
        A new keyframe is inserted when ``S_rel > threshold`` (strict, per
        the paper's Sec. subsec:skim).
    semantic_focus:
        The PSSS prompt's ``<Semantic Focus>`` field.
    min_segment_length / max_segment_length:
        ETRI additions — see module docstring. ``min_segment_length`` must be
        >= 1 (a keyframe's own frame counts as position 0 of its segment).
        ``max_segment_length`` (if given) must be >= ``min_segment_length``.
    scene_detector:
        Optional ``SceneChangeDetector`` (or any object exposing
        ``detect(frames) -> {"boundaries": [...], ...}``). When supplied,
        ``extract()`` runs it once up front and forces a keyframe at any
        detected scene boundary — the same way ``max_segment_length`` forces
        one — *in addition to* the PSSS-driven decisions, so a hard scene cut
        is never missed just because the caption/PSSS score happened to stay
        below ``threshold``. ``None`` (default) preserves the original
        PSSS-only behaviour exactly (no scene-change signal consulted).
        Every forced/PSSS decision is also recorded in the returned
        ``force_reason`` dict as one of the categorical values
        ``"first_frame"|"scene_change"|"max_segment_length"|"psss"`` —
        callers should read that field rather than pattern-matching the
        human-readable ``keyframe_reasons`` prose string.
    """

    selector_name = "psss"

    def __init__(
        self,
        caption_fn: Callable[[object, int], str],
        psss_backend: PsssBackend,
        threshold: float = 0.35,
        semantic_focus: str = DEFAULT_SEMANTIC_FOCUS,
        min_segment_length: int = 1,
        max_segment_length: Optional[int] = None,
        seed: Optional[int] = None,
        scene_detector=None,
    ) -> None:
        if min_segment_length < 1:
            raise ValueError(f"min_segment_length must be >= 1; got {min_segment_length}.")
        if max_segment_length is not None and max_segment_length < min_segment_length:
            raise ValueError(
                f"max_segment_length ({max_segment_length}) must be >= "
                f"min_segment_length ({min_segment_length})."
            )
        self.caption_fn = caption_fn
        self.psss_backend = psss_backend
        self.threshold = float(threshold)
        self.semantic_focus = str(semantic_focus)
        self.min_segment_length = int(min_segment_length)
        self.max_segment_length = None if max_segment_length is None else int(max_segment_length)
        self.seed = seed
        self.scene_detector = scene_detector

    def _empty_result(self) -> Dict:
        return {
            "keyframes": [], "gops": [], "frame_roles": [], "boundaries": [], "distances": [],
            "selector": self.selector_name,
            "psss_backend": getattr(self.psss_backend, "backend_name", None),
            "psss_backend_kind": self.psss_backend.backend_kind,
            "psss_model_id": getattr(self.psss_backend, "model_id", None),
            "psss_threshold": self.threshold,
            "psss_semantic_focus": self.semantic_focus,
            "psss_min_segment_length": self.min_segment_length,
            "psss_max_segment_length": self.max_segment_length,
            "psss_scores": [],
            "keyframe_reasons": {},
            "force_reason": {},
            "scene_change_used": self.scene_detector is not None,
        }

    def extract(self, frames: List) -> Dict:
        n = len(frames)
        if n == 0:
            return self._empty_result()

        captions: List[Optional[str]] = [None] * n

        def _caption(i: int) -> str:
            if captions[i] is None:
                captions[i] = self.caption_fn(frames[i], i)
            return captions[i]

        # Real scene-change signal, computed once up front (not inferred from
        # any reason string later) — see class docstring's scene_detector param.
        scene_boundaries: List[bool] = [False] * n
        if self.scene_detector is not None:
            scene_boundaries = list(self.scene_detector.detect(frames)["boundaries"])

        keyframes: List[int] = [0]
        boundaries = [False] * n
        boundaries[0] = True
        distances = [0.0] * n
        frame_roles = ["inter"] * n
        frame_roles[0] = "keyframe"
        psss_scores: List[Dict] = []
        keyframe_reasons: Dict[int, str] = {
            0: "first frame (K_1 = 1) — always a keyframe, no PSSS evaluated."
        }
        force_reason: Dict[int, str] = {0: "first_frame"}

        current_kf = 0
        _caption(0)

        i = 1
        while i < n:
            span = i - current_kf

            # Scene-change forcing takes priority over max_segment_length: a
            # detected hard cut is a stronger, content-derived signal than an
            # arbitrary length cap, and checking it first means a scene change
            # that happens to also land past max_segment_length is correctly
            # attributed to "scene_change" rather than the length cap.
            if scene_boundaries[i]:
                reason = (
                    "scene_change detected by scene_detector "
                    f"(segment length {span} since keyframe {current_kf}) — forced "
                    "keyframe; NOT a PSSS decision (real scene-change signal, "
                    "combined with SKEM/PSSS per the ETRI scene_detector integration)."
                )
                keyframes.append(i)
                boundaries[i] = True
                frame_roles[i] = "keyframe"
                keyframe_reasons[i] = reason
                force_reason[i] = "scene_change"
                current_kf = i
                _caption(i)
                i += 1
                continue

            if self.max_segment_length is not None and span >= self.max_segment_length:
                reason = (
                    f"max_segment_length={self.max_segment_length} reached "
                    f"(segment length {span} since keyframe {current_kf}) — forced "
                    "keyframe; NOT a PSSS decision (ETRI addition, not in the LGVSC "
                    "paper's SKEM description)."
                )
                keyframes.append(i)
                boundaries[i] = True
                frame_roles[i] = "keyframe"
                keyframe_reasons[i] = reason
                force_reason[i] = "max_segment_length"
                current_kf = i
                _caption(i)
                i += 1
                continue

            if span < self.min_segment_length:
                # ETRI addition: too soon to insert a new keyframe — skip PSSS
                # evaluation entirely (saves a real-MLLM call when backend="real").
                i += 1
                continue

            info_a = _caption(current_kf)
            info_b = _caption(i)
            score = self.psss_backend.score(info_a, info_b, self.semantic_focus)
            distances[i] = score.s_rel
            # Start from the FULL PsssScoreResult (raw_logits/evidence/
            # model_id/proxy_of/notes included — see psss.py's to_dict()) so
            # keyframes.json/segments.json carry the same provenance a caller
            # inspecting PsssScoreResult directly would see, not just the
            # handful of fields the selector's own decision needs.
            record = score.to_dict()
            record.update({
                "index": i,
                "compared_to_keyframe": current_kf,
                "threshold": self.threshold,
            })
            if score.s_rel > self.threshold:
                reason = (
                    f"S_rel={score.s_rel:.4f} > threshold={self.threshold:g} vs "
                    f"keyframe {current_kf} → semantic divergence, new keyframe."
                )
                record["decision"] = "new_keyframe"
                keyframes.append(i)
                boundaries[i] = True
                frame_roles[i] = "keyframe"
                keyframe_reasons[i] = reason
                force_reason[i] = "psss"
                current_kf = i
            else:
                record["decision"] = "continue_segment"
                keyframe_reasons[i] = (
                    f"S_rel={score.s_rel:.4f} <= threshold={self.threshold:g} vs "
                    f"keyframe {current_kf} → semantically similar, no new keyframe."
                )
            psss_scores.append(record)
            i += 1

        gops = self._build_gops(keyframes, n)

        return {
            "keyframes": keyframes,
            "gops": gops,
            "frame_roles": frame_roles,
            "boundaries": boundaries,
            "distances": distances,
            "selector": self.selector_name,
            "psss_backend": getattr(self.psss_backend, "backend_name", None),
            "psss_backend_kind": self.psss_backend.backend_kind,
            "psss_model_id": getattr(self.psss_backend, "model_id", None),
            "psss_threshold": self.threshold,
            "psss_semantic_focus": self.semantic_focus,
            "psss_min_segment_length": self.min_segment_length,
            "psss_max_segment_length": self.max_segment_length,
            "psss_scores": psss_scores,
            "keyframe_reasons": {str(k): v for k, v in keyframe_reasons.items()},
            "force_reason": {str(k): v for k, v in force_reason.items()},
            "scene_change_used": self.scene_detector is not None,
        }

    @staticmethod
    def _build_gops(keyframes: List[int], n: int) -> List[Dict]:
        gops = []
        for pos, kf in enumerate(keyframes):
            end = keyframes[pos + 1] - 1 if pos + 1 < len(keyframes) else n - 1
            inter = list(range(kf + 1, end + 1))
            gops.append({"keyframe": kf, "inter_frames": inter, "range": [kf, end]})
        return gops
