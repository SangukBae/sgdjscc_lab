"""video/keyframe_extractor.py – GOP-like keyframe grouping (Phase 4-B).

Turns scene-boundary flags into a GOP (group-of-pictures) structure: each
keyframe is reconstructed with the full image pipeline and full semantic packet,
while the inter-frames that follow it reuse the keyframe packet plus a semantic
delta.  A new keyframe is forced whenever:

1. the scene-change detector marks a boundary, or
2. the current GOP reaches ``max_gop`` frames (so long static shots still refresh
   the reference periodically).

The output mirrors a classic video GOP layout — keyframe indices and the
inter-frame index ranges that depend on each keyframe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GOP:
    """One group of pictures: a keyframe and its dependent inter-frames."""

    keyframe: int
    inter_frames: List[int] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.keyframe

    @property
    def end(self) -> int:
        return self.inter_frames[-1] if self.inter_frames else self.keyframe

    def as_dict(self) -> Dict:
        return {
            "keyframe": self.keyframe,
            "inter_frames": list(self.inter_frames),
            "range": [self.start, self.end],
        }


def extract_keyframes(
    boundaries: List[bool],
    max_gop: Optional[int] = None,
) -> Dict:
    """Group frame indices into GOPs from scene-boundary flags.

    Parameters
    ----------
    boundaries:
        ``boundaries[i]`` is True when frame *i* starts a new scene (frame 0
        should be True).  Typically from ``SceneChangeDetector.detect``.
    max_gop:
        Maximum number of frames per GOP (keyframe + inter-frames).  ``None`` or
        ``<= 0`` disables the cap.

    Returns
    -------
    dict with keys:
        ``keyframes``   – sorted list of keyframe indices.
        ``gops``        – list of GOP dicts (keyframe, inter_frames, range).
        ``frame_roles`` – list[str] of "keyframe"/"inter" per frame index.
    """
    n = len(boundaries)
    cap = max_gop if (max_gop and max_gop > 0) else None

    gops: List[GOP] = []
    current: Optional[GOP] = None

    for i in range(n):
        force_new = bool(boundaries[i])
        if cap is not None and current is not None:
            if (i - current.keyframe) >= cap:
                force_new = True
        if current is None or force_new:
            current = GOP(keyframe=i)
            gops.append(current)
        else:
            current.inter_frames.append(i)

    keyframes = [g.keyframe for g in gops]
    frame_roles = ["inter"] * n
    for k in keyframes:
        frame_roles[k] = "keyframe"

    return {
        "keyframes": keyframes,
        "gops": [g.as_dict() for g in gops],
        "frame_roles": frame_roles,
    }


class KeyframeExtractor:
    """OO wrapper bundling scene detection + GOP grouping.

    Parameters
    ----------
    scene_detector:
        A ``SceneChangeDetector`` (or any object with ``detect(frames)``).
    max_gop:
        Maximum GOP length passed to :func:`extract_keyframes`.
    """

    def __init__(self, scene_detector, max_gop: Optional[int] = 12) -> None:
        self.scene_detector = scene_detector
        self.max_gop = max_gop

    def extract(self, frames) -> Dict:
        """Detect boundaries on *frames* then return the keyframe/GOP structure.

        The scene-detector output (boundaries, distances) is merged into the
        returned dict for logging/analysis.
        """
        detection = self.scene_detector.detect(frames)
        result = extract_keyframes(detection["boundaries"], max_gop=self.max_gop)
        result["distances"] = detection["distances"]
        result["boundaries"] = detection["boundaries"]
        return result


class FixedIntervalKeyframeSelector:
    """Literal fixed-interval keyframe selector — the LGVSC paper's SKIM
    definition exactly (Sec. subsec:skim): equal-length segments of
    ``interval`` frames, keyframe = first frame of each segment, **zero**
    scene-change signal.

    This is deliberately a *different* class from :class:`KeyframeExtractor`
    above. ``KeyframeExtractor`` (scene-change distance + ``max_gop`` cap) is
    what every pre-PSSS 1C mode (`mock_baseline`/`svd_start_only`/
    `wan_skim_sfa`/`wan_skem_dsa`) uses, and those configs' own docs already
    flag it as only a "nearest reproducible" SKIM approximation — it still
    inserts extra keyframes on a real scene change, which the paper's SKIM
    never does (SKIM has no visual-difference signal at all, only a frame
    count). For a comparison line that is honestly "SKIM, not an
    approximation of it", use this class instead (``keyframe.selector:
    fixed_interval`` — see ``configs/experiments/lgvsc_1c/etri_lgvsc_1c_skim_sfa_fixed.yaml``).

    Parameters
    ----------
    interval:
        Segment length in frames (``>= 1``). The last segment may be shorter
        when ``len(frames)`` is not a multiple of ``interval`` (matches
        :class:`KeyframeExtractor`'s own last-GOP truncation behaviour, and
        the paper's own allowance for a final partial segment).
    """

    selector_name = "fixed_interval"

    def __init__(self, interval: int = 12) -> None:
        if interval < 1:
            raise ValueError(f"interval must be >= 1; got {interval}")
        self.interval = int(interval)

    def extract(self, frames) -> Dict:
        n = len(frames)
        if n == 0:
            return {
                "keyframes": [], "gops": [], "frame_roles": [], "boundaries": [], "distances": [],
                "selector": self.selector_name, "fixed_interval": self.interval,
                "psss_backend_kind": "not_applicable",
            }

        keyframes = list(range(0, n, self.interval))
        boundaries = [False] * n
        frame_roles = ["inter"] * n
        for k in keyframes:
            boundaries[k] = True
            frame_roles[k] = "keyframe"

        gops: List[Dict] = []
        for pos, kf in enumerate(keyframes):
            end = keyframes[pos + 1] - 1 if pos + 1 < len(keyframes) else n - 1
            gops.append({"keyframe": kf, "inter_frames": list(range(kf + 1, end + 1)), "range": [kf, end]})

        return {
            "keyframes": keyframes,
            "gops": gops,
            "frame_roles": frame_roles,
            "boundaries": boundaries,
            "distances": [0.0] * n,
            "selector": self.selector_name,
            "fixed_interval": self.interval,
            "psss_backend_kind": "not_applicable",
        }


class FixedCountKeyframeSelector:
    """Content-independent SKIM selector producing exactly ``count`` keyframes.

    Equal integer intervals cannot represent every ``(n_frames, count)`` pair:
    for example, no integer interval produces 6 keyframes from 10 frames.
    This selector partitions the clip at ``floor(j * n / count)`` for
    ``j = 0 .. count-1``. Segment lengths differ by at most one frame and the
    requested keyframe count is exact.
    """

    selector_name = "fixed_count"

    def __init__(self, count: int) -> None:
        if count < 1:
            raise ValueError(f"count must be >= 1; got {count}")
        self.count = int(count)

    def extract(self, frames) -> Dict:
        n = len(frames)
        if n == 0:
            return {
                "keyframes": [], "gops": [], "frame_roles": [], "boundaries": [],
                "distances": [], "selector": self.selector_name,
                "fixed_count": self.count, "psss_backend_kind": "not_applicable",
            }
        if self.count > n:
            raise ValueError(
                f"fixed_count={self.count} cannot be represented by a {n}-frame clip."
            )

        keyframes = [(j * n) // self.count for j in range(self.count)]
        boundaries = [False] * n
        frame_roles = ["inter"] * n
        for k in keyframes:
            boundaries[k] = True
            frame_roles[k] = "keyframe"

        gops: List[Dict] = []
        for pos, kf in enumerate(keyframes):
            end = keyframes[pos + 1] - 1 if pos + 1 < len(keyframes) else n - 1
            gops.append({
                "keyframe": kf,
                "inter_frames": list(range(kf + 1, end + 1)),
                "range": [kf, end],
            })

        return {
            "keyframes": keyframes,
            "gops": gops,
            "frame_roles": frame_roles,
            "boundaries": boundaries,
            "distances": [0.0] * n,
            "selector": self.selector_name,
            "fixed_count": self.count,
            "psss_backend_kind": "not_applicable",
        }


# ── Selector factory (PSSS/SKEM readiness step) ─────────────────────────────
#
# Two independent, interchangeable keyframe-selector backends now exist:
# ``KeyframeExtractor`` above (scene-change/histogram distance + max_gop cap —
# the pre-existing "SKIM-nearest" fixed/adaptive extractor every 1A-1C mode
# used) and ``video/skem_selector.py::PsssKeyframeSelector`` (PSSS-thresholded
# semantic keyframe selection, LGVSC's SKEM). ``build_keyframe_extractor``
# picks one from ``keyframe.selector`` (default ``"fixed"`` — the untouched,
# numerically-identical-to-before path); ``keyframe.selector: "psss"`` is
# opt-in only.

def build_caption_fn(caption_source: str, *, captions=None, text_extractor=None, device=None):
    """Build a ``(frame_tensor, frame_index) -> str`` caption function for
    :class:`~sgdjscc_lab.video.skem_selector.PsssKeyframeSelector` from one of
    three sources.

    Parameters
    ----------
    caption_source:
        ``"captions_file"`` — look up a pre-loaded per-frame caption list
        (e.g. from ``--captions``, the same list ``evaluate_video.py`` already
        loads for packet building). ``"model"`` — call a real BLIP2/Qwen
        ``TextExtractor`` on the frame itself (real captioning, has a real
        model-inference cost per frame — see the module docstring's
        performance note). ``"mock"`` — a deterministic, dependency-free
        placeholder derived from simple per-channel frame statistics; NOT
        real captioning, structural test/dry-run use only.
    captions:
        Required (and used) only for ``caption_source="captions_file"``.
    text_extractor:
        Required (and used) only for ``caption_source="model"`` — a
        ``guidance.text_extractor.TextExtractor``-like object (must implement
        ``.extract(img_tensor, device) -> List[List[str]]``).
    device:
        Forwarded to ``text_extractor.extract`` for ``caption_source="model"``.
    """
    caption_source = str(caption_source)

    if caption_source == "captions_file":
        if captions is None:
            raise ValueError(
                "keyframe.psss.caption_source='captions_file' requires captions to be "
                "supplied (e.g. evaluate_video.py's --captions)."
            )
        _captions = list(captions)

        def _from_file(frame, idx):
            return _captions[idx] if 0 <= idx < len(_captions) else ""

        return _from_file

    if caption_source == "model":
        if text_extractor is None:
            raise ValueError(
                "keyframe.psss.caption_source='model' requires a real text_extractor "
                "(BLIP2/Qwen) — run without --no-models."
            )
        import torch
        dev = device or torch.device("cpu")

        def _from_model(frame, idx):
            f = frame if frame.dim() == 4 else frame.unsqueeze(0)
            caps = text_extractor.extract(f, dev)
            return caps[0][0] if caps and caps[0] else ""

        return _from_model

    if caption_source == "mock":
        def _mock(frame, idx):
            f = frame if frame.dim() == 3 else frame[0]
            means = f.float().mean(dim=(1, 2)).tolist()
            return f"frame {idx} stats " + " ".join(f"{m:.4f}" for m in means)

        return _mock

    raise NotImplementedError(
        f"keyframe.psss.caption_source={caption_source!r} is not implemented; "
        "expected one of 'captions_file' | 'model' | 'mock'."
    )


def build_keyframe_extractor(
    cfg,
    *,
    scene_detector=None,
    caption_fn=None,
    psss_backend=None,
):
    """Build the configured keyframe extractor/selector from ``keyframe.*``.

    ``keyframe.selector`` (default ``"fixed"``): ``"fixed"`` returns
    ``KeyframeExtractor(scene_detector, max_gop=keyframe.max_gop)`` — exactly
    what every caller built by hand before this factory existed, so a config
    that never sets ``keyframe.selector`` is completely unaffected.
    ``"fixed_interval"`` builds a
    :class:`~sgdjscc_lab.video.keyframe_extractor.FixedIntervalKeyframeSelector`
    (LGVSC's literal SKIM — zero scene-change signal) from
    ``keyframe.fixed_interval.interval`` (defaults to ``keyframe.max_gop`` so
    it is directly comparable to a PSSS selector's ``max_segment_length`` —
    same worst-case segment length, same style of cap). ``"fixed_count"``
    builds a content-independent, near-equal-length partition with exactly
    ``keyframe.fixed_count.count`` keyframes. ``"psss"`` builds a
    :class:`~sgdjscc_lab.video.skem_selector.PsssKeyframeSelector` from
    ``keyframe.psss.*`` (``backend``/``threshold``/``semantic_focus``/
    ``min_segment_length``/``max_segment_length``/``seed``), using
    *psss_backend* if supplied (dependency injection for tests) or building
    one via :func:`sgdjscc_lab.video.psss.build_psss_backend` otherwise, and
    *caption_fn* (required for ``"psss"``; build it with
    :func:`build_caption_fn`).
    """
    from omegaconf import OmegaConf

    selector = str(OmegaConf.select(cfg, "keyframe.selector", default="fixed"))
    max_gop = int(OmegaConf.select(cfg, "keyframe.max_gop", default=12))

    if selector == "fixed":
        if scene_detector is None:
            raise ValueError("keyframe.selector='fixed' requires a scene_detector.")
        return KeyframeExtractor(scene_detector, max_gop=max_gop)

    if selector == "fixed_interval":
        interval = OmegaConf.select(cfg, "keyframe.fixed_interval.interval", default=None)
        interval = max_gop if interval is None else int(interval)
        return FixedIntervalKeyframeSelector(interval=interval)

    if selector == "fixed_count":
        count = OmegaConf.select(cfg, "keyframe.fixed_count.count", default=None)
        if count is None:
            raise ValueError(
                "keyframe.selector='fixed_count' requires keyframe.fixed_count.count."
            )
        return FixedCountKeyframeSelector(count=int(count))

    if selector == "psss":
        from sgdjscc_lab.video.psss import DEFAULT_SEMANTIC_FOCUS, build_psss_backend
        from sgdjscc_lab.video.skem_selector import PsssKeyframeSelector

        p = "keyframe.psss."
        if psss_backend is None:
            backend_name = str(OmegaConf.select(cfg, p + "backend", default="mock"))
            psss_backend = build_psss_backend(
                backend_name, cfg=OmegaConf.select(cfg, "keyframe.psss", default=None)
            )
        if caption_fn is None:
            raise ValueError(
                "keyframe.selector='psss' requires a caption_fn — build one with "
                "build_caption_fn(keyframe.psss.caption_source, ...)."
            )
        max_len = OmegaConf.select(cfg, p + "max_segment_length", default=None)
        # Default False preserves the original PSSS-only selector exactly (no
        # scene_detector passed unless explicitly requested) — combining real
        # scene-change detection with PSSS is opt-in via this flag.
        use_scene_detector = bool(OmegaConf.select(cfg, p + "use_scene_detector", default=False))
        if use_scene_detector and scene_detector is None:
            raise ValueError(
                "keyframe.psss.use_scene_detector=true requires a scene_detector "
                "(e.g. build_keyframe_extractor(cfg, scene_detector=SceneChangeDetector(), ...))."
            )
        return PsssKeyframeSelector(
            caption_fn=caption_fn,
            psss_backend=psss_backend,
            threshold=float(OmegaConf.select(cfg, p + "threshold", default=0.35)),
            semantic_focus=str(OmegaConf.select(cfg, p + "semantic_focus", default=DEFAULT_SEMANTIC_FOCUS)),
            min_segment_length=int(OmegaConf.select(cfg, p + "min_segment_length", default=1)),
            max_segment_length=(None if max_len is None else int(max_len)),
            seed=OmegaConf.select(cfg, p + "seed", default=None),
            scene_detector=(scene_detector if use_scene_detector else None),
        )

    raise NotImplementedError(
        f"keyframe.selector={selector!r} is not implemented; expected 'fixed', "
        "'fixed_interval', 'fixed_count', or 'psss'."
    )
