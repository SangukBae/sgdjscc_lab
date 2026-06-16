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
