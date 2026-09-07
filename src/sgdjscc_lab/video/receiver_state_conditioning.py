"""Video-pipeline helpers for frame-aligned SAVER receiver conditions."""

from __future__ import annotations

from typing import Mapping, Sequence

from .keyframe_extractor import extract_keyframes


class ReceiverStateBoundaryExtractor:
    """Preserve base keyframes and split GOPs whenever receiver state changes.

    One Wan call produces one segment with one text condition. Forcing a GOP
    boundary at every receiver snapshot transition prevents one call from
    silently applying a future revoke/restore state to earlier target frames.
    """

    def __init__(self, base_extractor, condition_rows: Sequence[Mapping]) -> None:
        self.base_extractor = base_extractor
        self.condition_rows = list(condition_rows)

    def extract(self, frames):
        if len(frames) != len(self.condition_rows):
            raise ValueError("receiver condition rows must align with frames")
        base = self.base_extractor.extract(frames)
        keyframes = {int(value) for value in base.get("keyframes") or []}
        state_boundaries = []
        previous = None
        for index, row in enumerate(self.condition_rows):
            fingerprint = row.get("snapshot_fingerprint")
            if not isinstance(fingerprint, str):
                raise ValueError("receiver condition row lacks snapshot_fingerprint")
            if index and fingerprint != previous:
                keyframes.add(index)
                state_boundaries.append(index)
            previous = fingerprint
        boundaries = [index in keyframes for index in range(len(frames))]
        result = extract_keyframes(boundaries, max_gop=None)
        for key, value in base.items():
            if key not in {"keyframes", "gops", "frame_roles", "boundaries"}:
                result[key] = value
        result["boundaries"] = boundaries
        result["receiver_state_boundaries"] = state_boundaries
        result["base_keyframes"] = sorted(int(value) for value in base.get("keyframes") or [])
        return result
