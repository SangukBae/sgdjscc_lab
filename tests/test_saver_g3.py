from __future__ import annotations

import pytest

from sgdjscc_lab.evaluators.negative_semantics_g3 import summarize_g3
from sgdjscc_lab.guidance.saver_receiver_state import (
    G3_RECEIVER_ARMS,
    build_g3_receiver_condition_manifests,
    validate_g3_receiver_condition_manifest,
)


def _event(event_type: str):
    transition = {
        "ENTER": ("confirmed_absent", "present"),
        "EXIT": ("present", "confirmed_absent"),
        "OCCLUDE": ("present", "unknown"),
        "REAPPEAR": ("unknown", "present"),
    }[event_type]
    return {
        "event_id": f"event-{event_type.lower()}",
        "event_type": event_type,
        "video_id": f"video-{event_type.lower()}",
        "entity_id": "track-1",
        "concept_id": "dog",
        "start_frame": 2,
        "end_frame": 2,
        "state_before": transition[0],
        "state_after": transition[1],
        "scene_epoch": 0,
        "official_gt_verified": True,
    }


def _index():
    events = [_event(name) for name in ("ENTER", "EXIT", "OCCLUDE", "REAPPEAR")]
    return {
        "events": events,
        "videos": {
            event["video_id"]: {"frame_names": [f"{i}.jpg" for i in range(5)]}
            for event in events
        },
    }


def test_g3_manifests_encode_append_only_and_revocable_event_semantics():
    index = _index()
    manifests = build_g3_receiver_condition_manifests(index, sorted(index["videos"]))
    expected = {video_id: 5 for video_id in index["videos"]}
    for arm in G3_RECEIVER_ARMS:
        validate_g3_receiver_condition_manifest(manifests[arm], expected)

    exit_video = "video-exit"
    append = manifests["append_only_rsm"]["videos"][exit_video]["conditions"]
    revocable = manifests["revocable_rsm"]["videos"][exit_video]["conditions"]
    assert append[3]["active_entities"] == ("dog",)
    assert append[3]["negative_entities"] == ()
    assert revocable[3]["active_entities"] == ()
    assert revocable[3]["negative_entities"] == ("dog",)

    occluded = manifests["revocable_rsm"]["videos"]["video-occlude"]["conditions"]
    assert occluded[3]["active_entities"] == ()
    assert occluded[3]["negative_entities"] == ()


def _evaluation_rows():
    rows = []
    for arm in G3_RECEIVER_ARMS:
        for event_id in ("exit-1", "exit-2"):
            for frame_index in range(4):
                present = frame_index < 2
                if present:
                    score = 0.9
                elif arm == "append_only_rsm":
                    score = 0.8
                elif arm == "revocable_rsm":
                    score = 0.05
                else:
                    score = 0.3
                identity = {
                    "no_rsm": 0.50,
                    "append_only_rsm": 0.80,
                    "revocable_rsm": 0.76,
                }[arm]
                rows.append({
                    "arm": arm,
                    "event_id": event_id,
                    "event_type": "EXIT",
                    "event_frame": 2,
                    "frame_index": frame_index,
                    "gt_present": present,
                    "score": score,
                    "identity_similarity": identity,
                })
    return rows


def test_g3_summary_separates_identity_gain_and_ghost_revocation():
    summary = summarize_g3(_evaluation_rows(), bootstrap_iterations=100)
    assert summary["metrics"]["append_only_rsm"]["ghost_survival_auc"] == 1.0
    assert summary["metrics"]["revocable_rsm"]["ghost_survival_auc"] == 0.0
    assert summary["ghost_auc_relative_reduction"] == 1.0
    assert summary["provisional_gate_status"] == "PASSED"


def test_g3_summary_rejects_unpaired_arm_grid():
    rows = _evaluation_rows()
    rows.pop()
    with pytest.raises(ValueError, match="grids differ"):
        summarize_g3(rows, bootstrap_iterations=10)


def test_receiver_snapshot_transition_forces_gop_boundary():
    from sgdjscc_lab.video.receiver_state_conditioning import ReceiverStateBoundaryExtractor

    class Base:
        def extract(self, frames):
            return {
                "keyframes": [0],
                "gops": [{"keyframe": 0, "inter_frames": [1, 2, 3], "range": [0, 3]}],
                "frame_roles": ["keyframe", "inter", "inter", "inter"],
                "boundaries": [True, False, False, False],
                "distances": [0.0] * 4,
            }

    rows = [
        {"snapshot_fingerprint": "a" * 64},
        {"snapshot_fingerprint": "a" * 64},
        {"snapshot_fingerprint": "b" * 64},
        {"snapshot_fingerprint": "b" * 64},
    ]
    structure = ReceiverStateBoundaryExtractor(Base(), rows).extract([0, 1, 2, 3])
    assert structure["keyframes"] == [0, 2]
    assert structure["receiver_state_boundaries"] == [2]
    assert structure["gops"][0]["inter_frames"] == [1]
