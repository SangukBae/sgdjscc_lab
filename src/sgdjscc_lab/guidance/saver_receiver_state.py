"""Oracle G3 receiver-state manifests for event-level revocation experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence

from sgdjscc_lab.models.saver.contracts import AssertionAction, SemanticState
from sgdjscc_lab.models.saver.rsm_conditioning import compile_receiver_state_condition
from sgdjscc_lab.transmission.saver_packet import SignedStatePacket, VersionedEntityLedger


G3_RECEIVER_ARMS = ("no_rsm", "append_only_rsm", "revocable_rsm")
G3_CONDITION_MANIFEST_SCHEMA = "negative_semantics_g3_receiver_condition_v1"

_STATE_BY_NAME = {
    "present": SemanticState.PRESENT,
    "confirmed_absent": SemanticState.CONFIRMED_ABSENT,
    "unknown": SemanticState.UNKNOWN,
}
_EVENT_ACTION = {
    "ENTER": AssertionAction.RESUME,
    "EXIT": AssertionAction.REVOKE,
    "OCCLUDE": AssertionAction.SUSPEND,
    "REAPPEAR": AssertionAction.RESUME,
}


def _packet(event: Mapping[str, Any], *, state: SemanticState, action: AssertionAction,
            version: int) -> SignedStatePacket:
    return SignedStatePacket(
        scene_epoch=int(event.get("scene_epoch", 0)),
        entity_id=str(event["entity_id"]),
        version=int(version),
        state=state,
        action=action,
        confidence=1.0,
        payload=b"",
        ack_requested=False,
    )


def _initialize(
    ledger: VersionedEntityLedger, event: Mapping[str, Any], arm: str,
) -> None:
    state = _STATE_BY_NAME[str(event["state_before"])]
    if arm == "append_only_rsm" and str(event["event_type"]) in {
        "EXIT", "OCCLUDE", "REAPPEAR",
    }:
        # Append-only memory has already observed the entity before these
        # events and deliberately never removes/suspends it.
        state = SemanticState.PRESENT
    if state == SemanticState.PRESENT:
        action = AssertionAction.ASSERT
    elif state == SemanticState.CONFIRMED_ABSENT:
        action = AssertionAction.ASSERT
    else:
        action = AssertionAction.SUSPEND
    ledger.apply(_packet(event, state=state, action=action, version=0))


def _condition_frames(
    event: Mapping[str, Any], frame_count: int, arm: str,
) -> list[dict]:
    ledger = VersionedEntityLedger(scene_epoch=int(event.get("scene_epoch", 0)))
    if arm != "no_rsm":
        _initialize(ledger, event, arm)
    transition = int(event["start_frame"])
    if transition < 0 or transition >= frame_count:
        raise ValueError(
            f"event={event['event_id']} transition frame {transition} outside 0..{frame_count - 1}"
        )
    rows = []
    for frame_index in range(frame_count):
        if frame_index == transition and arm == "revocable_rsm":
            state = _STATE_BY_NAME[str(event["state_after"])]
            ledger.apply(_packet(
                event, state=state, action=_EVENT_ACTION[str(event["event_type"])], version=1
            ))
        elif frame_index == transition and arm == "append_only_rsm":
            state_after = _STATE_BY_NAME[str(event["state_after"])]
            if state_after == SemanticState.PRESENT:
                ledger.apply(_packet(
                    event, state=state_after, action=AssertionAction.UPDATE, version=1
                ))
            # EXIT/OCCLUDE intentionally leave the active entry untouched.
        condition = compile_receiver_state_condition(
            ledger, entity_labels={str(event["entity_id"]): str(event["concept_id"])}
        ).to_side_info()
        condition.update({
            "frame_index": frame_index,
            "event_id": str(event["event_id"]),
            "event_type": str(event["event_type"]),
            "oracle_eval_only": True,
            "serialized_in_packet": False,
            "rate_accounted": False,
        })
        rows.append(condition)
    return rows


def build_g3_receiver_condition_manifests(
    ground_truth_index: Mapping[str, Any],
    video_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Build no/append/revocable receiver snapshots for one event per video."""

    videos = ground_truth_index.get("videos") or {}
    events = ground_truth_index.get("events") or []
    by_video: Dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        by_video.setdefault(str(event["video_id"]), []).append(event)
    selected = [str(value) for value in video_ids]
    if len(set(selected)) != len(selected) or not selected:
        raise ValueError("video_ids must be non-empty and unique")

    manifests = {
        arm: {
            "schema": G3_CONDITION_MANIFEST_SCHEMA,
            "arm": arm,
            "oracle_side_input": "ORACLE_EVAL_ONLY",
            "serialized_in_packet": False,
            "rate_accounted": False,
            "videos": {},
        }
        for arm in G3_RECEIVER_ARMS
    }
    for video_id in selected:
        if video_id not in videos:
            raise ValueError(f"unknown G3 video_id={video_id!r}")
        candidates = by_video.get(video_id, [])
        if len(candidates) != 1:
            raise ValueError(
                f"G3 requires exactly one frozen event for video={video_id}; got {len(candidates)}"
            )
        event = candidates[0]
        if event.get("official_gt_verified") is not True:
            raise ValueError(f"G3 event is not official-GT verified: {event['event_id']}")
        count = len(videos[video_id]["frame_names"])
        for arm in G3_RECEIVER_ARMS:
            manifests[arm]["videos"][video_id] = {
                "event": deepcopy(dict(event)),
                "frame_count": count,
                "conditions": _condition_frames(event, count, arm),
            }
    return manifests


def validate_g3_receiver_condition_manifest(
    manifest: Mapping[str, Any], expected_frame_counts: Mapping[str, int],
) -> Dict[str, list[dict]]:
    if manifest.get("schema") != G3_CONDITION_MANIFEST_SCHEMA:
        raise ValueError("unsupported G3 condition manifest schema")
    arm = str(manifest.get("arm"))
    if arm not in G3_RECEIVER_ARMS:
        raise ValueError(f"unknown G3 arm={arm!r}")
    if manifest.get("oracle_side_input") != "ORACLE_EVAL_ONLY":
        raise ValueError("G3 condition must be marked ORACLE_EVAL_ONLY")
    if manifest.get("serialized_in_packet") is not False or manifest.get("rate_accounted") is not False:
        raise ValueError("G3 Oracle condition cannot be serialized or rate-accounted")
    videos = manifest.get("videos") or {}
    if set(videos) != set(expected_frame_counts):
        raise ValueError("G3 condition video grid differs from expected videos")
    output: Dict[str, list[dict]] = {}
    for video_id, expected in expected_frame_counts.items():
        item = videos[video_id]
        rows = item.get("conditions") or []
        if int(item.get("frame_count", -1)) != int(expected) or len(rows) != int(expected):
            raise ValueError(f"G3 frame grid mismatch for video={video_id}")
        if [int(row.get("frame_index", -1)) for row in rows] != list(range(int(expected))):
            raise ValueError(f"G3 frame indices are not contiguous for video={video_id}")
        if any(row.get("schema") != "saver_receiver_condition_v1" for row in rows):
            raise ValueError(f"invalid SAVER condition row for video={video_id}")
        output[video_id] = [dict(row) for row in rows]
    return output
