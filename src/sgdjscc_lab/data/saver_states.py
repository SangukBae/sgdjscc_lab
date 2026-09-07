"""Deterministic target construction for SAVER-JSCC signed entity states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sgdjscc_lab.models.saver.contracts import (
    AssertionAction,
    EntityObservation,
    SemanticState,
    SignedEntityEvent,
    infer_action,
)


@dataclass(frozen=True)
class EntitySlotTargets:
    """Fixed-width labels produced without reading receiver reconstructions."""

    entity_ids: Tuple[str, ...]
    states: Tuple[SemanticState, ...]
    actions: Tuple[AssertionAction, ...]
    confidence: Tuple[float, ...]
    valid_mask: Tuple[bool, ...]
    events: Tuple[SignedEntityEvent, ...]


def _index(observations: Iterable[EntityObservation]) -> Dict[str, EntityObservation]:
    out: Dict[str, EntityObservation] = {}
    for observation in observations:
        if observation.entity_id in out:
            raise ValueError(f"duplicate entity_id={observation.entity_id!r}")
        out[observation.entity_id] = observation
    return out


def build_entity_slot_targets(
    previous: Sequence[EntityObservation],
    current: Sequence[EntityObservation],
    *,
    scene_epoch: int,
    previous_versions: Optional[Mapping[str, int]] = None,
    slot_count: int = 16,
    scene_reset: bool = False,
) -> EntitySlotTargets:
    """Build deterministic slot labels and monotonic events.

    Existing entity order is retained first to stabilize slot identity.  New
    entities are appended in lexical ID order.  Entities absent from
    ``current`` are emitted as ``UNKNOWN``/``SUSPEND`` (when previously
    present), never as ``CONFIRMED_ABSENT``.
    """

    if scene_epoch < 0 or slot_count <= 0:
        raise ValueError("scene_epoch must be non-negative and slot_count positive")
    prev = _index(previous)
    curr = _index(current)
    previous_versions = dict(previous_versions or {})

    ordered_ids = list(prev)
    ordered_ids.extend(sorted(entity_id for entity_id in curr if entity_id not in prev))
    if len(ordered_ids) > slot_count:
        raise ValueError(
            f"{len(ordered_ids)} entities exceed slot_count={slot_count}; "
            "apply an explicit eviction policy before target construction"
        )

    events: List[SignedEntityEvent] = []
    if scene_reset:
        events.append(SignedEntityEvent(
            scene_epoch=scene_epoch,
            entity_id="",
            version=0,
            state=SemanticState.UNKNOWN,
            action=AssertionAction.SCENE_RESET,
            confidence=1.0,
            provenance={"source": "scene_boundary"},
        ))

    ids: List[str] = []
    states: List[SemanticState] = []
    actions: List[AssertionAction] = []
    confidences: List[float] = []
    valid: List[bool] = []
    for entity_id in ordered_ids:
        before = prev.get(entity_id)
        now = curr.get(entity_id)
        if now is None:
            now = EntityObservation(
                entity_id=entity_id,
                state=SemanticState.UNKNOWN,
                confidence=0.0,
                feature=before.feature if before is not None else (),
                provenance={"source": "unobserved", "confirmed_absent": False},
            )
        action = infer_action(before.state if before else None, now.state)
        version = int(previous_versions.get(entity_id, -1)) + 1
        ids.append(entity_id)
        states.append(now.state)
        actions.append(action)
        confidences.append(now.confidence)
        valid.append(True)
        if action != AssertionAction.SKIP:
            events.append(SignedEntityEvent(
                scene_epoch=scene_epoch,
                entity_id=entity_id,
                version=version,
                state=now.state,
                action=action,
                confidence=now.confidence,
                feature=now.feature,
                provenance=now.provenance,
            ))

    pad = slot_count - len(ids)
    ids.extend([""] * pad)
    states.extend([SemanticState.UNKNOWN] * pad)
    actions.extend([AssertionAction.SKIP] * pad)
    confidences.extend([0.0] * pad)
    valid.extend([False] * pad)
    return EntitySlotTargets(
        entity_ids=tuple(ids),
        states=tuple(states),
        actions=tuple(actions),
        confidence=tuple(confidences),
        valid_mask=tuple(valid),
        events=tuple(events),
    )

