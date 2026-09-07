"""Deterministic receiver-state conditioning compiled from delivered packets.

This module is deliberately downstream of :class:`VersionedEntityLedger`.
It cannot inspect transmitter annotations or source frames, which makes the
resulting prompt condition legal at the receiver and auditable from wire
packets alone.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Mapping, Optional

from sgdjscc_lab.models.saver.contracts import RenderStatus
from sgdjscc_lab.transmission.saver_packet import VersionedEntityLedger


SAVER_RECEIVER_CONDITION_SCHEMA = "saver_receiver_condition_v1"
_SPACE = re.compile(r"\s+")


def _safe_label(value: object) -> str:
    text = _SPACE.sub(" ", str(value)).strip()
    text = "".join(ch for ch in text if ch.isprintable())
    return text[:80]


@dataclass(frozen=True)
class ReceiverStateCondition:
    """Prompt-level view of one deterministic receiver ledger snapshot."""

    scene_epoch: int
    max_version: int
    active_entities: tuple[str, ...]
    negative_entities: tuple[str, ...]
    positive_prompt: str
    negative_prompt: str
    snapshot_fingerprint: str
    schema: str = SAVER_RECEIVER_CONDITION_SCHEMA

    def to_side_info(self) -> dict:
        """Return a JSON-safe side-info object for the external Wan worker."""

        return asdict(self)


def compile_receiver_state_condition(
    ledger: VersionedEntityLedger,
    *,
    entity_labels: Optional[Mapping[str, str]] = None,
) -> ReceiverStateCondition:
    """Compile active and negative prompts from receiver-applied state only.

    ``entity_labels`` must be receiver-known metadata keyed by transmitted
    entity ID. Missing labels fall back to the ID itself. Suspended/unknown
    entities intentionally appear in neither prompt; uncertainty must not be
    converted into confirmed absence.
    """

    labels = entity_labels or {}
    active: list[str] = []
    negative: list[str] = []
    snapshot = []
    max_version = ledger.scene_reset_version
    for entity_id, entry in sorted(ledger.entries.items()):
        label = _safe_label(labels.get(entity_id, entity_id))
        if not label:
            label = _safe_label(entity_id)
        max_version = max(max_version, entry.version)
        snapshot.append({
            "entity_id": entity_id,
            "version": entry.version,
            "state": int(entry.state),
            "render_status": int(entry.render_status),
            "confidence": round(float(entry.confidence), 7),
            "payload_sha256": hashlib.sha256(entry.payload).hexdigest(),
        })
        if entry.render_status == RenderStatus.ACTIVE:
            active.append(label)
        elif entry.render_status in (RenderStatus.ABSENT, RenderStatus.REVOKED):
            negative.append(label)

    canonical = json.dumps(
        {
            "schema": SAVER_RECEIVER_CONDITION_SCHEMA,
            "scene_epoch": ledger.scene_epoch,
            "scene_reset_version": ledger.scene_reset_version,
            "entries": snapshot,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    positive_prompt = (
        "Currently visible entities: " + ", ".join(active) + "." if active else ""
    )
    negative_prompt = (
        "Do not render absent or revoked entities: " + ", ".join(negative) + "."
        if negative else ""
    )
    return ReceiverStateCondition(
        scene_epoch=ledger.scene_epoch,
        max_version=max_version,
        active_entities=tuple(active),
        negative_entities=tuple(negative),
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        snapshot_fingerprint=hashlib.sha256(canonical).hexdigest(),
    )
