"""Typed contracts shared by the SAVER-JSCC transmitter and receiver.

The enums in this module are part of the on-wire/checkpoint ABI.  Their
integer values must therefore never be reordered in-place: introduce a new
schema version if the vocabulary changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping, Optional, Tuple


SAVER_ARCHITECTURE_VERSION = "saver-jscc-v1"
SIGNED_PACKET_SCHEMA_VERSION = 1
DEFAULT_SLOT_COUNT = 16
DEFAULT_MEMORY_DIM = 512


class SemanticState(IntEnum):
    """Mutually exclusive evidence state for an entity slot."""

    PRESENT = 0
    CONFIRMED_ABSENT = 1
    UNKNOWN = 2


class AssertionAction(IntEnum):
    """State-transition operation carried by a signed assertion."""

    SKIP = 0
    ASSERT = 1
    UPDATE = 2
    SUSPEND = 3
    RESUME = 4
    REVOKE = 5
    SCENE_RESET = 6


ENTITY_ACTION_COUNT = 6  # SCENE_RESET is a scene-level packet, not a slot head.


class RenderStatus(IntEnum):
    """Receiver-side rendering eligibility, separate from identity memory."""

    EMPTY = 0
    ACTIVE = 1
    SUSPENDED = 2
    REVOKED = 3
    ABSENT = 4


class ProtectionAction(IntEnum):
    """Optional packet-protection action selected by JASR."""

    NONE = 0
    REPETITION = 1
    PIGGYBACK = 2
    SNAPSHOT = 3


@dataclass(frozen=True)
class EntityObservation:
    """One labelled entity observation used to construct training targets.

    ``CONFIRMED_ABSENT`` must originate from an authoritative annotation or a
    separately calibrated label source.  A detector miss is represented as
    ``UNKNOWN`` and cannot be silently promoted to absence.
    """

    entity_id: str
    state: SemanticState
    confidence: float = 1.0
    feature: Tuple[float, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id must be non-empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        object.__setattr__(self, "state", SemanticState(self.state))
        object.__setattr__(self, "feature", tuple(float(x) for x in self.feature))


@dataclass(frozen=True)
class SignedEntityEvent:
    """Versioned semantic event before neural tokenization/serialization."""

    scene_epoch: int
    entity_id: str
    version: int
    state: SemanticState
    action: AssertionAction
    confidence: float = 1.0
    feature: Tuple[float, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", SemanticState(self.state))
        object.__setattr__(self, "action", AssertionAction(self.action))
        object.__setattr__(self, "feature", tuple(float(x) for x in self.feature))
        if self.scene_epoch < 0 or self.version < 0:
            raise ValueError("scene_epoch and version must be non-negative")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if self.action == AssertionAction.SCENE_RESET:
            if self.entity_id:
                raise ValueError("SCENE_RESET must use an empty entity_id")
        elif not self.entity_id:
            raise ValueError("entity action requires a non-empty entity_id")
        if self.state == SemanticState.UNKNOWN and self.action == AssertionAction.REVOKE:
            raise ValueError("UNKNOWN evidence cannot issue REVOKE")


@dataclass(frozen=True)
class SaverCheckpointContract:
    """Architecture fields that must match when restoring a SAVER checkpoint."""

    architecture_version: str = SAVER_ARCHITECTURE_VERSION
    packet_schema_version: int = SIGNED_PACKET_SCHEMA_VERSION
    slot_count: int = DEFAULT_SLOT_COUNT
    memory_dim: int = DEFAULT_MEMORY_DIM
    injection_layers: Tuple[int, ...] = (1, 2, 3)
    action_vocabulary: Tuple[str, ...] = tuple(a.name for a in AssertionAction)

    def __post_init__(self) -> None:
        if self.slot_count <= 0 or self.memory_dim <= 0:
            raise ValueError("slot_count and memory_dim must be positive")
        if len(set(self.injection_layers)) != len(self.injection_layers):
            raise ValueError("injection_layers must be unique")
        if tuple(sorted(self.injection_layers)) != self.injection_layers:
            raise ValueError("injection_layers must be sorted")

    def to_dict(self) -> dict:
        return {
            "architecture_version": self.architecture_version,
            "packet_schema_version": self.packet_schema_version,
            "slot_count": self.slot_count,
            "memory_dim": self.memory_dim,
            "injection_layers": list(self.injection_layers),
            "action_vocabulary": list(self.action_vocabulary),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SaverCheckpointContract":
        return cls(
            architecture_version=str(value["architecture_version"]),
            packet_schema_version=int(value["packet_schema_version"]),
            slot_count=int(value["slot_count"]),
            memory_dim=int(value["memory_dim"]),
            injection_layers=tuple(int(x) for x in value["injection_layers"]),
            action_vocabulary=tuple(str(x) for x in value["action_vocabulary"]),
        )

    def assert_compatible(self, saved: Mapping[str, Any]) -> None:
        other = SaverCheckpointContract.from_dict(saved)
        if self != other:
            raise RuntimeError(
                "incompatible SAVER checkpoint contract: "
                f"expected={self.to_dict()} saved={other.to_dict()}"
            )


def infer_action(
    previous: Optional[SemanticState], current: SemanticState
) -> AssertionAction:
    """Derive a conservative event without turning missing evidence into absence."""

    current = SemanticState(current)
    previous = None if previous is None else SemanticState(previous)
    if current == SemanticState.UNKNOWN:
        return AssertionAction.SUSPEND if previous == SemanticState.PRESENT else AssertionAction.SKIP
    if current == SemanticState.CONFIRMED_ABSENT:
        return (
            AssertionAction.REVOKE
            if previous == SemanticState.PRESENT
            else AssertionAction.ASSERT
        )
    if previous is None or previous == SemanticState.UNKNOWN:
        return AssertionAction.ASSERT
    if previous == SemanticState.CONFIRMED_ABSENT:
        return AssertionAction.RESUME
    return AssertionAction.UPDATE

