"""Deterministic signed-state wire packets and Tx/Rx version synchronization.

This module intentionally keeps exact digital bytes separate from wireless
complex-symbol accounting.  ``serialize_signed_packet`` and
``serialize_state_ack`` return the actual byte strings placed on their
respective links, including headers and CRC32 checksums.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

from sgdjscc_lab.models.saver.contracts import (
    AssertionAction,
    ProtectionAction,
    RenderStatus,
    SIGNED_PACKET_SCHEMA_VERSION,
    SemanticState,
)


SIGNED_PACKET_MAGIC = b"SVRS"
STATE_ACK_MAGIC = b"SVRA"
NO_ENTITY_VERSION = 0xFFFFFFFF

# magic, schema, state, action, protection, flags, scene_epoch, version,
# entity_id_bytes, confidence, payload_bytes
_PACKET_HEADER_FMT = ">4sBBBBBIIHfI"
_PACKET_HEADER_SIZE = struct.calcsize(_PACKET_HEADER_FMT)

# magic, schema, status, scene_epoch, version, entity_id_bytes
_ACK_HEADER_FMT = ">4sBBIIH"
_ACK_HEADER_SIZE = struct.calcsize(_ACK_HEADER_FMT)


class SignedPacketError(ValueError):
    """Base class for signed-state packet failures."""


class SignedPacketMagicError(SignedPacketError):
    pass


class SignedPacketVersionError(SignedPacketError):
    pass


class SignedPacketLengthError(SignedPacketError):
    pass


class SignedPacketChecksumError(SignedPacketError):
    pass


class FeedbackMode(IntEnum):
    NONE = 0
    ACK = 1


class ApplyStatus(IntEnum):
    APPLIED = 0
    DUPLICATE = 1
    STALE_VERSION = 2
    STALE_EPOCH = 3
    FUTURE_EPOCH_REQUIRES_RESET = 4
    LOST = 5
    CORRUPT = 6


_FLAG_ACK_REQUESTED = 1 << 0


@dataclass(frozen=True)
class SignedStatePacket:
    scene_epoch: int
    entity_id: str
    version: int
    state: SemanticState
    action: AssertionAction
    confidence: float
    payload: bytes = b""
    protection: ProtectionAction = ProtectionAction.NONE
    ack_requested: bool = False
    schema_version: int = SIGNED_PACKET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", SemanticState(self.state))
        object.__setattr__(self, "action", AssertionAction(self.action))
        object.__setattr__(self, "protection", ProtectionAction(self.protection))
        object.__setattr__(self, "payload", bytes(self.payload))
        if self.scene_epoch < 0 or self.version < 0:
            raise ValueError("scene_epoch and version must be non-negative")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        encoded_id = self.entity_id.encode("utf-8")
        if len(encoded_id) > 0xFFFF:
            raise ValueError("entity_id is too long for the packet schema")
        if self.action == AssertionAction.SCENE_RESET:
            if self.entity_id:
                raise ValueError("SCENE_RESET must have an empty entity_id")
        elif not self.entity_id:
            raise ValueError("entity action requires a non-empty entity_id")
        if self.state == SemanticState.UNKNOWN and self.action == AssertionAction.REVOKE:
            raise ValueError("UNKNOWN evidence cannot issue REVOKE")

    @property
    def exact_wire_bytes(self) -> int:
        return len(serialize_signed_packet(self))


@dataclass(frozen=True)
class StateAck:
    scene_epoch: int
    entity_id: str
    version: int
    status: ApplyStatus
    schema_version: int = SIGNED_PACKET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ApplyStatus(self.status))
        if self.scene_epoch < 0 or self.version < 0:
            raise ValueError("scene_epoch and version must be non-negative")
        if len(self.entity_id.encode("utf-8")) > 0xFFFF:
            raise ValueError("entity_id is too long for the ACK schema")

    @property
    def exact_wire_bytes(self) -> int:
        return len(serialize_state_ack(self))


@dataclass(frozen=True)
class LedgerEntry:
    scene_epoch: int
    entity_id: str
    version: int
    state: SemanticState
    render_status: RenderStatus
    confidence: float
    payload: bytes


@dataclass(frozen=True)
class LinkAccounting:
    forward_bytes: int = 0
    feedback_bytes: int = 0
    retransmission_bytes: int = 0

    @property
    def total_on_wire_bytes(self) -> int:
        return self.forward_bytes + self.feedback_bytes + self.retransmission_bytes


def _append_crc(data: bytes) -> bytes:
    return data + struct.pack(">I", zlib.crc32(data) & 0xFFFFFFFF)


def _verify_crc(data: bytes, minimum: int) -> bytes:
    if len(data) < minimum + 4:
        raise SignedPacketLengthError(f"packet too short: {len(data)} bytes")
    body = data[:-4]
    declared = struct.unpack(">I", data[-4:])[0]
    actual = zlib.crc32(body) & 0xFFFFFFFF
    if declared != actual:
        raise SignedPacketChecksumError(
            f"checksum mismatch: declared={declared:#010x} actual={actual:#010x}"
        )
    return body


def serialize_signed_packet(packet: SignedStatePacket) -> bytes:
    """Serialize one signed state operation deterministically."""

    entity = packet.entity_id.encode("utf-8")
    flags = _FLAG_ACK_REQUESTED if packet.ack_requested else 0
    header = struct.pack(
        _PACKET_HEADER_FMT,
        SIGNED_PACKET_MAGIC,
        packet.schema_version,
        int(packet.state),
        int(packet.action),
        int(packet.protection),
        flags,
        packet.scene_epoch,
        packet.version,
        len(entity),
        packet.confidence,
        len(packet.payload),
    )
    return _append_crc(header + entity + packet.payload)


def parse_signed_packet(data: bytes) -> SignedStatePacket:
    """Validate CRC/length/enums and reconstruct a fresh packet object."""

    body = _verify_crc(bytes(data), _PACKET_HEADER_SIZE)
    values = struct.unpack(_PACKET_HEADER_FMT, body[:_PACKET_HEADER_SIZE])
    (
        magic,
        schema,
        state,
        action,
        protection,
        flags,
        scene_epoch,
        version,
        entity_len,
        confidence,
        payload_len,
    ) = values
    if magic != SIGNED_PACKET_MAGIC:
        raise SignedPacketMagicError(f"bad magic {magic!r}")
    if schema != SIGNED_PACKET_SCHEMA_VERSION:
        raise SignedPacketVersionError(f"unsupported schema version={schema}")
    expected = _PACKET_HEADER_SIZE + entity_len + payload_len
    if len(body) != expected:
        raise SignedPacketLengthError(f"declared body size={expected}, actual={len(body)}")
    offset = _PACKET_HEADER_SIZE
    try:
        entity_id = body[offset:offset + entity_len].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SignedPacketError("entity_id is not valid UTF-8") from exc
    offset += entity_len
    try:
        return SignedStatePacket(
            scene_epoch=scene_epoch,
            entity_id=entity_id,
            version=version,
            state=SemanticState(state),
            action=AssertionAction(action),
            confidence=float(confidence),
            payload=body[offset:],
            protection=ProtectionAction(protection),
            ack_requested=bool(flags & _FLAG_ACK_REQUESTED),
            schema_version=schema,
        )
    except (ValueError, TypeError) as exc:
        raise SignedPacketError(f"invalid signed packet fields: {exc}") from exc


def serialize_state_ack(ack: StateAck) -> bytes:
    entity = ack.entity_id.encode("utf-8")
    header = struct.pack(
        _ACK_HEADER_FMT,
        STATE_ACK_MAGIC,
        ack.schema_version,
        int(ack.status),
        ack.scene_epoch,
        ack.version,
        len(entity),
    )
    return _append_crc(header + entity)


def parse_state_ack(data: bytes) -> StateAck:
    body = _verify_crc(bytes(data), _ACK_HEADER_SIZE)
    magic, schema, status, scene_epoch, version, entity_len = struct.unpack(
        _ACK_HEADER_FMT, body[:_ACK_HEADER_SIZE]
    )
    if magic != STATE_ACK_MAGIC:
        raise SignedPacketMagicError(f"bad ACK magic {magic!r}")
    if schema != SIGNED_PACKET_SCHEMA_VERSION:
        raise SignedPacketVersionError(f"unsupported ACK schema version={schema}")
    if len(body) != _ACK_HEADER_SIZE + entity_len:
        raise SignedPacketLengthError("ACK entity_id length mismatch")
    try:
        entity_id = body[_ACK_HEADER_SIZE:].decode("utf-8")
        return StateAck(scene_epoch, entity_id, version, ApplyStatus(status), schema)
    except (UnicodeDecodeError, ValueError) as exc:
        raise SignedPacketError(f"invalid ACK fields: {exc}") from exc


def _render_status(packet: SignedStatePacket) -> RenderStatus:
    if packet.action == AssertionAction.SUSPEND:
        return RenderStatus.SUSPENDED
    if packet.action == AssertionAction.REVOKE:
        return RenderStatus.REVOKED
    if packet.state == SemanticState.CONFIRMED_ABSENT:
        return RenderStatus.ABSENT
    if packet.state == SemanticState.PRESENT:
        return RenderStatus.ACTIVE
    return RenderStatus.SUSPENDED


class VersionedEntityLedger:
    """Deterministic receiver ledger enforcing epoch/version monotonicity."""

    def __init__(self, scene_epoch: int = 0) -> None:
        if scene_epoch < 0:
            raise ValueError("scene_epoch must be non-negative")
        self.scene_epoch = int(scene_epoch)
        self.entries: Dict[str, LedgerEntry] = {}

    def apply(self, packet: SignedStatePacket) -> ApplyStatus:
        if packet.action == AssertionAction.SCENE_RESET:
            if packet.scene_epoch < self.scene_epoch:
                return ApplyStatus.STALE_EPOCH
            self.scene_epoch = packet.scene_epoch
            self.entries.clear()
            return ApplyStatus.APPLIED
        if packet.scene_epoch < self.scene_epoch:
            return ApplyStatus.STALE_EPOCH
        if packet.scene_epoch > self.scene_epoch:
            return ApplyStatus.FUTURE_EPOCH_REQUIRES_RESET
        previous = self.entries.get(packet.entity_id)
        if previous is not None:
            if packet.version == previous.version:
                return ApplyStatus.DUPLICATE
            if packet.version < previous.version:
                return ApplyStatus.STALE_VERSION
        self.entries[packet.entity_id] = LedgerEntry(
            scene_epoch=packet.scene_epoch,
            entity_id=packet.entity_id,
            version=packet.version,
            state=packet.state,
            render_status=_render_status(packet),
            confidence=packet.confidence,
            payload=packet.payload,
        )
        return ApplyStatus.APPLIED


class VersionedStateSynchronizer:
    """Small transport boundary with explicit forward/feedback byte accounting.

    In ``NONE`` mode the transmitter maintains only a belief updated when it
    sends.  In ``ACK`` mode it updates its acknowledged state only after an
    ``APPLIED`` ACK is parsed.  The implementation never lets the transmitter
    directly inspect the receiver ledger.
    """

    def __init__(self, feedback_mode: FeedbackMode = FeedbackMode.NONE) -> None:
        self.feedback_mode = FeedbackMode(feedback_mode)
        self.receiver = VersionedEntityLedger()
        self.tx_belief: Dict[str, Tuple[int, int]] = {}
        self.tx_acknowledged: Dict[str, Tuple[int, int]] = {}
        self.accounting = LinkAccounting()

    def send(self, packet: SignedStatePacket, *, deliver: bool = True) -> Tuple[ApplyStatus, Optional[bytes]]:
        wire = serialize_signed_packet(packet)
        self.accounting = LinkAccounting(
            forward_bytes=self.accounting.forward_bytes + len(wire),
            feedback_bytes=self.accounting.feedback_bytes,
            retransmission_bytes=self.accounting.retransmission_bytes,
        )
        if packet.action != AssertionAction.SCENE_RESET:
            self.tx_belief[packet.entity_id] = (packet.scene_epoch, packet.version)
        if not deliver:
            return ApplyStatus.LOST, None
        status = self.receiver.apply(parse_signed_packet(wire))
        ack_wire: Optional[bytes] = None
        if self.feedback_mode == FeedbackMode.ACK and packet.ack_requested:
            ack_wire = serialize_state_ack(StateAck(
                scene_epoch=packet.scene_epoch,
                entity_id=packet.entity_id,
                version=packet.version,
                status=status,
            ))
            self.accounting = LinkAccounting(
                forward_bytes=self.accounting.forward_bytes,
                feedback_bytes=self.accounting.feedback_bytes + len(ack_wire),
                retransmission_bytes=self.accounting.retransmission_bytes,
            )
            self.accept_ack(ack_wire)
        return status, ack_wire

    def receive_wire(self, wire: Optional[bytes]) -> ApplyStatus:
        """Receiver-only entrypoint for loss/corruption simulations."""

        if wire is None:
            return ApplyStatus.LOST
        try:
            packet = parse_signed_packet(wire)
        except SignedPacketError:
            return ApplyStatus.CORRUPT
        return self.receiver.apply(packet)

    def accept_ack(self, wire: bytes) -> StateAck:
        ack = parse_state_ack(wire)
        if ack.status == ApplyStatus.APPLIED and ack.entity_id:
            current = self.tx_acknowledged.get(ack.entity_id)
            candidate = (ack.scene_epoch, ack.version)
            if current is None or candidate > current:
                self.tx_acknowledged[ack.entity_id] = candidate
        return ack

    def account_retransmission(self, packet: SignedStatePacket) -> int:
        """Record an explicitly scheduled retransmission without delivering it."""

        nbytes = len(serialize_signed_packet(packet))
        self.accounting = LinkAccounting(
            forward_bytes=self.accounting.forward_bytes,
            feedback_bytes=self.accounting.feedback_bytes,
            retransmission_bytes=self.accounting.retransmission_bytes + nbytes,
        )
        return nbytes

