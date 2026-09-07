from __future__ import annotations

import pytest

from sgdjscc_lab.models.saver.contracts import (
    AssertionAction,
    ProtectionAction,
    RenderStatus,
    SemanticState,
)
from sgdjscc_lab.transmission.saver_packet import (
    ApplyStatus,
    FeedbackMode,
    SignedPacketChecksumError,
    SignedStatePacket,
    StateAck,
    VersionedEntityLedger,
    VersionedStateSynchronizer,
    parse_signed_packet,
    parse_state_ack,
    serialize_signed_packet,
    serialize_state_ack,
)


def packet(version=1, action=AssertionAction.ASSERT, state=SemanticState.PRESENT, **kwargs):
    return SignedStatePacket(
        scene_epoch=kwargs.pop("scene_epoch", 0),
        entity_id=kwargs.pop("entity_id", "person:7"),
        version=version,
        state=state,
        action=action,
        confidence=kwargs.pop("confidence", 0.875),
        payload=kwargs.pop("payload", b"feature"),
        **kwargs,
    )


def test_signed_packet_round_trip_is_deterministic_and_byte_exact():
    original = packet(
        protection=ProtectionAction.PIGGYBACK,
        ack_requested=True,
        payload=bytes(range(64)),
    )
    wire1 = serialize_signed_packet(original)
    wire2 = serialize_signed_packet(original)
    assert wire1 == wire2
    decoded = parse_signed_packet(wire1)
    assert decoded == original
    assert original.exact_wire_bytes == len(wire1)


def test_signed_packet_corruption_fails_closed():
    wire = bytearray(serialize_signed_packet(packet()))
    wire[-6] ^= 0x01
    with pytest.raises(SignedPacketChecksumError):
        parse_signed_packet(bytes(wire))


def test_ack_round_trip_counts_actual_bytes():
    ack = StateAck(2, "car:3", 11, ApplyStatus.APPLIED)
    wire = serialize_state_ack(ack)
    assert parse_state_ack(wire) == ack
    assert ack.exact_wire_bytes == len(wire)


def test_revoke_tombstone_rejects_late_assert_without_losing_identity_record():
    ledger = VersionedEntityLedger()
    assert ledger.apply(packet(version=4)) == ApplyStatus.APPLIED
    revoke = packet(
        version=5,
        action=AssertionAction.REVOKE,
        state=SemanticState.CONFIRMED_ABSENT,
    )
    assert ledger.apply(revoke) == ApplyStatus.APPLIED
    assert ledger.entries["person:7"].render_status == RenderStatus.REVOKED
    assert ledger.apply(packet(version=4)) == ApplyStatus.STALE_VERSION
    assert ledger.entries["person:7"].version == 5
    assert ledger.entries["person:7"].render_status == RenderStatus.REVOKED


def test_scene_epoch_requires_reset_and_old_epoch_cannot_leak_back():
    ledger = VersionedEntityLedger(scene_epoch=2)
    future = packet(scene_epoch=3)
    assert ledger.apply(future) == ApplyStatus.FUTURE_EPOCH_REQUIRES_RESET
    reset = SignedStatePacket(
        scene_epoch=3,
        entity_id="",
        version=0,
        state=SemanticState.UNKNOWN,
        action=AssertionAction.SCENE_RESET,
        confidence=1.0,
    )
    assert ledger.apply(reset) == ApplyStatus.APPLIED
    assert ledger.apply(packet(scene_epoch=2)) == ApplyStatus.STALE_EPOCH


def test_duplicate_scene_reset_cannot_erase_new_epoch_entities():
    ledger = VersionedEntityLedger()
    reset = SignedStatePacket(
        scene_epoch=1,
        entity_id="",
        version=3,
        state=SemanticState.UNKNOWN,
        action=AssertionAction.SCENE_RESET,
        confidence=1.0,
    )
    assert ledger.apply(reset) == ApplyStatus.APPLIED
    assert ledger.apply(packet(scene_epoch=1, version=4)) == ApplyStatus.APPLIED
    assert ledger.apply(reset) == ApplyStatus.DUPLICATE
    assert "person:7" in ledger.entries


def test_loss_duplicate_corruption_and_ack_accounting_are_explicit():
    sync = VersionedStateSynchronizer(FeedbackMode.ACK)
    p = packet(ack_requested=True)
    status, ack_wire = sync.send(p)
    assert status == ApplyStatus.APPLIED
    assert ack_wire is not None
    assert sync.accounting.forward_bytes == p.exact_wire_bytes
    assert sync.accounting.feedback_bytes == len(ack_wire)
    assert sync.tx_acknowledged[p.entity_id] == (p.scene_epoch, p.version)

    assert sync.receive_wire(None) == ApplyStatus.LOST
    corrupt = bytearray(serialize_signed_packet(p))
    corrupt[-1] ^= 1
    assert sync.receive_wire(bytes(corrupt)) == ApplyStatus.CORRUPT
    assert sync.receiver.apply(p) == ApplyStatus.DUPLICATE

    retransmitted = sync.account_retransmission(p)
    assert retransmitted == p.exact_wire_bytes
    assert sync.accounting.total_on_wire_bytes == (
        sync.accounting.forward_bytes
        + sync.accounting.feedback_bytes
        + sync.accounting.retransmission_bytes
    )


def test_unknown_revoke_rejected_before_serialization():
    with pytest.raises(ValueError, match="UNKNOWN.*REVOKE"):
        packet(action=AssertionAction.REVOKE, state=SemanticState.UNKNOWN)
