from __future__ import annotations

from sgdjscc_lab.models.saver.contracts import AssertionAction, ProtectionAction, SemanticState
from sgdjscc_lab.transmission.saver_channel import SignedPacketChannel, SignedPacketFaultProfile
from sgdjscc_lab.transmission.saver_packet import (
    ApplyStatus,
    SignedStatePacket,
    VersionedEntityLedger,
    serialize_signed_packet,
)


def wire(version, action, state, *, protection=ProtectionAction.NONE):
    return serialize_signed_packet(SignedStatePacket(
        scene_epoch=0,
        entity_id="person:1",
        version=version,
        state=state,
        action=action,
        confidence=1.0,
        payload=b"state",
        protection=protection,
    ))


def test_forced_reorder_cannot_revive_revoked_entity_with_stale_assert():
    asserted = wire(1, AssertionAction.ASSERT, SemanticState.PRESENT)
    revoked = wire(2, AssertionAction.REVOKE, SemanticState.CONFIRMED_ABSENT)
    channel = SignedPacketChannel(SignedPacketFaultProfile())
    report = channel.transmit([asserted, revoked], forced_order=[1, 0])
    ledger = VersionedEntityLedger()
    statuses = channel.apply_report(report, ledger)
    assert statuses == (ApplyStatus.APPLIED, ApplyStatus.STALE_VERSION)
    assert ledger.entries["person:1"].version == 2
    assert ledger.entries["person:1"].state == SemanticState.CONFIRMED_ABSENT


def test_repetition_protection_and_duplicates_are_counted_not_hidden():
    repeated = wire(
        1, AssertionAction.ASSERT, SemanticState.PRESENT,
        protection=ProtectionAction.REPETITION,
    )
    channel = SignedPacketChannel(SignedPacketFaultProfile(duplicate_probability=1.0, seed=4))
    report = channel.transmit([repeated])
    assert len(report.attempts) == 3
    assert report.total_on_wire_bytes == 3 * len(repeated)
    statuses = channel.apply_report(report, VersionedEntityLedger())
    assert statuses.count(ApplyStatus.APPLIED) == 1
    assert statuses.count(ApplyStatus.DUPLICATE) == 2


def test_loss_and_corruption_are_explicit_and_reproducible():
    wires = [wire(i, AssertionAction.UPDATE, SemanticState.PRESENT) for i in range(1, 8)]
    profile = SignedPacketFaultProfile(
        loss_probability=0.25,
        corruption_probability=0.5,
        reorder_window=3,
        seed=99,
    )
    first = SignedPacketChannel(profile).transmit(wires)
    second = SignedPacketChannel(profile).transmit(wires)
    assert first == second
    assert first.total_on_wire_bytes == sum(len(item) for item in wires)
    statuses = SignedPacketChannel.apply_report(first, VersionedEntityLedger())
    assert len(statuses) == len(first.delivered)
    assert any(status == ApplyStatus.CORRUPT for status in statuses)


def test_complete_loss_has_zero_delivery_but_nonzero_on_wire_rate():
    value = wire(1, AssertionAction.ASSERT, SemanticState.PRESENT)
    report = SignedPacketChannel(SignedPacketFaultProfile(loss_probability=1.0)).transmit([value])
    assert report.delivered == ()
    assert report.delivered_bytes == 0
    assert report.total_on_wire_bytes == len(value)

