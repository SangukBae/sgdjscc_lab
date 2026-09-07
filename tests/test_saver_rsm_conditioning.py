from __future__ import annotations

from sgdjscc_lab.models.saver.contracts import AssertionAction, SemanticState
from sgdjscc_lab.models.saver.rsm_conditioning import (
    SAVER_RECEIVER_CONDITION_SCHEMA,
    compile_receiver_state_condition,
)
from sgdjscc_lab.transmission.saver_packet import SignedStatePacket, VersionedEntityLedger


def _packet(entity_id, version, state, action):
    return SignedStatePacket(
        scene_epoch=0,
        entity_id=entity_id,
        version=version,
        state=state,
        action=action,
        confidence=0.9,
    )


def test_receiver_condition_uses_applied_active_and_revoked_state_only():
    ledger = VersionedEntityLedger()
    ledger.apply(_packet("car-1", 1, SemanticState.PRESENT, AssertionAction.ASSERT))
    ledger.apply(_packet("person-2", 1, SemanticState.UNKNOWN, AssertionAction.SUSPEND))
    ledger.apply(_packet(
        "dog-3", 1, SemanticState.CONFIRMED_ABSENT, AssertionAction.REVOKE
    ))

    condition = compile_receiver_state_condition(
        ledger, entity_labels={"car-1": "red car", "person-2": "person", "dog-3": "dog"}
    )

    assert condition.active_entities == ("red car",)
    assert condition.negative_entities == ("dog",)
    assert "person" not in condition.positive_prompt
    assert "person" not in condition.negative_prompt
    assert condition.to_side_info()["schema"] == SAVER_RECEIVER_CONDITION_SCHEMA


def test_receiver_condition_fingerprint_is_order_independent_and_version_sensitive():
    first = VersionedEntityLedger()
    second = VersionedEntityLedger()
    packets = [
        _packet("b", 1, SemanticState.PRESENT, AssertionAction.ASSERT),
        _packet("a", 1, SemanticState.PRESENT, AssertionAction.ASSERT),
    ]
    for packet in packets:
        first.apply(packet)
    for packet in reversed(packets):
        second.apply(packet)
    assert (
        compile_receiver_state_condition(first).snapshot_fingerprint
        == compile_receiver_state_condition(second).snapshot_fingerprint
    )
    second.apply(_packet("a", 2, SemanticState.PRESENT, AssertionAction.UPDATE))
    assert (
        compile_receiver_state_condition(first).snapshot_fingerprint
        != compile_receiver_state_condition(second).snapshot_fingerprint
    )
