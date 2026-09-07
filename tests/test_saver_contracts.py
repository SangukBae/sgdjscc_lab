from __future__ import annotations

import dataclasses

import pytest

from sgdjscc_lab.data.saver_states import build_entity_slot_targets
from sgdjscc_lab.models.saver.contracts import (
    AssertionAction,
    EntityObservation,
    SaverCheckpointContract,
    SemanticState,
    SignedEntityEvent,
    stable_entity_numeric_id,
)


def obs(entity_id, state, confidence=1.0):
    return EntityObservation(
        entity_id=entity_id,
        state=state,
        confidence=confidence,
        provenance={"source": "official_gt"},
    )


def test_missing_observation_becomes_unknown_suspend_not_revoke():
    targets = build_entity_slot_targets(
        [obs("person:7", SemanticState.PRESENT)],
        [],
        scene_epoch=3,
        previous_versions={"person:7": 8},
        slot_count=2,
    )
    assert targets.states[0] == SemanticState.UNKNOWN
    assert targets.actions[0] == AssertionAction.SUSPEND
    assert targets.events[0].version == 9
    assert targets.events[0].provenance["confirmed_absent"] is False


def test_confirmed_absence_revokes_previous_present_and_resume_is_explicit():
    absent = build_entity_slot_targets(
        [obs("dog:2", SemanticState.PRESENT)],
        [obs("dog:2", SemanticState.CONFIRMED_ABSENT)],
        scene_epoch=0,
        slot_count=1,
    )
    assert absent.actions == (AssertionAction.REVOKE,)

    resumed = build_entity_slot_targets(
        [obs("dog:2", SemanticState.CONFIRMED_ABSENT)],
        [obs("dog:2", SemanticState.PRESENT)],
        scene_epoch=0,
        slot_count=1,
    )
    assert resumed.actions == (AssertionAction.RESUME,)


def test_slots_keep_previous_order_and_pad_deterministically():
    targets = build_entity_slot_targets(
        [obs("z", SemanticState.PRESENT), obs("a", SemanticState.PRESENT)],
        [obs("b", SemanticState.PRESENT), obs("a", SemanticState.PRESENT)],
        scene_epoch=0,
        slot_count=4,
    )
    assert targets.entity_ids == ("z", "a", "b", "")
    assert targets.valid_mask == (True, True, True, False)
    assert targets.actions[-1] == AssertionAction.SKIP


def test_scene_reset_is_scene_level_event():
    targets = build_entity_slot_targets([], [], scene_epoch=5, slot_count=1, scene_reset=True)
    event = targets.events[0]
    assert event.action == AssertionAction.SCENE_RESET
    assert event.entity_id == ""


def test_unknown_revoke_is_structurally_rejected():
    with pytest.raises(ValueError, match="UNKNOWN.*REVOKE"):
        SignedEntityEvent(
            scene_epoch=0,
            entity_id="cat:1",
            version=1,
            state=SemanticState.UNKNOWN,
            action=AssertionAction.REVOKE,
        )


def test_checkpoint_contract_fails_closed_on_architecture_mismatch():
    expected = SaverCheckpointContract(slot_count=16, memory_dim=64, injection_layers=(1, 3))
    expected.assert_compatible(expected.to_dict())
    changed = dataclasses.replace(expected, slot_count=8).to_dict()
    with pytest.raises(RuntimeError, match="incompatible SAVER checkpoint"):
        expected.assert_compatible(changed)


def test_target_builder_rejects_implicit_eviction():
    with pytest.raises(ValueError, match="explicit eviction policy"):
        build_entity_slot_targets(
            [],
            [obs("a", SemanticState.PRESENT), obs("b", SemanticState.PRESENT)],
            scene_epoch=0,
            slot_count=1,
        )


def test_entity_numeric_id_is_stable_and_epoch_scoped():
    assert stable_entity_numeric_id(2, "person:7") == stable_entity_numeric_id(2, "person:7")
    assert stable_entity_numeric_id(2, "person:7") != stable_entity_numeric_id(3, "person:7")
    assert 0 <= stable_entity_numeric_id(2, "person:7") < 2**63
