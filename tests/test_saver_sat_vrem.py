from __future__ import annotations

import pytest
import torch

from sgdjscc_lab.models.saver import (
    AssertionAction,
    RenderStatus,
    SemanticState,
    SignedAssertionTokenizer,
    VersionedRevocableEntityMemory,
)


def test_sat_shapes_masks_unknown_revoke_and_backpropagates():
    torch.manual_seed(3)
    sat = SignedAssertionTokenizer(input_dim=12, memory_dim=16, slot_count=4, num_heads=4)
    source = torch.randn(2, 7, 12, requires_grad=True)
    previous = torch.randn(2, 4, 16)
    valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    evidence = torch.tensor([
        [SemanticState.PRESENT, SemanticState.UNKNOWN, SemanticState.UNKNOWN, SemanticState.UNKNOWN],
        [SemanticState.CONFIRMED_ABSENT, SemanticState.PRESENT, SemanticState.UNKNOWN, SemanticState.UNKNOWN],
    ])
    out = sat(source, previous, slot_mask=valid, evidence_states=evidence)
    assert out.slots.shape == (2, 4, 16)
    assert out.state_logits.shape == (2, 4, 3)
    assert out.action_logits.shape == (2, 4, 6)
    assert out.confidence.shape == (2, 4)
    assert torch.isneginf(out.action_logits[0, 1, AssertionAction.REVOKE]) or (
        out.action_logits[0, 1, AssertionAction.REVOKE] < -1e20
    )
    assert out.action_probabilities[0, 2, AssertionAction.SKIP] == 1
    assert out.confidence[0, 2] == 0
    (out.slots.square().mean() + out.state_logits.square().mean()).backward()
    assert source.grad is not None and source.grad.abs().sum() > 0
    assert sat.slot_queries.grad is not None


def _event_tensors(action, state, version, *, entity_id=7, slots=2):
    return {
        "entity_ids": torch.tensor([[entity_id] + [-1] * (slots - 1)]),
        "semantic_states": torch.tensor([[int(state)] + [int(SemanticState.UNKNOWN)] * (slots - 1)]),
        "operations": torch.tensor([[int(action)] + [int(AssertionAction.SKIP)] * (slots - 1)]),
        "versions": torch.tensor([[version] + [-1] * (slots - 1)]),
        "valid_mask": torch.tensor([[True] + [False] * (slots - 1)]),
    }


def test_vrem_keeps_identity_on_revoke_and_rejects_late_assert():
    torch.manual_seed(4)
    memory = VersionedRevocableEntityMemory(memory_dim=8, slot_count=2)
    initial = memory.initial_state(1, device="cpu")
    feature = torch.randn(1, 2, 8, requires_grad=True)
    asserted = memory(initial, feature, **_event_tensors(
        AssertionAction.ASSERT, SemanticState.PRESENT, 4
    ))
    state = asserted.state
    identity_before = state.identity.clone()
    assert asserted.accepted_mask[0, 0]
    assert state.active_mask[0, 0]

    revoked = memory(state, feature, **_event_tensors(
        AssertionAction.REVOKE, SemanticState.CONFIRMED_ABSENT, 5
    ))
    state = revoked.state
    assert torch.equal(state.identity, identity_before)
    assert state.negative_mask[0, 0]
    assert state.render_status[0, 0] == int(RenderStatus.REVOKED)
    assert torch.count_nonzero(state.render[0, 0]) == 0

    late = memory(state, feature, **_event_tensors(
        AssertionAction.ASSERT, SemanticState.PRESENT, 4
    ))
    assert not late.accepted_mask[0, 0]
    assert late.stale_mask[0, 0]
    assert late.state.render_status[0, 0] == int(RenderStatus.REVOKED)
    assert late.state.versions[0, 0] == 5

    late.state.negative.sum().backward()
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert memory.negative_projection[1].weight.grad is not None


def test_vrem_suspend_preserves_identity_and_resume_reactivates():
    memory = VersionedRevocableEntityMemory(memory_dim=4, slot_count=1)
    feature = torch.randn(1, 1, 4)
    state = memory(
        memory.initial_state(1, device="cpu"),
        feature,
        **_event_tensors(AssertionAction.ASSERT, SemanticState.PRESENT, 0, slots=1),
    ).state
    identity = state.identity.clone()
    state = memory(
        state,
        feature,
        **_event_tensors(AssertionAction.SUSPEND, SemanticState.UNKNOWN, 1, slots=1),
    ).state
    assert torch.equal(state.identity, identity)
    assert state.render_status.item() == int(RenderStatus.SUSPENDED)
    state = memory(
        state,
        feature,
        **_event_tensors(AssertionAction.RESUME, SemanticState.PRESENT, 2, slots=1),
    ).state
    assert state.active_mask.item()
    assert torch.count_nonzero(state.negative) == 0


def test_vrem_requires_explicit_scene_reset_and_blocks_identity_swap():
    memory = VersionedRevocableEntityMemory(memory_dim=4, slot_count=1)
    feature = torch.randn(1, 1, 4)
    state = memory.initial_state(1, device="cpu")
    event = _event_tensors(AssertionAction.ASSERT, SemanticState.PRESENT, 0, slots=1)
    future = memory(state, feature, scene_epochs=torch.tensor([1]), **event)
    assert not future.accepted_mask.item()
    reset = memory(
        state,
        feature,
        scene_epochs=torch.tensor([1]),
        scene_reset_mask=torch.tensor([True]),
        **event,
    )
    assert reset.accepted_mask.item()
    assert reset.state.scene_epoch.item() == 1

    swapped = _event_tensors(
        AssertionAction.UPDATE, SemanticState.PRESENT, 1, entity_id=8, slots=1
    )
    with pytest.raises(ValueError, match="identity swap"):
        memory(reset.state, feature, scene_epochs=torch.tensor([1]), **swapped)


def test_vrem_rejects_unknown_revoke():
    memory = VersionedRevocableEntityMemory(memory_dim=4, slot_count=1)
    with pytest.raises(ValueError, match="UNKNOWN.*REVOKE"):
        memory(
            memory.initial_state(1, device="cpu"),
            torch.randn(1, 1, 4),
            **_event_tensors(AssertionAction.REVOKE, SemanticState.UNKNOWN, 0, slots=1),
        )

