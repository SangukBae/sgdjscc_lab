from __future__ import annotations

import torch

from sgdjscc_lab.models.saver import (
    ActionConditionedSemanticChannelCodec,
    AssertionAction,
    JointAssertionSymbolRouter,
    SemanticState,
    SignedMemoryDiTAdapterStack,
    SignedMemoryDiTBlock,
)


def test_router_hard_forward_respects_budget_and_unknown_mask():
    torch.manual_seed(5)
    router = JointAssertionSymbolRouter(
        token_dim=16,
        channel_dim=4,
        rate_options=(0, 4, 8, 16),
        visual_rate_options=(0, 8, 16),
        protection_extra_symbols=(0, 2, 1, 4),
        hidden_dim=24,
    ).eval()
    events = torch.randn(2, 5, 16, requires_grad=True)
    visual = torch.randn(2, 7, 16)
    channel = torch.randn(2, 4)
    budget = torch.tensor([12, 25])
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    states = torch.tensor([
        [0, 2, 1, 2, 2],
        [0, 0, 2, 1, 0],
    ])
    out = router(events, visual, channel, budget, valid_mask=valid, semantic_states=states)
    assert torch.all(out.hard_total_symbols <= budget)
    assert not out.selected_mask[0, 3:].any()
    assert out.action_weights[0, 1, AssertionAction.REVOKE] == 0
    assert torch.all(out.hard_actions[~valid] == int(AssertionAction.SKIP))
    assert torch.all(out.hard_rate_symbols[~valid] == 0)
    (out.soft_total_symbols.mean() + out.action_logits[valid].mean()).backward()
    assert events.grad is not None and events.grad.abs().sum() > 0


def test_router_zero_budget_selects_nothing():
    router = JointAssertionSymbolRouter(
        token_dim=8,
        channel_dim=2,
        rate_options=(0, 4),
        visual_rate_options=(0, 8),
        hidden_dim=16,
    ).eval()
    out = router(torch.randn(1, 2, 8), torch.randn(1, 3, 8), torch.randn(1, 2), 0)
    assert out.hard_total_symbols.item() == 0
    assert out.hard_visual_symbols.item() == 0
    assert not out.selected_mask.any()


def test_wireless_codec_counts_complex_uses_and_has_end_to_end_gradient():
    torch.manual_seed(6)
    codec = ActionConditionedSemanticChannelCodec(token_dim=12, max_symbols=8, hidden_dim=32)
    tokens = torch.randn(2, 3, 12, requires_grad=True)
    actions = torch.tensor([[1, 2, 0], [5, 1, 3]])
    states = torch.tensor([[0, 0, 2], [1, 0, 2]])
    allocation = torch.tensor([[8, 3, 0], [4, 1, 6]])
    encoded = codec.encode_wireless(tokens, actions, states, allocation)
    assert torch.equal(encoded.actual_complex_channel_uses, allocation.sum(dim=-1))
    generator = torch.Generator().manual_seed(17)
    observed = codec.transmit_wireless(
        encoded, snr_db=10.0, channel="rayleigh", packet_loss_probability=0.0,
        generator=generator,
    )
    decoded = codec.decode_wireless(observed, actions, states)
    assert decoded.shape == tokens.shape
    assert torch.count_nonzero(decoded[0, 2]) == 0
    decoded.square().mean().backward()
    assert tokens.grad is not None and tokens.grad.abs().sum() > 0
    assert codec.encoder[0].weight.grad is not None


def test_wireless_packet_loss_is_explicit_not_hidden():
    codec = ActionConditionedSemanticChannelCodec(token_dim=4, max_symbols=3, hidden_dim=8)
    encoded = codec.encode_wireless(
        torch.randn(1, 2, 4),
        torch.tensor([[1, 1]]),
        torch.tensor([[0, 0]]),
        torch.tensor([[3, 2]]),
    )
    observed = codec.transmit_wireless(encoded, snr_db=5.0, packet_loss_probability=1.0)
    assert not observed.symbol_mask.any()
    assert torch.count_nonzero(observed.iq) == 0
    assert torch.count_nonzero(observed.reliability) == 0


def test_source_codec_round_trip_uses_actual_serialized_bytes():
    feature = torch.linspace(-1, 1, 16)
    wire = ActionConditionedSemanticChannelCodec.encode_source_packet(
        feature,
        scene_epoch=2,
        entity_id="bicycle:4",
        version=9,
        state=SemanticState.CONFIRMED_ABSENT,
        action=AssertionAction.REVOKE,
        confidence=0.95,
        bit_depth=4,
    )
    packet, decoded = ActionConditionedSemanticChannelCodec.decode_source_packet(wire)
    assert packet.entity_id == "bicycle:4"
    assert packet.version == 9
    assert packet.action == AssertionAction.REVOKE
    assert decoded.shape == feature.shape
    assert torch.max(torch.abs(decoded - feature)) <= 2.0 / 15.0 + 1e-6
    assert len(wire) == packet.exact_wire_bytes


def _dit_inputs():
    return {
        "hidden_states": torch.randn(2, 5, 24),
        "timestep_embedding": torch.randn(2, 24),
        "channel_tokens": torch.randn(2, 4, 8),
        "active_memory": torch.randn(2, 3, 16),
        "negative_memory": torch.randn(2, 3, 16),
        "active_mask": torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool),
        "negative_mask": torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool),
        "channel_reliability": torch.tensor([0.9, 0.4]),
        "assertion_confidence": torch.tensor([0.8, 0.7]),
    }


def test_signed_memory_dit_is_exact_identity_at_zero_init_then_uses_negative_bank():
    torch.manual_seed(7)
    block = SignedMemoryDiTBlock(
        hidden_dim=24, memory_dim=16, channel_dim=8, bottleneck_dim=16, num_heads=4
    )
    inputs = _dit_inputs()
    initial = block(**inputs)
    assert torch.equal(initial.hidden_states, inputs["hidden_states"])
    assert torch.count_nonzero(initial.negative_residual) > 0
    assert torch.count_nonzero(initial.positive_residual[1]) == 0

    with torch.no_grad():
        block.output_projection.weight.fill_(0.01)
    full = block(**inputs).hidden_states
    no_negative = block(**{**inputs, "negative_mask": torch.zeros(2, 3, dtype=torch.bool)}).hidden_states
    assert not torch.allclose(full, no_negative)


def test_signed_memory_adapter_stack_only_changes_registered_layers():
    stack = SignedMemoryDiTAdapterStack(
        (1, 3), hidden_dim=24, memory_dim=16, channel_dim=8,
        bottleneck_dim=16, num_heads=4,
    )
    inputs = _dit_inputs()
    bypass = stack.inject(2, **inputs)
    assert torch.equal(bypass.hidden_states, inputs["hidden_states"])
    assert stack.has_layer(1) and not stack.has_layer(2)
    injected = stack.inject(1, **inputs)
    assert injected.hidden_states.shape == inputs["hidden_states"].shape

