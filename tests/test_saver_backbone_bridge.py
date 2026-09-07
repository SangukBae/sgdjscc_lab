from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sgdjscc_lab.models.saver.backbone_bridge import (
    SignedMemoryBackboneBridge,
    SignedMemoryCondition,
)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, hidden):
        return hidden + self.linear(hidden)


def _condition(batch=2, hidden=16, memory=12):
    return SignedMemoryCondition(
        timestep_embedding=torch.randn(batch, hidden),
        channel_tokens=torch.randn(batch, 4, 8),
        active_memory=torch.randn(batch, 3, memory),
        negative_memory=torch.randn(batch, 3, memory),
        active_mask=torch.tensor([[True, False, False], [True, True, False]]),
        negative_mask=torch.tensor([[False, True, False], [False, False, True]]),
        channel_reliability=torch.tensor([0.9, 0.7]),
        assertion_confidence=torch.tensor([0.8, 0.6]),
    )


def test_bridge_is_noop_outside_context_and_zero_initialized_inside_context():
    torch.manual_seed(2)
    blocks = nn.ModuleList([_ResidualBlock(16) for _ in range(4)])
    bridge = SignedMemoryBackboneBridge(
        [1, 3], hidden_dim=16, memory_dim=12, bottleneck_dim=16, num_heads=4
    )
    hidden = torch.randn(2, 5, 16)
    baseline = nn.Sequential(*blocks)(hidden)
    bridge.attach(blocks)
    assert torch.equal(nn.Sequential(*blocks)(hidden), baseline)
    with bridge.condition(_condition()):
        assert torch.equal(nn.Sequential(*blocks)(hidden), baseline)
    bridge.detach()
    assert not bridge.attached


def test_bridge_changes_internal_block_output_after_adapter_learns():
    torch.manual_seed(3)
    blocks = nn.ModuleList([_ResidualBlock(16) for _ in range(3)])
    bridge = SignedMemoryBackboneBridge(
        [1], hidden_dim=16, memory_dim=12, bottleneck_dim=16, num_heads=4
    )
    bridge.attach(blocks)
    hidden = torch.randn(2, 5, 16)
    baseline = nn.Sequential(*blocks)(hidden)
    nn.init.normal_(bridge.adapters.adapters["1"].output_projection.weight, std=0.01)
    with bridge.condition(_condition()):
        changed = nn.Sequential(*blocks)(hidden)
    assert not torch.equal(changed, baseline)
    changed.sum().backward()
    assert bridge.adapters.adapters["1"].output_projection.weight.grad is not None


def test_bridge_rejects_out_of_range_layer_and_nested_activation_restores_context():
    blocks = nn.ModuleList([_ResidualBlock(16)])
    bridge = SignedMemoryBackboneBridge(
        [1], hidden_dim=16, memory_dim=12, bottleneck_dim=16, num_heads=4
    )
    with pytest.raises(ValueError, match="outside"):
        bridge.attach(blocks)
