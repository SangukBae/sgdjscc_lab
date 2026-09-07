from __future__ import annotations

import copy

import pytest
import torch
from omegaconf import OmegaConf

from sgdjscc_lab.models.saver import (
    ActionConditionedSemanticChannelCodec,
    AssertionAction,
    JointAssertionSymbolRouter,
    SaverCheckpointContract,
    SemanticState,
    SignedAssertionTokenizer,
    SignedMemoryDiTAdapterStack,
    VersionedRevocableEntityMemory,
)
from sgdjscc_lab.models.saver.factory import build_saver_video_pipeline, checkpoint_contract_from_config
from sgdjscc_lab.pipelines.saver_video_pipeline import SaverVideoPipeline
from sgdjscc_lab.training.saver_losses import SaverLoss, SaverLossWeights
from sgdjscc_lab.training.saver_stage_runner import SaverStageRunner


def small_pipeline(enabled=True):
    return SaverVideoPipeline(
        SignedAssertionTokenizer(6, memory_dim=8, slot_count=3, num_heads=2),
        JointAssertionSymbolRouter(
            8, 3, rate_options=(0, 2, 4), visual_rate_options=(0, 4, 8),
            protection_extra_symbols=(0, 2, 1, 4), hidden_dim=12,
        ),
        ActionConditionedSemanticChannelCodec(8, max_symbols=4, hidden_dim=16),
        VersionedRevocableEntityMemory(memory_dim=8, slot_count=3),
        SignedMemoryDiTAdapterStack(
            (1, 3), hidden_dim=12, memory_dim=8, channel_dim=4,
            timestep_dim=12, bottleneck_dim=8, num_heads=2,
        ),
        enabled=enabled,
    )


def inputs(pipeline):
    generator = torch.Generator().manual_seed(23)
    return {
        "source_features": torch.randn(1, 5, 6),
        "visual_tokens": torch.randn(1, 4, 8),
        "diffusion_hidden": torch.randn(1, 7, 12),
        "timestep_embedding": torch.randn(1, 12),
        "channel_features": torch.randn(1, 3),
        "channel_tokens": torch.randn(1, 4, 4),
        "memory_state": pipeline.memory.initial_state(1, device="cpu"),
        "entity_ids": torch.tensor([[1, 2, -1]]),
        "semantic_states": torch.tensor([[
            int(SemanticState.PRESENT),
            int(SemanticState.CONFIRMED_ABSENT),
            int(SemanticState.UNKNOWN),
        ]]),
        "versions": torch.tensor([[0, 0, -1]]),
        "scene_epochs": torch.tensor([0]),
        "total_budget": 32,
        "valid_mask": torch.tensor([[True, True, False]]),
        "teacher_actions": torch.tensor([[
            int(AssertionAction.ASSERT),
            int(AssertionAction.REVOKE),
            int(AssertionAction.SKIP),
        ]]),
        "teacher_allocated_symbols": torch.tensor([[4, 4, 0]]),
        "snr_db": 30.0,
        "generator": generator,
    }


def test_master_gate_off_is_strict_identity_and_does_not_mutate_memory():
    pipeline = small_pipeline(enabled=False)
    kwargs = inputs(pipeline)
    hidden = kwargs["diffusion_hidden"].clone()
    state = kwargs["memory_state"]
    out = pipeline(**kwargs)
    assert not out.enabled
    assert torch.equal(out.diffusion_hidden, hidden)
    assert out.memory_state is state
    assert out.assertion is None and out.routing is None


def test_end_to_end_pipeline_builds_active_and_negative_banks():
    pipeline = small_pipeline(enabled=True)
    kwargs = inputs(pipeline)
    out = pipeline(**kwargs)
    assert out.enabled
    assert out.assertion.state_logits.shape == (1, 3, 3)
    assert out.routing.hard_total_symbols.item() <= 32
    assert out.encoded_symbols.actual_complex_channel_uses.item() == 8
    assert out.memory_update.accepted_mask.tolist() == [[True, True, False]]
    assert out.memory_state.active_mask.tolist() == [[True, False, False]]
    assert out.memory_state.negative_mask.tolist() == [[False, True, False]]
    assert tuple(out.adapter_outputs) == (1, 3)
    # Zero-initialized adapters preserve the incoming backbone exactly.
    assert torch.equal(out.diffusion_hidden, kwargs["diffusion_hidden"])


def test_saver_loss_has_all_terms_and_backpropagates():
    torch.manual_seed(9)
    state_logits = torch.randn(1, 3, 3, requires_grad=True)
    action_logits = torch.randn(1, 3, 6, requires_grad=True)
    diffusion = torch.randn(1, 4, 5, requires_grad=True)
    decoded = torch.randn(1, 3, 8, requires_grad=True)
    anchor = torch.randn(1, 3, 8, requires_grad=True)
    presence = torch.randn(1, 3, requires_grad=True)
    losses = SaverLoss(SaverLossWeights())(
        diffusion_prediction=diffusion,
        diffusion_target=torch.zeros_like(diffusion),
        state_logits=state_logits,
        state_targets=torch.tensor([[0, 1, 2]]),
        action_logits=action_logits,
        action_targets=torch.tensor([[1, 5, 0]]),
        valid_mask=torch.tensor([[1, 1, 0]], dtype=torch.bool),
        memory_anchor=anchor,
        memory_positive=torch.randn_like(anchor),
        memory_negative=torch.randn_like(anchor),
        entity_presence_logits=presence,
        absent_mask=torch.tensor([[0, 1, 0]], dtype=torch.bool),
        present_mask=torch.tensor([[1, 0, 0]], dtype=torch.bool),
        rate_penalty=torch.tensor(0.25, requires_grad=True),
        decoded_event_features=decoded,
        event_target_features=torch.randn_like(decoded),
        delivered_mask=torch.tensor([[1, 1, 0]], dtype=torch.bool),
    )
    assert set(losses) == {
        "loss_diff", "loss_state", "loss_memory", "loss_ghost",
        "loss_preserve", "loss_rate", "loss_channel", "loss",
    }
    losses["loss"].backward()
    for tensor in (state_logits, action_logits, diffusion, decoded, anchor, presence):
        assert tensor.grad is not None and tensor.grad.abs().sum() > 0


def test_stage_runner_freezes_exact_modules_and_checkpoint_fails_closed():
    torch.manual_seed(11)
    pipeline = small_pipeline(enabled=True)
    contract = SaverCheckpointContract(
        slot_count=3, memory_dim=8, injection_layers=(1, 3)
    )
    runner = SaverStageRunner(pipeline, SaverLoss(), contract, stage="sv1")
    names = runner.trainable_parameter_names()
    assert names and all(name.startswith(("tokenizer.", "memory.")) for name in names)
    kwargs = inputs(pipeline)
    batch = {
        "pipeline_inputs": kwargs,
        "targets": {
            "memory_positive": torch.randn(1, 3, 8),
            "memory_negative": torch.randn(1, 3, 8),
            "event_target_features": torch.randn(1, 3, 8),
        },
    }
    metrics = runner.training_step(batch)
    assert runner.global_step == 1
    assert metrics["loss"] > 0
    payload = runner.checkpoint_payload({"effective_seed": 2025})
    assert payload["contract_fingerprint"] == contract.fingerprint
    restored = SaverStageRunner(small_pipeline(enabled=True), SaverLoss(), contract, stage="sv1")
    restored.load_checkpoint_payload(payload)
    assert restored.global_step == 1

    bad = copy.deepcopy(payload)
    bad["contract"]["slot_count"] = 4
    with pytest.raises(RuntimeError, match="incompatible SAVER checkpoint"):
        restored.load_checkpoint_payload(bad)


def test_config_factory_is_opt_in_and_preserves_contract_fields():
    cfg = OmegaConf.create({
        "use_saver_jscc": False,
        "saver": {
            "architecture_version": "saver-jscc-v1",
            "packet_schema_version": 1,
            "source_feature_dim": 6,
            "slot_count": 3,
            "memory_dim": 8,
            "num_heads": 2,
            "router": {
                "channel_feature_dim": 3,
                "hidden_dim": 12,
                "rate_options": [0, 2, 4],
                "visual_rate_options": [0, 4, 8],
                "protection_extra_symbols": [0, 2, 1, 4],
            },
            "codec": {"max_symbols": 4, "hidden_dim": 16},
            "dit": {
                "hidden_dim": 12,
                "timestep_dim": 12,
                "channel_token_dim": 4,
                "bottleneck_dim": 8,
                "injection_layers": [1, 3],
            },
        },
    })
    contract = checkpoint_contract_from_config(cfg)
    pipeline = build_saver_video_pipeline(cfg)
    assert contract.slot_count == 3 and contract.injection_layers == (1, 3)
    assert not pipeline.enabled
