"""Tensor-level end-to-end SAVER-JSCC video pipeline.

The pipeline composes the trainable SAVER modules without depending on a
specific multi-GB diffusion backbone.  A backbone calls ``inject`` at its
configured block indices, while CPU tests can exercise the complete gradient
path with small tensors.  ``enabled=False`` is a strict identity path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from sgdjscc_lab.models.saver import (
    ActionConditionedSemanticChannelCodec,
    AssertionAction,
    JointAssertionSymbolRouter,
    MemoryUpdateOutput,
    RouterOutput,
    SignedAssertionOutput,
    SignedAssertionTokenizer,
    SignedMemoryAdapterOutput,
    SignedMemoryDiTAdapterStack,
    VersionedMemoryState,
    VersionedRevocableEntityMemory,
    WirelessChannelObservation,
    WirelessSemanticSymbols,
)


@dataclass
class SaverForwardOutput:
    enabled: bool
    diffusion_hidden: torch.Tensor
    memory_state: VersionedMemoryState
    assertion: Optional[SignedAssertionOutput] = None
    routing: Optional[RouterOutput] = None
    encoded_symbols: Optional[WirelessSemanticSymbols] = None
    channel_observation: Optional[WirelessChannelObservation] = None
    decoded_event_features: Optional[torch.Tensor] = None
    memory_update: Optional[MemoryUpdateOutput] = None
    adapter_outputs: Dict[int, SignedMemoryAdapterOutput] = field(default_factory=dict)


class SaverVideoPipeline(nn.Module):
    """Compose SAT → JASR → codec → VREM → signed-memory adapters."""

    def __init__(
        self,
        tokenizer: SignedAssertionTokenizer,
        router: JointAssertionSymbolRouter,
        codec: ActionConditionedSemanticChannelCodec,
        memory: VersionedRevocableEntityMemory,
        dit_adapters: SignedMemoryDiTAdapterStack,
        *,
        enabled: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.router = router
        self.codec = codec
        self.memory = memory
        self.dit_adapters = dit_adapters
        self.enabled = bool(enabled)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def forward(
        self,
        *,
        source_features: torch.Tensor,
        visual_tokens: torch.Tensor,
        diffusion_hidden: torch.Tensor,
        timestep_embedding: torch.Tensor,
        channel_features: torch.Tensor,
        channel_tokens: torch.Tensor,
        memory_state: VersionedMemoryState,
        entity_ids: torch.Tensor,
        semantic_states: torch.Tensor,
        versions: torch.Tensor,
        scene_epochs: torch.Tensor,
        total_budget,
        valid_mask: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
        scene_reset_mask: Optional[torch.Tensor] = None,
        scene_reset_versions: Optional[torch.Tensor] = None,
        teacher_actions: Optional[torch.Tensor] = None,
        teacher_allocated_symbols: Optional[torch.Tensor] = None,
        snr_db: float = 10.0,
        channel: str = "awgn",
        packet_loss_probability: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> SaverForwardOutput:
        if not self.enabled:
            return SaverForwardOutput(
                enabled=False,
                diffusion_hidden=diffusion_hidden,
                memory_state=memory_state,
            )

        previous_slots = memory_state.identity
        assertion = self.tokenizer(
            source_features,
            previous_slots,
            source_mask=source_mask,
            slot_mask=valid_mask,
            evidence_states=semantic_states,
        )
        routing = self.router(
            assertion.slots,
            visual_tokens,
            channel_features,
            total_budget,
            valid_mask=assertion.valid_mask,
            semantic_states=semantic_states,
        )
        actions = routing.hard_actions if teacher_actions is None else teacher_actions.long()
        allocation = (
            routing.hard_rate_symbols
            if teacher_allocated_symbols is None
            else teacher_allocated_symbols.long()
        )
        expected = semantic_states.shape
        if tuple(actions.shape) != tuple(expected) or tuple(allocation.shape) != tuple(expected):
            raise ValueError("teacher actions/allocations must have shape [B,K]")
        selected = assertion.valid_mask & allocation.gt(0) & actions.ne(int(AssertionAction.SKIP))
        effective_actions = torch.where(
            selected, actions, torch.full_like(actions, int(AssertionAction.SKIP))
        )
        effective_allocation = torch.where(selected, allocation, torch.zeros_like(allocation))

        encoded = self.codec.encode_wireless(
            assertion.slots,
            effective_actions,
            semantic_states,
            effective_allocation,
            reliability=assertion.confidence,
        )
        observation = self.codec.transmit_wireless(
            encoded,
            snr_db=snr_db,
            channel=channel,
            packet_loss_probability=packet_loss_probability,
            generator=generator,
        )
        decoded = self.codec.decode_wireless(observation, effective_actions, semantic_states)
        delivered = selected & observation.symbol_mask.any(dim=-1)
        memory_update = self.memory(
            memory_state,
            decoded,
            entity_ids,
            semantic_states,
            effective_actions,
            versions,
            scene_epochs=scene_epochs,
            scene_reset_mask=scene_reset_mask,
            scene_reset_versions=scene_reset_versions,
            confidence=assertion.confidence * observation.reliability,
            valid_mask=delivered,
        )
        state = memory_update.state

        valid_count = delivered.sum(dim=1).clamp_min(1).to(assertion.confidence.dtype)
        mean_confidence = (
            assertion.confidence * observation.reliability * delivered
        ).sum(dim=1) / valid_count
        mean_reliability = (observation.reliability * delivered).sum(dim=1) / valid_count

        hidden = diffusion_hidden
        adapter_outputs: Dict[int, SignedMemoryAdapterOutput] = {}
        for layer in self.dit_adapters.injection_layers:
            adapted = self.dit_adapters.inject(
                layer,
                hidden,
                timestep_embedding=timestep_embedding,
                channel_tokens=channel_tokens,
                active_memory=state.render,
                negative_memory=state.negative,
                active_mask=state.active_mask,
                negative_mask=state.negative_mask,
                channel_reliability=mean_reliability,
                assertion_confidence=mean_confidence,
            )
            adapter_outputs[layer] = adapted
            hidden = adapted.hidden_states

        return SaverForwardOutput(
            enabled=True,
            diffusion_hidden=hidden,
            memory_state=state,
            assertion=assertion,
            routing=routing,
            encoded_symbols=encoded,
            channel_observation=observation,
            decoded_event_features=decoded,
            memory_update=memory_update,
            adapter_outputs=adapter_outputs,
        )
