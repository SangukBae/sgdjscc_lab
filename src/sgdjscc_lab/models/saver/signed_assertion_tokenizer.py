"""Learned entity-slot tokenizer for SAVER-JSCC signed assertions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .contracts import (
    AssertionAction,
    DEFAULT_MEMORY_DIM,
    DEFAULT_SLOT_COUNT,
    ENTITY_ACTION_COUNT,
    SemanticState,
)


@dataclass
class SignedAssertionOutput:
    slots: torch.Tensor
    state_logits: torch.Tensor
    action_logits: torch.Tensor
    confidence: torch.Tensor
    valid_mask: torch.Tensor

    @property
    def state_probabilities(self) -> torch.Tensor:
        return self.state_logits.softmax(dim=-1)

    @property
    def action_probabilities(self) -> torch.Tensor:
        return self.action_logits.softmax(dim=-1)


class SignedAssertionTokenizer(nn.Module):
    """Bind GOP features to persistent entity slots and predict signed events.

    The module only receives current source features and transmitter-side
    previous slots.  Receiver reconstructions and future frames are deliberately
    absent from the interface.
    """

    def __init__(
        self,
        input_dim: int,
        memory_dim: int = DEFAULT_MEMORY_DIM,
        slot_count: int = DEFAULT_SLOT_COUNT,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or memory_dim <= 0 or slot_count <= 0:
            raise ValueError("input_dim, memory_dim and slot_count must be positive")
        if memory_dim % num_heads:
            raise ValueError("memory_dim must be divisible by num_heads")
        self.input_dim = int(input_dim)
        self.memory_dim = int(memory_dim)
        self.slot_count = int(slot_count)
        self.source_projection = nn.Linear(input_dim, memory_dim)
        self.slot_queries = nn.Parameter(torch.empty(slot_count, memory_dim))
        nn.init.normal_(self.slot_queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            memory_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.slot_update = nn.GRUCell(memory_dim, memory_dim)
        self.slot_norm = nn.LayerNorm(memory_dim)
        self.state_head = nn.Linear(memory_dim, len(SemanticState))
        self.action_head = nn.Sequential(
            nn.Linear(2 * memory_dim, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, ENTITY_ACTION_COUNT),
        )
        self.confidence_head = nn.Linear(memory_dim, 1)

    def _validate(
        self,
        source_features: torch.Tensor,
        previous_slots: Optional[torch.Tensor],
        slot_mask: Optional[torch.Tensor],
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        if source_features.ndim != 3 or source_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"source_features must be [B,L,{self.input_dim}], "
                f"got {tuple(source_features.shape)}"
            )
        batch = source_features.shape[0]
        if previous_slots is None:
            previous_slots = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        if tuple(previous_slots.shape) != (batch, self.slot_count, self.memory_dim):
            raise ValueError(
                "previous_slots must be "
                f"[B,{self.slot_count},{self.memory_dim}], got {tuple(previous_slots.shape)}"
            )
        if slot_mask is None:
            slot_mask = torch.ones(
                batch, self.slot_count, dtype=torch.bool, device=source_features.device
            )
        if tuple(slot_mask.shape) != (batch, self.slot_count):
            raise ValueError(f"slot_mask must be [B,{self.slot_count}]")
        return batch, previous_slots, slot_mask.bool()

    def forward(
        self,
        source_features: torch.Tensor,
        previous_slots: Optional[torch.Tensor] = None,
        *,
        source_mask: Optional[torch.Tensor] = None,
        slot_mask: Optional[torch.Tensor] = None,
        evidence_states: Optional[torch.Tensor] = None,
    ) -> SignedAssertionOutput:
        batch, previous_slots, slot_mask = self._validate(
            source_features, previous_slots, slot_mask
        )
        source = self.source_projection(source_features)
        if source_mask is not None:
            if tuple(source_mask.shape) != source_features.shape[:2]:
                raise ValueError("source_mask must match source_features [B,L]")
            key_padding_mask = ~source_mask.bool()
        else:
            key_padding_mask = None

        query = self.slot_queries.unsqueeze(0).expand(batch, -1, -1) + previous_slots
        context, _ = self.cross_attention(
            query, source, source, key_padding_mask=key_padding_mask, need_weights=False
        )
        updated = self.slot_update(
            context.reshape(-1, self.memory_dim),
            previous_slots.reshape(-1, self.memory_dim),
        ).reshape(batch, self.slot_count, self.memory_dim)
        updated = self.slot_norm(updated)
        updated = torch.where(slot_mask.unsqueeze(-1), updated, previous_slots)

        state_logits = self.state_head(updated)
        difference = updated - previous_slots
        action_logits = self.action_head(torch.cat([updated, difference], dim=-1))
        confidence = self.confidence_head(updated).sigmoid().squeeze(-1)

        if evidence_states is None:
            evidence_states = state_logits.detach().argmax(dim=-1)
        if tuple(evidence_states.shape) != (batch, self.slot_count):
            raise ValueError(f"evidence_states must be [B,{self.slot_count}]")
        unknown = evidence_states.eq(int(SemanticState.UNKNOWN))
        action_logits = action_logits.clone()
        action_logits[..., int(AssertionAction.REVOKE)] = action_logits[
            ..., int(AssertionAction.REVOKE)
        ].masked_fill(unknown, torch.finfo(action_logits.dtype).min)

        # Padded slots deterministically select SKIP and contribute no confidence.
        invalid = ~slot_mask
        if invalid.any():
            action_logits = action_logits.masked_fill(invalid.unsqueeze(-1), torch.finfo(action_logits.dtype).min)
            skip_values = action_logits[..., int(AssertionAction.SKIP)]
            action_logits[..., int(AssertionAction.SKIP)] = torch.where(
                invalid, torch.zeros_like(skip_values), skip_values
            )
            confidence = confidence.masked_fill(invalid, 0.0)

        return SignedAssertionOutput(
            slots=updated,
            state_logits=state_logits,
            action_logits=action_logits,
            confidence=confidence,
            valid_mask=slot_mask,
        )

