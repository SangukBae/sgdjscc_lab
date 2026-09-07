"""Versioned, revocable dual-bank neural entity memory for SAVER-JSCC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .contracts import AssertionAction, DEFAULT_MEMORY_DIM, DEFAULT_SLOT_COUNT, RenderStatus, SemanticState


@dataclass
class VersionedMemoryState:
    identity: torch.Tensor
    render: torch.Tensor
    negative: torch.Tensor
    entity_ids: torch.Tensor
    versions: torch.Tensor
    semantic_states: torch.Tensor
    render_status: torch.Tensor
    occupied: torch.Tensor
    scene_epoch: torch.Tensor

    def detach(self) -> "VersionedMemoryState":
        return VersionedMemoryState(**{
            name: value.detach()
            for name, value in self.__dict__.items()
        })

    @property
    def active_mask(self) -> torch.Tensor:
        return self.occupied & self.render_status.eq(int(RenderStatus.ACTIVE))

    @property
    def negative_mask(self) -> torch.Tensor:
        return self.occupied & (
            self.render_status.eq(int(RenderStatus.REVOKED))
            | self.render_status.eq(int(RenderStatus.ABSENT))
        )


@dataclass
class MemoryUpdateOutput:
    state: VersionedMemoryState
    accepted_mask: torch.Tensor
    stale_mask: torch.Tensor


class VersionedRevocableEntityMemory(nn.Module):
    """Learned feature updater guarded by deterministic version transitions.

    State is passed in/out explicitly so batches, replicas and checkpoint
    boundaries cannot accidentally share hidden mutable memory.
    """

    def __init__(
        self,
        memory_dim: int = DEFAULT_MEMORY_DIM,
        slot_count: int = DEFAULT_SLOT_COUNT,
    ) -> None:
        super().__init__()
        if memory_dim <= 0 or slot_count <= 0:
            raise ValueError("memory_dim and slot_count must be positive")
        self.memory_dim = int(memory_dim)
        self.slot_count = int(slot_count)
        self.identity_updater = nn.GRUCell(memory_dim, memory_dim)
        self.render_projection = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, memory_dim))
        self.negative_projection = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, memory_dim))

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        scene_epoch: int = 0,
    ) -> VersionedMemoryState:
        if batch_size <= 0 or scene_epoch < 0:
            raise ValueError("batch_size must be positive and scene_epoch non-negative")
        shape = (batch_size, self.slot_count, self.memory_dim)
        slots = (batch_size, self.slot_count)
        return VersionedMemoryState(
            identity=torch.zeros(shape, device=device, dtype=dtype),
            render=torch.zeros(shape, device=device, dtype=dtype),
            negative=torch.zeros(shape, device=device, dtype=dtype),
            entity_ids=torch.full(slots, -1, device=device, dtype=torch.long),
            versions=torch.full(slots, -1, device=device, dtype=torch.long),
            semantic_states=torch.full(
                slots, int(SemanticState.UNKNOWN), device=device, dtype=torch.long
            ),
            render_status=torch.full(
                slots, int(RenderStatus.EMPTY), device=device, dtype=torch.long
            ),
            occupied=torch.zeros(slots, device=device, dtype=torch.bool),
            scene_epoch=torch.full(
                (batch_size,), scene_epoch, device=device, dtype=torch.long
            ),
        )

    def _validate_state(self, state: VersionedMemoryState, batch: int) -> None:
        vector_shape = (batch, self.slot_count, self.memory_dim)
        slot_shape = (batch, self.slot_count)
        for name in ("identity", "render", "negative"):
            if tuple(getattr(state, name).shape) != vector_shape:
                raise ValueError(f"memory {name} must have shape {vector_shape}")
        for name in ("entity_ids", "versions", "semantic_states", "render_status", "occupied"):
            if tuple(getattr(state, name).shape) != slot_shape:
                raise ValueError(f"memory {name} must have shape {slot_shape}")
        if tuple(state.scene_epoch.shape) != (batch,):
            raise ValueError("memory scene_epoch must have shape [B]")

    def forward(
        self,
        state: VersionedMemoryState,
        entity_features: torch.Tensor,
        entity_ids: torch.Tensor,
        semantic_states: torch.Tensor,
        operations: torch.Tensor,
        versions: torch.Tensor,
        *,
        scene_epochs: Optional[torch.Tensor] = None,
        scene_reset_mask: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> MemoryUpdateOutput:
        if entity_features.ndim != 3:
            raise ValueError("entity_features must be [B,K,d_m]")
        batch = entity_features.shape[0]
        expected = (batch, self.slot_count, self.memory_dim)
        if tuple(entity_features.shape) != expected:
            raise ValueError(f"entity_features must have shape {expected}")
        self._validate_state(state, batch)
        slot_shape = (batch, self.slot_count)
        for name, value in {
            "entity_ids": entity_ids,
            "semantic_states": semantic_states,
            "operations": operations,
            "versions": versions,
        }.items():
            if tuple(value.shape) != slot_shape:
                raise ValueError(f"{name} must have shape {slot_shape}")
        device = entity_features.device
        if scene_epochs is None:
            scene_epochs = state.scene_epoch
        if scene_reset_mask is None:
            scene_reset_mask = torch.zeros(batch, dtype=torch.bool, device=device)
        if confidence is None:
            confidence = torch.ones(slot_shape, dtype=entity_features.dtype, device=device)
        if valid_mask is None:
            valid_mask = torch.ones(slot_shape, dtype=torch.bool, device=device)
        if tuple(scene_epochs.shape) != (batch,) or tuple(scene_reset_mask.shape) != (batch,):
            raise ValueError("scene_epochs and scene_reset_mask must have shape [B]")

        invalid_unknown_revoke = (
            semantic_states.eq(int(SemanticState.UNKNOWN))
            & operations.eq(int(AssertionAction.REVOKE))
            & valid_mask
        )
        if invalid_unknown_revoke.any():
            raise ValueError("UNKNOWN evidence cannot issue REVOKE")

        # A new epoch is valid only with an explicit reset. Reset all banks first.
        reset = scene_reset_mask.bool() & scene_epochs.ge(state.scene_epoch)
        reset_slots = reset.unsqueeze(-1)
        reset_vectors = reset_slots.unsqueeze(-1)
        identity = torch.where(reset_vectors, torch.zeros_like(state.identity), state.identity)
        render = torch.where(reset_vectors, torch.zeros_like(state.render), state.render)
        negative = torch.where(reset_vectors, torch.zeros_like(state.negative), state.negative)
        entity_state = torch.where(reset_slots, torch.full_like(state.entity_ids, -1), state.entity_ids)
        version_state = torch.where(reset_slots, torch.full_like(state.versions, -1), state.versions)
        semantic_state = torch.where(
            reset_slots,
            torch.full_like(state.semantic_states, int(SemanticState.UNKNOWN)),
            state.semantic_states,
        )
        render_status = torch.where(
            reset_slots,
            torch.full_like(state.render_status, int(RenderStatus.EMPTY)),
            state.render_status,
        )
        occupied = torch.where(reset_slots, torch.zeros_like(state.occupied), state.occupied)
        effective_epoch = torch.where(reset, scene_epochs, state.scene_epoch)

        # Slot binding is explicit. Reusing an occupied slot for a different ID
        # without a scene reset/eviction would silently create identity swaps.
        identity_swap = valid_mask & occupied & entity_state.ne(entity_ids)
        if identity_swap.any():
            raise ValueError(
                "identity swap: entity_id changed in an occupied slot without "
                "explicit reset/eviction"
            )

        same_epoch = scene_epochs.unsqueeze(-1).eq(effective_epoch.unsqueeze(-1))
        newer = (~occupied) | versions.gt(version_state)
        non_skip = operations.ne(int(AssertionAction.SKIP))
        accepted = valid_mask.bool() & entity_ids.ge(0) & same_epoch & newer & non_skip
        stale = valid_mask.bool() & non_skip & ~accepted

        flat_features = entity_features.reshape(-1, self.memory_dim)
        flat_identity = identity.reshape(-1, self.memory_dim)
        candidate_identity = self.identity_updater(flat_features, flat_identity).reshape(expected)
        identity_actions = (
            operations.eq(int(AssertionAction.ASSERT))
            | operations.eq(int(AssertionAction.UPDATE))
            | operations.eq(int(AssertionAction.RESUME))
        ) & semantic_states.eq(int(SemanticState.PRESENT))
        update_identity = accepted & identity_actions
        identity = torch.where(update_identity.unsqueeze(-1), candidate_identity, identity)

        active = accepted & semantic_states.eq(int(SemanticState.PRESENT)) & (
            operations.eq(int(AssertionAction.ASSERT))
            | operations.eq(int(AssertionAction.UPDATE))
            | operations.eq(int(AssertionAction.RESUME))
        )
        suspended = accepted & operations.eq(int(AssertionAction.SUSPEND))
        revoked = accepted & operations.eq(int(AssertionAction.REVOKE))
        absent = accepted & semantic_states.eq(int(SemanticState.CONFIRMED_ABSENT)) & ~revoked
        render_candidate = self.render_projection(identity)
        negative_candidate = self.negative_projection(entity_features) * confidence.unsqueeze(-1)
        render = torch.where(active.unsqueeze(-1), render_candidate, render)
        render = torch.where((suspended | revoked | absent).unsqueeze(-1), torch.zeros_like(render), render)
        negative = torch.where(active.unsqueeze(-1), torch.zeros_like(negative), negative)
        negative = torch.where((revoked | absent).unsqueeze(-1), negative_candidate, negative)

        render_status = torch.where(active, torch.full_like(render_status, int(RenderStatus.ACTIVE)), render_status)
        render_status = torch.where(
            suspended, torch.full_like(render_status, int(RenderStatus.SUSPENDED)), render_status
        )
        render_status = torch.where(
            revoked, torch.full_like(render_status, int(RenderStatus.REVOKED)), render_status
        )
        render_status = torch.where(
            absent, torch.full_like(render_status, int(RenderStatus.ABSENT)), render_status
        )
        entity_state = torch.where(accepted, entity_ids.long(), entity_state)
        version_state = torch.where(accepted, versions.long(), version_state)
        semantic_state = torch.where(accepted, semantic_states.long(), semantic_state)
        occupied = occupied | accepted

        next_state = VersionedMemoryState(
            identity=identity,
            render=render,
            negative=negative,
            entity_ids=entity_state,
            versions=version_state,
            semantic_states=semantic_state,
            render_status=render_status,
            occupied=occupied,
            scene_epoch=effective_epoch,
        )
        return MemoryUpdateOutput(next_state, accepted, stale)
