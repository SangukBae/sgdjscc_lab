"""Zero-initialized signed-memory adapters for diffusion Transformer blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn


def _masked_attention(
    attention: nn.MultiheadAttention,
    query: torch.Tensor,
    memory: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if memory.ndim != 3 or memory.shape[0] != query.shape[0]:
        raise ValueError("memory must be [B,K,d] with the same batch as query")
    if memory.shape[1] == 0:
        return torch.zeros_like(query)
    if valid_mask is None:
        valid_mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
    if tuple(valid_mask.shape) != memory.shape[:2]:
        raise ValueError("memory mask must have shape [B,K]")
    valid_mask = valid_mask.bool()
    has_memory = valid_mask.any(dim=1)
    safe_mask = valid_mask.clone()
    safe_mask[~has_memory, 0] = True
    attended, _ = attention(
        query, memory, memory, key_padding_mask=~safe_mask, need_weights=False
    )
    return attended * has_memory[:, None, None].to(attended.dtype)


@dataclass
class SignedMemoryAdapterOutput:
    hidden_states: torch.Tensor
    positive_residual: torch.Tensor
    negative_residual: torch.Tensor
    positive_gate: torch.Tensor
    negative_gate: torch.Tensor


class SignedMemoryDiTBlock(nn.Module):
    """Identity-initialized adapter with asymmetric active/negative attention."""

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        channel_dim: int = 8,
        *,
        timestep_dim: Optional[int] = None,
        bottleneck_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        bottleneck_dim = int(bottleneck_dim or memory_dim)
        timestep_dim = int(timestep_dim or hidden_dim)
        if min(hidden_dim, memory_dim, channel_dim, timestep_dim, bottleneck_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        if bottleneck_dim % num_heads:
            raise ValueError("bottleneck_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.memory_dim = int(memory_dim)
        self.channel_dim = int(channel_dim)
        self.bottleneck_dim = bottleneck_dim

        self.hidden_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.hidden_down = nn.Linear(hidden_dim, bottleneck_dim)
        self.memory_projection = nn.Linear(memory_dim, bottleneck_dim)
        self.channel_projection = nn.Linear(channel_dim, bottleneck_dim)
        self.timestep_projection = nn.Linear(timestep_dim, bottleneck_dim)
        self.adaln = nn.Linear(bottleneck_dim, 2 * bottleneck_dim)
        self.self_attention = nn.MultiheadAttention(
            bottleneck_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.positive_attention = nn.MultiheadAttention(
            bottleneck_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.negative_attention = nn.MultiheadAttention(
            bottleneck_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(bottleneck_dim + 2, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, 2),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, 4 * bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * bottleneck_dim, bottleneck_dim),
        )
        self.output_projection = nn.Linear(bottleneck_dim, hidden_dim)
        # Exact identity at initialization; gradients still reach this layer on
        # the first step and the rest of the adapter thereafter.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _batch_scalar(value, batch: int, *, device, dtype, default: float) -> torch.Tensor:
        if value is None:
            return torch.full((batch,), default, device=device, dtype=dtype)
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand(batch)
        if tuple(tensor.shape) != (batch,):
            raise ValueError("reliability/confidence must be scalar or [B]")
        return tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        channel_tokens: torch.Tensor,
        active_memory: torch.Tensor,
        negative_memory: torch.Tensor,
        *,
        active_mask: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None,
        channel_reliability=None,
        assertion_confidence=None,
    ) -> SignedMemoryAdapterOutput:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(f"hidden_states must be [B,L,{self.hidden_dim}]")
        batch = hidden_states.shape[0]
        if timestep_embedding.ndim != 2 or timestep_embedding.shape[0] != batch:
            raise ValueError("timestep_embedding must be [B,d_t]")
        if channel_tokens.ndim != 3 or tuple(channel_tokens.shape[:1]) != (batch,):
            raise ValueError("channel_tokens must be [B,T,d_ch]")
        if channel_tokens.shape[-1] != self.channel_dim:
            raise ValueError(f"channel token dim must be {self.channel_dim}")

        hidden = self.hidden_down(self.hidden_norm(hidden_states))
        channel_condition = self.channel_projection(channel_tokens).mean(dim=1)
        condition = self.timestep_projection(timestep_embedding) + channel_condition
        scale, shift = self.adaln(condition).chunk(2, dim=-1)
        hidden = hidden * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        self_context, _ = self.self_attention(hidden, hidden, hidden, need_weights=False)
        hidden = hidden + self_context

        active = self.memory_projection(active_memory)
        negative = self.memory_projection(negative_memory)
        positive_residual = _masked_attention(
            self.positive_attention, hidden, active, active_mask
        )
        negative_residual = torch.tanh(_masked_attention(
            self.negative_attention, hidden, negative, negative_mask
        ))

        reliability = self._batch_scalar(
            channel_reliability, batch, device=hidden.device, dtype=hidden.dtype, default=1.0
        )
        confidence = self._batch_scalar(
            assertion_confidence, batch, device=hidden.device, dtype=hidden.dtype, default=1.0
        )
        gates = torch.sigmoid(self.gate(torch.cat([
            condition, reliability.unsqueeze(-1), confidence.unsqueeze(-1)
        ], dim=-1)))
        positive_gate = gates[:, 0:1, None]
        negative_gate = gates[:, 1:2, None] * confidence[:, None, None]
        signed = positive_gate * positive_residual - negative_gate * negative_residual
        adapted = signed + self.ffn(hidden + signed)
        output = hidden_states + self.output_projection(adapted)
        return SignedMemoryAdapterOutput(
            hidden_states=output,
            positive_residual=positive_residual,
            negative_residual=negative_residual,
            positive_gate=positive_gate,
            negative_gate=negative_gate,
        )


class SignedMemoryDiTAdapterStack(nn.Module):
    """Adapter registry invoked at selected backbone block indices."""

    def __init__(
        self,
        injection_layers: Sequence[int],
        *,
        hidden_dim: int,
        memory_dim: int,
        channel_dim: int = 8,
        timestep_dim: Optional[int] = None,
        bottleneck_dim: Optional[int] = None,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        layers = tuple(int(x) for x in injection_layers)
        if not layers or tuple(sorted(set(layers))) != layers or layers[0] < 0:
            raise ValueError("injection_layers must be unique, sorted and non-negative")
        self.injection_layers = layers
        self.adapters = nn.ModuleDict({
            str(layer): SignedMemoryDiTBlock(
                hidden_dim=hidden_dim,
                memory_dim=memory_dim,
                channel_dim=channel_dim,
                timestep_dim=timestep_dim,
                bottleneck_dim=bottleneck_dim,
                num_heads=num_heads,
            )
            for layer in layers
        })

    def has_layer(self, layer_index: int) -> bool:
        return str(int(layer_index)) in self.adapters

    def inject(self, layer_index: int, hidden_states: torch.Tensor, **kwargs) -> SignedMemoryAdapterOutput:
        key = str(int(layer_index))
        if key not in self.adapters:
            zeros = torch.zeros_like(hidden_states)
            one = torch.ones(
                hidden_states.shape[0], 1, 1, device=hidden_states.device, dtype=hidden_states.dtype
            )
            return SignedMemoryAdapterOutput(hidden_states, zeros, zeros, one, one)
        return self.adapters[key](hidden_states, **kwargs)

