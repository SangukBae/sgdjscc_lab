"""Action-conditioned source/wireless semantic channel codec for SAVER-JSCC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from sgdjscc_lab.transmission.saver_packet import (
    SignedStatePacket,
    parse_signed_packet,
    serialize_signed_packet,
)
from sgdjscc_lab.transmission.wire_packet import decode_latent_packet, encode_latent_packet

from .contracts import AssertionAction, ENTITY_ACTION_COUNT, ProtectionAction, SemanticState


@dataclass
class WirelessSemanticSymbols:
    iq: torch.Tensor
    symbol_mask: torch.Tensor
    allocated_symbols: torch.Tensor
    action_indices: torch.Tensor
    state_indices: torch.Tensor

    @property
    def actual_complex_channel_uses(self) -> torch.Tensor:
        return self.symbol_mask.sum(dim=(-1, -2))


@dataclass
class WirelessChannelObservation:
    iq: torch.Tensor
    symbol_mask: torch.Tensor
    reliability: torch.Tensor
    channel_gain_iq: torch.Tensor


class ActionConditionedSemanticChannelCodec(nn.Module):
    """Encode selected events into exact bytes or learned complex symbols.

    ``encode_source_packet`` is a non-differentiable deployment serializer and
    reports exact bytes.  ``encode_wireless``/``decode_wireless`` are the
    differentiable ``saver_wireless`` path and report actual complex uses.
    The two units are never added together by this class.
    """

    def __init__(
        self,
        token_dim: int,
        *,
        max_symbols: int = 64,
        hidden_dim: int = 256,
        action_embedding_dim: int = 16,
        state_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or max_symbols <= 0 or hidden_dim <= 0:
            raise ValueError("token_dim, max_symbols and hidden_dim must be positive")
        self.token_dim = int(token_dim)
        self.max_symbols = int(max_symbols)
        self.action_embedding = nn.Embedding(ENTITY_ACTION_COUNT, action_embedding_dim)
        self.state_embedding = nn.Embedding(len(SemanticState), state_embedding_dim)
        condition_dim = action_embedding_dim + state_embedding_dim + 1
        self.encoder = nn.Sequential(
            nn.Linear(token_dim + condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * max_symbols),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * max_symbols + condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def _condition(
        self,
        actions: torch.Tensor,
        states: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([
            self.action_embedding(actions.long()),
            self.state_embedding(states.long()),
            reliability.unsqueeze(-1),
        ], dim=-1)

    def encode_wireless(
        self,
        event_tokens: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        allocated_symbols: torch.Tensor,
        *,
        reliability: Optional[torch.Tensor] = None,
    ) -> WirelessSemanticSymbols:
        if event_tokens.ndim != 3 or event_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"event_tokens must be [B,K,{self.token_dim}]")
        shape = event_tokens.shape[:2]
        for name, value in {
            "actions": actions,
            "states": states,
            "allocated_symbols": allocated_symbols,
        }.items():
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape [B,K]")
        allocated_symbols = allocated_symbols.long()
        if (allocated_symbols < 0).any() or (allocated_symbols > self.max_symbols).any():
            raise ValueError(f"allocated_symbols must be in [0,{self.max_symbols}]")
        if reliability is None:
            reliability = torch.ones(shape, device=event_tokens.device, dtype=event_tokens.dtype)
        if tuple(reliability.shape) != shape:
            raise ValueError("reliability must have shape [B,K]")
        condition = self._condition(actions, states, reliability.to(event_tokens.dtype))
        iq = self.encoder(torch.cat([event_tokens, condition], dim=-1)).reshape(
            *shape, self.max_symbols, 2
        )
        indices = torch.arange(self.max_symbols, device=event_tokens.device)
        symbol_mask = indices.view(1, 1, -1).lt(allocated_symbols.unsqueeze(-1))
        iq = iq * symbol_mask.unsqueeze(-1).to(iq.dtype)
        # Unit average complex-symbol power for each non-empty event.
        energy = iq.square().sum(dim=-1).sum(dim=-1, keepdim=True)
        count = symbol_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(iq.dtype)
        scale = torch.sqrt(count / energy.clamp_min(torch.finfo(iq.dtype).eps))
        iq = iq * scale.unsqueeze(-1)
        return WirelessSemanticSymbols(iq, symbol_mask, allocated_symbols, actions.long(), states.long())

    def transmit_wireless(
        self,
        encoded: WirelessSemanticSymbols,
        *,
        snr_db: float,
        channel: str = "awgn",
        packet_loss_probability: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> WirelessChannelObservation:
        if channel not in {"awgn", "rayleigh"}:
            raise ValueError("channel must be 'awgn' or 'rayleigh'")
        if not 0.0 <= packet_loss_probability <= 1.0:
            raise ValueError("packet_loss_probability must be in [0,1]")
        iq = encoded.iq
        batch, slots = iq.shape[:2]
        if channel == "rayleigh":
            gain = torch.randn(
                batch, slots, 1, 2, device=iq.device, dtype=iq.dtype, generator=generator
            ) / math.sqrt(2.0)
        else:
            gain = torch.zeros(batch, slots, 1, 2, device=iq.device, dtype=iq.dtype)
            gain[..., 0] = 1.0
        real = gain[..., 0] * iq[..., 0] - gain[..., 1] * iq[..., 1]
        imag = gain[..., 0] * iq[..., 1] + gain[..., 1] * iq[..., 0]
        faded = torch.stack([real, imag], dim=-1)
        noise_std = math.sqrt(0.5 * 10.0 ** (-float(snr_db) / 10.0))
        noise = torch.randn(faded.shape, device=iq.device, dtype=iq.dtype, generator=generator)
        received = faded + noise_std * noise
        gain_power = gain.square().sum(dim=-1).clamp_min(torch.finfo(iq.dtype).eps)
        # Perfect-CSI equalization; reliability still exposes fade/noise quality.
        equal_real = (gain[..., 0] * received[..., 0] + gain[..., 1] * received[..., 1]) / gain_power
        equal_imag = (gain[..., 0] * received[..., 1] - gain[..., 1] * received[..., 0]) / gain_power
        equalized = torch.stack([equal_real, equal_imag], dim=-1)

        keep = torch.rand(
            batch, slots, device=iq.device, generator=generator
        ).ge(packet_loss_probability)
        mask = encoded.symbol_mask & keep.unsqueeze(-1)
        equalized = equalized * mask.unsqueeze(-1).to(equalized.dtype)
        reliability = keep.to(iq.dtype) * (
            gain_power.squeeze(-1) / (gain_power.squeeze(-1) + 10.0 ** (-float(snr_db) / 10.0))
        )
        return WirelessChannelObservation(equalized, mask, reliability, gain)

    def decode_wireless(
        self,
        observation: WirelessChannelObservation,
        actions: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        shape = observation.iq.shape[:2]
        if tuple(actions.shape) != shape or tuple(states.shape) != shape:
            raise ValueError("actions and states must match observation [B,K]")
        reliability = observation.reliability.to(observation.iq.dtype)
        condition = self._condition(actions, states, reliability)
        flat = observation.iq.reshape(*shape, 2 * self.max_symbols)
        decoded = self.decoder(torch.cat([flat, condition], dim=-1))
        valid = observation.symbol_mask.any(dim=-1)
        return decoded * valid.unsqueeze(-1).to(decoded.dtype)

    @staticmethod
    def encode_source_packet(
        feature: torch.Tensor,
        *,
        scene_epoch: int,
        entity_id: str,
        version: int,
        state: SemanticState,
        action: AssertionAction,
        confidence: float,
        bit_depth: int = 4,
        protection: ProtectionAction = ProtectionAction.NONE,
        ack_requested: bool = False,
    ) -> bytes:
        payload = encode_latent_packet(
            feature,
            bit_depth=bit_depth,
            granularity="per_tensor",
            channel_dim=0,
            metadata={"profile": "saver_source", "payload": "entity_feature"},
        )
        return serialize_signed_packet(SignedStatePacket(
            scene_epoch=scene_epoch,
            entity_id=entity_id,
            version=version,
            state=state,
            action=action,
            confidence=confidence,
            payload=payload,
            protection=protection,
            ack_requested=ack_requested,
        ))

    @staticmethod
    def decode_source_packet(
        wire: bytes,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> tuple[SignedStatePacket, torch.Tensor]:
        packet = parse_signed_packet(wire)
        feature = decode_latent_packet(packet.payload, dtype=dtype, device=device)
        return packet, feature

