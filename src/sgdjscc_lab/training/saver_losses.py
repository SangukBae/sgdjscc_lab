"""Multi-objective losses for staged and end-to-end SAVER-JSCC training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SaverLossWeights:
    diffusion: float = 1.0
    state: float = 1.0
    memory: float = 0.2
    ghost: float = 0.5
    preserve: float = 0.5
    rate: float = 0.05
    channel: float = 0.2


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)


class SaverLoss(nn.Module):
    """Compute the seven predeclared SAVER objectives without evaluator leakage."""

    def __init__(self, weights: SaverLossWeights = SaverLossWeights(), memory_margin: float = 0.2) -> None:
        super().__init__()
        self.weights = weights
        self.memory_margin = float(memory_margin)

    @staticmethod
    def _device_tensor(candidates) -> torch.Tensor:
        for value in candidates:
            if isinstance(value, torch.Tensor):
                return value.new_zeros(())
        raise ValueError("at least one tensor loss input is required")

    def forward(
        self,
        *,
        diffusion_prediction: Optional[torch.Tensor] = None,
        diffusion_target: Optional[torch.Tensor] = None,
        state_logits: Optional[torch.Tensor] = None,
        state_targets: Optional[torch.Tensor] = None,
        action_logits: Optional[torch.Tensor] = None,
        action_targets: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        memory_anchor: Optional[torch.Tensor] = None,
        memory_positive: Optional[torch.Tensor] = None,
        memory_negative: Optional[torch.Tensor] = None,
        entity_presence_logits: Optional[torch.Tensor] = None,
        absent_mask: Optional[torch.Tensor] = None,
        present_mask: Optional[torch.Tensor] = None,
        rate_penalty: Optional[torch.Tensor] = None,
        decoded_event_features: Optional[torch.Tensor] = None,
        event_target_features: Optional[torch.Tensor] = None,
        delivered_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        zero = self._device_tensor([
            diffusion_prediction, state_logits, action_logits, memory_anchor,
            entity_presence_logits, rate_penalty, decoded_event_features,
        ])
        losses: Dict[str, torch.Tensor] = {}

        if diffusion_prediction is not None and diffusion_target is not None:
            losses["loss_diff"] = F.mse_loss(diffusion_prediction, diffusion_target)
        else:
            losses["loss_diff"] = zero

        state_terms = []
        if state_logits is not None and state_targets is not None:
            raw = F.cross_entropy(
                state_logits.reshape(-1, state_logits.shape[-1]),
                state_targets.long().reshape(-1),
                reduction="none",
            ).reshape(state_targets.shape)
            state_terms.append(_masked_mean(raw, valid_mask if valid_mask is not None else torch.ones_like(raw)))
        if action_logits is not None and action_targets is not None:
            raw = F.cross_entropy(
                action_logits.reshape(-1, action_logits.shape[-1]),
                action_targets.long().reshape(-1),
                reduction="none",
            ).reshape(action_targets.shape)
            state_terms.append(_masked_mean(raw, valid_mask if valid_mask is not None else torch.ones_like(raw)))
        losses["loss_state"] = torch.stack(state_terms).mean() if state_terms else zero

        if memory_anchor is not None and memory_positive is not None:
            positive = 1.0 - F.cosine_similarity(memory_anchor, memory_positive, dim=-1)
            memory_loss = _masked_mean(
                positive, valid_mask if valid_mask is not None else torch.ones_like(positive)
            )
            if memory_negative is not None:
                negative = F.cosine_similarity(memory_anchor, memory_negative, dim=-1)
                memory_loss = memory_loss + _masked_mean(
                    torch.relu(negative - self.memory_margin),
                    valid_mask if valid_mask is not None else torch.ones_like(negative),
                )
            losses["loss_memory"] = memory_loss
        else:
            losses["loss_memory"] = zero

        if entity_presence_logits is not None and absent_mask is not None:
            ghost_raw = F.binary_cross_entropy_with_logits(
                entity_presence_logits, torch.zeros_like(entity_presence_logits), reduction="none"
            )
            losses["loss_ghost"] = _masked_mean(ghost_raw, absent_mask)
        else:
            losses["loss_ghost"] = zero
        if entity_presence_logits is not None and present_mask is not None:
            preserve_raw = F.binary_cross_entropy_with_logits(
                entity_presence_logits, torch.ones_like(entity_presence_logits), reduction="none"
            )
            losses["loss_preserve"] = _masked_mean(preserve_raw, present_mask)
        else:
            losses["loss_preserve"] = zero

        losses["loss_rate"] = rate_penalty.mean() if rate_penalty is not None else zero
        if decoded_event_features is not None and event_target_features is not None:
            channel_raw = (decoded_event_features - event_target_features).square()
            losses["loss_channel"] = _masked_mean(
                channel_raw,
                delivered_mask if delivered_mask is not None else torch.ones(channel_raw.shape[:-1], device=channel_raw.device),
            )
        else:
            losses["loss_channel"] = zero

        total = (
            self.weights.diffusion * losses["loss_diff"]
            + self.weights.state * losses["loss_state"]
            + self.weights.memory * losses["loss_memory"]
            + self.weights.ghost * losses["loss_ghost"]
            + self.weights.preserve * losses["loss_preserve"]
            + self.weights.rate * losses["loss_rate"]
            + self.weights.channel * losses["loss_channel"]
        )
        losses["loss"] = total
        return losses

