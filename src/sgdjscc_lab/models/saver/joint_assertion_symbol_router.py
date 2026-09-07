"""Budget-constrained joint visual/assertion router for SAVER-JSCC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import AssertionAction, ENTITY_ACTION_COUNT, ProtectionAction, SemanticState


@dataclass
class RouterOutput:
    action_logits: torch.Tensor
    rate_logits: torch.Tensor
    protection_logits: torch.Tensor
    visual_rate_logits: torch.Tensor
    action_weights: torch.Tensor
    rate_weights: torch.Tensor
    protection_weights: torch.Tensor
    visual_rate_weights: torch.Tensor
    hard_actions: torch.Tensor
    hard_rate_symbols: torch.Tensor
    hard_protection: torch.Tensor
    hard_visual_symbols: torch.Tensor
    selected_mask: torch.Tensor
    hard_total_symbols: torch.Tensor
    soft_total_symbols: torch.Tensor
    budget_penalty: torch.Tensor


def _st_one_hot(logits: torch.Tensor, *, training: bool, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    soft = (logits / temperature).softmax(dim=-1)
    if training:
        hard_st = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        hard_index = hard_st.detach().argmax(dim=-1)
        return hard_st, hard_index
    hard_index = logits.argmax(dim=-1)
    hard = F.one_hot(hard_index, logits.shape[-1]).to(logits.dtype)
    # Straight-through form retains a gradient when a caller deliberately uses
    # eval-mode routing in a differentiable diagnostic.
    return hard - soft.detach() + soft, hard_index


class JointAssertionSymbolRouter(nn.Module):
    """Jointly choose visual symbols and per-entity action/rate/protection.

    The hard forward pass is guaranteed not to exceed ``total_budget``.  Its
    discrete budget projection is paired with differentiable soft selections
    and ``budget_penalty`` for training.
    """

    def __init__(
        self,
        token_dim: int,
        channel_dim: int,
        *,
        rate_options: Sequence[int] = (0, 8, 16, 32, 64),
        visual_rate_options: Sequence[int] = (0, 128, 256, 512),
        protection_extra_symbols: Sequence[int] = (0, 8, 4, 16),
        hidden_dim: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or channel_dim <= 0 or hidden_dim <= 0:
            raise ValueError("dimensions must be positive")
        if not rate_options or rate_options[0] != 0 or sorted(rate_options) != list(rate_options):
            raise ValueError("rate_options must be sorted and start with zero")
        if (
            not visual_rate_options
            or visual_rate_options[0] != 0
            or sorted(visual_rate_options) != list(visual_rate_options)
        ):
            raise ValueError("visual_rate_options must be sorted and start with zero")
        if len(protection_extra_symbols) != len(ProtectionAction):
            raise ValueError("protection cost table must match ProtectionAction")
        self.token_dim = int(token_dim)
        self.channel_dim = int(channel_dim)
        self.temperature = float(temperature)
        self.register_buffer("rate_options", torch.tensor(rate_options, dtype=torch.float32))
        self.register_buffer("visual_rate_options", torch.tensor(visual_rate_options, dtype=torch.float32))
        self.register_buffer(
            "protection_extra_symbols", torch.tensor(protection_extra_symbols, dtype=torch.float32)
        )
        context_dim = token_dim + channel_dim + 1
        self.context = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.GELU())
        self.action_head = nn.Linear(hidden_dim, ENTITY_ACTION_COUNT)
        self.rate_head = nn.Linear(hidden_dim, len(rate_options))
        self.protection_head = nn.Linear(hidden_dim, len(protection_extra_symbols))
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.visual_head = nn.Sequential(
            nn.Linear(token_dim + channel_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(visual_rate_options)),
        )

    @staticmethod
    def _budget_tensor(total_budget, batch: int, device) -> torch.Tensor:
        budget = torch.as_tensor(total_budget, device=device, dtype=torch.long)
        if budget.ndim == 0:
            budget = budget.expand(batch)
        if tuple(budget.shape) != (batch,) or (budget < 0).any():
            raise ValueError("total_budget must be non-negative scalar or [B]")
        return budget

    def forward(
        self,
        event_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        channel_features: torch.Tensor,
        total_budget,
        *,
        valid_mask: Optional[torch.Tensor] = None,
        semantic_states: Optional[torch.Tensor] = None,
    ) -> RouterOutput:
        if event_tokens.ndim != 3 or event_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"event_tokens must be [B,K,{self.token_dim}]")
        if visual_tokens.ndim != 3 or visual_tokens.shape[0] != event_tokens.shape[0]:
            raise ValueError("visual_tokens must be [B,L,d] with the same batch")
        if visual_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"visual token dim must be {self.token_dim}")
        batch, slots, _ = event_tokens.shape
        if tuple(channel_features.shape) != (batch, self.channel_dim):
            raise ValueError(f"channel_features must be [B,{self.channel_dim}]")
        device = event_tokens.device
        budget = self._budget_tensor(total_budget, batch, device)
        if valid_mask is None:
            valid_mask = torch.ones(batch, slots, dtype=torch.bool, device=device)
        if tuple(valid_mask.shape) != (batch, slots):
            raise ValueError("valid_mask must be [B,K]")
        valid_mask = valid_mask.bool()

        budget_ratio = budget.to(event_tokens.dtype) / max(
            1.0, float(self.visual_rate_options.max().item() + slots * self.rate_options.max().item())
        )
        shared = torch.cat([
            channel_features,
            budget_ratio.unsqueeze(-1),
        ], dim=-1).unsqueeze(1).expand(-1, slots, -1)
        hidden = self.context(torch.cat([event_tokens, shared], dim=-1))
        action_logits = self.action_head(hidden)
        rate_logits = self.rate_head(hidden)
        protection_logits = self.protection_head(hidden)
        utility = self.utility_head(hidden).squeeze(-1)

        if semantic_states is not None:
            if tuple(semantic_states.shape) != (batch, slots):
                raise ValueError("semantic_states must be [B,K]")
            unknown = semantic_states.eq(int(SemanticState.UNKNOWN))
            action_logits = action_logits.clone()
            action_logits[..., int(AssertionAction.REVOKE)] = action_logits[
                ..., int(AssertionAction.REVOKE)
            ].masked_fill(unknown, torch.finfo(action_logits.dtype).min)

        invalid = ~valid_mask
        action_logits = action_logits.masked_fill(invalid.unsqueeze(-1), torch.finfo(action_logits.dtype).min)
        action_logits[..., int(AssertionAction.SKIP)] = torch.where(
            invalid,
            torch.zeros_like(action_logits[..., int(AssertionAction.SKIP)]),
            action_logits[..., int(AssertionAction.SKIP)],
        )
        rate_logits = rate_logits.masked_fill(invalid.unsqueeze(-1), torch.finfo(rate_logits.dtype).min)
        rate_logits[..., 0] = torch.where(invalid, torch.zeros_like(rate_logits[..., 0]), rate_logits[..., 0])
        protection_logits = protection_logits.masked_fill(
            invalid.unsqueeze(-1), torch.finfo(protection_logits.dtype).min
        )
        protection_logits[..., int(ProtectionAction.NONE)] = torch.where(
            invalid,
            torch.zeros_like(protection_logits[..., int(ProtectionAction.NONE)]),
            protection_logits[..., int(ProtectionAction.NONE)],
        )

        action_weights, action_index = _st_one_hot(
            action_logits, training=self.training, temperature=self.temperature
        )
        rate_weights, rate_index = _st_one_hot(
            rate_logits, training=self.training, temperature=self.temperature
        )
        protection_weights, protection_index = _st_one_hot(
            protection_logits, training=self.training, temperature=self.temperature
        )

        pooled_visual = visual_tokens.mean(dim=1)
        visual_logits = self.visual_head(torch.cat([
            pooled_visual, channel_features, budget_ratio.unsqueeze(-1)
        ], dim=-1))
        visual_weights, visual_index = _st_one_hot(
            visual_logits, training=self.training, temperature=self.temperature
        )

        hard_visual = self.visual_rate_options[visual_index].long()
        hard_rates = self.rate_options[rate_index].long()
        hard_protection_cost = self.protection_extra_symbols[protection_index].long()
        hard_actions = action_index.long()
        hard_protection = protection_index.long()

        # Deterministic hard projection: clamp visual allocation to the largest
        # legal option and admit entity events by learned utility.
        projected_visual = torch.zeros_like(hard_visual)
        projected_actions = torch.full_like(hard_actions, int(AssertionAction.SKIP))
        projected_rates = torch.zeros_like(hard_rates)
        projected_protection = torch.full_like(hard_protection, int(ProtectionAction.NONE))
        selected = torch.zeros_like(valid_mask)
        hard_total = torch.zeros(batch, dtype=torch.long, device=device)
        for b in range(batch):
            legal_visual = self.visual_rate_options[self.visual_rate_options <= budget[b]]
            chosen_visual = min(int(hard_visual[b].item()), int(legal_visual.max().item()))
            projected_visual[b] = chosen_visual
            remaining = int(budget[b].item()) - chosen_visual
            order = torch.argsort(utility[b], descending=True)
            for index in order.tolist():
                action = int(hard_actions[b, index].item())
                if not bool(valid_mask[b, index]) or action == int(AssertionAction.SKIP):
                    continue
                cost = int(hard_rates[b, index].item() + hard_protection_cost[b, index].item())
                if cost <= 0 or cost > remaining:
                    continue
                projected_actions[b, index] = action
                projected_rates[b, index] = hard_rates[b, index]
                projected_protection[b, index] = hard_protection[b, index]
                selected[b, index] = True
                remaining -= cost
            hard_total[b] = int(budget[b].item()) - remaining

        soft_rates = (rate_weights * self.rate_options).sum(dim=-1)
        soft_protection = (protection_weights * self.protection_extra_symbols).sum(dim=-1)
        soft_non_skip = 1.0 - action_weights[..., int(AssertionAction.SKIP)]
        soft_visual = (visual_weights * self.visual_rate_options).sum(dim=-1)
        soft_total = soft_visual + ((soft_rates + soft_protection) * soft_non_skip * valid_mask).sum(dim=-1)
        budget_penalty = torch.relu(soft_total - budget.to(soft_total.dtype)).mean()

        # Replace ST hard values with the budget-projected hard decision while
        # preserving gradients from their soft distributions.
        projected_action_oh = F.one_hot(projected_actions, ENTITY_ACTION_COUNT).to(action_weights.dtype)
        action_weights = projected_action_oh - action_weights.detach() + action_weights
        rate_indices_projected = torch.zeros_like(rate_index)
        for i, value in enumerate(self.rate_options.tolist()):
            rate_indices_projected = torch.where(
                projected_rates.eq(int(value)), torch.full_like(rate_indices_projected, i), rate_indices_projected
            )
        projected_rate_oh = F.one_hot(rate_indices_projected, len(self.rate_options)).to(rate_weights.dtype)
        rate_weights = projected_rate_oh - rate_weights.detach() + rate_weights
        projected_protection_oh = F.one_hot(
            projected_protection, len(self.protection_extra_symbols)
        ).to(protection_weights.dtype)
        protection_weights = projected_protection_oh - protection_weights.detach() + protection_weights

        return RouterOutput(
            action_logits=action_logits,
            rate_logits=rate_logits,
            protection_logits=protection_logits,
            visual_rate_logits=visual_logits,
            action_weights=action_weights,
            rate_weights=rate_weights,
            protection_weights=protection_weights,
            visual_rate_weights=visual_weights,
            hard_actions=projected_actions,
            hard_rate_symbols=projected_rates,
            hard_protection=projected_protection,
            hard_visual_symbols=projected_visual,
            selected_mask=selected,
            hard_total_symbols=hard_total,
            soft_total_symbols=soft_total,
            budget_penalty=budget_penalty,
        )
