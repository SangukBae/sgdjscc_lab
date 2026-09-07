"""Runtime bridge that injects signed-memory adapters into Transformer blocks."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import torch
import torch.nn as nn

from .signed_memory_dit import SignedMemoryDiTAdapterStack


@dataclass(frozen=True)
class SignedMemoryCondition:
    timestep_embedding: torch.Tensor
    channel_tokens: torch.Tensor
    active_memory: torch.Tensor
    negative_memory: torch.Tensor
    active_mask: Optional[torch.Tensor] = None
    negative_mask: Optional[torch.Tensor] = None
    channel_reliability: Optional[torch.Tensor] = None
    assertion_confidence: Optional[torch.Tensor] = None


class SignedMemoryBackboneBridge(nn.Module):
    """Attach trainable SM-DiT adapters after selected backbone blocks.

    The bridge owns the adapter parameters and forward-hook handles. A caller
    activates them only inside :meth:`condition`; outside that context the
    attached backbone is bit-for-bit unchanged. The hook accepts Tensor or
    tuple outputs and fails closed on unsupported structures.
    """

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
        self.adapters = SignedMemoryDiTAdapterStack(
            injection_layers,
            hidden_dim=hidden_dim,
            memory_dim=memory_dim,
            channel_dim=channel_dim,
            timestep_dim=timestep_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
        )
        self._condition: ContextVar[Optional[SignedMemoryCondition]] = ContextVar(
            f"saver_condition_{id(self)}", default=None
        )
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    def attach(self, blocks: Sequence[nn.Module]) -> None:
        if self.attached:
            raise RuntimeError("bridge is already attached")
        if not isinstance(blocks, (nn.ModuleList, list, tuple)):
            raise TypeError("blocks must be a ModuleList/list/tuple of modules")
        if self.adapters.injection_layers[-1] >= len(blocks):
            raise ValueError("an injection layer is outside the backbone block range")
        for layer_index in self.adapters.injection_layers:
            block = blocks[layer_index]
            if not isinstance(block, nn.Module):
                raise TypeError(f"blocks[{layer_index}] is not an nn.Module")
            self._handles.append(block.register_forward_hook(self._make_hook(layer_index)))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            condition = self._condition.get()
            if condition is None:
                return output
            if isinstance(output, torch.Tensor):
                hidden = output
                rebuild = lambda value: value
            elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                hidden = output[0]
                rebuild = lambda value: (value,) + output[1:]
            else:
                raise TypeError(
                    "SM-DiT bridge requires each selected block to return a Tensor "
                    "or a tuple whose first item is a Tensor"
                )
            adapted = self.adapters.inject(
                layer_index,
                hidden,
                timestep_embedding=condition.timestep_embedding,
                channel_tokens=condition.channel_tokens,
                active_memory=condition.active_memory,
                negative_memory=condition.negative_memory,
                active_mask=condition.active_mask,
                negative_mask=condition.negative_mask,
                channel_reliability=condition.channel_reliability,
                assertion_confidence=condition.assertion_confidence,
            ).hidden_states
            return rebuild(adapted)

        return hook

    @contextmanager
    def condition(self, condition: SignedMemoryCondition) -> Iterator[None]:
        if not self.attached:
            raise RuntimeError("attach the bridge before activating a condition")
        token = self._condition.set(condition)
        try:
            yield
        finally:
            self._condition.reset(token)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "SignedMemoryBackboneBridge is not called directly; attach it to "
            "backbone blocks and use its condition context manager"
        )
