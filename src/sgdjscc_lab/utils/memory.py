"""memory.py – CUDA memory management helpers."""

from __future__ import annotations

import gc

import torch


def release_cuda_memory() -> None:
    """Best-effort CUDA cache cleanup after temporary model offloading."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
