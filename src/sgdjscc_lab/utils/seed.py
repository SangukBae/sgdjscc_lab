"""seed.py – Reproducibility seed helper."""

from __future__ import annotations

import random
import zlib

import numpy as np
import torch


def set_global_seed(seed: int = 2025) -> None:
    """Set random seeds for Python, NumPy, PyTorch, and cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def derive_frame_seed(base_seed: int, video_key: str, frame_index: int) -> int:
    """Deterministic per-(video, frame) seed derived from *base_seed*.

    Two different configs (e.g. ``fixed_int4`` vs ``fixed_int8``) reconstructing
    the SAME ``(video_key, frame_index)`` get the IDENTICAL seed here, so any
    residual RNG-dependent step (e.g. a diffusion sampler's initial noise) is
    aligned across configs being compared — the comparison isolates the
    channel/quantization/selector difference instead of also mixing in
    incidental RNG-draw differences. Two different frames (or videos) get
    different seeds, so a run does not silently reuse one RNG state throughout
    an entire sweep. Deterministic and reproducible: the same
    ``(base_seed, video_key, frame_index)`` always derives the same value,
    across processes/machines (``zlib.crc32`` is not seed/PYTHONHASHSEED
    dependent, unlike the builtin ``hash()``).
    """
    key = f"{base_seed}:{video_key}:{frame_index}".encode("utf-8")
    return int(base_seed) ^ zlib.crc32(key)
