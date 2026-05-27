"""seed.py – Reproducibility seed helper."""

from __future__ import annotations

import random

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
