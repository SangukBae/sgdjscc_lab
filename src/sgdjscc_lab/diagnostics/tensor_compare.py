"""diagnostics/tensor_compare.py – Pairwise tensor comparison for diagnostics.

Given two tensors captured at the SAME named stage by two different Tx/Rx
paths (see ``tensor_recorder.py``), computes exact equality, max/mean
absolute error, MSE, cosine similarity and norm ratio. Used to build
``tensor_pair_comparison.csv``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def compare_tensors(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Dict[str, Any]:
    """Compare *a* (reference/first path) against *b* (second path).

    Returns ``comparable: False`` with a ``reason`` when either tensor is
    missing, not a tensor, or the shapes differ — never a fabricated number
    for an incomparable pair.
    """
    if a is None or b is None:
        return {"comparable": False, "reason": "missing_tensor"}
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return {"comparable": False, "reason": "non_tensor"}
    if tuple(a.shape) != tuple(b.shape):
        return {
            "comparable": False,
            "reason": f"shape_mismatch {tuple(a.shape)} vs {tuple(b.shape)}",
        }

    a_cpu = a.detach().cpu()
    b_cpu = b.detach().cpu()
    exact_equal = bool(torch.equal(a_cpu, b_cpu))

    af = a_cpu.float()
    bf = b_cpu.float()
    a_finite = bool(torch.isfinite(af).all().item())
    b_finite = bool(torch.isfinite(bf).all().item())
    if not (a_finite and b_finite):
        return {
            "comparable": True,
            "exact_equal": exact_equal,
            "both_finite": False,
            "max_abs_err": None, "mean_abs_err": None, "mse": None,
            "cosine_similarity": None, "norm_a": None, "norm_b": None,
            "norm_ratio": None,
        }

    diff = af - bf
    max_abs_err = float(diff.abs().max().item())
    mean_abs_err = float(diff.abs().mean().item())
    mse = float((diff ** 2).mean().item())
    norm_a = float(torch.linalg.norm(af).item())
    norm_b = float(torch.linalg.norm(bf).item())
    if norm_a > 0 and norm_b > 0:
        cosine = float(
            torch.dot(af.flatten(), bf.flatten()).item() / (norm_a * norm_b)
        )
    else:
        cosine = None
    if norm_a > 0:
        norm_ratio = norm_b / norm_a
    elif norm_b == 0:
        norm_ratio = 1.0
    else:
        norm_ratio = None

    return {
        "comparable": True,
        "exact_equal": exact_equal,
        "both_finite": True,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "mse": mse,
        "cosine_similarity": cosine,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "norm_ratio": norm_ratio,
    }
