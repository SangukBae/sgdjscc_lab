"""utils/metrics_io.py – Metric aggregation and serialisation helpers.

Functions
---------
summarize_metrics(rows)
    Compute per-column mean and std across a list of per-image metric dicts.

flatten_metric_dict(d, prefix)
    Recursively flatten a nested dict (e.g. SRS submetrics) into a flat dict.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def summarize_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and std for each numeric column across per-image rows.

    Parameters
    ----------
    rows:
        List of dicts, one per image.  All dicts should share the same keys
        (extra keys in some rows are silently ignored during per-key iteration).

    Returns
    -------
    dict with keys ``<col>_mean`` and ``<col>_std`` for each numeric column,
    plus ``n`` (row count).  Non-numeric or all-None columns are skipped.

    Example
    -------
    >>> rows = [{"psnr": 30.0, "ssim": 0.9}, {"psnr": 28.0, "ssim": 0.85}]
    >>> summarize_metrics(rows)
    {"psnr_mean": 29.0, "psnr_std": 1.0, "ssim_mean": 0.875, "ssim_std": 0.025, "n": 2}
    """
    if not rows:
        return {"n": 0}

    # Collect all column names across all rows
    all_keys = {k for row in rows for k in row}

    summary: Dict[str, Any] = {"n": len(rows)}

    for key in sorted(all_keys):
        values = [row[key] for row in rows if key in row and row[key] is not None]
        try:
            nums = [float(v) for v in values]
        except (TypeError, ValueError):
            continue  # skip non-numeric columns

        if not nums:
            continue

        n = len(nums)
        mean = sum(nums) / n
        variance = sum((x - mean) ** 2 for x in nums) / max(n - 1, 1)
        std = math.sqrt(variance)

        summary[f"{key}_mean"] = round(mean, 6)
        summary[f"{key}_std"]  = round(std, 6)

    return summary


def flatten_metric_dict(
    d: Dict[str, Any],
    prefix: str = "",
    sep: str = "_",
) -> Dict[str, Any]:
    """Recursively flatten a nested metric dict into a single-level dict.

    Parameters
    ----------
    d:
        Possibly nested dict (e.g. from SemanticReliabilityEvaluator.evaluate()).
    prefix:
        String prepended to each flattened key.
    sep:
        Separator between prefix and child key (default ``'_'``).

    Returns
    -------
    Flat dict with compound keys.

    Example
    -------
    >>> flatten_metric_dict({"srs": {"clip": 0.8, "weights": {"w_img": 0.3}}})
    {"srs_clip": 0.8, "srs_weights_w_img": 0.3}
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_metric_dict(v, prefix=full_key, sep=sep))
        else:
            out[full_key] = v
    return out


def format_summary_table(summary: Dict[str, Any]) -> str:
    """Format a summary dict as a readable table string for console output.

    Parameters
    ----------
    summary:
        Output of ``summarize_metrics()``.

    Returns
    -------
    Multi-line string with columns: metric, mean, std.
    """
    lines = [f"{'Metric':<40} {'Mean':>12} {'Std':>12}"]
    lines.append("-" * 66)

    n = summary.pop("n", "?")
    mean_keys = [k for k in sorted(summary) if k.endswith("_mean")]
    for mk in mean_keys:
        base = mk[:-5]   # strip "_mean"
        std_key = f"{base}_std"
        mean_val = summary[mk]
        std_val  = summary.get(std_key, 0.0)
        lines.append(f"  {base:<38} {mean_val:>12.4f} {std_val:>12.4f}")

    lines.append("-" * 66)
    lines.append(f"  n = {n}")
    summary["n"] = n   # restore
    return "\n".join(lines)
