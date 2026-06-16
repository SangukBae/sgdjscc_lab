"""utils/metric_profiles.py – Evaluation metric profiles (paper vs extended).

The SGD-JSCC paper (Sec. VI "Performance Metrics") reports a specific metric set
— **PSNR, LPIPS, CLIP score, FID** — to evaluate reconstruction + perceptual
quality. The `sgdjscc_lab` evaluation stack additionally computes ETRI-oriented
metrics (SSIM, object preservation, hallucination, SRS, …) that are NOT part of
the paper's set.

To keep paper comparisons honest, this module separates two profiles:

  ``paper``     PSNR / LPIPS / CLIP(img-img, txt-img) / FID   ← the paper's set
  ``extended``  + SSIM, object preservation/miss/add, hallucination, SRS
  ``full``      ``extended`` + FID

SSIM is **kept** (it is in ``extended``/``full``) but is explicitly flagged as a
non-paper metric via :data:`NON_PAPER_METRICS`, so reports can mark it.
"""

from __future__ import annotations

from typing import List, Set

# ── Paper metric set (Sec. VI) ────────────────────────────────────────────────
PAPER_METRICS: Set[str] = {
    "psnr", "lpips", "clip_image_image", "clip_text_image", "fid",
}

# ── ETRI/extended additions (NOT in the paper's reported set) ─────────────────
EXTENDED_ONLY_METRICS: Set[str] = {
    "ssim",
    "object_preservation_rate", "missing_object_rate", "additional_object_rate",
    "hallucination_score", "semantic_reliability_score",
}

# ── Profiles ──────────────────────────────────────────────────────────────────
# extended intentionally EXCLUDES fid by default (heavy / set-level); use "full".
EXTENDED_METRICS: Set[str] = (PAPER_METRICS - {"fid"}) | EXTENDED_ONLY_METRICS
FULL_METRICS: Set[str] = EXTENDED_METRICS | {"fid"}

METRIC_PROFILES = {
    "paper": PAPER_METRICS,
    "extended": EXTENDED_METRICS,
    "full": FULL_METRICS,
}
VALID_PROFILES = tuple(METRIC_PROFILES.keys())

#: Metrics that are NOT part of the paper's reported set (flag in reports).
NON_PAPER_METRICS: Set[str] = EXTENDED_ONLY_METRICS

#: Canonical CSV column order for the base metrics.
ORDERED_METRIC_COLUMNS: List[str] = [
    "psnr", "ssim", "lpips",
    "clip_image_image", "clip_text_image",
    "object_preservation_rate", "missing_object_rate", "additional_object_rate",
    "hallucination_score", "semantic_reliability_score",
    "fid",
]


def resolve_profile(name: str) -> Set[str]:
    """Return the enabled-metric set for *name* (``paper`` | ``extended`` | ``full``)."""
    key = str(name).lower().strip()
    if key not in METRIC_PROFILES:
        raise ValueError(
            f"Unknown metric profile {name!r}. Valid: {', '.join(VALID_PROFILES)}."
        )
    return set(METRIC_PROFILES[key])


def columns_for_metrics(metrics) -> List[str]:
    """Ordered CSV columns for an explicit *metrics* set.

    ``filename, snr_db`` + the canonical-ordered metrics present in *metrics*.
    When ``fid`` is selected, the ``fid_backend`` provenance column is appended so
    a results file records whether FID came from Inception or a proxy backend.

    Deriving columns from the FINAL enabled set (rather than a raw profile) keeps
    the CSV schema consistent when the set is narrowed afterwards (e.g. --no-clip).
    """
    metrics = set(metrics)
    cols = ["filename", "snr_db"] + [m for m in ORDERED_METRIC_COLUMNS if m in metrics]
    if "fid" in metrics:
        cols.append("fid_backend")
    return cols


def profile_columns(name: str) -> List[str]:
    """Ordered CSV columns for a named profile (see :func:`columns_for_metrics`)."""
    return columns_for_metrics(resolve_profile(name))


def is_paper_metric(metric: str) -> bool:
    """True if *metric* is part of the paper's reported set."""
    return metric in PAPER_METRICS
