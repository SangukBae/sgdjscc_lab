"""pipelines/generation_mode_comparison.py – start-only vs bidirectional
generate-mode comparison pipeline (ETRI 4차, step 6).

Runs the *same* frame sequence through a start-only ``TemporalPipeline`` and a
bidirectional ``TemporalPipeline`` and diffs their temporal metrics
(``PTC``/``SFR``/``SDI``) and decision counts (``n_generate`` /
``n_recompute_semantic`` / ``n_recompute_motion`` / ``n_reused``).

Scope note (read before citing a number from this module's output)
--------------------------------------------------------------------
Both pipelines use **mock** generator backends (ETRI 3차/4차 —
``video/video_generator.py``: copy/interpolation for start-only,
``BidirectionalInterpolationGenerator`` for bidirectional). This module only
proves the two conditioning modes can be run and diffed side by side in one
pipeline; **it does not certify bidirectional as reducing drift/flicker, and
its output is not a generation-quality claim.** A real quality/drift
comparison needs a real generator backend (5차+ follow-up per
docs/etri_strategy.md).

This module does not build ``reconstruct_fn``/``packet_fn``/models itself —
those are supplied via ``pipeline_factory`` (injected by the caller, typically
``scripts/evaluate_video.py``) so the comparison logic here stays testable
without SGD-JSCC checkpoints or CLIP.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# The metrics this module diffs — PTC/SFR/SDI (temporal_consistency.evaluate_sequence)
# plus the reuse/recompute/generate decision counts (TemporalPipeline._summarize).
COMPARISON_KEYS = (
    "ptc", "sfr", "sdi",
    "n_generate", "n_reused", "n_recompute_semantic", "n_recompute_motion",
)

_SCOPE_NOTE = (
    "Mock generator backends in both modes (ETRI 4차) — this comparison proves "
    "the pipeline runs and diffs are computed; it is NOT a generation-quality "
    "or drift-reduction claim. See docs/etri_strategy.md 4차 구현 결과."
)


def mode_metrics(result: Dict) -> Dict:
    """Compute the temporal-consistency + decision-count metrics for one
    ``TemporalPipeline.run()`` result (same computation evaluate_video.py uses
    for its primary ``temporal_metrics.csv``)."""
    from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence

    metrics = evaluate_sequence(result["records"])
    metrics.update(result["summary"])
    return metrics


def compare_metrics(start_only: Dict, bidirectional: Dict) -> Dict:
    """Build a flat, JSON/CSV-friendly diff dict over :data:`COMPARISON_KEYS`.

    For each key, emits ``{key}_start_only``, ``{key}_bidirectional`` and
    ``{key}_diff`` (``bidirectional - start_only``; ``None`` when either side
    is missing/non-numeric).
    """
    out: Dict = {}
    for key in COMPARISON_KEYS:
        a = start_only.get(key)
        b = bidirectional.get(key)
        out[f"{key}_start_only"] = a
        out[f"{key}_bidirectional"] = b
        diff = None
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            diff = b - a
        out[f"{key}_diff"] = diff
    out["note"] = _SCOPE_NOTE
    return out


def _write_metrics_csv(metrics: Dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)


def run_comparison(
    frames: List,
    pipeline_factory: Callable[[str], "TemporalPipeline"],  # noqa: F821 (forward ref, no import needed)
    output_json: Optional[str] = None,
    start_only_csv: Optional[str] = None,
    bidirectional_csv: Optional[str] = None,
) -> Dict:
    """Run the same *frames* through a start-only and a bidirectional pipeline.

    Parameters
    ----------
    frames:
        Ordered frame tensors (identical input to both runs).
    pipeline_factory:
        ``(conditioning_mode: "start_only" | "bidirectional") ->
        TemporalPipeline``. Left to the caller so this module never needs to
        know how to build ``reconstruct_fn``/``packet_fn``/models.
    output_json / start_only_csv / bidirectional_csv:
        Optional output paths. When given, the corresponding artefact is
        written; when ``None`` that artefact is skipped (still returned in the
        result dict either way).

    Returns
    -------
    dict with ``start_only`` (metrics dict), ``bidirectional`` (metrics dict),
    and ``comparison`` (the diff dict from :func:`compare_metrics`).
    """
    pipe_start = pipeline_factory("start_only")
    result_start = pipe_start.run(frames)
    metrics_start = mode_metrics(result_start)

    pipe_bidi = pipeline_factory("bidirectional")
    result_bidi = pipe_bidi.run(frames)
    metrics_bidi = mode_metrics(result_bidi)

    comparison = compare_metrics(metrics_start, metrics_bidi)

    if start_only_csv:
        _write_metrics_csv(metrics_start, start_only_csv)
        logger.info("start_only temporal metrics → %s", start_only_csv)
    if bidirectional_csv:
        _write_metrics_csv(metrics_bidi, bidirectional_csv)
        logger.info("bidirectional temporal metrics → %s", bidirectional_csv)
    if output_json:
        p = Path(output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(
                {"start_only": metrics_start, "bidirectional": metrics_bidi, "comparison": comparison},
                fh, indent=2,
            )
        logger.info("Generation-mode comparison → %s", output_json)

    return {"start_only": metrics_start, "bidirectional": metrics_bidi, "comparison": comparison}
