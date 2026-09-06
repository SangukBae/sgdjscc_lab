"""Pure logic for the fixed_int4 vs fixed_int6 negative-semantics bridge.

Kept separate from ``negative_semantics.py`` because this module aggregates
per-video *transmission accounting* (exact bytes, PSNR/SSIM/LPIPS, latency)
across bit-depth conditions, not detector presence/absence rows. Both modules
are pure (no torch/GPU import) so they stay usable from report-only tooling
and are fast to unit test.

Every metric comparison here is an **inner-joined, paired-by-video**
difference (``paired_delta_from_video_values`` / ``paired_condition_metrics``),
not a difference of two independently-computed means: fixed_int4 and
fixed_int6 reconstruct the same 40 videos, so the correlated, per-video
`candidate - reference` differences have much smaller variance than the
naive `mean(candidate) - mean(reference)` would suggest, and pairing also
catches a missing/duplicated video before it silently distorts an aggregate.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


def bootstrap_mean_ci(
    values: Sequence[float], *, iterations: int, seed: int = 7301
) -> Dict[str, Any]:
    """Video-clustered bootstrap 95% CI on the mean of ``values``.

    Each element of ``values`` is assumed to already be one aggregate number
    per video (e.g. mean_psnr for that video, or a per-video paired delta),
    so resampling elements with replacement is equivalent to resampling
    videos with replacement.
    """
    values = list(values)
    if not values:
        return {"lower": None, "upper": None}
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(int(iterations)):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    lower = samples[int(0.025 * (len(samples) - 1))]
    upper = samples[int(0.975 * (len(samples) - 1))]
    return {"lower": lower, "upper": upper}


def video_clustered_bootstrap_mean(
    values_by_video: Mapping[str, Sequence[float]], *, iterations: int, seed: int = 7301,
) -> Dict[str, Any]:
    """Bootstrap 95% CI on the mean of all values, resampling at the *video*
    level (with replacement) and including every value that video
    contributes for each resampled draw.

    This matters whenever a video can contribute more than one point (e.g.
    two EXIT ghost-track events in the same video): those points are not
    independent draws, so ``bootstrap_mean_ci`` -- which resamples flattened
    points directly -- would understate the true variance by treating
    same-video points as if they could be resampled independently of each
    other. Resampling video keys first preserves that within-video
    correlation.
    """
    videos = sorted(values_by_video)
    if not videos:
        return {"lower": None, "upper": None}
    all_values = [value for video in videos for value in values_by_video[video]]
    if not all_values:
        return {"lower": None, "upper": None}
    rng = random.Random(seed)
    samples = []
    for _ in range(int(iterations)):
        resampled: List[float] = []
        for _ in range(len(videos)):
            video = videos[rng.randrange(len(videos))]
            resampled.extend(values_by_video[video])
        if resampled:
            samples.append(sum(resampled) / len(resampled))
    if not samples:
        return {"lower": None, "upper": None}
    samples.sort()
    lower = samples[int(0.025 * (len(samples) - 1))]
    upper = samples[int(0.975 * (len(samples) - 1))]
    return {"lower": lower, "upper": upper}


def paired_delta_video_clustered(
    reference_by_video: Mapping[str, Sequence[float]],
    candidate_by_video: Mapping[str, Sequence[float]],
    *,
    bootstrap_iterations: int = 2000,
) -> Dict[str, Any]:
    """Paired ``candidate - reference`` delta per (video, point-within-video)
    -- e.g. one point per EXIT event -- with a video-clustered bootstrap CI
    on the mean delta (see ``video_clustered_bootstrap_mean``). Fails closed
    if the video sets differ or a video's point count differs between the
    two sides (its points must already be in the same order on both sides,
    e.g. sorted by event_id by the caller).
    """
    reference_ids = set(reference_by_video)
    candidate_ids = set(candidate_by_video)
    if reference_ids != candidate_ids:
        raise ValueError(
            "paired comparison requires identical video sets: "
            f"missing_in_candidate={sorted(reference_ids - candidate_ids)}, "
            f"missing_in_reference={sorted(candidate_ids - reference_ids)}"
        )
    if not reference_ids:
        raise ValueError("paired comparison requires at least one video")
    delta_by_video: Dict[str, List[float]] = {}
    for video in sorted(reference_ids):
        reference_values = list(reference_by_video[video])
        candidate_values = list(candidate_by_video[video])
        if len(reference_values) != len(candidate_values):
            raise ValueError(
                f"video={video!r}: point count differs between reference "
                f"({len(reference_values)}) and candidate ({len(candidate_values)})"
            )
        delta_by_video[video] = [c - r for r, c in zip(reference_values, candidate_values)]
    all_deltas = [delta for video in delta_by_video for delta in delta_by_video[video]]
    return {
        "n_videos": len(delta_by_video),
        "n_points": len(all_deltas),
        "per_video_delta": delta_by_video,
        "mean_delta": sum(all_deltas) / len(all_deltas) if all_deltas else None,
        "bootstrap_95ci": video_clustered_bootstrap_mean(delta_by_video, iterations=bootstrap_iterations),
    }


def validate_unique_keys(
    rows: Iterable[Mapping[str, Any]], *, key_of: Callable[[Mapping[str, Any]], Any], label: str,
) -> None:
    """Fail closed if ``key_of`` produces the same key for more than one row."""
    seen: Dict[Any, int] = {}
    for row in rows:
        key = key_of(row)
        seen[key] = seen.get(key, 0) + 1
    duplicates = sorted((key, count) for key, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(f"{label}: duplicate key(s) found: {duplicates[:10]}")


def validate_matching_video_sets(video_sets: Mapping[str, Any], *, label: str) -> None:
    """Fail closed unless every named set of video IDs in ``video_sets`` is
    exactly equal -- not just the same size. A count-only check (e.g.
    ``validate_exact_video_count`` on two conditions that both happen to
    have 40 videos) cannot catch two conditions covering *different* 40
    videos; this compares the actual sets pairwise.
    """
    names = sorted(video_sets)
    if len(names) < 2:
        return
    reference_name = names[0]
    reference_set = set(video_sets[reference_name])
    for name in names[1:]:
        candidate_set = set(video_sets[name])
        if candidate_set != reference_set:
            raise ValueError(
                f"{label}: video ID set for {name!r} differs from {reference_name!r} "
                f"(only_in_{name}={sorted(candidate_set - reference_set)}, "
                f"only_in_{reference_name}={sorted(reference_set - candidate_set)})"
            )


def validate_gt_present_consistency(
    rows_a: Iterable[Mapping[str, Any]], rows_b: Iterable[Mapping[str, Any]],
    *, key_of: Callable[[Mapping[str, Any]], Any], label_a: str, label_b: str,
) -> None:
    """Fail closed if the same ``(video, frame, concept)`` key has a
    different ``gt_present`` value in ``rows_a`` vs ``rows_b``. Ground truth
    does not depend on bit depth, so any disagreement means the two row sets
    were not actually scored against the same ground truth (wrong gt_index,
    a mislabeled concept, or a frame-alignment bug).
    """
    gt_a = {key_of(row): bool(row["gt_present"]) for row in rows_a}
    gt_b = {key_of(row): bool(row["gt_present"]) for row in rows_b}
    shared_keys = set(gt_a) & set(gt_b)
    mismatched = sorted(key for key in shared_keys if gt_a[key] != gt_b[key])
    if mismatched:
        raise ValueError(
            f"gt_present disagrees between {label_a!r} and {label_b!r} for key(s): "
            f"{mismatched[:10]}"
        )


def coerce_finite_float(value: Any, *, context: str) -> float:
    """Parse ``value`` as a float, failing closed on blank/missing/NaN/Inf.

    ``run_transmission_reduction_eval.py`` writes a blank string (not "nan")
    for a metric it did not compute for a given run (e.g. ``mean_lpips`` is
    ``""`` whenever the child run passed ``--no-lpips``). Silently coercing
    that to ``0.0`` or ``NaN`` would corrupt a mean/CI without any signal;
    raising here forces the caller to either supply a real value or exclude
    the field for that condition.
    """
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValueError(f"{context}: missing/blank numeric value")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: could not parse {value!r} as a float") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{context}: non-finite value {parsed!r}")
    return parsed


def validate_exact_video_count(
    video_ids: Sequence[str], *, expected: int, label: str
) -> None:
    """Fail closed unless ``video_ids`` has exactly ``expected`` unique entries.

    Used to enforce "formal means exactly 40 videos per condition, smoke
    means exactly 1" rather than silently accepting a partial/duplicated set.
    """
    counts = Counter(video_ids)
    duplicates = sorted(video_id for video_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{label}: duplicate video_id(s) {duplicates}")
    if len(counts) != expected:
        raise ValueError(f"{label}: expected exactly {expected} video_id(s), got {len(counts)}: {sorted(counts)}")


def one_value_per_video(
    rows: Iterable[Mapping[str, Any]],
    *,
    condition_of: Callable[[Mapping[str, Any]], str],
    video_id_of: Callable[[Mapping[str, Any]], str],
    condition: str,
    field: str,
) -> Dict[str, float]:
    """Extract ``{video_id: value}`` for one (condition, field), failing
    closed on a duplicate video within the condition or a blank/NaN/Inf value.
    """
    values: Dict[str, float] = {}
    for row in rows:
        if condition_of(row) != condition:
            continue
        video_id = video_id_of(row)
        if video_id in values:
            raise ValueError(f"condition={condition!r} field={field!r}: duplicate video_id {video_id!r}")
        values[video_id] = coerce_finite_float(
            row[field], context=f"condition={condition!r} field={field!r} video_id={video_id!r}"
        )
    return values


def paired_delta_from_video_values(
    reference_values: Mapping[str, float],
    candidate_values: Mapping[str, float],
    *,
    bootstrap_iterations: int = 2000,
) -> Dict[str, Any]:
    """Inner-join two ``{video_id: value}`` maps and summarize the per-video
    ``candidate - reference`` difference: fails closed if the video sets
    differ, otherwise returns the paired deltas, their mean, and a
    video-clustered bootstrap 95% CI on that mean.
    """
    reference_ids = set(reference_values)
    candidate_ids = set(candidate_values)
    if reference_ids != candidate_ids:
        raise ValueError(
            "paired comparison requires identical video sets: "
            f"missing_in_candidate={sorted(reference_ids - candidate_ids)}, "
            f"missing_in_reference={sorted(candidate_ids - reference_ids)}"
        )
    if not reference_ids:
        raise ValueError("paired comparison requires at least one video")
    video_ids = sorted(reference_ids)
    per_video_delta = {
        video_id: candidate_values[video_id] - reference_values[video_id]
        for video_id in video_ids
    }
    deltas = [per_video_delta[video_id] for video_id in video_ids]
    return {
        "n_videos": len(video_ids),
        "video_ids": video_ids,
        "per_video_delta": per_video_delta,
        "mean_delta": sum(deltas) / len(deltas),
        "bootstrap_95ci": bootstrap_mean_ci(deltas, iterations=bootstrap_iterations),
    }


def paired_condition_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    condition_of: Callable[[Mapping[str, Any]], str],
    video_id_of: Callable[[Mapping[str, Any]], str],
    reference: str,
    candidate: str,
    fields: Sequence[str],
    bootstrap_iterations: int = 2000,
) -> Dict[str, Dict[str, Any]]:
    """Per-field paired ``candidate - reference`` delta from row-shaped input
    (e.g. ``per_video_metrics.csv`` rows). See ``paired_delta_from_video_values``
    for the join/statistics contract each field goes through.
    """
    rows = list(rows)
    result: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        reference_values = one_value_per_video(
            rows, condition_of=condition_of, video_id_of=video_id_of, condition=reference, field=field,
        )
        candidate_values = one_value_per_video(
            rows, condition_of=condition_of, video_id_of=video_id_of, condition=candidate, field=field,
        )
        result[field] = paired_delta_from_video_values(
            reference_values, candidate_values, bootstrap_iterations=bootstrap_iterations,
        )
    return result


def summarize_condition_metrics(
    per_video_rows: Iterable[Mapping[str, Any]],
    *,
    condition_of: Callable[[Mapping[str, Any]], str],
    video_id_of: Callable[[Mapping[str, Any]], str],
    fields: Sequence[str],
    bootstrap_iterations: int = 2000,
) -> Dict[str, Any]:
    """Group per-video metric rows by condition and summarize each field.

    ``condition_of``/``video_id_of`` map a row to its condition label (e.g.
    ``f"{bit_depth}__{decoder}"``) and video id. Fails closed (``ValueError``)
    on a duplicate video within a condition or a blank/NaN/Inf field value
    rather than silently producing a mean over garbage input -- see
    ``coerce_finite_float``.
    """
    by_condition: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_video_rows:
        by_condition[condition_of(row)].append(row)
    result: Dict[str, Any] = {}
    for condition, rows in by_condition.items():
        entry: Dict[str, Any] = {"n_videos": len(rows)}
        for field in fields:
            values_by_video = one_value_per_video(
                rows, condition_of=condition_of, video_id_of=video_id_of, condition=condition, field=field,
            )
            values = list(values_by_video.values())
            entry[field] = {
                "mean": sum(values) / len(values) if values else None,
                "bootstrap_95ci": bootstrap_mean_ci(values, iterations=bootstrap_iterations),
            }
        result[condition] = entry
    return result


def quality_gate_pass(
    delta: Mapping[str, float],
    *,
    psnr_drop_db_max: float,
    ssim_drop_max: float,
    lpips_rise_max: float,
) -> Dict[str, Any]:
    """Apply the repo's standard quality gate to a fixed_int6-vs-fixed_int4
    paired mean delta (e.g. ``paired_condition_metrics(...)["mean_psnr"]["mean_delta"]``).

    Matches the thresholds already used for bit-depth comparisons in
    ``scripts/run_transmission_reduction_eval.py`` (``quality_gate`` in each
    child run's ``summary.json``): PSNR must not drop more than
    ``psnr_drop_db_max``, SSIM not more than ``ssim_drop_max``, and LPIPS
    (lower is better) must not rise more than ``lpips_rise_max``.
    """
    psnr_delta = delta.get("mean_psnr")
    ssim_delta = delta.get("mean_ssim")
    lpips_delta = delta.get("mean_lpips")
    checks = {
        "psnr_within_budget": psnr_delta is not None and psnr_delta >= -psnr_drop_db_max,
        "ssim_within_budget": ssim_delta is not None and ssim_delta >= -ssim_drop_max,
        "lpips_within_budget": lpips_delta is not None and lpips_delta <= lpips_rise_max,
    }
    return {**checks, "passed": all(checks.values())}
