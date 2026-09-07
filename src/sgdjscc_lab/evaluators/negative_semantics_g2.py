"""Pure, GPU-free metrics for G2 Oracle ABSENT controllability."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    index = min(len(ordered) - 1, max(0, int(math.floor(probability * (len(ordered) - 1)))))
    return ordered[index]


def _validate_detection_grid(
    rows: Sequence[Mapping[str, Any]], arms: Sequence[str], policies: Sequence[str],
) -> None:
    expected_arms = set(arms)
    expected_policies = set(policies)
    if {str(row["arm"]) for row in rows} != expected_arms:
        raise ValueError("detection rows do not contain exactly the declared G2 arms")
    if {str(row["policy"]) for row in rows} != expected_policies:
        raise ValueError("detection rows do not contain exactly the declared G2 policies")
    by_arm: Dict[str, set] = defaultdict(set)
    for row in rows:
        key = (
            str(row["video_id"]), str(row["policy"]), int(row["seed"]),
            int(row["frame_index"]), str(row["concept_id"]),
        )
        arm = str(row["arm"])
        if key in by_arm[arm]:
            raise ValueError(f"duplicate G2 detection key in arm {arm}: {key}")
        by_arm[arm].add(key)
        if not isinstance(row.get("gt_present"), bool):
            raise ValueError("gt_present must be bool in every G2 detection row")
        for field in ("score", "source_score"):
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {field} in G2 detection rows")
    reference = by_arm[arms[0]]
    for arm in arms[1:]:
        if by_arm[arm] != reference:
            raise ValueError(f"G2 detection key grid differs for arm {arm}")


def _h_counts(rows: Sequence[Mapping[str, Any]], threshold: float) -> Dict[str, int | float | None]:
    absent = [row for row in rows if not bool(row["gt_present"])]
    paired = [row for row in absent if float(row["source_score"]) < threshold]
    raw_add = sum(float(row["score"]) >= threshold for row in absent)
    paired_add = sum(float(row["score"]) >= threshold for row in paired)
    return {
        "raw_absent_opportunities": len(absent),
        "raw_additional_detections": raw_add,
        "raw_h_add": raw_add / len(absent) if absent else None,
        "source_paired_absent_opportunities": len(paired),
        "source_paired_additional_detections": paired_add,
        "source_paired_h_add": paired_add / len(paired) if paired else None,
    }

def _paired_h_bootstrap(
    candidate: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]],
    threshold: float, iterations: int, *, seed: int,
) -> Dict[str, Any]:
    def counts_by_video(values):
        out = defaultdict(lambda: [0, 0])
        for row in values:
            if not bool(row["gt_present"]) and float(row["source_score"]) < threshold:
                out[str(row["video_id"])][1] += 1
                out[str(row["video_id"])][0] += float(row["score"]) >= threshold
        return out

    cand = counts_by_video(candidate)
    ref = counts_by_video(reference)
    videos = sorted(set(cand) | set(ref))
    if not videos or set(cand) != set(ref):
        raise ValueError("paired G2 H_add requires the same non-empty video clusters")
    rng = random.Random(seed)
    deltas: List[float] = []
    reductions: List[float] = []
    for _ in range(int(iterations)):
        ca = co = ra = ro = 0
        for video in (rng.choice(videos) for _ in videos):
            ca += cand[video][0]
            co += cand[video][1]
            ra += ref[video][0]
            ro += ref[video][1]
        candidate_rate = ca / co if co else 0.0
        reference_rate = ra / ro if ro else 0.0
        deltas.append(candidate_rate - reference_rate)
        if reference_rate > 0:
            reductions.append((reference_rate - candidate_rate) / reference_rate)
    return {
        "delta_candidate_minus_reference": {
            "lower_two_sided_95": _quantile(deltas, 0.025),
            "upper_two_sided_95": _quantile(deltas, 0.975),
            "upper_one_sided_95": _quantile(deltas, 0.95),
        },
        "relative_reduction": ({
            "lower_one_sided_95": _quantile(reductions, 0.05),
            "lower_two_sided_95": _quantile(reductions, 0.025),
            "upper_two_sided_95": _quantile(reductions, 0.975),
        } if reductions else None),
        "cluster_unit": "video",
        "iterations": int(iterations),
    }


def _false_suppression(
    candidate: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]],
    threshold: float, iterations: int, *, seed: int,
) -> Dict[str, Any]:
    ref_lookup = {
        (str(row["video_id"]), str(row["policy"]), int(row["seed"]),
         int(row["frame_index"]), str(row["concept_id"])): row
        for row in reference
    }
    by_video = defaultdict(lambda: [0, 0])
    for row in candidate:
        if not bool(row["gt_present"]):
            continue
        key = (str(row["video_id"]), str(row["policy"]), int(row["seed"]),
               int(row["frame_index"]), str(row["concept_id"]))
        ref_row = ref_lookup[key]
        if float(ref_row["score"]) >= threshold:
            by_video[str(row["video_id"])][1] += 1
            by_video[str(row["video_id"])][0] += float(row["score"]) < threshold
    videos = sorted(by_video)
    suppressed = sum(value[0] for value in by_video.values())
    opportunities = sum(value[1] for value in by_video.values())
    rate = suppressed / opportunities if opportunities else None
    if not videos:
        return {
            "suppressed": 0, "baseline_detected_present_opportunities": 0,
            "rate": None, "upper_one_sided_95": None,
        }
    rng = random.Random(seed)
    samples = []
    for _ in range(int(iterations)):
        num = den = 0
        for video in (rng.choice(videos) for _ in videos):
            num += by_video[video][0]
            den += by_video[video][1]
        samples.append(num / den if den else 0.0)
    return {
        "suppressed": suppressed,
        "baseline_detected_present_opportunities": opportunities,
        "rate": rate,
        "upper_one_sided_95": _quantile(samples, 0.95),
        "definition": "GT-present concept detected in no_negative but not in candidate",
    }


def _quality_delta(
    accounting_rows: Sequence[Mapping[str, Any]], *, candidate_arm: str,
    reference_arm: str, policy: str, metric: str, iterations: int, seed: int,
) -> Dict[str, Any]:
    values: Dict[str, Dict[str, List[float]]] = {
        candidate_arm: defaultdict(list), reference_arm: defaultdict(list),
    }
    for row in accounting_rows:
        arm = str(row["arm"])
        if arm not in values or str(row["policy"]) != policy:
            continue
        raw = row.get(metric)
        if raw in (None, ""):
            raise ValueError(f"missing {metric} for formal G2 accounting row")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"non-finite {metric} for formal G2 accounting row")
        values[arm][str(row["video_id"])].append(value)
    videos = sorted(set(values[candidate_arm]) & set(values[reference_arm]))
    if set(values[candidate_arm]) != set(values[reference_arm]) or not videos:
        raise ValueError(f"quality grid mismatch for {policy}/{metric}")
    deltas = {
        video: (
            sum(values[candidate_arm][video]) / len(values[candidate_arm][video])
            - sum(values[reference_arm][video]) / len(values[reference_arm][video])
        )
        for video in videos
    }
    point = sum(deltas.values()) / len(deltas)
    rng = random.Random(seed)
    samples = [
        sum(deltas[rng.choice(videos)] for _ in videos) / len(videos)
        for _ in range(int(iterations))
    ]
    return {
        "mean_delta": point,
        "per_video_delta": deltas,
        "lower_one_sided_95": _quantile(samples, 0.05),
        "upper_one_sided_95": _quantile(samples, 0.95),
        "lower_two_sided_95": _quantile(samples, 0.025),
        "upper_two_sided_95": _quantile(samples, 0.975),
    }


def summarize_g2(
    detection_rows: Iterable[Mapping[str, Any]],
    accounting_rows: Iterable[Mapping[str, Any]],
    *,
    threshold: float,
    arms: Sequence[str],
    policies: Sequence[str],
    primary_policy: str,
    bootstrap_iterations: int,
    gate: Mapping[str, float],
) -> Dict[str, Any]:
    """Summarize four-arm G2 results and evaluate the provisional oracle gate."""
    rows = [dict(row) for row in detection_rows]
    accounting = [dict(row) for row in accounting_rows]
    _validate_detection_grid(rows, arms, policies)
    if arms[0] != "no_negative" or "oracle_negative" not in arms:
        raise ValueError("G2 arms must start with no_negative and include oracle_negative")
    reference_arm = "no_negative"

    by_condition: Dict[str, Any] = {}
    comparisons: Dict[str, Any] = {}
    quality: Dict[str, Any] = {}
    for policy in policies:
        policy_rows = [row for row in rows if str(row["policy"]) == policy]
        by_condition[policy] = {}
        reference = [row for row in policy_rows if str(row["arm"]) == reference_arm]
        comparisons[policy] = {}
        for arm in arms:
            arm_rows = [row for row in policy_rows if str(row["arm"]) == arm]
            by_condition[policy][arm] = _h_counts(arm_rows, float(threshold))
            if arm == reference_arm:
                continue
            arm_counts = by_condition[policy][arm]
            ref_counts = by_condition[policy][reference_arm]
            ref_rate = ref_counts["source_paired_h_add"]
            arm_rate = arm_counts["source_paired_h_add"]
            relative = (
                (float(ref_rate) - float(arm_rate)) / float(ref_rate)
                if ref_rate not in (None, 0.0) and arm_rate is not None else None
            )
            comparisons[policy][arm] = {
                "source_paired_h_add_delta": (
                    float(arm_rate) - float(ref_rate)
                    if arm_rate is not None and ref_rate is not None else None
                ),
                "source_paired_h_add_relative_reduction": relative,
                "h_add_bootstrap": _paired_h_bootstrap(
                    arm_rows, reference, float(threshold), int(bootstrap_iterations),
                    seed=19001 + sum(map(ord, policy + arm)),
                ),
                "false_suppression_vs_no_negative": _false_suppression(
                    arm_rows, reference, float(threshold), int(bootstrap_iterations),
                    seed=23003 + sum(map(ord, policy + arm)),
                ),
            }
        quality[policy] = {
            metric: _quality_delta(
                accounting, candidate_arm="oracle_negative", reference_arm=reference_arm,
                policy=policy, metric=metric, iterations=int(bootstrap_iterations),
                seed=29009 + sum(map(ord, policy + metric)),
            )
            for metric in ("mean_psnr", "mean_ssim", "mean_lpips")
        }

    primary = comparisons[primary_policy]["oracle_negative"]
    oracle_relative = primary["source_paired_h_add_relative_reduction"]
    h_ci = primary["h_add_bootstrap"]["delta_candidate_minus_reference"]
    false_suppression = primary["false_suppression_vs_no_negative"]
    primary_quality = quality[primary_policy]
    checks = {
        "oracle_h_add_relative_reduction": (
            oracle_relative is not None
            and oracle_relative >= float(gate["h_add_relative_reduction_min"])
        ),
        "oracle_h_add_one_sided_ci_improves": h_ci["upper_one_sided_95"] < 0.0,
        "oracle_false_suppression_upper_bound": (
            false_suppression["upper_one_sided_95"] is not None
            and false_suppression["upper_one_sided_95"]
            <= float(gate["false_suppression_increase_ci_upper_max"])
        ),
        "psnr_noninferiority": (
            primary_quality["mean_psnr"]["lower_one_sided_95"]
            >= -float(gate["psnr_drop_db_max"])
        ),
        "ssim_noninferiority": (
            primary_quality["mean_ssim"]["lower_one_sided_95"]
            >= -float(gate["ssim_drop_max"])
        ),
        "lpips_noninferiority": (
            primary_quality["mean_lpips"]["upper_one_sided_95"]
            <= float(gate["lpips_rise_max"])
        ),
    }
    return {
        "threshold": float(threshold),
        "primary_policy": primary_policy,
        "by_condition": by_condition,
        "comparisons_vs_no_negative": comparisons,
        "oracle_quality_delta_vs_no_negative": quality,
        "provisional_gate_checks": checks,
        "provisional_gate_status": "PASSED" if all(checks.values()) else "NOT_PASSED",
        "bootstrap": {
            "cluster_unit": "video", "iterations": int(bootstrap_iterations),
            "confidence_level": 0.95, "one_sided": True,
        },
    }
