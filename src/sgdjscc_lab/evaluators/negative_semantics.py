"""Pure metric logic for the frozen negative-semantics G1 evaluator."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def select_global_threshold(
    score_rows: Iterable[Mapping[str, Any]],
    thresholds: Sequence[float],
    *,
    min_recall: float,
    max_false_positive_rate: float,
) -> Dict[str, Any]:
    """Select the highest predeclared threshold satisfying both GT constraints."""
    rows = list(score_rows)
    positives = sum(bool(row["gt_present"]) for row in rows)
    negatives = len(rows) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("calibration requires at least one positive and one negative opportunity")
    candidates = []
    for threshold in sorted({float(value) for value in thresholds}):
        true_positive = sum(
            bool(row["gt_present"]) and float(row["score"]) >= threshold
            for row in rows
        )
        false_positive = sum(
            not bool(row["gt_present"]) and float(row["score"]) >= threshold
            for row in rows
        )
        recall = true_positive / positives
        false_positive_rate = false_positive / negatives
        eligible = recall >= min_recall and false_positive_rate <= max_false_positive_rate
        candidates.append({
            "threshold": threshold,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "positive_opportunities": positives,
            "negative_opportunities": negatives,
            "recall": recall,
            "false_positive_rate": false_positive_rate,
            "eligible": eligible,
        })
    eligible_rows = [row for row in candidates if row["eligible"]]
    selected = max(eligible_rows, key=lambda row: row["threshold"]) if eligible_rows else None
    return {
        "status": "PASSED" if selected is not None else "NOT_PASSED",
        "selected_threshold": selected["threshold"] if selected else None,
        "selection_rule": "highest_threshold_meeting_both_constraints",
        "min_recall": float(min_recall),
        "max_false_positive_rate": float(max_false_positive_rate),
        "selected_operating_point": selected,
        "candidates": candidates,
    }


def collapse_additional_events(
    rows: Iterable[Mapping[str, Any]], threshold: float
) -> List[Dict[str, Any]]:
    """Collapse consecutive absent detections into one concept-track event."""
    groups: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["video_id"], int(row["seed"]), row["policy"], row["concept_id"])
        groups[key].append(row)
    events: List[Dict[str, Any]] = []
    for (video_id, seed, policy, concept_id), values in sorted(groups.items()):
        active: List[Mapping[str, Any]] = []

        def flush() -> None:
            if not active:
                return
            events.append({
                "video_id": video_id,
                "seed": seed,
                "policy": policy,
                "concept_id": concept_id,
                "start_frame": int(active[0]["frame_index"]),
                "end_frame": int(active[-1]["frame_index"]),
                "duration": len(active),
                "max_score": max(float(item["score"]) for item in active),
            })
            active.clear()

        previous = None
        for row in sorted(values, key=lambda item: int(item["frame_index"])):
            frame = int(row["frame_index"])
            positive_absent = (
                not bool(row["gt_present"]) and float(row["score"]) >= threshold
            )
            if positive_absent and (previous is None or frame == previous + 1):
                active.append(row)
            elif positive_absent:
                flush()
                active.append(row)
            else:
                flush()
            previous = frame
        flush()
    return events


def _cluster_bootstrap_h_add(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    *,
    iterations: int,
    seed: int = 7301,
) -> Dict[str, float]:
    by_video: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not bool(row["gt_present"]):
            by_video[str(row["video_id"])].append(row)
    videos = sorted(by_video)
    if not videos:
        return {"lower": 0.0, "upper": 0.0}
    rng = random.Random(seed)
    samples = []
    for _ in range(int(iterations)):
        additions = opportunities = 0
        for video in (rng.choice(videos) for _ in videos):
            values = by_video[video]
            opportunities += len(values)
            additions += sum(float(row["score"]) >= threshold for row in values)
        samples.append(additions / opportunities if opportunities else 0.0)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return {"lower": lo, "upper": hi}


def _ghost_summary(
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    threshold: float,
    horizon: int,
) -> Dict[str, Any]:
    lookup = {
        (
            str(row["video_id"]), int(row["seed"]), str(row["policy"]),
            int(row["frame_index"]), str(row["concept_id"]),
        ): row
        for row in rows
    }
    policies = sorted({str(row["policy"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    exit_events = [row for row in events if row["event_type"] == "EXIT"]
    values = []
    censored = 0
    for event in exit_events:
        start = int(event["start_frame"])
        for policy in policies:
            for seed in seeds:
                window = []
                for frame in range(start, start + horizon):
                    row = lookup.get((event["video_id"], seed, policy, frame, event["concept_id"]))
                    if row is None or bool(row["gt_present"]):
                        window = []
                        break
                    window.append(float(row["score"]) >= threshold)
                if len(window) != horizon:
                    censored += 1
                    continue
                values.append({
                    "event_id": event["event_id"],
                    "video_id": event["video_id"],
                    "seed": seed,
                    "policy": policy,
                    "auc": sum(window) / horizon,
                    "survival": [int(value) for value in window],
                })
    by_policy: Dict[str, Any] = {}
    for policy in policies:
        policy_values = [row["auc"] for row in values if row["policy"] == policy]
        by_policy[policy] = {
            "mean_auc": sum(policy_values) / len(policy_values) if policy_values else None,
            "uncensored_count": len(policy_values),
        }
    return {
        "horizon_annotated_timesteps": horizon,
        "uncensored": values,
        "right_censored_count": censored,
        "by_policy": by_policy,
    }


def summarize_g1(
    rows: Iterable[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    primary_policy: str,
    bootstrap_iterations: int,
    ghost_horizon: int,
) -> Dict[str, Any]:
    values = list(rows)
    collapsed = collapse_additional_events(values, threshold)
    policies = sorted({str(row["policy"]) for row in values})
    by_policy: Dict[str, Any] = {}
    for policy in policies:
        policy_rows = [row for row in values if row["policy"] == policy]
        absent = [row for row in policy_rows if not bool(row["gt_present"])]
        additions = sum(float(row["score"]) >= threshold for row in absent)
        policy_events = [row for row in collapsed if row["policy"] == policy]
        video_counts = Counter(row["video_id"] for row in policy_events)
        seed_counts = Counter(int(row["seed"]) for row in policy_events)
        total_events = len(policy_events)
        by_policy[policy] = {
            "absent_opportunities": len(absent),
            "additional_detections": additions,
            "h_add": additions / len(absent) if absent else None,
            "h_add_video_clustered_bootstrap_95ci": _cluster_bootstrap_h_add(
                policy_rows, threshold, iterations=bootstrap_iterations
            ),
            "additional_event_count": total_events,
            "affected_video_count": len(video_counts),
            "affected_seed_count": len(seed_counts),
            "max_single_video_event_share": (
                max(video_counts.values()) / total_events if total_events else 0.0
            ),
            "max_single_seed_event_share": (
                max(seed_counts.values()) / total_events if total_events else 0.0
            ),
        }
    return {
        "threshold": float(threshold),
        "primary_policy": primary_policy,
        "by_policy": by_policy,
        "additional_events": collapsed,
        "ghost_survival": _ghost_summary(
            values, events, threshold, horizon=int(ghost_horizon)
        ),
    }
