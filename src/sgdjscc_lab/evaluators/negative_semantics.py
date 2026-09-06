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


# --- v1.2 amendment: effective-seed accounting and source-paired additions ---
#
# G1 v1.1 ran a deterministic reconstruction path (fixed selector, fixed
# diffusion schedule, no sampling noise consumes the ``seed`` argument) under
# three nominal seeds. Every reconstructed frame — and therefore every
# detector score — is bit-identical across seeds 2025/2026/2027. Treating
# these as three independent replicates triples event/opportunity counts and
# makes "not concentrated in one seed" trivially true without adding any real
# statistical evidence. The functions below make that duplication explicit
# and auditable instead of silently absorbing it into the v1.1 numbers, which
# are left unmodified in ``summarize_g1`` above for historical reproducibility.


def effective_seed_groups(
    rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Collapse seeds whose detector scores are bit-identical, per policy.

    Two seeds within the same policy are judged duplicate when every
    ``(video_id, frame_index, concept_id)`` score matches exactly — the score
    is a deterministic function of the reconstructed pixels, so an exact
    match is strong evidence the seed never reached the sampler (verified
    independently by reconstruction-frame hashes for G1 v1.1). Seeds that
    diverge anywhere are kept as distinct canonical (effective) units.
    """
    by_policy_seed: Dict[str, Dict[int, Dict[tuple, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        policy = str(row["policy"])
        seed = int(row["seed"])
        key = (str(row["video_id"]), int(row["frame_index"]), str(row["concept_id"]))
        by_policy_seed[policy][seed][key] = float(row["score"])
    result: Dict[str, Dict[str, Any]] = {}
    for policy, seed_scores in by_policy_seed.items():
        seeds = sorted(seed_scores)
        canonical_seeds: List[int] = []
        canonical_for: Dict[int, int] = {}
        for seed in seeds:
            match = next(
                (canonical for canonical in canonical_seeds if seed_scores[seed] == seed_scores[canonical]),
                None,
            )
            if match is None:
                canonical_seeds.append(seed)
                canonical_for[seed] = seed
            else:
                canonical_for[seed] = match
        result[policy] = {
            "declared_seed_count": len(seeds),
            "effective_seed_count": len(canonical_seeds),
            "canonical_seeds": canonical_seeds,
            "seed_to_canonical": canonical_for,
            "deterministic_collapse": len(canonical_seeds) < len(seeds),
        }
    return result


def effective_seed_groups_from_hashes(
    hash_by_policy_seed: Mapping[str, Mapping[int, str]],
) -> Dict[str, Dict[str, Any]]:
    """Same collapsing shape as ``effective_seed_groups``, but keyed on a
    caller-supplied content hash per (policy, seed) instead of inferring
    pixel identity from detector scores.

    A detector-score match is not proof that the underlying reconstruction
    frames are pixel-identical -- two different images could in principle
    score identically. Callers should compute ``hash_by_policy_seed`` from
    the actual frame bytes (see ``sgdjscc_lab.utils.frame_hash.frame_tree_sha256``)
    and cross-check it against ``effective_seed_groups``' score-based result
    with ``cross_validate_seed_evidence`` before trusting a collapse.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for policy, seed_hashes in hash_by_policy_seed.items():
        seeds = sorted(seed_hashes)
        canonical_seeds: List[int] = []
        canonical_for: Dict[int, int] = {}
        for seed in seeds:
            match = next(
                (canonical for canonical in canonical_seeds if seed_hashes[seed] == seed_hashes[canonical]),
                None,
            )
            if match is None:
                canonical_seeds.append(seed)
                canonical_for[seed] = seed
            else:
                canonical_for[seed] = match
        result[policy] = {
            "declared_seed_count": len(seeds),
            "effective_seed_count": len(canonical_seeds),
            "canonical_seeds": canonical_seeds,
            "seed_to_canonical": canonical_for,
            "deterministic_collapse": len(canonical_seeds) < len(seeds),
        }
    return result


def cross_validate_seed_evidence(
    score_groups: Mapping[str, Mapping[str, Any]],
    hash_groups: Mapping[str, Mapping[str, Any]],
) -> None:
    """Fail closed if detector-score and reconstruction-frame-hash evidence
    disagree about which seeds are duplicates of each other.

    Both ``score_groups`` (from ``effective_seed_groups``) and ``hash_groups``
    (from ``effective_seed_groups_from_hashes``) must cover the same set of
    policies and agree, per policy, on the exact ``seed_to_canonical``
    mapping. Any disagreement -- including one input missing a policy the
    other has -- raises, since it means the two independent kinds of
    evidence for "these seeds produced identical output" do not actually
    agree, and no automatic resolution is safe.
    """
    policies = set(score_groups) | set(hash_groups)
    for policy in sorted(policies):
        score_info = score_groups.get(policy)
        hash_info = hash_groups.get(policy)
        if score_info is None or hash_info is None:
            missing_from = "score-based" if score_info is None else "hash-based"
            raise ValueError(f"policy={policy!r} is missing from {missing_from} seed evidence")
        if score_info["seed_to_canonical"] != hash_info["seed_to_canonical"]:
            raise ValueError(
                f"policy={policy!r}: detector-score seed grouping "
                f"{score_info['seed_to_canonical']} disagrees with independent "
                f"reconstruction-frame-hash seed grouping {hash_info['seed_to_canonical']}; "
                "refusing to collapse seeds on conflicting evidence"
            )


def require_effective_seed_declaration(
    groups: Mapping[str, Mapping[str, Any]], *, declared_deterministic_decoder: bool
) -> None:
    """Fail closed if seeds silently collapsed without a matching declaration.

    A run must positively declare its decoder deterministic before its
    outputs are allowed to show fewer effective than declared seeds. This
    stops a future run — where the decoder is expected to be stochastic —
    from quietly losing seed diversity (e.g. a broken seed plumbing bug) and
    still reporting seed-level prevalence/CI numbers as if nothing changed.
    """
    for policy, info in groups.items():
        if info["deterministic_collapse"] and not declared_deterministic_decoder:
            raise ValueError(
                f"policy={policy!r} seeds collapsed to {info['effective_seed_count']} "
                f"effective unit(s) from {info['declared_seed_count']} declared seeds "
                "without a deterministic-decoder declaration; refusing to report "
                "seed-level statistics computed from undeclared duplicate seeds"
            )


def raw_and_paired_h_add(
    reconstruction_rows: Iterable[Mapping[str, Any]],
    source_rows: Iterable[Mapping[str, Any]],
    threshold: float,
    *,
    effective_seeds: Mapping[str, Sequence[int]] | None = None,
) -> Dict[str, Any]:
    """Split H_add into the v1.1 raw definition and a source-paired variant.

    ``raw_h_add`` matches ``summarize_g1``'s ``h_add`` exactly: detector
    positive on the reconstruction while official GT says the concept is
    absent, regardless of what the source frame looked like.

    ``source_paired_h_add`` restricts the denominator to opportunities where
    the *source* frame's detector score was also below threshold (correctly
    negative). This isolates additions the transmit/reconstruct pipeline
    itself introduces from cases where the source image already trips the
    detector — a limitation the source-only OWLv2 calibration cannot rule
    out, since calibration only bounds aggregate recall/FPR, not per-frame
    agreement with the reconstruction path.
    """
    source_index: Dict[tuple, float] = {
        (str(row["video_id"]), int(row["frame_index"]), str(row["concept_id"])): float(row["score"])
        for row in source_rows
    }
    recon_rows = list(reconstruction_rows)
    if effective_seeds is not None:
        recon_rows = [
            row for row in recon_rows
            if int(row["seed"]) in set(effective_seeds.get(str(row["policy"]), []))
        ]
    by_policy: Dict[str, Dict[str, Any]] = {}
    for policy in sorted({str(row["policy"]) for row in recon_rows}):
        absent = [row for row in recon_rows if row["policy"] == policy and not bool(row["gt_present"])]
        raw_additions = sum(float(row["score"]) >= threshold for row in absent)
        pairable = [
            row for row in absent
            if (str(row["video_id"]), int(row["frame_index"]), str(row["concept_id"])) in source_index
        ]
        source_negative = [
            row for row in pairable
            if source_index[(str(row["video_id"]), int(row["frame_index"]), str(row["concept_id"]))] < threshold
        ]
        paired_additions = sum(float(row["score"]) >= threshold for row in source_negative)
        by_policy[policy] = {
            "absent_opportunities": len(absent),
            "raw_additional_detections": raw_additions,
            "raw_h_add": raw_additions / len(absent) if absent else None,
            "unpairable_opportunities": len(absent) - len(pairable),
            "source_paired_opportunities": len(source_negative),
            "source_paired_additional_detections": paired_additions,
            "source_paired_h_add": paired_additions / len(source_negative) if source_negative else None,
        }
    return by_policy


def h_add_by_video(
    reconstruction_rows: Iterable[Mapping[str, Any]],
    source_rows: Iterable[Mapping[str, Any]],
    threshold: float,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Per-video breakdown of raw/source-paired H_add, one row per (policy, video).

    Unlike ``raw_and_paired_h_add`` (which pools all opportunities into one
    ratio per policy and tolerates unpairable rows -- appropriate for the
    frozen v1.1/v1.2 pipeline, which is not changed here), this keeps each
    video's own ratio so a caller can pair videos across two conditions
    (e.g. fixed_int4 vs fixed_int6, encoded as the ``policy`` field) and
    compute a paired per-video delta instead of subtracting two
    independently-pooled ratios. Because a bridge comparison controls its
    own small, closed dataset, this fails closed instead: a duplicate
    ``(video_id, frame_index, concept_id)`` key in ``source_rows``, or any
    reconstruction opportunity with no matching source key at all, raises
    rather than being silently overwritten or dropped.
    """
    source_index: Dict[tuple, float] = {}
    for row in source_rows:
        key = (str(row["video_id"]), int(row["frame_index"]), str(row["concept_id"]))
        if key in source_index:
            raise ValueError(f"duplicate source key {key} in source_rows")
        source_index[key] = float(row["score"])
    by_policy_video: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in reconstruction_rows:
        if bool(row["gt_present"]):
            continue
        by_policy_video[str(row["policy"])][str(row["video_id"])].append(row)
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for policy, by_video in by_policy_video.items():
        result[policy] = {}
        for video_id, rows in by_video.items():
            unpaired = [
                row for row in rows
                if (str(row["video_id"]), int(row["frame_index"]), str(row["concept_id"])) not in source_index
            ]
            if unpaired:
                missing_keys = sorted({
                    (str(row["video_id"]), int(row["frame_index"]), str(row["concept_id"]))
                    for row in unpaired
                })
                raise ValueError(
                    f"policy={policy!r} video_id={video_id!r}: {len(unpaired)} reconstruction "
                    f"opportunity row(s) have no matching source key, e.g. {missing_keys[:5]}"
                )
            raw_additions = sum(float(row["score"]) >= threshold for row in rows)
            source_negative = [
                row for row in rows
                if source_index[(str(row["video_id"]), int(row["frame_index"]), str(row["concept_id"]))] < threshold
            ]
            paired_additions = sum(float(row["score"]) >= threshold for row in source_negative)
            result[policy][video_id] = {
                "absent_opportunities": len(rows),
                "raw_h_add": raw_additions / len(rows) if rows else None,
                "source_paired_opportunities": len(source_negative),
                "source_paired_h_add": paired_additions / len(source_negative) if source_negative else None,
            }
    return result


def reapply_gate_checks(
    policy_metrics: Mapping[str, Any], gate_cfg: Mapping[str, Any]
) -> Dict[str, bool]:
    """Reapply the frozen G1 gate's per-policy thresholds to a metrics dict.

    Mirrors the subset of ``scripts/run_negative_semantics_g1.py::_write_summary``'s
    ``checks`` that depend on ``by_policy[primary_policy]`` (h_add prevalence,
    event count, affected video/seed counts, concentration shares). The
    remaining v1.1 gate checks (evaluator calibration, selector weight
    separation, formal-vs-smoke, all-policies/all-seeds-present) are
    structural and identical whether seeds are counted as declared or
    deduplicated, so they are intentionally not reproduced here.
    """
    return {
        "h_add_prevalence": (policy_metrics.get("h_add") or 0.0) >= float(gate_cfg["h_add_min"]),
        "additional_event_count": policy_metrics.get("additional_event_count", 0) >= int(gate_cfg["additional_events_min"]),
        "affected_video_count": policy_metrics.get("affected_video_count", 0) >= int(gate_cfg["affected_videos_min"]),
        "affected_seed_count": policy_metrics.get("affected_seed_count", 0) >= int(gate_cfg["affected_seeds_min"]),
        "not_concentrated_in_one_video": policy_metrics.get("max_single_video_event_share", 1.0) <= float(gate_cfg["max_single_video_event_share"]),
        "not_concentrated_in_one_seed": policy_metrics.get("max_single_seed_event_share", 1.0) <= float(gate_cfg["max_single_seed_event_share"]),
    }


def summarize_g1_v1_2(
    rows: Iterable[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    source_rows: Iterable[Mapping[str, Any]],
    *,
    threshold: float,
    primary_policy: str,
    bootstrap_iterations: int,
    ghost_horizon: int,
    declared_deterministic_decoder: bool,
    gate_cfg: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Derive v1.2 amendment metrics from already-frozen v1.1 detection rows.

    Read-only with respect to v1.1: this never mutates ``rows``/``events``
    and callers are expected to write the result to a new path rather than
    the frozen v1.1 run directory. Returns both the unmodified v1.1
    (declared-seed) view for continuity and a deduplicated effective-seed
    view alongside the source-paired additional-object split.
    """
    values = list(rows)
    source_values = list(source_rows)
    groups = effective_seed_groups(values)
    require_effective_seed_declaration(groups, declared_deterministic_decoder=declared_deterministic_decoder)
    effective_seeds = {policy: info["canonical_seeds"] for policy, info in groups.items()}
    effective_rows = [
        row for row in values
        if int(row["seed"]) in set(effective_seeds.get(str(row["policy"]), []))
    ]
    raw_view = summarize_g1(
        values, events, threshold=threshold, primary_policy=primary_policy,
        bootstrap_iterations=bootstrap_iterations, ghost_horizon=ghost_horizon,
    )
    effective_view = summarize_g1(
        effective_rows, events, threshold=threshold, primary_policy=primary_policy,
        bootstrap_iterations=bootstrap_iterations, ghost_horizon=ghost_horizon,
    )
    unique_exit_event_count = len({
        str(event["event_id"]) for event in events if event.get("event_type") == "EXIT"
    })
    effective_view["ghost_survival"]["unique_exit_event_count"] = unique_exit_event_count
    report: Dict[str, Any] = {
        "amendment": "negative_semantics_g1_v1_2",
        "threshold": float(threshold),
        "primary_policy": primary_policy,
        "declared_deterministic_decoder": bool(declared_deterministic_decoder),
        "seed_groups": groups,
        "raw_v1_1_view": {
            "by_policy": raw_view["by_policy"],
            "ghost_survival": raw_view["ghost_survival"],
            "note": "unchanged v1.1 definition; seeds counted as declared (2025/2026/2027), not deduplicated",
        },
        "effective_seed_view": {
            "by_policy": effective_view["by_policy"],
            "ghost_survival": effective_view["ghost_survival"],
            "note": "recomputed using only the canonical (deduplicated) seed per policy",
        },
        "source_paired_additional_object": {
            "declared_seeds": raw_and_paired_h_add(values, source_values, threshold),
            "effective_seeds": raw_and_paired_h_add(
                values, source_values, threshold, effective_seeds=effective_seeds
            ),
        },
    }
    if gate_cfg is not None:
        raw_primary = raw_view["by_policy"].get(primary_policy, {})
        effective_primary = effective_view["by_policy"].get(primary_policy, {})
        raw_gate = reapply_gate_checks(raw_primary, gate_cfg)
        effective_gate = reapply_gate_checks(effective_primary, gate_cfg)
        flipped = sorted(
            check for check, passed in effective_gate.items() if raw_gate.get(check) and not passed
        )
        report["gate_reapplication"] = {
            "primary_policy": primary_policy,
            "raw_declared_seed_checks": raw_gate,
            "effective_seed_checks": effective_gate,
            "raw_declared_seed_status": "PASSED" if all(raw_gate.values()) else "NOT_PASSED",
            "effective_seed_status": "PASSED" if all(effective_gate.values()) else "NOT_PASSED",
            "checks_that_flip_to_fail_under_effective_seeds": flipped,
        }
    return report
