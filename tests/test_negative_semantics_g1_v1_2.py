"""Unit/regression tests for the G1 v1.2 amendment metrics.

These exercise `effective_seed_groups`, `require_effective_seed_declaration`,
`raw_and_paired_h_add`, and `summarize_g1_v1_2` against small synthetic
fixtures that reproduce the pattern actually observed in the frozen
`outputs/negative_semantics_g1_pilot_rtx4080_v1_1` run: reconstruction scores
are bit-identical across nominal seeds because the decoder path never
consumes the seed. No GPU, network, or dataset access is required.
"""

from __future__ import annotations

import pytest

from sgdjscc_lab.evaluators.negative_semantics import (
    cross_validate_seed_evidence,
    effective_seed_groups,
    effective_seed_groups_from_hashes,
    h_add_by_video,
    raw_and_paired_h_add,
    reapply_gate_checks,
    require_effective_seed_declaration,
    summarize_g1_v1_2,
)


def _triplicated_rows():
    """Same scores for seeds 2025/2026/2027 (deterministic-decoder pattern)."""
    rows = []
    for seed in (2025, 2026, 2027):
        for frame, score, gt_present in [
            (0, 0.8, False), (1, 0.7, False), (2, 0.1, False),
            (3, 0.9, True), (4, 0.9, False),
        ]:
            rows.append({
                "video_id": "v1", "seed": seed, "policy": "few10",
                "concept_id": "dog", "frame_index": frame,
                "score": score, "gt_present": gt_present,
            })
    return rows


def test_effective_seed_groups_collapses_bit_identical_seeds():
    groups = effective_seed_groups(_triplicated_rows())
    info = groups["few10"]
    assert info["declared_seed_count"] == 3
    assert info["effective_seed_count"] == 1
    assert info["canonical_seeds"] == [2025]
    assert info["seed_to_canonical"] == {2025: 2025, 2026: 2025, 2027: 2025}
    assert info["deterministic_collapse"] is True


def test_effective_seed_groups_keeps_genuinely_distinct_seeds_separate():
    rows = []
    for seed, bump in ((2025, 0.0), (2026, 0.3)):
        rows.append({
            "video_id": "v1", "seed": seed, "policy": "few10",
            "concept_id": "dog", "frame_index": 0,
            "score": 0.1 + bump, "gt_present": False,
        })
    info = effective_seed_groups(rows)["few10"]
    assert info["declared_seed_count"] == 2
    assert info["effective_seed_count"] == 2
    assert info["deterministic_collapse"] is False


def test_require_effective_seed_declaration_fails_closed_when_undeclared():
    groups = effective_seed_groups(_triplicated_rows())
    with pytest.raises(ValueError, match="deterministic-decoder declaration"):
        require_effective_seed_declaration(groups, declared_deterministic_decoder=False)
    # Declaring it explicitly must not raise.
    require_effective_seed_declaration(groups, declared_deterministic_decoder=True)


def test_require_effective_seed_declaration_passes_without_collapse():
    rows = [
        {"video_id": "v1", "seed": 2025, "policy": "few10", "concept_id": "dog",
         "frame_index": 0, "score": 0.1, "gt_present": False},
        {"video_id": "v1", "seed": 2026, "policy": "few10", "concept_id": "dog",
         "frame_index": 0, "score": 0.9, "gt_present": False},
    ]
    groups = effective_seed_groups(rows)
    require_effective_seed_declaration(groups, declared_deterministic_decoder=False)


def test_raw_and_paired_h_add_separates_source_carried_from_new_additions():
    threshold = 0.5
    reconstruction_rows = [
        # Concept "cat" is GT-absent everywhere below; source already flags
        # frame 0 (score 0.9 >= threshold) so a reconstruction positive there
        # is source-carried, not a new addition.
        {"video_id": "v1", "seed": 2025, "policy": "few10", "concept_id": "cat",
         "frame_index": 0, "score": 0.8, "gt_present": False},
        # Frame 1: source is negative (0.1) but reconstruction is positive
        # (0.7) -- a genuine pipeline-introduced addition.
        {"video_id": "v1", "seed": 2025, "policy": "few10", "concept_id": "cat",
         "frame_index": 1, "score": 0.7, "gt_present": False},
        # Frame 2: both source and reconstruction stay negative.
        {"video_id": "v1", "seed": 2025, "policy": "few10", "concept_id": "cat",
         "frame_index": 2, "score": 0.2, "gt_present": False},
    ]
    source_rows = [
        {"video_id": "v1", "frame_index": 0, "concept_id": "cat", "score": 0.9},
        {"video_id": "v1", "frame_index": 1, "concept_id": "cat", "score": 0.1},
        {"video_id": "v1", "frame_index": 2, "concept_id": "cat", "score": 0.05},
    ]
    result = raw_and_paired_h_add(reconstruction_rows, source_rows, threshold)
    few10 = result["few10"]
    assert few10["absent_opportunities"] == 3
    assert few10["raw_additional_detections"] == 2
    assert few10["raw_h_add"] == pytest.approx(2 / 3)
    # Only frames 1 and 2 have a source-negative opportunity.
    assert few10["source_paired_opportunities"] == 2
    assert few10["source_paired_additional_detections"] == 1
    assert few10["source_paired_h_add"] == pytest.approx(0.5)
    assert few10["unpairable_opportunities"] == 0


def test_summarize_g1_v1_2_effective_view_dedupes_triplicated_seed_events():
    rows = _triplicated_rows()
    source_rows = [
        {"video_id": "v1", "frame_index": frame, "concept_id": "dog", "score": 0.0}
        for frame in range(5)
    ]
    events = [
        {"event_id": "exit-1", "event_type": "EXIT", "video_id": "v1", "concept_id": "dog", "start_frame": 4},
    ]
    report = summarize_g1_v1_2(
        rows, events, source_rows, threshold=0.5, primary_policy="few10",
        bootstrap_iterations=20, ghost_horizon=1,
        declared_deterministic_decoder=True,
    )
    assert report["seed_groups"]["few10"]["effective_seed_count"] == 1
    raw = report["raw_v1_1_view"]["by_policy"]["few10"]
    effective = report["effective_seed_view"]["by_policy"]["few10"]
    # Raw (declared-seed) counts triple the effective (deduplicated) counts.
    assert raw["additional_detections"] == 3 * effective["additional_detections"]
    assert raw["additional_event_count"] == 3 * effective["additional_event_count"]
    # h_add itself (a ratio) is identical either way -- triplication scales
    # numerator and denominator equally.
    assert raw["h_add"] == pytest.approx(effective["h_add"])
    assert report["effective_seed_view"]["ghost_survival"]["unique_exit_event_count"] == 1


_G1_GATE_CFG = {
    "h_add_min": 0.01,
    "additional_events_min": 20,
    "affected_videos_min": 5,
    "affected_seeds_min": 2,
    "max_single_video_event_share": 0.50,
    "max_single_seed_event_share": 0.75,
}


def test_reapply_gate_checks_flags_seed_concentration_when_only_one_effective_seed():
    # Mirrors the actual G1 v1.1 few10 shape: plenty of events/videos, but a
    # single effective seed means 100% of events sit in "one seed".
    effective_metrics = {
        "h_add": 0.0198, "additional_event_count": 165, "affected_video_count": 24,
        "affected_seed_count": 1, "max_single_video_event_share": 0.18,
        "max_single_seed_event_share": 1.0,
    }
    checks = reapply_gate_checks(effective_metrics, _G1_GATE_CFG)
    assert checks["h_add_prevalence"] is True
    assert checks["additional_event_count"] is True
    assert checks["affected_video_count"] is True
    assert checks["affected_seed_count"] is False
    assert checks["not_concentrated_in_one_seed"] is False


def test_summarize_g1_v1_2_gate_reapplication_flips_seed_checks_on_collapse():
    rows = _triplicated_rows()
    # Pad in more distinct videos so affected_video_count/event thresholds are
    # not the limiting factor -- isolate the seed-concentration flip.
    for video_number in range(1, 6):
        video_id = f"v{video_number + 1}"
        for seed in (2025, 2026, 2027):
            rows.append({
                "video_id": video_id, "seed": seed, "policy": "few10",
                "concept_id": "dog", "frame_index": 0,
                "score": 0.8, "gt_present": False,
            })
    source_rows = [
        {"video_id": row["video_id"], "frame_index": row["frame_index"],
         "concept_id": row["concept_id"], "score": 0.0}
        for row in rows
    ]
    report = summarize_g1_v1_2(
        rows, [], source_rows, threshold=0.5, primary_policy="few10",
        bootstrap_iterations=20, ghost_horizon=1,
        declared_deterministic_decoder=True, gate_cfg=_G1_GATE_CFG,
    )
    reapplied = report["gate_reapplication"]
    assert reapplied["raw_declared_seed_checks"]["affected_seed_count"] is True
    assert reapplied["effective_seed_checks"]["affected_seed_count"] is False
    assert reapplied["effective_seed_checks"]["not_concentrated_in_one_seed"] is False
    assert "affected_seed_count" in reapplied["checks_that_flip_to_fail_under_effective_seeds"]
    assert "not_concentrated_in_one_seed" in reapplied["checks_that_flip_to_fail_under_effective_seeds"]
    assert reapplied["raw_declared_seed_status"] == "PASSED"
    assert reapplied["effective_seed_status"] == "NOT_PASSED"


def test_summarize_g1_v1_2_fails_closed_without_deterministic_declaration():
    rows = _triplicated_rows()
    source_rows = [
        {"video_id": "v1", "frame_index": frame, "concept_id": "dog", "score": 0.0}
        for frame in range(5)
    ]
    with pytest.raises(ValueError, match="deterministic-decoder declaration"):
        summarize_g1_v1_2(
            rows, [], source_rows, threshold=0.5, primary_policy="few10",
            bootstrap_iterations=20, ghost_horizon=1,
            declared_deterministic_decoder=False,
        )


def test_effective_seed_groups_from_hashes_mirrors_score_based_shape():
    hashes = {"few10": {2025: "aaa", 2026: "aaa", 2027: "aaa"}}
    info = effective_seed_groups_from_hashes(hashes)["few10"]
    assert info["declared_seed_count"] == 3
    assert info["effective_seed_count"] == 1
    assert info["canonical_seeds"] == [2025]
    assert info["deterministic_collapse"] is True


def test_effective_seed_groups_from_hashes_keeps_distinct_hashes_separate():
    hashes = {"few10": {2025: "aaa", 2026: "bbb"}}
    info = effective_seed_groups_from_hashes(hashes)["few10"]
    assert info["effective_seed_count"] == 2
    assert info["deterministic_collapse"] is False


def test_cross_validate_seed_evidence_passes_when_score_and_hash_agree():
    rows = _triplicated_rows()
    score_groups = effective_seed_groups(rows)
    hash_groups = effective_seed_groups_from_hashes({"few10": {2025: "x", 2026: "x", 2027: "x"}})
    cross_validate_seed_evidence(score_groups, hash_groups)  # must not raise


def test_cross_validate_seed_evidence_fails_closed_on_disagreement():
    rows = _triplicated_rows()
    score_groups = effective_seed_groups(rows)  # says all 3 seeds collapse
    # Hash evidence disagrees: seed 2027 is actually pixel-distinct.
    hash_groups = effective_seed_groups_from_hashes({"few10": {2025: "x", 2026: "x", 2027: "y"}})
    with pytest.raises(ValueError, match="disagrees with independent"):
        cross_validate_seed_evidence(score_groups, hash_groups)


def test_cross_validate_seed_evidence_fails_closed_on_missing_policy():
    score_groups = {"few10": {"seed_to_canonical": {2025: 2025}}}
    hash_groups = {"full50": {"seed_to_canonical": {2025: 2025}}}
    with pytest.raises(ValueError, match="missing from"):
        cross_validate_seed_evidence(score_groups, hash_groups)


def test_h_add_by_video_gives_one_row_per_policy_video_pair():
    threshold = 0.5
    reconstruction_rows = [
        {"video_id": "v1", "seed": 2025, "policy": "few10", "concept_id": "cat",
         "frame_index": 0, "score": 0.8, "gt_present": False},
        {"video_id": "v1", "seed": 2025, "policy": "few10", "concept_id": "cat",
         "frame_index": 1, "score": 0.7, "gt_present": False},
        {"video_id": "v2", "seed": 2025, "policy": "few10", "concept_id": "cat",
         "frame_index": 0, "score": 0.1, "gt_present": False},
    ]
    source_rows = [
        {"video_id": "v1", "frame_index": 0, "concept_id": "cat", "score": 0.9},
        {"video_id": "v1", "frame_index": 1, "concept_id": "cat", "score": 0.1},
        {"video_id": "v2", "frame_index": 0, "concept_id": "cat", "score": 0.0},
    ]
    result = h_add_by_video(reconstruction_rows, source_rows, threshold)
    v1 = result["few10"]["v1"]
    v2 = result["few10"]["v2"]
    assert v1["absent_opportunities"] == 2
    assert v1["raw_h_add"] == pytest.approx(1.0)
    # Only frame 1 is source-negative for v1; frame 0 is source-carried.
    assert v1["source_paired_opportunities"] == 1
    assert v1["source_paired_h_add"] == pytest.approx(1.0)
    assert v2["absent_opportunities"] == 1
    assert v2["raw_h_add"] == pytest.approx(0.0)


def test_h_add_by_video_fails_closed_on_duplicate_source_key():
    reconstruction_rows = [
        {"video_id": "v1", "seed": 2025, "policy": "few10", "concept_id": "cat",
         "frame_index": 0, "score": 0.8, "gt_present": False},
    ]
    source_rows = [
        {"video_id": "v1", "frame_index": 0, "concept_id": "cat", "score": 0.9},
        {"video_id": "v1", "frame_index": 0, "concept_id": "cat", "score": 0.1},  # duplicate key
    ]
    with pytest.raises(ValueError, match="duplicate source key"):
        h_add_by_video(reconstruction_rows, source_rows, 0.5)


def test_h_add_by_video_fails_closed_on_missing_source_pair():
    reconstruction_rows = [
        {"video_id": "v1", "seed": 2025, "policy": "few10", "concept_id": "cat",
         "frame_index": 0, "score": 0.8, "gt_present": False},
        {"video_id": "v1", "seed": 2025, "policy": "few10", "concept_id": "cat",
         "frame_index": 1, "score": 0.2, "gt_present": False},
    ]
    source_rows = [
        {"video_id": "v1", "frame_index": 0, "concept_id": "cat", "score": 0.9},
        # frame_index 1 has no source row at all.
    ]
    with pytest.raises(ValueError, match="no matching source key"):
        h_add_by_video(reconstruction_rows, source_rows, 0.5)
