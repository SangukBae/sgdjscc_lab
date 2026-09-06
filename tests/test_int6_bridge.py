"""Unit tests for the pure fixed_int4 vs fixed_int6 bridge accounting logic.

No GPU, network, or dataset access -- these exercise
`src/sgdjscc_lab/evaluators/int6_bridge.py` against small synthetic
per-video metric rows shaped like `run_transmission_reduction_eval.py`'s
`per_video_metrics.csv`, including the blank ``mean_lpips`` pattern actually
observed there when a run passed ``--no-lpips``.
"""

from __future__ import annotations

import pytest

from sgdjscc_lab.evaluators.int6_bridge import (
    bootstrap_mean_ci,
    coerce_finite_float,
    one_value_per_video,
    paired_condition_metrics,
    paired_delta_from_video_values,
    paired_delta_video_clustered,
    quality_gate_pass,
    summarize_condition_metrics,
    validate_exact_video_count,
    validate_gt_present_consistency,
    validate_matching_video_sets,
    validate_unique_keys,
    video_clustered_bootstrap_mean,
)


def _condition_of(row):
    return f"{row['config']}__{row['decoder_mode']}"


def _video_id_of(row):
    return row["video"]


def test_bootstrap_mean_ci_brackets_the_sample_mean():
    values = [10.0, 12.0, 11.0, 9.0, 13.0]
    ci = bootstrap_mean_ci(values, iterations=500)
    mean = sum(values) / len(values)
    assert ci["lower"] <= mean <= ci["upper"]


def test_bootstrap_mean_ci_handles_empty_input():
    assert bootstrap_mean_ci([], iterations=500) == {"lower": None, "upper": None}


def test_coerce_finite_float_accepts_numeric_strings():
    assert coerce_finite_float("24.5", context="x") == pytest.approx(24.5)


@pytest.mark.parametrize("bad_value", ["", "  ", None, "nan", "inf", "-inf", "not_a_number"])
def test_coerce_finite_float_rejects_blank_nan_inf_and_unparseable(bad_value):
    with pytest.raises(ValueError):
        coerce_finite_float(bad_value, context="x")


def test_validate_exact_video_count_accepts_matching_unique_set():
    validate_exact_video_count(["v1", "v2", "v3"], expected=3, label="formal")


def test_validate_exact_video_count_rejects_wrong_count():
    with pytest.raises(ValueError, match="expected exactly 40"):
        validate_exact_video_count(["v1", "v2"], expected=40, label="formal")


def test_validate_exact_video_count_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        validate_exact_video_count(["v1", "v1", "v2"], expected=2, label="formal")


def _rows_with_blank_lpips():
    """Reproduces the actual v1.1 per_video_metrics.csv shape: mean_lpips is
    a blank string (child run used --no-lpips), everything else is populated.
    """
    rows = []
    for video in ("v1", "v2", "v3"):
        rows.append({
            "video": video, "config": "fixed_int4", "decoder_mode": "few10",
            "total_bundle_bytes": "100000", "mean_psnr": "24.0", "mean_ssim": "0.74",
            "mean_lpips": "", "total_elapsed_s": "40.0",
        })
        rows.append({
            "video": video, "config": "fixed_int6", "decoder_mode": "few10",
            "total_bundle_bytes": "130000", "mean_psnr": "24.4", "mean_ssim": "0.745",
            "mean_lpips": "", "total_elapsed_s": "40.0",
        })
    return rows


def test_summarize_condition_metrics_fails_closed_on_blank_lpips_field():
    with pytest.raises(ValueError, match="mean_lpips"):
        summarize_condition_metrics(
            _rows_with_blank_lpips(), condition_of=_condition_of, video_id_of=_video_id_of,
            fields=["total_bundle_bytes", "mean_psnr", "mean_lpips"], bootstrap_iterations=50,
        )


def test_summarize_condition_metrics_succeeds_when_blank_field_excluded():
    summary = summarize_condition_metrics(
        _rows_with_blank_lpips(), condition_of=_condition_of, video_id_of=_video_id_of,
        fields=["total_bundle_bytes", "mean_psnr"], bootstrap_iterations=50,
    )
    assert summary["fixed_int4__few10"]["total_bundle_bytes"]["mean"] == pytest.approx(100000.0)


def _rows_numeric():
    rows = []
    for video in ("v1", "v2", "v3"):
        rows.append({
            "video": video, "config": "fixed_int4", "decoder_mode": "few10",
            "total_bundle_bytes": 100000.0, "mean_psnr": 24.0, "mean_ssim": 0.74,
            "mean_lpips": 0.25, "total_elapsed_s": 40.0,
        })
        rows.append({
            "video": video, "config": "fixed_int6", "decoder_mode": "few10",
            "total_bundle_bytes": 130000.0, "mean_psnr": 24.4, "mean_ssim": 0.745,
            "mean_lpips": 0.245, "total_elapsed_s": 40.0,
        })
    return rows


def test_summarize_condition_metrics_groups_and_aggregates_by_condition():
    fields = ["total_bundle_bytes", "mean_psnr", "mean_ssim", "mean_lpips", "total_elapsed_s"]
    summary = summarize_condition_metrics(
        _rows_numeric(), condition_of=_condition_of, video_id_of=_video_id_of,
        fields=fields, bootstrap_iterations=200,
    )
    assert set(summary) == {"fixed_int4__few10", "fixed_int6__few10"}
    int4 = summary["fixed_int4__few10"]
    assert int4["n_videos"] == 3
    assert int4["total_bundle_bytes"]["mean"] == pytest.approx(100000.0)
    assert int4["mean_psnr"]["bootstrap_95ci"]["lower"] <= 24.0 <= int4["mean_psnr"]["bootstrap_95ci"]["upper"]


def test_summarize_condition_metrics_fails_closed_on_duplicate_video_in_condition():
    rows = _rows_numeric() + [{
        "video": "v1", "config": "fixed_int4", "decoder_mode": "few10",
        "total_bundle_bytes": 999.0, "mean_psnr": 1.0, "mean_ssim": 1.0,
        "mean_lpips": 1.0, "total_elapsed_s": 1.0,
    }]
    with pytest.raises(ValueError, match="duplicate video_id"):
        summarize_condition_metrics(
            rows, condition_of=_condition_of, video_id_of=_video_id_of,
            fields=["mean_psnr"], bootstrap_iterations=50,
        )


def test_one_value_per_video_extracts_only_the_requested_condition():
    values = one_value_per_video(
        _rows_numeric(), condition_of=_condition_of, video_id_of=_video_id_of,
        condition="fixed_int4__few10", field="mean_psnr",
    )
    assert values == {"v1": 24.0, "v2": 24.0, "v3": 24.0}


def test_paired_delta_from_video_values_computes_paired_mean_and_ci():
    reference = {"v1": 24.0, "v2": 25.0, "v3": 23.0}
    candidate = {"v1": 24.4, "v2": 25.5, "v3": 22.8}
    result = paired_delta_from_video_values(reference, candidate, bootstrap_iterations=500)
    assert result["n_videos"] == 3
    assert result["per_video_delta"]["v1"] == pytest.approx(0.4)
    assert result["per_video_delta"]["v3"] == pytest.approx(-0.2)
    assert result["mean_delta"] == pytest.approx((0.4 + 0.5 - 0.2) / 3)
    assert result["bootstrap_95ci"]["lower"] <= result["mean_delta"] <= result["bootstrap_95ci"]["upper"]


def test_paired_delta_from_video_values_rejects_mismatched_video_sets():
    reference = {"v1": 1.0, "v2": 2.0}
    candidate = {"v1": 1.0, "v3": 3.0}
    with pytest.raises(ValueError, match="missing_in_candidate"):
        paired_delta_from_video_values(reference, candidate)


def test_paired_delta_from_video_values_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one video"):
        paired_delta_from_video_values({}, {})


def test_paired_condition_metrics_computes_signed_paired_deltas():
    fields = ["total_bundle_bytes", "mean_psnr", "mean_ssim", "mean_lpips"]
    deltas = paired_condition_metrics(
        _rows_numeric(), condition_of=_condition_of, video_id_of=_video_id_of,
        reference="fixed_int4__few10", candidate="fixed_int6__few10",
        fields=fields, bootstrap_iterations=50,
    )
    assert deltas["total_bundle_bytes"]["mean_delta"] == pytest.approx(30000.0)
    assert deltas["mean_psnr"]["mean_delta"] == pytest.approx(0.4)
    assert deltas["mean_ssim"]["mean_delta"] == pytest.approx(0.005)
    assert deltas["mean_lpips"]["mean_delta"] == pytest.approx(-0.005)
    assert deltas["total_bundle_bytes"]["per_video_delta"] == {"v1": 30000.0, "v2": 30000.0, "v3": 30000.0}


def test_paired_condition_metrics_fails_closed_on_missing_video_in_one_condition():
    rows = _rows_numeric() + [{
        "video": "v4", "config": "fixed_int4", "decoder_mode": "few10",
        "total_bundle_bytes": 100000.0, "mean_psnr": 24.0, "mean_ssim": 0.74,
        "mean_lpips": 0.25, "total_elapsed_s": 40.0,
    }]
    with pytest.raises(ValueError, match="missing_in_candidate"):
        paired_condition_metrics(
            rows, condition_of=_condition_of, video_id_of=_video_id_of,
            reference="fixed_int4__few10", candidate="fixed_int6__few10",
            fields=["mean_psnr"], bootstrap_iterations=50,
        )


def test_quality_gate_pass_matches_repo_standard_thresholds():
    within_budget = {"mean_psnr": -0.4, "mean_ssim": -0.005, "mean_lpips": 0.005}
    result = quality_gate_pass(
        within_budget, psnr_drop_db_max=0.5, ssim_drop_max=0.01, lpips_rise_max=0.02,
    )
    assert result["passed"] is True

    over_budget = {"mean_psnr": -0.6, "mean_ssim": -0.005, "mean_lpips": 0.005}
    result = quality_gate_pass(
        over_budget, psnr_drop_db_max=0.5, ssim_drop_max=0.01, lpips_rise_max=0.02,
    )
    assert result["passed"] is False
    assert result["psnr_within_budget"] is False
    assert result["ssim_within_budget"] is True


def test_video_clustered_bootstrap_mean_brackets_the_true_mean_with_multi_point_videos():
    # v1 contributes 2 points, v2 contributes 1 -- resampling must keep v1's
    # two points together whenever v1 is drawn.
    values_by_video = {"v1": [0.0, 0.2], "v2": [1.0]}
    ci = video_clustered_bootstrap_mean(values_by_video, iterations=500)
    all_values = [0.0, 0.2, 1.0]
    mean = sum(all_values) / len(all_values)
    assert ci["lower"] <= mean <= ci["upper"]


def test_video_clustered_bootstrap_mean_handles_empty_input():
    assert video_clustered_bootstrap_mean({}, iterations=200) == {"lower": None, "upper": None}


def test_paired_delta_video_clustered_computes_delta_with_multiple_events_per_video():
    # v1 has two EXIT events, v2 has one -- exercises the "one video, several
    # ghost events" case explicitly.
    reference = {"v1": [0.5, 0.25], "v2": [0.1]}
    candidate = {"v1": [0.6, 0.20], "v2": [0.0]}
    result = paired_delta_video_clustered(reference, candidate, bootstrap_iterations=500)
    assert result["n_videos"] == 2
    assert result["n_points"] == 3
    assert result["per_video_delta"]["v1"] == [pytest.approx(0.1), pytest.approx(-0.05)]
    assert result["per_video_delta"]["v2"] == [pytest.approx(-0.1)]
    expected_mean = (0.1 - 0.05 - 0.1) / 3
    assert result["mean_delta"] == pytest.approx(expected_mean)
    assert result["bootstrap_95ci"]["lower"] <= result["mean_delta"] <= result["bootstrap_95ci"]["upper"]


def test_paired_delta_video_clustered_rejects_mismatched_video_sets():
    with pytest.raises(ValueError, match="missing_in_candidate"):
        paired_delta_video_clustered({"v1": [0.1]}, {"v2": [0.1]})


def test_paired_delta_video_clustered_rejects_mismatched_point_count_within_a_video():
    with pytest.raises(ValueError, match="point count differs"):
        paired_delta_video_clustered({"v1": [0.1, 0.2]}, {"v1": [0.1]})


def test_validate_unique_keys_passes_when_all_keys_distinct():
    rows = [{"k": "a"}, {"k": "b"}]
    validate_unique_keys(rows, key_of=lambda r: r["k"], label="test")


def test_validate_unique_keys_fails_closed_on_duplicate():
    rows = [{"k": "a"}, {"k": "a"}]
    with pytest.raises(ValueError, match="duplicate key"):
        validate_unique_keys(rows, key_of=lambda r: r["k"], label="test")


def test_validate_matching_video_sets_passes_when_all_equal():
    validate_matching_video_sets({"int4": ["v1", "v2"], "int6": ["v2", "v1"]}, label="test")


def test_validate_matching_video_sets_fails_closed_on_set_mismatch_same_count():
    with pytest.raises(ValueError, match="differs from"):
        validate_matching_video_sets({"int4": ["v1", "v2"], "int6": ["v1", "v3"]}, label="test")


def test_validate_gt_present_consistency_passes_when_equal():
    rows_a = [{"key": "k1", "gt_present": True}]
    rows_b = [{"key": "k1", "gt_present": True}]
    validate_gt_present_consistency(rows_a, rows_b, key_of=lambda r: r["key"], label_a="a", label_b="b")


def test_validate_gt_present_consistency_fails_closed_on_disagreement():
    rows_a = [{"key": "k1", "gt_present": True}]
    rows_b = [{"key": "k1", "gt_present": False}]
    with pytest.raises(ValueError, match="gt_present disagrees"):
        validate_gt_present_consistency(rows_a, rows_b, key_of=lambda r: r["key"], label_a="a", label_b="b")
