"""Tests for the fixed_int4 vs fixed_int6 bridge script.

Four kinds of coverage, all GPU-free:

1. Pure aggregation (`build_summary`, `build_ghost_delta`,
   `compute_condition_mean_lpips`, `fill_condition_lpips`,
   `lpips_evaluator_provenance`) against small synthetic fixtures, including
   every fail-closed validation path and the smoke `OMITTED_ASYMMETRIC_SMOKE`
   contract.
2. The canonical LPIPS resize/pad transform, reproducing the real
   1280x630 -> 512x256 smoke case with the actual
   `run_transmission_reduction_eval.py::_load_frames` function (no GPU
   needed -- resize/pad run on CPU).
3. `check_run_spec_resume`, `verify_critical_files_tracked_at_head` (using a
   throwaway temp git repo), and `gather_provenance`/`_execute`'s
   write-ordering guarantee, all in isolation.
4. Real, read-only checks (`verify_int4_reuse`, `load_int4_rows`,
   `load_int4_accounting`) against the actual frozen
   `outputs/negative_semantics_g1_pilot_rtx4080_v1_1` run already on this
   machine -- skipped if that run is not present.

Nothing here calls `run()` end to end or touches CUDA: reconstruction and
OWLv2 scoring are GPU steps the user runs separately (see
docs/experiments/2026-09-06_negative_semantics_int6_bridge_preparation.md).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
V1_1_RUN = ROOT / "outputs/negative_semantics_g1_pilot_rtx4080_v1_1"
SCRIPT_PATH = ROOT / "scripts/run_negative_semantics_int6_bridge.py"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_help_runs_as_a_subprocess_without_cuda_env():
    """`--help` must work even with CUDA hidden from the process -- argparse
    exits before any GPU-touching function is ever called.
    """
    import os

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert completed.returncode == 0
    assert "fixed_int4" in completed.stdout


_PROTOCOL_FIXTURE = {
    "protocol_id": "negative_semantics_int6_bridge_v1_0",
    "reconstruction": {
        "policies": {"few10": 10}, "reference_bit_depth": "fixed_int4",
        "guide_profile": "candidate_both_omit",
    },
    "evaluator": {"concepts": ["dog"]},
    "metrics": {
        "statistics": {"bootstrap_iterations": 50},
        "ghost_survival_auc": {"horizon_annotated_timesteps": 4},
        "quality_gate": {"psnr_drop_db_max": 0.5, "ssim_drop_max": 0.01, "lpips_rise_max": 0.02},
    },
}


def _accounting_row(video, condition, **overrides):
    row = {
        "video": video, "condition": condition,
        "total_bundle_bytes": "100000", "total_bundle_bytes_per_frame": "1000",
        "mean_psnr": "24.0", "mean_ssim": "0.74", "mean_lpips": "0.25", "total_elapsed_s": "40.0",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def _detection_rows(video_ids, condition, *, score=0.2, concepts=("dog",), gt_present=False):
    rows = []
    for video_id in video_ids:
        for frame in range(2):
            for concept in concepts:
                rows.append({
                    "video_id": video_id, "seed": 2025, "condition": condition,
                    "concept_id": concept, "frame_index": frame, "score": score, "gt_present": gt_present,
                })
    return rows


def _two_video_fixture(module):
    gt_index = {
        "events": [],
        "videos": {
            video: {
                "frame_names": ["frame_00000.png", "frame_00001.png"],
                "present_concepts": [[], []],
            }
            for video in ("v1", "v2")
        },
    }
    video_ids = ["v1", "v2"]
    detection_rows = []
    accounting_rows = []
    for bit_depth, psnr_bump in (("fixed_int4", 0.0), ("fixed_int6", 0.4)):
        for policy in ("few10",):
            condition = f"{bit_depth}__{policy}"
            detection_rows += _detection_rows(video_ids, condition)
            for video in video_ids:
                accounting_rows.append(_accounting_row(
                    video, condition, mean_psnr=24.0 + psnr_bump, mean_lpips=0.25 - psnr_bump / 100,
                ))
    source_rows = [
        {"video_id": video, "frame_index": frame, "concept_id": "dog", "score": 0.0}
        for video in video_ids for frame in range(2)
    ]
    return gt_index, detection_rows, accounting_rows, source_rows


def test_build_summary_computes_paired_delta_and_quality_gate_when_formal():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    report = module.build_summary(
        _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
        threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=None, smoke=False,
    )
    delta = report["fixed_int6_vs_reference_accounting_delta"]["few10"]
    assert delta["mean_psnr"]["mean_delta"] == pytest.approx(0.4)
    assert delta["mean_psnr"]["per_video_delta"] == {"v1": pytest.approx(0.4), "v2": pytest.approx(0.4)}
    assert report["fixed_int6_quality_gate"]["few10"]["passed"] is True
    absolute = report["by_condition_accounting_absolute"]
    assert absolute["fixed_int4__few10"]["mean_psnr"]["mean"] == pytest.approx(24.0)
    assert absolute["fixed_int6__few10"]["mean_psnr"]["mean"] == pytest.approx(24.4)
    assert report["heldout_accessed"] is False
    assert "evidence_scope" not in report  # stamped by _execute, not build_summary


def test_build_summary_omits_asymmetric_accounting_and_quality_when_smoke():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    report = module.build_summary(
        _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
        threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=2, smoke=True,
    )
    assert report["by_condition_accounting_absolute"] == module.OMITTED_ASYMMETRIC_SMOKE
    assert report["fixed_int6_vs_reference_accounting_delta"] == module.OMITTED_ASYMMETRIC_SMOKE
    assert report["fixed_int6_quality_gate"] == module.OMITTED_ASYMMETRIC_SMOKE
    # Detection-based metrics remain real computations even for smoke, since
    # both conditions are restricted to the same frame budget.
    assert report["h_add_estimands"]["pooled_absolute"]["by_condition"]
    assert report["h_add_estimands"]["paired_video_delta"]["by_policy"]["few10"]


def test_build_summary_reports_pooled_absolute_and_paired_delta_as_distinct_estimands():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    report = module.build_summary(
        _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
        threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=None, smoke=False,
    )
    estimands = report["h_add_estimands"]
    assert "pooled" in estimands["pooled_absolute"]["definition"] or "pooled" in estimands["pooled_absolute"]["definition"].lower()
    assert "per-video" in estimands["paired_video_delta"]["definition"]
    pooled = estimands["pooled_absolute"]["by_condition"]["fixed_int4__few10"]
    assert "absent_opportunities" in pooled and "raw_additional_detections" in pooled  # numerator/denominator


def test_build_summary_fails_closed_on_wrong_video_count():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    with pytest.raises(ValueError, match="expected exactly 3"):
        module.build_summary(
            _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
            threshold=0.5, expected_video_ids=["v1", "v2", "v3"], max_frames=None, smoke=False,
        )


def test_build_summary_fails_closed_on_missing_condition():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    detection_rows = [row for row in detection_rows if row["condition"] != "fixed_int6__few10"]
    accounting_rows = [row for row in accounting_rows if row["condition"] != "fixed_int6__few10"]
    with pytest.raises(ValueError, match="missing (detection|accounting) rows for condition"):
        module.build_summary(
            _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
            threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=None, smoke=False,
        )


def test_build_summary_fails_closed_on_detector_key_mismatch():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    for row in detection_rows:
        if row["condition"] == "fixed_int6__few10":
            row["concept_id"] = "cat"
    with pytest.raises(ValueError, match="detector concept_id sets differ"):
        module.build_summary(
            _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
            threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=None, smoke=False,
        )


def test_build_summary_fails_closed_on_duplicate_video_in_condition():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    accounting_rows.append(_accounting_row("v1", "fixed_int4__few10"))
    with pytest.raises(ValueError, match="duplicate"):
        module.build_summary(
            _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
            threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=None, smoke=False,
        )


def test_build_summary_fails_closed_on_blank_numeric_value():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    accounting_rows[0]["mean_lpips"] = ""
    with pytest.raises(ValueError, match="mean_lpips"):
        module.build_summary(
            _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
            threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=None, smoke=False,
        )


def test_build_summary_fails_closed_on_duplicate_detection_key():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    # Duplicate an existing (video, frame, concept) row within one condition.
    dup = dict(detection_rows[0])
    detection_rows.append(dup)
    with pytest.raises(ValueError, match="duplicate key"):
        module.build_summary(
            _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
            threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=None, smoke=False,
        )


def test_build_summary_fails_closed_on_video_set_mismatch_same_count():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    # fixed_int6 covers {v1, v3} instead of {v1, v2} -- same count, different set.
    for row in detection_rows:
        if row["condition"] == "fixed_int6__few10" and row["video_id"] == "v2":
            row["video_id"] = "v3"
    with pytest.raises(ValueError, match="key set differs"):
        module.build_summary(
            _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
            threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=None, smoke=False,
        )


def test_build_summary_fails_closed_on_gt_present_disagreement():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    for row in detection_rows:
        if row["condition"] == "fixed_int6__few10" and row["video_id"] == "v1" and row["frame_index"] == 0:
            row["gt_present"] = True
    with pytest.raises(ValueError, match="gt_present disagrees"):
        module.build_summary(
            _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
            threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=None, smoke=False,
        )


@pytest.mark.parametrize("row_kind", ["detection", "accounting"])
def test_build_summary_rejects_unexpected_condition(row_kind):
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    if row_kind == "detection":
        detection_rows.append(dict(detection_rows[0], condition="unexpected"))
    else:
        accounting_rows.append(dict(accounting_rows[0], condition="unexpected"))
    with pytest.raises(ValueError, match=f"unexpected {row_kind} condition"):
        module.build_summary(
            _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
            threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=None, smoke=False,
        )


def test_build_summary_rejects_missing_source_key_even_when_gt_present():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    gt_index["videos"]["v1"]["present_concepts"][0] = ["dog"]
    for row in detection_rows:
        if row["video_id"] == "v1" and row["frame_index"] == 0 and row["concept_id"] == "dog":
            row["gt_present"] = True
    source_rows = [
        row for row in source_rows
        if not (row["video_id"] == "v1" and row["frame_index"] == 0 and row["concept_id"] == "dog")
    ]
    with pytest.raises(ValueError, match="source key set differs"):
        module.build_summary(
            _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
            threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=None, smoke=False,
        )


def test_build_summary_rejects_conditions_that_agree_on_wrong_official_gt():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index, detection_rows, accounting_rows, source_rows = _two_video_fixture(module)
    for row in detection_rows:
        if row["video_id"] == "v1" and row["frame_index"] == 0 and row["concept_id"] == "dog":
            row["gt_present"] = True
    with pytest.raises(ValueError, match="official gt_index"):
        module.build_summary(
            _PROTOCOL_FIXTURE, gt_index, detection_rows, accounting_rows, source_rows,
            threshold=0.5, expected_video_ids=["v1", "v2"], max_frames=None, smoke=False,
        )


def test_build_ghost_delta_classifies_all_four_censoring_buckets_with_multi_event_video():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index = {
        "events": [
            {"event_id": "exit-a", "event_type": "EXIT", "video_id": "v1", "concept_id": "dog", "start_frame": 0},
            {"event_id": "exit-b", "event_type": "EXIT", "video_id": "v1", "concept_id": "dog", "start_frame": 10},
            {"event_id": "exit-c", "event_type": "EXIT", "video_id": "v2", "concept_id": "dog", "start_frame": 0},
            {"event_id": "exit-d", "event_type": "EXIT", "video_id": "v3", "concept_id": "dog", "start_frame": 0},
        ],
    }
    rows = []
    # v1: two EXIT events, both jointly uncensored in both conditions.
    # Per-frame scores (threshold=0.5) chosen so the AUC (fraction of the
    # 4-frame horizon scoring >= threshold) differs between conditions:
    # event exit-a: reference AUC=1.0 (4/4), candidate AUC=0.5 (2/4) -> delta -0.5
    # event exit-b: reference AUC=0.0 (0/4), candidate AUC=0.25 (1/4) -> delta +0.25
    event_frame_scores = {
        (0, "fixed_int4__few10"): [0.6, 0.6, 0.6, 0.6],
        (0, "fixed_int6__few10"): [0.6, 0.6, 0.1, 0.1],
        (10, "fixed_int4__few10"): [0.1, 0.1, 0.1, 0.1],
        (10, "fixed_int6__few10"): [0.6, 0.1, 0.1, 0.1],
    }
    for (event_start, condition), scores in event_frame_scores.items():
        for offset, score in enumerate(scores):
            rows.append({
                "video_id": "v1", "seed": 2025, "condition": condition, "concept_id": "dog",
                "frame_index": event_start + offset, "score": score, "gt_present": False,
            })
    # v2: uncensored in fixed_int4, right-censored in fixed_int6 (GT reappears).
    for frame in range(4):
        rows.append({
            "video_id": "v2", "seed": 2025, "condition": "fixed_int4__few10", "concept_id": "dog",
            "frame_index": frame, "score": 0.1, "gt_present": False,
        })
    for frame in range(4):
        rows.append({
            "video_id": "v2", "seed": 2025, "condition": "fixed_int6__few10", "concept_id": "dog",
            "frame_index": frame, "score": 0.1, "gt_present": frame >= 2,
        })
    # v3: right-censored in both conditions.
    for condition in ("fixed_int4__few10", "fixed_int6__few10"):
        for frame in range(4):
            rows.append({
                "video_id": "v3", "seed": 2025, "condition": condition, "concept_id": "dog",
                "frame_index": frame, "score": 0.1, "gt_present": frame >= 1,
            })
    result = module.build_ghost_delta(
        gt_index, rows, threshold=0.5, ghost_horizon=4, bootstrap_iterations=100,
        reference_bit_depth="fixed_int4", policies=["few10"],
        expected_video_ids=["v1", "v2", "v3"],
    )
    breakdown = result["few10"]["censoring_breakdown"]
    assert breakdown["jointly_uncensored"] == {"n_events": 2, "n_videos": 1}
    assert breakdown["candidate_only_censored"] == {"n_events": 1, "n_videos": 1}  # v2, censored in candidate (fixed_int6)
    assert breakdown["reference_only_censored"] == {"n_events": 0, "n_videos": 0}
    assert breakdown["jointly_censored"] == {"n_events": 1, "n_videos": 1}  # v3
    paired = result["few10"]["paired_delta"]
    assert paired["n_videos"] == 1
    assert paired["n_points"] == 2
    assert paired["per_video_delta"]["v1"] == [pytest.approx(-0.5), pytest.approx(0.25)]
    assert paired["mean_delta"] == pytest.approx(-0.125)


def test_build_ghost_report_omits_two_frame_smoke_before_touching_full_gt_events():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    gt_index = {
        "events": [
            {"event_id": f"exit-{index}", "event_type": "EXIT", "video_id": f"v{index}",
             "concept_id": "dog", "start_frame": 0}
            for index in range(10)
        ]
    }
    report = module.build_ghost_report(
        gt_index, [], threshold=0.5, ghost_horizon=16, bootstrap_iterations=50,
        reference_bit_depth="fixed_int4", policies=["few10"],
        expected_video_ids=["v0"], smoke=True, max_frames=2,
    )
    assert report["status"] == module.OMITTED_INSUFFICIENT_HORIZON_SMOKE
    assert report["available_frame_cap"] == 2
    assert report["required_horizon_frames"] == 16


def test_fill_condition_lpips_overwrites_blank_values_in_place():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    rows = [_accounting_row("v1", "fixed_int4__few10", mean_lpips="")]
    module.fill_condition_lpips(rows, condition_label="fixed_int4__few10", lpips_by_video={"v1": 0.31})
    assert rows[0]["mean_lpips"] == 0.31


def test_fill_condition_lpips_fails_closed_on_missing_video():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    rows = [_accounting_row("v1", "fixed_int4__few10")]
    with pytest.raises(ValueError, match="no computed LPIPS"):
        module.fill_condition_lpips(rows, condition_label="fixed_int4__few10", lpips_by_video={})


# --- LPIPS canonical transform + strict comparison --------------------------

def test_canonical_transform_reproduces_the_real_1280x630_to_512x256_smoke_case(tmp_path):
    """The real int6 bridge smoke video is 1280x630; run_negative_semantics_g1.py
    resizes+pads it to 512x256 before reconstruction (see the v1.1 memory
    amendment doc). This reproduces that exact case with the actual
    `_load_frames` transform, GPU-free.
    """
    from PIL import Image

    module = _load_script("run_negative_semantics_int6_bridge.py")
    tre = module._load_transmission_reduction_eval_support()

    source_dir = tmp_path / "frames"
    source_dir.mkdir()
    Image.new("RGB", (1280, 630), (10, 20, 30)).save(source_dir / "img_0000001.jpg")

    tensors, info = tre._load_frames(source_dir, tmp_path, image_long_side=512, image_pad_multiple=128)
    assert len(tensors) == 1
    assert tuple(tensors[0].shape[-2:]) == (256, 512)
    assert info["padding_rule"] == "symmetric_constant_zero_extra_pixel_bottom_right"


def test_build_canonical_source_frames_matches_recon_shape_and_lpips_succeeds(tmp_path, monkeypatch):
    """End-to-end (GPU-free, fake evaluator): a 1280x630 source and a
    512x256 reconstruction frame -- after the canonical transform -- have
    matching shapes, so compute_condition_mean_lpips succeeds.
    """
    from PIL import Image

    module = _load_script("run_negative_semantics_int6_bridge.py")
    tre = module._load_transmission_reduction_eval_support()
    monkeypatch.setattr(module, "ROOT", tmp_path)

    source_dir = tmp_path / "source" / "v1"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (1280, 630), (10, 20, 30)).save(source_dir / "img_0000001.jpg")

    recon_dir = tmp_path / "recon"
    recon_dir.mkdir()
    recon_path = recon_dir / "frame_00000.png"
    Image.new("RGB", (512, 256), (12, 22, 32)).save(recon_path)

    gt_index = {"videos": {"v1": {"source_path": "source/v1", "frame_names": ["img_0000001.jpg"]}}}
    source_tensors_by_video = module.build_canonical_source_frames(
        tre, gt_index, ["v1"], image_long_side=512, image_pad_multiple=128, max_frames=None,
    )
    assert tuple(source_tensors_by_video["v1"][0].shape[-2:]) == (256, 512)

    class FakeEvaluator:
        def evaluate(self, source_tensor, recon_tensor):
            assert tuple(source_tensor.shape) == tuple(recon_tensor.shape)
            return {"lpips": 0.2}

    result = module.compute_condition_mean_lpips(
        ["v1"], source_tensors_by_video, {"v1": [recon_path]}, evaluator=FakeEvaluator(),
    )
    assert result == {"v1": pytest.approx(0.2)}


def test_build_canonical_source_frames_fails_closed_on_frame_count_mismatch(tmp_path, monkeypatch):
    from PIL import Image

    module = _load_script("run_negative_semantics_int6_bridge.py")
    tre = module._load_transmission_reduction_eval_support()
    monkeypatch.setattr(module, "ROOT", tmp_path)

    source_dir = tmp_path / "source" / "v1"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (1, 2, 3)).save(source_dir / "img_0000001.jpg")

    # Declares 2 frames but only 1 actually exists on disk.
    gt_index = {"videos": {"v1": {"source_path": "source/v1", "frame_names": ["a.jpg", "b.jpg"]}}}
    with pytest.raises(ValueError, match="canonical source frame count"):
        module.build_canonical_source_frames(
            tre, gt_index, ["v1"], image_long_side=128, image_pad_multiple=128, max_frames=None,
        )


def test_compute_condition_mean_lpips_fails_closed_on_raw_vs_canonical_shape_mismatch(tmp_path):
    """Comparing a RAW (untransformed) source tensor against a canonically
    -sized reconstruction must fail closed with a clear shape error -- this
    is exactly the bug the canonical-transform requirement prevents.
    """
    import torch

    module = _load_script("run_negative_semantics_int6_bridge.py")
    raw_source_tensor = torch.zeros(1, 3, 630, 1280)  # untransformed 1280x630
    recon_path = tmp_path / "frame_00000.png"
    from PIL import Image
    Image.new("RGB", (512, 256), (1, 2, 3)).save(recon_path)

    class FakeEvaluator:
        def evaluate(self, a, b):
            raise AssertionError("must not be called after a shape mismatch is detected")

    with pytest.raises(ValueError, match="shape"):
        module.compute_condition_mean_lpips(
            ["v1"], {"v1": [raw_source_tensor]}, {"v1": [recon_path]}, evaluator=FakeEvaluator(),
        )


def test_compute_condition_mean_lpips_fails_closed_on_frame_count_mismatch():
    module = _load_script("run_negative_semantics_int6_bridge.py")

    class FakeEvaluator:
        def evaluate(self, a, b):
            return {"lpips": 0.1}

    with pytest.raises(ValueError, match="frame"):
        module.compute_condition_mean_lpips(
            ["v1"], {"v1": [object(), object()]}, {"v1": [Path("r0")]}, evaluator=FakeEvaluator(),
        )


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf")])
def test_compute_condition_mean_lpips_fails_closed_on_none_nan_inf(tmp_path, bad_value):
    from PIL import Image

    module = _load_script("run_negative_semantics_int6_bridge.py")
    recon_path = tmp_path / "frame_00000.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(recon_path)

    class FakeEvaluator:
        def evaluate(self, a, b):
            return {"lpips": bad_value}

    class FakeTensor:
        shape = (1, 3, 4, 4)  # matches both sides so the shape check passes

    with pytest.raises(ValueError):
        module.compute_condition_mean_lpips(
            ["v1"], {"v1": [FakeTensor()]}, {"v1": [recon_path]}, evaluator=FakeEvaluator(),
            load_recon_tensor=lambda p: FakeTensor(),
        )


def test_lpips_evaluator_provenance_reports_net_version_preprocessing_and_deterministic_hash():
    import torch

    module = _load_script("run_negative_semantics_int6_bridge.py")

    class FakeLpipsModel:
        def state_dict(self):
            return {"a.weight": torch.tensor([1.0, 2.0]), "b.weight": torch.tensor([3.0])}

    class FakeEvaluator:
        lpips_net = "vgg"

        def _get_lpips(self):
            return FakeLpipsModel()

    provenance_1 = module.lpips_evaluator_provenance(FakeEvaluator())
    provenance_2 = module.lpips_evaluator_provenance(FakeEvaluator())
    assert provenance_1["net"] == "vgg"
    assert provenance_1["preprocessing"] == module.LPIPS_PREPROCESSING_DESCRIPTION
    assert provenance_1["state_dict_sha256"] == provenance_2["state_dict_sha256"]  # deterministic
    assert len(provenance_1["state_dict_sha256"]) == 64


def test_lpips_input_frame_hashes_covers_source_int4_int6(tmp_path):
    module = _load_script("run_negative_semantics_int6_bridge.py")
    for name in ("s.png", "i4.png", "i6.png"):
        (tmp_path / name).write_bytes(b"x")
    hashes = module.lpips_input_frame_hashes(
        ["v1"], {"v1": [tmp_path / "s.png"]}, {"v1": [tmp_path / "i4.png"]}, {"v1": [tmp_path / "i6.png"]},
    )
    assert set(hashes) == {"source", "fixed_int4", "fixed_int6"}
    assert set(hashes["source"]) == {"v1"}


# --- provenance/resume ------------------------------------------------------

def test_check_run_spec_resume_allows_matching_spec():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    spec = {"mode": "formal", "seed": 2025}
    module.check_run_spec_resume(dict(spec), spec, run_root=Path("/tmp/x"))  # must not raise


def test_check_run_spec_resume_allows_first_run_with_no_existing_spec():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    module.check_run_spec_resume(None, {"mode": "formal"}, run_root=Path("/tmp/x"))  # must not raise


def test_check_run_spec_resume_rejects_any_field_mismatch():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    with pytest.raises(SystemExit, match="resume run_spec mismatch"):
        module.check_run_spec_resume(
            {"mode": "formal", "seed": 2025}, {"mode": "formal", "seed": 2026}, run_root=Path("/tmp/x"),
        )


def _init_git_repo(root: Path) -> None:
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "test"],
    ):
        subprocess.run(args, cwd=root, check=True, capture_output=True)


def _git_commit_all(root: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=root, check=True, capture_output=True)


def test_verify_critical_files_tracked_at_head_passes_when_clean(tmp_path):
    module = _load_script("run_negative_semantics_int6_bridge.py")
    _init_git_repo(tmp_path)
    (tmp_path / "critical.py").write_text("x = 1\n")
    (tmp_path / "unrelated_untracked.txt").write_text("scratch\n")  # must never block
    _git_commit_all(tmp_path, "initial")
    checked = module.verify_critical_files_tracked_at_head(tmp_path, ["critical.py"])
    assert "critical.py" in checked


def test_verify_critical_files_tracked_at_head_rejects_untracked_critical_file(tmp_path):
    module = _load_script("run_negative_semantics_int6_bridge.py")
    _init_git_repo(tmp_path)
    (tmp_path / "committed.py").write_text("x = 1\n")
    _git_commit_all(tmp_path, "initial")
    (tmp_path / "critical.py").write_text("x = 1\n")  # never git add-ed
    with pytest.raises(SystemExit, match="not tracked by git"):
        module.verify_critical_files_tracked_at_head(tmp_path, ["critical.py"])


def test_verify_critical_files_tracked_at_head_rejects_modified_critical_file(tmp_path):
    module = _load_script("run_negative_semantics_int6_bridge.py")
    _init_git_repo(tmp_path)
    (tmp_path / "critical.py").write_text("x = 1\n")
    _git_commit_all(tmp_path, "initial")
    (tmp_path / "critical.py").write_text("x = 2\n")  # uncommitted edit
    with pytest.raises(SystemExit, match="uncommitted changes"):
        module.verify_critical_files_tracked_at_head(tmp_path, ["critical.py"])


def test_verify_critical_files_tracked_at_head_ignores_unrelated_untracked_files(tmp_path):
    module = _load_script("run_negative_semantics_int6_bridge.py")
    _init_git_repo(tmp_path)
    (tmp_path / "critical.py").write_text("x = 1\n")
    _git_commit_all(tmp_path, "initial")
    (tmp_path / "unrelated.txt").write_text("not tracked, not critical\n")
    checked = module.verify_critical_files_tracked_at_head(tmp_path, ["critical.py"])
    assert "critical.py" in checked


def test_verify_critical_files_tracked_at_head_rejects_missing_file(tmp_path):
    module = _load_script("run_negative_semantics_int6_bridge.py")
    _init_git_repo(tmp_path)
    (tmp_path / "committed.py").write_text("x = 1\n")
    _git_commit_all(tmp_path, "initial")
    with pytest.raises(SystemExit, match="does not exist"):
        module.verify_critical_files_tracked_at_head(tmp_path, ["missing.py"])


@pytest.mark.skipif(not V1_1_RUN.is_dir(), reason="frozen G1 v1.1 run_root is not present on this machine")
def test_incompatible_resume_does_not_touch_existing_preflight_or_reuse_metadata(tmp_path):
    module = _load_script("run_negative_semantics_int6_bridge.py")
    g1 = module._load_g1_support()
    run_root = tmp_path / "run_root"
    run_root.mkdir()
    sentinel_run_spec = {"mode": "formal", "seed": 999999999}
    (run_root / "run_spec.json").write_text(json.dumps(sentinel_run_spec))
    (run_root / "preflight.json").write_text(json.dumps({"sentinel": "preflight"}))
    (run_root / "int4_reuse_verification.json").write_text(json.dumps({"sentinel": "reuse"}))

    def fake_preflight(g1, protocol, formal):
        return {
            "status": "PASSED", "git": {"commit": "deadbeef", "dirty": False},
            "checkpoint_sha256": {}, "evaluator_model": {}, "lpips_provenance": None,
            "critical_file_sha256": None,
        }

    args = argparse.Namespace(run_root=run_root, device="cpu", smoke=True, preflight_only=False)
    with pytest.raises(SystemExit, match="resume run_spec mismatch"):
        module._execute(args, g1, preflight_fn=fake_preflight)

    assert json.loads((run_root / "run_spec.json").read_text()) == sentinel_run_spec
    assert json.loads((run_root / "preflight.json").read_text()) == {"sentinel": "preflight"}
    assert json.loads((run_root / "int4_reuse_verification.json").read_text()) == {"sentinel": "reuse"}


@pytest.mark.skipif(not V1_1_RUN.is_dir(), reason="frozen G1 v1.1 run_root is not present on this machine")
def test_preflight_only_is_fully_read_only_even_for_future_formal_root(tmp_path):
    module = _load_script("run_negative_semantics_int6_bridge.py")
    g1 = module._load_g1_support()
    run_root = tmp_path / "run_root"
    run_root.mkdir()
    (run_root / "preflight.json").write_text(json.dumps({"sentinel": "formal preflight, do not touch"}))

    def fake_preflight(g1, protocol, formal):
        return {
            "status": "PASSED", "git": {"commit": "deadbeef", "dirty": False},
            "checkpoint_sha256": {}, "evaluator_model": {}, "lpips_provenance": None,
            "critical_file_sha256": None,
        }

    args = argparse.Namespace(run_root=run_root, device="cpu", smoke=False, preflight_only=True)
    exit_code = module._execute(args, g1, preflight_fn=fake_preflight)
    assert exit_code == 0
    assert not (run_root / "preflight_only.json").exists()
    assert json.loads((run_root / "preflight.json").read_text()) == {"sentinel": "formal preflight, do not touch"}
    assert not (run_root / "run_spec.json").is_file()  # preflight-only never commits to a run_spec


def test_artifact_checksum_payload_recursively_covers_scientific_outputs_only(tmp_path):
    module = _load_script("run_negative_semantics_int6_bridge.py")
    scientific = {
        "run_spec.json": b"spec\n",
        "reconstruction/fixed_int6/few10/seed_2025/per_video_metrics.csv": b"video,bytes\nv1,10\n",
        "reconstruction/fixed_int6/few10/seed_2025/summary.json": b"{}\n",
        "reconstruction/fixed_int6/few10/seed_2025/recon_videos/v1/cfg/frame_00000.png": b"png-bytes",
        "evaluator/reconstruction/fixed_int6/few10/seed_2025/v1.json": b"{}\n",
    }
    for relative, content in scientific.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    excluded = {
        "status.json": b"mutable",
        "operator.log": b"still-open",
        "reconstruction/fixed_int6/few10/seed_2025/logs/run.log": b"diagnostic",
        "stale.tmp": b"temporary",
    }
    for relative, content in excluded.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    payload = module.build_artifact_checksum_payload(tmp_path)
    assert set(payload["files"]) == set(scientific)
    assert payload["file_count"] == len(scientific)
    assert payload["total_size_bytes"] == sum(len(content) for content in scientific.values())
    for relative, content in scientific.items():
        assert payload["files"][relative]["size_bytes"] == len(content)
        assert len(payload["files"][relative]["sha256"]) == 64


@pytest.mark.skipif(not V1_1_RUN.is_dir(), reason="frozen G1 v1.1 run_root is not present on this machine")
def test_verify_int4_reuse_accepts_the_real_frozen_v1_1_run():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    g1 = module._load_g1_support()
    protocol = module._load_protocol()
    run_spec = g1._load_json(V1_1_RUN / "run_spec.json")
    video_ids = run_spec["video_ids"]
    seed = run_spec["seeds"][0]
    reuse_info = module.verify_int4_reuse(g1, protocol, video_ids, seed)
    assert reuse_info["seed"] == seed
    assert set(reuse_info["policies_verified"]) == {"few10", "full50"}
    assert reuse_info["threshold"] == pytest.approx(0.2)
    assert len(reuse_info["detection_rows_sha256"]) == 64
    assert set(reuse_info["per_video_metrics_sha256"]) == {"few10", "full50"}


@pytest.mark.skipif(not V1_1_RUN.is_dir(), reason="frozen G1 v1.1 run_root is not present on this machine")
def test_verify_int4_reuse_rejects_a_seed_outside_the_source_run():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    g1 = module._load_g1_support()
    protocol = module._load_protocol()
    run_spec = g1._load_json(V1_1_RUN / "run_spec.json")
    with pytest.raises(SystemExit, match="was not part of"):
        module.verify_int4_reuse(g1, protocol, run_spec["video_ids"], 9999)


@pytest.mark.skipif(not V1_1_RUN.is_dir(), reason="frozen G1 v1.1 run_root is not present on this machine")
def test_load_int4_rows_and_accounting_restrict_to_smoke_video_and_frame_budget():
    module = _load_script("run_negative_semantics_int6_bridge.py")
    g1 = module._load_g1_support()
    protocol = module._load_protocol()
    seed = protocol["reconstruction"]["seeds"][0]
    smoke_video = protocol["gate"]["smoke_video_id"]

    formal_rows = module.load_int4_rows(g1, protocol, seed, ["ovis_valid_0299d8d6", smoke_video], max_frames=None)
    assert {row["video_id"] for row in formal_rows} == {"ovis_valid_0299d8d6", smoke_video}

    smoke_rows = module.load_int4_rows(g1, protocol, seed, [smoke_video], max_frames=2)
    assert {row["video_id"] for row in smoke_rows} == {smoke_video}
    assert all(int(row["frame_index"]) < 2 for row in smoke_rows)
    assert len(smoke_rows) < len(formal_rows)

    accounting = module.load_int4_accounting(protocol, seed, [smoke_video])
    assert {row["video"] for row in accounting} == {smoke_video}
    assert {row["condition"] for row in accounting} == {"fixed_int4__few10", "fixed_int4__full50"}


@pytest.mark.skipif(not V1_1_RUN.is_dir(), reason="frozen G1 v1.1 run_root is not present on this machine")
def test_owlv2_cache_invalidated_when_frame_tree_hash_changes(tmp_path):
    """Item 7 (previous round): the OWLv2 cache must key on the
    reconstruction frame-tree hash, not just a frame count.
    """
    from PIL import Image

    from sgdjscc_lab.utils.frame_hash import frame_tree_sha256

    module = _load_script("run_negative_semantics_int6_bridge.py")
    g1 = module._load_g1_support()

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    path = frame_dir / "frame_00000.png"
    Image.new("RGB", (4, 4), (10, 10, 10)).save(path)

    class FakeScorer:
        concepts = ["dog"]

        def __init__(self):
            self.calls = 0

        def score_image(self, image):
            self.calls += 1
            return {"dog": 0.5}

    scorer = FakeScorer()
    cache_path = tmp_path / "cache.json"
    frame_hash_1 = frame_tree_sha256([path])
    g1._score_sequence(
        scorer, [path], cache_path, long_side=None, pad_multiple=None, crop_size=None,
        cache_metadata={"reconstruction_frame_tree_sha256": frame_hash_1},
    )
    assert scorer.calls == 1

    g1._score_sequence(
        scorer, [path], cache_path, long_side=None, pad_multiple=None, crop_size=None,
        cache_metadata={"reconstruction_frame_tree_sha256": frame_hash_1},
    )
    assert scorer.calls == 1

    Image.new("RGB", (4, 4), (200, 200, 200)).save(path)
    frame_hash_2 = frame_tree_sha256([path])
    assert frame_hash_2 != frame_hash_1
    g1._score_sequence(
        scorer, [path], cache_path, long_side=None, pad_multiple=None, crop_size=None,
        cache_metadata={"reconstruction_frame_tree_sha256": frame_hash_2},
    )
    assert scorer.calls == 2
