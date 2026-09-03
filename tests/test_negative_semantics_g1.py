from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from sgdjscc_lab.evaluators.negative_semantics import (
    collapse_additional_events,
    select_global_threshold,
    summarize_g1,
)


ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_global_threshold_is_frozen_from_gt_constraints():
    rows = [
        {"score": 0.90, "gt_present": True},
        {"score": 0.20, "gt_present": True},
        {"score": 0.10, "gt_present": False},
        {"score": 0.01, "gt_present": False},
    ]
    report = select_global_threshold(
        rows, [0.05, 0.15, 0.25], min_recall=0.5, max_false_positive_rate=0.0
    )
    assert report["status"] == "PASSED"
    assert report["selected_threshold"] == 0.25
    assert report["selected_operating_point"]["recall"] == 0.5
    assert report["selected_operating_point"]["false_positive_rate"] == 0.0


def test_additional_detections_collapse_by_consecutive_run():
    rows = []
    for frame, score, gt_present in [
        (0, 0.8, False), (1, 0.7, False), (2, 0.1, False),
        (3, 0.9, True), (4, 0.9, False),
    ]:
        rows.append({
            "video_id": "v1", "seed": 2025, "policy": "few10",
            "concept_id": "dog", "frame_index": frame,
            "score": score, "gt_present": gt_present,
        })
    events = collapse_additional_events(rows, 0.5)
    assert [(event["start_frame"], event["end_frame"]) for event in events] == [(0, 1), (4, 4)]


def test_g1_summary_reports_h_add_distribution_and_right_censoring():
    rows = []
    for video in ("v1", "v2"):
        for frame in range(18):
            rows.append({
                "video_id": video, "seed": 2025, "policy": "few10",
                "concept_id": "person", "frame_index": frame,
                "score": 0.8 if video == "v1" and frame in (2, 3) else 0.1,
                "gt_present": frame < 2,
            })
    events = [
        {"event_id": "exit-1", "event_type": "EXIT", "video_id": "v1", "concept_id": "person", "start_frame": 2},
        {"event_id": "exit-2", "event_type": "EXIT", "video_id": "v2", "concept_id": "person", "start_frame": 10},
    ]
    report = summarize_g1(
        rows, events, threshold=0.5, primary_policy="few10",
        bootstrap_iterations=50, ghost_horizon=16,
    )
    policy = report["by_policy"]["few10"]
    assert policy["absent_opportunities"] == 32
    assert policy["additional_detections"] == 2
    assert policy["additional_event_count"] == 1
    assert policy["affected_video_count"] == 1
    assert report["ghost_survival"]["by_policy"]["few10"]["uncensored_count"] == 1
    assert report["ghost_survival"]["right_censored_count"] == 1
    assert report["ghost_survival"]["by_policy"]["few10"]["mean_auc"] == 2 / 16


def test_image_sequence_loader_preserves_order_and_resizes(tmp_path):
    module = _load_script("run_transmission_reduction_eval.py")
    source = tmp_path / "frames"
    source.mkdir()
    Image.new("RGB", (800, 400), "red").save(source / "img_0000002.jpg")
    Image.new("RGB", (800, 400), "blue").save(source / "img_0000001.jpg")
    frames, info = module._load_frames(source, tmp_path / "unused", image_long_side=512)
    assert [path.name for path in info["files"]] == ["img_0000001.jpg", "img_0000002.jpg"]
    assert info["source_kind"] == "annotated_image_sequence"
    assert frames[0].shape[-2:] == (256, 512)


@pytest.mark.skipif(
    not (ROOT / "data/external/ovis/extracted/valid/annotations_by_video").is_dir(),
    reason="OVIS G0 payload is intentionally external",
)
def test_g1_preparation_uses_only_frozen_pilot_and_is_deterministic(tmp_path):
    module = _load_script("prepare_negative_semantics_g1.py")
    first = module.prepare(ROOT, tmp_path / "pilot")
    second = module.prepare(ROOT, tmp_path / "pilot")
    index = json.loads((tmp_path / "pilot/ground_truth_index.json").read_text(encoding="utf-8"))
    assert first == second
    assert first["status"] == "PREPARED"
    assert first["heldout_accessed"] is False
    assert first["video_count"] == 40
    assert first["event_count"] == 40
    assert len(index["videos"]) == 40
    assert len(index["events"]) == 40
    assert {event["event_type"] for event in index["events"]} == {
        "ENTER", "EXIT", "OCCLUDE", "REAPPEAR"
    }
    assert all((tmp_path / "pilot/frames" / video_id).is_symlink() for video_id in index["videos"])


def test_g1_protocol_keeps_heldout_sealed_and_freezes_full_matrix():
    from omegaconf import OmegaConf

    protocol = OmegaConf.to_container(OmegaConf.load(
        ROOT / "configs/experiments/negative_semantics/g1_protocol.yaml"
    ), resolve=True)
    assert protocol["freeze_status"] == "frozen"
    assert protocol["scope"]["training"] is False
    assert protocol["scope"]["heldout_access"] == "prohibited"
    assert protocol["reconstruction"]["policies"] == {"few10": 10, "full50": 50}
    assert protocol["reconstruction"]["seeds"] == [2025, 2026, 2027]
    assert protocol["evaluator"]["selector_model_ids"] == []
