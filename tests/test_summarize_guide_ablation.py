from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_guide_summary_test", _ROOT / "scripts" / "summarize_guide_ablation.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(profile: str, bytes_: int, psnr: float):
    return {
        "config": f"fixed_int4__{profile}", "selector": "fixed", "channel": "int4",
        "bit_depth": 4, "guide_profile": profile,
        "guide_family": "baseline" if profile == "baseline" else "edge",
        "guide_stage": "baseline" if profile == "baseline" else "individual",
        "edge_bit_depth": 8 if profile == "baseline" else 4,
        "uncertainty_bit_depth": 8, "edge_downsample": 1,
        "uncertainty_downsample": 1, "edge_stride": 1, "uncertainty_stride": 1,
        "edge_omit": False, "uncertainty_omit": False,
        "n_videos": 1, "all_expected_videos_present": True,
        "all_finite_metrics": True, "total_nan_or_inf_frames": 0,
        "valid_frame_ratio": 1.0, "mean_psnr": psnr, "mean_ssim": 0.8,
        "mean_lpips": 0.2, "mean_total_bundle_bytes_per_video": bytes_,
    }


def _per_video(profile: str):
    return {
        "video": "v1", "config": f"fixed_int4__{profile}",
        "guide_profile": profile, "fixed_reference_snr_db": 10.0,
    }


def _packet(profile: str, edge_bytes: int):
    return {
        "video": "v1", "config": f"fixed_int4__{profile}", "frame_index": 0,
        "guide_profile": profile, "caption_bytes": 1, "edge_bytes": edge_bytes,
        "edge_uncertainty_bytes": 4, "manifest_bytes": 1,
        "semantic_packet_bytes": 1, "visual_bytes": 2, "bundle_overhead_bytes": 1,
        "total_bundle_bytes": 10 + edge_bytes, "edge_action": 0,
        "uncertainty_action": 0,
    }


def test_summary_validates_pairs_components_quality_gate_and_selects_minimum(tmp_path):
    _write_csv(tmp_path / "aggregate.csv", [
        _aggregate("baseline", 100, 25.0),
        _aggregate("edge_q4", 80, 24.9),
    ])
    _write_csv(tmp_path / "per_video_metrics.csv", [
        _per_video("baseline"), _per_video("edge_q4"),
    ])
    _write_csv(tmp_path / "packet_components.csv", [
        _packet("baseline", 1), _packet("edge_q4", 1),
    ])
    _write_csv(tmp_path / "failed_pairs.csv", [])
    (tmp_path / "run_signature.json").write_text(json.dumps({
        "guide_profiles": ["baseline", "edge_q4"],
        "video_keys": ["v1"],
    }), encoding="utf-8")

    validation = mod.summarize(tmp_path)

    assert validation["validation_passed"] is True
    assert validation["n_observed_pairs"] == validation["n_expected_pairs"] == 2
    assert validation["selected_minimum_bytes_in_gate"] == "edge_q4"
    effect_rows = mod._read_csv(tmp_path / "guide_ablation_effect.csv")
    edge_row = next(row for row in effect_rows if row["guide_profile"] == "edge_q4")
    assert float(edge_row["byte_reduction_ratio"]) == 0.2
    assert edge_row["within_quality_gate"] == "True"


def test_parallel_provenance_requires_three_matching_clean_worker_manifests(tmp_path):
    commit = "a" * 40
    assignments = []
    statuses = []
    for index in range(3):
        worker_id = f"worker_{index:02d}"
        device = f"cuda:{index}"
        assignments.append({"worker_id": worker_id, "device": device, "videos": [f"v{index}"]})
        statuses.append({"worker_id": worker_id, "device": device, "returncode": 0})
        path = tmp_path / "workers" / worker_id / "run_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "git": {"commit": commit, "dirty": False, "branch": "main"},
            "extra": {"physical_cuda_device": device},
        }), encoding="utf-8")
    (tmp_path / "parallel_plan.json").write_text(json.dumps({
        "git": {"commit": commit, "dirty": False, "branch": "main"},
        "devices": ["cuda:0", "cuda:1", "cuda:2"],
        "assignments": assignments,
    }), encoding="utf-8")
    (tmp_path / "parallel_worker_status.json").write_text(
        json.dumps(statuses), encoding="utf-8"
    )
    assert mod._parallel_provenance(tmp_path)["passed"] is True
    statuses[1]["device"] = "cuda:2"
    (tmp_path / "parallel_worker_status.json").write_text(
        json.dumps(statuses), encoding="utf-8"
    )
    assert mod._parallel_provenance(tmp_path)["passed"] is False
