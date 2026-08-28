"""CPU-only tests for the fail-closed matched-rate result validator."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_matched_rate_validator_test",
        _ROOT / "scripts" / "validate_fixed_skem_matched_rate.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _valid_run(root: Path):
    videos = [f"v{index}" for index in range(10)]
    channels = ["float32", "int16", "int8", "int6", "int4"]
    configs = [f"{selector}_{channel}" for selector in ("fixed", "skem") for channel in channels]
    assignments = [
        {
            "worker_id": f"worker_{index:02d}", "device": f"cuda:{index}",
            "videos": videos[index::3],
        }
        for index in range(3)
    ]
    plan = {
        "devices": ["cuda:0", "cuda:1", "cuda:2"],
        "assignments": assignments,
        "settings": {
            "configs": ",".join(configs),
            "match_actual_transmissions": True,
            "match_fixed_keyframes": False,
        },
    }
    (root / "parallel_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    statuses = [
        {"worker_id": item["worker_id"], "device": item["device"], "returncode": 0}
        for item in assignments
    ]
    (root / "parallel_worker_status.json").write_text(json.dumps(statuses), encoding="utf-8")
    for item in assignments:
        worker_root = root / "workers" / item["worker_id"]
        worker_root.mkdir(parents=True)
        (worker_root / "run_manifest.json").write_text(json.dumps({
            "extra": {
                "logical_device": "cuda:0",
                "physical_cuda_device": item["device"],
                "run_signature": {"physical_cuda_device": item["device"]},
            },
        }), encoding="utf-8")

    per_video = []
    for video in videos:
        for config in configs:
            per_video.append({
                "video": video, "config": config,
                "digital_step_policy": "fixed_reference", "fixed_reference_snr_db": 10,
                "n_nan_or_inf_frames": 0, "mean_psnr": 24, "mean_ssim": 0.8,
                "mean_lpips": 0.2,
            })
    _write_csv(root / "per_video_metrics.csv", per_video)
    _write_csv(root / "failed_pairs.csv", [])
    _write_csv(root / "matched_rate_plan.csv", [{
        "video": video, "mode": "actual_transmissions",
        "transmitting_frame_count_exact": True,
        "target_n_transmitting_frames": 5,
        "skem_planned_n_transmitting_frames": 5,
    } for video in videos])
    _write_csv(root / "rate_matching.csv", [{
        "video": video, "channel": channel,
        "matched_rate_mode": "actual_transmissions",
        "transmitting_frame_count_matched": True,
        "fixed_n_transmitting_frames": 5, "skem_n_transmitting_frames": 5,
        "byte_diff_ratio": 0.005, "raw_rate_matched": True,
        "fixed_padding_bytes": 50, "skem_padding_bytes": 0,
        "fixed_effective_total_bytes": 10050,
        "skem_effective_total_bytes": 10050,
        "effective_bytes_exact": True,
    } for video in videos for channel in channels])


def test_validator_passes_complete_three_gpu_exact_run(tmp_path):
    _valid_run(tmp_path)
    result = mod.validate(tmp_path)
    assert result["validation_passed"] is True
    assert result["counts"]["observed_video_config_pairs"] == 100
    assert result["counts"]["observed_rate_rows"] == 50
    mod.write_outputs(tmp_path, result)
    assert (tmp_path / "matched_rate_validation.json").is_file()
    assert (tmp_path / "matched_rate_quality_effect.csv").is_file()
    assert "**PASS**" in (tmp_path / "MATCHED_RATE_REPORT.md").read_text(encoding="utf-8")


def test_validator_fails_raw_byte_or_worker_provenance_violation(tmp_path):
    _valid_run(tmp_path)
    rates = mod._rows(tmp_path / "rate_matching.csv")
    rates[0]["byte_diff_ratio"] = "0.02"
    rates[0]["raw_rate_matched"] = "False"
    _write_csv(tmp_path / "rate_matching.csv", rates)
    workers = json.loads((tmp_path / "parallel_worker_status.json").read_text())
    workers[0]["returncode"] = 1
    (tmp_path / "parallel_worker_status.json").write_text(json.dumps(workers))
    result = mod.validate(tmp_path)
    assert result["validation_passed"] is False
    assert "raw_byte_difference_within_1pct" in result["failed_checks"]
    assert "worker_status_complete" in result["failed_checks"]
