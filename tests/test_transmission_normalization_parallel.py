"""CPU-only tests for the safe multi-GPU transmission orchestrator."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_txnorm_parallel_test_module",
        _ROOT / "scripts" / "run_transmission_normalization_parallel.py",
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


def _per_video(video: str, config: str, channel: str, byte_count: int, psnr: float):
    selector = config.split("_", 1)[0]
    return {
        "video": video, "config": config, "selector": selector, "channel": channel,
        "bit_depth": 16 if channel == "int16" else 8,
        "psss_backend_kind": "", "digital_step_policy": "fixed_reference",
        "ablation_label": "", "n_frames_total": 10, "n_transmitting_frames": 2,
        "n_keyframes_selected": 2, "n_nan_or_inf_frames": 0,
        "fixed_selector_kind": "fixed_count", "fixed_count_target": 2,
        "fixed_max_gop_used": "", "keyframe_count_matched": True,
        "nonfinite_stages": "", "n_quality_frames": 10, "valid_frame_ratio": 1.0,
        "mean_psnr": psnr, "mean_ssim": 0.8, "mean_lpips": 0.2,
        "latent_elements_total": 100, "analog_channel_symbols_total": "",
        "source_packet_bits_total": byte_count * 8,
        "digital_side_information_bytes_total": "", "total_bundle_bytes": byte_count,
        "total_bundle_bytes_per_frame": byte_count / 10,
        "analog_no_wire_bytes": False, "visual_transport_complete": True,
        "total_elapsed_s": 1.0,
    }


def test_assign_videos_balances_frame_load_deterministically():
    videos = [
        {"key": "a", "n_frames": 100}, {"key": "b", "n_frames": 90},
        {"key": "c", "n_frames": 80}, {"key": "d", "n_frames": 70},
        {"key": "e", "n_frames": 60}, {"key": "f", "n_frames": 50},
    ]
    workers = mod.assign_videos(videos, ["cuda:0", "cuda:1", "cuda:2"])
    assert sorted(v for worker in workers for v in worker["videos"]) == list("abcdef")
    loads = [worker["estimated_frames"] for worker in workers]
    assert max(loads) - min(loads) <= 30
    assert workers == mod.assign_videos(videos, ["cuda:0", "cuda:1", "cuda:2"])


def test_devices_must_be_unique_cuda_devices():
    with pytest.raises(ValueError):
        mod._validate_devices("cuda:0,cuda:0")
    with pytest.raises(ValueError):
        mod._validate_devices("cuda:0,cpu")


def test_worker_commands_have_independent_output_roots(tmp_path):
    args = mod._parse_args(["--output-root", str(tmp_path)])
    workers = [
        {"worker_id": "worker_00", "device": "cuda:0", "videos": ["a"]},
        {"worker_id": "worker_01", "device": "cuda:1", "videos": ["b"]},
    ]
    commands = [mod._worker_command(args, worker, tmp_path) for worker in workers]
    assert commands[0] != commands[1]
    assert str(tmp_path / "workers" / "worker_00") in commands[0]
    assert str(tmp_path / "workers" / "worker_01") in commands[1]
    assert commands[0][commands[0].index("--device") + 1] == "cuda:0"
    assert commands[1][commands[1].index("--device") + 1] == "cuda:1"


def test_resume_plan_mismatch_is_rejected(tmp_path):
    plan = {"commit": "a", "assignments": []}
    mod._check_or_write_plan(tmp_path, plan)
    mod._check_or_write_plan(tmp_path, dict(plan))
    with pytest.raises(RuntimeError, match="plan mismatch"):
        mod._check_or_write_plan(tmp_path, {"commit": "b", "assignments": []})


def test_merge_builds_global_aggregate_effects_and_manifest(tmp_path):
    plan = {
        "schema_version": 1,
        "git": {"commit": "a" * 40, "dirty": False, "branch": "main"},
        "dataset_root": "/dataset", "dataset_manifest_sha256": "b" * 64,
        "devices": ["cuda:0", "cuda:1"],
        "assignments": [
            {"worker_id": "worker_00", "device": "cuda:0", "videos": ["v1"], "estimated_frames": 10},
            {"worker_id": "worker_01", "device": "cuda:1", "videos": ["v2"], "estimated_frames": 10},
        ],
        "settings": {"seed": 2025, "digital_step_policy": "fixed_reference"},
    }
    mod._atomic_json(tmp_path / "parallel_plan.json", plan)
    statuses = []
    for index, video in enumerate(("v1", "v2")):
        root = tmp_path / "workers" / f"worker_{index:02d}"
        rows = [
            _per_video(video, "fixed_int16", "int16", 1000, 24.0),
            _per_video(video, "fixed_int8", "int8", 600, 23.8),
        ]
        _write_csv(root / "per_video_metrics.csv", rows)
        for filename in mod.MERGED_CSV_KEYS:
            if filename != "per_video_metrics.csv":
                _write_csv(root / filename, [])
        (root / "run_manifest.json").write_text("{}", encoding="utf-8")
        statuses.append({"worker_id": f"worker_{index:02d}", "returncode": 0})
    mod._atomic_json(tmp_path / "parallel_worker_status.json", statuses)

    summary = mod.merge_worker_outputs(tmp_path, plan, statuses, ["parallel.py"])
    assert summary["run_status"] == "completed"
    with (tmp_path / "aggregate.csv").open(newline="", encoding="utf-8") as handle:
        aggregate = list(csv.DictReader(handle))
    assert {row["config"] for row in aggregate} == {"fixed_int16", "fixed_int8"}
    assert all(row["all_expected_videos_present"] == "True" for row in aggregate)
    assert (tmp_path / "quantization_effect.csv").is_file()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["run_status"] == "completed"
    assert len(manifest["extra"]["worker_manifest_sha256"]) == 2
    assert "normalization_effect_summary.json" in manifest["extra"]["output_artifact_sha256"]
