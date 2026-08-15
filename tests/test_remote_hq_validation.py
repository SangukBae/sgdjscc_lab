from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality = _load("evaluate_video_frame_quality", _REPO / "scripts" / "evaluate_video_frame_quality.py")
hq = _load("run_remote_hq_validation", _REPO / "scripts" / "run_remote_hq_validation.py")


def _png(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 16), color).save(path)


def test_frame_quality_matches_numeric_indices_and_separates_kinds(tmp_path):
    run = tmp_path / "generation" / "wan" / "01_toy"
    _png(run / "extracted_frames" / "01_toy" / "frame_00000.png", (0, 0, 0))
    _png(run / "extracted_frames" / "01_toy" / "frame_00001.png", (10, 10, 10))
    _png(run / "recon_frames" / "recon_00000.png", (1, 1, 1))
    _png(run / "generated_frames" / "generated_00001.png", (20, 20, 20))

    rows, summary = quality.evaluate_root(
        tmp_path / "generation", kinds=["auto"], device="cpu",
        use_lpips=False, use_clip=False,
    )
    assert {(row["kind"], row["frame_index"]) for row in rows} == {
        ("recon", 0), ("generated", 1),
    }
    assert {row["kind"] for row in summary} == {"recon", "generated"}
    assert all(row["n_frames"] == 1 for row in summary)
    assert all(row["psnr_mean"] is not None for row in summary)


def test_frame_quality_resizes_candidate_and_records_it(tmp_path):
    run = tmp_path / "01_toy"
    _png(run / "extracted_frames" / "01_toy" / "frame_00000.png", (0, 0, 0))
    path = run / "recon_frames" / "recon_00000.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 8), (5, 5, 5)).save(path)
    rows = quality.evaluate_run(
        run, kinds=["recon"], device="cpu", use_lpips=False, use_clip=False,
    )
    assert rows[0]["resized_for_metric"] is True
    assert rows[0]["candidate_width"] == 16
    assert rows[0]["source_width"] == 32


def test_hq_plan_uses_real_models_full_resolution_and_three_gpu_wan(tmp_path):
    args = hq._parse_args(["--output-root", str(tmp_path)])
    commands = {item["name"]: item["cmd"] for item in hq.build_commands(args, tmp_path, list(hq._ALL_PHASES))}
    all_tokens = [token for cmd in commands.values() for token in cmd]
    assert "--no-models" not in all_tokens
    assert commands["video"][commands["video"].index("--parallel") + 1] == "3"
    assert commands["svd"][commands["svd"].index("--worker-num-inference-steps") + 1] == "25"
    assert commands["wan"][commands["wan"].index("--worker-num-inference-steps") + 1] == "30"
    assert commands["wan"][commands["wan"].index("--worker-height") + 1] == "256"
    assert commands["wan"][commands["wan"].index("--worker-width") + 1] == "512"
    assert "balanced" in commands["wan"]
    assert "recon" in commands["quality_video"]
    assert "auto" in commands["quality_generation"]


def test_hq_dry_run_writes_plan_without_environment_preflight(tmp_path):
    rc = hq.main([
        "--dry-run", "--phases", "wan", "--output-root", str(tmp_path),
    ])
    assert rc == 0
    plan = json.loads((tmp_path / "hq_validation_plan.json").read_text())
    assert [row["name"] for row in plan] == ["wan"]
    assert not (tmp_path / "preflight.json").exists()
