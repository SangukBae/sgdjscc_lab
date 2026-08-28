from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load_summarizer():
    spec = importlib.util.spec_from_file_location(
        "_integrated_summary_test", ROOT / "scripts/summarize_integrated_semantic_validation.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_integrated_summary_requires_complete_grid_and_selects_smallest_passing_candidate(tmp_path):
    mod = load_summarizer()
    videos = ["v1", "v2"]
    assignments = [
        {"worker_id": "worker_00", "device": "cuda:0", "videos": ["v1"]},
        {"worker_id": "worker_01", "device": "cuda:1", "videos": ["v2"]},
    ]
    (tmp_path / "integrated_plan.json").write_text(
        json.dumps({"assignments": assignments}), encoding="utf-8"
    )
    bytes_by_profile = {
        "baseline": 1000, "combined_ds4": 400,
        "candidate_edge_ds4_uncertainty_omit": 300, "candidate_both_omit": 200,
    }
    for assignment in assignments:
        video = assignment["videos"][0]
        rows = []
        for policy in mod.POLICIES:
            for profile in mod.PROFILES:
                rows.append({
                    "video": video, "decoder_policy": policy, "guide_profile": profile,
                    "n_frames": 100, "mean_psnr": 30, "mean_ssim": 0.9,
                    "mean_lpips": 0.1, "total_bundle_bytes": bytes_by_profile[profile],
                    "total_elapsed_s": {"full50": 30, "few10": 10, "vae_direct": 1}[policy],
                    "closed_n_items": 100, "closed_mean_severity": 0.1,
                    "closed_ptc": 0.9, "closed_sfr": 0.01, "closed_sdi": 0.0,
                    "open_temporal_hallucination_rate": 0.01,
                    "open_total_additional_objects": 1,
                    "closed_backend_clip": 2, "closed_backend_owlv2": 2,
                    "closed_backend_vqa": 2, "open_backend_clip": 2,
                    "open_backend_owlv2": 2, "open_backend_vqa": 2,
                })
        worker_root = tmp_path / "semantic" / "workers" / assignment["worker_id"]
        write_csv(worker_root / "integrated_semantic_rows.csv", rows)
        (worker_root / "worker_summary.json").write_text(json.dumps({
            "status": "completed", "worker_id": assignment["worker_id"], "n_pairs": len(rows),
        }), encoding="utf-8")

    validation = mod.summarize(tmp_path)
    assert validation["validation_passed"] is True
    assert validation["n_completed_pairs"] == 24
    selected = validation["selected_operating_point"]
    assert selected["decoder_policy"] == "vae_direct"
    assert selected["guide_profile"] == "candidate_both_omit"
    assert selected["mean_total_bundle_bytes"] == 200
    assert (tmp_path / "artifact_sha256.json").is_file()
