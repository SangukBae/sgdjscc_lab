from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from sgdjscc_lab.evaluators.negative_semantics_g2 import summarize_g2
from sgdjscc_lab.guidance.negative_conditioning import (
    G2_ARMS,
    QUALITY_NEGATIVE_PROMPT,
    build_g2_condition_manifests,
    resolve_negative_prompts,
    validate_g2_condition_manifest,
)


ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_test_g2_{name}", ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_negative_prompt_is_byte_for_byte_legacy_value():
    assert resolve_negative_prompts(["one", "two"], OmegaConf.create({})) == [
        QUALITY_NEGATIVE_PROMPT, QUALITY_NEGATIVE_PROMPT,
    ]


def test_semantic_negative_append_broadcasts_and_replace_is_explicit():
    append = OmegaConf.create({
        "negative_conditioning": {"enabled": True, "mode": "append", "prompt": "a dog"}
    })
    assert resolve_negative_prompts(["p1", "p2"], append) == [
        f"{QUALITY_NEGATIVE_PROMPT}, a dog",
        f"{QUALITY_NEGATIVE_PROMPT}, a dog",
    ]
    replace = OmegaConf.create({
        "negative_conditioning": {
            "enabled": True, "mode": "replace", "prompt": ["x", "y"],
        }
    })
    assert resolve_negative_prompts(["p1", "p2"], replace) == ["x", "y"]
    with pytest.raises(ValueError, match="length must match"):
        resolve_negative_prompts(["p1", "p2"], OmegaConf.create({
            "negative_conditioning": {"enabled": True, "prompt": ["x"]}
        }))


def test_diffusion_standard_path_consumes_resolved_semantic_negative():
    import torch
    from unittest.mock import MagicMock
    from sgdjscc_lab.pipelines.infer_pipeline import _run_diffusion

    pipe = MagicMock()
    pipe.generate.return_value = (None, torch.zeros(1, 16, 16, 16))
    pipe.alphas_cumprod = torch.ones(1000)
    cfg = OmegaConf.create({
        "use_phase5": False,
        "negative_conditioning": {"enabled": True, "mode": "append", "prompt": "a dog"},
    })
    _run_diffusion(
        pipe=pipe, encode_features_hat=torch.zeros(1, 16, 16, 16),
        power_scalar=torch.ones(1, 1, 1, 1), semantic_text=["a street"],
        canny_latent=torch.zeros(1, 4, 32, 32), cur_step=10,
        cfg_method="constant", guidance_scale=4.0, ctrl_scale=1.0,
        not_control=[False], use_jscc_feat=True, use_controlnet=False,
        diffusion_step=10, step_style="linear", mask_token=None, cfg=cfg,
    )
    assert pipe.generate.call_args.kwargs["negative_prompt"] == [
        f"{QUALITY_NEGATIVE_PROMPT}, a dog"
    ]


def _gt_index():
    return {
        "videos": {
            "v1": {
                "frame_names": ["000.jpg", "001.jpg"],
                "present_concepts": [["dog"], []],
            },
            "v2": {
                "frame_names": ["000.jpg", "001.jpg"],
                "present_concepts": [[], ["cat"]],
            },
        }
    }


def test_g2_condition_builder_is_deterministic_cardinality_matched_and_oracle_safe():
    kwargs = {
        "concepts": ["cat", "dog", "bird"],
        "queries": {"cat": "a cat", "dog": "a dog", "bird": "a bird"},
        "frequency_ranking": ["dog", "cat", "bird"],
        "random_seed": 27091,
    }
    first = build_g2_condition_manifests(_gt_index(), ["v1", "v2"], **kwargs)
    second = build_g2_condition_manifests(_gt_index(), ["v1", "v2"], **kwargs)
    assert first == second
    expected = {"v1": 2, "v2": 2}
    for arm in G2_ARMS:
        validate_g2_condition_manifest(first[arm], expected)
    oracle = first["oracle_negative"]["videos"]["v1"]["frames"]
    assert oracle[0]["negative_concepts"] == ["cat", "bird"]
    assert oracle[1]["negative_concepts"] == ["cat", "dog", "bird"]
    for video_id in expected:
        oracle_frames = first["oracle_negative"]["videos"][video_id]["frames"]
        for arm in ("random_negative", "frequency_negative"):
            control_frames = first[arm]["videos"][video_id]["frames"]
            assert [len(row["negative_concepts"]) for row in control_frames] == [
                len(row["negative_concepts"]) for row in oracle_frames
            ]
    assert first["oracle_negative"]["oracle_eval_only"] is True
    assert first["oracle_negative"]["serialized_in_packet"] is False
    assert first["oracle_negative"]["rate_accounted"] is False


def test_condition_validator_rejects_rate_accounting_and_incomplete_grid():
    manifest = build_g2_condition_manifests(
        _gt_index(), ["v1"], concepts=["cat", "dog"],
        queries={"cat": "a cat", "dog": "a dog"},
        frequency_ranking=["dog", "cat"], random_seed=1,
    )["oracle_negative"]
    manifest["rate_accounted"] = True
    with pytest.raises(ValueError, match="outside packet/rate accounting"):
        validate_g2_condition_manifest(manifest, {"v1": 2})
    manifest["rate_accounted"] = False
    manifest["videos"]["v1"]["frames"].pop()
    with pytest.raises(ValueError, match="frame count mismatch"):
        validate_g2_condition_manifest(manifest, {"v1": 2})


def test_transmission_runner_loads_manifest_and_injects_receiver_only_cfg(tmp_path):
    module = _load_script("run_transmission_reduction_eval.py")
    manifest = build_g2_condition_manifests(
        _gt_index(), ["v1"], concepts=["cat", "dog"],
        queries={"cat": "a cat", "dog": "a dog"},
        frequency_ranking=["dog", "cat"], random_seed=1,
    )["oracle_negative"]
    path = tmp_path / "condition.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded, rows = module._load_negative_condition_plan(
        path, [{"key": "v1", "row": {"n_frames": "2"}}], max_frames=1,
    )
    assert loaded["arm"] == "oracle_negative"
    assert len(rows["v1"]) == 1
    cfg = module._with_negative_condition(
        OmegaConf.create({"diffusion_step": 10}), rows["v1"][0], "oracle_negative"
    )
    assert cfg.negative_conditioning.oracle_eval_only is True
    assert cfg.negative_conditioning.serialized_in_packet is False
    assert cfg.negative_conditioning.rate_accounted is False
    assert "a cat" in cfg.negative_conditioning.prompt


def _g2_metric_fixture():
    detection = []
    accounting = []
    for arm in G2_ARMS:
        for video in ("v1", "v2"):
            # Absent opportunity: no_negative hallucinates, only oracle removes.
            detection.append({
                "video_id": video, "arm": arm, "policy": "few10", "seed": 2025,
                "frame_index": 0, "concept_id": "dog", "gt_present": False,
                "source_score": 0.1, "score": 0.1 if arm == "oracle_negative" else 0.9,
            })
            # Present opportunity stays detected in every arm (no false suppression).
            detection.append({
                "video_id": video, "arm": arm, "policy": "few10", "seed": 2025,
                "frame_index": 1, "concept_id": "dog", "gt_present": True,
                "source_score": 0.9, "score": 0.9,
            })
            accounting.append({
                "video_id": video, "arm": arm, "policy": "few10", "seed": 2025,
                "mean_psnr": 25.0, "mean_ssim": 0.8, "mean_lpips": 0.2,
            })
    return detection, accounting


def test_g2_summary_passes_clear_oracle_control_with_no_quality_or_safety_cost():
    detection, accounting = _g2_metric_fixture()
    report = summarize_g2(
        detection, accounting, threshold=0.5, arms=list(G2_ARMS), policies=["few10"],
        primary_policy="few10", bootstrap_iterations=100,
        gate={
            "h_add_relative_reduction_min": 0.25,
            "false_suppression_increase_ci_upper_max": 0.02,
            "psnr_drop_db_max": 0.5, "ssim_drop_max": 0.01, "lpips_rise_max": 0.02,
        },
    )
    oracle = report["comparisons_vs_no_negative"]["few10"]["oracle_negative"]
    assert oracle["source_paired_h_add_relative_reduction"] == 1.0
    assert oracle["false_suppression_vs_no_negative"]["rate"] == 0.0
    assert report["provisional_gate_status"] == "PASSED"


def test_g2_summary_fails_closed_on_arm_grid_mismatch():
    detection, accounting = _g2_metric_fixture()
    detection.pop()
    with pytest.raises(ValueError, match="grid differs"):
        summarize_g2(
            detection, accounting, threshold=0.5, arms=list(G2_ARMS), policies=["few10"],
            primary_policy="few10", bootstrap_iterations=10,
            gate={
                "h_add_relative_reduction_min": 0.25,
                "false_suppression_increase_ci_upper_max": 0.02,
                "psnr_drop_db_max": 0.5, "ssim_drop_max": 0.01, "lpips_rise_max": 0.02,
            },
        )


def test_g2_protocol_freezes_oracle_boundary_and_fixed_int4():
    protocol = OmegaConf.to_container(OmegaConf.load(
        ROOT / "configs/experiments/negative_semantics/g2_oracle_absent_protocol.yaml"
    ), resolve=True)
    assert protocol["freeze_status"] == "frozen"
    assert protocol["scope"]["heldout_access"] == "prohibited"
    assert protocol["reconstruction"]["base_config"] == "fixed_int4"
    assert protocol["negative_conditioning"]["arms"] == list(G2_ARMS)
    assert protocol["negative_conditioning"]["oracle_eval_only"] is True
    assert protocol["negative_conditioning"]["serialized_in_packet"] is False
    assert protocol["negative_conditioning"]["rate_accounted"] is False


def test_g2_matched_compute_audit_uses_real_child_csv_schema(tmp_path):
    module = _load_script("run_negative_semantics_g2.py")
    fields = [
        "video", "config", "channel", "bit_depth", "digital_step_policy",
        "fixed_reference_snr_db", "decoder_mode", "diffusion_step",
        "effective_diffusion_step", "n_frames_total", "n_transmitting_frames",
        "n_keyframes_selected", "latent_elements_total", "source_packet_bits_total",
        "total_bundle_bytes", "negative_condition_arm", "mean_psnr", "mean_ssim",
        "mean_lpips",
    ]
    selection_fields = [
        "video", "frame_index", "decision", "visual_transmitted", "source_packet_bits",
    ]
    for arm in G2_ARMS:
        child = tmp_path / "reconstruction" / arm / "few10" / "seed_2025"
        child.mkdir(parents=True)
        with (child / "per_video_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for video in ("v1", "v2"):
                writer.writerow({
                    "video": video, "config": "fixed_int4__candidate_both_omit",
                    "channel": "int4", "bit_depth": 4,
                    "digital_step_policy": "fixed_reference", "fixed_reference_snr_db": 10,
                    "decoder_mode": "diffusion", "diffusion_step": 10,
                    "effective_diffusion_step": 10, "n_frames_total": 2,
                    "n_transmitting_frames": 1, "n_keyframes_selected": 1,
                    "latent_elements_total": 100, "source_packet_bits_total": 800,
                    "total_bundle_bytes": 100, "negative_condition_arm": arm,
                    "mean_psnr": 25, "mean_ssim": 0.8, "mean_lpips": 0.2,
                })
        with (child / "keyframe_selection.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=selection_fields)
            writer.writeheader()
            for video in ("v1", "v2"):
                writer.writerows([
                    {"video": video, "frame_index": 0, "decision": "keyframe",
                     "visual_transmitted": True, "source_packet_bits": 700},
                    {"video": video, "frame_index": 1, "decision": "reuse",
                     "visual_transmitted": False, "source_packet_bits": 100},
                ])
    accounting, report = module._collect_accounting_and_verify_matching(
        {}, ["v1", "v2"], tmp_path, {"few10": 10}, [2025], smoke=False,
    )
    assert len(accounting) == 8
    assert report["status"] == "PASSED"
    assert report["mismatch_count"] == 0
