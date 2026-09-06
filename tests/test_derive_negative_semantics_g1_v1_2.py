"""Regression tests for scripts/derive_negative_semantics_g1_v1_2.py.

Builds a small synthetic v1.1-shaped run_root + dataset (no real GPU data,
no dependency on outputs/negative_semantics_g1_pilot_rtx4080_v1_1 being
present) to exercise the provenance-integrity checks, the hash-vs-score
seed-evidence cross-validation, and the output-root resume/signature
protection end to end. `PROTOCOL_PATH`/`G1_DATASET` are monkeypatched on the
loaded module object -- the same technique other tests in this repo use to
load a script as a module and drive it in isolation.

`frame_tree_sha256` only reads raw bytes, so fixture "frames" are plain text
files with a `.png` extension -- no PIL/real image data needed to exercise
the hashing/cross-validation logic here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


PROTOCOL_YAML = {
    "metrics": {
        "statistics": {"bootstrap_iterations": 50},
        "ghost_survival_auc": {"horizon_annotated_timesteps": 4},
    },
    "gate": {
        "primary_policy": "few10",
        "h_add_min": 0.0,
        "additional_events_min": 0,
        "affected_videos_min": 0,
        "affected_seeds_min": 2,
        "max_single_video_event_share": 1.0,
        "max_single_seed_event_share": 0.75,
    },
    "reconstruction": {"base_config": "fixed_int4", "guide_profile": "candidate_both_omit"},
}

VIDEO_IDS = ["ovis_valid_fake1", "ovis_valid_fake2"]
FRAME_NAMES = ["img_0000001.jpg", "img_0000002.jpg"]
POLICIES = ["few10", "full50"]
SEEDS = [2025, 2026, 2027]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _build_fixture(tmp_path: Path, *, seed_frames_identical: bool = True):
    """Build a minimal but internally-consistent v1.1-shaped run_root +
    dataset. Returns (module, run_root, output_root) with PROTOCOL_PATH and
    G1_DATASET already monkeypatched onto the loaded module.
    """
    module = _load_script("derive_negative_semantics_g1_v1_2.py")

    dataset_root = tmp_path / "dataset"
    protocol_path = tmp_path / "g1_protocol.yaml"
    run_root = tmp_path / "run_root"
    output_root = tmp_path / "output"

    protocol_path.write_text(yaml.safe_dump(PROTOCOL_YAML), encoding="utf-8")
    protocol_sha256 = _sha256_file(protocol_path)

    gt_index = {
        "videos": {
            video_id: {
                "frame_names": FRAME_NAMES,
                "present_concepts": [[], []],
            }
            for video_id in VIDEO_IDS
        },
        "events": [],
    }
    gt_index_path = dataset_root / "ground_truth_index.json"
    _write_json(gt_index_path, gt_index)
    gt_index_sha256 = _sha256_file(gt_index_path)

    manifest = {"g1_ground_truth_index_sha256": gt_index_sha256}
    manifest_path = dataset_root / "preparation_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256_file(manifest_path)

    run_spec = {
        "smoke": False,
        "protocol_sha256": protocol_sha256,
        "preparation_manifest_sha256": manifest_sha256,
        "git_commit": "deadbeef",
    }
    _write_json(run_root / "run_spec.json", run_spec)

    evaluator_freeze = {
        "status": "PASSED", "selected_threshold": 0.5, "protocol_sha256": protocol_sha256,
    }
    _write_json(run_root / "evaluator" / "evaluator_freeze.json", evaluator_freeze)

    # Source-only OWLv2 cache: everything scores 0.0 (always source-negative).
    for video_id in VIDEO_IDS:
        _write_json(run_root / "evaluator" / "source" / f"{video_id}.json", {
            "scores": [
                {"frame_index": index, "scores": {"dog": 0.0}}
                for index in range(len(FRAME_NAMES))
            ],
        })

    # Reconstruction frames: identical bytes across seeds unless told otherwise.
    for policy in POLICIES:
        for seed in SEEDS:
            for video_id in VIDEO_IDS:
                recon_dir = (
                    run_root / "reconstruction" / policy / f"seed_{seed}" /
                    "recon_videos" / video_id / "fixed_int4__candidate_both_omit"
                )
                recon_dir.mkdir(parents=True, exist_ok=True)
                for index in range(len(FRAME_NAMES)):
                    content = f"{policy}-{video_id}-{index}"
                    if not seed_frames_identical and seed == SEEDS[-1]:
                        content += "-DIFFERENT"
                    (recon_dir / f"frame_{index:05d}.png").write_text(content, encoding="utf-8")

    # Detection rows: scores identical across seeds (matches the identical
    # frame bytes above) so score-based and hash-based collapse should agree
    # when seed_frames_identical=True.
    detection_rows = []
    for policy in POLICIES:
        for seed in SEEDS:
            for video_id in VIDEO_IDS:
                for index in range(len(FRAME_NAMES)):
                    detection_rows.append({
                        "video_id": video_id, "seed": seed, "policy": policy,
                        "frame_index": index, "concept_id": "dog",
                        "score": 0.9, "gt_present": False,
                    })
    detection_path = run_root / "evaluator" / "detection_rows.jsonl"
    detection_path.parent.mkdir(parents=True, exist_ok=True)
    with detection_path.open("w", encoding="utf-8") as handle:
        for row in detection_rows:
            handle.write(json.dumps(row) + "\n")

    module.PROTOCOL_PATH = protocol_path
    module.G1_DATASET = dataset_root
    return module, run_root, output_root


def test_derive_succeeds_with_dual_verified_evidence_on_consistent_fixture(tmp_path):
    module, run_root, output_root = _build_fixture(tmp_path)
    report = module.derive(run_root, output_root, declared_deterministic_decoder=False)
    assert report["seed_evidence_method"] == "score_and_hash_dual_verified"
    assert report["seed_groups"]["few10"]["effective_seed_count"] == 1
    assert (output_root / "g1_summary_v1_2.json").is_file()
    assert (output_root / "g1_policy_summary_v1_2.csv").is_file()


def test_derive_rejects_protocol_hash_mismatch(tmp_path):
    module, run_root, output_root = _build_fixture(tmp_path)
    # Mutate the "current" protocol file after run_spec.json was written.
    module.PROTOCOL_PATH.write_text(
        module.PROTOCOL_PATH.read_text(encoding="utf-8") + "\n# mutated\n", encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="g1_protocol.yaml has changed"):
        module.derive(run_root, output_root, declared_deterministic_decoder=False)


def test_derive_rejects_dataset_manifest_hash_mismatch(tmp_path):
    module, run_root, output_root = _build_fixture(tmp_path)
    manifest_path = module.G1_DATASET / "preparation_manifest.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace("}", ', "extra": true}'), encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="preparation_manifest.json has changed"):
        module.derive(run_root, output_root, declared_deterministic_decoder=False)


def test_derive_rejects_ground_truth_index_hash_mismatch(tmp_path):
    module, run_root, output_root = _build_fixture(tmp_path)
    gt_index_path = module.G1_DATASET / "ground_truth_index.json"
    data = json.loads(gt_index_path.read_text(encoding="utf-8"))
    data["events"] = [{"extra": "tampered"}]
    gt_index_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match the hash recorded"):
        module.derive(run_root, output_root, declared_deterministic_decoder=False)


def test_derive_rejects_evaluator_freeze_protocol_drift(tmp_path):
    module, run_root, output_root = _build_fixture(tmp_path)
    freeze_path = run_root / "evaluator" / "evaluator_freeze.json"
    data = json.loads(freeze_path.read_text(encoding="utf-8"))
    data["protocol_sha256"] = "not_the_real_hash"
    freeze_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit, match="drifted apart"):
        module.derive(run_root, output_root, declared_deterministic_decoder=False)


def test_derive_defaults_to_not_declared_and_requires_hash_or_flag(tmp_path):
    """Without --declared-deterministic-decoder, a genuine score/hash
    disagreement must fail closed rather than silently collapsing seeds.
    """
    module, run_root, output_root = _build_fixture(tmp_path, seed_frames_identical=False)
    with pytest.raises(ValueError, match="disagrees with independent"):
        module.derive(run_root, output_root, declared_deterministic_decoder=False)


def test_derive_operator_override_skips_hash_verification(tmp_path):
    """--declared-deterministic-decoder trusts score-only evidence even when
    the (unhashed) frames actually differ -- an explicit, visible trade-off,
    not the default.
    """
    module, run_root, output_root = _build_fixture(tmp_path, seed_frames_identical=False)
    report = module.derive(run_root, output_root, declared_deterministic_decoder=True)
    assert report["seed_evidence_method"] == "operator_declared"
    assert report["reconstruction_frame_hash_manifest"] is None


def test_derive_resume_is_idempotent_on_matching_signature(tmp_path):
    module, run_root, output_root = _build_fixture(tmp_path)
    first = module.derive(run_root, output_root, declared_deterministic_decoder=False)
    second = module.derive(run_root, output_root, declared_deterministic_decoder=False)
    assert first["derivation_signature"] == second["derivation_signature"]


def test_derive_rejects_output_root_signature_conflict_without_allow_overwrite(tmp_path):
    module, run_root, output_root = _build_fixture(tmp_path)
    module.derive(run_root, output_root, declared_deterministic_decoder=False)
    with pytest.raises(SystemExit, match="different\\s+derivation_signature|derivation_signature"):
        module.derive(run_root, output_root, declared_deterministic_decoder=True)


def test_derive_allows_output_root_overwrite_when_explicitly_requested(tmp_path):
    module, run_root, output_root = _build_fixture(tmp_path)
    module.derive(run_root, output_root, declared_deterministic_decoder=False)
    report = module.derive(
        run_root, output_root, declared_deterministic_decoder=True, allow_overwrite=True,
    )
    assert report["seed_evidence_method"] == "operator_declared"


def test_derive_rejects_writing_output_root_inside_run_root(tmp_path):
    module, run_root, _output_root = _build_fixture(tmp_path)
    with pytest.raises(SystemExit, match="refusing to write"):
        module.derive(run_root, run_root / "nested_output", declared_deterministic_decoder=False)


def test_derive_csv_output_uses_lf_only(tmp_path):
    module, run_root, output_root = _build_fixture(tmp_path)
    module.derive(run_root, output_root, declared_deterministic_decoder=False)
    raw = (output_root / "g1_policy_summary_v1_2.csv").read_bytes()
    assert b"\r\n" not in raw
    assert b"\n" in raw


@pytest.mark.skipif(
    not (ROOT / "outputs/negative_semantics_g1_pilot_rtx4080_v1_1").is_dir(),
    reason="frozen G1 v1.1 run_root is not present on this machine",
)
def test_derive_against_real_v1_1_run_is_dual_hash_verified(tmp_path):
    module = _load_script("derive_negative_semantics_g1_v1_2.py")
    report = module.derive(
        ROOT / "outputs/negative_semantics_g1_pilot_rtx4080_v1_1",
        tmp_path / "derived",
        declared_deterministic_decoder=False,
    )
    assert report["seed_evidence_method"] == "score_and_hash_dual_verified"
    assert report["gate_reapplication"]["effective_seed_status"] == "NOT_PASSED"
    assert report["seed_groups"]["few10"]["effective_seed_count"] == 1
    assert report["seed_groups"]["full50"]["effective_seed_count"] == 1
