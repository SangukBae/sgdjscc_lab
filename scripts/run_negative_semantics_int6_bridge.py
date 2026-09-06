#!/usr/bin/env python3
"""Paired fixed_int4 vs fixed_int6 bridge under the frozen G1 conditions.

Frozen protocol: ``configs/experiments/negative_semantics/g1_int6_bridge_protocol.yaml``.
Answers one question without re-litigating the G1 phenomenon gate: does the
2026-09-04 ETRI operating-point change (``fixed_int4`` -> ``fixed_int6``)
change exact bundle bytes, PSNR/SSIM/LPIPS, latency, or additional-object /
ghost-track prevalence, under the identical OVIS Pilot / ``candidate_both_omit``
guide / ``few10``+``full50`` decoder / single effective seed used by G1 v1.1?

``fixed_int4`` is never recomputed: its reconstruction frames and detector
scores are read directly from the frozen
``outputs/negative_semantics_g1_pilot_rtx4080_v1_1`` run. Only ``fixed_int6``
reconstruction + OWLv2 scoring is new GPU work, and it reuses G1's frozen
evaluator threshold (never recalibrated per bit depth) and G1's already
-generated captions (bit-depth independent, read from the shared
``data/negative_semantics/g1/pilot/captions`` dataset directory).

Every fixed_int6-vs-fixed_int4 comparison is computed as an inner-joined,
paired-by-video difference with a paired video-clustered bootstrap 95% CI
(see ``src/sgdjscc_lab/evaluators/int6_bridge.py``), covering exact bytes,
PSNR/SSIM/LPIPS, latency, raw/source-paired H_add, and ghost-track AUC. LPIPS
is computed post-hoc by this script for *both* conditions with one shared
evaluator instance against the *canonical* (resized/no-upscale/center-padded)
source frame -- the exact transform
``run_transmission_reduction_eval.py::_load_frames`` applies before
reconstruction, reused here rather than reimplemented, so LPIPS never
compares a raw differently-sized source image against a reconstruction --
rather than relying on that script's own ``--no-lpips``-gated column, which
is blank for the reused v1.1 fixed_int4 rows.

Formal runs additionally verify that every critical file this run depends on
(this script, its imported support modules, the frozen protocol/config) is
git-tracked, uncommitted-change-free, and byte-identical to what HEAD
records -- a plain "is the tracked tree dirty" check would miss an untracked
new file entirely, silently letting `git_commit` be recorded as if it
reproduced code that commit does not actually contain.

Importing this module transitively imports torch (``sgdjscc_lab.evaluators``'s
package ``__init__`` eagerly imports ``quality.py``, which imports torch --
true of the original ``run_negative_semantics_g1.py`` as well), but that
alone never touches a GPU: no CUDA call, device allocation, or
reconstruction happens at module scope. Every GPU-touching step (CUDA
availability checks, the OWLv2 scorer, LPIPS, subprocess reconstruction)
lives inside a function invoked only from ``run()``/``_execute()``, so
loading this module or calling ``--help`` never requires a GPU to actually
be present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sgdjscc_lab.evaluators.negative_semantics import (  # noqa: E402
    h_add_by_video,
    raw_and_paired_h_add,
    summarize_g1,
)
from sgdjscc_lab.evaluators.int6_bridge import (  # noqa: E402
    paired_condition_metrics,
    paired_delta_from_video_values,
    paired_delta_video_clustered,
    quality_gate_pass,
    summarize_condition_metrics,
    validate_exact_video_count,
    validate_gt_present_consistency,
    validate_unique_keys,
)
from sgdjscc_lab.utils.frame_hash import frame_tree_sha256  # noqa: E402


PROTOCOL_PATH = ROOT / "configs/experiments/negative_semantics/g1_int6_bridge_protocol.yaml"
G1_SCRIPT = ROOT / "scripts/run_negative_semantics_g1.py"
TRANSMISSION_REDUCTION_EVAL_SCRIPT = ROOT / "scripts/run_transmission_reduction_eval.py"
G1_DATASET = ROOT / "data/negative_semantics/g1/pilot"

_ACCOUNTING_FIELDS = [
    "total_bundle_bytes", "total_bundle_bytes_per_frame",
    "mean_psnr", "mean_ssim", "mean_lpips", "total_elapsed_s",
]
OMITTED_ASYMMETRIC_SMOKE = "OMITTED_ASYMMETRIC_SMOKE"
OMITTED_INSUFFICIENT_HORIZON_SMOKE = "OMITTED_INSUFFICIENT_HORIZON_SMOKE"

# Files this run directly depends on for a formal reconstruction+scoring
# pass. Every one of these must be git-tracked, unmodified, and identical to
# HEAD before a formal run (or a preflight check standing in for one) is
# allowed to proceed -- see `verify_critical_files_tracked_at_head`.
CRITICAL_RELATIVE_PATHS = [
    "scripts/run_negative_semantics_int6_bridge.py",
    "scripts/run_negative_semantics_g1.py",
    "scripts/run_transmission_reduction_eval.py",
    "configs/experiments/negative_semantics/g1_int6_bridge_protocol.yaml",
    "configs/experiments/negative_semantics/g1_reconstruction.yaml",
    "src/sgdjscc_lab/evaluators/negative_semantics.py",
    "src/sgdjscc_lab/evaluators/int6_bridge.py",
    "src/sgdjscc_lab/evaluators/quality.py",
    "src/sgdjscc_lab/utils/frame_hash.py",
    "src/sgdjscc_lab/io.py",
    "src/sgdjscc_lab/paths.py",
]

LPIPS_PREPROCESSING_DESCRIPTION = (
    "source: run_transmission_reduction_eval.py::_load_frames canonical transform "
    "(aspect_preserving_bilinear_antialias_no_upscale resize to image_long_side, then "
    "symmetric_constant_zero_extra_pixel_bottom_right pad to image_pad_multiple); "
    "reconstruction: sgdjscc_lab.io.load_image_as_tensor on the saved PNG, no further "
    "transform (already at the canonical resolution); both as [1,3,H,W] float32 in "
    "[0,1], compared via QualityEvaluator.evaluate (internally rescaled to [-1,1] "
    "before the LPIPS backbone)."
)


def _load_g1_support():
    """Import run_negative_semantics_g1.py by path to reuse its frozen,
    already-tested helpers (hashing, atomic writers, OWLv2 scorer, resize/pad,
    HF snapshot resolution) instead of duplicating them. This mirrors the
    pattern tests/test_negative_semantics_g1.py already uses to load scripts
    as modules.
    """
    spec = importlib.util.spec_from_file_location("_negative_semantics_g1_support", G1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_transmission_reduction_eval_support():
    """Import run_transmission_reduction_eval.py by path to reuse its
    canonical frame resize/no-upscale/center-pad transform (``_load_frames``)
    for LPIPS's source side, instead of re-implementing that math and
    risking it drifting from what reconstruction actually targets.
    """
    spec = importlib.util.spec_from_file_location(
        "_transmission_reduction_eval_support", TRANSMISSION_REDUCTION_EVAL_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_protocol() -> Dict[str, Any]:
    from omegaconf import OmegaConf

    protocol = OmegaConf.to_container(OmegaConf.load(PROTOCOL_PATH), resolve=True)
    if protocol.get("freeze_status") != "frozen" or protocol["scope"]["heldout_access"] != "prohibited":
        raise SystemExit("invalid or unfrozen int6 bridge protocol")
    return protocol


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
    temporary.replace(path)


# --- provenance: git-tracked critical files, run_spec resume gate ---------

def verify_critical_files_tracked_at_head(
    repo_root: Path, relative_paths: Sequence[str],
) -> Dict[str, str]:
    """Fail closed unless every path in ``relative_paths`` is git-tracked,
    free of uncommitted changes, and byte-identical to ``HEAD``.

    A whole-repo "is the tracked tree dirty" check (``git diff --quiet``)
    says nothing about a file that was never ``git add``ed in the first
    place: it would report "clean" while a brand-new, untracked runner
    script silently does the actual work, making the recorded
    ``git_commit`` a false promise of reproducibility. This checks each
    critical path individually and only that path, so unrelated untracked
    files elsewhere in the working tree (a stray notebook, a draft doc) are
    never blocked.
    """
    problems: List[str] = []
    checked: Dict[str, str] = {}
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            problems.append(f"{relative}: file does not exist")
            continue
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative], cwd=repo_root,
            capture_output=True, text=True,
        ).returncode == 0
        if not tracked:
            problems.append(f"{relative}: not tracked by git")
            continue
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", relative], cwd=repo_root,
            capture_output=True, text=True,
        ).stdout.strip()
        if status:
            problems.append(f"{relative}: has uncommitted changes ({status})")
            continue
        head_result = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=repo_root, capture_output=True,
        )
        if head_result.returncode != 0:
            problems.append(f"{relative}: not present at HEAD")
            continue
        head_sha256 = hashlib.sha256(head_result.stdout).hexdigest()
        working_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if head_sha256 != working_sha256:
            problems.append(f"{relative}: working tree content differs from HEAD despite clean git status")
            continue
        checked[relative] = working_sha256
    if problems:
        raise SystemExit(
            "formal run refused: the following critical file(s) are not safely "
            "reproducible from the recorded git commit alone:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
    return checked


def check_run_spec_resume(
    existing_spec: Optional[Mapping[str, Any]], new_spec: Mapping[str, Any], *, run_root: Path,
) -> None:
    """Fail closed if a prior run_spec.json under this --run-root disagrees
    with what this invocation would record: mode, protocol/dataset/
    checkpoint/evaluator/LPIPS/critical-file hashes, git commit, device,
    seed, or video_ids. A silent resume across a changed protocol or
    checkpoint would let two reconstructions in the same run_root be
    provenance-inconsistent.
    """
    if existing_spec is not None and existing_spec != new_spec:
        raise SystemExit(
            f"resume run_spec mismatch at {run_root / 'run_spec.json'}; a prior run "
            "recorded different provenance (mode/protocol/dataset/checkpoint/evaluator/"
            "lpips/critical-files/git/device/seed/video_ids) for this --run-root. Choose "
            "another --run-root instead of resuming across incompatible settings."
        )


def verify_int4_reuse(g1, protocol: Mapping[str, Any], video_ids: Sequence[str], seed: int) -> Dict[str, Any]:
    """Confirm the v1.1 int4 reuse source is complete for this seed/video set
    without recomputing anything. Pure file-reading; no GPU/network access.
    """
    reuse_cfg = protocol["reuse"]
    source_run = (ROOT / reuse_cfg["int4_source_run"]).resolve()
    run_spec = g1._load_json(source_run / "run_spec.json")
    if run_spec.get("protocol_sha256") != reuse_cfg["int4_source_protocol_sha256"]:
        raise SystemExit(
            "int4 reuse source protocol_sha256 does not match the frozen bridge "
            "protocol's expectation; refusing an ambiguous reuse"
        )
    if run_spec.get("smoke") is not False:
        raise SystemExit("int4 reuse source is a smoke run, not formal evidence")
    if int(seed) not in [int(value) for value in run_spec.get("seeds", [])]:
        raise SystemExit(f"seed {seed} was not part of the int4 reuse source run")
    if not set(video_ids).issubset(set(run_spec.get("video_ids", []))):
        raise SystemExit("int4 reuse source does not cover all requested video_ids")
    policies = protocol["reconstruction"]["policies"]
    verified_policies = []
    for policy in policies:
        summary_path = source_run / "reconstruction" / policy / f"seed_{seed}" / "summary.json"
        summary = g1._load_json(summary_path)
        if summary.get("run_status") != "completed":
            raise SystemExit(f"int4 reuse source child run not completed: {policy}/seed_{seed}")
        if int(summary.get("n_videos", 0)) < len(video_ids):
            raise SystemExit(f"int4 reuse source child run covers fewer videos than requested: {policy}/seed_{seed}")
        if int(summary.get("n_failed_pairs", -1)) != 0:
            raise SystemExit(f"int4 reuse source child run has failed pairs: {policy}/seed_{seed}")
        verified_policies.append(policy)
    evaluator_freeze = g1._load_json(source_run / "evaluator" / "evaluator_freeze.json")
    if evaluator_freeze.get("status") != "PASSED":
        raise SystemExit("int4 reuse source evaluator calibration did not pass; cannot reuse its threshold")
    detection_rows_path = source_run / "evaluator" / "detection_rows.jsonl"
    return {
        "source_run": str(source_run),
        "seed": int(seed),
        "policies_verified": verified_policies,
        "threshold": evaluator_freeze["selected_threshold"],
        "evaluator_model": evaluator_freeze["evaluator_model"],
        "detection_rows_sha256": _sha256_file(detection_rows_path),
        "per_video_metrics_sha256": {
            policy: _sha256_file(source_run / "reconstruction" / policy / f"seed_{seed}" / "per_video_metrics.csv")
            for policy in policies
        },
    }


def load_int4_rows(
    g1, protocol: Mapping[str, Any], seed: int, video_ids: Sequence[str], max_frames: Optional[int],
) -> List[Dict[str, Any]]:
    """Filter the frozen v1.1 detection_rows.jsonl to this single seed, this
    run's video_ids, and (for smoke) this run's frame budget, relabeling
    `policy` as a bridge condition (`fixed_int4__<decoder>`), rather than
    rescoring already-scored reconstruction frames.

    Restricting to `video_ids`/`max_frames` matters for smoke: fixed_int6
    only reconstructs+scores one video for two frames there, so comparing it
    against fixed_int4's full 40-video/full-length rows would silently
    compare disjoint populations instead of the same 1 video x 2 frames.
    """
    video_id_set = set(video_ids)
    source_run = ROOT / protocol["reuse"]["int4_source_run"]
    rows = []
    with (source_run / "evaluator" / "detection_rows.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row["seed"]) != int(seed):
                continue
            if row["video_id"] not in video_id_set:
                continue
            if max_frames is not None and int(row["frame_index"]) >= max_frames:
                continue
            row = dict(row)
            row["condition"] = f"fixed_int4__{row['policy']}"
            rows.append(row)
    return rows


def load_int4_accounting(protocol: Mapping[str, Any], seed: int, video_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """Per-video accounting rows are not frame-indexed, so (unlike
    `load_int4_rows`) this can only restrict by `video_ids`, not by
    `max_frames`. During smoke this means the fixed_int4 side still reflects
    the video's full length while fixed_int6 covers only its first frames --
    exactly why `build_summary` never computes a paired accounting/quality
    delta for smoke (see `OMITTED_ASYMMETRIC_SMOKE`).
    """
    video_id_set = set(video_ids)
    source_run = ROOT / protocol["reuse"]["int4_source_run"]
    guide_profile = protocol["reconstruction"]["guide_profile"]
    # run_transmission_reduction_eval.py writes the combined
    # "{base_config}__{guide_profile}" label into per_video_metrics.csv's
    # `config` column (see its `config_name` construction), not the bare
    # bit-depth name.
    expected_config = f"fixed_int4__{guide_profile}"
    rows = []
    for policy in protocol["reconstruction"]["policies"]:
        csv_path = source_run / "reconstruction" / policy / f"seed_{seed}" / "per_video_metrics.csv"
        for row in _read_csv_rows(csv_path):
            if row.get("config") != expected_config or row.get("video") not in video_id_set:
                continue
            row = dict(row)
            row["condition"] = f"fixed_int4__{policy}"
            rows.append(row)
    return rows


def load_source_rows(
    g1, reuse_info: Mapping[str, Any], gt_index: Mapping[str, Any],
    video_ids: Sequence[str], max_frames: Optional[int],
) -> List[Dict[str, Any]]:
    """Reuse v1.1's cached source-only OWLv2 scores (bit-depth independent:
    they score the original OVIS frames, never a reconstruction), restricted
    to this run's video_ids/max_frames for the same reason as `load_int4_rows`.
    """
    source_run = Path(reuse_info["source_run"])
    rows = []
    for video_id in video_ids:
        cached = g1._load_json(source_run / "evaluator" / "source" / f"{video_id}.json")
        for item in cached["scores"]:
            if max_frames is not None and int(item["frame_index"]) >= max_frames:
                continue
            for concept, score in item["scores"].items():
                rows.append({
                    "video_id": video_id, "frame_index": item["frame_index"],
                    "concept_id": concept, "score": score,
                })
    return rows


def _run_int6_reconstruction(
    protocol: Mapping[str, Any], video_ids: Sequence[str], run_root: Path,
    *, policy: str, steps: int, seed: int, device: str, max_frames: Optional[int],
) -> None:
    recon = protocol["reconstruction"]
    spatial = recon["spatial_transform"]
    output = run_root / "reconstruction" / "fixed_int6" / policy / f"seed_{seed}"
    command = [
        sys.executable, str(TRANSMISSION_REDUCTION_EVAL_SCRIPT),
        "--dataset-root", str(G1_DATASET),
        "--config", str(ROOT / recon["config"]),
        "--output-root", str(output),
        "--video-ids", ",".join(video_ids),
        "--configs", "fixed_int6",
        "--guide-profiles", recon["guide_profile"],
        "--decoder-mode", "diffusion",
        "--diffusion-step", str(steps),
        "--device", device,
        "--seed", str(seed),
        "--snr", str(recon["snr_db"]),
        "--digital-step-policy", recon["digital_step_policy"],
        "--fixed-reference-snr-db", str(recon["fixed_reference_snr_db"]),
        "--fixed-max-gop", str(recon["fixed_max_gop"]),
        "--image-long-side", str(spatial["image_long_side"]),
        "--image-pad-multiple", str(spatial["pad_to_multiple"]),
        "--fps", "1",
        "--skip-keyframe-sweep", "--skip-source-size-report", "--no-lpips",
    ]
    if max_frames is not None:
        command.extend(["--max-frames", str(max_frames)])
    print(f"[int6-bridge] reconstruction fixed_int6 policy={policy} seed={seed} (resume-safe)", flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"fixed_int6 reconstruction failed: policy={policy}, seed={seed}, exit={completed.returncode}")


def _score_int6_reconstruction(
    g1, protocol: Mapping[str, Any], gt_index: Mapping[str, Any], video_ids: Sequence[str],
    evaluator_model: Mapping[str, str], threshold: float, run_root: Path,
    *, policy: str, seed: int, device: str, max_frames: Optional[int],
) -> List[Dict[str, Any]]:
    """Score fixed_int6 reconstructions with the frozen OWLv2 threshold.

    The per-video cache metadata includes `reconstruction_frame_tree_sha256`
    (see `sgdjscc_lab.utils.frame_hash.frame_tree_sha256`), not just a frame
    count: without a content hash, regenerating the same number of frames
    with different pixels would silently reuse a stale cached score.
    """
    recon = protocol["reconstruction"]
    config_name = f"fixed_int6__{recon['guide_profile']}"
    scorer = g1._OwlV2Scorer(evaluator_model, protocol["evaluator"]["queries"], device)
    rows: List[Dict[str, Any]] = []
    try:
        for number, video_id in enumerate(video_ids, start=1):
            metadata = gt_index["videos"][video_id]
            expected = len(metadata["frame_names"])
            if max_frames is not None:
                expected = min(expected, max_frames)
            recon_dir = (
                run_root / "reconstruction" / "fixed_int6" / policy / f"seed_{seed}" /
                "recon_videos" / video_id / config_name
            )
            paths = [recon_dir / f"frame_{index:05d}.png" for index in range(expected)]
            missing = [path for path in paths if not path.is_file()]
            if missing:
                raise SystemExit(f"missing {len(missing)} fixed_int6 reconstruction frames for {policy}/{video_id}")
            from PIL import Image

            source_first = ROOT / metadata["source_path"] / metadata["frame_names"][0]
            with Image.open(source_first) as source_image:
                content_size = g1._resized_size(
                    source_image.size, int(recon["spatial_transform"]["image_long_side"])
                )
            frame_tree_hash = frame_tree_sha256(paths)
            scored = g1._score_sequence(
                scorer, paths,
                run_root / "evaluator" / "reconstruction" / "fixed_int6" / policy / f"seed_{seed}" / f"{video_id}.json",
                long_side=None, pad_multiple=None, crop_size=content_size,
                cache_metadata={
                    "kind": "reconstruction", "bit_depth": "fixed_int6",
                    "model_revision": evaluator_model["revision"], "frame_count": expected,
                    "evaluation_crop_size": list(content_size),
                    "reconstruction_frame_tree_sha256": frame_tree_hash,
                },
            )
            for item in scored:
                present = set(metadata["present_concepts"][item["frame_index"]])
                for concept, score in item["scores"].items():
                    rows.append({
                        "video_id": video_id, "seed": int(seed),
                        "policy": policy, "condition": f"fixed_int6__{policy}",
                        "frame_index": item["frame_index"], "concept_id": concept,
                        "score": score, "gt_present": concept in present,
                    })
            print(f"[int6-bridge] evaluator {number}/{len(video_ids)}: fixed_int6/{policy}/{video_id}", flush=True)
    finally:
        scorer.close()
    return rows


def load_int6_accounting(run_root: Path, protocol: Mapping[str, Any], seed: int, video_ids: Sequence[str]) -> List[Dict[str, Any]]:
    video_id_set = set(video_ids)
    guide_profile = protocol["reconstruction"]["guide_profile"]
    expected_config = f"fixed_int6__{guide_profile}"
    rows = []
    for policy in protocol["reconstruction"]["policies"]:
        csv_path = run_root / "reconstruction" / "fixed_int6" / policy / f"seed_{seed}" / "per_video_metrics.csv"
        for row in _read_csv_rows(csv_path):
            if row.get("config") != expected_config or row.get("video") not in video_id_set:
                continue
            row = dict(row)
            row["condition"] = f"fixed_int6__{policy}"
            rows.append(row)
    return rows


# --- LPIPS: canonical transform, shared evaluator, strict validation ------

def build_canonical_source_frames(
    tre, gt_index: Mapping[str, Any], video_ids: Sequence[str], *,
    image_long_side: int, image_pad_multiple: int, max_frames: Optional[int],
) -> Dict[str, List[Any]]:
    """Canonical (resized, no-upscale, center-padded) source tensors per
    video, using the exact transform run_transmission_reduction_eval.py
    applies before reconstruction (``tre._load_frames``). Computed once and
    shared between the fixed_int4 and fixed_int6 LPIPS legs, which both
    compare against the identical target.
    """
    result: Dict[str, List[Any]] = {}
    for video_id in video_ids:
        metadata = gt_index["videos"][video_id]
        source_dir = ROOT / metadata["source_path"]
        tensors, _info = tre._load_frames(
            source_dir, ROOT, image_long_side=image_long_side, image_pad_multiple=image_pad_multiple,
        )
        if max_frames is not None:
            tensors = tensors[:max_frames]
        expected = len(metadata["frame_names"])
        if max_frames is not None:
            expected = min(expected, max_frames)
        if len(tensors) != expected:
            raise ValueError(
                f"canonical source frame count for video_id={video_id!r} is {len(tensors)}, "
                f"expected {expected}"
            )
        result[video_id] = tensors
    return result


def build_shared_lpips_evaluator(device: str, net: str):
    """One LPIPS-capable evaluator, constructed exactly once per run and
    reused for both fixed_int4 and fixed_int6 (and across policies) --
    `run()` never calls this more than once, so both conditions are scored
    under literally the same backbone instance and weights.
    """
    import torch
    from sgdjscc_lab.evaluators.quality import QualityEvaluator

    return QualityEvaluator(use_lpips=True, lpips_net=net, device=torch.device(device))


def lpips_evaluator_provenance(evaluator: Any) -> Dict[str, Any]:
    """net / package version / preprocessing description / deterministic
    state-dict hash for the shared LPIPS evaluator, so a resumed or repeated
    run can detect (via `run_spec.json`'s exact-match resume check) that the
    installed `lpips` package or its pretrained weights changed, rather than
    silently comparing two conditions under different backbones.
    """
    try:
        package_version = importlib.metadata.version("lpips")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    model = evaluator._get_lpips()
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return {
        "net": evaluator.lpips_net,
        "package": "lpips",
        "package_version": package_version,
        "preprocessing": LPIPS_PREPROCESSING_DESCRIPTION,
        "state_dict_sha256": digest.hexdigest(),
    }


def _default_load_recon_tensor(path: Path):
    from sgdjscc_lab.io import load_image_as_tensor

    return load_image_as_tensor(path)


def compute_condition_mean_lpips(
    video_ids: Sequence[str],
    source_tensors_by_video: Mapping[str, Sequence[Any]],
    recon_frame_paths: Mapping[str, Sequence[Path]],
    *,
    evaluator: Any,
    load_recon_tensor: Callable[[Path], Any] = _default_load_recon_tensor,
) -> Dict[str, float]:
    """Per-video mean LPIPS between the canonical source tensor and the
    matching reconstruction frame.

    Both the source side (``source_tensors_by_video``, already transformed
    by ``build_canonical_source_frames``) and the reconstruction side use
    the same ``evaluator`` instance -- see ``build_shared_lpips_evaluator``.
    Fails closed (no swallowed exceptions, no silent cropping) on a
    frame-count mismatch, a per-frame shape mismatch, or a None/NaN/Inf
    LPIPS value, rather than producing a mean over a corrupted comparison.
    """
    result: Dict[str, float] = {}
    for video_id in video_ids:
        source_tensors = source_tensors_by_video[video_id]
        recon_paths = recon_frame_paths[video_id]
        if len(source_tensors) != len(recon_paths):
            raise ValueError(
                f"video_id={video_id!r}: {len(source_tensors)} canonical source frame(s) vs "
                f"{len(recon_paths)} reconstruction frame(s) -- LPIPS requires an exact pairing"
            )
        if not source_tensors:
            raise ValueError(f"no frames to compute LPIPS for video_id={video_id!r}")
        values = []
        for index, (source_tensor, recon_path) in enumerate(zip(source_tensors, recon_paths)):
            recon_tensor = load_recon_tensor(recon_path)
            if tuple(source_tensor.shape) != tuple(recon_tensor.shape):
                raise ValueError(
                    f"video_id={video_id!r} frame {index}: canonical source shape "
                    f"{tuple(source_tensor.shape)} != reconstruction shape "
                    f"{tuple(recon_tensor.shape)} ({recon_path})"
                )
            metrics = evaluator.evaluate(source_tensor, recon_tensor)
            lpips_value = metrics["lpips"]
            if lpips_value is None:
                raise ValueError(f"video_id={video_id!r} frame {index}: LPIPS evaluator returned None")
            lpips_value = float(lpips_value)
            if not math.isfinite(lpips_value):
                raise ValueError(
                    f"video_id={video_id!r} frame {index}: LPIPS evaluator returned non-finite value {lpips_value}"
                )
            values.append(lpips_value)
        result[video_id] = sum(values) / len(values)
    return result


def _frame_paths_under(
    gt_index: Mapping[str, Any], video_ids: Sequence[str], max_frames: Optional[int],
    *, base_dir: Path, config_name: str,
) -> Dict[str, List[Path]]:
    """``base_dir`` is the directory directly containing ``recon_videos/``.

    int4 (reused from v1.1) and int6 (this bridge's own tree) have different
    depths above that point -- v1.1 only ever ran fixed_int4, so its tree is
    ``{source_run}/reconstruction/{policy}/seed_{seed}/recon_videos/...``,
    while this bridge's own int6 tree is
    ``{run_root}/reconstruction/fixed_int6/{policy}/seed_{seed}/recon_videos/...``
    -- so callers pass the already-qualified ``base_dir`` for each condition
    rather than this function trying to reconstruct it generically.
    """
    result = {}
    for video_id in video_ids:
        metadata = gt_index["videos"][video_id]
        count = len(metadata["frame_names"])
        if max_frames is not None:
            count = min(count, max_frames)
        recon_dir = base_dir / "recon_videos" / video_id / config_name
        result[video_id] = [recon_dir / f"frame_{index:05d}.png" for index in range(count)]
    return result


def _source_frame_paths(
    gt_index: Mapping[str, Any], video_ids: Sequence[str], max_frames: Optional[int],
) -> Dict[str, List[Path]]:
    result = {}
    for video_id in video_ids:
        metadata = gt_index["videos"][video_id]
        names = metadata["frame_names"]
        if max_frames is not None:
            names = names[:max_frames]
        source_dir = ROOT / metadata["source_path"]
        result[video_id] = [source_dir / name for name in names]
    return result


def lpips_input_frame_hashes(
    video_ids: Sequence[str],
    source_paths_by_video: Mapping[str, Sequence[Path]],
    int4_recon_paths: Mapping[str, Sequence[Path]],
    int6_recon_paths: Mapping[str, Sequence[Path]],
) -> Dict[str, Dict[str, str]]:
    """Per-video content hash of exactly the files fed into LPIPS for each
    side, recorded in the completion manifest for provenance -- independent
    of, and in addition to, the OWLv2 scoring cache's own frame-tree hash.
    """
    return {
        "source": {video_id: frame_tree_sha256(source_paths_by_video[video_id]) for video_id in video_ids},
        "fixed_int4": {video_id: frame_tree_sha256(int4_recon_paths[video_id]) for video_id in video_ids},
        "fixed_int6": {video_id: frame_tree_sha256(int6_recon_paths[video_id]) for video_id in video_ids},
    }


def fill_condition_lpips(
    accounting_rows: List[Dict[str, Any]],
    *,
    condition_label: str,
    lpips_by_video: Mapping[str, float],
) -> None:
    """Overwrite `mean_lpips` in-place for every row of `condition_label`
    with the post-hoc value from `compute_condition_mean_lpips`, replacing
    whatever `run_transmission_reduction_eval.py` wrote (blank, for the
    `--no-lpips` runs both legs of this bridge actually use).
    """
    for row in accounting_rows:
        if row.get("condition") != condition_label:
            continue
        video_id = row.get("video")
        if video_id not in lpips_by_video:
            raise ValueError(f"condition={condition_label!r}: no computed LPIPS for video_id={video_id!r}")
        row["mean_lpips"] = lpips_by_video[video_id]


# --- structural validation shared by smoke and formal ----------------------

def _detection_key(row: Mapping[str, Any]) -> tuple:
    return (str(row["video_id"]), int(row["frame_index"]), str(row["concept_id"]))


def validate_detection_structure(
    detection_rows: Sequence[Mapping[str, Any]], source_rows: Sequence[Mapping[str, Any]],
    *, expected_conditions: Sequence[str], expected_video_ids: Sequence[str],
    expected_concepts: Sequence[str], gt_index: Mapping[str, Any],
    max_frames: Optional[int],
) -> None:
    """Validate the complete frozen detector grid, not only row counts.

    Every condition and the source scorer must cover exactly the requested
    ``(video, frame, concept)`` keys derived from ground truth.  Each
    reconstruction row's ``gt_present`` value is also recomputed from the
    official GT index, so two conditions cannot pass merely by agreeing on
    the same wrong label.
    """
    requested_videos = set(expected_video_ids)
    validate_exact_video_count(
        list(expected_video_ids), expected=len(requested_videos), label="expected_video_ids",
    )
    expected_keys = set()
    expected_gt_present: Dict[tuple, bool] = {}
    for video_id in sorted(requested_videos):
        if video_id not in gt_index.get("videos", {}):
            raise ValueError(f"expected video_id={video_id!r} is absent from gt_index")
        metadata = gt_index["videos"][video_id]
        frame_names = list(metadata.get("frame_names", []))
        present_concepts = list(metadata.get("present_concepts", []))
        frame_count = len(frame_names)
        if max_frames is not None:
            frame_count = min(frame_count, int(max_frames))
        if len(present_concepts) < frame_count:
            raise ValueError(
                f"gt_index video_id={video_id!r} has {len(present_concepts)} present_concepts "
                f"entries for {frame_count} evaluated frame(s)"
            )
        for frame_index in range(frame_count):
            present = set(present_concepts[frame_index])
            for concept in expected_concepts:
                key = (video_id, frame_index, str(concept))
                expected_keys.add(key)
                expected_gt_present[key] = str(concept) in present

    for condition in expected_conditions:
        rows = [row for row in detection_rows if row["condition"] == condition]
        validate_unique_keys(rows, key_of=_detection_key, label=f"detection[{condition}]")
        keys = {_detection_key(row) for row in rows}
        if keys != expected_keys:
            raise ValueError(
                f"detection[{condition}] key set differs from the frozen GT grid: "
                f"missing={sorted(expected_keys - keys)[:10]}, "
                f"unexpected={sorted(keys - expected_keys)[:10]}"
            )
        for row in rows:
            key = _detection_key(row)
            if bool(row["gt_present"]) != expected_gt_present[key]:
                raise ValueError(
                    f"gt_present disagrees with official gt_index for condition={condition!r}, "
                    f"key={key}: row={bool(row['gt_present'])}, "
                    f"official={expected_gt_present[key]}"
                )

    validate_unique_keys(source_rows, key_of=_detection_key, label="source_rows")
    source_keys = {_detection_key(row) for row in source_rows}
    if source_keys != expected_keys:
        raise ValueError(
            "source key set differs from the frozen GT grid: "
            f"missing={sorted(expected_keys - source_keys)[:10]}, "
            f"unexpected={sorted(source_keys - expected_keys)[:10]}"
        )

    int4_conditions = [c for c in expected_conditions if c.startswith("fixed_int4__")]
    int6_conditions = [c for c in expected_conditions if c.startswith("fixed_int6__")]
    for int4_condition, int6_condition in zip(sorted(int4_conditions), sorted(int6_conditions)):
        int4_rows = [row for row in detection_rows if row["condition"] == int4_condition]
        int6_rows = [row for row in detection_rows if row["condition"] == int6_condition]
        int4_keys = {_detection_key(row) for row in int4_rows}
        int6_keys = {_detection_key(row) for row in int6_rows}
        if int4_keys != int6_keys:
            raise ValueError(
                f"detection key sets differ between {int4_condition!r} and {int6_condition!r}: "
                f"only_in_{int4_condition}={sorted(int4_keys - int6_keys)[:5]}, "
                f"only_in_{int6_condition}={sorted(int6_keys - int4_keys)[:5]}"
            )
        validate_gt_present_consistency(
            int4_rows, int6_rows, key_of=_detection_key,
            label_a=int4_condition, label_b=int6_condition,
        )


def build_summary(
    protocol: Mapping[str, Any], gt_index: Mapping[str, Any],
    detection_rows: Sequence[Dict[str, Any]], accounting_rows: Sequence[Dict[str, Any]],
    source_rows: Sequence[Dict[str, Any]], *, threshold: float,
    expected_video_ids: Sequence[str], max_frames: Optional[int], smoke: bool,
) -> Dict[str, Any]:
    """Pure aggregation step (no I/O): combine per-condition detection rows +
    accounting rows into the bridge comparison report.

    Structural validation (condition/video coverage, no duplicate detection
    keys, int4/int6/source key equality, gt_present agreement) always runs.
    Paired accounting/quality-gate deltas are computed only for formal runs:
    during smoke, fixed_int4's accounting reflects its full video length
    while fixed_int6 only covers a 2-frame budget, so any paired delta there
    would silently compare mismatched populations -- those fields are
    reported as ``OMITTED_ASYMMETRIC_SMOKE`` instead, alongside the
    genuinely comparable (frame-restricted on both sides) detection-based
    metrics, which are computed normally even for smoke.
    """
    reference_bit_depth = protocol["reconstruction"]["reference_bit_depth"]
    policies = sorted(protocol["reconstruction"]["policies"])
    expected_conditions = sorted(
        f"{bit_depth}__{policy}"
        for bit_depth in (reference_bit_depth, "fixed_int6")
        for policy in policies
    )

    detection_conditions = {row["condition"] for row in detection_rows}
    accounting_conditions = {row["condition"] for row in accounting_rows}
    expected_condition_set = set(expected_conditions)
    missing_detection = expected_condition_set - detection_conditions
    missing_accounting = expected_condition_set - accounting_conditions
    if missing_detection:
        raise ValueError(f"missing detection rows for condition(s): {sorted(missing_detection)}")
    if missing_accounting:
        raise ValueError(f"missing accounting rows for condition(s): {sorted(missing_accounting)}")
    unexpected_detection = detection_conditions - expected_condition_set
    unexpected_accounting = accounting_conditions - expected_condition_set
    if unexpected_detection:
        raise ValueError(f"unexpected detection condition(s): {sorted(unexpected_detection)}")
    if unexpected_accounting:
        raise ValueError(f"unexpected accounting condition(s): {sorted(unexpected_accounting)}")

    int4_concepts = {row["concept_id"] for row in detection_rows if row["condition"].startswith(f"{reference_bit_depth}__")}
    int6_concepts = {row["concept_id"] for row in detection_rows if row["condition"].startswith("fixed_int6__")}
    if int4_concepts != int6_concepts:
        raise ValueError(
            f"detector concept_id sets differ between conditions: "
            f"{reference_bit_depth}={sorted(int4_concepts)} vs fixed_int6={sorted(int6_concepts)}"
        )

    expected_video_id_set = set(expected_video_ids)
    validate_exact_video_count(
        list(expected_video_ids), expected=len(expected_video_id_set), label="expected_video_ids",
    )
    for condition in expected_conditions:
        condition_video_ids = [row["video"] for row in accounting_rows if row["condition"] == condition]
        validate_exact_video_count(
            condition_video_ids, expected=len(expected_video_id_set), label=f"accounting[{condition}]",
        )
        if set(condition_video_ids) != expected_video_id_set:
            raise ValueError(
                f"accounting[{condition}] video ID set differs from expected_video_ids: "
                f"missing={sorted(expected_video_id_set - set(condition_video_ids))}, "
                f"unexpected={sorted(set(condition_video_ids) - expected_video_id_set)}"
            )

    validate_detection_structure(
        detection_rows, source_rows,
        expected_conditions=expected_conditions, expected_video_ids=expected_video_ids,
        expected_concepts=protocol["evaluator"]["concepts"], gt_index=gt_index,
        max_frames=max_frames,
    )

    def accounting_video_id_of(row: Mapping[str, Any]) -> str:
        return str(row["video"])

    metric_cfg = protocol["metrics"]
    bootstrap_iterations = int(metric_cfg["statistics"]["bootstrap_iterations"])

    # Pooled *absolute* per-condition accounting (exact bytes/frame/PSNR/
    # SSIM/LPIPS/latency, mean + CI) -- distinct from the paired delta below.
    if smoke:
        accounting_absolute: Any = OMITTED_ASYMMETRIC_SMOKE
        accounting_paired: Any = OMITTED_ASYMMETRIC_SMOKE
        quality_verdicts: Any = OMITTED_ASYMMETRIC_SMOKE
    else:
        accounting_absolute = summarize_condition_metrics(
            accounting_rows, condition_of=lambda row: str(row["condition"]),
            video_id_of=accounting_video_id_of, fields=_ACCOUNTING_FIELDS,
            bootstrap_iterations=bootstrap_iterations,
        )
        accounting_paired = {}
        quality_verdicts = {}
        quality_gate = metric_cfg["quality_gate"]
        for policy in policies:
            reference = f"{reference_bit_depth}__{policy}"
            candidate = f"fixed_int6__{policy}"
            deltas = paired_condition_metrics(
                accounting_rows, condition_of=lambda row: str(row["condition"]),
                video_id_of=accounting_video_id_of, reference=reference, candidate=candidate,
                fields=_ACCOUNTING_FIELDS, bootstrap_iterations=bootstrap_iterations,
            )
            accounting_paired[policy] = deltas
            quality_verdicts[policy] = quality_gate_pass(
                {field: entry["mean_delta"] for field, entry in deltas.items()},
                psnr_drop_db_max=float(quality_gate["psnr_drop_db_max"]),
                ssim_drop_max=float(quality_gate["ssim_drop_max"]),
                lpips_rise_max=float(quality_gate["lpips_rise_max"]),
            )
        if set(quality_verdicts) != set(policies):
            raise ValueError(
                f"quality_verdicts covers {sorted(quality_verdicts)}, expected {sorted(policies)}; "
                "refusing to report an incomplete quality gate"
            )

    # H_add: two different estimands, both reported and explicitly labeled.
    # "pooled_absolute" pools every opportunity across all videos into one
    # ratio per condition (numerator/denominator exposed) -- the same
    # definition `raw_and_paired_h_add` uses for the frozen v1.1/v1.2
    # pipeline. "paired_video_delta" instead keeps each video's own ratio
    # and reports the inner-joined per-video candidate-reference difference
    # with a paired bootstrap CI. These answer different questions ("how
    # prevalent is it overall" vs "does switching bit depth change it,
    # video by video") and are never collapsed into a single number.
    relabeled_detection_rows = [dict(row, policy=row["condition"]) for row in detection_rows]
    pooled_absolute_h_add = raw_and_paired_h_add(relabeled_detection_rows, source_rows, threshold)

    h_add_paired: Dict[str, Any] = {}
    h_add_video = h_add_by_video(relabeled_detection_rows, source_rows, threshold)
    for policy in policies:
        reference = f"{reference_bit_depth}__{policy}"
        candidate = f"fixed_int6__{policy}"
        reference_values = h_add_video.get(reference, {})
        candidate_values = h_add_video.get(candidate, {})
        h_add_paired[policy] = {
            field: paired_delta_from_video_values(
                {video: values[field] for video, values in reference_values.items() if values[field] is not None},
                {video: values[field] for video, values in candidate_values.items() if values[field] is not None},
                bootstrap_iterations=bootstrap_iterations,
            )
            for field in ("raw_h_add", "source_paired_h_add")
        }

    return {
        "protocol_id": protocol["protocol_id"],
        "threshold": float(threshold),
        "expected_video_count": len(expected_video_id_set),
        "expected_video_ids": sorted(expected_video_id_set),
        "conditions": expected_conditions,
        "by_condition_accounting_absolute": accounting_absolute,
        "fixed_int6_vs_reference_accounting_delta": accounting_paired,
        "fixed_int6_quality_gate": quality_verdicts,
        "h_add_estimands": {
            "pooled_absolute": {
                "definition": "all opportunities across all videos pooled into one ratio per condition",
                "by_condition": pooled_absolute_h_add,
            },
            "paired_video_delta": {
                "definition": "per-video ratio, inner-joined candidate-reference difference with paired bootstrap CI",
                "by_policy": h_add_paired,
            },
        },
        "heldout_accessed": False,
    }


def build_ghost_delta(
    gt_index: Mapping[str, Any], detection_rows: Sequence[Mapping[str, Any]],
    *, threshold: float, ghost_horizon: int, bootstrap_iterations: int,
    reference_bit_depth: str, policies: Sequence[str],
    expected_video_ids: Sequence[str],
) -> Dict[str, Any]:
    """Paired ghost-survival AUC delta per policy, restricted to EXIT events
    that are uncensored in BOTH conditions for that video/event, with the
    bootstrap CI computed at *video* granularity (`paired_delta_video_clustered`)
    so a video contributing more than one EXIT event does not have its
    events treated as independent draws.

    Every event is classified into exactly one of four buckets: jointly
    uncensored (contributes a paired point), reference-only censored
    (censored in the reference condition, uncensored in the candidate),
    candidate-only censored (censored in the candidate condition,
    uncensored in the reference), or jointly censored in both -- with
    counts for each, plus how many distinct videos each touches.
    """
    expected_video_id_set = set(expected_video_ids)
    row_video_ids = {str(row["video_id"]) for row in detection_rows}
    if row_video_ids != expected_video_id_set:
        raise ValueError(
            "ghost detection video ID set differs from expected_video_ids: "
            f"missing={sorted(expected_video_id_set - row_video_ids)}, "
            f"unexpected={sorted(row_video_ids - expected_video_id_set)}"
        )
    selected_events = [
        event for event in gt_index["events"]
        if event.get("event_type") == "EXIT" and str(event["video_id"]) in expected_video_id_set
    ]
    all_exit_keys = {
        (str(event["video_id"]), str(event["event_id"]))
        for event in selected_events
    }
    result: Dict[str, Any] = {}
    for policy in policies:
        reference = f"{reference_bit_depth}__{policy}"
        candidate = f"fixed_int6__{policy}"

        def _uncensored_auc(condition: str) -> Dict[tuple, float]:
            rows = [
                dict(row, policy=row["condition"])
                for row in detection_rows if row["condition"] == condition
            ]
            summary = summarize_g1(
                rows, selected_events, threshold=threshold, primary_policy=condition,
                bootstrap_iterations=50, ghost_horizon=ghost_horizon,
            )
            return {
                (str(item["video_id"]), str(item["event_id"])): item["auc"]
                for item in summary["ghost_survival"]["uncensored"]
            }

        reference_auc = _uncensored_auc(reference)
        candidate_auc = _uncensored_auc(candidate)
        jointly_uncensored = sorted(set(reference_auc) & set(candidate_auc))
        # Uncensored in reference but not candidate => candidate is the one censored.
        candidate_only_censored = sorted(set(reference_auc) - set(candidate_auc))
        # Uncensored in candidate but not reference => reference is the one censored.
        reference_only_censored = sorted(set(candidate_auc) - set(reference_auc))
        jointly_censored = sorted(all_exit_keys - set(reference_auc) - set(candidate_auc))

        def _by_video(keys: Sequence[tuple]) -> Dict[str, List[str]]:
            grouped: Dict[str, List[str]] = {}
            for video_id, event_id in keys:
                grouped.setdefault(video_id, []).append(event_id)
            return grouped

        censoring_breakdown = {
            "jointly_uncensored": {"n_events": len(jointly_uncensored), "n_videos": len(_by_video(jointly_uncensored))},
            "reference_only_censored": {"n_events": len(reference_only_censored), "n_videos": len(_by_video(reference_only_censored))},
            "candidate_only_censored": {"n_events": len(candidate_only_censored), "n_videos": len(_by_video(candidate_only_censored))},
            "jointly_censored": {"n_events": len(jointly_censored), "n_videos": len(_by_video(jointly_censored))},
        }

        if not jointly_uncensored:
            result[policy] = {"censoring_breakdown": censoring_breakdown, "paired_delta": None}
            continue

        reference_by_video: Dict[str, List[float]] = {}
        candidate_by_video: Dict[str, List[float]] = {}
        for video_id, event_id in sorted(jointly_uncensored):
            reference_by_video.setdefault(video_id, []).append(reference_auc[(video_id, event_id)])
            candidate_by_video.setdefault(video_id, []).append(candidate_auc[(video_id, event_id)])
        result[policy] = {
            "censoring_breakdown": censoring_breakdown,
            "paired_delta": paired_delta_video_clustered(
                reference_by_video, candidate_by_video, bootstrap_iterations=bootstrap_iterations,
            ),
        }
    return result


def build_ghost_report(
    gt_index: Mapping[str, Any], detection_rows: Sequence[Mapping[str, Any]],
    *, threshold: float, ghost_horizon: int, bootstrap_iterations: int,
    reference_bit_depth: str, policies: Sequence[str],
    expected_video_ids: Sequence[str], smoke: bool, max_frames: Optional[int],
) -> Any:
    """Return a valid ghost comparison or an explicit smoke omission.

    The two-frame smoke cannot contain a complete 16-frame ghost horizon, so
    treating every out-of-scope EXIT event as right-censored would be a false
    statistic.  Formal runs delegate to ``build_ghost_delta`` after limiting
    events to the exact run video set.
    """
    if smoke and (max_frames is None or int(max_frames) < int(ghost_horizon)):
        return {
            "status": OMITTED_INSUFFICIENT_HORIZON_SMOKE,
            "available_frame_cap": max_frames,
            "required_horizon_frames": int(ghost_horizon),
            "reason": "smoke frame cap is shorter than the frozen ghost-survival horizon",
        }
    return build_ghost_delta(
        gt_index, detection_rows, threshold=threshold, ghost_horizon=ghost_horizon,
        bootstrap_iterations=bootstrap_iterations, reference_bit_depth=reference_bit_depth,
        policies=policies, expected_video_ids=expected_video_ids,
    )


# --- top-level orchestration -----------------------------------------------

def gather_provenance(
    g1, protocol: Mapping[str, Any], *, formal: bool, mode_label: str,
    video_ids: Sequence[str], seed: int, device: str,
    preflight_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute everything needed to decide whether this invocation may
    proceed/resume -- int4 reuse verification, preflight, and the candidate
    run_spec -- without writing anything to disk.

    Split out from `_execute` so "no writes before the resume check" is a
    structural property, not just a claim: this function's signature has no
    ``run_root`` parameter at all, so it cannot touch one.
    """
    preflight_fn = preflight_fn or _preflight
    reuse_info = verify_int4_reuse(g1, protocol, video_ids, seed)
    preflight = preflight_fn(g1, protocol, formal=formal)
    run_spec = {
        "mode": mode_label,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "dataset_preparation_manifest_sha256": _sha256_file(G1_DATASET / "preparation_manifest.json"),
        "video_ids": list(video_ids),
        "policies": {policy: int(steps) for policy, steps in protocol["reconstruction"]["policies"].items()},
        "seed": seed,
        "device": device,
        "int4_source_run": reuse_info["source_run"],
        "int4_source_protocol_sha256": protocol["reuse"]["int4_source_protocol_sha256"],
        "int4_source_detection_rows_sha256": reuse_info["detection_rows_sha256"],
        "int4_source_per_video_metrics_sha256": reuse_info["per_video_metrics_sha256"],
        "checkpoint_sha256": preflight["checkpoint_sha256"],
        "evaluator_model": reuse_info["evaluator_model"],
        "lpips_provenance": preflight.get("lpips_provenance"),
        "critical_file_sha256": preflight.get("critical_file_sha256"),
        "git_commit": preflight["git"]["commit"],
        "git_dirty": preflight["git"]["dirty"],
        "heldout_accessed": False,
    }
    return {"reuse_info": reuse_info, "preflight": preflight, "run_spec": run_spec}


def _write_status(g1, run_root: Path, status: str, **extra: Any) -> None:
    g1._atomic_json(run_root / "status.json", {"status": status, "updated_at": time.time(), **extra})


def build_artifact_checksum_payload(run_root: Path) -> Dict[str, Any]:
    """Hash every immutable scientific output under ``run_root``.

    Mutable/operator-owned files are intentionally excluded: ``status.json``
    is written after this manifest, ``operator.log`` may still be open via
    ``tee``, and child ``logs/`` are diagnostic streams rather than metric
    inputs.  Reconstruction PNG/MP4 files, child manifests/summaries/CSVs,
    evaluator caches, and the top-level report/provenance files are included.
    """
    excluded_names = {"artifact_checksums.json", "status.json", "operator.log"}
    files: Dict[str, Dict[str, Any]] = {}
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative_path = path.relative_to(run_root)
        if path.name in excluded_names or path.name.endswith(".tmp") or "logs" in relative_path.parts:
            continue
        relative = relative_path.as_posix()
        files[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    if not files:
        raise ValueError(f"no immutable scientific artifacts found under {run_root}")
    return {
        "algorithm": "sha256",
        "scope": "all immutable scientific artifacts recursively under run_root",
        "excluded": ["artifact_checksums.json", "status.json", "operator.log", "**/logs/**", "*.tmp"],
        "file_count": len(files),
        "total_size_bytes": sum(entry["size_bytes"] for entry in files.values()),
        "files": files,
    }


def _write_artifact_checksums(g1, run_root: Path) -> None:
    g1._atomic_json(run_root / "artifact_checksums.json", build_artifact_checksum_payload(run_root))


def run(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Two-frame structural GPU check; never paper evidence.")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    g1 = _load_g1_support()
    return _execute(args, g1)


def _execute(
    args: argparse.Namespace, g1, *, preflight_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> int:
    run_root = args.run_root.resolve()
    protocol = _load_protocol()
    if protocol["gate"]["require_g0_pass"]:
        g1._require_g0_pass()
    g1._prepare_dataset()
    gt_index = g1._load_json(G1_DATASET / "ground_truth_index.json")

    smoke = args.smoke
    seed = int(protocol["reconstruction"]["seeds"][0])
    video_ids = [protocol["gate"]["smoke_video_id"]] if smoke else sorted(gt_index["videos"])
    max_frames = 2 if smoke else None
    expected_video_count = 1 if smoke else int(protocol["data"]["expected_videos"])
    validate_exact_video_count(video_ids, expected=expected_video_count, label="run video_ids")

    # Preflight-only forecasts a formal run's own strictness (critical-file
    # tracking, clean checkout) whenever not run under --smoke, so the user
    # gets that warning before committing to the long-running job.
    formal_strictness = not smoke
    mode_label = "preflight_only" if args.preflight_only else ("smoke" if smoke else "formal")

    # Nothing below this point writes to disk until check_run_spec_resume
    # has passed -- gather_provenance takes no run_root and cannot write.
    provenance = gather_provenance(
        g1, protocol, formal=formal_strictness, mode_label=mode_label,
        video_ids=video_ids, seed=seed, device=args.device, preflight_fn=preflight_fn,
    )
    reuse_info, preflight, run_spec = provenance["reuse_info"], provenance["preflight"], provenance["run_spec"]

    if args.preflight_only:
        # A preflight is a read-only forecast.  In particular, the documented
        # workflow may point this at the future formal run-root, which must not
        # be created or changed merely by checking whether a run is permitted.
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    spec_path = run_root / "run_spec.json"
    existing_spec = g1._load_json(spec_path) if spec_path.is_file() else None
    check_run_spec_resume(existing_spec, run_spec, run_root=run_root)

    # Only now, after confirming this run-root is either fresh or a
    # compatible resume, do we touch the filesystem.
    run_root.mkdir(parents=True, exist_ok=True)
    g1._atomic_json(run_root / "int4_reuse_verification.json", reuse_info)
    g1._atomic_json(run_root / "preflight.json", preflight)
    g1._atomic_json(spec_path, run_spec)
    _write_status(g1, run_root, "RUNNING", mode=mode_label)

    try:
        for policy, steps in protocol["reconstruction"]["policies"].items():
            _run_int6_reconstruction(
                protocol, video_ids, run_root, policy=policy, steps=int(steps),
                seed=seed, device=args.device, max_frames=max_frames,
            )

        int6_rows: List[Dict[str, Any]] = []
        for policy in protocol["reconstruction"]["policies"]:
            int6_rows.extend(_score_int6_reconstruction(
                g1, protocol, gt_index, video_ids, reuse_info["evaluator_model"],
                reuse_info["threshold"], run_root, policy=policy, seed=seed,
                device=args.device, max_frames=max_frames,
            ))
        g1._atomic_jsonl(run_root / "evaluator" / "int6_detection_rows.jsonl", int6_rows)

        int4_rows = load_int4_rows(g1, protocol, seed, video_ids, max_frames)
        int4_accounting = load_int4_accounting(protocol, seed, video_ids)
        int6_accounting = load_int6_accounting(run_root, protocol, seed, video_ids)
        source_rows = load_source_rows(g1, reuse_info, gt_index, video_ids, max_frames)

        # LPIPS: shared evaluator, canonical source transform computed once
        # and reused for both conditions.
        reference_bit_depth = protocol["reconstruction"]["reference_bit_depth"]
        guide_profile = protocol["reconstruction"]["guide_profile"]
        spatial = protocol["reconstruction"]["spatial_transform"]
        tre = _load_transmission_reduction_eval_support()
        lpips_net = protocol["metrics"]["lpips"]["net"]
        lpips_evaluator = build_shared_lpips_evaluator(device=args.device, net=lpips_net)
        canonical_source_tensors = build_canonical_source_frames(
            tre, gt_index, video_ids, image_long_side=int(spatial["image_long_side"]),
            image_pad_multiple=int(spatial["pad_to_multiple"]), max_frames=max_frames,
        )
        source_paths = _source_frame_paths(gt_index, video_ids, max_frames)
        lpips_hashes_by_policy = {}
        for policy in protocol["reconstruction"]["policies"]:
            int4_base_dir = Path(reuse_info["source_run"]) / "reconstruction" / policy / f"seed_{seed}"
            int6_base_dir = run_root / "reconstruction" / "fixed_int6" / policy / f"seed_{seed}"
            int4_recon_paths = _frame_paths_under(
                gt_index, video_ids, max_frames, base_dir=int4_base_dir,
                config_name=f"{reference_bit_depth}__{guide_profile}",
            )
            int6_recon_paths = _frame_paths_under(
                gt_index, video_ids, max_frames, base_dir=int6_base_dir,
                config_name=f"fixed_int6__{guide_profile}",
            )
            int4_lpips = compute_condition_mean_lpips(
                video_ids, canonical_source_tensors, int4_recon_paths, evaluator=lpips_evaluator,
            )
            int6_lpips = compute_condition_mean_lpips(
                video_ids, canonical_source_tensors, int6_recon_paths, evaluator=lpips_evaluator,
            )
            fill_condition_lpips(int4_accounting, condition_label=f"{reference_bit_depth}__{policy}", lpips_by_video=int4_lpips)
            fill_condition_lpips(int6_accounting, condition_label=f"fixed_int6__{policy}", lpips_by_video=int6_lpips)
            lpips_hashes_by_policy[policy] = lpips_input_frame_hashes(
                video_ids, source_paths, int4_recon_paths, int6_recon_paths,
            )

        report = build_summary(
            protocol, gt_index, int4_rows + int6_rows, int4_accounting + int6_accounting,
            source_rows, threshold=float(reuse_info["threshold"]), expected_video_ids=video_ids,
            max_frames=max_frames, smoke=smoke,
        )
        metric_cfg = protocol["metrics"]
        report["ghost_survival_delta"] = build_ghost_report(
            gt_index, int4_rows + int6_rows, threshold=float(reuse_info["threshold"]),
            ghost_horizon=int(metric_cfg["ghost_survival_auc"]["horizon_annotated_timesteps"]),
            bootstrap_iterations=int(metric_cfg["statistics"]["bootstrap_iterations"]),
            reference_bit_depth=reference_bit_depth, policies=sorted(protocol["reconstruction"]["policies"]),
            expected_video_ids=video_ids, smoke=smoke, max_frames=max_frames,
        )
        report["lpips_provenance"] = preflight.get("lpips_provenance")
        report["lpips_input_frame_hashes"] = lpips_hashes_by_policy
        report["evidence_scope"] = "NOT_EVIDENCE_SMOKE" if smoke else "OVIS_PILOT_INT6_BRIDGE"
        report["status"] = "PASSED"
        g1._atomic_json(run_root / "int6_bridge_summary.json", report)
        _write_artifact_checksums(g1, run_root)
        _write_status(g1, run_root, "PASSED", mode=mode_label, summary_path=str(run_root / "int6_bridge_summary.json"))
    except BaseException as error:
        _write_status(g1, run_root, "FAILED", mode=mode_label, error=str(error))
        raise
    print(f"[int6-bridge] complete -> {run_root / 'int6_bridge_summary.json'}", flush=True)
    return 0


def _preflight(g1, protocol: Mapping[str, Any], formal: bool) -> Dict[str, Any]:
    import torch
    from sgdjscc_lab.paths import model_root

    git = g1._git_state()
    if formal and git["dirty"]:
        raise SystemExit("formal int6 bridge run requires a tracked-clean checkout")
    critical_file_sha256 = verify_critical_files_tracked_at_head(ROOT, CRITICAL_RELATIVE_PATHS) if formal else None
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this environment")
    checkpoints = [
        "JSCC_model.pth", "diffusion_backbone.pth", "diffusion_controlnet.pth",
        "muge-epoch-19-checkpoint.pth",
    ]
    checkpoint_root = Path(model_root())
    missing = [name for name in checkpoints if not (checkpoint_root / name).is_file()]
    if missing:
        raise SystemExit(f"missing SGD-JSCC checkpoints: {missing}")
    evaluator = g1._resolve_hf_snapshot(protocol["evaluator"]["model_id"])
    lpips_net = protocol["metrics"]["lpips"]["net"]
    lpips_evaluator = build_shared_lpips_evaluator(device="cpu", net=lpips_net)
    lpips_provenance = lpips_evaluator_provenance(lpips_evaluator)
    if lpips_provenance["net"] != lpips_net:
        raise SystemExit(
            f"LPIPS evaluator net={lpips_provenance['net']!r} does not match the frozen "
            f"protocol's declared net={lpips_net!r}"
        )
    return {
        "status": "PASSED", "git": git, "python": sys.executable,
        "torch_version": torch.__version__,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_sha256": {name: g1._sha256(checkpoint_root / name) for name in checkpoints},
        "evaluator_model": evaluator,
        "lpips_provenance": lpips_provenance,
        "critical_file_sha256": critical_file_sha256,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
