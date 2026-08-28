#!/usr/bin/env python
"""Three-GPU reconstruction + calibrated integrated evaluation orchestrator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sgdjscc_lab.utils import run_manifest as rm  # noqa: E402

POLICIES = {
    "full50": {"decoder_mode": "diffusion", "diffusion_step": 50},
    "few10": {"decoder_mode": "diffusion", "diffusion_step": 10},
    "vae_direct": {"decoder_mode": "vae_direct", "diffusion_step": 50},
}
PROFILES = (
    "baseline", "combined_ds4",
    "candidate_edge_ds4_uncertainty_omit", "candidate_both_omit",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--dataset-root", default=str(ROOT / "data/etri_video_eval"))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke-test only; full validator requires 100.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _load_parallel():
    spec = importlib.util.spec_from_file_location(
        "_integrated_parallel", ROOT / "scripts/run_transmission_normalization_parallel.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_videos(dataset_root):
    with (Path(dataset_root) / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [{"key": f"{r['id']}_{r['name']}", "n_frames": int(r["n_frames"])} for r in rows]


def _assignments(dataset_root, devices):
    parallel = _load_parallel()
    values = parallel.assign_videos(_manifest_videos(dataset_root), devices)
    for index, item in enumerate(values):
        item["worker_id"] = f"worker_{index:02d}"
    return values


def _policy_command(args, run_root, policy, *, preflight=False):
    child = run_root / "reconstruction" / policy
    setting = POLICIES[policy]
    command = [
        sys.executable, "scripts/run_transmission_normalization_parallel.py",
        "--devices", args.devices, "--dataset-root", str(Path(args.dataset_root).resolve()),
        "--configs", "fixed_int4", "--guide-profiles", ",".join(PROFILES),
        "--decoder-mode", setting["decoder_mode"],
        "--diffusion-step", str(setting["diffusion_step"]),
        "--digital-step-policy", "fixed_reference", "--fixed-reference-snr-db", "10",
        "--seed", str(args.seed), "--no-match-fixed-keyframes",
        "--skip-keyframe-sweep", "--skip-source-size-report",
    ]
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    if preflight:
        command.extend(["--output-root", str(child), "--preflight-only"])
    elif (child / "parallel_plan.json").is_file():
        command.extend(["--resume", str(child)])
    else:
        command.extend(["--output-root", str(child)])
    return command


def _semantic_command(args, run_root, worker, *, preflight=False):
    command = [
        sys.executable, "scripts/evaluate_integrated_semantics.py",
        "--run-root", str(run_root), "--worker-id", worker["worker_id"],
        "--device", "cuda:0", "--dataset-root", str(Path(args.dataset_root).resolve()),
    ]
    if preflight:
        command.append("--preflight-only")
    else:
        command.append("--resume")
    return command


def _environment(worker):
    env = dict(os.environ)
    physical = worker["device"].split(":", 1)[1]
    env["CUDA_VISIBLE_DEVICES"] = physical
    env["SGDJSCC_PHYSICAL_CUDA_DEVICE"] = worker["device"]
    env["PYTHON_BIN"] = sys.executable
    return env


def _write_plan(path, plan):
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError("integrated resume plan mismatch; use a new --output-root")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(argv=None):
    args = parse_args(argv)
    if args.output_root and args.resume:
        raise ValueError("pass at most one of --output-root and --resume")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if devices != ["cuda:0", "cuda:1", "cuda:2"]:
        raise ValueError("integrated protocol requires exactly cuda:0,cuda:1,cuda:2")
    if args.seed != 2025:
        raise ValueError("integrated protocol requires seed 2025")
    run_root = Path(args.resume or args.output_root or (
        ROOT / "outputs" / f"integrated_semantic_validation_10db_{time.strftime('%Y%m%d_%H%M%S')}"
    )).resolve()
    assignments = _assignments(args.dataset_root, devices)
    state = rm.get_git_state(ROOT)
    if state["commit"] == rm.UNKNOWN or state["dirty"] is True:
        raise RuntimeError("integrated experiment requires an exact clean git commit")
    plan = {
        "schema_version": 1, "git": state,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_manifest_sha256": rm.sha256_file(Path(args.dataset_root) / "manifest.csv"),
        "seed": args.seed, "fixed_reference_snr_db": 10.0,
        "configs": ["fixed_int4"], "policies": POLICIES,
        "guide_profiles": list(PROFILES), "assignments": assignments,
        "expected_pairs": 10 * len(POLICIES) * len(PROFILES),
        "max_frames": args.max_frames,
    }
    policy_commands = [_policy_command(args, run_root, name) for name in POLICIES]
    semantic_commands = [_semantic_command(args, run_root, worker) for worker in assignments]
    if args.dry_run:
        print(json.dumps({
            "output_root": str(run_root), "plan": plan,
            "reconstruction_commands": policy_commands,
            "semantic_commands": semantic_commands,
            "cuda_visible_devices": [worker["device"].split(":", 1)[1] for worker in assignments],
        }, indent=2))
        return 0

    env = dict(os.environ)
    env["PYTHON_BIN"] = sys.executable
    for policy in POLICIES:
        subprocess.run(_policy_command(args, run_root, policy, preflight=True), cwd=ROOT, env=env, check=True)
    for worker in assignments:
        subprocess.run(_semantic_command(args, run_root, worker, preflight=True), cwd=ROOT, env=_environment(worker), check=True)
    if args.preflight_only:
        print("integrated 3-GPU preflight passed")
        return 0

    _write_plan(run_root / "integrated_plan.json", plan)
    for policy in POLICIES:
        command = _policy_command(args, run_root, policy)
        log_path = run_root / f"reconstruction_{policy}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((run_root / "reconstruction" / policy / "summary.json").read_text(encoding="utf-8"))
        if summary.get("run_status") != "completed":
            raise RuntimeError(f"{policy} reconstruction ended as {summary.get('run_status')}")

    processes = []
    for worker in assignments:
        log_path = run_root / "semantic" / "workers" / worker["worker_id"] / "semantic_run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            _semantic_command(args, run_root, worker), cwd=ROOT,
            env=_environment(worker), stdout=log, stderr=subprocess.STDOUT,
        )
        processes.append((worker, process, log))
        print(f"started semantic {worker['worker_id']} on {worker['device']}: {','.join(worker['videos'])}")
    failures = []
    for worker, process, log in processes:
        code = process.wait()
        log.close()
        print(f"finished semantic {worker['worker_id']} returncode={code}")
        if code:
            failures.append({"worker_id": worker["worker_id"], "returncode": code})
    if failures:
        raise RuntimeError(f"semantic workers failed before safe merge: {failures}")

    subprocess.run([
        sys.executable, "scripts/summarize_integrated_semantic_validation.py",
        "--run-root", str(run_root),
    ], cwd=ROOT, env=env, check=True)
    child_hashes = {
        policy: rm.sha256_file(run_root / "reconstruction" / policy / "run_manifest.json")
        for policy in POLICIES
    }
    artifact_hashes = json.loads((run_root / "artifact_sha256.json").read_text(encoding="utf-8"))
    manifest = rm.build_run_manifest(
        run_id=run_root.name, command_argv=sys.argv, command_source="captured",
        seed=args.seed, dataset_ref=plan["dataset_root"], dataset_hash=plan["dataset_manifest_sha256"],
        include_environment=False, repo_root=ROOT,
        exact_fields=["serialized bundle bytes", "saved reconstruction PNGs", "paired video rows"],
        proxy_fields=["CLIP/OWLv2/VQA semantic judgments"],
        nan_or_failure_counts={"n_failed_pairs": 0},
        extra={"run_status": "completed", "integrated_plan": plan,
               "child_manifest_sha256": child_hashes, "output_artifact_sha256": artifact_hashes},
    )
    rm.write_run_manifest(run_root / "run_manifest.json", manifest)
    print(f"integrated evaluation completed: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
