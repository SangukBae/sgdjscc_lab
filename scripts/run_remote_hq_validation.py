#!/usr/bin/env python3
"""Run the complete RTX-4090 high-quality ETRI validation workflow.

This intentionally skips mock/no-model/low-step smoke experiments.  Every
model-producing phase uses real checkpoints, writes to an isolated output
root, records the exact subprocess command, and supports ``--skip-existing``
for safe continuation after a long run.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
_ALL_PHASES = ("image", "video", "remeasure", "svd", "wan", "quality")


def _parse_devices(value: str) -> List[str]:
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if not devices or any(not item.startswith("cuda:") for item in devices):
        raise argparse.ArgumentTypeError("use comma-separated CUDA devices, e.g. cuda:0,cuda:1,cuda:2")
    return devices


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Direct high-quality validation on the 3x RTX-4090 remote server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", default=str(_REPO_ROOT / "outputs" / "remote_hq_4090"))
    parser.add_argument("--phases", default="all",
                        help=f"Comma list from {_ALL_PHASES} or all.")
    parser.add_argument("--devices", type=_parse_devices, default=_parse_devices("cuda:0,cuda:1,cuda:2"))
    parser.add_argument("--worker-python", default=(
        "/home/wilco/SangukBae/Semantic/.venvs/lgvsc_gen/bin/python"
    ))
    parser.add_argument("--image-input", default=str(_REPO_ROOT / "data" / "kodak"))
    parser.add_argument("--snr-list", default="-5,0,5,10,15,20,25")
    parser.add_argument("--video-snr", type=float, default=5.0)
    parser.add_argument("--diffusion-step", type=int, default=50)
    parser.add_argument("--videos", default=None,
                        help="Optional comma-separated subset; default is all 10 videos.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Default None evaluates full clips. Setting this makes the run non-full.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--svd-steps", type=int, default=25)
    parser.add_argument("--wan-steps", type=int, default=30)
    parser.add_argument("--worker-max-memory", default=(
        '{"0":"8GiB","1":"22GiB","2":"22GiB","cpu":"40GiB"}'
    ))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write/print the full command plan without running subprocesses.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _selected_phases(value: str) -> List[str]:
    phases = list(_ALL_PHASES) if value.strip() == "all" else [p.strip() for p in value.split(",") if p.strip()]
    unknown = set(phases) - set(_ALL_PHASES)
    if unknown:
        raise ValueError(f"Unknown phase(s): {sorted(unknown)}; valid={_ALL_PHASES}")
    return phases


def preflight(args, output_root: Path) -> dict:
    required = [
        _REPO_ROOT / "checkpoints" / "JSCC_model.pth",
        _REPO_ROOT / "checkpoints" / "diffusion_backbone.pth",
        _REPO_ROOT / "checkpoints" / "diffusion_controlnet.pth",
        _REPO_ROOT / "checkpoints" / "muge-epoch-19-checkpoint.pth",
        _REPO_ROOT / "data" / "etri_video_eval" / "manifest.csv",
        Path(args.image_input),
        Path(args.worker_python),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing required paths:\n  " + "\n  ".join(missing))
    if args.height <= 0 or args.width <= 0 or args.svd_steps <= 0 or args.wan_steps <= 0:
        raise RuntimeError("height/width/inference steps must all be positive")
    if args.height % 16 or args.width % 16:
        raise RuntimeError("Wan/SVD HQ height and width must be multiples of 16")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the current container")
    if torch.cuda.device_count() < len(args.devices):
        raise RuntimeError(
            f"Requested {len(args.devices)} GPU(s), but torch sees {torch.cuda.device_count()}"
        )

    worker_check = subprocess.run(
        [args.worker_python, "-c", (
            "import torch,diffusers,transformers,accelerate; "
            "from diffusers import WanImageToVideoPipeline,StableVideoDiffusionPipeline; "
            "print(torch.cuda.is_available(),torch.cuda.device_count(),diffusers.__version__)"
        )],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    if worker_check.returncode:
        raise RuntimeError(f"LGVSC worker environment failed:\n{worker_check.stderr}")

    disk = shutil.disk_usage(_REPO_ROOT)
    report = {
        "repo_root": str(_REPO_ROOT),
        "output_root": str(output_root),
        "python": sys.executable,
        "worker_python": args.worker_python,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "disk_free_gib": round(disk.free / 2**30, 2),
        "worker_check": worker_check.stdout.strip(),
        "hq": {
            "height": args.height, "width": args.width,
            "svd_steps": args.svd_steps, "wan_steps": args.wan_steps,
            "diffusion_step": args.diffusion_step,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if disk.free < 250 * 2**30:
        print(f"WARNING: only {report['disk_free_gib']} GiB free; monitor disk usage during Wan/SVD runs.")
    return report


def build_commands(args, output_root: Path, phases: List[str]) -> List[dict]:
    py = sys.executable
    image_csv = output_root / "image" / "kodak_snr_sweep.csv"
    video_root = output_root / "video_real_step50"
    baseline_root = video_root / "baseline"
    remeasure_root = output_root / "remeasure"
    generation_root = output_root / "generation"
    quality_root = output_root / "quality"
    common_video_filter = ["--videos", args.videos] if args.videos else []
    max_frames = ["--max-frames", str(args.max_frames)] if args.max_frames is not None else []
    resume = ["--skip-existing"] if args.skip_existing else []
    keep_going = ["--continue-on-error"] if args.continue_on_error else []
    devices_text = ",".join(args.devices)

    commands: List[dict] = []
    if "image" in phases:
        commands.append({"name": "image", "marker": image_csv, "cmd": [
            py, str(_SCRIPTS / "evaluate.py"),
            "--config", str(_REPO_ROOT / "configs" / "recipes" / "inference" / "composed.yaml"),
            "--input", str(Path(args.image_input).resolve()),
            "--output-csv", str(image_csv),
            f"--snr-list={args.snr_list}", "--device", args.devices[0], "--profile", "extended",
        ]})
    if "video" in phases:
        commands.append({"name": "video", "marker": video_root / "batch_status.json", "cmd": [
            py, str(_SCRIPTS / "run_etri_video_eval.py"),
            "--stages", "baseline", "--snr", str(args.video_snr),
            "--diffusion-step", str(args.diffusion_step),
            "--parallel", str(len(args.devices)), "--devices", devices_text,
            "--gpu-log-interval", "10", "--profile",
            "--output-root", str(video_root),
        ] + common_video_filter + max_frames + resume})
    if "remeasure" in phases:
        commands.append({"name": "remeasure", "marker": remeasure_root / "summary_metrics.csv", "cmd": [
            py, str(_SCRIPTS / "batch_remeasure_owlv2_vqa_10videos.py"),
            "--baseline-root", str(baseline_root), "--output-root", str(remeasure_root),
            "--device", args.devices[0],
        ] + common_video_filter + resume + keep_going})
    if "svd" in phases:
        commands.append({"name": "svd", "marker": generation_root / "batch_status.json", "cmd": [
            py, str(_SCRIPTS / "batch_lgvsc_1c_reproduce.py"),
            "--modes", "svd_start_only", "--device", args.devices[0],
            "--output-root", str(generation_root),
            "--worker-height", str(args.height), "--worker-width", str(args.width),
            "--worker-num-inference-steps", str(args.svd_steps),
            "--worker-decode-chunk-size", "1",
        ] + common_video_filter + max_frames + resume + keep_going})
    if "wan" in phases:
        commands.append({"name": "wan", "marker": generation_root / "batch_status.json", "cmd": [
            py, str(_SCRIPTS / "batch_lgvsc_1c_reproduce.py"),
            "--modes", "wan_skim_sfa,wan_skem_dsa", "--device", args.devices[0],
            "--output-root", str(generation_root),
            "--worker-height", str(args.height), "--worker-width", str(args.width),
            "--worker-num-inference-steps", str(args.wan_steps),
            "--worker-device-map", "balanced", "--worker-max-memory", args.worker_max_memory,
        ] + common_video_filter + max_frames + resume + keep_going})
    if "quality" in phases:
        commands.extend([
            {"name": "quality_video", "marker": quality_root / "video_summary.csv", "cmd": [
                py, str(_SCRIPTS / "evaluate_video_frame_quality.py"),
                "--run-root", str(baseline_root), "--kinds", "recon", "--device", args.devices[0],
                "--output-csv", str(quality_root / "video_frames.csv"),
                "--summary-csv", str(quality_root / "video_summary.csv"),
                "--summary-json", str(quality_root / "video_summary.json"),
            ]},
            {"name": "quality_generation", "marker": quality_root / "generation_summary.csv", "cmd": [
                py, str(_SCRIPTS / "evaluate_video_frame_quality.py"),
                "--run-root", str(generation_root), "--kinds", "auto", "--device", args.devices[0],
                "--output-csv", str(quality_root / "generation_frames.csv"),
                "--summary-csv", str(quality_root / "generation_summary.csv"),
                "--summary-json", str(quality_root / "generation_summary.json"),
            ]},
        ])
    return commands


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        phases = _selected_phases(args.phases)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_root = Path(args.output_root).resolve()

    if args.dry_run:
        report = {"dry_run": True, "phases": phases}
    else:
        report = preflight(args, output_root)
        print(json.dumps(report, indent=2))
    if args.preflight_only:
        return 0

    commands = build_commands(args, output_root, phases)
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "hq_validation_plan.json"
    plan = [{"name": item["name"], "cmd": item["cmd"], "marker": str(item["marker"])} for item in commands]
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Plan -> {plan_path}")

    env = os.environ.copy()
    env["SGDJSCC_LGVSC_WORKER_PYTHON"] = str(Path(args.worker_python).resolve())
    env["CUDA_VISIBLE_DEVICES"] = ",".join(device.split(":", 1)[1] for device in args.devices)
    env["HF_ENABLE_PARALLEL_LOADING"] = "YES"
    status: List[Dict] = []
    status_path = output_root / "hq_validation_status.json"

    for item in commands:
        command_text = shlex.join(item["cmd"])
        print(f"\n=== {item['name']} ===\n{command_text}", flush=True)
        if args.dry_run:
            status.append({"name": item["name"], "status": "dry_run", "cmd": command_text})
            continue
        if args.skip_existing and Path(item["marker"]).exists() and item["name"] in {
            "image", "quality_video", "quality_generation"
        }:
            status.append({"name": item["name"], "status": "skipped", "cmd": command_text})
            continue
        started = time.time()
        proc = subprocess.run(item["cmd"], cwd=_REPO_ROOT, env=env)
        row = {
            "name": item["name"], "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode, "duration_sec": round(time.time() - started, 2),
            "cmd": command_text,
        }
        status.append(row)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if proc.returncode and not args.continue_on_error:
            print(f"Stopped after {item['name']} failed. Resume with --skip-existing after fixing it.", file=sys.stderr)
            return proc.returncode

    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"\nHQ validation complete. Status -> {status_path}")
    return 1 if any(row["status"] == "failed" for row in status) else 0


if __name__ == "__main__":
    raise SystemExit(main())
