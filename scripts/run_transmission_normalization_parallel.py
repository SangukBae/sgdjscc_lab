#!/usr/bin/env python
"""Safely distribute transmission-normalization videos across multiple GPUs.

Every GPU writes to an independent worker directory.  Only after all workers
finish does the parent process merge tabular results and regenerate aggregate,
Pareto, effect-summary, and manifest artifacts.  No two processes ever append
to the same CSV or manifest.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sgdjscc_lab.utils import run_manifest as rm  # noqa: E402

DEFAULT_CONFIGS = (
    "fixed_awgn,fixed_float32,fixed_int16,fixed_int8,fixed_int6,fixed_int4,"
    "skem_float32,skem_int16,skem_int8,skem_int6,skem_int4"
)
MERGED_CSV_KEYS = {
    "per_video_metrics.csv": ("video", "config"),
    "keyframe_selection.csv": ("video", "config", "frame_index"),
    "packet_components.csv": ("video", "config", "frame_index"),
    "keyframe_sweep.csv": ("video", "threshold", "max_segment_length"),
    "failed_pairs.csv": ("video", "config"),
    "source_size_report.csv": ("video",),
    "quantization_diagnostics.csv": ("video", "config", "frame_index", "patch_index"),
    "matched_rate_plan.csv": ("video",),
}
PARENT_ARTIFACTS = (
    "parallel_plan.json", "parallel_worker_status.json", "per_video_metrics.csv",
    "aggregate.csv", "pareto_frontier.csv", "rate_matching.csv", "matched_rate_plan.csv",
    "matched_rate_quality_effect.csv", "matched_rate_validation.json", "MATCHED_RATE_REPORT.md",
    "keyframe_selection.csv", "packet_components.csv", "keyframe_sweep.csv",
    "failed_pairs.csv", "source_size_report.csv", "quantization_diagnostics.csv",
    "quantization_effect.csv", "selector_effect.csv",
    "quantization_effect_ablation.csv", "selector_effect_ablation.csv",
    "normalization_effect_summary.json", "summary.json", "README.md",
)


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run transmission normalization on independent GPU workers and merge safely.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--dataset-root", default=str(_REPO_ROOT / "data/etri_video_eval"))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", default=None, help="Resume an existing parallel output root.")
    parser.add_argument("--video-ids", default=None)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--digital-step-policy", default="fixed_reference",
        choices=("fixed_reference", "bitdepth_proxy", "quant_nmse"),
    )
    parser.add_argument("--fixed-reference-snr-db", type=float, default=10.0)
    parser.add_argument("--ablation-label", default=None)
    parser.add_argument("--psss-backend", default="proxy", choices=("mock", "proxy", "real"))
    parser.add_argument("--psss-model-id", default=None)
    parser.add_argument("--psss-device", default="cpu")
    parser.add_argument("--psss-dtype", default="fp32")
    parser.add_argument("--psss-threshold", type=float, default=None)
    parser.add_argument("--psss-max-segment-length", type=int, default=None)
    parser.add_argument("--use-scene-detector", action="store_true")
    parser.add_argument("--no-match-fixed-keyframes", action="store_true")
    parser.add_argument("--match-actual-transmissions", action="store_true")
    parser.add_argument(
        "--matched-rate-thresholds",
        default=",".join(str(value) for value in (
            tuple(round(-0.95 + 0.05 * index, 10) for index in range(39)) + (0.999999,)
        )),
    )
    parser.add_argument(
        "--matched-rate-max-segment-lengths",
        default="8,10,12,14,16,20,24,32,48,64,100",
    )
    parser.add_argument("--skip-keyframe-sweep", action="store_true")
    parser.add_argument("--skip-source-size-report", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _read_video_manifest(dataset_root: Path) -> List[Dict[str, Any]]:
    path = dataset_root / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    videos = []
    for row in rows:
        key = f"{row['id']}_{row['name']}"
        videos.append({"key": key, "n_frames": int(row.get("n_frames") or 1)})
    return videos


def assign_videos(videos: Sequence[Dict[str, Any]], devices: Sequence[str]) -> List[Dict[str, Any]]:
    """Deterministic longest-first balancing by manifest frame count."""
    workers = [{"device": device, "videos": [], "estimated_frames": 0} for device in devices]
    for video in sorted(videos, key=lambda item: (-int(item["n_frames"]), str(item["key"]))):
        worker = min(workers, key=lambda item: (item["estimated_frames"], devices.index(item["device"])))
        worker["videos"].append(str(video["key"]))
        worker["estimated_frames"] += int(video["n_frames"])
    return workers


def _validate_devices(text: str) -> List[str]:
    devices = [item.strip() for item in text.split(",") if item.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain one or more unique device names")
    if any(not item.startswith("cuda:") for item in devices):
        raise ValueError("parallel normalization requires CUDA devices such as cuda:0,cuda:1")
    return devices


def _atomic_json(path: Path, value: Any) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _plan(args: argparse.Namespace, devices: Sequence[str], videos: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    state = rm.get_git_state(_REPO_ROOT)
    if state["commit"] == rm.UNKNOWN:
        raise RuntimeError("git commit is unknown; parallel experiment provenance is unsafe")
    if state["dirty"] is True:
        raise RuntimeError("tracked checkout is dirty; commit the changes before running")
    assignments = assign_videos(videos, devices)
    for index, worker in enumerate(assignments):
        worker["worker_id"] = f"worker_{index:02d}"
    return {
        "schema_version": 1,
        "git": state,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_manifest_sha256": rm.sha256_file(Path(args.dataset_root) / "manifest.csv"),
        "devices": list(devices),
        "assignments": assignments,
        "settings": {
            "configs": args.configs,
            "max_frames": args.max_frames,
            "seed": args.seed,
            "digital_step_policy": args.digital_step_policy,
            "fixed_reference_snr_db": args.fixed_reference_snr_db,
            "ablation_label": args.ablation_label,
            "psss_backend": args.psss_backend,
            "psss_model_id": args.psss_model_id,
            "psss_device": args.psss_device,
            "psss_dtype": args.psss_dtype,
            "psss_threshold": args.psss_threshold,
            "psss_max_segment_length": args.psss_max_segment_length,
            "use_scene_detector": args.use_scene_detector,
            "match_fixed_keyframes": (
                not args.no_match_fixed_keyframes and not args.match_actual_transmissions
            ),
            "match_actual_transmissions": args.match_actual_transmissions,
            "matched_rate_thresholds": args.matched_rate_thresholds,
            "matched_rate_max_segment_lengths": args.matched_rate_max_segment_lengths,
            "skip_keyframe_sweep": args.skip_keyframe_sweep,
            "skip_source_size_report": args.skip_source_size_report,
        },
    }


def _check_or_write_plan(output_root: Path, plan: Dict[str, Any]) -> None:
    path = output_root / "parallel_plan.json"
    if not path.exists():
        _atomic_json(path, plan)
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != plan:
        raise RuntimeError(
            f"parallel resume plan mismatch at {path}; devices, video assignment, commit, "
            "or experiment settings changed"
        )


def _worker_command(
    args: argparse.Namespace, worker: Dict[str, Any], output_root: Path, *, preflight_only: bool = False,
) -> List[str]:
    command = [
        "bash", "scripts/run_transmission_normalization.sh",
        # Each process sees exactly one physical GPU, remapped by
        # CUDA_VISIBLE_DEVICES to logical cuda:0. This contains upstream code
        # that hard-codes cuda:0 and prevents cross-device tensor creation.
        "--device", "cuda:0", "--dataset-root", str(Path(args.dataset_root).resolve()),
    ]
    if preflight_only:
        return [*command, "--preflight-only"]
    command.extend([
        "--output-root", str(output_root / "workers" / worker["worker_id"]),
        "--video-ids", ",".join(worker["videos"]),
        "--configs", args.configs,
        "--seed", str(args.seed),
        "--digital-step-policy", args.digital_step_policy,
        "--fixed-reference-snr-db", str(args.fixed_reference_snr_db),
        "--psss-backend", args.psss_backend,
        "--psss-device", args.psss_device,
        "--psss-dtype", args.psss_dtype,
    ])
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.ablation_label:
        command.extend(["--ablation-label", args.ablation_label])
    if args.psss_model_id:
        command.extend(["--psss-model-id", args.psss_model_id])
    if args.psss_threshold is not None:
        command.extend(["--psss-threshold", str(args.psss_threshold)])
    if args.psss_max_segment_length is not None:
        command.extend(["--psss-max-segment-length", str(args.psss_max_segment_length)])
    if args.use_scene_detector:
        command.append("--use-scene-detector")
    if args.match_actual_transmissions:
        command.extend([
            "--no-match-fixed-keyframes", "--match-actual-transmissions",
            "--matched-rate-thresholds", args.matched_rate_thresholds,
            "--matched-rate-max-segment-lengths", args.matched_rate_max_segment_lengths,
        ])
    elif args.no_match_fixed_keyframes:
        command.append("--no-match-fixed-keyframes")
    if args.skip_keyframe_sweep:
        command.append("--skip-keyframe-sweep")
    if args.skip_source_size_report:
        command.append("--skip-source-size-report")
    if args.retry_failed:
        command.append("--retry-failed")
    return command


def _worker_environment(worker: Dict[str, Any]) -> Dict[str, str]:
    physical_index = str(worker["device"]).split(":", 1)[1]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = physical_index
    environment["SGDJSCC_PHYSICAL_CUDA_DEVICE"] = str(worker["device"])
    return environment


def _deduplicate(rows: Iterable[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in keys)
        if key in by_key and by_key[key] != row:
            raise RuntimeError(f"conflicting duplicate merged row for key={key}")
        by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def merge_worker_outputs(
    output_root: Path, plan: Dict[str, Any], worker_statuses: Sequence[Dict[str, Any]], argv: Sequence[str],
) -> Dict[str, Any]:
    driver = _load_script("_txnorm_parallel_driver", "run_transmission_reduction_eval.py")
    summarizer = _load_script("_txnorm_parallel_summarizer", "summarize_transmission_normalization.py")
    worker_roots = [output_root / "workers" / item["worker_id"] for item in plan["assignments"]]

    merged: Dict[str, List[Dict[str, Any]]] = {}
    for filename, keys in MERGED_CSV_KEYS.items():
        rows: List[Dict[str, Any]] = []
        for root in worker_roots:
            rows.extend(driver._read_csv_dicts(root / filename))
        merged[filename] = _deduplicate(rows, keys)
        driver._write_csv(output_root / filename, merged[filename])

    per_video_rows = [driver._coerce_per_video_row(row) for row in merged["per_video_metrics.csv"]]
    expected_videos = {
        video for worker in plan["assignments"] for video in worker["videos"]
    }
    aggregate_rows = driver._aggregate(per_video_rows, expected_video_keys=expected_videos)
    driver._write_csv(output_root / "aggregate.csv", aggregate_rows)
    pareto_rows, baseline_info = driver._pareto_frontier(aggregate_rows)
    driver._write_csv(output_root / "pareto_frontier.csv", pareto_rows)
    rate_rows = driver._compute_rate_matching(per_video_rows)
    driver._write_csv(output_root / "rate_matching.csv", rate_rows)
    summarizer.run(["--run-root", str(output_root)])

    failed_pairs = merged["failed_pairs.csv"]
    matched_rate_validation = None
    if plan.get("settings", {}).get("match_actual_transmissions"):
        validator = _load_script(
            "_fixed_skem_matched_rate_validator", "validate_fixed_skem_matched_rate.py"
        )
        matched_rate_validation = validator.validate(output_root)
        validator.write_outputs(output_root, matched_rate_validation)
    run_status = (
        "completed_with_failures" if failed_pairs else
        "completed_validation_failed"
        if matched_rate_validation is not None
        and not matched_rate_validation["validation_passed"] else
        "completed"
    )
    summary = {
        "run_status": run_status,
        "parallel": True,
        "devices": plan["devices"],
        "n_workers": len(plan["assignments"]),
        "n_expected_videos": len(expected_videos),
        "n_completed_pairs": len(per_video_rows),
        "n_failed_pairs": len(failed_pairs),
        "pareto_baseline": baseline_info,
        "pareto_selected": next(
            (row for row in pareto_rows if row.get("selected_as_smallest_in_budget")), None
        ),
        "worker_statuses": list(worker_statuses),
        "matched_rate_validation": matched_rate_validation,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _atomic_json(output_root / "summary.json", summary)
    readme = f"""# Parallel transmission normalization — {output_root.name}

- 상태
  - `{run_status}`
  - GPU worker: {len(plan['assignments'])}개 (`{', '.join(plan['devices'])}`)
  - 완료 pair: {len(per_video_rows)}개
  - 실패 pair: {len(failed_pairs)}개
  - exact matched-rate 검증: {"해당 없음" if matched_rate_validation is None else ("PASS" if matched_rate_validation["validation_passed"] else "FAIL — matched_rate_validation.json 확인")}
- 안전성
  - worker별 독립 디렉터리: `workers/worker_NN/`
  - 공용 CSV 동시 쓰기 없음
  - 모든 worker 종료 후 상위 CSV·Pareto·effect·manifest 생성
  - 재개 시 `parallel_plan.json`과 다른 commit/device/영상 배분/설정은 거부
- 대용량 산출물
  - packet과 복원 영상은 복사하지 않고 각 worker 디렉터리에 보존
  - 상위 디렉터리는 병합된 표와 재현성 metadata만 제공
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    worker_manifests = {}
    for root in worker_roots:
        path = root / "run_manifest.json"
        worker_manifests[str(root.relative_to(output_root))] = rm.sha256_file(path)
    artifact_hashes = {
        name: rm.sha256_file(output_root / name)
        for name in PARENT_ARTIFACTS if (output_root / name).is_file()
    }
    manifest = rm.build_run_manifest(
        run_id=output_root.name,
        command_argv=argv,
        command_source="captured",
        seed=plan["settings"]["seed"],
        dataset_ref=plan["dataset_root"],
        dataset_hash=plan["dataset_manifest_sha256"],
        include_environment=False,
        repo_root=_REPO_ROOT,
        exact_fields=["worker packet bytes", "merged bytes/video", "merged bytes/frame"],
        proxy_fields=["estimated channel symbols", "estimated FEC wire bytes"],
        nan_or_failure_counts={"n_failed_pairs": len(failed_pairs)},
        extra={
            "run_status": run_status,
            "parallel_plan": plan,
            "worker_manifest_sha256": worker_manifests,
            "output_artifact_sha256": artifact_hashes,
        },
    )
    rm.write_run_manifest(output_root / "run_manifest.json", manifest)
    return summary


def run(argv=None) -> int:
    args = _parse_args(argv)
    if args.output_root and args.resume:
        raise ValueError("pass at most one of --output-root / --resume")
    if args.digital_step_policy != "fixed_reference" and not args.ablation_label:
        raise ValueError("non-default digital step policy requires --ablation-label")
    if args.psss_backend == "real" and not args.psss_model_id:
        raise ValueError("--psss-backend real requires --psss-model-id")

    devices = _validate_devices(args.devices)
    videos = _read_video_manifest(Path(args.dataset_root))
    if args.video_ids:
        requested = [item for item in args.video_ids.split(",") if item]
        available = {item["key"] for item in videos}
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(f"unknown --video-ids: {missing}")
        videos = [item for item in videos if item["key"] in set(requested)]
    if not videos:
        raise ValueError("no videos selected")
    if len(devices) > len(videos):
        devices = devices[:len(videos)]

    plan = _plan(args, devices, videos)
    output_root = Path(
        args.resume or args.output_root
        or (_REPO_ROOT / "outputs" / f"transmission_normalization_parallel_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    if not output_root.is_absolute():
        output_root = (_REPO_ROOT / output_root).resolve()
    commands = [
        _worker_command(args, worker, output_root) for worker in plan["assignments"]
    ]

    if args.dry_run:
        print(json.dumps({
            "output_root": str(output_root), "plan": plan, "commands": commands,
            "cuda_visible_devices": [
                _worker_environment(worker)["CUDA_VISIBLE_DEVICES"]
                for worker in plan["assignments"]
            ],
        }, indent=2))
        return 0

    for worker in plan["assignments"]:
        subprocess.run(
            _worker_command(args, worker, output_root, preflight_only=True),
            check=True, env=_worker_environment(worker),
        )
    if args.preflight_only:
        print(f"parallel preflight passed for {', '.join(devices)}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    _check_or_write_plan(output_root, plan)
    processes = []
    for worker, command in zip(plan["assignments"], commands):
        log_path = output_root / f"{worker['worker_id']}.log"
        handle = log_path.open("a", encoding="utf-8")
        print(f"starting {worker['worker_id']} on {worker['device']}: {','.join(worker['videos'])}")
        processes.append((
            worker, handle,
            subprocess.Popen(
                command, stdout=handle, stderr=subprocess.STDOUT,
                env=_worker_environment(worker),
            ),
        ))

    statuses = []
    for worker, handle, process in processes:
        returncode = process.wait()
        handle.close()
        statuses.append({
            "worker_id": worker["worker_id"], "device": worker["device"],
            "videos": worker["videos"], "returncode": returncode,
        })
        print(f"finished {worker['worker_id']} returncode={returncode}")
    _atomic_json(output_root / "parallel_worker_status.json", statuses)

    unexpected = [item for item in statuses if item["returncode"] not in (0, 3)]
    if unexpected:
        print(f"parallel run failed before safe merge: {unexpected}", file=sys.stderr)
        return 1
    summary = merge_worker_outputs(output_root, plan, statuses, sys.argv if argv is None else [sys.argv[0], *argv])
    print(json.dumps(summary, indent=2))
    if summary["run_status"] == "completed_with_failures":
        return 3
    if summary["run_status"] == "completed_validation_failed":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
