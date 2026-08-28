#!/usr/bin/env python
"""Validate a completed fixed-vs-SKEM actual-transmission matched-rate run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate(run_root: Path) -> Dict[str, Any]:
    plan_path = run_root / "parallel_plan.json"
    worker_path = run_root / "parallel_worker_status.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else {}
    workers = json.loads(worker_path.read_text(encoding="utf-8")) if worker_path.is_file() else []
    per_video = _rows(run_root / "per_video_metrics.csv")
    rate_rows = _rows(run_root / "rate_matching.csv")
    matched_plans = _rows(run_root / "matched_rate_plan.csv")
    failures = _rows(run_root / "failed_pairs.csv")

    assignments = plan.get("assignments", [])
    assignment_by_id = {item.get("worker_id"): item for item in assignments}
    worker_manifests = {}
    for assignment in assignments:
        worker_id = assignment.get("worker_id")
        path = run_root / "workers" / str(worker_id) / "run_manifest.json"
        if path.is_file():
            worker_manifests[worker_id] = json.loads(path.read_text(encoding="utf-8"))
    expected_videos = sorted({
        video for assignment in assignments for video in assignment.get("videos", [])
    })
    config_text = str(plan.get("settings", {}).get("configs", ""))
    expected_configs = sorted(item for item in config_text.split(",") if item)
    expected_pairs = {
        (video, config) for video in expected_videos for config in expected_configs
    }
    observed_pairs = {(row.get("video", ""), row.get("config", "")) for row in per_video}
    digital_channels = sorted({
        config.split("_", 1)[1]
        for config in expected_configs
        if config.startswith("fixed_") and config.split("_", 1)[1] != "awgn"
    })
    expected_rate_pairs = {
        (video, channel) for video in expected_videos for channel in digital_channels
    }
    observed_rate_pairs = {
        (row.get("video", ""), row.get("channel", "")) for row in rate_rows
    }

    checks: Dict[str, bool] = {
        "parallel_plan_present": bool(plan),
        "actual_transmission_mode_locked": (
            bool(plan.get("settings", {}).get("match_actual_transmissions"))
            and not bool(plan.get("settings", {}).get("match_fixed_keyframes"))
        ),
        "three_unique_gpu_workers": (
            len(assignments) == 3
            and len(set(plan.get("devices", []))) == 3
            and all(str(device).startswith("cuda:") for device in plan.get("devices", []))
        ),
        "worker_status_complete": (
            len(workers) == len(assignments) == 3
            and all(int(worker.get("returncode", -1)) == 0 for worker in workers)
            and {worker.get("worker_id") for worker in workers}
            == {assignment.get("worker_id") for assignment in assignments}
            and all(
                worker.get("device")
                == assignment_by_id.get(worker.get("worker_id"), {}).get("device")
                for worker in workers
            )
        ),
        "worker_manifest_gpu_provenance": (
            len(worker_manifests) == len(assignments) == 3
            and all(
                manifest.get("extra", {}).get("logical_device") == "cuda:0"
                and manifest.get("extra", {}).get("physical_cuda_device")
                == assignment_by_id[worker_id].get("device")
                and manifest.get("extra", {}).get("run_signature", {}).get(
                    "physical_cuda_device"
                ) == assignment_by_id[worker_id].get("device")
                for worker_id, manifest in worker_manifests.items()
            )
        ),
        "all_video_config_pairs_present_once": (
            len(per_video) == len(expected_pairs)
            and observed_pairs == expected_pairs
        ),
        "failed_pairs_zero": len(failures) == 0,
        "nonfinite_zero": (
            bool(per_video)
            and all(int(float(row.get("n_nan_or_inf_frames") or 0)) == 0 for row in per_video)
            and all(
                _finite(row.get(metric))
                for row in per_video for metric in ("mean_psnr", "mean_ssim", "mean_lpips")
            )
        ),
        "snr_10db_all_pairs": (
            bool(per_video)
            and all(
                row.get("digital_step_policy") == "fixed_reference"
                and _finite(row.get("fixed_reference_snr_db"))
                and abs(float(row["fixed_reference_snr_db"]) - 10.0) <= 1e-12
                for row in per_video
            )
        ),
        "one_exact_plan_per_video": (
            len(matched_plans) == len(expected_videos)
            and {row.get("video") for row in matched_plans} == set(expected_videos)
            and all(
                row.get("mode") == "actual_transmissions"
                and _bool(row.get("transmitting_frame_count_exact"))
                and int(float(row.get("target_n_transmitting_frames") or -1))
                == int(float(row.get("skem_planned_n_transmitting_frames") or -2))
                for row in matched_plans
            )
        ),
        "all_rate_rows_present_once": (
            len(rate_rows) == len(expected_rate_pairs)
            and observed_rate_pairs == expected_rate_pairs
        ),
        "actual_transmission_counts_exact": (
            bool(rate_rows)
            and all(
                row.get("matched_rate_mode") == "actual_transmissions"
                and _bool(row.get("transmitting_frame_count_matched"))
                and int(float(row.get("fixed_n_transmitting_frames") or -1))
                == int(float(row.get("skem_n_transmitting_frames") or -2))
                for row in rate_rows
            )
        ),
        "raw_byte_difference_within_1pct": (
            bool(rate_rows)
            and all(
                _finite(row.get("byte_diff_ratio"))
                and float(row["byte_diff_ratio"]) <= 0.01 + 1e-12
                and _bool(row.get("raw_rate_matched"))
                for row in rate_rows
            )
        ),
        "effective_bytes_exact_after_padding": (
            bool(rate_rows)
            and all(
                _bool(row.get("effective_bytes_exact"))
                and int(float(row.get("fixed_effective_total_bytes") or -1))
                == int(float(row.get("skem_effective_total_bytes") or -2))
                for row in rate_rows
            )
        ),
    }

    channel_summary = []
    for channel in digital_channels:
        selected = [row for row in rate_rows if row.get("channel") == channel]
        quality_pairs = []
        for video in expected_videos:
            fixed = next((
                row for row in per_video
                if row.get("video") == video and row.get("config") == f"fixed_{channel}"
            ), None)
            skem = next((
                row for row in per_video
                if row.get("video") == video and row.get("config") == f"skem_{channel}"
            ), None)
            if fixed is not None and skem is not None:
                quality_pairs.append((fixed, skem))

        def mean_pair(field: str, side: int):
            values = [float(pair[side][field]) for pair in quality_pairs if _finite(pair[side].get(field))]
            return sum(values) / len(values) if values else None

        fixed_psnr, skem_psnr = mean_pair("mean_psnr", 0), mean_pair("mean_psnr", 1)
        fixed_ssim, skem_ssim = mean_pair("mean_ssim", 0), mean_pair("mean_ssim", 1)
        fixed_lpips, skem_lpips = mean_pair("mean_lpips", 0), mean_pair("mean_lpips", 1)
        channel_summary.append({
            "channel": channel,
            "n_videos": len(selected),
            "fixed_mean_psnr": fixed_psnr,
            "skem_mean_psnr": skem_psnr,
            "psnr_delta_skem_minus_fixed": (
                skem_psnr - fixed_psnr if fixed_psnr is not None and skem_psnr is not None else None
            ),
            "fixed_mean_ssim": fixed_ssim,
            "skem_mean_ssim": skem_ssim,
            "ssim_delta_skem_minus_fixed": (
                skem_ssim - fixed_ssim if fixed_ssim is not None and skem_ssim is not None else None
            ),
            "fixed_mean_lpips": fixed_lpips,
            "skem_mean_lpips": skem_lpips,
            "lpips_delta_skem_minus_fixed": (
                skem_lpips - fixed_lpips
                if fixed_lpips is not None and skem_lpips is not None else None
            ),
            "mean_raw_byte_diff_ratio": (
                sum(float(row["byte_diff_ratio"]) for row in selected) / len(selected)
                if selected else None
            ),
            "max_raw_byte_diff_ratio": (
                max(float(row["byte_diff_ratio"]) for row in selected) if selected else None
            ),
            "fixed_padding_bytes_total": sum(
                int(float(row.get("fixed_padding_bytes") or 0)) for row in selected
            ),
            "skem_padding_bytes_total": sum(
                int(float(row.get("skem_padding_bytes") or 0)) for row in selected
            ),
        })

    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "validation_passed": not failed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
        "counts": {
            "expected_videos": len(expected_videos),
            "expected_configs": len(expected_configs),
            "expected_video_config_pairs": len(expected_pairs),
            "observed_video_config_pairs": len(per_video),
            "expected_rate_rows": len(expected_rate_pairs),
            "observed_rate_rows": len(rate_rows),
            "matched_rate_plan_rows": len(matched_plans),
            "failed_pairs": len(failures),
        },
        "devices": plan.get("devices", []),
        "worker_statuses": workers,
        "channel_summary": channel_summary,
    }


def write_outputs(run_root: Path, result: Dict[str, Any]) -> None:
    (run_root / "matched_rate_validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    quality_path = run_root / "matched_rate_quality_effect.csv"
    with quality_path.open("w", newline="", encoding="utf-8") as handle:
        rows = result["channel_summary"]
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Fixed–SKEM exact matched-rate validation",
        "",
        f"- Verdict: **{'PASS' if result['validation_passed'] else 'FAIL'}**",
        f"- Video-config pairs: {result['counts']['observed_video_config_pairs']} / "
        f"{result['counts']['expected_video_config_pairs']}",
        f"- Rate rows: {result['counts']['observed_rate_rows']} / "
        f"{result['counts']['expected_rate_rows']}",
        f"- Failed pairs: {result['counts']['failed_pairs']}",
        f"- GPUs: {', '.join(result['devices'])}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in result["checks"].items()
    )
    lines.extend(["", "## Per-channel raw byte matching", ""])
    for row in result["channel_summary"]:
        lines.append(
            f"- `{row['channel']}`: n={row['n_videos']}, "
            f"ΔPSNR={row['psnr_delta_skem_minus_fixed']!s}, "
            f"ΔSSIM={row['ssim_delta_skem_minus_fixed']!s}, "
            f"ΔLPIPS={row['lpips_delta_skem_minus_fixed']!s}, "
            f"mean diff={row['mean_raw_byte_diff_ratio']!s}, "
            f"max diff={row['max_raw_byte_diff_ratio']!s}, "
            f"fixed padding={row['fixed_padding_bytes_total']}, "
            f"SKEM padding={row['skem_padding_bytes_total']} bytes"
        )
    if result["failed_checks"]:
        lines.extend(["", "Failed checks: " + ", ".join(result["failed_checks"])])
    (run_root / "MATCHED_RATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.run_root).resolve()
    result = validate(root)
    write_outputs(root, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["validation_passed"] else 4


if __name__ == "__main__":
    raise SystemExit(run())
