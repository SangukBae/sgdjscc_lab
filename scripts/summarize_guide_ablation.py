#!/usr/bin/env python
"""Validate and summarize edge/uncertainty transport ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sgdjscc_lab.transmission.guide_transport import (  # noqa: E402
    GUIDE_PROFILES,
    guide_actions,
)

QUALITY_GATE = {"psnr_drop_db": 0.5, "ssim_drop": 0.01, "lpips_rise": 0.02}
COMPONENT_FIELDS = (
    "caption_bytes", "edge_bytes", "edge_uncertainty_bytes", "manifest_bytes",
    "semantic_packet_bytes", "visual_bytes", "bundle_overhead_bytes",
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _float(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def _int(row: Dict[str, Any], key: str) -> int:
    return int(float(row.get(key) or 0))


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _expected_profiles(run_root: Path, aggregate_rows: List[Dict[str, str]]) -> List[str]:
    plan_path = run_root / "parallel_plan.json"
    signature_path = run_root / "run_signature.json"
    values = None
    if plan_path.is_file():
        values = json.loads(plan_path.read_text(encoding="utf-8")).get("settings", {}).get(
            "guide_profiles"
        )
    elif signature_path.is_file():
        values = json.loads(signature_path.read_text(encoding="utf-8")).get("guide_profiles")
    if isinstance(values, str):
        values = [item for item in values.split(",") if item]
    if values:
        return [str(item) for item in values]
    return sorted({row.get("guide_profile", "") for row in aggregate_rows if row.get("guide_profile")})


def _expected_videos(run_root: Path, per_video_rows: List[Dict[str, str]]) -> List[str]:
    plan_path = run_root / "parallel_plan.json"
    signature_path = run_root / "run_signature.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        return sorted({
            str(video)
            for assignment in plan.get("assignments", [])
            for video in assignment.get("videos", [])
        })
    if signature_path.is_file():
        values = json.loads(signature_path.read_text(encoding="utf-8")).get("video_keys", [])
        if values:
            return sorted(str(value) for value in values)
    return sorted({row.get("video", "") for row in per_video_rows if row.get("video")})


def _component_rows(packet_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in packet_rows:
        profile = row.get("guide_profile") or "baseline"
        target = grouped.setdefault(profile, {
            "guide_profile": profile,
            "n_packet_rows": 0,
            **{field: 0 for field in COMPONENT_FIELDS},
            "total_bundle_bytes": 0,
            "edge_transmit_rows": 0,
            "edge_reuse_rows": 0,
            "edge_zero_rows": 0,
            "uncertainty_transmit_rows": 0,
            "uncertainty_reuse_rows": 0,
            "uncertainty_zero_rows": 0,
        })
        target["n_packet_rows"] += 1
        for field in COMPONENT_FIELDS:
            target[field] += _int(row, field)
        target["total_bundle_bytes"] += _int(row, "total_bundle_bytes")
        for component, field in (("edge", "edge_action"), ("uncertainty", "uncertainty_action")):
            action = str(row.get(field, "")).strip()
            if action in {"0", "1", "2"}:
                name = {"0": "transmit", "1": "reuse", "2": "zero"}[action]
                target[f"{component}_{name}_rows"] += 1
    return [grouped[key] for key in sorted(grouped)]


def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    no_worse = (
        a["mean_total_bundle_bytes_per_video"] <= b["mean_total_bundle_bytes_per_video"]
        and a["mean_psnr"] >= b["mean_psnr"]
        and a["mean_ssim"] >= b["mean_ssim"]
        and a["mean_lpips"] <= b["mean_lpips"]
    )
    strict = (
        a["mean_total_bundle_bytes_per_video"] < b["mean_total_bundle_bytes_per_video"]
        or a["mean_psnr"] > b["mean_psnr"]
        or a["mean_ssim"] > b["mean_ssim"]
        or a["mean_lpips"] < b["mean_lpips"]
    )
    return no_worse and strict


def _guide_actions_match_protocol(packet_rows: List[Dict[str, str]]) -> bool:
    grouped: Dict[tuple[str, str], List[Dict[str, str]]] = {}
    for row in packet_rows:
        if str(row.get("edge_action", "")).strip() not in {"0", "1", "2"}:
            continue
        key = (row.get("video", ""), row.get("guide_profile", ""))
        grouped.setdefault(key, []).append(row)
    for (_video, profile_name), rows in grouped.items():
        if profile_name not in GUIDE_PROFILES:
            return False
        ordered = sorted(rows, key=lambda row: _int(row, "frame_index"))
        for ordinal, row in enumerate(ordered):
            expected_edge, expected_uncertainty = guide_actions(
                GUIDE_PROFILES[profile_name], ordinal,
            )
            actual_edge = _int(row, "edge_action")
            actual_uncertainty = _int(row, "uncertainty_action")
            if (actual_edge, actual_uncertainty) != (expected_edge, expected_uncertainty):
                return False
            if (actual_edge == 0) != (_int(row, "edge_bytes") > 0):
                return False
            if (actual_uncertainty == 0) != (_int(row, "edge_uncertainty_bytes") > 0):
                return False
    return bool(grouped)


def _parallel_provenance(run_root: Path) -> Dict[str, Any]:
    plan_path = run_root / "parallel_plan.json"
    if not plan_path.is_file():
        return {"applicable": False, "passed": True}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    status_path = run_root / "parallel_worker_status.json"
    statuses = (
        json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else []
    )
    assignments = plan.get("assignments", [])
    status_by_worker = {row.get("worker_id"): row for row in statuses}
    manifests = []
    for assignment in assignments:
        worker_id = assignment.get("worker_id")
        path = run_root / "workers" / str(worker_id) / "run_manifest.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            manifests.append({
                "worker_id": worker_id,
                "planned_device": assignment.get("device"),
                "recorded_device": payload.get("extra", {}).get("physical_cuda_device"),
                "git": payload.get("git", {}),
            })
    expected_commit = plan.get("git", {}).get("commit")
    passed = (
        plan.get("devices") == ["cuda:0", "cuda:1", "cuda:2"]
        and len(assignments) == len(statuses) == len(manifests) == 3
        and all(
            status_by_worker.get(item.get("worker_id"), {}).get("device") == item.get("device")
            and status_by_worker.get(item.get("worker_id"), {}).get("returncode") == 0
            for item in assignments
        )
        and all(
            item["planned_device"] == item["recorded_device"]
            and item["git"].get("commit") == expected_commit
            and item["git"].get("dirty") is False
            for item in manifests
        )
    )
    return {
        "applicable": True,
        "passed": passed,
        "planned_devices": plan.get("devices"),
        "worker_statuses": statuses,
        "worker_manifests": manifests,
    }


def summarize(run_root: Path) -> Dict[str, Any]:
    aggregate_rows = _read_csv(run_root / "aggregate.csv")
    per_video_rows = _read_csv(run_root / "per_video_metrics.csv")
    packet_rows = _read_csv(run_root / "packet_components.csv")
    failed_rows = _read_csv(run_root / "failed_pairs.csv")
    expected_profiles = _expected_profiles(run_root, aggregate_rows)

    baseline_candidates = [
        row for row in aggregate_rows if row.get("guide_profile") == "baseline"
    ]
    baseline = baseline_candidates[0] if len(baseline_candidates) == 1 else None
    effect_rows: List[Dict[str, Any]] = []
    if baseline is not None:
        baseline_bytes = _float(baseline, "mean_total_bundle_bytes_per_video")
        baseline_psnr = _float(baseline, "mean_psnr")
        baseline_ssim = _float(baseline, "mean_ssim")
        baseline_lpips = _float(baseline, "mean_lpips")
        for row in aggregate_rows:
            profile = row.get("guide_profile") or "baseline"
            current_bytes = _float(row, "mean_total_bundle_bytes_per_video")
            psnr = _float(row, "mean_psnr")
            ssim = _float(row, "mean_ssim")
            lpips = _float(row, "mean_lpips")
            psnr_drop = baseline_psnr - psnr
            ssim_drop = baseline_ssim - ssim
            lpips_rise = lpips - baseline_lpips
            complete = (
                _bool(row.get("all_expected_videos_present"))
                and _bool(row.get("all_finite_metrics"))
                and _int(row, "total_nan_or_inf_frames") == 0
                and math.isclose(_float(row, "valid_frame_ratio"), 1.0)
            )
            within_gate = (
                complete
                and psnr_drop <= QUALITY_GATE["psnr_drop_db"]
                and ssim_drop <= QUALITY_GATE["ssim_drop"]
                and lpips_rise <= QUALITY_GATE["lpips_rise"]
            )
            effect_rows.append({
                "config": row["config"],
                "guide_profile": profile,
                "guide_family": row.get("guide_family", ""),
                "guide_stage": row.get("guide_stage", ""),
                "n_videos": _int(row, "n_videos"),
                "mean_psnr": psnr,
                "mean_ssim": ssim,
                "mean_lpips": lpips,
                "mean_total_bundle_bytes_per_video": current_bytes,
                "psnr_drop_db": psnr_drop,
                "ssim_drop": ssim_drop,
                "lpips_rise": lpips_rise,
                "byte_reduction": baseline_bytes - current_bytes,
                "byte_reduction_ratio": (
                    (baseline_bytes - current_bytes) / baseline_bytes if baseline_bytes else 0.0
                ),
                "complete_and_finite": complete,
                "within_quality_gate": within_gate,
                "edge_bit_depth": row.get("edge_bit_depth", ""),
                "uncertainty_bit_depth": row.get("uncertainty_bit_depth", ""),
                "edge_downsample": row.get("edge_downsample", ""),
                "uncertainty_downsample": row.get("uncertainty_downsample", ""),
                "edge_stride": row.get("edge_stride", ""),
                "uncertainty_stride": row.get("uncertainty_stride", ""),
                "edge_omit": row.get("edge_omit", ""),
                "uncertainty_omit": row.get("uncertainty_omit", ""),
            })

    component_rows = _component_rows(packet_rows)
    eligible = [row for row in effect_rows if _bool(row["within_quality_gate"])]
    pareto_rows: List[Dict[str, Any]] = []
    for row in eligible:
        candidate = dict(row)
        candidate["on_pareto_frontier"] = not any(
            other is not row and _dominates(other, row) for other in eligible
        )
        pareto_rows.append(candidate)
    pareto_rows.sort(key=lambda row: row["mean_total_bundle_bytes_per_video"])
    for index, row in enumerate(pareto_rows):
        row["rank_by_bytes"] = index + 1
        row["selected_minimum_bytes_in_gate"] = index == 0

    expected_videos = _expected_videos(run_root, per_video_rows)
    expected_pairs = len(expected_videos) * len(expected_profiles)
    observed_pairs = {(row.get("video"), row.get("guide_profile")) for row in per_video_rows}
    observed_profiles = sorted({row.get("guide_profile", "") for row in per_video_rows})
    fixed_reference_values = {
        float(row["fixed_reference_snr_db"])
        for row in per_video_rows if row.get("fixed_reference_snr_db") not in ("", None)
    }
    component_reconciles = all(
        sum(_int(row, field) for field in COMPONENT_FIELDS)
        == _int(row, "total_bundle_bytes")
        for row in packet_rows
    )
    parallel_provenance = _parallel_provenance(run_root)
    checks = {
        "one_baseline_aggregate_row": len(baseline_candidates) == 1,
        "expected_profiles_present": sorted(expected_profiles) == observed_profiles,
        "all_video_profile_pairs_present": len(observed_pairs) == expected_pairs,
        "no_failed_pairs": len(failed_rows) == 0,
        "all_rows_finite_and_complete": bool(effect_rows) and all(
            _bool(row["complete_and_finite"]) for row in effect_rows
        ),
        "fixed_reference_snr_exactly_10db": fixed_reference_values == {10.0},
        "packet_component_bytes_reconcile": component_reconciles,
        "guide_actions_match_profile_and_item_presence": _guide_actions_match_protocol(
            packet_rows
        ),
        "all_profiles_known": all(name in GUIDE_PROFILES for name in observed_profiles),
        "baseline_fixed_int4": bool(baseline) and str(baseline.get("config", "")).startswith(
            "fixed_int4__baseline"
        ),
        "three_gpu_worker_provenance": parallel_provenance["passed"],
    }
    validation = {
        "schema_version": 1,
        "run_root": str(run_root),
        "quality_gate": QUALITY_GATE,
        "expected_profiles": expected_profiles,
        "observed_profiles": observed_profiles,
        "n_expected_videos": len(expected_videos),
        "n_expected_pairs": expected_pairs,
        "n_observed_pairs": len(observed_pairs),
        "n_failed_pairs": len(failed_rows),
        "fixed_reference_snr_db_values": sorted(fixed_reference_values),
        "parallel_provenance": parallel_provenance,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "validation_passed": all(checks.values()),
        "selected_minimum_bytes_in_gate": (
            pareto_rows[0]["guide_profile"] if pareto_rows else None
        ),
    }

    _write_csv(run_root / "guide_ablation_effect.csv", effect_rows)
    _write_csv(run_root / "guide_component_bytes.csv", component_rows)
    _write_csv(run_root / "guide_pareto_frontier.csv", pareto_rows)
    (run_root / "guide_ablation_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# Edge·uncertainty ablation validation — {run_root.name}

- 검증 상태: `{'PASS' if validation['validation_passed'] else 'FAIL'}`
- 영상: {validation['n_expected_videos']}개
- profile: {len(expected_profiles)}개
- 완료 pair: {validation['n_observed_pairs']}/{validation['n_expected_pairs']}
- 실패 pair: {validation['n_failed_pairs']}개
- fixed-reference SNR: {validation['fixed_reference_snr_db_values']}
- 품질 gate: PSNR 하락 ≤ {QUALITY_GATE['psnr_drop_db']} dB, SSIM 하락 ≤ {QUALITY_GATE['ssim_drop']}, LPIPS 증가 ≤ {QUALITY_GATE['lpips_rise']}
- gate 내 최소 byte profile: `{validation['selected_minimum_bytes_in_gate']}`
- failed checks: {validation['failed_checks'] or '없음'}

`guide_ablation_effect.csv`는 `fixed_int4__baseline` 대비 품질·byte 변화를,
`guide_component_bytes.csv`는 실제 직렬화 bundle 구성과 transmit/reuse/zero 횟수를,
`guide_pareto_frontier.csv`는 품질 gate 내 다목적 Pareto 후보를 기록한다.
"""
    (run_root / "GUIDE_ABLATION_REPORT.md").write_text(report, encoding="utf-8")
    return validation


def run(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    validation = summarize(Path(args.run_root))
    return 0 if validation["validation_passed"] else 4


if __name__ == "__main__":
    raise SystemExit(run())
