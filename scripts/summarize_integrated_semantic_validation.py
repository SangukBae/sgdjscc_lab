#!/usr/bin/env python
"""Fail-closed merge, paired effects and screening report for integrated eval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

POLICIES = ("full50", "few10", "vae_direct")
PROFILES = (
    "baseline", "combined_ds4",
    "candidate_edge_ds4_uncertainty_omit", "candidate_both_omit",
)
BASELINE = ("full50", "baseline")
EXPECTED_BACKENDS = ("clip", "owlv2", "vqa")
MARGINS = {
    "psnr_drop_db": 0.5,
    "ssim_drop": 0.01,
    "lpips_rise": 0.02,
    "closed_ptc_drop": 0.05,
    "closed_severity_rise": 0.05,
    "open_hallucination_rise": 0.05,
    "open_additional_per_frame_rise": 0.05,
    "closed_sfr_rise": 0.05,
    "closed_sdi_abs_change": 0.01,
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def f(row, key):
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"missing required metric {key} for {row.get('video')}/{row.get('decoder_policy')}/{row.get('guide_profile')}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric {key}: {value}")
    return number


def mean(values):
    return sum(values) / len(values)


def bootstrap_ci(values, *, seed=2025, iterations=5000):
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(iterations))
    return samples[int(0.025 * iterations)], samples[min(int(0.975 * iterations), iterations - 1)]


def _deltas(candidate, baseline):
    n = f(candidate, "closed_n_items")
    bn = f(baseline, "closed_n_items")
    return {
        "psnr_drop_db": f(baseline, "mean_psnr") - f(candidate, "mean_psnr"),
        "ssim_drop": f(baseline, "mean_ssim") - f(candidate, "mean_ssim"),
        "lpips_rise": f(candidate, "mean_lpips") - f(baseline, "mean_lpips"),
        "closed_ptc_drop": f(baseline, "closed_ptc") - f(candidate, "closed_ptc"),
        "closed_severity_rise": f(candidate, "closed_mean_severity") - f(baseline, "closed_mean_severity"),
        "open_hallucination_rise": f(candidate, "open_temporal_hallucination_rate") - f(baseline, "open_temporal_hallucination_rate"),
        "open_additional_per_frame_rise": (
            f(candidate, "open_total_additional_objects") / n
            - f(baseline, "open_total_additional_objects") / bn
        ),
        "closed_sfr_rise": f(candidate, "closed_sfr") - f(baseline, "closed_sfr"),
        "closed_sdi_abs_change": abs(f(candidate, "closed_sdi") - f(baseline, "closed_sdi")),
    }


def summarize(run_root: Path):
    plan = json.loads((run_root / "integrated_plan.json").read_text(encoding="utf-8"))
    expected_videos = sorted(video for a in plan["assignments"] for video in a["videos"])
    rows = []
    worker_summaries = []
    for assignment in plan["assignments"]:
        worker_root = run_root / "semantic" / "workers" / assignment["worker_id"]
        summary = json.loads((worker_root / "worker_summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise RuntimeError(f"semantic worker incomplete: {assignment['worker_id']}")
        worker_summaries.append(summary)
        rows.extend(read_csv(worker_root / "integrated_semantic_rows.csv"))

    by_key = {}
    for row in rows:
        key = (row["video"], row["decoder_policy"], row["guide_profile"])
        if key in by_key:
            raise RuntimeError(f"duplicate integrated row {key}")
        by_key[key] = row
    expected_keys = {
        (video, policy, profile)
        for video in expected_videos for policy in POLICIES for profile in PROFILES
    }
    if set(by_key) != expected_keys:
        raise RuntimeError(
            f"integrated coverage mismatch: missing={sorted(expected_keys-set(by_key))}, "
            f"unexpected={sorted(set(by_key)-expected_keys)}"
        )
    for row in rows:
        if int(float(row["n_frames"])) != 100:
            raise RuntimeError(f"expected 100 frames: {row['video']}/{row['decoder_policy']}/{row['guide_profile']}")
        for metric in (
            "mean_psnr", "mean_ssim", "mean_lpips", "total_bundle_bytes", "total_elapsed_s",
            "closed_mean_severity", "closed_ptc", "closed_sfr", "closed_sdi",
            "open_temporal_hallucination_rate", "open_total_additional_objects",
        ):
            f(row, metric)

    backend_totals = {name: 0 for name in EXPECTED_BACKENDS}
    for row in rows:
        for name in EXPECTED_BACKENDS:
            backend_totals[name] += int(float(row[f"closed_backend_{name}"]))
            backend_totals[name] += int(float(row[f"open_backend_{name}"]))
    missing_backend_evidence = [name for name, count in backend_totals.items() if count <= 0]
    if missing_backend_evidence:
        raise RuntimeError(f"no real calibration evidence from backends: {missing_backend_evidence}")

    write_csv(run_root / "integrated_per_video.csv", sorted(rows, key=lambda r: (
        r["decoder_policy"], r["guide_profile"], r["video"]
    )))
    effects = []
    for policy in POLICIES:
        for profile in PROFILES:
            pair_deltas = [
                _deltas(by_key[(video, policy, profile)], by_key[(video, *BASELINE)])
                for video in expected_videos
            ]
            row = {"decoder_policy": policy, "guide_profile": profile, "n_videos": len(expected_videos)}
            gates = []
            for index, (name, margin) in enumerate(MARGINS.items()):
                values = [item[name] for item in pair_deltas]
                low, high = bootstrap_ci(values, seed=2025 + index)
                value = mean(values)
                row[name] = value
                row[f"{name}_ci95_low"] = low
                row[f"{name}_ci95_high"] = high
                row[f"{name}_margin"] = margin
                gates.append(value <= margin)
            candidates = [by_key[(v, policy, profile)] for v in expected_videos]
            row["mean_total_bundle_bytes"] = mean([f(item, "total_bundle_bytes") for item in candidates])
            row["mean_total_elapsed_s"] = mean([f(item, "total_elapsed_s") for item in candidates])
            row["screening_gate_passed"] = all(gates)
            effects.append(row)

    passing = [row for row in effects if row["screening_gate_passed"]]
    selected = min(
        passing, key=lambda r: (r["mean_total_bundle_bytes"], r["mean_total_elapsed_s"])
    ) if passing else None
    for row in effects:
        row["selected_operating_point"] = bool(selected is row)
    write_csv(run_root / "integrated_effect.csv", effects)
    write_csv(run_root / "integrated_screening_frontier.csv", sorted(
        passing, key=lambda r: (r["mean_total_bundle_bytes"], r["mean_total_elapsed_s"])
    ))

    validation = {
        "validation_passed": bool(selected),
        "run_status": "completed" if selected else "completed_no_candidate_in_budget",
        "n_expected_pairs": len(expected_keys), "n_completed_pairs": len(rows),
        "n_videos": len(expected_videos), "policies": list(POLICIES),
        "guide_profiles": list(PROFILES), "baseline": {"decoder_policy": BASELINE[0], "guide_profile": BASELINE[1]},
        "screening_margins": MARGINS, "backend_evidence_counts": backend_totals,
        "worker_summaries": worker_summaries,
        "selected_operating_point": selected,
        "interpretation": (
            "Development-set provisional screening only. Paired 95% bootstrap CIs are reported; "
            "the later held-out run must confirm the final operating point."
        ),
    }
    (run_root / "integrated_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# Integrated semantic · hallucination · temporal evaluation

- status: `{validation['run_status']}`
- coverage: {len(rows)}/{len(expected_keys)} video-policy-profile pairs, 100 frames each
- fixed condition: fixed selector, int4 digital packet, fixed-reference SNR 10 dB, seed 2025
- baseline: `full50 + baseline`
- presence ensemble: CLIP + OWLv2 + VQA; evidence {backend_totals}
- closed-world preservation: GT vocabulary filter
- open-world hallucination: non-object noise filter without GT vocabulary restriction
- selected development operating point: `{None if selected is None else selected['decoder_policy'] + ' + ' + selected['guide_profile']}`

The screening margins are provisional development gates, not a final claim.
Every effect is paired by video and includes a 95% bootstrap confidence interval
in `integrated_effect.csv`. A separate held-out validation remains required.
"""
    (run_root / "INTEGRATED_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")
    hashes = {}
    for name in (
        "integrated_plan.json", "integrated_per_video.csv", "integrated_effect.csv",
        "integrated_screening_frontier.csv", "integrated_validation.json",
        "INTEGRATED_EVALUATION_REPORT.md",
    ):
        path = run_root / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run_root / "artifact_sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validation


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    validation = summarize(Path(args.run_root).resolve())
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0 if validation["validation_passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
