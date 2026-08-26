#!/usr/bin/env python
"""summarize_transmission_normalization.py – quantization vs. selector effect tables.

Reads ``aggregate.csv`` (and, for keyframe-count context, ``per_video_metrics.csv``)
written by ``scripts/run_transmission_reduction_eval.py`` and splits the
combined fixed/SKEM x bit_depth grid into two SEPARATE effect tables, so a
bit_depth-driven quality/size change is never conflated with a
selector-driven one:

    quantization_effect.csv
        Selector held constant. For each selector (fixed/skem), every
        bit_depth row is compared against that SAME selector's float32
        (falling back to int16) row — isolates what quantization alone costs,
        with the keyframe selector unchanged.

    selector_effect.csv
        bit_depth held constant. For each bit_depth present under BOTH
        selectors, the skem row is compared against the fixed row at that
        SAME bit_depth — isolates what changing the keyframe selector alone
        costs/saves, with quantization unchanged.

AWGN rows are excluded from both tables (never a quantization-ladder member,
never a reliable-digital reference — see run_transmission_reduction_eval.py's
BASELINE_PREFERENCE). A row (or comparison) touching any config with
``total_nan_or_inf_frames > 0`` is marked ``valid=False`` and its delta
columns are left blank rather than silently computing a delta against
garbage — still listed, never hidden.

Deliberately pure stdlib (csv/json/argparse/pathlib only) and imports nothing
from ``sgdjscc_lab`` — mirrors ``summarize_etri_video_eval.py``'s contract of
staying importable without torch installed, for report-only environments and
for ``scripts/run_transmission_normalization.sh --preflight-only`` (which
needs to sanity check the whole toolchain without a GPU).

Usage
-----
    python scripts/summarize_transmission_normalization.py --run-root outputs/transmission_normalization_20260826_120000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reliable-digital reference channels only (never AWGN) — mirrors
# run_transmission_reduction_eval.py::BASELINE_PREFERENCE's channel half.
_REFERENCE_CHANNELS = ("float32", "int16")
_QUALITY_FIELDS = ("mean_psnr", "mean_ssim", "mean_lpips")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split a transmission_reduction_eval run's aggregate.csv into "
                    "separate quantization-effect and selector-effect tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-root", required=True, help="Output dir from run_transmission_reduction_eval.py.")
    p.add_argument("--output-root", default=None, help="Where to write the two CSVs; default = --run-root.")
    return p.parse_args(argv)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required input not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_valid(row: Dict[str, Any]) -> bool:
    """Task validity gate: zero non-finite frames, all quality metrics
    finite, full quality coverage, and every expected video present —
    mirrors run_transmission_reduction_eval.py::_pareto_frontier's
    ``_row_is_valid`` (this script reads that function's own output columns,
    already carrying these fields in aggregate.csv, so it never re-derives
    them from raw per-frame data)."""
    n_nan = _to_float(row.get("total_nan_or_inf_frames", 0)) or 0.0
    if n_nan != 0.0:
        return False
    if str(row.get("all_finite_metrics", "True")).strip().lower() == "false":
        return False
    vfr = _to_float(row.get("valid_frame_ratio", 1.0))
    if vfr is not None and vfr != 1.0:
        return False
    if str(row.get("all_expected_videos_present", "True")).strip().lower() == "false":
        return False
    return True


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# quantization_effect.csv — selector held constant, bit_depth varies
# ─────────────────────────────────────────────────────────────────────────────

def _pick_reference(rows_by_channel: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for channel in _REFERENCE_CHANNELS:
        row = rows_by_channel.get(channel)
        if row is not None and _is_valid(row):
            return channel
    return None


def build_quantization_effect(aggregate_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_selector: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in aggregate_rows:
        if row.get("channel") == "awgn":
            continue
        by_selector.setdefault(row["selector"], {})[row["channel"]] = row

    out: List[Dict[str, Any]] = []
    for selector, by_channel in sorted(by_selector.items()):
        reference_channel = _pick_reference(by_channel)
        reference = by_channel.get(reference_channel) if reference_channel else None
        for channel, row in sorted(by_channel.items()):
            valid_row = _is_valid(row)
            has_reference = reference is not None
            comparable = valid_row and has_reference and channel != reference_channel
            entry: Dict[str, Any] = {
                "selector": selector,
                "channel": channel,
                "bit_depth": row.get("bit_depth", ""),
                "digital_step_policy": row.get("digital_step_policy", ""),
                "psss_backend_kind": row.get("psss_backend_kind", ""),
                "reference_channel": reference_channel or "",
                "n_videos": row.get("n_videos", ""),
                "mean_psnr": row.get("mean_psnr", ""),
                "mean_ssim": row.get("mean_ssim", ""),
                "mean_lpips": row.get("mean_lpips", ""),
                "mean_total_bundle_bytes_per_video": row.get("mean_total_bundle_bytes_per_video", ""),
                "mean_total_bundle_bytes_per_frame": row.get("mean_total_bundle_bytes_per_frame", ""),
                "total_nan_or_inf_frames": row.get("total_nan_or_inf_frames", ""),
                "nonfinite_stages": row.get("nonfinite_stages", ""),
                "valid": valid_row,
                "is_reference": channel == reference_channel,
            }
            if comparable:
                ref_bytes = _to_float(reference.get("mean_total_bundle_bytes_per_video"))
                row_bytes = _to_float(row.get("mean_total_bundle_bytes_per_video"))
                entry["psnr_drop_db"] = _to_float(reference["mean_psnr"]) - _to_float(row["mean_psnr"])
                entry["ssim_drop"] = _to_float(reference["mean_ssim"]) - _to_float(row["mean_ssim"])
                ref_lpips, row_lpips = _to_float(reference.get("mean_lpips")), _to_float(row.get("mean_lpips"))
                entry["lpips_rise"] = (row_lpips - ref_lpips) if (ref_lpips is not None and row_lpips is not None) else ""
                entry["byte_ratio_vs_reference"] = (
                    (row_bytes / ref_bytes) if (ref_bytes and row_bytes is not None) else ""
                )
            else:
                entry.update({
                    "psnr_drop_db": "", "ssim_drop": "", "lpips_rise": "", "byte_ratio_vs_reference": "",
                })
                if not has_reference:
                    entry["note"] = "no valid float32/int16 reference for this selector"
                elif not valid_row:
                    entry["note"] = "excluded: this config has non-finite frames"
                else:
                    entry["note"] = ""
            entry.setdefault("note", "")
            out.append(entry)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# selector_effect.csv — bit_depth held constant, selector varies (fixed vs skem)
# ─────────────────────────────────────────────────────────────────────────────

def build_selector_effect(aggregate_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_channel: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in aggregate_rows:
        if row.get("channel") == "awgn":
            continue
        by_channel.setdefault(row["channel"], {})[row["selector"]] = row

    out: List[Dict[str, Any]] = []
    for channel, by_selector in sorted(by_channel.items()):
        fixed_row = by_selector.get("fixed")
        skem_row = by_selector.get("skem")
        if fixed_row is None or skem_row is None:
            continue
        both_valid = _is_valid(fixed_row) and _is_valid(skem_row)
        entry: Dict[str, Any] = {
            "channel": channel,
            "bit_depth": fixed_row.get("bit_depth", ""),
            # skem's real/mock/proxy PSSS backend, so a proxy-backend result is
            # never presented as "real SKEM" (task requirement) -- fixed has no
            # PSSS backend at all (not_applicable) so only skem's is reported.
            "skem_psss_backend_kind": skem_row.get("psss_backend_kind", ""),
            "fixed_mean_psnr": fixed_row.get("mean_psnr", ""),
            "skem_mean_psnr": skem_row.get("mean_psnr", ""),
            "fixed_mean_ssim": fixed_row.get("mean_ssim", ""),
            "skem_mean_ssim": skem_row.get("mean_ssim", ""),
            "fixed_mean_lpips": fixed_row.get("mean_lpips", ""),
            "skem_mean_lpips": skem_row.get("mean_lpips", ""),
            "fixed_mean_total_bundle_bytes_per_video": fixed_row.get("mean_total_bundle_bytes_per_video", ""),
            "skem_mean_total_bundle_bytes_per_video": skem_row.get("mean_total_bundle_bytes_per_video", ""),
            "fixed_mean_total_bundle_bytes_per_frame": fixed_row.get("mean_total_bundle_bytes_per_frame", ""),
            "skem_mean_total_bundle_bytes_per_frame": skem_row.get("mean_total_bundle_bytes_per_frame", ""),
            "fixed_mean_n_keyframes_selected": fixed_row.get("mean_n_keyframes_selected", ""),
            "skem_mean_n_keyframes_selected": skem_row.get("mean_n_keyframes_selected", ""),
            "keyframe_count_matched": fixed_row.get("keyframe_count_matched", ""),
            "fixed_total_nan_or_inf_frames": fixed_row.get("total_nan_or_inf_frames", ""),
            "skem_total_nan_or_inf_frames": skem_row.get("total_nan_or_inf_frames", ""),
            "valid": both_valid,
        }
        if both_valid:
            entry["psnr_delta_skem_minus_fixed"] = _to_float(skem_row["mean_psnr"]) - _to_float(fixed_row["mean_psnr"])
            entry["ssim_delta_skem_minus_fixed"] = _to_float(skem_row["mean_ssim"]) - _to_float(fixed_row["mean_ssim"])
            fl, sl = _to_float(fixed_row.get("mean_lpips")), _to_float(skem_row.get("mean_lpips"))
            entry["lpips_delta_skem_minus_fixed"] = (sl - fl) if (fl is not None and sl is not None) else ""
            fb, sb = _to_float(fixed_row.get("mean_total_bundle_bytes_per_video")), _to_float(skem_row.get("mean_total_bundle_bytes_per_video"))
            entry["byte_ratio_skem_over_fixed"] = (sb / fb) if fb else ""
            fk, sk = _to_float(fixed_row.get("mean_n_keyframes_selected")), _to_float(skem_row.get("mean_n_keyframes_selected"))
            entry["keyframe_count_delta_skem_minus_fixed"] = (sk - fk) if (fk is not None and sk is not None) else ""
            entry["note"] = ""
        else:
            entry.update({
                "psnr_delta_skem_minus_fixed": "", "ssim_delta_skem_minus_fixed": "",
                "lpips_delta_skem_minus_fixed": "", "byte_ratio_skem_over_fixed": "",
                "keyframe_count_delta_skem_minus_fixed": "",
                "note": "excluded: fixed and/or skem side has non-finite frames at this bit_depth",
            })
        out.append(entry)
    return out


def run(argv=None) -> int:
    args = _parse_args(argv)
    run_root = Path(args.run_root)
    output_root = Path(args.output_root) if args.output_root else run_root

    aggregate_rows = _read_csv(run_root / "aggregate.csv")

    quant_rows = build_quantization_effect(aggregate_rows)
    selector_rows = build_selector_effect(aggregate_rows)

    _write_csv(output_root / "quantization_effect.csv", quant_rows)
    _write_csv(output_root / "selector_effect.csv", selector_rows)

    ablation_policies = sorted({
        r.get("digital_step_policy", "") for r in aggregate_rows
        if r.get("digital_step_policy", "") not in ("", "fixed_reference")
    })
    if ablation_policies:
        print(
            f"WARNING: this run mixes digital_step_policy={ablation_policies} into "
            "quantization_effect.csv -- those rows are a decoder-step ABLATION, not "
            "the pure quantization comparison ('fixed_reference' is); do not present "
            "them as isolating quantization alone.",
            file=sys.stderr,
        )

    skem_backend_kinds = sorted({
        r.get("psss_backend_kind", "") for r in aggregate_rows
        if r.get("selector") == "skem" and r.get("psss_backend_kind", "")
    })

    summary = {
        "run_root": str(run_root),
        "n_quantization_effect_rows": len(quant_rows),
        "n_selector_effect_rows": len(selector_rows),
        "n_invalid_configs": sum(1 for r in quant_rows if not r["valid"]),
        "ablation_policies_present": ablation_policies,
        "skem_backend_kinds_present": skem_backend_kinds,
    }
    (output_root / "normalization_effect_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
