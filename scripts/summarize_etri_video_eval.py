#!/usr/bin/env python
"""summarize_etri_video_eval.py – aggregate the ETRI 10-video batch outputs.

Reads the per-stage / per-video run directories produced by
``scripts/run_etri_video_eval.py`` and writes, under ``<output-root>/summary/``:

- ``videos_summary.csv/json/md``       per video: PTC/SFR/SDI, keyframe/segment
                                       counts, reuse/recompute/generate ratios,
                                       semantic-unit overhead + (if the
                                       accounting stage ran) bit/symbol values
- ``motion_sweep_summary.csv/json/md`` threshold × video decision/metric table,
                                       with a focus section for
                                       05_camera_pan_person / 06_handheld_sign
- ``verifier_summary.csv/json/md``     per video missing/additional/structural
                                       error counts + controller decision mix
                                       (rows with no packet_match_report.json
                                       are marked status=missing_artifact
                                       instead of being dropped)
- ``generate_summary.csv/json/md``     3-way decision counts, generate ratio,
                                       generated_frames/ file check, and
                                       PTC/SFR/SDI delta vs the same video's
                                       baseline (generate-off) stage
- ``bidirectional_summary.csv/json/md`` start_only vs bidirectional metric diff
- ``heldout_summary.csv/json/md``      clip_only vs calibrated + metric_delta
- ``accounting_summary_all.csv/json/md`` bit/symbol/semantic-unit reductions +
                                       rate/reliability rows
- ``rate_reliability_curve.csv``       every video's accounting-stage
                                       rate/reliability point merged into one
                                       curve (see merge_accounting_curve below)
- ``model_mode_comparison_summary.csv/json/md`` no-models (identity-recon
                                       smoke) vs real-model baseline, joined
                                       from two separately-run sibling batch
                                       dirs (default: ``<output-root>_nomodels``
                                       / ``<output-root>_real_full_step50``,
                                       overridable via --nomodels-root /
                                       --real-model-root); baseline stage only

Every summary is derived purely from files on disk — this script never re-runs
any pipeline. Missing stages are skipped silently (their summary files are not
written), so it can run after any subset of stages.

The rate/reliability curve merge specifically (rather than an in-run append)
matters for parallel batches: each accounting run writes only its own
``accounting/<video>/accounting/rate_reliability_curve.csv`` (see
run_etri_video_eval.py's build_run_config docstring) — no subprocess ever
appends to a file another run also writes to. This script is the one place
that combines them, via
``pipelines.rate_reliability_report.merge_rate_reliability_curves``, and only
after every run has finished.

Usage
-----
python scripts/summarize_etri_video_eval.py \
    --output-root outputs/etri_video_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FOCUS_VIDEOS = ("05_camera_pan_person", "06_handheld_sign")


# ──────────────────────────────────────────────────────────────────────────────
# Generic readers / writers
# ──────────────────────────────────────────────────────────────────────────────

def _read_single_row_csv(path: Path):
    """Read a one-row CSV (e.g. temporal_metrics.csv) into a dict, or None."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[0] if rows else None


def _read_json(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_multi_row_csv(path: Path):
    """Read a multi-row CSV (e.g. packet_match_report.csv) into a list of
    dicts, or None if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(v):
    """Best-effort str→float conversion for CSV-loaded values ('' → None)."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f == int(f) and "." not in str(v) else f


def _fmt(v, nd=4):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_summary(rows: list, out_base: Path, title: str, extra_md: str = "") -> None:
    """Write rows as <out_base>.csv/.json/.md (md = pipe table)."""
    if not rows:
        return
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())

    with open(out_base.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    out_base.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join("---" for _ in fields) + "|")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(f)) for f in fields) + " |")
    if extra_md:
        lines += ["", extra_md]
    out_base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _video_dirs(stage_dir: Path) -> list:
    if not stage_dir.is_dir():
        return []
    return sorted(p for p in stage_dir.iterdir() if p.is_dir())


# ──────────────────────────────────────────────────────────────────────────────
# Stage summarizers (each returns a list of row dicts)
# ──────────────────────────────────────────────────────────────────────────────

def summarize_videos(root: Path, stage: str = "baseline") -> list:
    """Per-video headline summary from the *stage* runs (default baseline)."""
    rows = []
    for vdir in _video_dirs(Path(root) / stage):
        tm = _read_single_row_csv(vdir / "temporal_metrics.csv")
        if tm is None:
            continue
        segments = _read_json(vdir / "segments.json") or []
        keyframes = _read_json(vdir / "keyframes.json") or {}
        n_frames = _num(tm.get("n_frames")) or 0
        n_inter = _num(tm.get("n_interframes")) or 0
        n_reused = _num(tm.get("n_reused")) or 0
        n_gen = _num(tm.get("n_generate")) or 0
        n_rs = _num(tm.get("n_recompute_semantic")) or 0
        n_rm = _num(tm.get("n_recompute_motion")) or 0
        row = {
            "video": vdir.name,
            "n_frames": n_frames,
            "ptc": _num(tm.get("ptc")),
            "sfr": _num(tm.get("sfr")),
            "sdi": _num(tm.get("sdi")),
            "temporal_srs": _num(tm.get("temporal_srs")),
            "n_keyframes": _num(tm.get("n_keyframes")) or len(keyframes.get("keyframes", [])),
            "n_segments": len(segments),
            "n_reused": n_reused,
            "n_recompute_semantic": n_rs,
            "n_recompute_motion": n_rm,
            "n_generate": n_gen,
            "reuse_ratio": round(n_reused / n_inter, 4) if n_inter else None,
            "recompute_ratio": round((n_rs + n_rm) / n_inter, 4) if n_inter else None,
            "generate_ratio": round(n_gen / n_inter, 4) if n_inter else None,
            "transmitted_units": _num(tm.get("transmitted_units")),
            "naive_units": _num(tm.get("naive_units")),
            "overhead_reduction": _num(tm.get("overhead_reduction")),
        }
        # Bit/symbol values when the accounting stage ran for this video.
        acc = _read_json(Path(root) / "accounting" / vdir.name / "accounting" / "accounting_summary.json")
        if acc:
            row.update({
                "total_bits": acc.get("total_bits"),
                "total_channel_symbols": acc.get("total_channel_symbols"),
                "bit_reduction": acc.get("bit_reduction"),
                "symbol_reduction": acc.get("symbol_reduction"),
            })
        rows.append(row)
    return rows


def summarize_motion_sweep(root: Path) -> list:
    """threshold × video table: decisions + PTC/SFR/SDI."""
    sweep_dir = Path(root) / "motion_sweep"
    rows = []
    for th_dir in sorted(sweep_dir.glob("th_*")):
        th = th_dir.name[len("th_"):]
        for vdir in _video_dirs(th_dir):
            tm = _read_single_row_csv(vdir / "temporal_metrics.csv")
            if tm is None:
                continue
            rows.append({
                "motion_threshold": th,
                "video": vdir.name,
                "n_reused": _num(tm.get("n_reused")),
                "n_recompute_semantic": _num(tm.get("n_recompute_semantic")),
                "n_recompute_motion": _num(tm.get("n_recompute_motion")),
                "n_generate": _num(tm.get("n_generate")),
                "ptc": _num(tm.get("ptc")),
                "sfr": _num(tm.get("sfr")),
                "sdi": _num(tm.get("sdi")),
                "transmitted_units": _num(tm.get("transmitted_units")),
                "overhead_reduction": _num(tm.get("overhead_reduction")),
            })
    rows.sort(key=lambda r: (r["video"], _sort_th(r["motion_threshold"])))
    return rows


def _sort_th(th: str) -> float:
    return -1.0 if th == "off" else float(th)


def motion_sweep_focus_md(rows: list) -> str:
    """Markdown focus section for the pan/handheld stress videos."""
    lines = ["## Focus: camera pan / handheld (한계 1 — 의미 동일·모션 큼 구간)", ""]
    for video in FOCUS_VIDEOS:
        sub = [r for r in rows if r["video"] == video]
        if not sub:
            continue
        lines.append(f"### {video}")
        lines.append("")
        lines.append("| motion_threshold | n_reused | n_recompute_motion | n_recompute_semantic | ptc | sfr | sdi |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sorted(sub, key=lambda r: _sort_th(r["motion_threshold"])):
            lines.append(
                f"| {r['motion_threshold']} | {r['n_reused']} | {r['n_recompute_motion']} "
                f"| {r['n_recompute_semantic']} | {_fmt(r['ptc'])} | {_fmt(r['sfr'])} | {_fmt(r['sdi'])} |")
        lines.append("")
    lines.append("motion gate가 켜질수록(threshold ↓) reuse가 recompute_motion으로 이동하면 "
                 "의미는 같지만 모션이 큰 구간을 놓치지 않는다는 뜻이다 (docs/etri_strategy.md 한계 1).")
    return "\n".join(lines)


_VERIFIER_MISSING_ROW_FIELDS = (
    "n_frames_verified", "missing_objects_total", "additional_objects_total",
    "structural_errors_total", "mean_severity", "max_severity",
    "decision_accept", "decision_suppress_extra", "decision_strengthen_missing",
    "decision_strengthen_structure", "decision_fallback_recompute",
    "decision_keyframe_fallback", "accept_ratio",
)


def _load_packet_match_report(vdir: Path):
    """Read packet_match_report.json, falling back to packet_match_report.csv
    when only the CSV was written for this run. Returns a list of dicts with
    severity/error-count fields coerced to numbers (CSV values are strings
    otherwise), or None if neither file exists."""
    report = _read_json(vdir / "packet_match_report.json")
    if report is not None:
        return report
    rows = _read_multi_row_csv(vdir / "packet_match_report.csv")
    if rows is None:
        return None
    numeric_fields = ("severity", "missing_object_count", "additional_object_count",
                      "relation_error_count", "attribute_error_count")
    normalized = []
    for r in rows:
        nr = dict(r)
        for key in numeric_fields:
            if key in nr:
                nr[key] = _num(nr[key])
        normalized.append(nr)
    return normalized


def summarize_verifier(root: Path) -> list:
    """Per-video packet-verifier error counts + controller decision mix.

    Accepts either ``packet_match_report.json`` or, if only that was written,
    ``packet_match_report.csv`` (see ``_load_packet_match_report``). A video
    subdirectory under ``verifier/`` with neither file (verifier stage
    attempted but failed, or the report was never written) gets a
    ``status: missing_artifact`` row with every metric field left ``None``
    rather than being dropped silently — the whole ``verifier/`` stage being
    absent still yields an empty list (no summary file at all, i.e. "not
    run"), but a per-video gap within a stage that did run is reported
    explicitly.
    """
    rows = []
    for vdir in _video_dirs(Path(root) / "verifier"):
        report = _load_packet_match_report(vdir)
        if report is None:
            rows.append({
                "video": vdir.name,
                "status": "missing_artifact",
                **{f: None for f in _VERIFIER_MISSING_ROW_FIELDS},
            })
            continue
        n = len(report)
        severities = [r.get("severity") for r in report if r.get("severity") is not None]
        decisions = Counter(r.get("controller_decision") for r in report)
        missing = sum(int(r.get("missing_object_count") or 0) for r in report)
        additional = sum(int(r.get("additional_object_count") or 0) for r in report)
        structural = sum(int(r.get("relation_error_count") or 0)
                         + int(r.get("attribute_error_count") or 0) for r in report)
        rows.append({
            "video": vdir.name,
            "status": "ok",
            "n_frames_verified": n,
            "missing_objects_total": missing,
            "additional_objects_total": additional,
            "structural_errors_total": structural,
            "mean_severity": round(sum(severities) / len(severities), 4) if severities else None,
            "max_severity": round(max(severities), 4) if severities else None,
            "decision_accept": decisions.get("accept", 0),
            "decision_suppress_extra": decisions.get("suppress_extra", 0),
            "decision_strengthen_missing": decisions.get("strengthen_missing", 0),
            "decision_strengthen_structure": decisions.get("strengthen_structure_guidance", 0),
            "decision_fallback_recompute": decisions.get("fallback_recompute", 0),
            "decision_keyframe_fallback": decisions.get("keyframe_fallback", 0),
            "accept_ratio": round(decisions.get("accept", 0) / n, 4) if n else None,
        })
    return rows


def summarize_generate(root: Path) -> list:
    """3-way decision counts + generated_frames/ artefact check.

    Also joins each video's ``baseline/`` stage (generate branch off) PTC/SFR/SDI
    so the generate-branch run can be read as a delta against the same video's
    reuse/recompute-only baseline, not just in isolation. If the baseline stage
    wasn't run for a video, the ``baseline_*``/``*_delta_vs_baseline`` fields are
    left ``None`` rather than the row being dropped.
    """
    root = Path(root)
    baseline_by_video = {}
    for vdir in _video_dirs(root / "baseline"):
        tm = _read_single_row_csv(vdir / "temporal_metrics.csv")
        if tm is not None:
            baseline_by_video[vdir.name] = tm

    rows = []
    for vdir in _video_dirs(root / "generate"):
        tm = _read_single_row_csv(vdir / "temporal_metrics.csv")
        if tm is None:
            continue
        gen_dir = vdir / "generated_frames"
        n_saved = len(list(gen_dir.glob("generated_*.png"))) if gen_dir.is_dir() else 0
        n_inter = _num(tm.get("n_interframes")) or 0
        n_gen = _num(tm.get("n_generate")) or 0
        segs = _read_json(vdir / "segments.json") or []
        segs_with_gen = sum(1 for s in segs if s.get("generation"))
        row = {
            "video": vdir.name,
            "n_interframes": n_inter,
            "n_reused": _num(tm.get("n_reused")),
            "n_recompute_semantic": _num(tm.get("n_recompute_semantic")),
            "n_recompute_motion": _num(tm.get("n_recompute_motion")),
            "n_generate": n_gen,
            "generate_ratio": round(n_gen / n_inter, 4) if n_inter else None,
            "generated_frames_saved": n_saved,
            "saved_matches_decisions": (n_saved == n_gen),
            "segments_with_generation": segs_with_gen,
            "ptc": _num(tm.get("ptc")),
            "sfr": _num(tm.get("sfr")),
            "sdi": _num(tm.get("sdi")),
        }
        base_tm = baseline_by_video.get(vdir.name)
        for key in ("ptc", "sfr", "sdi"):
            base_val = _num(base_tm.get(key)) if base_tm else None
            gen_val = row[key]
            row[f"baseline_{key}"] = base_val
            row[f"{key}_delta_vs_baseline"] = (
                round(gen_val - base_val, 4)
                if gen_val is not None and base_val is not None else None
            )
        rows.append(row)
    return rows


def summarize_bidirectional(root: Path) -> list:
    """start_only vs bidirectional per-video comparison rows."""
    rows = []
    for vdir in _video_dirs(Path(root) / "bidirectional"):
        comp = _read_json(vdir / "generation_mode_comparison.json")
        if comp is None:
            continue
        so, bi = comp.get("start_only", {}), comp.get("bidirectional", {})
        diff = comp.get("comparison", {})
        row = {"video": vdir.name}
        for key in ("ptc", "sfr", "sdi", "n_generate", "n_reused",
                    "n_recompute_semantic", "n_recompute_motion"):
            row[f"start_{key}"] = so.get(key)
            row[f"bidi_{key}"] = bi.get(key)
            # compare_metrics() writes flat "<key>_diff" entries.
            row[f"diff_{key}"] = diff.get(f"{key}_diff")
        rows.append(row)
    return rows


def summarize_heldout(root: Path) -> list:
    """clip_only vs calibrated held-out remeasurement per video."""
    rows = []
    for vdir in _video_dirs(Path(root) / "heldout"):
        delta = _read_json(vdir / "heldout" / "metric_delta.json")
        clip_only = _read_json(vdir / "heldout" / "clip_only_metrics.json")
        calibrated = _read_json(vdir / "heldout" / "calibrated_metrics.json")
        if clip_only is None:
            continue
        cm = clip_only.get("metrics", clip_only)
        km = (calibrated or {}).get("metrics", calibrated or {})
        row = {"video": vdir.name, "n_items": cm.get("n_items")}
        for key in ("ptc", "sfr", "sdi", "mean_severity"):
            row[f"clip_only_{key}"] = cm.get(key)
            row[f"calibrated_{key}"] = km.get(key)
        row["metric_delta_exists"] = delta is not None
        row["calibrated_equals_clip_only"] = all(
            cm.get(k) == km.get(k) for k in ("ptc", "sfr", "sdi", "mean_severity"))
        rows.append(row)
    return rows


def merge_accounting_curve(root: Path):
    """Merge every video's per-run rate/reliability curve CSV into one shared
    ``summary/rate_reliability_curve.csv``.

    Reads-only aggregation, run post-hoc after all accounting-stage
    subprocesses have finished — no in-flight run ever writes to the merged
    path, so this is safe to call regardless of how the accounting runs
    themselves were parallelized. Returns the number of merged rows, or 0 if
    no per-video curve CSVs exist yet.
    """
    from sgdjscc_lab.pipelines.rate_reliability_report import merge_rate_reliability_curves

    curves = sorted((Path(root) / "accounting").glob("*/accounting/rate_reliability_curve.csv"))
    if not curves:
        return 0
    out_path = Path(root) / "summary" / "rate_reliability_curve.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return merge_rate_reliability_curves([str(p) for p in curves], str(out_path))


def summarize_accounting(root: Path) -> list:
    """bit/symbol/semantic-unit reductions + rate/reliability per video."""
    rows = []
    for vdir in _video_dirs(Path(root) / "accounting"):
        acc = _read_json(vdir / "accounting" / "accounting_summary.json")
        if acc is None:
            continue
        rr = _read_json(vdir / "accounting" / "rate_reliability_summary.json") or {}
        rows.append({
            "video": vdir.name,
            "n_frames": acc.get("n_frames"),
            "total_bits": acc.get("total_bits"),
            "total_channel_symbols": acc.get("total_channel_symbols"),
            "total_semantic_units": acc.get("total_semantic_units"),
            "baseline": acc.get("baseline"),
            "bit_reduction": acc.get("bit_reduction"),
            "symbol_reduction": acc.get("symbol_reduction"),
            "semantic_unit_reduction": acc.get("semantic_unit_reduction"),
            "proxy_fraction": acc.get("proxy_fraction"),
            "bits_per_frame": rr.get("bits_per_frame"),
            "symbols_per_frame": rr.get("symbols_per_frame"),
            "ptc": rr.get("ptc"),
            "sfr": rr.get("sfr"),
            "sdi": rr.get("sdi"),
            "mean_severity": rr.get("mean_severity"),
        })
    return rows


_MODEL_MODE_METRIC_KEYS = (
    "ptc", "sfr", "sdi", "temporal_srs", "n_reused",
    "n_recompute_semantic", "n_recompute_motion", "n_generate", "overhead_reduction",
)


def _locate_temporal_metrics(root: Path, video: str, stage: str):
    """Find <video>'s temporal_metrics.csv under *root*, supporting both the
    stage-driver tree (root/<stage>/<video>/temporal_metrics.csv, as written
    by run_etri_video_eval.py) and an older/ad-hoc flat tree
    (root/<video>/temporal_metrics.csv, e.g. a single-stage manual batch that
    never went through the stage driver). Returns None if neither exists."""
    staged = Path(root) / stage / video / "temporal_metrics.csv"
    if staged.exists():
        return staged
    flat = Path(root) / video / "temporal_metrics.csv"
    if flat.exists():
        return flat
    return None


def _model_mode_videos(root: Path, stage: str) -> set:
    root = Path(root)
    staged = root / stage
    if staged.is_dir():
        return {p.name for p in staged.iterdir() if p.is_dir()}
    if not root.is_dir():
        return set()
    return {p.name for p in root.iterdir()
            if p.is_dir() and (p / "temporal_metrics.csv").exists()}


def summarize_model_mode_comparison(nomodels_root, real_model_root, stage: str = "baseline") -> list:
    """Per-video comparison of a no-models (identity-recon smoke) batch
    against a real-model (actual SGD-JSCC + diffusion checkpoints) batch —
    e.g. ``outputs/etri_video_eval_nomodels`` vs
    ``outputs/etri_video_eval_real_full_step50``. These are typically two
    separately-run local batches with different output trees (not stages of
    the same run), so ``summarize_videos``/``summarize_all`` above never join
    them; this function is the one place that reads both roots together.

    Only the *stage* named (``baseline`` by default) is compared, because a
    real-model batch that only ran the baseline stage has no motion_sweep/
    verifier/generate/bidirectional/heldout/accounting stage output to compare
    against — comparing anything beyond baseline would silently overreach.

    Either root may be ``None`` or a non-existent path (batch not run yet /
    not provided); in that case its columns are left ``None`` per video
    instead of the whole comparison being skipped, so a one-sided comparison
    (e.g. only the no-models batch exists so far) still shows what's there.
    Returns ``[]`` only when *neither* root is available at all.
    """
    nomodels_root = Path(nomodels_root) if nomodels_root else None
    real_model_root = Path(real_model_root) if real_model_root else None
    nomodels_ok = nomodels_root is not None and nomodels_root.is_dir()
    real_model_ok = real_model_root is not None and real_model_root.is_dir()
    if not nomodels_ok and not real_model_ok:
        return []

    videos = set()
    if nomodels_ok:
        videos |= _model_mode_videos(nomodels_root, stage)
    if real_model_ok:
        videos |= _model_mode_videos(real_model_root, stage)

    rows = []
    for video in sorted(videos):
        nm_path = _locate_temporal_metrics(nomodels_root, video, stage) if nomodels_ok else None
        rm_path = _locate_temporal_metrics(real_model_root, video, stage) if real_model_ok else None
        nm = _read_single_row_csv(nm_path) if nm_path else None
        rm = _read_single_row_csv(rm_path) if rm_path else None
        row = {
            "video": video,
            "nomodels_available": nm is not None,
            "real_model_available": rm is not None,
        }
        for key in _MODEL_MODE_METRIC_KEYS:
            nv = _num(nm.get(key)) if nm else None
            rv = _num(rm.get(key)) if rm else None
            row[f"nomodels_{key}"] = nv
            row[f"real_model_{key}"] = rv
            row[f"{key}_diff_real_minus_nomodels"] = (
                round(rv - nv, 4)
                if isinstance(nv, (int, float)) and isinstance(rv, (int, float)) else None
            )
        rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def summarize_all(output_root: Path, nomodels_root=None, real_model_root=None) -> dict:
    """Run every stage summarizer against *output_root*; write summary files.

    *nomodels_root*/*real_model_root* are optional sibling batch directories
    (see ``summarize_model_mode_comparison``) — pass either or both to also
    write ``model_mode_comparison_summary.csv/json/md`` under the same
    ``output_root/summary/``. Neither is required; omitting both just skips
    that one summary, same as any other missing stage.

    Returns {summary_name: row_count} for reporting/tests.
    """
    root = Path(output_root)
    summary_dir = root / "summary"
    written = {}

    specs = [
        ("videos_summary", summarize_videos(root),
         "ETRI 10-video summary (baseline stage)", ""),
        ("motion_sweep_summary", summarize_motion_sweep(root),
         "Motion gate sweep (threshold × video)", None),  # focus md filled below
        ("verifier_summary", summarize_verifier(root),
         "Packet verifier + controller decisions (loop-internal)", ""),
        ("generate_summary", summarize_generate(root),
         "3-way reuse/recompute/generate decisions (mock backend)", ""),
        ("bidirectional_summary", summarize_bidirectional(root),
         "start_only vs bidirectional generation comparison (mock backend)", ""),
        ("heldout_summary", summarize_heldout(root),
         "Held-out remeasurement: clip_only vs calibrated", ""),
        ("accounting_summary_all", summarize_accounting(root),
         "Transmission accounting + rate/reliability (PoC)", ""),
        ("model_mode_comparison_summary",
         summarize_model_mode_comparison(nomodels_root, real_model_root),
         "no-models (identity-recon smoke) vs real-model baseline comparison", ""),
    ]
    for name, rows, title, extra in specs:
        if not rows:
            continue
        if name == "motion_sweep_summary":
            extra = motion_sweep_focus_md(rows)
        write_summary(rows, summary_dir / name, title, extra_md=extra or "")
        written[name] = len(rows)

    n_curve = merge_accounting_curve(root)
    if n_curve:
        written["rate_reliability_curve"] = n_curve

    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize ETRI 10-video batch outputs")
    p.add_argument("--output-root", default=str(_REPO_ROOT / "outputs" / "etri_video_eval"))
    p.add_argument("--nomodels-root", default=None,
                   help="Sibling no-models batch dir to compare against real-model baseline "
                        "(default: <output-root>_nomodels if it exists)")
    p.add_argument("--real-model-root", default=None,
                   help="Sibling real-model batch dir to compare against no-models baseline "
                        "(default: <output-root>_real_full_step50 if it exists)")
    args = p.parse_args()

    root = Path(args.output_root)
    if not root.is_dir():
        sys.exit(f"Error: output root not found: {root}")

    nomodels_root = Path(args.nomodels_root) if args.nomodels_root else \
        Path(f"{root}_nomodels")
    real_model_root = Path(args.real_model_root) if args.real_model_root else \
        Path(f"{root}_real_full_step50")
    if not nomodels_root.is_dir():
        nomodels_root = None
    if not real_model_root.is_dir():
        real_model_root = None

    written = summarize_all(root, nomodels_root=nomodels_root, real_model_root=real_model_root)
    if not written:
        sys.exit(f"Error: no stage outputs found under {root} — run scripts/run_etri_video_eval.py first.")
    print(f"Summaries → {root / 'summary'}")
    for name, n in written.items():
        ext = ".csv" if name == "rate_reliability_curve" else ".csv/.json/.md"
        print(f"  {name:<28} {n} row(s)  ({ext})")


if __name__ == "__main__":
    main()
