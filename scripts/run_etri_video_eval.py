#!/usr/bin/env python
"""run_etri_video_eval.py – ETRI 10-video batch evaluation driver.

Runs the 10-video ETRI evaluation set (data/etri_video_eval) through
``scripts/evaluate_video.py`` / ``scripts/remeasure_video_metrics.py`` in
per-stage, per-video isolated output directories so no run ever overwrites
another. Each run directory keeps its own generated ``config.yaml`` and a
``run.log`` capturing the exact command + full stdout/stderr.

Stages (docs/etri_strategy.md 구현 실행 순서와의 대응)
------------------------------------------------------
- ``baseline``       1차 산출물: recon.mp4, temporal_frames.csv,
                     temporal_metrics.csv, keyframes.json, segments.json
- ``motion_sweep``   1차 step 3 motion gate를 여러 threshold로 반복 실행
- ``verifier``       2차 step 7: packet_match_report / controller_decisions
- ``generate``       3차 step 5: reuse/recompute/generate 3-way (mock backend)
- ``bidirectional``  4차 step 6: start_only vs bidirectional 비교 (mock backend)
- ``heldout``        5차 step 9: clip_only vs calibrated 재측정 + metric_delta
                     (**독립 재구성** — remeasure_video_metrics.py가 baseline/
                     verifier가 이미 저장한 recon/packet을 재사용하는 게 아니라
                     같은 config로 TemporalPipeline을 처음부터 다시 돌린다.
                     --no-models identity 복원에서는 결정적이라 baseline과
                     동일한 숫자가 나오지만, 확률적 실모델 샘플링에서는 값이
                     달라질 수 있다 — remeasure_video_metrics.py 자체의
                     fidelity note 참조)
- ``accounting``     6차 step 11-12: bit/symbol accounting + rate/reliability

Output layout (default --output-root outputs/etri_video_eval):

    outputs/etri_video_eval/
      baseline/<video>/            config.yaml, run.log, recon.mp4,
                                   extracted_frames/ (per-run mp4→frame cache), …
      motion_sweep/th_<t>/<video>/
      verifier/<video>/
      generate/<video>/
      bidirectional/<video>/
      heldout/<video>/
      accounting/<video>/accounting/rate_reliability_curve.csv  (per-video, 1 row)
      summary/rate_reliability_curve.csv   (built by summarize_etri_video_eval.py
                                            merging the per-video curves above —
                                            never written directly by run subprocesses,
                                            so parallel accounting runs cannot race
                                            on a shared file)
      batch_status.json            (each row also records no_models/snr/device so
                                    "ok" can be traced back to what config produced it)

Every run subprocess writes only inside its own <stage>/<video>[/th_<t>]/
directory — nothing here is shared/appended-to across concurrent runs, so
parallelizing this driver across processes (e.g. one per GPU) is safe.

Usage
-----
# Smoke run (identity reconstruction, no checkpoints/GPU) on one video:
python scripts/run_etri_video_eval.py --stages all --videos 01_person_walk --no-models

# Full batch, all stages, real models:
python scripts/run_etri_video_eval.py --stages all --snr 5 --device cuda:0

# Motion gate sweep only (05/06 are the pan/handheld stress videos):
python scripts/run_etri_video_eval.py --stages motion_sweep \
    --motion-thresholds off,0.02,0.05,0.08,0.12 --no-models

Scope note: with ``--no-models`` the reconstruction is identity and packets
are caption-derived — the outputs validate pipeline/artefact structure, not
reconstruction quality. Generate/bidirectional stages use the mock backends
(copy / interpolation) per the 3~4차 scope; see docs/etri_strategy.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

STAGES = (
    "baseline", "motion_sweep", "verifier", "generate",
    "bidirectional", "heldout", "accounting",
)

# Config fragments composed for every generated run config (same set as
# configs/composed_video.yaml). Written as absolute paths so the generated
# config can live inside its run directory.
_FRAGMENTS = ("channel/awgn", "model/sgdjscc", "infer/awgn", "eval/default", "video/default")


# ──────────────────────────────────────────────────────────────────────────────
# Manifest / dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_manifest(data_root: Path) -> list:
    """Parse data/etri_video_eval/manifest.csv into per-video entries.

    Each entry: {key, processed, captions, gt, row} where *key* is the
    processed-file stem (e.g. ``01_person_walk``) used for output directories,
    and captions/gt may be None when the side files are absent.
    """
    data_root = Path(data_root)
    manifest = data_root / "manifest.csv"
    entries = []
    with open(manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            processed = data_root / row["processed_file"]
            key = processed.stem
            captions = data_root / "captions" / f"{key}.txt"
            gt = data_root / "gt" / f"{key}.json"
            entries.append({
                "key": key,
                "processed": processed,
                "captions": captions if captions.exists() else None,
                "gt": gt if gt.exists() else None,
                "row": dict(row),
            })
    return entries


def convert_gt_to_presence(gt: dict) -> dict:
    """Convert a data/etri_video_eval/gt/<video>.json segment-level GT file
    into the per-item ``{item_id: {object_name: bool}}`` mapping the ETRI 5차
    ``gt`` presence backend expects (``remeasure_video_metrics.py
    --gt-metadata``, ``--from-packets`` mode).

    Item ids follow the still-image packet stem convention ``frame_{i:05d}``.
    This is an input-format bridge only — actual GT-backed calibration runs
    are a follow-up once packet dumps exist for these videos.
    """
    n_frames = int(gt.get("n_frames", 0))
    labels = set()
    for seg in gt.get("segments", []):
        for obj in seg.get("objects", []):
            labels.add(str(obj.get("label")))
    presence = {}
    for i in range(n_frames):
        frame_presence = {label: False for label in sorted(labels)}
        for seg in gt.get("segments", []):
            if int(seg.get("start_frame", 0)) <= i <= int(seg.get("end_frame", -1)):
                for obj in seg.get("objects", []):
                    if str(obj.get("presence", "visible")) == "visible":
                        frame_presence[str(obj.get("label"))] = True
        presence[f"frame_{i:05d}"] = frame_presence
    return presence


# ──────────────────────────────────────────────────────────────────────────────
# Per-run config generation
# ──────────────────────────────────────────────────────────────────────────────

def stage_out_dir(output_root: Path, stage: str, video_key: str, threshold=None) -> Path:
    """Isolated run directory: stage/video (+ /th_<t>/ level for the sweep)."""
    if stage == "motion_sweep":
        tag = "off" if threshold is None else f"{float(threshold):g}"
        return Path(output_root) / stage / f"th_{tag}" / video_key
    return Path(output_root) / stage / video_key


def build_run_config(
    repo_root: Path,
    input_path: Path,
    stage: str,
    *,
    motion_threshold=None,
    snr=None,
    save_video: bool = True,
    generate_delta_min: float = 0.0,
    generate_delta_max: float = 1.0,
    generate_motion_max: float = 1.0,
    generate_stage_motion_threshold: float = 0.05,
    rate_reliability_label: str = None,
) -> dict:
    """Build the per-run config dict for *stage*.

    The dict is written as YAML inside the run directory, so every relative
    output path below resolves to that directory (config.py resolves paths
    relative to the config file's own dir). ``_defaults_`` entries are
    absolute, pointing at the standard composed_video fragment set.

    All output paths here are relative (never a shared/global path) so that
    two runs — even for the same source video across different stages, or
    launched by separate concurrent processes — never write into a directory
    another run also touches. This includes ``video_io.extracted_frames_dir``:
    it used to point at one cache shared by every stage for a given video,
    but ``video_io.extract_frames()`` deletes-then-rewrites its target
    directory on every call, so two runs extracting the same video
    concurrently could race (one deleting frames the other is mid-read on).
    Keeping it per-run costs a redundant mp4 decode per stage (cheap for
    these 100-frame/10s clips) in exchange for that race going away entirely.
    """
    cfg = {
        "_defaults_": [str((Path(repo_root) / "configs" / f)) for f in _FRAGMENTS],
        "use_phase4": True,
        "use_phase5": False,
        "use_packet_eval": True,
        "input_path": str(input_path),
        "keyframe_json": "keyframes.json",
        "segment_json": "segments.json",
        "temporal_csv": "temporal_metrics.csv",
        "frame_log_csv": "temporal_frames.csv",
        "video_io": {
            "extracted_frames_dir": "extracted_frames",
            "recon_frames_dir": "recon_frames",
            "save_recon_frames": True,
            "save_recon_video": bool(save_video),
            "recon_video": "recon.mp4",
        },
    }
    if snr is not None:
        cfg["snr_db"] = float(snr)

    if stage == "motion_sweep" and motion_threshold is not None:
        cfg["temporal"] = {"motion_threshold": float(motion_threshold)}

    if stage in ("verifier", "accounting"):
        cfg["use_packet_verifier"] = True
        cfg["verifier"] = {
            "enabled": True,
            "report_json": "packet_match_report.json",
            "report_csv": "packet_match_report.csv",
            "decisions_json": "controller_decisions.json",
            "decisions_csv": "controller_decisions.csv",
        }

    if stage in ("generate", "bidirectional"):
        # PoC-exercise settings: the motion gate is turned ON and the
        # generate candidacy band widened so real pan/handheld motion can
        # actually route frames into the generate branch (mock backends).
        # These are NOT tuned evaluation values — see docs/etri_strategy.md 3차.
        cfg["temporal"] = {"motion_threshold": float(generate_stage_motion_threshold)}
        cfg["use_video_gen"] = True
        cfg["video_generator"] = {
            "enabled": True,
            "backend": "auto",
            "conditioning_mode": "start_only",
            "generate_delta_min": float(generate_delta_min),
            "generate_delta_max": float(generate_delta_max),
            "generate_motion_max": float(generate_motion_max),
            "save_generated_frames": True,
            "generated_frames_dir": "generated_frames",
        }
        if stage == "bidirectional":
            cfg["video_generator"].update({
                "conditioning_mode": "bidirectional",
                "bidirectional_missing_end_policy": "fallback_start_only",
                "comparison_enabled": True,
                "comparison_output": "generation_mode_comparison.json",
                "comparison_start_only_csv": "temporal_metrics_start_only.csv",
                "comparison_bidirectional_csv": "temporal_metrics_bidirectional.csv",
            })

    if stage == "heldout":
        cfg["heldout"] = {
            "enabled": True,
            "clip_only_json": "heldout/clip_only_metrics.json",
            "clip_only_csv": "heldout/clip_only_metrics.csv",
            "calibrated_json": "heldout/calibrated_metrics.json",
            "calibrated_csv": "heldout/calibrated_metrics.csv",
            "output_json": "heldout/metric_delta.json",
            "output_csv": "heldout/metric_delta.csv",
        }

    if stage == "accounting":
        cfg["accounting"] = {
            "enabled": True,
            "output_dir": "accounting",
        }
        cfg["rate_reliability"] = {
            "enabled": True,
            "output_json": "accounting/rate_reliability_summary.json",
            # Always a per-run local file — never a path shared across videos.
            # append_rate_reliability_row() does a check-then-write-header
            # (TOCTOU) before appending, so pointing every video's run at one
            # shared curve_csv would race under any parallel execution of
            # this driver. scripts/summarize_etri_video_eval.py merges all
            # per-video curve CSVs into summary/rate_reliability_curve.csv
            # afterwards via pipelines.rate_reliability_report.merge_rate_reliability_curves().
            "curve_csv": "accounting/rate_reliability_curve.csv",
            "label": rate_reliability_label,
        }
    return cfg


def build_command(repo_root: Path, stage: str, cfg_path: Path, captions=None,
                  no_models: bool = False, save_video: bool = True, device=None) -> list:
    """Assemble the subprocess argv for one run."""
    scripts = Path(repo_root) / "scripts"
    if stage == "heldout":
        cmd = [sys.executable, str(scripts / "remeasure_video_metrics.py"), "--config", str(cfg_path)]
        # --from-run mode (--input, no --from-packets): remeasure_video_metrics.py
        # rebuilds a FRESH TemporalPipeline from this config and reconstructs
        # every frame again — it does NOT load the baseline/verifier stage's
        # already-saved recon_frames/packet_match_report. With --no-models
        # (identity reconstruction) this is deterministic, so heldout/'s
        # numbers match baseline/'s; with a stochastic real-model sampler they
        # can differ run-to-run. See remeasure_video_metrics.py's own
        # "Fidelity note" docstring — --from-packets is the byte-exact option
        # when that matters, but it needs saved packet JSON pairs this batch
        # driver doesn't currently produce.
    else:
        cmd = [sys.executable, str(scripts / "evaluate_video.py"), "--config", str(cfg_path)]
        if save_video:
            cmd.append("--save-video")
    if captions is not None:
        cmd += ["--captions", str(captions)]
    if no_models:
        cmd.append("--no-models")
    if device is not None:
        cmd += ["--device", str(device)]
    return cmd


# ──────────────────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────────────────────

def _write_yaml(cfg: dict, path: Path) -> None:
    from omegaconf import OmegaConf
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), str(path))


def run_one(repo_root: Path, stage: str, entry: dict, out_dir: Path, cfg: dict, *,
            no_models: bool, save_video: bool, device=None, snr=None,
            skip_existing: bool = False, prior: dict = None) -> dict:
    """Generate the config, run the stage subprocess, tee output to run.log.

    ``no_models``/``snr``/``device`` on the returned row always describe the
    config that actually PRODUCED the files now on disk for this
    (stage, video, motion_threshold) — never just this invocation's CLI args.
    ``requested_no_models``/``requested_snr``/``requested_device`` separately
    record what THIS invocation asked for, which can differ from what
    produced the files when ``skip_existing`` short-circuits execution (e.g.
    files were made with ``--no-models``, this invocation passed
    ``--device cuda:0 --skip-existing``). ``prior`` — the existing
    batch_status.json row for this run id, looked up by the caller — is how a
    skip recovers the real produced-by config and preserves the original
    cmd/duration_sec/returncode instead of a fabricated/lost one.

    When ``prior["status"] == "failed"`` the marker file still exists (e.g.
    partial output written before the subprocess crashed, or a stale marker
    left by an even earlier successful run this failed attempt didn't clean
    up), so its no_models/snr/device/cmd/duration_sec are only the LAST
    ATTEMPTED config, not a confirmed produced-by one — the row still carries
    them (better than nothing) but is flagged with a note rather than treated
    as equally trustworthy as an "ok" prior.
    """
    out_dir = Path(out_dir)
    marker = out_dir / ("heldout/metric_delta.json" if stage == "heldout" else "temporal_metrics.csv")
    # The merge key in main() stays (stage, video, motion_threshold) because
    # that triple already equals the output directory 1:1 (stage_out_dir());
    # adding config fields to the key would create stale entries describing a
    # config a later rerun into the SAME directory has since overwritten.
    if skip_existing and marker.exists():
        if prior is not None:
            # Carry forward exactly what actually produced these files —
            # including cmd/duration_sec/returncode, which a naive "replace
            # with a fresh skip row" would otherwise silently discard.
            row = dict(prior)
            row["status"] = "skipped"
            row["out_dir"] = str(out_dir)
            if prior.get("status") == "failed":
                row["prior_status"] = "failed"
                row["note"] = (
                    "marker file exists but the last recorded run for this output "
                    "was 'failed' — no_models/snr/device/cmd/duration_sec here are "
                    "the LAST ATTEMPTED config, not a confirmed produced-by one "
                    "(the marker may be a stale artefact from an even earlier ok "
                    "run, or partial output from the failed attempt). Re-run "
                    "without --skip-existing to regenerate cleanly."
                )
        else:
            row = {"stage": stage, "video": entry["key"], "out_dir": str(out_dir),
                   "status": "skipped", "returncode": None, "duration_sec": 0.0, "cmd": None,
                   "no_models": None, "snr": None, "device": None,
                   "note": "skip_existing found output but no prior batch_status.json "
                          "record for it — provenance of the files on disk is unknown."}
        row["requested_no_models"] = no_models
        row["requested_snr"] = snr
        row["requested_device"] = device
        return row

    cfg_path = out_dir / "config.yaml"
    _write_yaml(cfg, cfg_path)

    cmd = build_command(repo_root, stage, cfg_path, captions=entry.get("captions"),
                        no_models=no_models, save_video=save_video, device=device)
    if stage == "heldout":
        cmd += ["--input", str(entry["processed"])]
        # Input-format bridge for the ETRI 5차 gt presence backend (see
        # convert_gt_to_presence). Written even though the clip_only run
        # doesn't consume it — it documents the exact --gt-metadata format
        # for a later --from-packets + gt-backend run.
        if entry.get("gt"):
            gt = json.loads(Path(entry["gt"]).read_text(encoding="utf-8"))
            gt_out = out_dir / "gt_presence.json"
            gt_out.write_text(json.dumps(convert_gt_to_presence(gt), indent=2), encoding="utf-8")

    log_path = out_dir / "run.log"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# stage={stage} video={entry['key']}\n# cmd: {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(repo_root))
    return {"stage": stage, "video": entry["key"], "out_dir": str(out_dir),
            "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode, "duration_sec": round(time.time() - t0, 2),
            "cmd": " ".join(cmd),
            # This invocation actually ran the subprocess, so requested ==
            # produced by construction — kept for schema symmetry with the
            # skip branch above (where they can legitimately differ).
            "no_models": no_models, "snr": snr, "device": device,
            "requested_no_models": no_models, "requested_snr": snr, "requested_device": device}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ETRI 10-video batch evaluation driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data" / "etri_video_eval"))
    p.add_argument("--output-root", default=str(_REPO_ROOT / "outputs" / "etri_video_eval"))
    p.add_argument("--stages", default="baseline",
                   help=f"Comma list from {STAGES} or 'all'.")
    p.add_argument("--videos", default=None,
                   help="Comma list of video keys (e.g. 01_person_walk) or ids (01); default all.")
    p.add_argument("--motion-thresholds", default="off,0.02,0.05,0.08,0.12",
                   help="motion_sweep stage: comma list of thresholds; 'off' = gate disabled.")
    p.add_argument("--no-models", action="store_true",
                   help="Identity-reconstruction dry run (no checkpoints/GPU).")
    p.add_argument("--snr", type=float, default=None, help="AWGN SNR dB (config snr_db).")
    p.add_argument("--device", default=None, help="Compute device (real-model runs).")
    p.add_argument("--no-save-video", action="store_true", help="Skip recon.mp4 assembly.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs whose primary output already exists (resume).")
    p.add_argument("--generate-motion-threshold", type=float, default=0.05,
                   help="generate/bidirectional stages: temporal.motion_threshold PoC value.")
    p.add_argument("--generate-delta-min", type=float, default=0.0)
    p.add_argument("--generate-delta-max", type=float, default=1.0)
    p.add_argument("--generate-motion-max", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    entries = read_manifest(Path(args.data_root))

    if args.videos:
        wanted = {v.strip() for v in args.videos.split(",") if v.strip()}
        entries = [e for e in entries
                   if e["key"] in wanted or e["row"].get("id") in wanted or e["row"].get("name") in wanted]
        if not entries:
            sys.exit(f"Error: --videos matched nothing among {sorted(wanted)}")

    stages = list(STAGES) if args.stages.strip() == "all" else [
        s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            sys.exit(f"Error: unknown stage '{s}' (valid: {', '.join(STAGES)})")

    thresholds = []
    for tok in args.motion_thresholds.split(","):
        tok = tok.strip()
        if not tok:
            continue
        thresholds.append(None if tok.lower() in ("off", "none", "null") else float(tok))

    save_video = not args.no_save_video

    # Loaded up front (not just at the end) so a skip_existing run can look
    # up what config actually produced each already-there output — see
    # run_one()'s prior= parameter.
    def _run_id(r):
        return (r.get("stage"), r.get("video"), r.get("motion_threshold"))

    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "batch_status.json"
    existing = []
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []
    existing_by_id = {_run_id(r): r for r in existing}

    runs = []
    for stage in stages:
        for entry in entries:
            per_run_thresholds = thresholds if stage == "motion_sweep" else [None]
            for th in per_run_thresholds:
                out_dir = stage_out_dir(output_root, stage, entry["key"], threshold=th)
                cfg = build_run_config(
                    _REPO_ROOT, entry["processed"], stage,
                    motion_threshold=th, snr=args.snr,
                    save_video=(save_video and stage != "heldout"),
                    generate_delta_min=args.generate_delta_min,
                    generate_delta_max=args.generate_delta_max,
                    generate_motion_max=args.generate_motion_max,
                    generate_stage_motion_threshold=args.generate_motion_threshold,
                    rate_reliability_label=entry["key"],
                )
                label = f"[{stage}{'' if th is None else f' th={th}'}] {entry['key']}"
                print(f"→ {label}", flush=True)
                status = run_one(
                    _REPO_ROOT, stage, entry, out_dir, cfg,
                    no_models=args.no_models, save_video=(save_video and stage != "heldout"),
                    device=args.device, snr=args.snr, skip_existing=args.skip_existing,
                    prior=existing_by_id.get((stage, entry["key"], th)),
                )
                if stage == "motion_sweep":
                    status["motion_threshold"] = th
                print(f"   {status['status']} ({status['duration_sec']}s) → {status['out_dir']}", flush=True)
                runs.append(status)

    # Keep prior entries for (stage, video, threshold) combos not re-run now.
    new_ids = {_run_id(r) for r in runs}
    merged = [r for r in existing if _run_id(r) not in new_ids] + runs
    status_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    n_fail = sum(1 for r in runs if r["status"] == "failed")
    print(f"\nBatch complete: {len(runs)} run(s), {n_fail} failed. Status → {status_path}")
    if n_fail:
        for r in runs:
            if r["status"] == "failed":
                print(f"  FAILED: {r['stage']} / {r['video']} → {r['out_dir']}/run.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
