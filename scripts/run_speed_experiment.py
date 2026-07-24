#!/usr/bin/env python
"""run_speed_experiment.py – 4-mode real-model video-eval speed/quality comparison.

Runs the same ETRI video(s) through ``evaluate_video.py`` under four
comparable configurations and writes a side-by-side markdown report:

  no_models_captions          identity reconstruction + given captions
                               (structural validation only — NOT real image
                               quality; see evaluate_video.py's --no-models help)
  real_keyframe_only_step10   real models, diffusion_step=10,
                               --force-interframe-reuse (only GOP keyframes
                               ever run diffusion — weaker validation than
                               the full per-frame pipeline)
  real_all_frames_step10      real models, diffusion_step=10, normal
                               reuse/recompute decisions
  real_all_frames_step50      real models, diffusion_step=50 (paper default),
                               normal reuse/recompute decisions

Each (mode, video) run writes to an isolated
``<output-root>/<mode>/<video>/`` directory (config.yaml, run.log, recon.mp4,
temporal_metrics.csv, temporal_frames.csv), so results are directly inspectable
per run in addition to the aggregated report. profiling_summary.json / progress.json
are also written here — evaluate_video.py's profiling is opt-in (off by default),
but this driver's own report depends on it, so it always passes --profile.

Usage
-----
# Quick throughput ESTIMATE (few frames per mode) before committing to a full run:
python scripts/run_speed_experiment.py --videos 01_person_walk --max-frames 8 \\
    --device cuda:0

# Full comparison on one video:
python scripts/run_speed_experiment.py --videos 01_person_walk --device cuda:0

Scope note: this compares SPEED and call counts, not reconstruction quality —
no_models_captions numbers must never be cited as real-model image quality
(see CLAUDE.md's algorithm-preservation / paper_mode notes), and
real_keyframe_only_step10's PTC/SFR/SDI are a structurally weaker validation
than real_all_frames_* (inter-frame drift/hallucination is never exercised —
see video/temporal_pipeline.py's force_interframe_reuse docstring).
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import run_etri_video_eval as batch  # noqa: E402  (manifest/config/command builders)

MODES = {
    "no_models_captions": dict(no_models=True, diffusion_step=None, force_interframe_reuse=False),
    "real_keyframe_only_step10": dict(no_models=False, diffusion_step=10, force_interframe_reuse=True),
    "real_all_frames_step10": dict(no_models=False, diffusion_step=10, force_interframe_reuse=False),
    "real_all_frames_step50": dict(no_models=False, diffusion_step=50, force_interframe_reuse=False),
}
MODE_ORDER = list(MODES)

_TEMPORAL_FIELDS = (
    "n_frames", "n_keyframes", "n_interframes", "n_reused", "n_generate",
    "n_recompute_semantic", "n_recompute_motion", "overhead_reduction",
    "ptc", "sfr", "sdi",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="4-mode real-model video-eval speed/quality comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data" / "etri_video_eval"))
    p.add_argument("--output-root", default=str(_REPO_ROOT / "outputs" / "speed_experiment"))
    p.add_argument("--videos", default=None,
                   help="Comma list of video keys/ids (e.g. 01_person_walk or 01); "
                        "default: the first video in the manifest only.")
    p.add_argument("--modes", default=",".join(MODE_ORDER),
                   help=f"Comma list from {MODE_ORDER}.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--snr", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None,
                   help="Passthrough to evaluate_video.py --max-frames for every mode: cheap "
                        "throughput estimate (avg_frame_sec) before running full-length. The "
                        "report is explicitly marked as an estimate when this is set.")
    p.add_argument("--no-clip", action="store_true",
                   help="Passthrough --no-clip to every real-model mode.")
    p.add_argument("--recon-caption-mode", default=None, choices=[None, "own", "skip"])
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (mode, video) runs whose temporal_metrics.csv already exists.")
    return p.parse_args()


def run_mode_video(mode: str, mode_cfg: dict, entry: dict, output_root: Path, *,
                    device: str, snr, max_frames, no_clip: bool, recon_caption_mode,
                    skip_existing: bool) -> dict:
    out_dir = Path(output_root) / mode / entry["key"]
    marker = out_dir / "temporal_metrics.csv"
    if skip_existing and marker.exists():
        return _collect(mode, entry["key"], out_dir, returncode=0, duration_sec=0.0, skipped=True)

    cfg = batch.build_run_config(_REPO_ROOT, entry["processed"], "baseline", snr=snr, save_video=True)
    # batch.build_run_config already overrides model_root absolutely (see its
    # comment) — this experiment's <output-root>/<mode>/<video>/config.yaml
    # nesting depth matches the batch driver's, so the same fix applies here
    # for free; kept explicit below only as a defensive re-assertion in case a
    # future edit changes build_run_config's default.
    cfg["model_root"] = str((_REPO_ROOT / "checkpoints").resolve())
    if mode_cfg["diffusion_step"] is not None:
        cfg["diffusion_step"] = mode_cfg["diffusion_step"]
    cfg_path = out_dir / "config.yaml"
    batch._write_yaml(cfg, cfg_path)

    # This driver's whole report (frames_done/avg_frame_sec/call counts) is
    # read back from profiling_summary.json (see _collect) — profiling is
    # opt-in in evaluate_video.py (default off, see its --profile help), so
    # it must be requested explicitly here regardless of mode.
    extra = ["--profile"]
    if mode_cfg["diffusion_step"] is not None:
        extra += ["--diffusion-step", str(mode_cfg["diffusion_step"])]
    if mode_cfg["force_interframe_reuse"]:
        extra.append("--force-interframe-reuse")
    if no_clip and not mode_cfg["no_models"]:
        extra.append("--no-clip")
    if recon_caption_mode and not mode_cfg["no_models"]:
        extra += ["--recon-caption-mode", recon_caption_mode]
    if max_frames is not None:
        extra += ["--max-frames", str(max_frames)]

    # Work around SGDJSCC's hardcoded .cuda() (see
    # run_etri_video_eval.remap_device_for_cuda_visible's docstring) — this
    # driver is meant to be run with different modes pinned to different GPUs
    # (this task's own remote verification ran real_keyframe_only_step10 on
    # cuda:0 and real_all_frames_step10 on cuda:1 simultaneously), so any
    # non-cuda:0 --device here needs the same CUDA_VISIBLE_DEVICES remap the
    # batch driver's --parallel path already applies.
    run_env, cli_device = batch.remap_device_for_cuda_visible(device, None)

    cmd = batch.build_command(
        _REPO_ROOT, "baseline", cfg_path, captions=entry.get("captions"),
        no_models=mode_cfg["no_models"], save_video=True, device=cli_device, extra_args=extra,
    )

    log_path = out_dir / "run.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_kwargs = {"stderr": subprocess.STDOUT, "cwd": str(_REPO_ROOT)}
    if run_env is not None:
        run_kwargs["env"] = run_env
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# mode={mode} video={entry['key']}\n# cmd: {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, **run_kwargs)
    duration = round(time.time() - t0, 2)
    return _collect(mode, entry["key"], out_dir, returncode=proc.returncode, duration_sec=duration)


def _collect(mode: str, video: str, out_dir: Path, *, returncode: int, duration_sec: float,
             skipped: bool = False) -> dict:
    result = {
        "mode": mode, "video": video, "out_dir": str(out_dir),
        "status": "skipped" if skipped else ("ok" if returncode == 0 else "failed"),
        "returncode": returncode, "duration_sec": duration_sec,
    }
    prof_path = out_dir / "profiling_summary.json"
    if prof_path.exists():
        try:
            prof = json.loads(prof_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prof = {}
        result["frames_done"] = prof.get("frames_done")
        result["avg_frame_sec"] = prof.get("avg_frame_sec")
        result["counters"] = prof.get("counters", {})
    tcsv = out_dir / "temporal_metrics.csv"
    if tcsv.exists():
        with open(tcsv, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            for k in _TEMPORAL_FIELDS:
                if k in rows[0]:
                    result[k] = rows[0][k]
    return result


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_report(results: list, path: Path, *, truncated: bool, modes: list) -> None:
    lines = ["# Speed experiment results\n"]
    if truncated:
        lines.append(
            "> **--max-frames was set: this is a THROUGHPUT ESTIMATE from a partial "
            "clip, not a full-video wall-clock measurement.** Use `avg_frame_sec` to "
            "extrapolate (`avg_frame_sec * total_frames_in_video`); `duration_sec` "
            "here only covers the truncated frames actually processed.\n"
        )
    lines.append(
        "| mode | video | status | duration_sec | frames_done | avg_frame_sec | "
        "diffusion_calls | blip2_calls | clip_calls | n_keyframes | n_reused | "
        "n_recompute | ptc | sfr | sdi |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        c = r.get("counters") or {}
        diff = c.get("diffusion_calls", 0)
        blip2 = c.get("blip2_calls", 0)
        clip = c.get("clip_image_calls", 0) + c.get("clip_text_calls", 0)
        n_recompute = (int(r.get("n_recompute_semantic") or 0) if str(r.get("n_recompute_semantic", "")).strip()
                       else 0) + (int(r.get("n_recompute_motion") or 0) if str(r.get("n_recompute_motion", "")).strip()
                       else 0)
        lines.append(
            f"| {r['mode']} | {r['video']} | {r['status']} | {_fmt(r.get('duration_sec'))} | "
            f"{_fmt(r.get('frames_done'))} | {_fmt(r.get('avg_frame_sec'))} | {diff} | {blip2} | "
            f"{clip} | {_fmt(r.get('n_keyframes'))} | {_fmt(r.get('n_reused'))} | {n_recompute} | "
            f"{_fmt(r.get('ptc'))} | {_fmt(r.get('sfr'))} | {_fmt(r.get('sdi'))} |"
        )
    lines.append("")
    lines.append("Modes run: " + ", ".join(modes))
    lines.append("")
    lines.append(
        "Notes: `no_models_captions` reconstruction is identity — its ptc/sfr/sdi validate "
        "orchestration only, not image quality. `real_keyframe_only_step10` forces every "
        "inter-frame to reuse (diffusion only at keyframes) — a weaker validation than the "
        "`real_all_frames_*` modes, which exercise the normal reuse/recompute decision on "
        "every frame."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_write_results_locked(results_path: Path, new_rows: list) -> list:
    """Read-merge-write ``results.json`` under an exclusive file lock.

    This driver is meant to be launched multiple times concurrently against
    the SAME ``--output-root`` (e.g. one mode per GPU — this task's own
    remote verification ran two modes simultaneously this way). Each run's
    own artefacts (config.yaml, recon.mp4, ...) live in an isolated
    ``<mode>/<video>/`` directory and never collide, but ``results.json`` is
    ONE shared file every invocation reads, merges into, and rewrites — a
    plain read-then-write has a TOCTOU race where two processes' reads both
    see the same "before" state and the second write silently discards the
    first's rows. ``fcntl.flock`` on the file itself serialises the whole
    read-merge-write critical section across processes (POSIX/Linux only,
    which is this project's only supported runtime — see CLAUDE.md).
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # "a+" both creates the file if missing and opens it read/write without
    # truncating, so the lock is acquired before any content is read or lost.
    with open(results_path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            raw = fh.read()
            try:
                existing = json.loads(raw) if raw.strip() else []
            except json.JSONDecodeError:
                existing = []
            new_ids = {(r.get("mode"), r.get("video")) for r in new_rows}
            merged = [r for r in existing if (r.get("mode"), r.get("video")) not in new_ids] + new_rows
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(merged, indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return merged


def main() -> None:
    args = _parse_args()
    entries = batch.read_manifest(Path(args.data_root))
    if args.videos:
        wanted = {v.strip() for v in args.videos.split(",") if v.strip()}
        entries = [e for e in entries
                   if e["key"] in wanted or e["row"].get("id") in wanted or e["row"].get("name") in wanted]
        if not entries:
            sys.exit(f"Error: --videos matched nothing among {sorted(wanted)}")
    else:
        entries = entries[:1]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            sys.exit(f"Error: unknown mode '{m}' (valid: {', '.join(MODE_ORDER)})")

    output_root = Path(args.output_root)
    results = []
    for mode in modes:
        for entry in entries:
            print(f"=== {mode} / {entry['key']} ===", flush=True)
            r = run_mode_video(
                mode, MODES[mode], entry, output_root,
                device=args.device, snr=args.snr, max_frames=args.max_frames,
                no_clip=args.no_clip, recon_caption_mode=args.recon_caption_mode,
                skip_existing=args.skip_existing,
            )
            print(f"    {r['status']} ({r['duration_sec']}s) frames_done={r.get('frames_done')} "
                  f"avg_frame_sec={r.get('avg_frame_sec')}", flush=True)
            results.append(r)

    output_root.mkdir(parents=True, exist_ok=True)
    # Merge with any existing results.json instead of overwriting wholesale —
    # this driver is meant to be launched multiple times concurrently against
    # the SAME --output-root (e.g. one mode per GPU, as in this task's remote
    # verification), and each run's out_dir is already isolated by (mode,
    # video), so a naive read-then-write would race across processes. See
    # _merge_write_results_locked's docstring for the flock-based fix.
    results_path = output_root / "results.json"
    merged = _merge_write_results_locked(results_path, results)
    report_path = output_root / "speed_experiment_report.md"
    merged_modes = list(dict.fromkeys([r["mode"] for r in merged]))  # de-duped, first-seen order
    write_report(merged, report_path, truncated=(args.max_frames is not None), modes=merged_modes)
    print(f"\nResults → {output_root / 'results.json'}")
    print(f"Report  → {report_path}")

    n_fail = sum(1 for r in results if r["status"] == "failed")
    if n_fail:
        print(f"\n{n_fail} run(s) FAILED — see out_dir/run.log for each.")
        sys.exit(1)


if __name__ == "__main__":
    main()
