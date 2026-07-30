#!/usr/bin/env python
"""batch_remeasure_owlv2_vqa_10videos.py – ETRI 5차 follow-up: batch driver for
re-running ``scripts/remeasure_video_metrics.py`` (real OWLv2/VQA/ensemble
presence calibration + the object-vocabulary filter) over all 10 ETRI
evaluation videos, reusing an already-completed real-model batch's saved
frames.

Scope note (preparation only)
------------------------------
This script prepares and drives the REAL-WEIGHT remeasurement — it does not
run it itself as part of any automated task. It reuses
``outputs/etri_video_eval_real_full_step50/baseline/<video_id>/{extracted_frames,
recon_frames}`` (see ``pipelines.heldout_remeasurement.items_from_recon_frame_dirs``)
so no diffusion/JSCC reconstruction is repeated — only CLIP (for packet
re-extraction) + whichever presence backend(s) a mode configures (OWLv2/VQA/
CLIP) actually run. Each (mode, video) job still needs real network access the
first time (OWLv2/BLIP-2 weight download) and real GPU time — this is a
genuinely heavy operation, left for the operator to launch explicitly.

Five modes (see MODE_SPECS)
-----------------------------
owlv2                    OWLv2-only detector calibration (configs/etri_video_eval_owlv2.yaml)
vqa                      VQA-only calibration (configs/etri_video_eval_vqa.yaml)
ensemble_nofilter        ensemble_weighted (clip+owlv2+vqa), object_vocabulary_filter OFF —
                          the pre-filter baseline; comparison-only, see caveat below
ensemble_gt_filter       ensemble_weighted + object_vocabulary_filter ON, use_gt_vocabulary=True —
                          closed-world GT-object-only preservation evaluation
ensemble_openworld_filter ensemble_weighted + object_vocabulary_filter ON, use_gt_vocabulary=False —
                          strips count/action/scene noise but keeps non-GT objects, for
                          hallucination/additional-object analysis

CAVEAT (see docs/etri_owlv2_vqa_readiness.md "10-video batch remeasurement" section
for the full writeup): ``ensemble_nofilter`` numbers are NOT suitable for a final
object-preservation claim — caption-noun contamination ("one"/"walking"/"sidewalk")
inflates its severity/PTC/SFR improvement. Use ``ensemble_gt_filter`` for
object-preservation claims and ``ensemble_openworld_filter`` for
hallucination/additional-object analysis; ``ensemble_nofilter`` is comparison-only.

Each (mode, video_id) job:
1. Loads the mode's base config (``configs/etri_video_eval_{owlv2,vqa,ensemble}.yaml``)
   via ``sgdjscc_lab.config.load_config`` (fragments + path resolution already
   applied), then overrides every output-artefact path (``heldout.*`` — the only
   paths ``remeasure_video_metrics.py`` actually reads/writes — plus the other
   declared-but-currently-unused manual_* paths, for forward-compatibility and to
   avoid any silent collision) to live under
   ``<output-root>/<mode>/<video_id>/`` and applies the mode's
   ``verifier.object_vocabulary_filter``/other config overrides.
2. Writes the resulting (already fully composed, no ``_defaults_`` left — see
   ``build_mode_config``'s docstring) config to
   ``<output-root>/_generated_configs/<mode>/<video_id>.yaml``.
3. Invokes ``scripts/remeasure_video_metrics.py --config <generated> \\
   --from-recon-frames outputs/etri_video_eval_real_full_step50/baseline/<video_id> \\
   --captions data/etri_video_eval/captions/<video_id>.txt \\
   --gt-metadata data/etri_video_eval/gt/<video_id>.json --device <device>``
   as a subprocess (auto-omitting --captions/--gt-metadata if those per-video
   files don't exist), capturing stdout/stderr to
   ``<output-root>/<mode>/<video_id>/run.log``.

Usage
-----
# See exactly what would run, without running anything (no GPU/network use):
python scripts/batch_remeasure_owlv2_vqa_10videos.py --dry-run

# Actually run all 10 videos × all 5 modes (real GPU + weight downloads):
python scripts/batch_remeasure_owlv2_vqa_10videos.py --device cuda:0

# Only a subset:
python scripts/batch_remeasure_owlv2_vqa_10videos.py \\
    --videos 01_person_walk,03_dog_walk --modes owlv2,ensemble_gt_filter

# Resume an interrupted batch without re-running finished jobs:
python scripts/batch_remeasure_owlv2_vqa_10videos.py --skip-existing

# Regenerate only the aggregate summary from whatever has already run:
python scripts/batch_remeasure_owlv2_vqa_10videos.py --summary-only

See also: docs/etri_owlv2_vqa_readiness.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sgdjscc_lab.batch_remeasure_owlv2_vqa_10videos")

DEFAULT_BASELINE_ROOT = _REPO_ROOT / "outputs" / "etri_video_eval_real_full_step50" / "baseline"
DEFAULT_CAPTIONS_DIR = _REPO_ROOT / "data" / "etri_video_eval" / "captions"
DEFAULT_GT_DIR = _REPO_ROOT / "data" / "etri_video_eval" / "gt"
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "etri_video_eval" / "remeasure_10videos"
DEFAULT_CONFIGS_DIR = _REPO_ROOT / "configs"
REMEASURE_SCRIPT = _REPO_ROOT / "scripts" / "remeasure_video_metrics.py"

GENERATED_CONFIGS_DIRNAME = "_generated_configs"
SUMMARY_BASENAME = "summary_metrics"   # → <output-root>/summary_metrics.{csv,md}


# ─────────────────────────────────────────────────────────────────────────────
# Mode registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModeSpec:
    base_config: str
    purpose: str
    caveat: str = ""
    overrides: Dict[str, object] = field(default_factory=dict)


MODE_SPECS: Dict[str, ModeSpec] = {
    "owlv2": ModeSpec(
        base_config="etri_video_eval_owlv2.yaml",
        purpose="OWLv2-only zero-shot detector presence calibration.",
        overrides={},
    ),
    "vqa": ModeSpec(
        base_config="etri_video_eval_vqa.yaml",
        purpose="VQA-only (BLIP-2 yes/no) presence calibration.",
        overrides={},
    ),
    "ensemble_nofilter": ModeSpec(
        base_config="etri_video_eval_ensemble.yaml",
        purpose="clip+owlv2+vqa ensemble_weighted, object_vocabulary_filter OFF — "
                "the pre-filter baseline, for comparison against the two filtered modes below.",
        caveat="NOT suitable for a final object-preservation claim: caption-noun "
               "contamination ('one'/'walking'/'sidewalk' ...) inflates its severity/PTC/SFR "
               "improvement. Comparison-only.",
        overrides={"verifier.object_vocabulary_filter.enabled": False},
    ),
    "ensemble_gt_filter": ModeSpec(
        base_config="etri_video_eval_ensemble.yaml",
        purpose="clip+owlv2+vqa ensemble_weighted + object_vocabulary_filter ON, "
                "use_gt_vocabulary=True — closed-world GT-object-only preservation evaluation.",
        caveat="Use this mode's numbers for object-preservation claims.",
        overrides={
            "verifier.object_vocabulary_filter.enabled": True,
            "verifier.object_vocabulary_filter.use_gt_vocabulary": True,
        },
    ),
    "ensemble_openworld_filter": ModeSpec(
        base_config="etri_video_eval_ensemble.yaml",
        purpose="clip+owlv2+vqa ensemble_weighted + object_vocabulary_filter ON, "
                "use_gt_vocabulary=False — count/action/scene noise removed but non-GT "
                "objects retained (open-world vocabulary).",
        caveat="Use this mode's numbers for hallucination/additional-object analysis, "
               "NOT for object-preservation claims (it is not restricted to GT objects).",
        overrides={
            "verifier.object_vocabulary_filter.enabled": True,
            "verifier.object_vocabulary_filter.use_gt_vocabulary": False,
        },
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Video discovery / per-video file lookup
# ─────────────────────────────────────────────────────────────────────────────

def discover_videos(baseline_root, videos_filter: Optional[Sequence[str]] = None) -> List[str]:
    """List video ids under *baseline_root* that have a ``recon_frames/`` dir
    (i.e. a completed real-model baseline run), sorted by name.

    *videos_filter*, if given, restricts (and reorders to match) the result;
    a name not found under *baseline_root* raises ``ValueError`` rather than
    silently being dropped, so a typo in ``--videos`` fails loudly.
    """
    baseline_root = Path(baseline_root)
    if not baseline_root.is_dir():
        return []
    found = sorted(
        p.name for p in baseline_root.iterdir()
        if p.is_dir() and (p / "recon_frames").is_dir()
    )
    if not videos_filter:
        return found
    wanted = list(dict.fromkeys(videos_filter))   # de-dup, preserve order
    missing = [v for v in wanted if v not in found]
    if missing:
        raise ValueError(
            f"Requested video id(s) not found under {baseline_root} "
            f"(no recon_frames/ dir): {missing}. Available: {found}"
        )
    return wanted


def captions_path_for(video_id: str, captions_dir=None) -> Path:
    return Path(captions_dir or DEFAULT_CAPTIONS_DIR) / f"{video_id}.txt"


def gt_path_for(video_id: str, gt_dir=None) -> Path:
    return Path(gt_dir or DEFAULT_GT_DIR) / f"{video_id}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Per-mode/video output paths + generated config
# ─────────────────────────────────────────────────────────────────────────────

def mode_output_dir(output_root, mode: str, video_id: str) -> Path:
    """``<output-root>/<mode>/<video_id>/`` — the root every artefact for this
    (mode, video) job lives under, matching the layout the task requested
    (e.g. ``.../owlv2/01_person_walk/heldout/metric_delta.json``)."""
    return Path(output_root) / mode / video_id


def generated_config_path(output_root, mode: str, video_id: str) -> Path:
    return Path(output_root) / GENERATED_CONFIGS_DIRNAME / mode / f"{video_id}.yaml"


def metric_delta_path(output_root, mode: str, video_id: str) -> Path:
    return mode_output_dir(output_root, mode, video_id) / "heldout" / "metric_delta.json"


def _apply_output_path_overrides(cfg, video_out_dir: Path) -> None:
    """Redirect every output-artefact path this config declares into
    *video_out_dir* so concurrent/sequential (mode, video) jobs never share a
    path. Only the six ``heldout.*`` keys are actually read/written by
    ``remeasure_video_metrics.py --from-recon-frames`` today, but the other
    manual_* keys are still declared in the base configs (unused by this
    codepath, but real fields other tooling — e.g. ``evaluate_video.py`` —
    does read); overriding all of them keeps the generated config internally
    consistent and avoids a silent collision if this script is ever reused
    against a codepath that does read them.
    """
    from omegaconf import OmegaConf

    video_out_dir = Path(video_out_dir)
    heldout_dir = video_out_dir / "heldout"

    updates = {
        "keyframe_json": video_out_dir / "keyframes.json",
        "segment_json": video_out_dir / "segments.json",
        "temporal_csv": video_out_dir / "temporal_metrics.csv",
        "frame_log_csv": video_out_dir / "temporal_frames.csv",
        "video_io.extracted_frames_dir": video_out_dir / "_extracted",
        "video_io.recon_frames_dir": video_out_dir / "recon_frames",
        "video_io.recon_video": video_out_dir / "recon.mp4",
        "verifier.report_json": video_out_dir / "packet_match_report.json",
        "verifier.report_csv": video_out_dir / "packet_match_report.csv",
        "verifier.decisions_json": video_out_dir / "controller_decisions.json",
        "verifier.decisions_csv": video_out_dir / "controller_decisions.csv",
        "heldout.clip_only_json": heldout_dir / "clip_only_metrics.json",
        "heldout.clip_only_csv": heldout_dir / "clip_only_metrics.csv",
        "heldout.calibrated_json": heldout_dir / "calibrated_metrics.json",
        "heldout.calibrated_csv": heldout_dir / "calibrated_metrics.csv",
        "heldout.output_json": heldout_dir / "metric_delta.json",
        "heldout.output_csv": heldout_dir / "metric_delta.csv",
    }
    for dotted_key, path_value in updates.items():
        OmegaConf.update(cfg, dotted_key, str(path_value), force_add=True)


def build_mode_config(mode: str, video_id: str, output_root, configs_dir=None):
    """Load *mode*'s base config, redirect all output paths under
    ``<output_root>/<mode>/<video_id>/``, and apply the mode's config
    overrides (see ``MODE_SPECS``). Returns an OmegaConf ``DictConfig`` ready
    to be written to disk (via :func:`write_generated_config`) — the returned
    config has already gone through ``sgdjscc_lab.config.load_config``'s
    ``_defaults_`` fragment composition and path resolution, so it has no
    ``_defaults_`` key left and every path is already absolute; it is
    therefore safe to write it to and re-load it from ANY directory (it does
    not need to live under ``configs/`` the way a hand-written fragment-using
    config does).
    """
    from omegaconf import OmegaConf
    from sgdjscc_lab.config import load_config

    if mode not in MODE_SPECS:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(MODE_SPECS)}")
    spec = MODE_SPECS[mode]
    configs_dir = Path(configs_dir) if configs_dir else DEFAULT_CONFIGS_DIR
    base_path = configs_dir / spec.base_config
    if not base_path.is_file():
        raise FileNotFoundError(f"Base config for mode {mode!r} not found: {base_path}")

    cfg = load_config(base_path)
    _apply_output_path_overrides(cfg, mode_output_dir(output_root, mode, video_id))
    for dotted_key, value in spec.overrides.items():
        OmegaConf.update(cfg, dotted_key, value, force_add=True)
    return cfg


def write_generated_config(cfg, path) -> Path:
    from omegaconf import OmegaConf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess command construction + single-job execution
# ─────────────────────────────────────────────────────────────────────────────

def build_command(
    video_id: str, config_path, baseline_root, device: str,
    captions_dir=None, gt_dir=None, python_exe: Optional[str] = None,
) -> List[str]:
    """The exact ``scripts/remeasure_video_metrics.py`` invocation for one
    (mode, video) job. ``--captions``/``--gt-metadata`` are only added when
    the corresponding per-video file actually exists — a video missing one
    (e.g. no GT annotated yet) still gets remeasured, just without that input,
    rather than the whole job being skipped or crashing on a missing path.
    """
    python_exe = python_exe or sys.executable
    cmd = [
        python_exe, str(REMEASURE_SCRIPT),
        "--config", str(config_path),
        "--from-recon-frames", str(Path(baseline_root) / video_id),
        "--device", str(device),
    ]
    cap = captions_path_for(video_id, captions_dir)
    if cap.is_file():
        cmd += ["--captions", str(cap)]
    else:
        logger.warning("No captions file for %s (%s) — running without --captions.", video_id, cap)
    gt = gt_path_for(video_id, gt_dir)
    if gt.is_file():
        cmd += ["--gt-metadata", str(gt)]
    else:
        logger.warning("No GT metadata file for %s (%s) — running without --gt-metadata.", video_id, gt)
    return cmd


@dataclass
class JobResult:
    mode: str
    video_id: str
    status: str                       # "ok" | "skipped" | "failed" | "dry_run"
    command: List[str]
    config_path: str
    metric_delta_path: str
    returncode: Optional[int] = None
    log_path: Optional[str] = None
    elapsed_seconds: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode, "video_id": self.video_id, "status": self.status,
            "command": self.command, "config_path": self.config_path,
            "metric_delta_path": self.metric_delta_path, "returncode": self.returncode,
            "log_path": self.log_path, "elapsed_seconds": self.elapsed_seconds,
        }


def run_job(
    mode: str, video_id: str, output_root, baseline_root, device: str,
    captions_dir=None, gt_dir=None, configs_dir=None,
    dry_run: bool = False, skip_existing: bool = False, python_exe: Optional[str] = None,
) -> JobResult:
    """Prepare (and, unless *dry_run*, execute) one (mode, video_id) job.

    Config generation always happens (cheap, useful for inspection even in
    dry-run) — only the actual ``subprocess.run`` of
    ``remeasure_video_metrics.py`` is skipped when *dry_run* is set.
    """
    delta_path = metric_delta_path(output_root, mode, video_id)
    config_path = generated_config_path(output_root, mode, video_id)

    if skip_existing and delta_path.is_file():
        logger.info("[%s/%s] SKIP (metric_delta.json already exists: %s)", mode, video_id, delta_path)
        return JobResult(mode, video_id, "skipped", [], str(config_path), str(delta_path))

    cfg = build_mode_config(mode, video_id, output_root, configs_dir=configs_dir)
    write_generated_config(cfg, config_path)
    cmd = build_command(video_id, config_path, baseline_root, device,
                         captions_dir=captions_dir, gt_dir=gt_dir, python_exe=python_exe)

    if dry_run:
        logger.info("[%s/%s] DRY-RUN command: %s", mode, video_id, " ".join(cmd))
        logger.info("[%s/%s] DRY-RUN output → %s", mode, video_id, delta_path)
        return JobResult(mode, video_id, "dry_run", cmd, str(config_path), str(delta_path))

    log_path = mode_output_dir(output_root, mode, video_id) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[%s/%s] RUN: %s", mode, video_id, " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout or ''}\n--- stderr ---\n{proc.stderr or ''}\n",
        encoding="utf-8",
    )
    status = "ok" if proc.returncode == 0 else "failed"
    level = logging.INFO if status == "ok" else logging.ERROR
    logger.log(level, "[%s/%s] %s (returncode=%s, %.1fs) — log: %s",
               mode, video_id, status, proc.returncode, elapsed, log_path)
    return JobResult(mode, video_id, status, cmd, str(config_path), str(delta_path),
                      returncode=proc.returncode, log_path=str(log_path), elapsed_seconds=elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# Batch driver
# ─────────────────────────────────────────────────────────────────────────────

def plan_jobs(modes: Sequence[str], videos: Sequence[str]) -> List[Tuple[str, str]]:
    unknown = [m for m in modes if m not in MODE_SPECS]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}; expected one of {sorted(MODE_SPECS)}")
    return [(mode, video_id) for mode in modes for video_id in videos]


def run_batch(
    modes: Sequence[str], videos: Sequence[str], output_root, baseline_root, device: str,
    captions_dir=None, gt_dir=None, configs_dir=None,
    dry_run: bool = False, skip_existing: bool = False, continue_on_error: bool = False,
    python_exe: Optional[str] = None,
) -> List[JobResult]:
    """Run every (mode, video) job in *modes* × *videos*.

    On the first ``"failed"`` job, stops immediately (returning the results
    gathered so far) unless *continue_on_error* is set — this is the "죽일지
    계속 진행할지" switch the task asked for; default is fail-fast so a
    systemic problem (bad weights path, no GPU) doesn't burn through 50
    multi-minute jobs before anyone notices.
    """
    jobs = plan_jobs(modes, videos)
    results: List[JobResult] = []
    for mode, video_id in jobs:
        result = run_job(
            mode, video_id, output_root, baseline_root, device,
            captions_dir=captions_dir, gt_dir=gt_dir, configs_dir=configs_dir,
            dry_run=dry_run, skip_existing=skip_existing, python_exe=python_exe,
        )
        results.append(result)
        if result.status == "failed" and not continue_on_error:
            logger.error("Stopping batch after first failure (%s/%s) — pass --continue-on-error "
                         "to keep going instead.", mode, video_id)
            break
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary (CSV + Markdown)
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARY_METRIC_KEYS = (
    "mean_severity", "ptc", "sfr", "sdi",
    "total_missing_objects", "total_additional_objects", "temporal_hallucination_rate",
)


def discover_result_dirs(output_root) -> List[Tuple[str, str]]:
    """``[(mode, video_id), ...]`` for every ``<output_root>/<mode>/<video_id>``
    directory that looks like a job's output dir (excludes the generated-config
    dir and any pre-existing summary output)."""
    output_root = Path(output_root)
    if not output_root.is_dir():
        return []
    skip = {GENERATED_CONFIGS_DIRNAME}
    pairs = []
    for mode_dir in sorted(p for p in output_root.iterdir() if p.is_dir() and p.name not in skip):
        for video_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
            pairs.append((mode_dir.name, video_dir.name))
    return pairs


def _delta_to_row(mode: str, video_id: str, delta: Dict) -> Dict:
    row = {"video_id": video_id, "mode": mode, "n_items": delta.get("n_items_clip_only")}
    for key in _SUMMARY_METRIC_KEYS:
        row[f"{key}_clip_only"] = delta.get(f"{key}_clip_only")
        row[f"{key}_calibrated"] = delta.get(f"{key}_calibrated")
        row[f"{key}_diff"] = delta.get(f"{key}_diff")
    return row


def summarize_batch(
    output_root, modes: Optional[Sequence[str]] = None, videos: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """Read every completed job's ``heldout/metric_delta.json`` under
    *output_root* and return one summary row per (mode, video_id) — purely
    from files already on disk, no subprocess/model involved (safe to call
    any time, including with nothing having finished yet, in which case it
    returns ``[]``).

    *modes*/*videos*, if given, restrict which (mode, video_id) pairs are
    included — jobs found on disk outside that set are skipped, not errored
    on (a partial batch is a normal thing to summarize).
    """
    rows = []
    mode_filter = set(modes) if modes else None
    video_filter = set(videos) if videos else None
    for mode, video_id in discover_result_dirs(output_root):
        if mode_filter is not None and mode not in mode_filter:
            continue
        if video_filter is not None and video_id not in video_filter:
            continue
        delta_path = metric_delta_path(output_root, mode, video_id)
        if not delta_path.is_file():
            continue
        delta = json.loads(delta_path.read_text(encoding="utf-8"))
        rows.append(_delta_to_row(mode, video_id, delta))
    rows.sort(key=lambda r: (r["mode"], r["video_id"]))
    return rows


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_summary_files(rows: List[Dict], output_root) -> Tuple[Optional[Path], Optional[Path]]:
    """Write ``<output_root>/summary_metrics.csv`` and ``.md``. Returns
    ``(csv_path, md_path)``, or ``(None, None)`` (writes nothing) if *rows*
    is empty — an empty batch produces no misleading empty-but-present file."""
    if not rows:
        return None, None
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())

    csv_path = output_root / f"{SUMMARY_BASENAME}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md_path = output_root / f"{SUMMARY_BASENAME}.md"
    lines = ["# ETRI 10-video OWLv2/VQA/ensemble held-out remeasurement summary", ""]
    lines.append(
        "GT-object-only preservation claims → `ensemble_gt_filter` rows. "
        "Hallucination/additional-object analysis → `ensemble_openworld_filter` rows. "
        "`ensemble_nofilter` is comparison-only (caption-noun-contaminated), not for final claims. "
        "See docs/etri_owlv2_vqa_readiness.md."
    )
    lines.append("")
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join("---" for _ in fields) + "|")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(f)) for f in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return csv_path, md_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _csv_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-drive scripts/remeasure_video_metrics.py (OWLv2/VQA/ensemble, real "
                    "weights) over the 10 ETRI evaluation videos, reusing an already-completed "
                    "real-model baseline's saved frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT),
                   help="Completed real-model baseline batch dir "
                        "(<baseline-root>/<video_id>/{extracted_frames,recon_frames})")
    p.add_argument("--captions-dir", default=str(DEFAULT_CAPTIONS_DIR))
    p.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    p.add_argument("--configs-dir", default=str(DEFAULT_CONFIGS_DIR),
                   help="Directory containing the base configs/etri_video_eval_{owlv2,vqa,ensemble}.yaml")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--videos", default=None,
                   help="Comma-separated video id subset (default: every video id discovered "
                        "under --baseline-root)")
    p.add_argument("--modes", default=None,
                   help=f"Comma-separated mode subset (default: all — {', '.join(MODE_SPECS)})")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate configs and print the commands/output paths that would run, "
                        "without invoking remeasure_video_metrics.py")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a (mode, video) job if its heldout/metric_delta.json already exists")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep running remaining jobs after one fails (default: stop immediately)")
    p.add_argument("--summary-only", action="store_true",
                   help="Skip running any jobs — just (re)generate summary_metrics.csv/.md from "
                        "whatever metric_delta.json files already exist under --output-root")
    p.add_argument("--python", default=None,
                   help="Python interpreter used for the remeasure_video_metrics.py subprocess "
                        "(default: this interpreter, i.e. sys.executable)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    # Resolve to absolute paths up front. build_mode_config() writes these
    # into the generated per-(mode, video) config, which remeasure_video_metrics.py
    # then re-loads via sgdjscc_lab.config.load_config() — any path still
    # relative at that point gets re-resolved relative to the GENERATED
    # config's own directory (outputs/.../_generated_configs/<mode>/), not the
    # CWD this script was invoked from. A relative --output-root (e.g.
    # "tmp_out") would otherwise silently double up
    # (tmp_out/_generated_configs/.../tmp_out/...). Resolving here, once,
    # keeps every downstream path absolute and unambiguous regardless of
    # where load_config() re-resolves it from.
    output_root = Path(args.output_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    captions_dir = Path(args.captions_dir).resolve()
    gt_dir = Path(args.gt_dir).resolve()
    configs_dir = Path(args.configs_dir).resolve()

    modes = _csv_list(args.modes) or list(MODE_SPECS)
    unknown = [m for m in modes if m not in MODE_SPECS]
    if unknown:
        sys.exit(f"Error: unknown mode(s) {unknown}; expected one of {sorted(MODE_SPECS)}")

    if args.summary_only:
        rows = summarize_batch(output_root, modes=modes if args.modes else None,
                               videos=_csv_list(args.videos))
        csv_path, md_path = write_summary_files(rows, output_root)
        if csv_path is None:
            sys.exit(f"Error: no metric_delta.json found under {output_root} — nothing to summarize.")
        print(f"Summary ({len(rows)} row(s)) → {csv_path}\n                        → {md_path}")
        return

    videos_filter = _csv_list(args.videos)
    try:
        videos = discover_videos(baseline_root, videos_filter=videos_filter)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    if not videos:
        sys.exit(f"Error: no video(s) with recon_frames/ found under {baseline_root}")

    logger.info("Batch: %d mode(s) × %d video(s) = %d job(s). modes=%s videos=%s",
               len(modes), len(videos), len(modes) * len(videos), modes, videos)
    for mode in modes:
        spec = MODE_SPECS[mode]
        logger.info("  mode=%-26s purpose: %s%s", mode, spec.purpose,
                   f"  [CAVEAT: {spec.caveat}]" if spec.caveat else "")

    results = run_batch(
        modes, videos, output_root, baseline_root, args.device,
        captions_dir=captions_dir, gt_dir=gt_dir, configs_dir=configs_dir,
        dry_run=args.dry_run, skip_existing=args.skip_existing,
        continue_on_error=args.continue_on_error, python_exe=args.python,
    )

    n_ok = sum(1 for r in results if r.status == "ok")
    n_skipped = sum(1 for r in results if r.status == "skipped")
    n_failed = sum(1 for r in results if r.status == "failed")
    n_dry = sum(1 for r in results if r.status == "dry_run")
    print(f"\nBatch finished: {len(results)}/{len(modes) * len(videos)} job(s) attempted "
         f"— ok={n_ok} skipped={n_skipped} failed={n_failed} dry_run={n_dry}")
    for r in results:
        if r.status == "failed":
            print(f"  FAILED  {r.mode}/{r.video_id}  (returncode={r.returncode})  log={r.log_path}")

    if args.dry_run:
        print(f"\nDry-run only — no remeasurement was executed. Generated configs under "
             f"{output_root / GENERATED_CONFIGS_DIRNAME}")
        return

    rows = summarize_batch(output_root, modes=modes, videos=videos)
    csv_path, md_path = write_summary_files(rows, output_root)
    if csv_path is not None:
        print(f"Summary ({len(rows)} row(s)) → {csv_path}\n                        → {md_path}")

    if n_failed and not args.continue_on_error:
        sys.exit(f"{n_failed} job(s) failed and --continue-on-error was not set — batch stopped early.")
    elif n_failed:
        sys.exit(f"{n_failed} job(s) failed (see run.log per job) — batch continued (--continue-on-error).")


if __name__ == "__main__":
    main()
