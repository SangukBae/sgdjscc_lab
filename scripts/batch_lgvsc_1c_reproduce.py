#!/usr/bin/env python
"""scripts/batch_lgvsc_1c_reproduce.py – ETRI 후속 1단계 step 1C
LGVSC-reproduction-baseline batch driver.

Scope note (read before touching this file)
--------------------------------------------
1A built the Rx-legal segment-level generation contract, 1B connected real
generative backends (mock / SVD / Wan start-only / Wan bidirectional) behind
that contract and verified each on real GPU. **1C's job is not to run the
actual validation** — that is explicitly the user's call, done by hand with
the commands this driver prints/generates — **it is to make that validation
one command away**: isolated per-mode/per-video configs, a batch driver that
never overwrites another run's output, and a summary table generator that
turns whatever is on disk into a comparison CSV/Markdown.

**Nothing here claims a faithful LGVSC reproduction.** LGVSC's paper
components (`SKIM`/`SKEM` keyframe selection, `SFA`/`DSA` decoder
adapters, PSSS scoring, side-info encoding) are not publicly released in
enough detail to reproduce byte-for-byte. What this driver runs are
**"LGVSC-style reproducible baselines"** — real, runnable pipelines chosen to
sit at increasing distance from LGVSC's stated design, each documented with
exactly which part is a genuine analog and which part is an approximation.
See docs/lgvsc_1c_reproduction_readiness.md (the original four modes) and
docs/lgvsc_psss_skem_readiness.md (the PSSS/SKEM modes added after it) for
the full mapping — the one-line version:

- ``mock_baseline``   — no real generative model (BidirectionalInterpolationGenerator,
                        a linear blend). The "zero real generation" floor.
- ``svd_start_only``  — real diffusion, image-only conditioning (no caption,
                        no end-keyframe). Cannot represent SFA or DSA (no
                        text conditioning at all).
- ``wan_skim_sfa``    — real diffusion, start-keyframe + caption conditioning.
                        Nearest reproducible analog of LGVSC's SFA
                        (start-frame adapter) side of SKIM+SFA. Keyframe
                        SELECTION is the pre-existing fixed/scene-change
                        extractor (see below) — same as wan_skem_dsa.
- ``wan_skem_dsa``    — real diffusion, start+end-keyframe + caption
                        conditioning. Nearest reproducible analog of LGVSC's
                        DSA (dual-side adapter) side of SKEM+DSA. Same
                        keyframe selection as wan_skim_sfa.
- ``skim_sfa_fixed``  — like wan_skim_sfa, but keyframe SELECTION is REPLACED
                        by `keyframe.selector: fixed_interval`
                        (`FixedIntervalKeyframeSelector` — zero scene-change
                        signal, the paper's literal SKIM). The canonical
                        SKIM/SFA side of the SKIM/SFA-vs-SKEM/DSA comparison
                        pair below. NOTE: its static `interval` only bounds
                        the WORST-CASE segment length the same way SKEM's
                        `max_segment_length` does — it does not, by itself,
                        match SKEM's ACTUAL (data-dependent, usually smaller)
                        keyframe count/CBR for a given video. Use
                        ``--keyframe-count-match-from`` (see below) to replace
                        the interval selector with an exact-count, near-equal
                        partition calibrated from an already-run SKEM mode.
                        This matches keyframe count; measured CBR still needs
                        channel-symbol accounting.
- ``skem_dsa_psss``   — like wan_skem_dsa, but keyframe SELECTION is REPLACED
                        by `keyframe.selector: psss` (SKEM,
                        src/sgdjscc_lab/video/skem_selector.py) with a REAL
                        MLLM PSSS backend — genuinely variable-length
                        segments driven by semantic divergence, not a fixed
                        interval. The canonical SKEM/DSA side of the
                        comparison pair; real GPU + real MLLM required.
- ``skem_dsa_mock_psss`` / ``skem_dsa_proxy_psss`` — CPU-only diagnostic
                        twins of skem_dsa_psss with a mock/proxy PSSS backend
                        and a mock (bidirectional-interpolation) decoder —
                        prove the selector wiring/variable-length-segment
                        machinery structurally without any GPU generation
                        cost. NEVER cite either as evidence of real PSSS/SKEM
                        quality.

**Keyframe SELECTION (the SKIM-vs-SKEM distinction) is identical across
mock_baseline/svd_start_only/wan_skim_sfa/wan_skem_dsa** — those four modes
all use this repository's pre-existing keyframe extractor (`keyframe.max_gop`
+ scene-change detector, `configs/base/video/default.yaml`), not a semantic/
PSSS-driven selector. The four PSSS/SKEM-readiness modes above (skim_sfa_fixed
plus the three skem_dsa_* modes) are what actually distinguish the two
selectors — skim_sfa_fixed keeps the fixed extractor, the skem_dsa_* modes use
`video/skem_selector.py::PsssKeyframeSelector`. Only the GENERATE-branch
decoder conditioning differs between the *_skim_sfa/*_sfa_fixed family (single
keyframe) and the *_skem_dsa family (two keyframes, when available — see the
last-open-GOP fallback in each mode's own config header). `side_infos`
(motion/delta dicts — the closer analog of a real side-info adapter) are
accepted by the manifest but not folded into Wan/SVD conditioning by any mode
(see scripts/lgvsc_generate_worker.py::run_wan_backend's docstring) — a
documented gap, not a silent omission.

Each mode's config (``configs/experiments/lgvsc_1c/etri_lgvsc_1c_<mode>.yaml``) is copied from an
already real-GPU-verified 1B config — ``wan_skim_sfa`` from
``configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_start_only.yaml``, ``wan_skem_dsa``
from ``configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_fixed.yaml`` —
so this driver never introduces new, unverified generation-backend wiring;
it only orchestrates the already-working per-video/per-mode config
generation, subprocess dispatch, and result aggregation.

Output layout (default --output-root outputs/etri_video_eval/lgvsc_1c_reproduce)
---------------------------------------------------------------------------
    outputs/etri_video_eval/lgvsc_1c_reproduce/
      mock_baseline/<video_id>/        recon.mp4, temporal_metrics.csv,
                                       segments.json, generated_frames/, run.log
      svd_start_only/<video_id>/       (same file set)
      wan_skim_sfa/<video_id>/         (same file set)
      wan_skem_dsa/<video_id>/         (same file set)
      _generated_configs/<mode>/<video_id>.yaml   (what evaluate_video.py's
                                                    --config actually points at)
      batch_status.json                per-(mode,video) run bookkeeping
      summary_metrics.csv / .md / .json
      summary_aggregate_comparison.csv / .md / .json   skim_sfa_fixed vs
                                       skem_dsa_psss per-video + MEAN row
                                       (PSSS/SKEM readiness step — see
                                       build_aggregate_comparison())

Every run writes only inside its own ``<mode>/<video_id>/`` directory and its
own ``_generated_configs/<mode>/<video_id>.yaml`` — no two (mode, video) jobs
ever share a path, so runs are safe to re-invoke/resume/parallelize by hand.

``--no-models`` note
--------------------
``--no-models`` (forwarded to ``scripts/evaluate_video.py``) only disables
SGD-JSCC's own keyframe reconstruction models (identity reconstruction
instead) — it has **no effect** on the generate-branch worker for
``svd_start_only``/``wan_skim_sfa``/``wan_skem_dsa``, which is a fully
separate subprocess in a different conda env (``semantic-diffusers`` by
default) that still loads and runs its real generative model regardless.
Passing ``--no-models`` to this driver therefore still exercises real GPU
generation for those three modes — it just skips the (comparatively slow)
Rx-side diffusion reconstruction so the run finishes faster. Only
``mock_baseline`` has no real-model cost either way.

Usage
-----
# Show what WOULD run (no subprocess, no GPU) for two videos, one mode:
python scripts/batch_lgvsc_1c_reproduce.py --modes wan_skim_sfa \\
    --videos 01_person_walk,05_camera_pan_person --max-frames 14 --dry-run

# Actual smoke run — cheap, no GPU, mock backend only:
python scripts/batch_lgvsc_1c_reproduce.py --modes mock_baseline \\
    --videos 01_person_walk --max-frames 14 --no-models

# Real GPU validation across all 4 modes / all 10 videos (the user runs this):
python scripts/batch_lgvsc_1c_reproduce.py --modes all --device cuda:0

# Regenerate the summary table from whatever is already on disk, no new runs:
python scripts/batch_lgvsc_1c_reproduce.py --summary-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sgdjscc_lab.paths import model_root as _model_root  # noqa: E402
from sgdjscc_lab.paths import run_root as _run_root  # noqa: E402

MODES = (
    "mock_baseline", "svd_start_only", "wan_skim_sfa", "wan_skem_dsa",
    # PSSS/SKEM readiness step: the SKIM/SFA-vs-SKEM/DSA comparison pair
    # (skim_sfa_fixed / skem_dsa_psss, both required by the task brief) plus
    # two CPU-only diagnostic twins of skem_dsa_psss that swap in a mock/proxy
    # PSSS backend instead of a real MLLM — see
    # docs/lgvsc_psss_skem_readiness.md before treating any of these four as
    # more than what their name says.
    "skim_sfa_fixed", "skem_dsa_psss", "skem_dsa_mock_psss", "skem_dsa_proxy_psss",
)

_LGVSC_1C_CONFIG_DIR = _REPO_ROOT / "configs" / "experiments" / "lgvsc_1c"

_MODE_CONFIG = {
    "mock_baseline": _LGVSC_1C_CONFIG_DIR / "etri_lgvsc_1c_mock_baseline.yaml",
    "svd_start_only": _LGVSC_1C_CONFIG_DIR / "etri_lgvsc_1c_svd_start_only.yaml",
    "wan_skim_sfa": _LGVSC_1C_CONFIG_DIR / "etri_lgvsc_1c_wan_skim_sfa.yaml",
    "wan_skem_dsa": _LGVSC_1C_CONFIG_DIR / "etri_lgvsc_1c_wan_skem_dsa.yaml",
    "skim_sfa_fixed": _LGVSC_1C_CONFIG_DIR / "etri_lgvsc_1c_skim_sfa_fixed.yaml",
    "skem_dsa_psss": _LGVSC_1C_CONFIG_DIR / "etri_lgvsc_1c_skem_dsa_psss.yaml",
    "skem_dsa_mock_psss": _LGVSC_1C_CONFIG_DIR / "etri_lgvsc_1c_skem_dsa_mock_psss.yaml",
    "skem_dsa_proxy_psss": _LGVSC_1C_CONFIG_DIR / "etri_lgvsc_1c_skem_dsa_proxy_psss.yaml",
}

# SKIM/SFA-vs-SKEM/DSA comparison axis (PSSS/SKEM readiness step): which
# "family" each mode belongs to, for build_aggregate_comparison() below. Modes
# not on this axis at all (mock_baseline/svd_start_only, which don't
# distinguish a keyframe selector) are simply absent.
_SKIM_SKEM_FAMILY = {
    "wan_skim_sfa": "skim_sfa", "skim_sfa_fixed": "skim_sfa",
    "wan_skem_dsa": "skem_dsa", "skem_dsa_psss": "skem_dsa",
    "skem_dsa_mock_psss": "skem_dsa", "skem_dsa_proxy_psss": "skem_dsa",
}

# Same fragment set every etri_lgvsc_1c_*.yaml / etri_video_eval_lgvsc_worker_*.yaml
# config composes via `_defaults_` (configs/base/video/default.yaml being the one
# that turns on use_temporal / keyframe / video_generator machinery).
_FRAGMENTS = (
    "base/channel/awgn", "base/model/sgdjscc", "base/infer/awgn",
    "base/eval/default", "base/video/default",
)

_SUMMARY_METRIC_FIELDS = (
    "n_frames", "n_keyframes", "n_interframes", "n_generate", "n_reused",
    "n_recompute_semantic", "n_recompute_motion", "temporal_srs", "srs_flicker",
    "ptc", "sfr", "sdi", "temporal_hallucination_rate",
    "transmitted_units", "naive_units", "overhead_reduction",
)


def mode_config_path(mode: str) -> Path:
    """Return the base template config path for *mode* (raises ValueError for
    an unknown mode — this is the single source of truth callers/tests should
    use rather than re-deriving the mode → config mapping themselves)."""
    if mode not in _MODE_CONFIG:
        raise ValueError(f"Unknown 1C mode {mode!r}; expected one of {MODES}")
    return _MODE_CONFIG[mode]


# ──────────────────────────────────────────────────────────────────────────────
# Video discovery
# ──────────────────────────────────────────────────────────────────────────────

def read_manifest(data_root: Path) -> list:
    """Parse data/etri_video_eval/manifest.csv into per-video entries (same
    shape as scripts/run_etri_video_eval.py's read_manifest, re-implemented
    locally so this driver stays a standalone script)."""
    data_root = Path(data_root)
    manifest = data_root / "manifest.csv"
    entries = []
    with open(manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            processed = data_root / row["processed_file"]
            key = processed.stem
            captions = data_root / "captions" / f"{key}.txt"
            entries.append({
                "key": key,
                "processed": processed,
                "captions": captions if captions.exists() else None,
                "row": dict(row),
            })
    return entries


def select_videos(entries: list, videos: str = None) -> list:
    if not videos:
        return entries
    wanted = {v.strip() for v in videos.split(",") if v.strip()}
    selected = [e for e in entries
                if e["key"] in wanted or e["row"].get("id") in wanted or e["row"].get("name") in wanted]
    if not selected:
        raise ValueError(f"--videos matched nothing among {sorted(wanted)}")
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Per-run config generation
# ──────────────────────────────────────────────────────────────────────────────

def out_dir_for(output_root: Path, mode: str, video_key: str) -> Path:
    """Isolated output directory for (mode, video) — never shared with any
    other (mode, video) combination."""
    return Path(output_root) / mode / video_key


def generated_config_path(output_root: Path, mode: str, video_key: str) -> Path:
    return Path(output_root) / "_generated_configs" / mode / f"{video_key}.yaml"


def compute_cbr_matched_interval(n_frames: int, n_keyframes: int) -> int:
    """Return an integer interval only when it exactly represents the count.

    This compatibility helper is deliberately *not* used for matched runs:
    many ``(n_frames, n_keyframes)`` pairs have no exact integer interval
    (10 frames / 6 keyframes is one example). Matched runs use
    ``FixedCountKeyframeSelector`` instead.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be > 0; got {n_frames}")
    if n_keyframes <= 0:
        raise ValueError(f"n_keyframes must be > 0; got {n_keyframes}")
    if n_keyframes > n_frames:
        raise ValueError(
            f"n_keyframes ({n_keyframes}) cannot exceed n_frames ({n_frames})."
        )
    for interval in range(1, n_frames + 1):
        if math.ceil(n_frames / interval) == n_keyframes:
            return interval
    raise ValueError(
        f"No integer fixed interval produces exactly {n_keyframes} keyframes "
        f"from {n_frames} frames; use FixedCountKeyframeSelector."
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_cbr_match(
    output_root: Path,
    source_mode: str,
    video_key: str,
    *,
    target_n_frames: int = None,
) -> dict:
    """Look up *source_mode*'s ALREADY-RUN ``keyframes.json`` for *video_key*
    and plan a ``FixedCountKeyframeSelector`` with the same keyframe count.
    ``target_n_frames`` must describe the target invocation after
    ``--max-frames`` truncation; a source/target mismatch is a hard
    precondition failure rather than a silent static-interval fallback.

    Never raises for missing/malformed source state. It returns a diagnostic
    status, and an actual matched run treats that status as a failed
    precondition instead of silently running the static interval baseline.
    """
    source_path = out_dir_for(output_root, source_mode, video_key) / "keyframes.json"
    data = _read_json(source_path)
    if data is None:
        return {
            "source_mode": source_mode, "status": "source_missing",
            "source_keyframes_json": str(source_path),
            "source_n_frames": None, "source_n_keyframes": None,
            "target_n_frames": target_n_frames, "requested_keyframes": None,
            "source_sha256": None, "matched_interval": None,
        }
    n_keyframes = len(data.get("keyframes") or [])
    n_frames = len(data.get("boundaries") or data.get("frame_roles") or [])
    if n_keyframes <= 0 or n_frames <= 0:
        return {
            "source_mode": source_mode, "status": "source_empty",
            "source_keyframes_json": str(source_path),
            "source_n_frames": n_frames, "source_n_keyframes": n_keyframes,
            "target_n_frames": target_n_frames, "requested_keyframes": None,
            "source_sha256": _sha256_file(source_path), "matched_interval": None,
        }
    source_metrics_path = out_dir_for(
        output_root, source_mode, video_key
    ) / "temporal_metrics.csv"
    source_metrics = _read_single_row_csv(source_metrics_path)
    if source_metrics is None:
        return {
            "source_mode": source_mode, "status": "source_incomplete",
            "source_keyframes_json": str(source_path),
            "source_n_frames": n_frames, "source_n_keyframes": n_keyframes,
            "target_n_frames": target_n_frames, "requested_keyframes": None,
            "source_sha256": _sha256_file(source_path),
            "source_config_sha256": None, "matched_interval": None,
        }
    try:
        metrics_n_frames = int(float(source_metrics.get("n_frames")))
    except (TypeError, ValueError):
        metrics_n_frames = None
    if metrics_n_frames != n_frames:
        return {
            "source_mode": source_mode, "status": "source_inconsistent",
            "source_keyframes_json": str(source_path),
            "source_n_frames": n_frames, "source_metrics_n_frames": metrics_n_frames,
            "source_n_keyframes": n_keyframes, "target_n_frames": target_n_frames,
            "requested_keyframes": None, "source_sha256": _sha256_file(source_path),
            "source_config_sha256": None, "matched_interval": None,
        }
    source_cfg_path = generated_config_path(output_root, source_mode, video_key)
    source_cfg_sha256 = (
        _sha256_file(source_cfg_path) if source_cfg_path.is_file() else None
    )
    source_metrics_sha256 = _sha256_file(source_metrics_path)
    if n_keyframes > n_frames:
        return {
            "source_mode": source_mode, "status": "source_invalid",
            "source_keyframes_json": str(source_path),
            "source_n_frames": n_frames, "source_n_keyframes": n_keyframes,
            "target_n_frames": target_n_frames, "requested_keyframes": None,
            "source_sha256": _sha256_file(source_path),
            "source_temporal_metrics_csv": str(source_metrics_path),
            "source_temporal_metrics_sha256": source_metrics_sha256,
            "source_config_path": str(source_cfg_path),
            "source_config_sha256": source_cfg_sha256, "matched_interval": None,
        }
    if target_n_frames is not None and int(target_n_frames) != n_frames:
        return {
            "source_mode": source_mode, "status": "frame_count_mismatch",
            "source_keyframes_json": str(source_path),
            "source_n_frames": n_frames, "source_n_keyframes": n_keyframes,
            "target_n_frames": int(target_n_frames), "requested_keyframes": n_keyframes,
            "source_sha256": _sha256_file(source_path),
            "source_temporal_metrics_csv": str(source_metrics_path),
            "source_temporal_metrics_sha256": source_metrics_sha256,
            "source_config_path": str(source_cfg_path),
            "source_config_sha256": source_cfg_sha256, "matched_interval": None,
        }
    return {
        "source_mode": source_mode, "status": "count_planned",
        "source_keyframes_json": str(source_path),
        "source_n_frames": n_frames, "source_n_keyframes": n_keyframes,
        "target_n_frames": n_frames if target_n_frames is None else int(target_n_frames),
        "requested_keyframes": n_keyframes,
        "source_sha256": _sha256_file(source_path),
        "source_temporal_metrics_csv": str(source_metrics_path),
        "source_temporal_metrics_sha256": source_metrics_sha256,
        "source_config_path": str(source_cfg_path),
        "source_config_sha256": source_cfg_sha256,
        "matched_interval": None,
    }


def build_run_config(mode: str, out_dir: Path, cbr_match: dict = None, *,
                     worker_device_map: str = None,
                     worker_max_memory: dict = None) -> dict:
    """Load ``configs/experiments/lgvsc_1c/etri_lgvsc_1c_<mode>.yaml`` and rewrite it into a
    per-video-ready config dict.

    Only three kinds of fields are rewritten:
    - ``_defaults_`` → absolute fragment paths (the generated config lives
      under ``<output_root>/_generated_configs/<mode>/``, several directory
      levels away from ``configs/``, so the base template's relative
      ``_defaults_: [channel/awgn, ...]`` would otherwise resolve to a
      nonexistent path — same fix as scripts/run_etri_video_eval.py's
      ``build_run_config``).
    - ``model_root`` → absolute (same reason).
    - Output-path fields (``keyframe_json``/``segment_json``/``temporal_csv``/
      ``frame_log_csv``/``video_io.*``/``video_generator.generated_frames_dir``)
      → **absolute** paths under ``out_dir`` (NOT bare relative filenames —
      config.py resolves relative paths against the config FILE's own
      directory, which here is ``_generated_configs/<mode>/``, not
      ``out_dir``; a bare filename would land every video's output in the
      same shared ``_generated_configs/<mode>/`` directory instead of its own
      isolated ``out_dir``, silently colliding across videos).

    Everything else — most importantly ``video_generator.backend``/
    ``conditioning_mode``/``worker.*`` — is carried over UNCHANGED from the
    base template unless a worker placement override is explicitly supplied.
    A placement override preserves backend/model-specific extra JSON (such as
    ``bidirectional_model_id``), replaces CPU offload with Diffusers' pipeline
    ``device_map``, and optionally supplies per-device memory limits.
    ``input_path`` is deliberately left unset
    here — the driver passes ``--input``/``--captions`` as CLI flags (see
    build_command), matching how every other 1B/1C config in this repo is
    actually invoked.

    *cbr_match* (from :func:`resolve_cbr_match`, via the keyframe-count-match
    CLI option):
    when given AND this mode's own ``keyframe.selector`` is
    ``"fixed_interval"``, replaces that selector with ``"fixed_count"`` and
    sets ``keyframe.fixed_count.count``. This guarantees the exact count even
    when no integer interval can do so. The plan is recorded under
    ``_keyframe_count_match`` and verified against the target keyframes.json
    after the subprocess completes.
    """
    from omegaconf import OmegaConf

    base_path = mode_config_path(mode)
    cfg = OmegaConf.to_container(OmegaConf.load(base_path), resolve=False)
    if not isinstance(cfg, dict):
        raise ValueError(f"{base_path} did not parse to a mapping")

    if cbr_match is not None:
        kf = dict(cfg.get("keyframe") or {})
        if str(kf.get("selector")) == "fixed_interval":
            cfg["_keyframe_count_match"] = dict(cbr_match)
            if cbr_match.get("status") == "count_planned":
                kf["selector"] = "fixed_count"
                kf["fixed_count"] = {
                    "count": int(cbr_match["requested_keyframes"]),
                }
                cfg["keyframe"] = kf

    out_dir = Path(out_dir)
    cfg["_defaults_"] = [str((_REPO_ROOT / "configs" / f)) for f in _FRAGMENTS]
    cfg["model_root"] = str(_model_root())
    cfg["keyframe_json"] = str(out_dir / "keyframes.json")
    cfg["segment_json"] = str(out_dir / "segments.json")
    cfg["temporal_csv"] = str(out_dir / "temporal_metrics.csv")
    cfg["frame_log_csv"] = str(out_dir / "temporal_frames.csv")
    cfg["video_io"] = {
        "extracted_frames_dir": str(out_dir / "extracted_frames"),
        "recon_frames_dir": str(out_dir / "recon_frames"),
        "save_recon_frames": True,
        "save_recon_video": True,
        "recon_video": str(out_dir / "recon.mp4"),
    }
    vg = dict(cfg.get("video_generator") or {})
    vg["generated_frames_dir"] = str(out_dir / "generated_frames")
    if worker_device_map is not None or worker_max_memory is not None:
        worker = dict(vg.get("worker") or {})
        if worker_device_map is None:
            raise ValueError("worker_max_memory requires worker_device_map")
        # SVD and mock modes do not use the Wan placement contract. This lets
        # --modes all apply one command-line override only where it is valid.
        if str(worker.get("backend")) == "wan":
            try:
                extra = json.loads(worker.get("extra_json") or "{}")
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"Mode {mode!r} has invalid worker.extra_json"
                ) from exc
            if not isinstance(extra, dict):
                raise ValueError(f"Mode {mode!r} worker.extra_json must be a JSON object")
            extra.pop("offload_mode", None)
            extra["device_map"] = worker_device_map
            if worker_max_memory is not None:
                extra["max_memory"] = dict(worker_max_memory)
            worker["extra_json"] = json.dumps(extra, separators=(",", ":"))
            extra_env = dict(worker.get("extra_env") or {})
            extra_env["HF_ENABLE_PARALLEL_LOADING"] = "YES"
            worker["extra_env"] = extra_env
            vg["worker"] = worker
    cfg["video_generator"] = vg
    return cfg


def _write_yaml(cfg: dict, path: Path) -> None:
    from omegaconf import OmegaConf
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), str(path))


def build_command(cfg_path: Path, input_path: Path, captions_path=None, *,
                   device: str = None, max_frames: int = None,
                   no_models: bool = False, save_video: bool = True,
                   recon_caption_mode: str = "own") -> list:
    """Assemble the evaluate_video.py subprocess argv — matches exactly the
    command shape documented in this module's docstring / the 1C task brief:
    --config, --input, --captions, --device, --max-frames, --save-video, and
    optionally --no-models/--recon-caption-mode.

    *recon_caption_mode*: forwarded as --recon-caption-mode only when
    "skip" (the evaluate_video.py default is already "own", so omitting the
    flag there is equivalent — but run_job always passes it explicitly,
    derived from the config's use_text, so the log/cmd string never claims
    "own" while BLIP2 is actually disabled)."""
    cmd = [
        sys.executable, str(_REPO_ROOT / "scripts" / "evaluate_video.py"),
        "--config", str(cfg_path),
        "--input", str(input_path),
    ]
    if captions_path is not None:
        cmd += ["--captions", str(captions_path)]
    if device is not None:
        cmd += ["--device", str(device)]
    if max_frames is not None:
        cmd += ["--max-frames", str(max_frames)]
    if save_video:
        cmd.append("--save-video")
    if no_models:
        cmd.append("--no-models")
    if recon_caption_mode == "skip":
        cmd += ["--recon-caption-mode", "skip"]
    return cmd


# ──────────────────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────────────────────

def _target_n_frames(entry: dict, max_frames: int = None):
    raw = (entry.get("row") or {}).get("n_frames")
    try:
        n_frames = int(raw)
    except (TypeError, ValueError):
        return None
    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))
    return n_frames


def verify_keyframe_count_match(out_dir: Path, plan: dict) -> dict:
    """Verify a completed fixed-count run against its source artifact."""
    result = dict(plan)
    source_path = Path(plan.get("source_keyframes_json") or "")
    if not source_path.is_file():
        result["status"] = "source_missing_after_run"
        return result
    if plan.get("source_sha256") and _sha256_file(source_path) != plan["source_sha256"]:
        result["status"] = "source_changed"
        return result
    source_metrics_path = Path(plan.get("source_temporal_metrics_csv") or "")
    if plan.get("source_temporal_metrics_sha256"):
        if not source_metrics_path.is_file():
            result["status"] = "source_metrics_missing_after_run"
            return result
        if _sha256_file(source_metrics_path) != plan["source_temporal_metrics_sha256"]:
            result["status"] = "source_metrics_changed"
            return result
    source_config_path = Path(plan.get("source_config_path") or "")
    if plan.get("source_config_sha256"):
        if not source_config_path.is_file():
            result["status"] = "source_config_missing_after_run"
            return result
        if _sha256_file(source_config_path) != plan["source_config_sha256"]:
            result["status"] = "source_config_changed"
            return result

    target_path = Path(out_dir) / "keyframes.json"
    target = _read_json(target_path)
    if target is None:
        result["status"] = "target_missing"
        return result

    actual_keyframes = len(target.get("keyframes") or [])
    actual_frames = len(target.get("boundaries") or target.get("frame_roles") or [])
    requested = int(plan["requested_keyframes"])
    expected_frames = int(plan["target_n_frames"])
    result.update({
        "target_keyframes_json": str(target_path),
        "actual_target_n_frames": actual_frames,
        "actual_fixed_keyframes": actual_keyframes,
        "keyframe_count_delta": actual_keyframes - requested,
    })
    if actual_frames != expected_frames:
        result["status"] = "target_frame_count_mismatch"
    elif actual_keyframes != requested:
        result["status"] = "keyframe_count_mismatch"
    else:
        result["status"] = "keyframe_count_verified"
    return result


def run_job(mode: str, entry: dict, output_root: Path, *, device=None, max_frames=None,
            no_models: bool = False, skip_existing: bool = False, dry_run: bool = False,
            cbr_match_from: str = None, worker_device_map: str = None,
            worker_max_memory: dict = None) -> dict:
    """Generate the per-video config, then either print (dry_run) or actually
    run the evaluate_video.py subprocess for one (mode, video) job.

    *cbr_match_from*: an already-run mode name (typically ``skem_dsa_psss``)
    whose ACTUAL keyframe count for this video should configure a
    ``fixed_count`` baseline — see
    :func:`resolve_cbr_match`/:func:`build_run_config`. ``None`` (default)
    leaves every mode's config exactly as its own template specifies.

    Returns a batch_status.json row: {mode, video, status, out_dir, cmd,
    returncode, duration_sec}. ``status`` is one of "ok"/"failed"/"skipped"/
    "dry_run".
    """
    video_key = entry["key"]
    out_dir = out_dir_for(output_root, mode, video_key)
    cfg_path = generated_config_path(output_root, mode, video_key)
    marker = out_dir / "temporal_metrics.csv"

    cbr_match = (
        resolve_cbr_match(
            output_root, cbr_match_from, video_key,
            target_n_frames=_target_n_frames(entry, max_frames),
        )
        if cbr_match_from else None
    )
    cfg = build_run_config(
        mode, out_dir, cbr_match=cbr_match,
        worker_device_map=worker_device_map,
        worker_max_memory=worker_max_memory,
    )

    match_plan = cfg.get("_keyframe_count_match")
    if skip_existing and marker.exists():
        if match_plan is None:
            return {
                "mode": mode, "video": video_key, "status": "skipped",
                "out_dir": str(out_dir), "cmd": None, "returncode": None, "duration_sec": 0.0,
            }
        verification = verify_keyframe_count_match(out_dir, match_plan)
        if verification.get("status") == "keyframe_count_verified":
            return {
                "mode": mode, "video": video_key, "status": "skipped",
                "out_dir": str(out_dir), "cmd": None, "returncode": None,
                "duration_sec": 0.0, "keyframe_count_match": verification,
            }

    _write_yaml(cfg, cfg_path)

    # use_text: false (see the four etri_lgvsc_1c_*.yaml headers — added to fit
    # the main SGD-JSCC reconstruction pipeline + external generate worker on a
    # single 16GB card) disables BLIP2 loading, so recon-side "own" captioning
    # would silently degrade to no-caption anyway; label that explicitly as
    # "skip" rather than leaving the log/cmd claiming the "own" default.
    recon_caption_mode = "skip" if not cfg.get("use_text", True) else "own"
    cmd = build_command(
        cfg_path, entry["processed"], entry.get("captions"),
        device=device, max_frames=max_frames, no_models=no_models, save_video=True,
        recon_caption_mode=recon_caption_mode,
    )

    if dry_run:
        print(f"[DRY-RUN] mode={mode} video={video_key}")
        print(f"  config: {cfg_path}")
        print(f"  out_dir: {out_dir}")
        print(f"  cmd: {' '.join(cmd)}")
        if cbr_match is not None:
            print(f"  cbr_match: {cbr_match}")
        return {
            "mode": mode, "video": video_key, "status": "dry_run",
            "out_dir": str(out_dir), "cmd": " ".join(cmd), "returncode": None, "duration_sec": 0.0,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    if match_plan is not None and match_plan.get("status") != "count_planned":
        message = (
            "Keyframe-count match precondition failed: "
            f"{match_plan.get('status')} (source={cbr_match_from}, video={video_key})."
        )
        log_path.write_text(message + "\n", encoding="utf-8")
        (out_dir / "keyframe_count_match.json").write_text(
            json.dumps(match_plan, indent=2), encoding="utf-8",
        )
        return {
            "mode": mode, "video": video_key, "status": "failed",
            "out_dir": str(out_dir), "cmd": " ".join(cmd), "returncode": 2,
            "duration_sec": 0.0, "error": message,
            "keyframe_count_match": match_plan,
        }

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# mode={mode} video={video_key}\n# cmd: {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(_REPO_ROOT))
    verification = None
    status = "ok" if proc.returncode == 0 else "failed"
    if proc.returncode == 0 and match_plan is not None:
        verification = verify_keyframe_count_match(out_dir, match_plan)
        (out_dir / "keyframe_count_match.json").write_text(
            json.dumps(verification, indent=2), encoding="utf-8",
        )
        if verification.get("status") != "keyframe_count_verified":
            status = "failed"
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(
                    "\nKeyframe-count postcondition failed: "
                    f"{verification.get('status')}\n"
                )

    result = {
        "mode": mode, "video": video_key,
        "status": status,
        "out_dir": str(out_dir), "cmd": " ".join(cmd),
        "returncode": proc.returncode, "duration_sec": round(time.time() - t0, 2),
    }
    if verification is not None:
        result["keyframe_count_match"] = verification
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Summary generation
# ──────────────────────────────────────────────────────────────────────────────

def _read_single_row_csv(path: Path):
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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _num(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f == int(f) and "." not in str(v) else f


def _read_generated_config(output_root: Path, mode: str, video_id: str) -> dict:
    """Best-effort read of the per-video generated config (the one actually
    passed to evaluate_video.py --config) for keyframe.selector/psss.* and
    video_generator.conditioning_mode provenance. Returns {} if missing/
    unparseable (e.g. a --dry-run job, or one that hasn't been generated yet)
    — every caller below treats an empty dict as "assume the fixed-selector
    default", matching build_keyframe_extractor()'s own default."""
    path = generated_config_path(output_root, mode, video_id)
    if not path.exists():
        return {}
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=False) or {}
    except Exception:  # noqa: BLE001
        return {}


def _segment_length(seg: dict) -> int:
    return 1 + len(seg.get("inter_frame_indices") or [])


def _mean(vals):
    return (sum(vals) / len(vals)) if vals else None


def _stdev(vals):
    if len(vals) < 2:
        return 0.0 if vals else None
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _accounting_for_run(out_dir: Path):
    """Read only this run's accounting artifact.

    Never fall back to a mode-level parent directory: that could associate one
    video's accounting with another and falsely report a CBR match.
    """
    path = Path(out_dir) / "accounting" / "accounting_summary.json"
    data = _read_json(path)
    if not data or data.get("total_channel_symbols") is None:
        return None
    return {
        "path": str(path),
        "total_channel_symbols": _num(data["total_channel_symbols"]),
        "proxy_fraction": _num(data.get("proxy_fraction")),
    }


def collect_run_metrics(output_root: Path, mode: str, video_id: str, status_by_id: dict = None) -> dict:
    """Build one summary_metrics row for (mode, video_id) from whatever is on
    disk under output_root/mode/video_id/ — works even when no batch_status.json
    is available (pure post-hoc, disk-only reconstruction), which is what lets
    --summary-only regenerate the table without re-running anything."""
    out_dir = out_dir_for(output_root, mode, video_id)
    status_by_id = status_by_id or {}
    prior = status_by_id.get((mode, video_id))

    row = {"mode": mode, "video_id": video_id}
    tm = _read_single_row_csv(out_dir / "temporal_metrics.csv")

    if prior is not None:
        status = prior.get("status")
    elif tm is not None:
        status = "ok"
    elif (out_dir / "run.log").exists():
        status = "failed"
    else:
        status = "missing"
    row["status"] = status

    for f in _SUMMARY_METRIC_FIELDS:
        row[f] = _num(tm.get(f)) if tm else None

    gen_dir = out_dir / "generated_frames"
    row["generated_frame_count"] = len(list(gen_dir.glob("generated_*.png"))) if gen_dir.is_dir() else 0

    segments = _read_json(out_dir / "segments.json") or []
    conditioning_modes, backends, has_end_kf = set(), set(), False
    for seg in segments:
        gen = seg.get("generation")
        if not gen:
            continue
        if gen.get("conditioning_mode"):
            conditioning_modes.add(gen["conditioning_mode"])
        if gen.get("backend"):
            backends.add(gen["backend"])
        if gen.get("end_keyframe_index") is not None:
            has_end_kf = True
    row["conditioning_modes_observed"] = ";".join(sorted(conditioning_modes))
    row["backends_observed"] = ";".join(sorted(backends))
    row["has_end_keyframe"] = has_end_kf

    # ── PSSS/SKEM readiness step: selector/segment-length/PSSS-score/
    # conditioning-mode-breakdown fields, so a skim_sfa_* vs skem_dsa_*
    # comparison table can be built purely from summary_metrics rows. ────────
    gen_cfg = _read_generated_config(output_root, mode, video_id)
    kf_cfg = dict(gen_cfg.get("keyframe") or {})
    vg_cfg = dict(gen_cfg.get("video_generator") or {})
    selector_backend = str(kf_cfg.get("selector", "fixed"))
    conditioning_mode_cfg = str(vg_cfg.get("conditioning_mode", "start_only"))
    row["selector_backend"] = selector_backend

    # Exact keyframe-count matching provenance. This is intentionally distinct
    # from measured CBR: equal keyframe counts do not prove equal serialized
    # channel-symbol payloads.
    match_plan = (
        gen_cfg.get("_keyframe_count_match")
        or gen_cfg.get("_cbr_match")  # read-only compatibility with old configs
        or {}
    )
    verification = _read_json(out_dir / "keyframe_count_match.json")
    if verification is None and match_plan.get("status") == "count_planned":
        verification = verify_keyframe_count_match(out_dir, match_plan)
    match_result = verification or match_plan

    row["fixed_interval_value"] = (kf_cfg.get("fixed_interval") or {}).get("interval")
    row["fixed_count_value"] = (kf_cfg.get("fixed_count") or {}).get("count")
    row["keyframe_match_status"] = match_result.get("status", "not_requested")
    row["keyframe_match_source"] = match_result.get("source_mode", "")
    row["requested_keyframes"] = match_result.get("requested_keyframes")
    row["actual_fixed_keyframes"] = match_result.get("actual_fixed_keyframes")
    row["keyframe_count_delta"] = match_result.get("keyframe_count_delta")
    row["source_n_frames"] = match_result.get("source_n_frames")
    row["target_n_frames"] = match_result.get(
        "actual_target_n_frames", match_result.get("target_n_frames")
    )
    row["source_keyframes_sha256"] = match_result.get("source_sha256", "")

    source_mode = match_result.get("source_mode")
    source_accounting = (
        _accounting_for_run(out_dir_for(output_root, source_mode, video_id))
        if source_mode else None
    )
    target_accounting = _accounting_for_run(out_dir)
    source_symbols = (
        source_accounting.get("total_channel_symbols")
        if source_accounting else None
    )
    target_symbols = (
        target_accounting.get("total_channel_symbols")
        if target_accounting else None
    )
    row["source_channel_symbols"] = source_symbols
    row["target_channel_symbols"] = target_symbols
    proxy_fractions = [
        accounting.get("proxy_fraction")
        for accounting in (source_accounting, target_accounting)
        if accounting is not None
    ]
    accounting_exact = (
        len(proxy_fractions) == 2
        and all(value == 0 for value in proxy_fractions)
    )
    if source_accounting is None or target_accounting is None:
        row["cbr_accounting_kind"] = "unavailable"
    elif accounting_exact:
        row["cbr_accounting_kind"] = "exact"
    else:
        row["cbr_accounting_kind"] = "proxy_or_unknown"
    row["measured_cbr_delta"] = (
        target_symbols - source_symbols
        if isinstance(target_symbols, (int, float))
        and isinstance(source_symbols, (int, float))
        else None
    )
    if row["keyframe_match_status"] == "keyframe_count_verified":
        if row["measured_cbr_delta"] is None:
            row["cbr_match_status"] = "count_only"
        elif row["measured_cbr_delta"] != 0:
            row["cbr_match_status"] = "mismatch"
        elif accounting_exact:
            row["cbr_match_status"] = "verified"
        else:
            row["cbr_match_status"] = "count_only"
    elif row["keyframe_match_status"] == "not_requested":
        row["cbr_match_status"] = "not_requested"
    else:
        row["cbr_match_status"] = "mismatch"
    # Legacy column retained but deliberately empty: exact count matching no
    # longer pretends every requested count has an integer interval.
    row["cbr_match_source"] = row["keyframe_match_source"]
    row["cbr_matched_interval"] = None

    seg_lengths = [_segment_length(s) for s in segments]
    row["n_segments"] = len(segments)
    row["segment_length_min"] = min(seg_lengths) if seg_lengths else None
    row["segment_length_max"] = max(seg_lengths) if seg_lengths else None
    row["segment_length_mean"] = _mean(seg_lengths)
    row["segment_length_std"] = _stdev(seg_lengths)

    psss_backend_kinds, trigger_psss_scores = set(), []
    worker_model_ids = set()
    n_start_only = n_bidirectional = n_fallback = 0
    for seg in segments:
        ks = seg.get("keyframe_selection")
        if ks:
            if ks.get("backend_kind"):
                psss_backend_kinds.add(str(ks["backend_kind"]))
            # Only the score that actually TRIGGERED this segment's keyframe
            # (decision == "new_keyframe") — see the separate population-wide
            # stats below (from keyframes.json's full psss_scores, which also
            # include every "continue_segment" evaluation) for the unbiased
            # distribution. Conflating the two would understate how often
            # PSSS decided NOT to insert a keyframe.
            score = ks.get("psss_score")
            if score and score.get("s_rel") is not None:
                trigger_psss_scores.append(float(score["s_rel"]))
        gen = seg.get("generation")
        if not gen:
            continue
        cm = gen.get("conditioning_mode")
        if cm == "start_only":
            n_start_only += 1
            if conditioning_mode_cfg == "bidirectional":
                n_fallback += 1
        elif cm == "bidirectional":
            n_bidirectional += 1
        backend_val = gen.get("backend")
        backend_candidates = backend_val if isinstance(backend_val, list) else ([backend_val] if backend_val else [])
        for b in backend_candidates:
            if b and str(b).startswith("external_segment_worker:"):
                parts = str(b).split(":")
                if len(parts) >= 3:
                    worker_model_ids.add(parts[2])

    if selector_backend != "psss":
        row["psss_backend_kind"] = "not_applicable"
    elif psss_backend_kinds:
        row["psss_backend_kind"] = ";".join(sorted(psss_backend_kinds))
    else:
        row["psss_backend_kind"] = str((kf_cfg.get("psss") or {}).get("backend", ""))

    # Population-wide PSSS score stats: EVERY frame the selector actually
    # evaluated (both "new_keyframe" and "continue_segment" decisions), read
    # from keyframes.json's psss_scores (video/skem_selector.py::
    # PsssKeyframeSelector.extract()'s full log) — NOT just the ones that
    # happened to trigger a keyframe. Aggregating only the triggering scores
    # (as an earlier version of this function did) is a biased sample: it
    # silently excludes every frame PSSS decided was "similar enough", which
    # is most frames in a typical run.
    keyframes_data = _read_json(out_dir / "keyframes.json") or {}
    all_psss_scores = [
        float(s["s_rel"]) for s in (keyframes_data.get("psss_scores") or [])
        if s.get("s_rel") is not None
    ]
    row["psss_score_mean"] = _mean(all_psss_scores)
    row["psss_score_min"] = min(all_psss_scores) if all_psss_scores else None
    row["psss_score_max"] = max(all_psss_scores) if all_psss_scores else None
    row["psss_score_n"] = len(all_psss_scores)

    # Trigger-only stats (kept separately, clearly named): the score
    # distribution restricted to the evaluations that actually inserted a
    # new keyframe — useful for looking at "how divergent was the typical
    # keyframe-triggering moment", which population stats alone don't show.
    row["trigger_psss_score_mean"] = _mean(trigger_psss_scores)
    row["trigger_psss_score_min"] = min(trigger_psss_scores) if trigger_psss_scores else None
    row["trigger_psss_score_max"] = max(trigger_psss_scores) if trigger_psss_scores else None

    row["n_start_only_segments"] = n_start_only
    row["n_bidirectional_segments"] = n_bidirectional
    row["n_fallback_segments"] = n_fallback
    row["worker_model_id"] = ";".join(sorted(worker_model_ids))

    row["accounting_total_channel_symbols"] = target_symbols

    log_path = out_dir / "run.log"
    row["error_log_path"] = str(log_path) if status == "failed" and log_path.exists() else ""
    row["run_log_path"] = str(log_path) if log_path.exists() else ""
    row["segments_json_path"] = str(out_dir / "segments.json") if (out_dir / "segments.json").exists() else ""
    row["keyframes_json_path"] = str(out_dir / "keyframes.json") if (out_dir / "keyframes.json").exists() else ""
    row["recon_video_path"] = str(out_dir / "recon.mp4") if (out_dir / "recon.mp4").exists() else ""
    return row


_COMPARISON_FIELDS = (
    "n_segments", "segment_length_mean", "segment_length_std", "segment_length_min",
    "segment_length_max", "psss_score_mean", "temporal_srs", "ptc", "sfr", "sdi",
    "overhead_reduction", "n_start_only_segments", "n_bidirectional_segments",
    "n_fallback_segments", "requested_keyframes", "actual_fixed_keyframes",
    "keyframe_count_delta", "source_channel_symbols", "target_channel_symbols",
    "measured_cbr_delta",
)


def build_aggregate_comparison(rows: list, modes_pair: tuple = ("skim_sfa_fixed", "skem_dsa_psss")) -> list:
    """Per-video + overall-mean SKIM/SFA-vs-SKEM/DSA comparison table.

    One row per video (plus a final "MEAN" row) with ``<mode>.<field>``
    columns for each of *modes_pair* side by side — lets a reader see, e.g.,
    ``skim_sfa_fixed.segment_length_mean`` next to
    ``skem_dsa_psss.segment_length_mean`` for the same video without cross-
    referencing two separate summary_metrics rows by hand. Missing (mode,
    video) combinations (job not yet run) leave that side's cells ``None``
    rather than dropping the video from the table, so a partial batch still
    produces a (partially populated) comparison.
    """
    by_key = {(r["mode"], r["video_id"]): r for r in rows}
    video_ids = sorted({r["video_id"] for r in rows if r["mode"] in modes_pair})
    if not video_ids:
        return []
    mode_a, mode_b = modes_pair
    collected = {f: {mode_a: [], mode_b: []} for f in _COMPARISON_FIELDS}

    out_rows = []
    for vid in video_ids:
        row = {"video_id": vid}
        for mode in (mode_a, mode_b):
            r = by_key.get((mode, vid))
            for f in _COMPARISON_FIELDS:
                v = r.get(f) if r else None
                row[f"{mode}.{f}"] = v
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    collected[f][mode].append(v)
        out_rows.append(row)

    mean_row = {"video_id": "MEAN"}
    for mode in (mode_a, mode_b):
        for f in _COMPARISON_FIELDS:
            vals = collected[f][mode]
            mean_row[f"{mode}.{f}"] = _mean(vals)
    out_rows.append(mean_row)
    return out_rows


def write_summary(rows: list, out_base: Path) -> None:
    """Write rows as <out_base>.csv/.md/.json (mirrors
    scripts/summarize_etri_video_eval.py::write_summary's format)."""
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

    def _fmt(v):
        if v is None or v == "":
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = ["# LGVSC 1C reproduction-baseline summary", "",
             "See docs/lgvsc_1c_reproduction_readiness.md before treating any "
             "wan_skim_sfa/wan_skem_dsa row as a faithful LGVSC reproduction — "
             "these are LGVSC-style approximations (see that doc's caveats).",
             ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join("---" for _ in fields) + "|")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(f)) for f in fields) + " |")
    out_base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(output_root: Path, modes: list, entries: list, status_by_id: dict = None) -> list:
    rows = []
    for mode in modes:
        for entry in entries:
            rows.append(collect_run_metrics(output_root, mode, entry["key"], status_by_id))
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_worker_max_memory(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--worker-max-memory must be valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict) or not parsed:
        raise argparse.ArgumentTypeError(
            "--worker-max-memory must be a non-empty JSON object"
        )
    return parsed


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ETRI 후속 1단계 1C — LGVSC-reproduction-baseline batch driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data" / "etri_video_eval"))
    p.add_argument("--output-root", default=str(_run_root() / "etri_video_eval" / "lgvsc_1c_reproduce"))
    p.add_argument("--modes", default="all",
                   help=f"Comma list from {MODES} or 'all'.")
    p.add_argument("--videos", default=None,
                   help="Comma list of video keys (e.g. 01_person_walk) or ids (01); default all 10.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Forwarded to evaluate_video.py's --max-frames (smoke: 14; omit for full clip).")
    p.add_argument("--device", default=None, help="Forwarded to evaluate_video.py's --device.")
    p.add_argument(
        "--worker-device-map", choices=("balanced",), default=None,
        help="Shard each Wan worker pipeline across all visible GPUs. Jobs still run sequentially.",
    )
    p.add_argument(
        "--worker-max-memory", type=_parse_worker_max_memory, default=None,
        help='JSON memory map for the worker, e.g. {"0":"8GiB","1":"22GiB",'
             '"2":"22GiB","cpu":"40GiB"}; requires --worker-device-map.',
    )
    p.add_argument("--no-models", action="store_true",
                   help="Forwarded to evaluate_video.py's --no-models — disables SGD-JSCC Rx "
                        "reconstruction ONLY; svd_start_only/wan_* worker backends still run for "
                        "real (see module docstring's '--no-models note').")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command + output paths that WOULD run; never invokes subprocess.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (mode, video) jobs whose temporal_metrics.csv already exists.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep running remaining jobs after a failure instead of stopping the batch.")
    p.add_argument("--summary-only", action="store_true",
                   help="Skip all job dispatch; just (re)build summary_metrics.csv/.md/.json from "
                        "whatever is already on disk for the selected --modes/--videos.")
    p.add_argument(
        "--keyframe-count-match-from", default=None,
        help="Mode name (e.g. skem_dsa_psss) whose already-run keyframes.json supplies "
             "the exact per-video keyframe count for a fixed SKIM baseline. The target "
             "uses FixedCountKeyframeSelector, then verifies the actual output count. "
             "This matches keyframe count, not necessarily measured channel-symbol CBR.",
    )
    p.add_argument(
        "--cbr-match-from", default=None,
        help="Deprecated alias for --keyframe-count-match-from. Kept for command compatibility; "
             "summary provenance reports count_only unless measured channel symbols also match.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.worker_max_memory is not None and args.worker_device_map is None:
        print("Error: --worker-max-memory requires --worker-device-map.", file=sys.stderr)
        return 1
    # Always absolute — build_run_config() writes out_dir-derived paths
    # verbatim into the generated per-video config, which config.py then
    # resolves relative to the GENERATED config file's own directory (not the
    # CWD). A relative --output-root would otherwise silently double-nest
    # every output path under _generated_configs/<mode>/ (see
    # docs/lgvsc_1c_reproduction_readiness.md / lgvsc_psss_skem_readiness.md's
    # "generated config uses absolute output paths" requirement).
    output_root = Path(args.output_root).resolve()
    data_root = Path(args.data_root)
    if (
        args.keyframe_count_match_from
        and args.cbr_match_from
        and args.keyframe_count_match_from != args.cbr_match_from
    ):
        print(
            "Error: --keyframe-count-match-from and deprecated --cbr-match-from "
            "must name the same mode when both are provided.",
            file=sys.stderr,
        )
        return 1
    keyframe_count_match_from = (
        args.keyframe_count_match_from or args.cbr_match_from
    )

    modes = list(MODES) if args.modes.strip() == "all" else [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            print(f"Error: unknown mode {m!r} (valid: {', '.join(MODES)})", file=sys.stderr)
            return 1

    entries = read_manifest(data_root)
    try:
        entries = select_videos(entries, args.videos)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "batch_status.json"
    existing = []
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []
    status_by_id = {(r.get("mode"), r.get("video")): r for r in existing}

    if args.summary_only:
        rows = build_summary(output_root, modes, entries, status_by_id)
        write_summary(rows, output_root / "summary_metrics")
        comparison_rows = build_aggregate_comparison(rows)
        write_summary(comparison_rows, output_root / "summary_aggregate_comparison")
        print(f"Summary regenerated from disk → {output_root / 'summary_metrics.csv'} ({len(rows)} rows)")
        return 0

    runs = []
    n_fail = 0
    for mode in modes:
        for entry in entries:
            status = run_job(
                mode, entry, output_root,
                device=args.device, max_frames=args.max_frames, no_models=args.no_models,
                skip_existing=args.skip_existing, dry_run=args.dry_run,
                cbr_match_from=keyframe_count_match_from,
                worker_device_map=args.worker_device_map,
                worker_max_memory=args.worker_max_memory,
            )
            runs.append(status)
            tag = "DRY-RUN" if status["status"] == "dry_run" else status["status"].upper()
            print(f"[{tag}] {mode} / {entry['key']} → {status['out_dir']}")
            if status["status"] == "failed":
                n_fail += 1
                if not args.continue_on_error:
                    print(f"Stopping (no --continue-on-error) after failure: "
                          f"{mode}/{entry['key']} — see {status['out_dir']}/run.log", file=sys.stderr)
                    new_ids = {(r["mode"], r["video"]) for r in runs}
                    merged = [r for r in existing if (r.get("mode"), r.get("video")) not in new_ids] + runs
                    status_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
                    return 1

    new_ids = {(r["mode"], r["video"]) for r in runs}
    merged = [r for r in existing if (r.get("mode"), r.get("video")) not in new_ids] + runs
    status_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    if not args.dry_run:
        status_by_id = {(r.get("mode"), r.get("video")): r for r in merged}
        rows = build_summary(output_root, modes, entries, status_by_id)
        write_summary(rows, output_root / "summary_metrics")
        comparison_rows = build_aggregate_comparison(rows)
        write_summary(comparison_rows, output_root / "summary_aggregate_comparison")
        print(f"Summary → {output_root / 'summary_metrics.csv'} ({len(rows)} rows)")

    print(f"\nBatch complete: {len(runs)} job(s), {n_fail} failed. Status → {status_path}")
    # Reaching here means either there were no failures, or --continue-on-error
    # was set (an early failure without it already returned 1 above).
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
