#!/usr/bin/env python
"""diagnose_float32_digital_quality.py – float32 digital quality diagnostic CLI.

Compares, at the SAME (video, frame, seed), three real production Tx/Rx
paths — see ``src/sgdjscc_lab/diagnostics/float32_digital_paths.py`` for the
full design note:

  awgn                current production AWGN path
  digital_inprocess   DigitalPacketChannel swapped in, no frame-bundle byte boundary
  digital_wire        real transmission.receiver_runtime byte boundary

Judgment criteria (see ``diagnostics/verdict.py``):
  - in-process and wire DIFFER at some stage         -> packet/Tx-Rx problem
  - both digital paths MATCH but worse than AWGN     -> edge/ControlNet/diffusion problem
  - diffusion_bypass_vae_direct ablation ALREADY low -> latent scaling/normalization problem
  - insufficient evidence                            -> inconclusive

Examples
--------
Dry run (prints the resolved plan, touches nothing):
    python scripts/diagnose_float32_digital_quality.py \\
        --output-root outputs/f32dig_smoke --video-ids 01_person_walk \\
        --frames 0 --dry-run

CPU/mock structural smoke test (no checkpoints, no GPU):
    python scripts/diagnose_float32_digital_quality.py \\
        --output-root outputs/f32dig_smoke --video-ids 01_person_walk \\
        --frames 0 --no-models --device cpu

Real single-frame, all-ablation diagnostic on a GPU server:
    python scripts/diagnose_float32_digital_quality.py \\
        --output-root outputs/f32dig_run --video-ids 01_person_walk \\
        --frames 0 --ablations all --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sgdjscc_lab.utils import run_manifest as rm  # noqa: E402 — hard dependency, see run_transmission_reduction_eval.py precedent

_CFG_FRAGMENTS = (
    "base/channel/awgn", "base/model/sgdjscc", "base/infer/awgn", "base/eval/default",
)

PATH_CHOICES = ("awgn", "digital_inprocess", "digital_wire")

# These ablations force edge_already_received identical across
# digital_inprocess and digital_wire (see diagnostics/ablations.py), which is
# what makes their edge/decoder-stage tensor comparisons meaningful
# packet_tx_rx_issue evidence (diagnostics/verdict.py's
# edge_handling_equalized=True) rather than the expected-by-design divergence
# under the baseline ablation.
EDGE_EQUALIZING_ABLATIONS = ("serialized_raw_edge", "awgn_edge_retransmit")

# Baseline verdicts from different server stages are not necessarily based on
# the same evidence: stage 3 deliberately runs baseline only, while stages 4/5
# also run the VAE-direct ablation.  Persist this scope explicitly so the
# integrated report can prefer the richer judgment instead of treating a
# legitimate evidence-driven reclassification as non-determinism.
EVIDENCE_BASELINE_PENDING_VAE = "baseline_pending_vae_direct"
EVIDENCE_BASELINE_ONLY = "baseline_only"
EVIDENCE_BASELINE_WITH_VAE = "baseline_with_vae_direct"
EVIDENCE_AUXILIARY_EDGE = "auxiliary_edge_equalized"

PATH_COMPARISON_FIELDS = [
    "video", "frame", "seed", "ablation", "path",
    "psnr", "ssim", "lpips", "psnr_delta_vs_awgn", "ssim_delta_vs_awgn", "lpips_delta_vs_awgn",
    "latency_ms", "diffusion_steps", "n_patches", "wire_bytes", "roundtrip_bitexact",
    "failed", "failure_stage", "failure_message",
]

TENSOR_PAIR_FIELDS = [
    "video", "frame", "seed", "ablation", "stage", "path_a", "path_b",
    "comparable", "reason", "exact_equal", "both_finite",
    "max_abs_err", "mean_abs_err", "mse", "cosine_similarity", "norm_a", "norm_b", "norm_ratio",
]

FAILED_CASES_FIELDS = ["video", "frame", "seed", "ablation", "path", "stage", "message"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_frame_spec(spec: str) -> List[int]:
    """``"0"`` / ``"0,5,9"`` / ``"0-19"`` / ``"0-4,10,20-24"`` -> sorted unique ints."""
    out: set = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="float32 digital Tx/Rx quality diagnostic harness (awgn vs digital_inprocess vs digital_wire).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-root", required=True)
    p.add_argument("--dataset-root", default=str(_REPO_ROOT / "data/etri_video_eval"))
    p.add_argument("--video-ids", default=None, help="Comma-separated video keys (e.g. 01_person_walk); default = all in manifest.")
    p.add_argument("--frames", default="0", help='Frame indices: "0", "0,5,9", "0-19", "0-4,10,20-24".')
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--paths", default=",".join(PATH_CHOICES), help="Comma-separated subset of " + ",".join(PATH_CHOICES))
    p.add_argument("--ablations", default="baseline",
                    help='Comma-separated ablation names, "all", or "baseline" (default = baseline only).')
    p.add_argument("--bit-depth", type=int, default=32, help="Digital bit depth for digital_inprocess/digital_wire.")
    p.add_argument("--granularity", default="per_tensor", choices=["per_tensor", "per_channel"])
    p.add_argument("--digital-step-policy", default="fixed_reference",
                    choices=["fixed_reference", "bitdepth_proxy", "quant_nmse"])
    p.add_argument("--fixed-step-value", type=float, default=0.5)
    p.add_argument("--minimal-denoise-steps", type=int, default=1)
    p.add_argument("--record-patch-index", type=int, default=0,
                    help="Which patch index gets tensor-stage instrumentation (representative patch).")
    p.add_argument("--no-instrument-tensors", action="store_true",
                    help="Disable tensor_stage_stats.jsonl/tensor_pair_comparison.csv (path_comparison.csv metrics only). "
                         "Use for large multi-frame runs where per-stage tensors are not needed.")
    p.add_argument("--save-tensors", action="store_true",
                    help="Additionally persist each recorded tensor as a .pt file under <output-root>/tensors/. "
                         "Off by default; only use for small, targeted runs.")
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--config", default=None, help="Optional composed config path; default = this harness's own fragment set.")
    p.add_argument("--model-root", default=None)
    p.add_argument("--no-models", action="store_true",
                    help="Use a deterministic CPU synthetic ModelBundle instead of loading real checkpoints "
                         "(structural smoke test only — never a real quality measurement).")
    p.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit; touches nothing.")
    p.add_argument("--resume", action="store_true",
                    help="Reuse an existing --output-root, skipping (video, frame, ablation) groups already "
                         "fully recorded in path_comparison.csv. Requires run_signature.json to match.")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Config / model construction
# ─────────────────────────────────────────────────────────────────────────────

def _make_cfg(output_root: Path, model_root: Path, snr_db: float, config_path: Optional[str], device: str):
    """Compose a real config via the project's own fragment set — see
    ``docs/protocols/float32_digital_diagnostics.md``. Kept intentionally
    tiny/duplicated (config-loading glue, not algorithm logic) rather than
    importing a sibling script module for two functions.
    """
    from omegaconf import OmegaConf
    from sgdjscc_lab.config import load_config

    if config_path:
        source_path = Path(config_path).resolve()
        cfg = load_config(source_path)
    else:
        composed_path = output_root / "configs" / "composed.yaml"
        composed_path.parent.mkdir(parents=True, exist_ok=True)
        frag_paths = [str((_REPO_ROOT / "configs" / f)) for f in _CFG_FRAGMENTS]
        composed_path.write_text(
            "_defaults_: [" + ", ".join(f'"{p}"' for p in frag_paths) + "]\n", encoding="utf-8",
        )
        cfg = load_config(composed_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "model_root": str(model_root), "snr_db": float(snr_db),
        "use_phase4": True, "mask_method": "none",
    }))
    return cfg


def _build_models(no_models: bool, cfg, device: str):
    if no_models:
        from sgdjscc_lab.diagnostics.mock_models import build_mock_models
        return build_mock_models(device=device, snr_db=float(cfg.snr_db))
    from sgdjscc_lab.runtime import build_models
    import torch
    return build_models(cfg, torch.device(device))


_CHECKPOINT_NAMES = (
    "JSCC_model.pth", "diffusion_backbone.pth", "diffusion_controlnet.pth", "muge-epoch-19-checkpoint.pth",
)


def _checkpoint_hashes(model_root: Path) -> Dict[str, str]:
    return {name: rm.sha256_file(model_root / name) for name in _CHECKPOINT_NAMES if (model_root / name).exists()}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _load_manifest_reader():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_f32dig_run_etri_video_eval", _REPO_ROOT / "scripts" / "run_etri_video_eval.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.read_manifest


def _load_video_frames(video_path: Path, work_dir: Path, frame_indices: List[int]):
    from sgdjscc_lab.utils.video_io import extract_frames
    from sgdjscc_lab.io import load_image_as_tensor

    info = extract_frames(video_path, work_dir)
    files = info["files"]
    max_idx = max(frame_indices)
    if max_idx >= len(files):
        raise ValueError(f"requested frame {max_idx} but {video_path} has only {len(files)} frames")
    return {i: load_image_as_tensor(files[i]) for i in frame_indices}


# ─────────────────────────────────────────────────────────────────────────────
# Incremental, resume-safe CSV/JSONL writers
# ─────────────────────────────────────────────────────────────────────────────

def _append_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def _append_jsonl_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _completed_groups(path: Path) -> set:
    if not path.exists():
        return set()
    completed: set = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            completed.add((row["video"], row["frame"], row["ablation"]))
    return completed


def _count_csv_rows(path: Path) -> int:
    """Exact row count (never a proxy for "unique groups") — used for
    failed_count so an ablation that fails on 3 of 3 paths for one group
    is not undercounted as "1 failure"."""
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def _dataset_manifest_hash(dataset_root: str) -> str:
    manifest_path = Path(dataset_root) / "manifest.csv"
    return rm.sha256_file(manifest_path) if manifest_path.exists() else rm.UNKNOWN


def _dataset_content_hash(dataset_root: str, entries: List[Dict[str, Any]]) -> str:
    """SHA-256 covering manifest.csv AND every SELECTED video's actual file
    bytes (+ captions/GT side files when present) — hashing manifest.csv
    alone cannot detect a video file being swapped while manifest.csv stays
    byte-identical, which would let --resume silently continue against
    different source video content."""
    import hashlib

    parts = [_dataset_manifest_hash(dataset_root)]
    for e in sorted(entries, key=lambda e: e["key"]):
        parts.append(e["key"])
        for field in ("processed", "captions", "gt"):
            value = e.get(field)
            if value is not None and Path(value).is_file():
                parts.append(f"{field}:{rm.sha256_file(value)}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _load_verdicts_index(path: Path) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    """Loads previously-recorded per-(video, frame, ablation-kind) verdicts
    (if any) so a ``--resume`` run's verdict_summary/REPORT.md reflect
    everything completed across every invocation targeting this
    --output-root, not just what the current invocation (re)processed.
    ``ablation-kind`` is ``"baseline"`` (transport-only classification) or
    one of ``EDGE_EQUALIZING_ABLATIONS`` (edge_handling_equalized=True)."""
    index: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    if not path.exists():
        return index
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[(row["video"], int(row["frame"]), row["ablation"])] = row
    return index


def _csv_bool(value: Optional[str]) -> Optional[bool]:
    if value in (None, ""):
        return None
    return value == "True"


def _csv_float(value: Optional[str]) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_psnr_index(path: Path) -> Dict[Tuple[str, str, str], Dict[str, Dict[str, Optional[float]]]]:
    """``{(video, frame_str, ablation): {path: {"psnr", "ssim", "lpips"}}}``
    reloaded from an existing path_comparison.csv. Evidence for a group
    completed in a PRIOR invocation is available immediately from this index
    — never only from this invocation's in-memory results — which is what
    lets a --resume run recompute a verdict for a baseline group that was
    already-done (skipped) this invocation (see _persist_verdict in run())."""
    index: Dict[Tuple[str, str, str], Dict[str, Dict[str, Optional[float]]]] = {}
    if not path.exists():
        return index
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["video"], row["frame"], row["ablation"])
            index.setdefault(key, {})[row["path"]] = {
                "psnr": _csv_float(row.get("psnr")),
                "ssim": _csv_float(row.get("ssim")),
                "lpips": _csv_float(row.get("lpips")),
            }
    return index


def _load_pair_index(path: Path) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """``{(video, frame_str, ablation): [digital_inprocess-vs-digital_wire pair rows]}``
    reloaded from an existing tensor_pair_comparison.csv (order-independent —
    path_a/path_b may have been recorded in either order)."""
    index: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    if not path.exists():
        return index
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if {row.get("path_a"), row.get("path_b")} != {"digital_inprocess", "digital_wire"}:
                continue
            key = (row["video"], row["frame"], row["ablation"])
            index.setdefault(key, []).append({
                "stage": row.get("stage"),
                "comparable": _csv_bool(row.get("comparable")) or False,
                "exact_equal": _csv_bool(row.get("exact_equal")),
                "both_finite": _csv_bool(row.get("both_finite")),
                "mean_abs_err": _csv_float(row.get("mean_abs_err")),
                "cosine_similarity": _csv_float(row.get("cosine_similarity")),
            })
    return index


def _inprocess_wire_pairs(pair_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filters a freshly-computed group's pair_rows (any path combination)
    down to digital_inprocess-vs-digital_wire, matching what
    _load_pair_index keeps from disk."""
    return [
        r for r in pair_rows
        if {r.get("path_a"), r.get("path_b")} == {"digital_inprocess", "digital_wire"}
    ]


def _psnr_triplet(psnr_index: Dict[Tuple[str, str, str], Dict[str, Dict[str, Optional[float]]]],
                   key: Tuple[str, str, str]) -> Dict[str, Optional[float]]:
    entry = psnr_index.get(key, {})
    return {
        "awgn_psnr": entry.get("awgn", {}).get("psnr"),
        "digital_inprocess_psnr": entry.get("digital_inprocess", {}).get("psnr"),
        "digital_wire_psnr": entry.get("digital_wire", {}).get("psnr"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run signature (resume safety)
# ─────────────────────────────────────────────────────────────────────────────

def _build_run_signature(args, cfg, entries: List[Dict[str, Any]], frames: List[int], model_root: Path) -> Dict[str, Any]:
    from omegaconf import OmegaConf
    import hashlib

    git_state = rm.get_git_state(_REPO_ROOT)
    try:
        resolved_config = OmegaConf.to_container(cfg, resolve=True)
        config_hash = hashlib.sha256(json.dumps(resolved_config, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    except Exception:  # noqa: BLE001 — signature must still be buildable if config dump fails
        config_hash = rm.UNKNOWN

    return {
        "git_commit": git_state["commit"], "git_dirty": git_state["dirty"], "git_branch": git_state["branch"],
        "video_ids": sorted(e["key"] for e in entries), "frames": frames, "seed": args.seed,
        "paths": sorted(args.paths.split(",")), "ablations": args.ablations,
        "bit_depth": args.bit_depth, "granularity": args.granularity,
        "digital_step_policy": args.digital_step_policy,
        "fixed_step_value": args.fixed_step_value, "minimal_denoise_steps": args.minimal_denoise_steps,
        "resolved_config_sha256": config_hash,
        # Covers manifest.csv AND the actual selected video/caption/GT file
        # bytes (see _dataset_content_hash) -- a video swap with an
        # unchanged manifest.csv must still be detected.
        "dataset_content_sha256": _dataset_content_hash(args.dataset_root, entries),
        "checkpoint_sha256": {} if args.no_models else _checkpoint_hashes(model_root),
        "no_models": bool(args.no_models), "device": args.device,
        # Both affect what actually gets recorded/skippable across a resumed
        # run -- must match exactly, or a resume could silently mix
        # instrumented and non-instrumented frames for the same output-root.
        "instrument_tensors": not args.no_instrument_tensors,
        "record_patch_index": args.record_patch_index,
    }


def _diff_signature(old: Dict[str, Any], new: Dict[str, Any]) -> str:
    lines = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            lines.append(f"  {key}:\n    was: {json.dumps(old.get(key), default=str)}\n    now: {json.dumps(new.get(key), default=str)}")
    return "\n".join(lines)


def _check_resume_signature(output_root: Path, signature: Dict[str, Any], resume: bool) -> None:
    sig_path = output_root / "run_signature.json"
    if not sig_path.exists():
        sig_path.parent.mkdir(parents=True, exist_ok=True)
        sig_path.write_text(json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8")
        return
    existing = json.loads(sig_path.read_text(encoding="utf-8"))
    if existing != signature:
        diff = _diff_signature(existing, signature)
        raise SystemExit(
            f"run_signature mismatch at {sig_path} — refusing to {'resume' if resume else 'overwrite'} "
            f"a different run in this output-root:\n{diff}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

_STOP_REQUESTED = False


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        print(f"[diagnose_float32_digital_quality] received signal {signum}; "
              "finishing the current (video, frame, ablation) group then exiting.", file=sys.stderr)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    video_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()] if args.video_ids else None
    frames = _parse_frame_spec(args.frames)
    requested_paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    for p in requested_paths:
        if p not in PATH_CHOICES:
            raise SystemExit(f"unknown --paths entry {p!r}; expected one of {PATH_CHOICES}")

    from sgdjscc_lab.diagnostics.ablations import build_default_ablations
    all_ablations = build_default_ablations(
        fixed_step_value=args.fixed_step_value, minimal_denoise_steps=args.minimal_denoise_steps,
    )
    if args.ablations == "all":
        ablation_names = list(all_ablations.keys())
    elif args.ablations == "baseline":
        ablation_names = ["baseline"]
    else:
        ablation_names = [a.strip() for a in args.ablations.split(",") if a.strip()]
    for name in ablation_names:
        if name not in all_ablations:
            raise SystemExit(f"unknown ablation {name!r}; available: {sorted(all_ablations)}")
    if "baseline" not in ablation_names:
        ablation_names = ["baseline"] + ablation_names  # baseline always first (verdict anchor)
    # Whether this run is expected to also produce diffusion_bypass_vae_direct
    # evidence for the SAME frames -- if so, the baseline verdict must stay
    # "provisional" until that evidence actually exists (possibly only after
    # a later --resume), rather than freezing a verdict computed without it.
    vae_direct_requested = "diffusion_bypass_vae_direct" in ablation_names

    output_root = Path(args.output_root)
    model_root = Path(args.model_root) if args.model_root else None
    if model_root is None and not args.no_models:
        from sgdjscc_lab.paths import model_root as resolve_model_root
        model_root = Path(resolve_model_root())

    read_manifest = _load_manifest_reader()
    entries = read_manifest(Path(args.dataset_root))
    if video_ids is not None:
        wanted = set(video_ids)
        entries = [e for e in entries if e["key"] in wanted]
        missing = wanted - {e["key"] for e in entries}
        if missing:
            raise SystemExit(f"video-ids not found in manifest: {sorted(missing)}")
    if not entries:
        raise SystemExit("no videos selected (check --video-ids / --dataset-root)")

    if args.dry_run:
        print("[dry-run] float32 digital diagnostic plan:")
        print(f"  output_root      : {output_root}")
        print(f"  videos           : {[e['key'] for e in entries]}")
        print(f"  frames           : {frames}")
        print(f"  paths            : {requested_paths}")
        print(f"  ablations        : {ablation_names}")
        print(f"  bit_depth        : {args.bit_depth}  granularity: {args.granularity}")
        print(f"  digital_step_policy: {args.digital_step_policy}")
        print(f"  seed             : {args.seed}")
        print(f"  no_models (mock) : {args.no_models}  device: {args.device}")
        print(f"  instrument_tensors: {not args.no_instrument_tensors}")
        n_groups = len(entries) * len(frames) * len(ablation_names)
        print(f"  total (video,frame,ablation) groups: {n_groups} x up to {len(requested_paths)} paths each")
        return 0

    _install_signal_handlers()
    output_root.mkdir(parents=True, exist_ok=True)
    tensor_dir = output_root / "tensors" if args.save_tensors else None

    cfg = _make_cfg(output_root, model_root or Path("."), snr_db=10.0, config_path=args.config, device=args.device)
    signature = _build_run_signature(args, cfg, entries, frames, model_root or Path("."))
    _check_resume_signature(output_root, signature, args.resume)

    path_comparison_csv = output_root / "path_comparison.csv"
    tensor_stage_jsonl = output_root / "tensor_stage_stats.jsonl"
    tensor_pair_csv = output_root / "tensor_pair_comparison.csv"
    failed_csv = output_root / "failed_cases.csv"
    verdicts_jsonl = output_root / "verdicts.jsonl"

    # Skipping already-completed (video, frame, ablation) groups is ALWAYS
    # on (matches the project's existing convention — see
    # run_transmission_reduction_eval.py — where resume-skip is unconditional,
    # not gated behind a flag). What --resume actually gates is whether a
    # NON-EMPTY output-root is allowed to be reused at all: a bare re-run
    # (no --resume) into a directory that already has completed groups is
    # refused outright, rather than silently duplicating rows — this is the
    # fix for the reproduced "3 rows -> 6 rows" bug (a bare re-run used to
    # neither skip nor refuse).
    already_done = _completed_groups(path_comparison_csv)
    # Verdicts already recorded by a PRIOR invocation targeting this
    # --output-root -- reloaded so a --resume run's verdict_summary/REPORT.md
    # cover every frame ever completed here, not just the ones (re)processed
    # in THIS invocation (whose baseline group may already be in already_done
    # and therefore never touched below).
    verdicts_index = _load_verdicts_index(verdicts_jsonl)
    # Verdict EVIDENCE (PSNR/SSIM/LPIPS per path, digital_inprocess-vs-
    # digital_wire tensor comparisons) for every group ever completed here,
    # reloaded from disk. Verdict computation below reads exclusively from
    # these two indices (updated in-memory as this invocation processes new
    # groups too) rather than from local variables scoped to "what THIS
    # invocation's ablation loop just did" -- the previous design lost a
    # frame's verdict permanently whenever the baseline group finished and
    # was written to disk, but a SIGINT/SIGTERM arrived before the verdict
    # itself was computed: a later --resume would skip that now-already-done
    # baseline group and never revisit it, so baseline_psnr/
    # baseline_comparisons stayed empty forever. Reading from these
    # disk-backed indices instead means the verdict is computed the moment
    # its evidence exists on disk, regardless of which invocation wrote it.
    psnr_index = _load_psnr_index(path_comparison_csv)
    pair_index = _load_pair_index(tensor_pair_csv) if not args.no_instrument_tensors else {}
    if already_done and not args.resume:
        raise SystemExit(
            f"{path_comparison_csv} already has {len(already_done)} completed (video, frame, "
            "ablation) group(s) from a previous run. Pass --resume to continue safely (skips "
            "already-completed groups, keeps prior verdicts) or use a different --output-root "
            "for a fresh run."
        )

    manifest_initial = rm.build_run_manifest(
        run_id=f"float32_digital_diagnostics_{int(time.time())}",
        command_argv=sys.argv, command_source="captured",
        seed=args.seed, resolved_config_path=None, config_source_path=args.config,
        dataset_ref=str(args.dataset_root), dataset_hash=_dataset_content_hash(args.dataset_root, entries),
        checkpoints=(None if args.no_models else {n: model_root / n for n in _CHECKPOINT_NAMES if (model_root / n).exists()}),
        repo_root=_REPO_ROOT, cuda_device_index=0,
        extra={"phase": "initial", "signature": signature},
    )
    rm.write_run_manifest(output_root / "run_manifest_initial.json", manifest_initial)

    models = _build_models(args.no_models, cfg, args.device)

    from sgdjscc_lab.diagnostics.float32_digital_paths import (
        PathOutcome, run_frame_awgn, run_frame_digital_inprocess, run_frame_digital_wire,
        run_path_with_failure_capture,
    )
    from sgdjscc_lab.diagnostics.tensor_compare import compare_tensors
    from sgdjscc_lab.diagnostics.tensor_recorder import TensorRecorder
    from sgdjscc_lab.diagnostics.verdict import aggregate_verdicts, classify
    from sgdjscc_lab.diagnostics.report import write_report_md, write_summary_json
    from sgdjscc_lab.evaluators.quality import compute_psnr, compute_ssim
    from sgdjscc_lab.utils.seed import derive_frame_seed, set_global_seed

    lpips_fn = None
    if not args.no_lpips and not args.no_models:
        try:
            from sgdjscc_lab.evaluators.quality import compute_lpips
            lpips_fn = compute_lpips
        except Exception as exc:  # noqa: BLE001 — LPIPS availability is environment-dependent
            print(f"[diagnose_float32_digital_quality] LPIPS unavailable ({exc}); recording None.", file=sys.stderr)

    def _persist_verdict(video: str, frame: int, kind: str, *, path_quality, comparisons,
                          vae_direct_quality=None, edge_handling_equalized: bool = False,
                          provisional: bool = False, evidence_level: str) -> None:
        """Computes + appends one verdict row.

        A row with ``status: "final"`` is frozen — never recomputed or
        overwritten once written. A row with ``status: "provisional"`` (used
        for the "baseline" kind when ``diffusion_bypass_vae_direct`` was
        requested for this run but its evidence has not arrived yet) stays
        open to being recomputed on a later call — e.g. after a --resume
        picks up the VAE-direct group's evidence — and is only re-appended
        to verdicts.jsonl (never edited in place; the loader keeps the LAST
        line per key) when the recomputed content actually differs, so an
        unchanged provisional state is not re-written every invocation.
        """
        key = (video, frame, kind)
        existing = verdicts_index.get(key)
        if existing is not None and existing.get("status", "final") == "final":
            return

        verdict = classify(
            inprocess_vs_wire_comparisons=comparisons, path_quality=path_quality,
            vae_direct_quality=vae_direct_quality, edge_handling_equalized=edge_handling_equalized,
        )
        row = {
            "video": video, "frame": frame, "ablation": kind,
            "verdict": verdict.verdict, "reason": verdict.reason,
            "first_divergent_stage": verdict.first_divergent_stage,
            "edge_handling_equalized": edge_handling_equalized,
            "status": "provisional" if provisional else "final",
            "evidence_level": evidence_level,
            "vae_direct_considered": evidence_level == EVIDENCE_BASELINE_WITH_VAE,
        }
        if existing is not None and all(existing.get(k) == row[k] for k in row):
            return  # nothing actually changed -- skip the redundant append

        if existing is not None:
            print(
                f"[diagnose_float32_digital_quality] verdict for {key} updated: "
                f"{existing.get('status')}/{existing.get('verdict')} -> {row['status']}/{row['verdict']}",
                file=sys.stderr,
            )
        verdicts_index[key] = row
        _append_jsonl_rows(verdicts_jsonl, [row])

    n_frames_processed = 0
    interrupted = False

    for entry in entries:
        video_key = entry["key"]
        work_dir = output_root / "frames_cache" / video_key
        frame_tensors = _load_video_frames(entry["processed"], work_dir, frames)

        for frame_idx in frames:
            frame_tensor = frame_tensors[frame_idx]
            awgn_step_ref: Optional[Tuple[Any, Any]] = None

            for ablation_name in ablation_names:
                group_key = (video_key, str(frame_idx), ablation_name)
                if group_key in already_done:
                    continue

                ablation = all_ablations[ablation_name]
                seed = derive_frame_seed(args.seed, video_key, frame_idx)
                recorder = TensorRecorder(
                    enabled=not args.no_instrument_tensors,
                    save_tensor_files=args.save_tensors, tensor_dir=tensor_dir,
                )

                outcomes: Dict[str, PathOutcome] = {}
                for path_name in requested_paths:
                    set_global_seed(seed)
                    common_kwargs = dict(
                        recorder=recorder, video=video_key, frame_index=frame_idx, seed=seed,
                        record_patch_index=args.record_patch_index,
                    )
                    if path_name == "awgn":
                        outcome = run_path_with_failure_capture(
                            run_frame_awgn, "awgn", frame_tensor, models, cfg, ablation, **common_kwargs,
                        )
                        if outcome.cur_step_ref is not None:
                            awgn_step_ref = outcome.cur_step_ref
                    elif path_name == "digital_inprocess":
                        outcome = run_path_with_failure_capture(
                            run_frame_digital_inprocess, "digital_inprocess", frame_tensor, models, cfg, ablation,
                            bit_depth=args.bit_depth, granularity=args.granularity,
                            awgn_step_ref=awgn_step_ref, **common_kwargs,
                        )
                    else:
                        outcome = run_path_with_failure_capture(
                            run_frame_digital_wire, "digital_wire", frame_tensor, models, cfg, ablation,
                            bit_depth=args.bit_depth, granularity=args.granularity,
                            digital_step_policy=args.digital_step_policy,
                            awgn_step_ref=awgn_step_ref, **common_kwargs,
                        )
                    outcomes[path_name] = outcome

                # ── metrics + CSV rows for this (video, frame, ablation) ──
                rows = []
                failed_rows = []
                psnr_by_path: Dict[str, Optional[float]] = {}
                ssim_by_path: Dict[str, Optional[float]] = {}
                lpips_by_path: Dict[str, Optional[float]] = {}
                for path_name, outcome in outcomes.items():
                    if outcome.failed:
                        failed_rows.append({
                            "video": video_key, "frame": frame_idx, "seed": seed, "ablation": ablation_name,
                            "path": path_name, "stage": outcome.failure_stage, "message": outcome.failure_message,
                        })
                        psnr = ssim = lpips_val = None
                    else:
                        original = frame_tensor
                        recon = outcome.reconstructed
                        psnr = compute_psnr(original, recon)
                        ssim = compute_ssim(original, recon)
                        lpips_val = lpips_fn(original, recon) if lpips_fn is not None else None
                    psnr_by_path[path_name] = psnr
                    ssim_by_path[path_name] = ssim
                    lpips_by_path[path_name] = lpips_val
                    rows.append({
                        "video": video_key, "frame": frame_idx, "seed": seed, "ablation": ablation_name,
                        "path": path_name, "psnr": psnr, "ssim": ssim, "lpips": lpips_val,
                        "latency_ms": outcome.latency_ms, "diffusion_steps": outcome.diffusion_steps,
                        "n_patches": outcome.n_patches, "wire_bytes": outcome.wire_bytes,
                        "roundtrip_bitexact": outcome.roundtrip_bitexact,
                        "failed": outcome.failed, "failure_stage": outcome.failure_stage,
                        "failure_message": outcome.failure_message,
                    })

                awgn_psnr = psnr_by_path.get("awgn")
                awgn_ssim = ssim_by_path.get("awgn")
                awgn_lpips = lpips_by_path.get("awgn")
                for row in rows:
                    row["psnr_delta_vs_awgn"] = (
                        row["psnr"] - awgn_psnr if row["psnr"] is not None and awgn_psnr is not None else None
                    )
                    row["ssim_delta_vs_awgn"] = (
                        row["ssim"] - awgn_ssim if row["ssim"] is not None and awgn_ssim is not None else None
                    )
                    row["lpips_delta_vs_awgn"] = (
                        row["lpips"] - awgn_lpips if row["lpips"] is not None and awgn_lpips is not None else None
                    )

                _append_csv_rows(path_comparison_csv, rows, PATH_COMPARISON_FIELDS)
                _append_csv_rows(failed_csv, failed_rows, FAILED_CASES_FIELDS)

                # Update the verdict-evidence indices for THIS group immediately
                # (in addition to whatever was reloaded from disk at start) so a
                # verdict can be computed for it below without waiting for a
                # future --resume to reload it from path_comparison.csv.
                psnr_index[group_key] = {
                    path_name: {
                        "psnr": psnr_by_path.get(path_name), "ssim": ssim_by_path.get(path_name),
                        "lpips": lpips_by_path.get(path_name),
                    }
                    for path_name in outcomes
                }

                # ── tensor stage stats + pairwise comparisons ──
                if recorder.enabled:
                    _append_jsonl_rows(tensor_stage_jsonl, recorder.rows)
                    stage_names = sorted({live_key[5] for live_key in recorder.live.keys()})
                    pair_rows = []
                    path_order = [p for p in requested_paths if p in outcomes and not outcomes[p].failed]
                    for i_a in range(len(path_order)):
                        for i_b in range(i_a + 1, len(path_order)):
                            path_a, path_b = path_order[i_a], path_order[i_b]
                            for stage in stage_names:
                                key_a = (video_key, frame_idx, seed, ablation_name, path_a, stage)
                                key_b = (video_key, frame_idx, seed, ablation_name, path_b, stage)
                                cmp = compare_tensors(recorder.live.get(key_a), recorder.live.get(key_b))
                                pair_rows.append({
                                    "video": video_key, "frame": frame_idx, "seed": seed, "ablation": ablation_name,
                                    "stage": stage, "path_a": path_a, "path_b": path_b, **cmp,
                                })
                    _append_csv_rows(tensor_pair_csv, pair_rows, TENSOR_PAIR_FIELDS)

                    inprocess_wire_pairs = _inprocess_wire_pairs(pair_rows)
                    if inprocess_wire_pairs:
                        pair_index[group_key] = inprocess_wire_pairs
                    recorder.clear_live()

                if _STOP_REQUESTED:
                    interrupted = True
                    break

            # Verdict computation: ALWAYS attempt this (reads exclusively from
            # psnr_index/pair_index, which by now include both this
            # invocation's fresh results above AND anything reloaded from
            # disk at start) — regardless of whether this frame's ablation
            # loop was just interrupted. Any group whose CSV/JSONL rows
            # already reached disk must not have its verdict evidence
            # orphaned (see the psnr_index/pair_index comment above for why).
            if not args.no_instrument_tensors:
                baseline_group = (video_key, str(frame_idx), "baseline")
                if baseline_group in psnr_index:
                    vae_direct_group = (video_key, str(frame_idx), "diffusion_bypass_vae_direct")
                    vae_direct_entry = psnr_index.get(vae_direct_group)
                    vae_direct_quality = (
                        {
                            "digital_inprocess_psnr": vae_direct_entry.get("digital_inprocess", {}).get("psnr"),
                            "digital_wire_psnr": vae_direct_entry.get("digital_wire", {}).get("psnr"),
                        }
                        if vae_direct_entry else None
                    )
                    # "Evidence arrived" means the vae-direct group was
                    # ATTEMPTED (its row exists in psnr_index), not that it
                    # succeeded -- a genuinely failed vae-direct group must
                    # still let the baseline verdict finalize (classify()
                    # already tolerates None PSNR values inside
                    # vae_direct_quality), not wait forever.
                    baseline_provisional = vae_direct_requested and vae_direct_entry is None
                    if baseline_provisional:
                        baseline_evidence_level = EVIDENCE_BASELINE_PENDING_VAE
                    elif vae_direct_requested:
                        baseline_evidence_level = EVIDENCE_BASELINE_WITH_VAE
                    else:
                        baseline_evidence_level = EVIDENCE_BASELINE_ONLY
                    _persist_verdict(
                        video_key, frame_idx, "baseline",
                        path_quality=_psnr_triplet(psnr_index, baseline_group),
                        comparisons=pair_index.get(baseline_group, []),
                        vae_direct_quality=vae_direct_quality,
                        edge_handling_equalized=False,
                        provisional=baseline_provisional,
                        evidence_level=baseline_evidence_level,
                    )
                # Edge-equalizing ablations (serialized_raw_edge / awgn_edge_retransmit)
                # get their OWN verdict, classified with edge_handling_equalized=True
                # so their (previously ignored) edge/decoder-stage divergence is
                # actually usable as packet_tx_rx_issue evidence.
                for eq_ablation in EDGE_EQUALIZING_ABLATIONS:
                    eq_group = (video_key, str(frame_idx), eq_ablation)
                    if eq_group in psnr_index:
                        _persist_verdict(
                            video_key, frame_idx, eq_ablation,
                            path_quality=_psnr_triplet(psnr_index, eq_group),
                            comparisons=pair_index.get(eq_group, []),
                            vae_direct_quality=None,
                            edge_handling_equalized=True,
                            evidence_level=EVIDENCE_AUXILIARY_EDGE,
                        )

            n_frames_processed += 1
            if interrupted:
                break
        if interrupted:
            break

    # Authoritative verdict set = everything ever recorded for this
    # --output-root (loaded at start + appended above), not just what this
    # invocation processed -- this is what keeps verdict_summary/REPORT.md
    # correct across --resume.
    all_per_video_verdicts = [
        verdicts_index[k] for k in sorted(verdicts_index, key=lambda k: (k[0], k[1], k[2]))
    ]

    # The run's ONE summary tally (verdict_summary / "종합 판정") reflects ONLY
    # the primary per-frame classification: ablation == "baseline" AND
    # status == "final". Folding in the serialized_raw_edge/
    # awgn_edge_retransmit auxiliary ablations' verdicts here would count the
    # SAME frame's evidence multiple times (baseline + N auxiliary rows all
    # agreeing is not N independent confirmations), and a still-"provisional"
    # baseline (waiting on diffusion_bypass_vae_direct evidence) has not yet
    # committed to a final classification. Every row is still fully preserved
    # in verdicts.jsonl / all_per_video_verdicts / REPORT.md's per-row table —
    # this filtering only narrows what feeds the ONE aggregate count.
    final_baseline_verdicts = [
        r for r in all_per_video_verdicts
        if r.get("ablation") == "baseline" and r.get("status", "final") == "final"
    ]
    n_provisional = sum(1 for r in all_per_video_verdicts if r.get("status") == "provisional")
    n_auxiliary = sum(1 for r in all_per_video_verdicts if r.get("ablation") in EDGE_EQUALIZING_ABLATIONS)
    n_baseline_with_vae = sum(
        1 for r in final_baseline_verdicts if r.get("evidence_level") == EVIDENCE_BASELINE_WITH_VAE
    )
    n_baseline_only = sum(
        1 for r in final_baseline_verdicts if r.get("evidence_level") == EVIDENCE_BASELINE_ONLY
    )

    verdict_summary = None
    if final_baseline_verdicts:
        from sgdjscc_lab.diagnostics.verdict import VerdictResult
        verdict_objs = [
            VerdictResult(verdict=r["verdict"], reason=r["reason"], first_divergent_stage=r["first_divergent_stage"])
            for r in final_baseline_verdicts
        ]
        verdict_summary = aggregate_verdicts(verdict_objs)

    failed_total = _count_csv_rows(failed_csv)
    write_summary_json(
        output_root, run_kind="float32_digital_diagnostics", dry_run=False,
        args=vars(args), verdict_summary=verdict_summary,
        counts={
            "n_frames_processed": n_frames_processed, "n_videos": len(entries), "interrupted": interrupted,
            "n_baseline_verdicts_final": len(final_baseline_verdicts),
            "n_baseline_verdicts_provisional": n_provisional,
            "n_auxiliary_edge_equalized_verdicts": n_auxiliary,
            "n_baseline_verdicts_with_vae_direct": n_baseline_with_vae,
            "n_baseline_verdicts_without_vae_direct": n_baseline_only,
        },
    )
    write_report_md(
        output_root, run_kind="float32_digital_diagnostics", dry_run=args.no_models,
        n_videos=len(entries), n_frames=len(frames), n_ablations=len(ablation_names),
        verdict_summary=verdict_summary, per_video_verdicts=all_per_video_verdicts,
        n_provisional=n_provisional, n_auxiliary=n_auxiliary,
        failed_count=failed_total,
        outputs={
            "path_comparison.csv": "path_comparison.csv", "tensor_stage_stats.jsonl": "tensor_stage_stats.jsonl",
            "tensor_pair_comparison.csv": "tensor_pair_comparison.csv", "failed_cases.csv": "failed_cases.csv",
            "verdicts.jsonl": "verdicts.jsonl", "summary.json": "summary.json",
        },
    )

    manifest_final = rm.build_run_manifest(
        run_id=manifest_initial["run_id"], command_argv=sys.argv, command_source="captured",
        seed=args.seed, resolved_config_path=None, config_source_path=args.config,
        dataset_ref=str(args.dataset_root), dataset_hash=_dataset_content_hash(args.dataset_root, entries),
        checkpoints=(None if args.no_models else {n: model_root / n for n in _CHECKPOINT_NAMES if (model_root / n).exists()}),
        repo_root=_REPO_ROOT, cuda_device_index=0,
        extra={"phase": "final", "signature": signature, "interrupted": interrupted,
               "output_artifact_sha256": {
                   name: rm.sha256_file(output_root / name)
                   for name in ("path_comparison.csv", "tensor_stage_stats.jsonl",
                                "tensor_pair_comparison.csv", "failed_cases.csv", "verdicts.jsonl", "summary.json")
                   if (output_root / name).exists()
               }},
    )
    rm.write_run_manifest(output_root / "run_manifest.json", manifest_final)

    if interrupted:
        print("[diagnose_float32_digital_quality] interrupted by signal; state saved, safe to --resume.", file=sys.stderr)
        return 130
    if failed_total:
        print(f"[diagnose_float32_digital_quality] completed with {failed_total} failed case(s); see failed_cases.csv", file=sys.stderr)
        return 3
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
