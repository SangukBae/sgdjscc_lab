#!/usr/bin/env python3
"""Compare saved video reconstruction/generation frames with source frames.

The temporal metrics produced by ``evaluate_video.py`` describe semantic-packet
consistency; they are not a substitute for direct visual fidelity.  This tool
adds the missing paired-frame evidence for completed batch runs:

* PSNR / SSIM / LPIPS
* CLIP image-image similarity
* separate aggregates for all reconstructed frames and generated-only frames

It accepts either one run directory or a batch root containing mode/video run
directories.  Source frames are discovered below ``extracted_frames/`` and
matched by the numeric suffix in ``frame_XXXXX.png``, ``recon_XXXXX.png``, and
``generated_XXXXX.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
from sgdjscc_lab.evaluators.quality import QualityEvaluator


def _frame_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Frame filename has no numeric suffix: {path}") from exc


def discover_run_dirs(root: Path) -> List[Path]:
    """Return completed run dirs under *root*, without duplicate nesting."""
    root = Path(root).resolve()
    if (root / "recon_frames").is_dir():
        return [root]
    return sorted({p.parent for p in root.rglob("recon_frames") if p.is_dir()})


def _source_index(run_dir: Path) -> Dict[int, Path]:
    extracted = run_dir / "extracted_frames"
    if not extracted.is_dir():
        raise FileNotFoundError(f"Missing extracted_frames/: {run_dir}")
    files = sorted(extracted.rglob("frame_*.png"))
    if not files:
        raise FileNotFoundError(f"No frame_*.png under {extracted}")
    result: Dict[int, Path] = {}
    for path in files:
        index = _frame_index(path)
        if index in result:
            raise ValueError(
                f"Ambiguous source frame index {index} under {extracted}: "
                f"{result[index]} and {path}"
            )
        result[index] = path
    return result


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _candidate_paths(run_dir: Path, kind: str) -> List[Path]:
    if kind == "recon":
        return sorted((run_dir / "recon_frames").glob("recon_*.png"))
    if kind == "generated":
        return sorted((run_dir / "generated_frames").glob("generated_*.png"))
    raise ValueError(f"Unknown kind: {kind}")


def _finite_mean(values: Iterable[object]):
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else None


def _finite_std(values: Iterable[object]):
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None)


def evaluate_run(
    run_dir: Path,
    *,
    kinds: Sequence[str],
    device: str,
    use_lpips: bool,
    use_clip: bool,
    clip_model: str = "ViT-B/32",
) -> List[dict]:
    run_dir = Path(run_dir).resolve()
    sources = _source_index(run_dir)
    torch_device = torch.device(device)
    quality = QualityEvaluator(use_lpips=use_lpips, device=torch_device)
    clip_eval = CLIPScoreEvaluator(model_name=clip_model, device=torch_device) if use_clip else None
    rows: List[dict] = []

    for kind in kinds:
        for candidate_path in _candidate_paths(run_dir, kind):
            index = _frame_index(candidate_path)
            source_path = sources.get(index)
            if source_path is None:
                raise FileNotFoundError(
                    f"No source frame for {candidate_path.name} (index={index}) in {run_dir}"
                )
            with Image.open(source_path) as src_image, Image.open(candidate_path) as cand_image:
                src_rgb = src_image.convert("RGB")
                cand_rgb = cand_image.convert("RGB")
                source_size = src_rgb.size
                candidate_size = cand_rgb.size
                resized = candidate_size != source_size
                if resized:
                    cand_rgb = cand_rgb.resize(source_size, Image.Resampling.BICUBIC)
                source = _to_tensor(src_rgb)
                candidate = _to_tensor(cand_rgb)

            metrics = quality.evaluate(source, candidate)
            clip_score = (
                clip_eval.image_image_score(source, candidate) if clip_eval is not None else None
            )
            rows.append({
                "run_dir": str(run_dir),
                "mode": run_dir.parent.name,
                "video_id": run_dir.name,
                "kind": kind,
                "frame_index": index,
                "source_path": str(source_path),
                "candidate_path": str(candidate_path),
                "source_width": source_size[0],
                "source_height": source_size[1],
                "candidate_width": candidate_size[0],
                "candidate_height": candidate_size[1],
                "resized_for_metric": resized,
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "clip_image_image": clip_score,
            })
    return rows


def summarize(rows: Sequence[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for row in rows:
        key = (row["run_dir"], row["mode"], row["video_id"], row["kind"])
        groups.setdefault(key, []).append(row)
    result = []
    for (run_dir, mode, video_id, kind), items in sorted(groups.items()):
        summary = {
            "run_dir": run_dir,
            "mode": mode,
            "video_id": video_id,
            "kind": kind,
            "n_frames": len(items),
            "n_resized": sum(bool(row["resized_for_metric"]) for row in items),
        }
        for metric in ("psnr", "ssim", "lpips", "clip_image_image"):
            summary[f"{metric}_mean"] = _finite_mean(row[metric] for row in items)
            summary[f"{metric}_std"] = _finite_std(row[metric] for row in items)
        result.append(summary)
    return result


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_root(
    run_root: Path,
    *,
    kinds: Sequence[str],
    device: str = "cuda:0",
    use_lpips: bool = True,
    use_clip: bool = True,
    clip_model: str = "ViT-B/32",
) -> tuple:
    rows: List[dict] = []
    run_dirs = discover_run_dirs(run_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run directory with recon_frames/ below {run_root}")
    for run_dir in run_dirs:
        selected = list(kinds)
        if "auto" in selected:
            selected = ["recon"]
            if (run_dir / "generated_frames").is_dir():
                selected.append("generated")
        rows.extend(evaluate_run(
            run_dir, kinds=selected, device=device, use_lpips=use_lpips,
            use_clip=use_clip, clip_model=clip_model,
        ))
    return rows, summarize(rows)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Paired source-vs-reconstruction/generation frame quality evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-root", required=True,
                        help="One run dir or a batch root containing run dirs.")
    parser.add_argument("--kinds", default="auto",
                        help="Comma list: recon,generated; auto evaluates recon and generated when present.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    run_root = Path(args.run_root).resolve()
    kinds = [part.strip() for part in args.kinds.split(",") if part.strip()]
    allowed = {"auto", "recon", "generated"}
    unknown = set(kinds) - allowed
    if unknown or ("auto" in kinds and len(kinds) != 1):
        raise SystemExit(f"Invalid --kinds {args.kinds!r}; use auto or recon,generated")

    output_csv = Path(args.output_csv) if args.output_csv else run_root / "frame_quality.csv"
    summary_csv = Path(args.summary_csv) if args.summary_csv else run_root / "frame_quality_summary.csv"
    summary_json = Path(args.summary_json) if args.summary_json else run_root / "frame_quality_summary.json"
    rows, summary = evaluate_root(
        run_root, kinds=kinds, device=args.device,
        use_lpips=not args.no_lpips, use_clip=not args.no_clip,
        clip_model=args.clip_model,
    )
    _write_csv(output_csv, rows)
    _write_csv(summary_csv, summary)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Frame rows: {len(rows)} -> {output_csv}")
    print(f"Summary rows: {len(summary)} -> {summary_csv} / {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
