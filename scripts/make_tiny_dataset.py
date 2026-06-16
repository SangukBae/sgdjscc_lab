#!/usr/bin/env python
"""make_tiny_dataset.py – Generate a tiny dataset for real-model SMOKE training.

Smoke training only needs a handful of *valid* samples — enough for the real
model to take 1–2 optimizer steps and write/restore a checkpoint.  This script
materialises a stage-appropriate tiny dataset:

  --stage jscc        N images (image-only)
  --stage text_dm     N images + per-image .txt caption sidecars
  --stage controlnet  N images + .txt captions (+ optional precomputed edges)
  --stage edge_codec  N images (edges are computed on the fly via Canny)
  --stage all         a shared train/ + val/ tree usable by every stage

Layout (``--stage all``):

  <out>/
  ├── train/
  │   ├── sample_000.png      tiny RGB image
  │   ├── sample_000.txt      caption sidecar ("a photo of sample 000")
  │   ├── sample_000_edge.png precomputed edge (only with --edges sidecar)
  │   └── …
  └── val/
      └── … (same structure)

Images are small random-but-structured tiles (a few coloured rectangles) so the
on-the-fly Canny edge detector produces non-trivial edges.

Usage
-----
python scripts/make_tiny_dataset.py --stage all --out ../data/tiny --n 6 --val 2
python scripts/make_tiny_dataset.py --stage edge_codec --out ../data/tiny_edges --n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch


def _structured_image(seed: int, size: int) -> torch.Tensor:
    """A small image with a few solid rectangles → real edges for Canny."""
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(3, size, size, generator=g) * 0.15  # faint background
    for _ in range(3):
        c = torch.rand(3, 1, 1, generator=g)
        x0 = int(torch.randint(0, size // 2, (1,), generator=g))
        y0 = int(torch.randint(0, size // 2, (1,), generator=g))
        x1 = x0 + int(torch.randint(size // 4, size // 2, (1,), generator=g))
        y1 = y0 + int(torch.randint(size // 4, size // 2, (1,), generator=g))
        img[:, y0:min(y1, size), x0:min(x1, size)] = c
    return img.clamp(0, 1)


def _write_split(split_dir: Path, n: int, size: int, *, captions: bool,
                 edges: bool, start: int) -> None:
    from sgdjscc_lab.io import save_tensor_as_image
    from sgdjscc_lab.data.datasets import _canny_edge
    split_dir.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + n):
        stem = f"sample_{i:03d}"
        img = _structured_image(seed=1000 + i, size=size)
        save_tensor_as_image(img, split_dir / f"{stem}.png")
        if captions:
            (split_dir / f"{stem}.txt").write_text(
                f"a photo of sample {i:03d}", encoding="utf-8")
        if edges:
            save_tensor_as_image(
                _canny_edge(img).repeat(3, 1, 1), split_dir / f"{stem}_edge.png")
    print(f"  wrote {n} samples → {split_dir}"
          f"  (captions={captions}, edges={edges})")


def main() -> None:
    p = argparse.ArgumentParser(description="Tiny smoke-training dataset generator")
    p.add_argument("--stage", default="all",
                   choices=["jscc", "text_dm", "controlnet", "edge_codec", "all"])
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--n", type=int, default=6, help="Number of train samples")
    p.add_argument("--val", type=int, default=2, help="Number of val samples")
    p.add_argument("--size", type=int, default=128, help="Image size (multiple of 128)")
    p.add_argument("--edges", action="store_true",
                   help="Also write precomputed *_edge.png sidecars "
                        "(for edge_source: sidecar). Canny-on-the-fly needs none.")
    args = p.parse_args()

    # Which fields each stage consumes.
    captions = args.stage in ("text_dm", "controlnet", "all")
    edges = args.edges and args.stage in ("controlnet", "edge_codec", "all")

    out = Path(args.out)
    print(f"Generating tiny '{args.stage}' dataset at {out} "
          f"(train={args.n}, val={args.val}, size={args.size})")
    _write_split(out / "train", args.n, args.size,
                 captions=captions, edges=edges, start=0)
    if args.val > 0:
        _write_split(out / "val", args.val, args.size,
                     captions=captions, edges=edges, start=args.n)
    print("Done. Use it with --train-list", out / "train",
          "(--val-list", str(out / "val") + ").")


if __name__ == "__main__":
    main()
