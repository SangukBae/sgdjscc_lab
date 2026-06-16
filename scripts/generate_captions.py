#!/usr/bin/env python
"""scripts/generate_captions.py – Generate per-image caption sidecars (<stem>.txt).

Promotes an **image-only** folder (e.g. CelebA / CelebA-HQ, which ship without
captions) into a text-image **pair** dataset usable by the text-guided stages
(`text_dm` / `controlnet`).  Writes a `<stem>.txt` next to each image so the
existing `caption_source: sidecar` path picks them up — no loader change needed.

Caption modes (``--mode``)
--------------------------
  ``fixed``     a single fixed template for every image (default; CPU-only).
                e.g. "a portrait photo of a person".
  ``filename``  derive a pseudo-caption from the filename stem (cheap; smoke).
  ``model``     run the repo's BLIP-2 caption extractor (guidance/text_extractor).
                Heavy: needs transformers + weights + ideally a GPU.

Fidelity note
-------------
Auto-generated captions are **paper-like, NOT paper-faithful**.  The paper's
CelebA-HQ captions are its own; ``fixed``/``filename`` are placeholders and
``model`` (BLIP-2) approximates captioning but is not the paper's exact text.
Use this to *enable* text-stage training on caption-less data, not to claim
paper-exact reproduction.

Usage
-----
    # fixed template (CPU)
    python scripts/generate_captions.py --input data/celeba/train \
        --mode fixed --text "a portrait photo of a person"

    # all CelebA splits at once (the scan is ALWAYS recursive)
    python scripts/generate_captions.py --input data/celeba --mode fixed

    # BLIP-2 model captions (GPU recommended)
    python scripts/generate_captions.py --input data/celeba_hq/train \
        --mode model --device cuda:0
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Make ``src/`` importable when run as a script.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.data.image_dataset import _IMG_EXTS, _list_images  # noqa: E402

logger = logging.getLogger("generate_captions")

DEFAULT_FIXED = "a portrait photo of a person"


def _filename_caption(stem: str) -> str:
    """Pseudo-caption from a filename stem (``_``/``-`` → spaces)."""
    cap = stem.replace("_", " ").replace("-", " ").strip()
    return cap or DEFAULT_FIXED


def generate_captions(
    input_dir: str | Path,
    *,
    mode: str = "fixed",
    text: str = DEFAULT_FIXED,
    overwrite: bool = False,
    device: str = "cpu",
    limit: Optional[int] = None,
) -> dict:
    """Write ``<stem>.txt`` next to every image under *input_dir*.

    Returns a summary dict ``{written, skipped, total}``.  ``mode`` ∈
    {``fixed``, ``filename``, ``model``}.  Existing sidecars are skipped unless
    *overwrite*.  Recursion is always on (uses the same recursive image scan as
    the datasets), so passing ``data/celeba`` covers train/val/test at once.
    """
    mode = str(mode).lower()
    if mode not in ("fixed", "filename", "model"):
        raise ValueError(f"--mode must be fixed|filename|model, got {mode!r}")

    files: List[Path] = _list_images(input_dir)
    if limit is not None:
        files = files[: int(limit)]
    total = len(files)
    logger.info("Found %d images under %s (mode=%s)", total, input_dir, mode)

    extractor = None
    dev = None
    if mode == "model":
        import torch
        from sgdjscc_lab.guidance.text_extractor import build_text_extractor
        from sgdjscc_lab.io import load_image_as_tensor
        dev = torch.device(device)
        extractor = build_text_extractor(dev)

    written = skipped = 0
    for i, fpath in enumerate(files):
        txt_path = fpath.with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            skipped += 1
            continue
        if mode == "fixed":
            caption = text
        elif mode == "filename":
            caption = _filename_caption(fpath.stem)
        else:  # model
            import torch
            import torch.nn.functional as F
            img = load_image_as_tensor(fpath)            # [1,3,H,W] in [0,1]
            img = F.interpolate(img, size=(128, 128), mode="bilinear", align_corners=False)
            out = extractor.extract(img.to(dev), dev)    # list-of-lists
            caption = (out[0][0] if out and out[0] else text)
        txt_path.write_text(str(caption).strip() + "\n", encoding="utf-8")
        written += 1
        if (i + 1) % 5000 == 0:
            logger.info("  %d/%d processed (%d written)", i + 1, total, written)

    logger.info("Done: %d written, %d skipped, %d total", written, skipped, total)
    return {"written": written, "skipped": skipped, "total": total}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate per-image caption sidecars (<stem>.txt).")
    p.add_argument("--input", "-i", required=True,
                   help="Image folder (recursively scanned; e.g. data/celeba or data/celeba/train).")
    p.add_argument("--mode", default="fixed", choices=["fixed", "filename", "model"],
                   help="Caption source (default: fixed).")
    p.add_argument("--text", default=DEFAULT_FIXED,
                   help=f"Fixed caption template for --mode fixed (default: {DEFAULT_FIXED!r}).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing <stem>.txt sidecars (default: skip).")
    p.add_argument("--device", default="cpu", help="Device for --mode model (e.g. cuda:0).")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N images (debug).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    summary = generate_captions(
        args.input, mode=args.mode, text=args.text, overwrite=args.overwrite,
        device=args.device, limit=args.limit,
    )
    print(f"captions: written={summary['written']} skipped={summary['skipped']} "
          f"total={summary['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
