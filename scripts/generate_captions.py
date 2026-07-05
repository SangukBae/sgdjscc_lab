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
  ``model``     run the Qwen2.5-VL-3B-Instruct caption extractor
                (guidance/qwen_caption).  Heavy: needs a recent transformers
                (>=4.49) + weights + ideally a GPU.  This is SEPARATE from the
                BLIP-2 model used inside the inference/eval pipelines, so it does
                not affect the paper-faithful forward pass.

Fidelity note
-------------
Auto-generated captions are **paper-like, NOT paper-faithful**.  The paper's
CelebA-HQ captions are its own; ``fixed``/``filename`` are placeholders and
``model`` (Qwen2.5-VL) approximates captioning but is not the paper's exact text.
Use this to *enable* text-stage training on caption-less data, not to claim
paper-exact reproduction.

Usage
-----
    # fixed template (CPU)
    python scripts/generate_captions.py --input data/celeba/train \
        --mode fixed --text "a portrait photo of a person"

    # all CelebA splits at once (the scan is ALWAYS recursive)
    python scripts/generate_captions.py --input data/celeba --mode fixed

    # Qwen2.5-VL model captions (GPU recommended; needs transformers>=4.49)
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
from sgdjscc_lab.utils.text_progress import format_unit_progress  # noqa: E402

logger = logging.getLogger("generate_captions")

DEFAULT_FIXED = "a portrait photo of a person"

# Qwen2.5-VL knobs for --mode model. Kept as plain literals here (not imported
# from guidance.qwen_caption) so --help / fixed / filename modes never import
# torch. ``None`` prompt → the module default prompt.
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MODEL_MAX_NEW_TOKENS = 64
DEFAULT_MODEL_BATCH_SIZE = 1
DEFAULT_MODEL_MAX_PIXELS = 512 * 512


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
    progress_every: int = 500,
    model_id: str = QWEN_MODEL_ID,
    prompt: Optional[str] = None,
    max_new_tokens: int = DEFAULT_MODEL_MAX_NEW_TOKENS,
    batch_size: int = DEFAULT_MODEL_BATCH_SIZE,
    max_pixels: Optional[int] = DEFAULT_MODEL_MAX_PIXELS,
    min_pixels: Optional[int] = None,
    label: Optional[str] = None,
    unit_index: int = 1,
    unit_total: int = 1,
    gpu: Optional[str] = None,
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
    if mode == "model":
        import torch
        from sgdjscc_lab.guidance.qwen_caption import build_qwen_caption_extractor
        dev = torch.device(device)
        extractor = build_qwen_caption_extractor(
            dev, model_id=model_id, prompt=prompt, max_new_tokens=max_new_tokens,
            max_pixels=max_pixels, min_pixels=min_pixels)

    progress_every = max(1, int(progress_every))
    batch_size = max(1, int(batch_size))
    written = skipped = 0
    processed = 0
    caption_dirs = set()

    def _log_progress() -> None:
        if not (total and (processed % progress_every == 0 or processed == total)):
            return
        if label:
            logger.info("%s", format_unit_progress(
                stage="caption", gpu=gpu, label=label, unit_index=unit_index,
                unit_total=unit_total, processed=processed, total=total))
        else:
            logger.info("  %.1f%% (%d/%d)", 100.0 * processed / total, processed, total)

    # --mode model batches images through Qwen2.5-VL; ``_pending`` buffers the
    # image paths awaiting a batched generate() call.
    _pending: List[Path] = []

    def _flush_model_batch() -> None:
        nonlocal written, processed
        if not _pending:
            return
        from PIL import Image
        images = []
        for fp in _pending:
            with Image.open(fp) as im:
                images.append(im.convert("RGB"))
        captions = extractor.caption_images(images)
        for fp, cap in zip(_pending, captions):
            caption = cap.strip() if cap and cap.strip() else text
            out_path = fp.with_suffix(".txt")
            out_path.write_text(caption + "\n", encoding="utf-8")
            caption_dirs.add(out_path.parent)
            written += 1
            processed += 1
            _log_progress()
        _pending.clear()

    for fpath in files:
        txt_path = fpath.with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            skipped += 1
            processed += 1
            _log_progress()
            continue
        if mode == "model":
            _pending.append(fpath)
            if len(_pending) >= batch_size:
                _flush_model_batch()
            continue
        caption = text if mode == "fixed" else _filename_caption(fpath.stem)
        txt_path.write_text(str(caption).strip() + "\n", encoding="utf-8")
        caption_dirs.add(txt_path.parent)
        written += 1
        processed += 1
        _log_progress()

    if mode == "model":
        _flush_model_batch()

    # Provenance marker: drop a sentinel in every directory that received an
    # auto-caption (and the input root) so paper_mode can REFUSE to train a
    # text stage on these (auto-captions are NOT paper-faithful — see
    # sgdjscc_lab/paper_mode.py and docs/paper_gap_closure.md).
    _write_provenance(set(caption_dirs) | {Path(input_dir)}, mode=mode, text=text,
                      written=written, model_id=(model_id if mode == "model" else None))

    logger.info("Done: %d written, %d skipped, %d total", written, skipped, total)
    return {"written": written, "skipped": skipped, "total": total}


def _write_provenance(dirs, *, mode: str, text: str, written: int,
                      model_id: Optional[str] = None) -> None:
    """Write the auto-caption provenance sentinel into each directory."""
    import json
    import time

    from sgdjscc_lab.paper_mode import AUTOCAPTION_SENTINEL
    payload = {
        "tool": "scripts/generate_captions.py",
        "mode": mode,
        "fixed_text": text if mode == "fixed" else None,
        "model_id": model_id,
        "written": int(written),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fidelity": "paper-like (auto-generated; NOT paper-faithful)",
        "note": ("These .txt captions are auto-generated and are blocked under "
                 "paper_mode. Use a dataset whose captions ship with it for the "
                 "paper-faithful path."),
    }
    for d in dirs:
        try:
            (Path(d) / AUTOCAPTION_SENTINEL).write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass


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
    p.add_argument("--progress-every", type=int, default=500,
                   help="Log progress every N images (default: 500).")
    # --mode model (Qwen2.5-VL) knobs. Ignored by fixed/filename modes.
    p.add_argument("--model-id", default=QWEN_MODEL_ID,
                   help=f"HF model id for --mode model (default: {QWEN_MODEL_ID!r}).")
    p.add_argument("--prompt", default=None,
                   help="Caption instruction for --mode model (default: built-in concise prompt).")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MODEL_MAX_NEW_TOKENS,
                   help=f"Max new tokens per caption (default: {DEFAULT_MODEL_MAX_NEW_TOKENS}).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_MODEL_BATCH_SIZE,
                   help=f"Images per Qwen batch for --mode model (default: {DEFAULT_MODEL_BATCH_SIZE}).")
    p.add_argument("--max-pixels", type=int, default=DEFAULT_MODEL_MAX_PIXELS,
                   help=f"Max H*W per image (vision-token cap; default: {DEFAULT_MODEL_MAX_PIXELS}).")
    p.add_argument("--min-pixels", type=int, default=None,
                   help="Min H*W per image (default: model default).")
    # Progress-bar metadata (set by the parallel prep script; optional standalone).
    p.add_argument("--label", default=None,
                   help="Dataset label for the progress bar, e.g. 'sa1b_images/train'. "
                        "When set, logs a '[caption][gpu=..][label][i/N] |bar| pct' line.")
    p.add_argument("--unit-index", type=int, default=1,
                   help="This unit's 1-based index within the dataset (default: 1).")
    p.add_argument("--unit-total", type=int, default=1,
                   help="Total units (shards) in the dataset (default: 1).")
    p.add_argument("--gpu", default=None, help="GPU id shown in the progress bar (display only).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    summary = generate_captions(
        args.input, mode=args.mode, text=args.text, overwrite=args.overwrite,
        device=args.device, limit=args.limit, progress_every=args.progress_every,
        model_id=args.model_id, prompt=args.prompt, max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size, max_pixels=args.max_pixels, min_pixels=args.min_pixels,
        label=args.label, unit_index=args.unit_index, unit_total=args.unit_total, gpu=args.gpu,
    )
    print(f"captions: written={summary['written']} skipped={summary['skipped']} "
          f"total={summary['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
