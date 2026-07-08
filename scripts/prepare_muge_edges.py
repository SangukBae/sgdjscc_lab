#!/usr/bin/env python
"""scripts/prepare_muge_edges.py – Precompute MuGE soft-edge sidecars.

The paper / public SGDJSCC use **MuGE soft edges** as the structural guidance.
For the paper-faithful training path (``edge_source: muge_sidecar``), precompute a
MuGE sidecar next to (or into a dir alongside) each image, so the Stage-3
ControlNet / edge_codec datasets read it without running MuGE in every
DataLoader worker.

This reuses the **same** MuGE network the inference path uses
(``guidance/edge_extractor.build_edge_extractor``) and the representation
helpers in ``data/datasets.py`` — see docs/paper_gap_closure.md [1].

Usage
-----
    python scripts/prepare_muge_edges.py \
        --input data/coco/train2017 --model-root ../checkpoints --device cuda:0
    # → writes data/coco/train2017/<stem>_muge.png for every image

    # write into a separate dir (referenced via train.dataset.edge_dir):
    python scripts/prepare_muge_edges.py --input data/coco/train2017 \
        --out-dir data/coco/train2017_muge --device cuda:0

    # preserve the inference-carried edge+uncertainty representation:
    python scripts/prepare_muge_edges.py --input data/coco/train2017 \
        --repr edge_uncertainty --out-dir data/coco/train2017_muge_eu
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from PIL import UnidentifiedImageError

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.utils.bad_images import quarantine_bad_image  # noqa: E402
from sgdjscc_lab.utils.text_progress import UnitProgress  # noqa: E402

logger = logging.getLogger("prepare_muge_edges")


def prepare_muge_edges(
    input_dir: str | Path,
    *,
    model_root: str | Path,
    out_dir: Optional[str | Path] = None,
    device: str = "cpu",
    size: int = 128,
    repr_name: str = "reduced",
    overwrite: bool = False,
    limit: Optional[int] = None,
    progress_every: int = 500,
    label: Optional[str] = None,
    unit_index: int = 1,
    unit_total: int = 1,
    gpu: Optional[str] = None,
) -> dict:
    """Write MuGE soft-edge sidecars for every image under *input_dir*.

    ``repr_name`` selects the saved representation (see data/datasets._MUGE_REPRS):
      * ``reduced`` (default) → 1-channel ``<stem>_muge.png`` (legacy / smallest).
      * ``edge_uncertainty`` (2ch) / ``multi`` (11ch) → ``<stem>_muge.npy`` (float16),
        preserving the multi-channel MuGE information (loaded as-is, not collapsed).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    from sgdjscc_lab.data.image_dataset import _list_images
    from sgdjscc_lab.data.datasets import muge_channels
    from sgdjscc_lab.guidance.edge_extractor import build_edge_extractor
    from sgdjscc_lab.io import load_image_as_tensor, save_tensor_as_image

    repr_name = str(repr_name).lower()
    as_png = repr_name == "reduced"
    dev = torch.device(device)
    extractor = build_edge_extractor(model_root, dev)   # same MuGE as inference

    files = _list_images(input_dir)
    if limit is not None:
        files = files[: int(limit)]
    progress_every = max(1, int(progress_every))
    out_base = Path(out_dir) if out_dir else None
    if out_base is not None:
        out_base.mkdir(parents=True, exist_ok=True)
    ext = ".png" if as_png else ".npy"
    written = skipped = bad = 0
    total = len(files)
    # Single-line in-place progress on a TTY; line-by-line logging otherwise.
    progress = UnitProgress(stage="muge", total=total, label=label, gpu=gpu,
                            unit_index=unit_index, unit_total=unit_total, logger=logger)
    for i, fpath in enumerate(files):
        dst = (out_base / f"{fpath.stem}_muge{ext}") if out_base is not None \
            else fpath.with_name(f"{fpath.stem}_muge{ext}")
        if dst.exists() and not overwrite:
            skipped += 1
        else:
            try:
                img = load_image_as_tensor(fpath)               # [1,3,H,W] in [0,1]
                img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
                data, unc = extractor.extract(img.to(dev), dev)
                edge = muge_channels(data, unc, repr_name)       # [C,H,W] in [0,1]
                if as_png:
                    save_tensor_as_image(edge.repeat(3, 1, 1).cpu(), dst)  # 1ch → 3ch png
                else:
                    np.save(dst, edge.cpu().numpy().astype(np.float16))    # [C,H,W]
                written += 1
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                moved_to = quarantine_bad_image(fpath)
                bad += 1
                logger.warning("Bad image quarantined: %s -> %s (%s)", fpath, moved_to, exc)
        processed = i + 1
        if total and (processed % progress_every == 0 or processed == total):
            progress.update(processed)
    if total:
        progress.close(total)  # commit the final 100% line (TTY only)
    logger.info("Done: %d written, %d skipped, %d bad, %d total (repr=%s, %s)",
                written, skipped, bad, total, repr_name, ext)
    return {"written": written, "skipped": skipped, "bad": bad, "total": total}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Precompute MuGE soft-edge sidecars.")
    p.add_argument("--input", "-i", required=True, help="Image folder (recursive).")
    p.add_argument("--model-root", default="../checkpoints",
                   help="Dir with muge-epoch-19-checkpoint.pth (default ../checkpoints).")
    p.add_argument("--out-dir", default=None,
                   help="Write edges here as <stem>_muge.(png|npy) "
                        "(default: next to images).")
    p.add_argument("--device", default="cpu", help="cpu | cuda:0 …")
    p.add_argument("--size", type=int, default=128, help="Edge map size (default 128).")
    p.add_argument("--repr", dest="repr_name", default="reduced",
                   choices=["reduced", "edge_uncertainty", "multi"],
                   help="MuGE representation: reduced=1ch png (default) | "
                        "edge_uncertainty=2ch npy (closest to inference) | multi=11ch npy.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=500,
                   help="Log progress every N images (default: 500).")
    # Progress-bar metadata (set by the parallel prep script; optional standalone).
    p.add_argument("--label", default=None,
                   help="Dataset label for the progress bar, e.g. 'journey_pairs/val'. "
                        "When set, logs a '[muge][gpu=..][label][i/N] |bar| pct' line.")
    p.add_argument("--unit-index", type=int, default=1,
                   help="This unit's 1-based index within the dataset (default: 1).")
    p.add_argument("--unit-total", type=int, default=1,
                   help="Total units (shards) in the dataset (default: 1).")
    p.add_argument("--gpu", default=None, help="GPU id shown in the progress bar (display only).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = _parse_args(argv)
    s = prepare_muge_edges(a.input, model_root=a.model_root, out_dir=a.out_dir,
                           device=a.device, size=a.size, repr_name=a.repr_name,
                           overwrite=a.overwrite, limit=a.limit,
                           progress_every=a.progress_every,
                           label=a.label, unit_index=a.unit_index,
                           unit_total=a.unit_total, gpu=a.gpu)
    print("muge edges: "
          f"written={s['written']} skipped={s['skipped']} bad={s['bad']} total={s['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
