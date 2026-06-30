#!/usr/bin/env python
"""prepare_sa1b.py – Convert SA-1B (Segment Anything) tar shards into image-only
training folders, **one shard at a time**, deleting each source tar only after its
output is verified (disk-bounded unpack of the ~750 GB SA-1B set).

Why image-only?  SA-1B shards hold ``sa_<id>.jpg`` (image) + ``sa_<id>.json`` (mask
annotations) — **no captions**. This repo's training loaders read images by a
recursive scan (``data/image_dataset._list_images`` → ``_IMG_EXTS``) and the
caption/edge stages need text/edge inputs SA-1B does not provide. So SA-1B is
usable as an **image-only** dataset for the image stages (``jscc`` / ``edge_codec``
/ ``csi_estimation``); the ``.json`` masks are not consumed by any current training
dataset, so they are dropped (the rationale, in code: only ``.jpg`` is extracted).

  sa_000020.tar              ->  sa1b_images/train/sa_000020/sa_226692.jpg
    ./sa_226692.jpg                               + …
    ./sa_226692.json (mask)   ← dropped

Output layout (matches the recursive loader; pass the split root as --train-list):

  sa1b_images/
  ├── train/<shard_tag>/*.jpg
  └── val/<shard_tag>/*.jpg

Safety / disk (mirrors scripts/prepare_cc3m.py's sequential design):
* **Atomic per shard**: images are extracted into a staging dir OUTSIDE the split
  (``<out>/.sa1b_tmp/<tag>.incoming/``); only after the whole shard succeeds is a
  ``.shard_done`` completion marker written and the staging dir atomically renamed
  to ``<split>/<tag>/``. A crash leaves only staging (cleaned + retried next run);
  a committed shard always has the marker → resume skips it.
* **Delete only after a VERIFIED commit**: ``--delete-shard-on-success`` removes the
  source ``.tar`` strictly after the shard committed AND a sample image decodes.
  A failed shard keeps its tar. Source tars are never touched otherwise.
* Broken / truncated images (PIL ``verify``) are skipped and counted, not fatal.

train/val split (deterministic by SHARD NUMBER, stable under tar deletion):
  a shard ``sa_<N>`` goes to **val** iff ``N % --val-every == 0``, else **train**
  (``--val-every 0`` or ``--all-train`` → everything to train). The number comes
  from the filename, not the position, so deletion does not shift the split.

Usage
-----
# preview (writes / deletes nothing)
python scripts/prepare_sa1b.py --dry-run

# convert shard-by-shard, freeing each tar after it is verified (resumable):
python scripts/prepare_sa1b.py --delete-shard-on-success

# bound a run to N shards; custom paths:
python scripts/prepare_sa1b.py --shard-dir data/sa1b/raw --output-dir data/sa1b_images \
    --limit-shards 2 --delete-shard-on-success
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("prepare_sa1b")

# Image suffixes (matches data/image_dataset._IMG_EXTS).
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Completion marker (hidden, no extension → excluded by the loader's _IMG_EXTS filter).
_SHARD_DONE_MARKER = ".shard_done"
_INCOMING_SUFFIX = ".incoming"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shard_tag(shard: Path) -> str:
    """Per-shard output subdir name = the tar stem (e.g. ``sa_000020``)."""
    return shard.stem


def _shard_number(shard: Path) -> int:
    """Trailing number in the shard name (for the deterministic train/val split)."""
    m = re.search(r"(\d+)(?!.*\d)", shard.stem)
    return int(m.group(1)) if m else 0


def _split_for_shard(shard: Path, val_every: int, all_train: bool) -> str:
    if all_train or val_every <= 0:
        return "train"
    return "val" if (_shard_number(shard) % val_every == 0) else "train"


def _valid_image_bytes(raw: bytes) -> bool:
    """True if *raw* decodes as a non-truncated image (PIL); permissive if no PIL."""
    try:
        from PIL import Image
    except Exception:  # pragma: no cover
        return True
    try:
        Image.open(io.BytesIO(raw)).verify()
        return True
    except Exception:
        return False


def _count_images(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)


def _is_shard_done(final_dir: Path) -> bool:
    return (final_dir / _SHARD_DONE_MARKER).is_file()


def _stamp_marker(final_dir: Path, shard_name: str, tag: str, images: int,
                  *, adopted: bool = False) -> None:
    import json
    import time
    (final_dir / _SHARD_DONE_MARKER).write_text(
        json.dumps({"shard": shard_name, "tag": tag, "images": images,
                    "adopted": adopted, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n",
        encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_images(tar: tarfile.TarFile, dest_dir: Path, *,
                    counters: Counter, dry_run: bool) -> int:
    """Extract every valid ``.jpg`` image from *tar* into *dest_dir* (lazily made).

    Drops non-image members (the ``.json`` masks). Skips broken images. Returns
    the number of images written (or that would be written in dry-run).
    """
    written = 0
    made_dir = False
    for member in tar:                      # streaming iteration (low memory)
        if not member.isfile():
            continue
        name = Path(member.name).name       # strip leading "./" / any dir
        if Path(name).suffix.lower() not in _IMG_EXTS:
            counters["skip_non_image"] += 1   # .json masks etc.
            continue
        counters["images_seen"] += 1
        fh = tar.extractfile(member)
        if fh is None:
            counters["skip_unreadable"] += 1
            continue
        raw = fh.read()
        if not _valid_image_bytes(raw):
            counters["skip_broken_image"] += 1
            continue
        if not dry_run:
            if not made_dir:
                dest_dir.mkdir(parents=True, exist_ok=True)
                made_dir = True
            (dest_dir / name).write_bytes(raw)
        written += 1
    return written


def _atomic_commit(tmp_dir: Path, final_dir: Path) -> None:
    """Publish *tmp_dir* as *final_dir* atomically (rename; cross-FS copy fallback)."""
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(tmp_dir, final_dir)
        return
    except OSError:
        pass
    staging = final_dir.with_name(final_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(tmp_dir, staging)
    os.rename(staging, final_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _sample_loads(final_dir: Path) -> bool:
    """Full-decode one committed image (not just header) — the pre-marker check."""
    imgs = [p for p in final_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS]
    if not imgs:
        return False
    try:
        from PIL import Image
        Image.open(imgs[0]).convert("RGB").load()
        return True
    except Exception:
        return False


def process_shard(shard: Path, final_dir: Path, tmp_root: Path, tag: str,
                  *, counters: Counter) -> int:
    """Convert one shard atomically: stage → verify → mark → commit. Returns images."""
    tmp_dir = tmp_root / f"{tag}{_INCOMING_SUFFIX}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(shard, "r") as tar:
            written = _extract_images(tar, tmp_dir, counters=counters, dry_run=False)
        if written == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return 0
        # Pre-commit verification: a sample image must fully decode (catches a
        # half-written / corrupt extraction before we mark the shard done).
        if not _sample_loads(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError("sample image failed to decode after extraction")
        _stamp_marker(tmp_dir, shard.name, tag, written)
        _atomic_commit(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    counters["images_written"] += written
    return written


def _clean_stale_staging(tmp_root: Path, out_root: Path) -> None:
    for p in list(tmp_root.glob(f"*{_INCOMING_SUFFIX}")):
        if p.is_dir():
            logger.info("  cleaning stale staging %s", p)
            shutil.rmtree(p, ignore_errors=True)
    for split in ("train", "val"):
        d = out_root / split
        if d.is_dir():
            for p in d.glob("*.staging"):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)


def discover_shards(shard_dir: Path, glob: str, limit: Optional[int]) -> List[Path]:
    """Return matching shards (possibly empty — e.g. all already converted+deleted)."""
    shards = sorted(shard_dir.glob(glob))
    return shards[:limit] if limit is not None else shards


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def run(shards: List[Path], out_root: Path, tmp_root: Path,
        args: argparse.Namespace, counters: Counter) -> None:
    tmp_root.mkdir(parents=True, exist_ok=True)
    _clean_stale_staging(tmp_root, out_root)
    total = len(shards)
    for i, shard in enumerate(shards, 1):
        tag = _shard_tag(shard)
        split = _split_for_shard(shard, args.val_every, args.all_train)
        final_dir = out_root / split / tag
        prog = f"[{i}/{total}]"

        if _is_shard_done(final_dir):
            counters["shards_skipped"] += 1
            logger.info("%s %-14s skip (already done, split=%s)", prog, shard.name, split)
            if args.delete_shard_on_success and shard.exists():
                shard.unlink(); counters["tars_deleted"] += 1
                logger.info("%s %-14s tar deleted (output already verified)", prog, shard.name)
            continue
        if final_dir.exists():                   # output dir exists but has no marker
            # A committed shard always carries its `.shard_done` marker (it is
            # stamped before the atomic rename), so an *unmarked* final dir is an
            # ambiguous leftover (legacy/external/partial). Don't destroy it by
            # default — skip and keep the tar. `--rebuild-unmarked` opts into wiping.
            if not args.rebuild_unmarked:
                counters["shards_unmarked"] += 1
                logger.warning("%s %-14s unmarked output exists — skipping (tar kept; "
                               "pass --rebuild-unmarked to wipe + rebuild)", prog, shard.name)
                continue
            logger.warning("%s %-14s unmarked output — wiping + rebuilding "
                           "(--rebuild-unmarked)", prog, shard.name)
            shutil.rmtree(final_dir, ignore_errors=True)

        try:
            written = process_shard(shard, final_dir, tmp_root, tag, counters=counters)
        except BaseException as exc:             # noqa: BLE001
            counters["shards_failed"] += 1
            logger.error("%s %-14s FAILED (%s) — tar kept", prog, shard.name, exc)
            continue
        if written == 0:
            counters["shards_empty"] += 1
            logger.warning("%s %-14s 0 images — tar kept", prog, shard.name)
            continue
        counters["shards_done"] += 1
        logger.info("%s %-14s → %6d images (committed, split=%s)", prog, shard.name, written, split)
        if args.delete_shard_on_success:
            shard.unlink(); counters["tars_deleted"] += 1
            logger.info("%s %-14s tar deleted (verified)", prog, shard.name)

    _clean_stale_staging(tmp_root, out_root)
    try:
        tmp_root.rmdir()
    except OSError:
        pass


def dry_run(shards: List[Path], out_root: Path, args: argparse.Namespace,
            counters: Counter) -> None:
    logger.info("[dry-run] no files written, NO tars deleted.")
    total = len(shards)
    for i, shard in enumerate(shards, 1):
        tag = _shard_tag(shard)
        split = _split_for_shard(shard, args.val_every, args.all_train)
        final_dir = out_root / split / tag
        prog = f"[{i}/{total}]"
        if _is_shard_done(final_dir):
            counters["shards_skipped"] += 1
            logger.info("%s %-14s would SKIP (done, split=%s)", prog, shard.name, split)
            if args.delete_shard_on_success:
                logger.info("%s %-14s would delete tar (verified)", prog, shard.name)
            continue
        if final_dir.exists():                   # unmarked leftover (see run())
            if not args.rebuild_unmarked:
                counters["shards_unmarked"] += 1
                logger.warning("%s %-14s would SKIP (unmarked output exists; "
                               "tar kept; needs --rebuild-unmarked)", prog, shard.name)
                continue
            logger.warning("%s %-14s would WIPE + rebuild (--rebuild-unmarked)", prog, shard.name)
        try:
            with tarfile.open(shard, "r") as tar:
                n = _extract_images(tar, final_dir, counters=counters, dry_run=True)
        except BaseException as exc:             # noqa: BLE001
            counters["shards_failed"] += 1
            logger.error("%s %-14s would FAIL (%s)", prog, shard.name, exc)
            continue
        counters["shards_done"] += 1
        counters["images_written"] += n
        logger.info("%s %-14s → %6d images (would commit, split=%s)", prog, shard.name, n, split)
        if args.delete_shard_on_success:
            logger.info("%s %-14s would delete tar after commit", prog, shard.name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert SA-1B tar shards → image-only training folders, shard-by-shard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--shard-dir", default="data/sa1b/raw",
                   help="Dir with the SA-1B sa_*.tar shards (relative → repo root).")
    p.add_argument("--output-dir", default="data/sa1b_images",
                   help="Output root; images land in <output-dir>/<split>/<shard_tag>/.")
    p.add_argument("--shard-glob", default="sa_*.tar", help="Shard glob.")
    p.add_argument("--limit-shards", type=int, default=None,
                   help="Process at most this many shards (resumable partial run).")
    p.add_argument("--val-every", type=int, default=10,
                   help="A shard sa_<N> goes to val iff N %% this == 0 (0 → all train).")
    p.add_argument("--all-train", action="store_true", help="Put every shard in train.")
    p.add_argument("--delete-shard-on-success", action="store_true",
                   help="DELETE each source .tar after its output is committed AND a "
                        "sample image decodes (verified). Failed shards keep their tar.")
    p.add_argument("--rebuild-unmarked", action="store_true",
                   help="If an output dir exists WITHOUT a completion marker, wipe and "
                        "rebuild it. Default: such a dir is left untouched and the shard "
                        "is skipped (tar kept) — a marker-less dir is ambiguous "
                        "(legacy/external/partial), so it is not destroyed automatically.")
    p.add_argument("--tmp-dir", default=None,
                   help="Staging dir (default <output-dir>/.sa1b_tmp; keep on the SAME "
                        "filesystem as the output for an atomic rename).")
    p.add_argument("--dry-run", action="store_true", help="Scan + count only; write/delete nothing.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    shard_dir = Path(args.shard_dir)
    if not shard_dir.is_absolute():
        shard_dir = (_REPO_ROOT / shard_dir).resolve()
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = (_REPO_ROOT / out_root).resolve()
    tmp_root = (Path(args.tmp_dir) if args.tmp_dir else out_root / ".sa1b_tmp")
    if not tmp_root.is_absolute():
        tmp_root = (_REPO_ROOT / tmp_root).resolve()

    shards = discover_shards(shard_dir, args.shard_glob, args.limit_shards)
    logger.info("shard_dir:   %s", shard_dir)
    logger.info("output_dir:  %s   (train/ + val/, image-only)", out_root)
    logger.info("split rule:  val iff shard_number %% %d == 0%s",
                args.val_every, "  (--all-train)" if args.all_train else "")
    logger.info("shards:      %d %s", len(shards), "[dry-run]" if args.dry_run else "")
    if not shards:
        # No tars left under the glob — typically "everything already converted &
        # deleted". This is a clean, idempotent no-op (not an error).
        logger.info("No shards matched %r under %s — nothing to do "
                    "(all converted+deleted?). Exiting.", args.shard_glob, shard_dir)
        return

    counters: Counter = Counter()
    if args.dry_run:
        dry_run(shards, out_root, args, counters)
    else:
        run(shards, out_root, tmp_root, args, counters)

    logger.info("──────── summary ────────")
    logger.info("shards converted:     %d", counters["shards_done"])
    logger.info("shards skipped (done):%d", counters["shards_skipped"])
    if counters["shards_unmarked"]:
        logger.info("shards skipped (unmarked, kept): %d  (use --rebuild-unmarked)",
                    counters["shards_unmarked"])
    logger.info("shards failed:        %d", counters["shards_failed"])
    if counters["shards_empty"]:
        logger.info("shards empty:         %d", counters["shards_empty"])
    logger.info("tars deleted:         %d", counters["tars_deleted"])
    logger.info("images written:       %d%s", counters["images_written"],
                "  [dry-run]" if args.dry_run else "")
    logger.info("skipped (masks/json): %d", counters["skip_non_image"])
    logger.info("skipped (broken img): %d", counters["skip_broken_image"])
    if not args.dry_run:
        logger.info("Output: %s/{train,val}/  →  use the split root as --train-list", out_root)


if __name__ == "__main__":
    main()
