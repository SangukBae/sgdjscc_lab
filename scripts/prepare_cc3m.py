#!/usr/bin/env python
"""prepare_cc3m.py – Convert CC3M WebDataset shards into training-ready pairs.

The repo's Stage 2/3 loaders (``data/datasets.py``, ``caption_source: sidecar``)
do **not** read WebDataset ``.tar`` shards directly — they expect a flat folder of
``<stem>.<img>`` images each with a same-stem ``<stem>.txt`` caption sidecar.
``pixparse/cc3m-wds`` already stores exactly that triple inside every shard
(``{key}.jpg`` + ``{key}.txt`` + ``{key}.json``), so this script only has to
**extract + de-collide**, not re-caption.

  cc3m-train-0000.tar          ->  cc3m_pairs/train/train0000/train0000_000000000.jpg
    000000000.jpg                                            + train0000_000000000.txt
    000000000.txt  (caption)
    000000000.json (metadata)

Why prefix the stem?  Keys (``000000000``…) **reset per shard** — every shard
contains ``000000000`` — so a naive extract would clobber pairs across shards.
We prefix each output stem with the shard tag (``train0000_``) so the
``(shard, key)`` pair is globally unique while the image/caption stems still match.

Why per-shard subdirs?  Full CC3M is ~2.9M pairs (~5.8M files). Piling those into
one flat directory stresses the filesystem (inode/lookup cost). By default each
shard's pairs go to ``<split>/<shard_tag>/`` so no directory holds more than one
shard (~5k pairs). The loader scans recursively (``_list_images`` → ``rglob``),
so nested dirs are read transparently. Use ``--flat`` for a handful of shards.

Modes
-----
* **convert** (default): refuses a non-empty split; writes the requested shards.
* **regenerate** (``--overwrite``): WIPES ``<output>/<split>/`` first, then writes
  — a smaller re-run shrinks the set and flat/per-shard layouts never mix under
  one split (the recursive loader would otherwise double-count them).
* **sequential append** (``--append`` and/or ``--delete-shard-on-success``): adds
  shards to a (possibly non-empty) split WITHOUT wiping, skipping already-done
  shards. Each shard is staged in a temp dir and **atomically committed** only
  after it fully succeeds, so an interrupted run never leaves a half-written
  shard in the dataset. With ``--delete-shard-on-success`` the source ``.tar`` is
  deleted **only after** its output is committed (success-verified) — the
  "append + verified-delete" flow that lets you unpack full CC3M while keeping
  disk usage bounded.

Design notes
------------
* Realistic flow: convert **a few shards first** (``--limit-shards``) before
  committing to the full ~575-shard train set.
* Non-destructive unless asked: the only destructive acts (``--overwrite`` wipe,
  ``--delete-shard-on-success`` tar removal) require their explicit flag; tar
  deletion happens strictly after a verified atomic commit.
* Durable: a crashed sequential run leaves only a staging dir OUTSIDE the split
  (``<output>/.cc3m_tmp_<split>/``); the next run cleans it and reprocesses the
  shard. Committed shards carry a ``.shard_done`` marker (invisible to the loader)
  used to skip them on re-run. An unmarked-but-populated shard dir (legacy /
  ``--overwrite`` output) is *adopted* — the marker is stamped and the shard is
  skipped, so existing datasets are append-safe without rework (``--rebuild-unmarked``
  forces a rebuild instead).
* Faithful dry-run: ``--dry-run`` in sequential mode mirrors the real run's
  per-shard decisions (skip done / adopt unmarked / build new), so capacity and
  resume planning match what an actual run would do.
* Robust: broken images, missing/empty captions, and unpaired members are
  skipped and counted, not fatal.

Usage
-----
# Dry-run: count what 2 train shards would yield (writes nothing)
python scripts/prepare_cc3m.py --split train --limit-shards 2 --dry-run

# Convert 4 train shards + 1 val shard (per-shard subdirs by default)
python scripts/prepare_cc3m.py --split train --limit-shards 4
python scripts/prepare_cc3m.py --split val   --limit-shards 1

# Cap total samples (handy for a quick Stage-2 smoke set)
python scripts/prepare_cc3m.py --split train --limit-shards 8 --max-samples 2000

# Regenerate / flat layout / custom paths
python scripts/prepare_cc3m.py \
    --shard-dir data/cc3m_wds --output-dir data/cc3m_pairs \
    --split train --shard-glob 'cc3m-train-00*.tar' --flat --overwrite

# Sequential append + free disk as you go (unpack full CC3M safely):
#   resumable — re-run the same command to continue where it left off.
python scripts/prepare_cc3m.py --split train --append --delete-shard-on-success
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Make src/ importable (parity with the other scripts; not strictly needed) ─
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_cc3m")

# Image suffixes a CC3M shard may carry (matches data/image_dataset._IMG_EXTS).
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Default shard globs per split. CC3M-WDS names val shards "cc3m-validation-*".
_SPLIT_GLOBS = {
    "train": "cc3m-train-*.tar",
    "val": "cc3m-validation-*.tar",
}

# Marker file dropped inside a committed shard subdir. Its presence means the
# shard was fully extracted AND atomically committed (see process_shard_sequential).
# A leading dot + no real extension → suffix is "" → the loader's _IMG_EXTS /
# ``.txt`` filters never pick it up, so it is invisible to training.
_SHARD_DONE_MARKER = ".shard_done"

# Suffix for the per-shard staging dir used while a shard is being written.
# Lives outside the split dir so a half-written shard is never visible to a
# loader pointed at <split>/ (see _resolve_tmp_root).
_INCOMING_SUFFIX = ".incoming"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shard_tag(shard: Path, split: str) -> str:
    """A short, collision-free per-shard prefix, e.g. ``train0000``.

    Derived from the trailing shard number in the filename so that the
    ``(shard, key)`` pair maps to a globally unique output stem. Falls back to
    the raw filename stem if no number is present.
    """
    m = re.search(r"(\d+)(?!.*\d)", shard.stem)
    num = m.group(1) if m else shard.stem
    return f"{split}{num}"


def _group_members(tar: tarfile.TarFile) -> Dict[str, Dict[str, tarfile.TarInfo]]:
    """Group a shard's members by key → {suffix: TarInfo}.

    A CC3M sample is the triple ``{key}.jpg`` / ``{key}.txt`` / ``{key}.json``.
    """
    groups: Dict[str, Dict[str, tarfile.TarInfo]] = {}
    for member in tar.getmembers():
        if not member.isfile():
            continue
        name = Path(member.name)
        key = name.stem
        suffix = name.suffix.lower()
        groups.setdefault(key, {})[suffix] = member
    return groups


def _pick_image(members: Dict[str, tarfile.TarInfo]) -> Optional[Tuple[str, tarfile.TarInfo]]:
    """Return (suffix, TarInfo) for the first image member, or None."""
    for suffix, info in members.items():
        if suffix in _IMG_EXTS:
            return suffix, info
    return None


def _read_caption(
    tar: tarfile.TarFile,
    members: Dict[str, tarfile.TarInfo],
) -> Optional[str]:
    """Caption text from the ``.txt`` sidecar, falling back to ``.json:caption``.

    Returns None when no non-empty caption can be found.
    """
    txt = members.get(".txt")
    if txt is not None:
        fh = tar.extractfile(txt)
        if fh is not None:
            cap = fh.read().decode("utf-8", errors="replace").strip()
            if cap:
                return cap
    # Fallback: the metadata json carries the same caption string.
    js = members.get(".json")
    if js is not None:
        fh = tar.extractfile(js)
        if fh is not None:
            try:
                cap = str(json.loads(fh.read().decode("utf-8")).get("caption", "")).strip()
                if cap:
                    return cap
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
    return None


def _valid_image_bytes(raw: bytes) -> bool:
    """True if *raw* decodes as a non-truncated image (PIL); permissive if no PIL."""
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - PIL is a hard dep here in practice
        return True  # cannot validate → trust the shard
    try:
        Image.open(io.BytesIO(raw)).verify()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction (shared by legacy + sequential paths)
# ─────────────────────────────────────────────────────────────────────────────

def _write_pairs_from_tar(
    tar: tarfile.TarFile,
    dest_dir: Path,
    tag: str,
    *,
    remaining: Optional[int],
    counters: Counter,
    dry_run: bool,
) -> Tuple[int, bool]:
    """Extract a shard's image+caption pairs from *tar* into *dest_dir*.

    Writes ``<tag>_<key>.<img>`` + ``<tag>_<key>.txt`` for each valid sample.
    The dest dir is created lazily on first write (so empty shards leave no dir).
    Returns ``(written, truncated)`` where *truncated* is True if extraction
    stopped early because the ``remaining`` budget was hit (i.e. the shard was
    NOT fully consumed — important for the delete-on-success safety check).
    Broken/unpaired samples are skipped and counted, never fatal.
    """
    written = 0
    truncated = False
    made_dir = False
    groups = _group_members(tar)
    for key in sorted(groups):
        if remaining is not None and written >= remaining:
            truncated = True
            break
        members = groups[key]
        counters["samples_seen"] += 1

        picked = _pick_image(members)
        if picked is None:
            counters["skip_no_image"] += 1
            continue
        img_suffix, img_info = picked

        caption = _read_caption(tar, members)
        if caption is None:
            counters["skip_no_caption"] += 1
            continue

        fh = tar.extractfile(img_info)
        if fh is None:
            counters["skip_no_image"] += 1
            continue
        raw = fh.read()
        if not _valid_image_bytes(raw):
            counters["skip_broken_image"] += 1
            continue

        stem = f"{tag}_{key}"
        if not dry_run:
            if not made_dir:
                dest_dir.mkdir(parents=True, exist_ok=True)   # lazily, only if used
                made_dir = True
            (dest_dir / f"{stem}{img_suffix}").write_bytes(raw)
            (dest_dir / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
        written += 1

    return written, truncated


# ─────────────────────────────────────────────────────────────────────────────
# Legacy conversion (default + --overwrite): direct write, no atomic commit
# ─────────────────────────────────────────────────────────────────────────────

def convert_shard(
    shard: Path,
    out_dir: Path,
    split: str,
    *,
    dry_run: bool,
    flat: bool,
    remaining: Optional[int],
    counters: Counter,
) -> int:
    """Extract one shard into ``out_dir`` as image + ``.txt`` pairs (legacy path).

    Returns the number of pairs written (or that *would* be written in dry-run).
    Honours ``remaining`` (global ``--max-samples`` budget) and updates *counters*.

    Layout
    ------
    * ``flat=False`` (default): pairs go to ``out_dir/<shard_tag>/`` — one
      subdirectory per shard so no single directory holds millions of files
      (inode/filesystem pressure at full CC3M scale). The recursive loader
      (``_list_images`` → ``rglob``) reads nested dirs transparently, and stems
      already carry the shard tag so they stay globally unique.
    * ``flat=True``: pairs go directly into ``out_dir`` (simplest for a handful
      of shards / smoke sets).
    """
    tag = _shard_tag(shard, split)
    target = out_dir if flat else out_dir / tag
    try:
        tar = tarfile.open(shard, "r")
    except (tarfile.TarError, OSError) as exc:
        logger.warning("  ! cannot open shard %s (%s) — skipped", shard.name, exc)
        counters["shards_broken"] += 1
        return 0

    with tar:
        written, _ = _write_pairs_from_tar(
            tar, target, tag, remaining=remaining, counters=counters, dry_run=dry_run)

    counters["pairs_written"] += written
    counters["shards_done"] += 1
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Sequential conversion (--append / --delete-shard-on-success): atomic per shard
# ─────────────────────────────────────────────────────────────────────────────

def _is_shard_done(final_dir: Path) -> bool:
    """True iff *final_dir* holds a committed shard (completion marker present)."""
    return (final_dir / _SHARD_DONE_MARKER).is_file()


def _count_images(d: Path) -> int:
    """Number of image files directly in *d* (the marker/.txt sidecars excluded)."""
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir()
               if p.is_file() and p.suffix.lower() in _IMG_EXTS)


def _stamp_marker(final_dir: Path, shard_name: str, tag: str, pairs: int,
                  *, adopted: bool = False) -> None:
    """Write the ``.shard_done`` completion marker into an already-built dir."""
    (final_dir / _SHARD_DONE_MARKER).write_text(
        json.dumps({"shard": shard_name, "tag": tag, "pairs": pairs,
                    "adopted": adopted}) + "\n",
        encoding="utf-8",
    )


def _classify_existing(final_dir: Path, *, rebuild_unmarked: bool) -> str:
    """Decide how the sequential path treats an existing *final_dir*.

    Returns one of:
      ``"done"``    — completion marker present → skip (already converted).
      ``"adopt"``   — unmarked but holds pairs (legacy/``--overwrite`` output, or a
                      prior tool version): treat as complete, stamp a marker, skip.
                      Atomic commits never leave an unmarked *non-empty* final dir,
                      so this can only be a complete build from another path.
      ``"rebuild"`` — unmarked + ``--rebuild-unmarked`` set, OR an empty/partial
                      leftover dir → wipe and rebuild.
      ``"new"``     — nothing there yet → build.
    """
    if _is_shard_done(final_dir):
        return "done"
    if final_dir.exists():
        if _count_images(final_dir) > 0:
            return "rebuild" if rebuild_unmarked else "adopt"
        return "rebuild"        # empty / partial leftover
    return "new"


def _atomic_commit(tmp_dir: Path, final_dir: Path) -> None:
    """Atomically publish *tmp_dir* as *final_dir* (which must not yet exist).

    Same-filesystem fast path: a single ``os.rename`` (atomic). Cross-filesystem
    fallback (when ``--tmp-dir`` is on another device): copy into a sibling
    ``*.staging`` dir that lives on the destination FS, then atomically rename
    that into place. Either way ``final_dir`` only ever appears fully formed.
    """
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(tmp_dir, final_dir)
        return
    except OSError:
        pass  # likely EXDEV (cross-device) — fall back to copy-then-swap
    staging = final_dir.with_name(final_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(tmp_dir, staging)
    os.rename(staging, final_dir)            # sibling of final → same FS → atomic
    shutil.rmtree(tmp_dir, ignore_errors=True)


def process_shard_sequential(
    shard: Path,
    final_dir: Path,
    tmp_root: Path,
    tag: str,
    *,
    counters: Counter,
) -> int:
    """Convert one shard atomically: stage → mark → commit. Returns pairs written.

    The shard is extracted into a staging dir under *tmp_root*; only after the
    whole shard succeeds is a completion marker written and the staging dir
    atomically renamed to *final_dir*. On any exception the staging dir is
    removed (no partial output under the split) and the error propagates so the
    caller can keep the source tar. A shard yielding 0 usable pairs commits
    nothing and returns 0.
    """
    tmp_dir = tmp_root / f"{tag}{_INCOMING_SUFFIX}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)                # clear any stale leftover
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(shard, "r") as tar:
            written, _ = _write_pairs_from_tar(
                tar, tmp_dir, tag, remaining=None, counters=counters, dry_run=False)
        if written == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return 0
        # Marker is written LAST, inside the staging dir, so it rides along in
        # the atomic rename — its presence in final_dir == "fully committed".
        _stamp_marker(tmp_dir, shard.name, tag, written)
        _atomic_commit(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    counters["pairs_written"] += written
    return written


def _clean_stale_staging(tmp_root: Path, out_dir: Path) -> None:
    """Remove leftover staging dirs from a previously interrupted run.

    These are partial outputs that were never committed, so deleting them is
    safe and keeps the next run consistent (the shard is simply reprocessed).
    """
    for p in tmp_root.glob(f"*{_INCOMING_SUFFIX}"):
        if p.is_dir():
            logger.info("  cleaning stale staging dir %s", p)
            shutil.rmtree(p, ignore_errors=True)
    for p in out_dir.glob("*.staging"):          # cross-FS fallback leftovers
        if p.is_dir():
            logger.info("  cleaning stale staging dir %s", p)
            shutil.rmtree(p, ignore_errors=True)


def run_sequential(
    shards: List[Path],
    out_dir: Path,
    tmp_root: Path,
    split: str,
    args: argparse.Namespace,
    counters: Counter,
) -> None:
    """Per-shard atomic append loop with optional success-verified tar deletion.

    For each shard: skip if already committed (marker present); else stage +
    atomically commit; on success optionally delete the source tar. Failures
    keep the tar and continue (the run is durable: one bad shard never corrupts
    the dataset or aborts the rest).
    """
    tmp_root.mkdir(parents=True, exist_ok=True)
    _clean_stale_staging(tmp_root, out_dir)

    for shard in shards:
        tag = _shard_tag(shard, split)
        final_dir = out_dir / tag
        status = _classify_existing(final_dir, rebuild_unmarked=args.rebuild_unmarked)

        # ── Output already present? (append guard against re-processing) ─────
        if status in ("done", "adopt"):
            if args.fail_on_existing:
                sys.exit(
                    f"--fail-on-existing: output for {shard.name} already exists at "
                    f"{final_dir}. Remove it or drop --fail-on-existing to skip it."
                )
            if status == "adopt":
                # Unmarked legacy/--overwrite output: migrate it in place by
                # stamping a marker so future runs recognise it as done (rather
                # than rebuilding good data). Counts the pairs already on disk.
                n = _count_images(final_dir)
                _stamp_marker(final_dir, shard.name, tag, n, adopted=True)
                logger.info("  %-28s adopted existing output (stamped marker, "
                            "%d pairs)", shard.name, n)
            counters["shards_skipped"] += 1
            logger.info("  %-28s skip (already converted)", shard.name)
            # The output is verified-complete, so freeing its tar is safe.
            if args.delete_shard_on_success and shard.exists():
                shard.unlink()
                counters["tars_deleted"] += 1
                logger.info("  %-28s tar deleted (output already verified)", shard.name)
            continue

        # ── Empty / partial leftover, or --rebuild-unmarked: wipe & rebuild ──
        if status == "rebuild" and final_dir.exists():
            logger.warning("  %-28s unmarked/partial output — rebuilding", shard.name)
            shutil.rmtree(final_dir)

        # ── Stage + atomic commit ────────────────────────────────────────────
        try:
            written = process_shard_sequential(
                shard, final_dir, tmp_root, tag, counters=counters)
        except BaseException as exc:                 # noqa: BLE001 - report + continue
            counters["shards_failed"] += 1
            logger.error("  %-28s FAILED (%s) — tar kept", shard.name, exc)
            continue

        if written == 0:
            counters["shards_empty"] += 1
            logger.warning("  %-28s 0 usable pairs — nothing committed, tar kept",
                           shard.name)
            continue

        counters["shards_done"] += 1
        logger.info("  %-28s → %6d pairs (committed)", shard.name, written)

        # ── Success-verified deletion (only reached AFTER a committed shard) ──
        if args.delete_shard_on_success:
            shard.unlink()
            counters["tars_deleted"] += 1
            logger.info("  %-28s tar deleted", shard.name)

    # Best-effort tidy of the (now-empty) staging root.
    _clean_stale_staging(tmp_root, out_dir)
    try:
        tmp_root.rmdir()
    except OSError:
        pass


def _dry_run_sequential(
    shards: List[Path],
    out_dir: Path,
    split: str,
    args: argparse.Namespace,
    counters: Counter,
) -> None:
    """Preview a sequential run WITHOUT writing or deleting anything.

    Mirrors :func:`run_sequential`'s per-shard decisions (skip already-done /
    adopt unmarked / build new) so capacity & resume planning from a dry-run
    matches the real run — unlike a blind legacy re-count of every shard.
    """
    logger.info("[dry-run] sequential mode: no files written, NO tars deleted.")
    for shard in shards:
        tag = _shard_tag(shard, split)
        final_dir = out_dir / tag
        status = _classify_existing(final_dir, rebuild_unmarked=args.rebuild_unmarked)

        if status in ("done", "adopt"):
            counters["shards_skipped"] += 1
            why = "already converted" if status == "done" else \
                  f"adopt existing unmarked output ({_count_images(final_dir)} pairs)"
            logger.info("  %-28s would SKIP (%s)", shard.name, why)
            if args.delete_shard_on_success and shard.exists():
                logger.info("  %-28s would delete tar (output verified)", shard.name)
            continue

        # new / rebuild → would extract: count pairs by scanning the tar.
        action = "rebuild" if status == "rebuild" else "commit"
        try:
            with tarfile.open(shard, "r") as tar:
                n, _ = _write_pairs_from_tar(
                    tar, final_dir, tag, remaining=None, counters=counters, dry_run=True)
        except BaseException as exc:                 # noqa: BLE001 - report + continue
            counters["shards_failed"] += 1
            logger.error("  %-28s would FAIL to read (%s) — tar kept", shard.name, exc)
            continue
        counters["shards_done"] += 1
        counters["pairs_written"] += n
        logger.info("  %-28s → %6d pairs (would %s)", shard.name, n, action)
        if args.delete_shard_on_success:
            logger.info("  %-28s would delete tar after commit", shard.name)


def discover_shards(shard_dir: Path, glob: str, limit: Optional[int]) -> List[Path]:
    shards = sorted(shard_dir.glob(glob))
    if not shards:
        raise FileNotFoundError(
            f"No shards matched {glob!r} under {shard_dir}. "
            f"Check --shard-dir / --shard-glob / --split."
        )
    if limit is not None:
        shards = shards[:limit]
    return shards


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CC3M WebDataset shards → image+.txt training pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--shard-dir", default="data/cc3m_wds",
                   help="Directory holding the cc3m-*.tar shards "
                        "(relative paths resolve against the repo root).")
    p.add_argument("--output-dir", default="data/cc3m_pairs",
                   help="Root output dir; pairs land in <output-dir>/<split>/ "
                        "(relative paths resolve against the repo root).")
    p.add_argument("--split", choices=["train", "val"], default="train",
                   help="Which split to build (chooses default shard glob + subdir).")
    p.add_argument("--shard-glob", default=None,
                   help="Override the shard glob (default derives from --split).")
    p.add_argument("--limit-shards", type=int, default=None,
                   help="Process at most this many shards (realistic partial flow).")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Stop after writing this many pairs total (across shards).")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate a non-empty output split: the <split> dir is "
                        "CLEARED before writing (so a smaller --max-samples shrinks "
                        "the set and flat/per-shard layouts never mix). Without it, "
                        "a non-empty split is refused.")
    p.add_argument("--flat", action="store_true",
                   help="Write all pairs flat into <output-dir>/<split>/ instead of "
                        "the default per-shard subdirs (<split>/<shard_tag>/). Flat is "
                        "fine for a few shards; AVOID for full CC3M — millions of files "
                        "in one directory hurt inode/filesystem performance. "
                        "Incompatible with --append / --delete-shard-on-success.")
    p.add_argument("--append", action="store_true",
                   help="Sequential append mode: add new shards to a (possibly "
                        "non-empty) split WITHOUT wiping it, skipping shards already "
                        "converted. Each shard is staged then atomically committed, so "
                        "an interrupted run never leaves a half-written shard in the "
                        "dataset. Use with --delete-shard-on-success to free disk as "
                        "you go.")
    p.add_argument("--delete-shard-on-success", action="store_true",
                   help="DELETE each source .tar after its output is fully extracted "
                        "AND atomically committed (success-verified). Enables the "
                        "sequential atomic path (like --append). Destructive: only the "
                        "verified-converted shard's tar is removed; a failed shard's "
                        "tar is kept.")
    p.add_argument("--fail-on-existing", action="store_true",
                   help="In sequential mode, abort if a shard's output already exists "
                        "instead of skipping it (strict re-run detection).")
    p.add_argument("--rebuild-unmarked", action="store_true",
                   help="In sequential mode, REBUILD a shard whose output dir exists "
                        "but lacks a completion marker (legacy/--overwrite output) "
                        "instead of adopting it. Default: adopt (stamp a marker and "
                        "skip) so existing datasets are append-safe without rework.")
    p.add_argument("--tmp-dir", default=None,
                   help="Staging dir for per-shard atomic commits (sequential mode). "
                        "Default: a hidden sibling of the split dir "
                        "(<output-dir>/.cc3m_tmp_<split>) so partial shards are never "
                        "visible to a loader pointed at <split>/. Keep it on the SAME "
                        "filesystem as --output-dir for an atomic rename (a cross-FS "
                        "tmp-dir still commits safely, just via an extra copy).")
    p.add_argument("--dry-run", action="store_true",
                   help="Scan + count only; write nothing and delete nothing.")
    return p.parse_args()


def _resolve_tmp_root(args: argparse.Namespace, out_root: Path, split: str) -> Path:
    """Staging root for sequential commits.

    Defaults to a hidden sibling of the split dir (``<out_root>/.cc3m_tmp_<split>``)
    so it is (a) on the same filesystem as the split (atomic rename) and (b) NOT
    under ``<split>/`` — a loader pointed at the split never sees in-flight shards.
    """
    if args.tmp_dir:
        tmp = Path(args.tmp_dir)
        if not tmp.is_absolute():
            tmp = (_REPO_ROOT / tmp).resolve()
        return tmp
    return out_root / f".cc3m_tmp_{split}"


def main() -> None:
    args = _parse_args()

    # Sequential atomic path is triggered by either append or delete-on-success.
    sequential = args.append or args.delete_shard_on_success

    # ── Mutually-exclusive / unsafe flag combinations (fail BEFORE any work) ──
    if args.append and args.overwrite:
        sys.exit("Error: --append (add shards) and --overwrite (regenerate) are "
                 "mutually exclusive — pick one.")
    if sequential and args.flat:
        sys.exit("Error: --append / --delete-shard-on-success require the per-shard "
                 "subdir layout (atomic commit + per-shard skip work on subdirs). "
                 "Drop --flat.")
    if sequential and args.max_samples is not None:
        sys.exit("Error: --max-samples truncates shards mid-way, which is unsafe with "
                 "per-shard commit/delete (a partial shard would be marked done / its "
                 "tar deleted). Bound a sequential run with --limit-shards instead.")
    if args.fail_on_existing and not sequential:
        sys.exit("Error: --fail-on-existing only applies in sequential mode "
                 "(--append / --delete-shard-on-success).")
    if args.rebuild_unmarked and not sequential:
        sys.exit("Error: --rebuild-unmarked only applies in sequential mode "
                 "(--append / --delete-shard-on-success).")

    # Resolve paths relative to the repo root (parity with the config loader,
    # which resolves data paths relative to the project, not the CWD).
    shard_dir = Path(args.shard_dir)
    if not shard_dir.is_absolute():
        shard_dir = (_REPO_ROOT / shard_dir).resolve()
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = (_REPO_ROOT / out_root).resolve()
    out_dir = out_root / args.split

    glob = args.shard_glob or _SPLIT_GLOBS[args.split]
    shards = discover_shards(shard_dir, glob, args.limit_shards)

    if sequential:
        mode = "sequential append" + (" + delete-on-success"
                                      if args.delete_shard_on_success else "")
    elif args.overwrite:
        mode = "regenerate (--overwrite)"
    else:
        mode = "convert"
    logger.info("shard_dir:   %s", shard_dir)
    logger.info("output_dir:  %s", out_dir)
    logger.info("split:       %s   glob: %s", args.split, glob)
    logger.info("mode:        %s", mode)
    logger.info("layout:      %s", "flat (<split>/)" if args.flat
                else "per-shard subdirs (<split>/<shard_tag>/)")
    logger.info("shards:      %d (of glob match) %s",
                len(shards), "[dry-run]" if args.dry_run else "")

    counters: Counter = Counter()

    # ─────────────────────────────────────────────────────────────────────────
    # DRY-RUN: always count-only via the legacy path (no writes, no deletes),
    # regardless of mode, so sequential flags can be previewed safely.
    # ─────────────────────────────────────────────────────────────────────────
    if args.dry_run:
        if sequential:
            _dry_run_sequential(shards, out_dir, args.split, args, counters)
        else:
            for shard in shards:
                n = convert_shard(shard, out_dir, args.split, dry_run=True,
                                  flat=args.flat, remaining=None, counters=counters)
                logger.info("  %-28s → %6d pairs", shard.name, n)
        _print_summary(args, counters, out_dir, sequential=sequential)
        return

    # ── Output dir guard ─────────────────────────────────────────────────────
    nonempty = out_dir.exists() and any(out_dir.iterdir())
    if args.overwrite:
        # Regenerate: WIPE first so a smaller re-run shrinks the set and layouts
        # never mix under one split (recursive loader would double-count).
        if nonempty:
            logger.info("--overwrite: clearing existing split dir %s", out_dir)
            shutil.rmtree(out_dir)
    elif sequential:
        # Append: a non-empty split is expected. A non-empty split WITHOUT
        # --append (delete-only) is refused — that combo would silently mutate
        # an existing dataset's tars without an explicit "add" intent.
        if nonempty and not args.append:
            sys.exit(
                f"Error: output dir {out_dir} is not empty. Add --append to extend it "
                f"in place (sequential mode), or --overwrite to regenerate it."
            )
    else:
        # Default convert: refuse a non-empty split (offer the explicit modes).
        if nonempty:
            sys.exit(
                f"Error: output dir {out_dir} is not empty. Use --append to add shards "
                f"in place, or --overwrite to regenerate it, or pick another --output-dir."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Convert ──────────────────────────────────────────────────────────────
    if sequential:
        tmp_root = _resolve_tmp_root(args, out_root, args.split)
        logger.info("staging dir: %s", tmp_root)
        run_sequential(shards, out_dir, tmp_root, args.split, args, counters)
    else:
        remaining = args.max_samples
        for shard in shards:
            if remaining is not None and remaining <= 0:
                logger.info("Reached --max-samples budget; stopping.")
                break
            n = convert_shard(
                shard, out_dir, args.split,
                dry_run=False, flat=args.flat,
                remaining=remaining, counters=counters,
            )
            if remaining is not None:
                remaining -= n
            logger.info("  %-28s → %6d pairs", shard.name, n)

    _print_summary(args, counters, out_dir, sequential=sequential)

    # Surface partial failure to the caller (CI / scripts) without discarding
    # the shards that did succeed.
    if counters["shards_failed"]:
        sys.exit(f"Completed with {counters['shards_failed']} failed shard(s) "
                 f"(their tars were kept). See log above.")


def _print_summary(args: argparse.Namespace, counters: Counter, out_dir: Path,
                   *, sequential: bool) -> None:
    logger.info("──────── summary ────────")
    logger.info("shards processed:     %d", counters["shards_done"])
    if sequential:
        logger.info("shards skipped (done):%d", counters["shards_skipped"])
        logger.info("shards failed:        %d", counters["shards_failed"])
        if counters["shards_empty"]:
            logger.info("shards empty (0 pair):%d", counters["shards_empty"])
        logger.info("tars deleted:         %d", counters["tars_deleted"])
    if counters["shards_broken"]:
        logger.info("shards unreadable:    %d", counters["shards_broken"])
    logger.info("samples seen:         %d", counters["samples_seen"])
    logger.info("pairs written:        %d%s",
                counters["pairs_written"], "  [dry-run: not on disk]" if args.dry_run else "")
    logger.info("skipped (no image):   %d", counters["skip_no_image"])
    logger.info("skipped (no caption): %d", counters["skip_no_caption"])
    logger.info("skipped (broken img): %d", counters["skip_broken_image"])
    if not args.dry_run:
        logger.info("Output: %s/  →  use as --train-list / --val-list", out_dir)


if __name__ == "__main__":
    main()
