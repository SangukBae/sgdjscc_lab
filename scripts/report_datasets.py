#!/usr/bin/env python
"""report_datasets.py - Generate a machine-local dataset inventory report.

This solves the "data/README.md does not travel via git" problem by separating:

* tracked, machine-agnostic dataset docs -> docs/dataset_status.md
* machine-specific inventory -> generated markdown under ignored data/_reports/

The output is intentionally lightweight and operational: existence, size, and a
few shallow signals for known dataset layouts. It avoids expensive full recursive
file counting on giant datasets unless the signal is cheap and meaningful.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


KNOWN = {
    "imagenet": ("image-only", "Stage 1 / image stages"),
    "coco": ("text-image", "COCO images + annotations/captions"),
    "journey_pairs": ("text-image pairs", "jpg/txt sidecar pairs"),
    "cc3m_wds": ("raw shards", "convert with prepare_cc3m.py"),
    "cc3m_pairs": ("text-image pairs", "converted CC3M output"),
    "sa1b": ("raw shards", "convert with prepare_sa1b.py"),
    "sa1b_images": ("image-only", "converted SA-1B output"),
    "celeba": ("image-only", "text stages need generated captions"),
    "celeba_hq": ("image or captioned image", "depends on local preparation"),
    "datacomp_pairs": ("text-image pairs", "pair-format training set"),
    "datacomp_small": ("raw or intermediate", "local prep staging"),
    "kodak": ("evaluation", "small image benchmark"),
    "ade20k": ("evaluation", "segmentation benchmark"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a machine-local dataset inventory markdown report."
    )
    p.add_argument(
        "--data-root",
        default="data",
        help="Repo-relative or absolute data root to scan.",
    )
    p.add_argument(
        "--output",
        default="data/_reports/dataset_status.md",
        help="Markdown output path (repo-relative or absolute).",
    )
    return p.parse_args()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


def _human_bytes(n: int) -> str:
    units = ["B", "K", "M", "G", "T", "P"]
    size = float(n)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{n}B"


def _disk_usage(path: Path) -> int:
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True)
        return int(out.split()[0])
    except Exception:
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
        return total


def _count(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern))


def _signal(path: Path) -> str:
    name = path.name
    if name == "sa1b":
        raw = path / "raw"
        imgs = path.parent / "sa1b_images"
        parts = []
        if raw.is_dir():
            parts.append(f"raw_tars={_count(raw, 'sa_*.tar')}")
        if imgs.is_dir():
            parts.append(f"done_markers={sum(1 for _ in imgs.rglob('.shard_done'))}")
        return ", ".join(parts) or "-"
    if name == "sa1b_images":
        train = path / "train"
        val = path / "val"
        return (
            f"train_shards={sum(1 for _ in train.glob('*') if _.is_dir()) if train.is_dir() else 0}, "
            f"val_shards={sum(1 for _ in val.glob('*') if _.is_dir()) if val.is_dir() else 0}, "
            f"done_markers={sum(1 for _ in path.rglob('.shard_done'))}"
        )
    if name == "cc3m_wds":
        return (
            f"train_tars={_count(path, 'cc3m-train-*.tar')}, "
            f"val_tars={_count(path, 'cc3m-validation-*.tar')}"
        )
    if name == "cc3m_pairs":
        train = path / "train"
        val = path / "val"
        return (
            f"train_shards={sum(1 for _ in train.glob('*') if _.is_dir()) if train.is_dir() else 0}, "
            f"val_shards={sum(1 for _ in val.glob('*') if _.is_dir()) if val.is_dir() else 0}"
        )
    if name == "coco":
        parts = []
        for d in ("train2017", "val2017", "test2017"):
            pp = path / d
            if pp.is_dir():
                parts.append(f"{d}={sum(1 for _ in pp.iterdir() if _.is_file())}")
        ann = path / "annotations"
        if ann.is_dir():
            parts.append("annotations=yes")
        return ", ".join(parts) or "-"
    top = [p.name for p in sorted(path.iterdir())[:6]]
    return ", ".join(top) if top else "-"


def _rows(data_root: Path) -> list[tuple[str, str, str, str, str]]:
    rows = []
    for path in sorted(p for p in data_root.iterdir() if not p.name.startswith(".")):
        role, notes = KNOWN.get(path.name, ("local data", "unclassified"))
        size = _human_bytes(_disk_usage(path))
        rows.append((path.name, role, size, _signal(path) if path.is_dir() else "-", notes))
    return rows


def _render(data_root: Path, rows: list[tuple[str, str, str, str, str]]) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Dataset Inventory",
        "",
        f"Generated: `{now}`",
        f"Data root: `{data_root}`",
        "",
        "Machine-local inventory generated by `scripts/report_datasets.py`.",
        "Canonical dataset roles and stage mapping live in `docs/dataset_status.md`.",
        "",
        "| Path | Role | Size | Signal | Notes |",
        "|---|---|---:|---|---|",
    ]
    for path, role, size, signal, notes in rows:
        lines.append(f"| `{path}` | {role} | {size} | {signal} | {notes} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    data_root = _resolve(args.data_root)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not data_root.is_dir():
        raise SystemExit(f"Data root not found: {data_root}")
    rows = _rows(data_root)
    output.write_text(_render(data_root, rows), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
