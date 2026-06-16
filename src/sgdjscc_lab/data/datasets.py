"""data/datasets.py – Stage-aware datasets for the SGD-JSCC training stages.

Dataset types, one (or more) per training stage (see ``training/stages.py``):

  ``image``            -> :class:`ImageOnlyDataset`      (stage 1 / JSCC)
  ``text_image``       -> :class:`TextImageDataset`      (stage 2 / text DM)
  ``text_image_edge``  -> :class:`TextImageEdgeDataset`  (stage 3 / ControlNet)
  ``edge``             -> :class:`EdgeOnlyDataset`       (edge_codec)

Every item is a ``dict`` so the collate function is uniform and new fields can be
added without breaking call sites:

  {"image": Tensor[3,H,W], "path": str,
   "caption": str,                # text_image, text_image_edge
   "edge":  Tensor[1,H,W]}        # text_image_edge

All three share the common :class:`~sgdjscc_lab.data.transforms.ImageTransform`
so the paper's "center-crop + resize 128×128" preprocessing is applied once.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from sgdjscc_lab.data.image_dataset import _IMG_EXTS, _list_images
from sgdjscc_lab.data.transforms import ImageTransform, build_transform
from sgdjscc_lab.training.stages import (
    STAGE_CONTROLNET,
    VALID_CAPTION_SOURCES,
    VALID_EDGE_SOURCES,
    StageConfigError,
    resolve_dataset_type,
    resolve_stage,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class _BaseImageDataset(Dataset):
    """Shared image loading + transform for the stage datasets."""

    def __init__(self, input_path, transform: Optional[ImageTransform] = None,
                 files: Optional[List[Path]] = None) -> None:
        # ``files`` (file-list mode) takes precedence over a recursive folder scan,
        # so large datasets can be driven by an explicit path list. Folder mode
        # (files=None) keeps the original recursive ``_list_images`` behaviour.
        self.files: List[Path] = list(files) if files is not None else _list_images(input_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def _load_image(self, fpath: Path) -> torch.Tensor:
        from sgdjscc_lab.io import load_image_as_tensor
        img = load_image_as_tensor(fpath)[0]   # [3, H, W] in [0, 1]
        if self.transform is not None:
            img = self.transform(img)
        return img


class ImageOnlyDataset(_BaseImageDataset):
    """Stage 1 (JSCC): image-only dataset.

    Returns ``{"image": Tensor[3,H,W], "path": str}``.
    """

    def __getitem__(self, idx: int) -> Dict:
        fpath = self.files[idx]
        return {"image": self._load_image(fpath), "path": str(fpath)}


# ─────────────────────────────────────────────────────────────────────────────
# Caption resolution
# ─────────────────────────────────────────────────────────────────────────────

class _CaptionResolver:
    """Resolve a caption for an image path from one of several sources.

    sources
    -------
    ``sidecar``        read ``<image_stem>.txt`` next to the image.
    ``manifest``       look up filename (or stem) in a JSON dict / CSV file
                       (single caption per image).
    ``filename``       derive a pseudo-caption from the filename stem (cheap;
                       mainly for smoke tests — *not* a paper-grade source).
    ``coco_json``      read a COCO ``captions_*.json`` (MULTIPLE captions per
                       image, joined via ``image_id`` → ``file_name``).
    ``multi_manifest`` read a JSON ``{filename: [caption, ...]}`` (a string value
                       becomes a 1-item list) — a generic multi-caption manifest.

    Multi-caption sources (``coco_json`` / ``multi_manifest``) pick ONE caption
    per access via ``select`` ∈ {``first``, ``longest``, ``random``}.  Single
    sources ignore ``select``.  Backward-compatible: ``sidecar`` / ``manifest`` /
    ``filename`` behave exactly as before.
    """

    def __init__(
        self,
        source: str,
        manifest_path: Optional[str] = None,
        fallback: str = "",
        select: str = "first",
        seed: Optional[int] = None,
    ) -> None:
        source = str(source).lower()
        if source not in VALID_CAPTION_SOURCES:
            raise StageConfigError(
                f"caption_source must be one of {VALID_CAPTION_SOURCES}, got {source!r}"
            )
        self.source = source
        self.fallback = fallback
        self.select = str(select).lower()
        self._manifest: Dict[str, str] = {}        # single-caption sources
        self._multi: Dict[str, List[str]] = {}     # multi-caption sources
        self._rng = random.Random(seed)
        if source == "manifest":
            if not manifest_path:
                raise StageConfigError("caption_source='manifest' needs caption_path")
            self._manifest = self._load_manifest(Path(manifest_path))
        elif source == "coco_json":
            if not manifest_path:
                raise StageConfigError(
                    "caption_source='coco_json' needs caption_path (a COCO captions_*.json)")
            self._multi = self._load_coco_json(Path(manifest_path))
        elif source == "multi_manifest":
            if not manifest_path:
                raise StageConfigError(
                    "caption_source='multi_manifest' needs caption_path "
                    "(a JSON {filename: [caption, ...]})")
            self._multi = self._load_multi_manifest(Path(manifest_path))

    @staticmethod
    def _load_manifest(path: Path) -> Dict[str, str]:
        if not path.exists():
            raise FileNotFoundError(f"Caption manifest not found: {path}")
        out: Dict[str, str] = {}
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in data.items():
                out[Path(str(k)).name] = str(v)
                out[Path(str(k)).stem] = str(v)
        else:  # csv: filename,caption
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.reader(fh):
                    if len(row) >= 2:
                        out[Path(row[0]).name] = row[1]
                        out[Path(row[0]).stem] = row[1]
        return out

    @staticmethod
    def _load_coco_json(path: Path) -> Dict[str, List[str]]:
        """Parse a COCO ``captions_*.json`` → ``{file_name|stem: [captions]}``.

        COCO stores ``images:[{id,file_name}]`` and ``annotations:[{image_id,
        caption}]``; we join them so a caption list is looked up by image filename.
        """
        if not path.exists():
            raise FileNotFoundError(f"COCO caption JSON not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        id2name = {img["id"]: img["file_name"] for img in data.get("images", [])}
        out: Dict[str, List[str]] = {}
        for ann in data.get("annotations", []):
            name = id2name.get(ann.get("image_id"))
            if name is None:
                continue
            cap = str(ann.get("caption", "")).strip()
            if not cap:
                continue
            for key in (name, Path(name).stem):
                out.setdefault(key, []).append(cap)
        return out

    @staticmethod
    def _load_multi_manifest(path: Path) -> Dict[str, List[str]]:
        """Parse a JSON ``{filename: [caption, ...]}`` (string value → 1-item list)."""
        if not path.exists():
            raise FileNotFoundError(f"Multi-caption manifest not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[str, List[str]] = {}
        for k, v in data.items():
            caps = [str(v)] if isinstance(v, str) else [str(x) for x in v]
            caps = [c.strip() for c in caps if str(c).strip()]
            if not caps:
                continue
            out[Path(str(k)).name] = caps
            out[Path(str(k)).stem] = caps
        return out

    def _select_caption(self, caps: List[str]) -> str:
        if not caps:
            return self.fallback
        if self.select == "longest":
            return max(caps, key=len)
        if self.select == "random":
            return self._rng.choice(caps)
        return caps[0]                              # "first" (default)

    def __call__(self, fpath: Path) -> str:
        if self.source == "sidecar":
            txt = fpath.with_suffix(".txt")
            if txt.exists():
                return txt.read_text(encoding="utf-8").strip() or self.fallback
            return self.fallback
        if self.source == "manifest":
            return self._manifest.get(
                fpath.name, self._manifest.get(fpath.stem, self.fallback)
            )
        if self.source in ("coco_json", "multi_manifest"):
            caps = self._multi.get(fpath.name) or self._multi.get(fpath.stem)
            return self._select_caption(caps) if caps else self.fallback
        # filename
        return fpath.stem.replace("_", " ").replace("-", " ").strip() or self.fallback


class TextImageDataset(_BaseImageDataset):
    """Stage 2 (text DM): text-image pair dataset.

    Returns ``{"image": Tensor[3,H,W], "caption": str, "path": str}``.
    """

    def __init__(
        self,
        input_path,
        caption_resolver: _CaptionResolver,
        transform: Optional[ImageTransform] = None,
        files: Optional[List[Path]] = None,
    ) -> None:
        super().__init__(input_path, transform, files=files)
        self.caption_resolver = caption_resolver

    def __getitem__(self, idx: int) -> Dict:
        fpath = self.files[idx]
        return {
            "image": self._load_image(fpath),
            "caption": self.caption_resolver(fpath),
            "path": str(fpath),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Edge resolution
# ─────────────────────────────────────────────────────────────────────────────

def _canny_edge(img: torch.Tensor) -> torch.Tensor:
    """Compute a single-channel edge map ``[1,H,W]`` in ``[0,1]`` from a CHW image.

    Prefers OpenCV's Canny detector; falls back to a torch Sobel gradient
    magnitude when cv2 is unavailable (e.g. minimal test environments).
    """
    try:
        import cv2
        import numpy as np

        gray = (img.mean(dim=0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        edges = cv2.Canny(gray, 100, 200)
        return torch.from_numpy(edges.astype("float32") / 255.0).unsqueeze(0)
    except Exception:  # pragma: no cover - exercised only without cv2
        # Sobel fallback (dependency-light, deterministic).
        import torch.nn.functional as F

        gray = img.mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.t()
        kx = kx.view(1, 1, 3, 3)
        ky = ky.view(1, 1, 3, 3)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2)
        mag = mag / (mag.amax() + 1e-8)
        return mag.squeeze(0)


def _load_edge_map(
    fpath: Path,
    img: torch.Tensor,
    edge_source: str,
    edge_dir: Optional[Path],
    transform: Optional[ImageTransform],
) -> torch.Tensor:
    """Resolve a single-channel edge map ``[1,H,W]`` for *fpath*.

    Shared by :class:`TextImageEdgeDataset` (stage 3) and
    :class:`EdgeOnlyDataset` (edge_codec).  ``canny`` computes the edge on the
    fly from *img*; ``sidecar`` reads a precomputed map from *edge_dir* (or
    ``<stem>_edge.png`` next to the image).
    """
    if edge_source == "canny":
        return _canny_edge(img)
    candidates = []
    if edge_dir is not None:
        # Support both "<stem>.<ext>" and "<stem>_edge.<ext>" inside edge_dir
        # (the latter matches the "<stem>_edge.png" convention used next to images
        # and in the docs / make_tiny_dataset.py --edges output).
        for ext in _IMG_EXTS:
            candidates.append(edge_dir / f"{fpath.stem}{ext}")
            candidates.append(edge_dir / f"{fpath.stem}_edge{ext}")
    candidates.append(fpath.with_name(f"{fpath.stem}_edge.png"))
    for cand in candidates:
        if cand.exists():
            from sgdjscc_lab.io import load_image_as_tensor
            edge = load_image_as_tensor(cand)[0]
            if transform is not None:
                edge = transform(edge)
            return edge.mean(dim=0, keepdim=True)  # → single channel
    raise FileNotFoundError(
        f"edge_source='sidecar' but no edge map found for {fpath.name} "
        f"(looked in edge_dir={edge_dir} and <stem>_edge.png)."
    )


class TextImageEdgeDataset(TextImageDataset):
    """Stage 3 (ControlNet): text-image-edge tuple dataset.

    Returns ``{"image", "caption", "edge": Tensor[1,H,W], "path"}``.

    Edge source
    -----------
    ``canny``   compute a Canny edge map from the (transformed) image on the fly.
    ``sidecar`` read a precomputed edge map matching the image filename from
                ``edge_dir`` (or ``<image>_edge.png`` next to the image).
    """

    def __init__(
        self,
        input_path,
        caption_resolver: _CaptionResolver,
        edge_source: str = "canny",
        edge_dir: Optional[str] = None,
        transform: Optional[ImageTransform] = None,
        files: Optional[List[Path]] = None,
    ) -> None:
        super().__init__(input_path, caption_resolver, transform, files=files)
        edge_source = str(edge_source).lower()
        if edge_source not in VALID_EDGE_SOURCES:
            raise StageConfigError(
                f"edge_source must be one of {VALID_EDGE_SOURCES}, got {edge_source!r}"
            )
        self.edge_source = edge_source
        self.edge_dir = Path(edge_dir) if edge_dir else None

    def _load_edge(self, fpath: Path, img: torch.Tensor) -> torch.Tensor:
        return _load_edge_map(fpath, img, self.edge_source, self.edge_dir, self.transform)

    def __getitem__(self, idx: int) -> Dict:
        fpath = self.files[idx]
        img = self._load_image(fpath)
        return {
            "image": img,
            "caption": self.caption_resolver(fpath),
            "edge": self._load_edge(fpath, img),
            "path": str(fpath),
        }


class EdgeOnlyDataset(_BaseImageDataset):
    """edge_codec stage: edge-map-only dataset (no captions).

    Returns ``{"edge": Tensor[1,H,W], "path": str}``.  The edge map is the
    codec's input *and* its reconstruction target (self-supervised BCE+Dice).
    Edge source matches :class:`TextImageEdgeDataset` (``canny`` | ``sidecar``).
    """

    def __init__(
        self,
        input_path,
        edge_source: str = "canny",
        edge_dir: Optional[str] = None,
        transform: Optional[ImageTransform] = None,
        files: Optional[List[Path]] = None,
    ) -> None:
        super().__init__(input_path, transform, files=files)
        edge_source = str(edge_source).lower()
        if edge_source not in VALID_EDGE_SOURCES:
            raise StageConfigError(
                f"edge_source must be one of {VALID_EDGE_SOURCES}, got {edge_source!r}"
            )
        self.edge_source = edge_source
        self.edge_dir = Path(edge_dir) if edge_dir else None

    def __getitem__(self, idx: int) -> Dict:
        fpath = self.files[idx]
        img = self._load_image(fpath)
        edge = _load_edge_map(fpath, img, self.edge_source, self.edge_dir, self.transform)
        return {"edge": edge, "path": str(fpath)}


# ─────────────────────────────────────────────────────────────────────────────
# Collate + builders
# ─────────────────────────────────────────────────────────────────────────────

def collate_stage_batch(items: List[Dict]) -> Dict:
    """Collate a list of stage-item dicts into a batched dict.

    Tensors are stacked; strings are collected into lists.
    """
    out: Dict = {}
    keys = items[0].keys()
    for key in keys:
        vals = [it[key] for it in items]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = list(vals)
    return out


def _make_caption_resolver(cfg, *, training: bool = True) -> _CaptionResolver:
    src = OmegaConf.select(cfg, "train.dataset.caption_source", default=None)
    manifest = OmegaConf.select(cfg, "train.dataset.caption_path", default=None)
    # Manifest-style sources (manifest / coco_json / multi_manifest) usually need a
    # DIFFERENT file for val (e.g. captions_train2017.json vs captions_val2017.json).
    # The val loader uses ``val_caption_path`` when set, else falls back to the train
    # path. (sidecar/filename ignore caption_path, so this is a no-op for them.)
    if not training:
        val_manifest = OmegaConf.select(cfg, "train.dataset.val_caption_path", default=None)
        if val_manifest is not None:
            manifest = val_manifest
    fallback = OmegaConf.select(cfg, "train.dataset.fallback_caption", default="")
    select = str(OmegaConf.select(cfg, "train.dataset.caption_select", default="first")).lower()
    # Reproducible validation: a "random" multi-caption pick is non-deterministic,
    # so for the val loader we fall back to "first" (stable captions for eval).
    if not training and select == "random":
        select = "first"
    seed = OmegaConf.select(cfg, "train.seed", default=None)
    return _CaptionResolver(src, manifest_path=manifest, fallback=str(fallback),
                            select=select, seed=seed)


def _resolve_input_files(cfg, *, training: bool) -> Optional[List[Path]]:
    """Return an explicit file list (file-list mode) or None (folder mode).

    ``train.dataset.input_mode: file_list`` reads image paths from
    ``file_list_path`` (training) / ``val_file_list_path`` (validation), one path
    per line (``#`` comments and blank lines ignored). Relative entries resolve
    against the list file's directory; absolute entries are used as-is. Folder
    mode (the default) returns None → the datasets keep their recursive
    ``_list_images`` scan (backward-compatible).
    """
    mode = str(OmegaConf.select(cfg, "train.dataset.input_mode", default="folder")).lower()
    if mode != "file_list":
        return None
    key = "train.dataset.file_list_path" if training else "train.dataset.val_file_list_path"
    lp = OmegaConf.select(cfg, key, default=None)
    if not lp:
        raise StageConfigError(
            f"input_mode='file_list' requires {key} "
            f"({'training' if training else 'validation'} image-path list)."
        )
    p = Path(lp)
    if not p.exists():
        raise FileNotFoundError(f"file list not found: {p}")
    base = p.parent
    files: List[Path] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fp = Path(line)
        files.append(fp.resolve() if fp.is_absolute() else (base / fp).resolve())
    if not files:
        raise FileNotFoundError(f"file list {p} contains no image paths.")
    return files


def build_dataset_for_stage(
    input_path: str,
    cfg,
    *,
    training: bool = True,
    stage: Optional[str] = None,
) -> Dataset:
    """Build the dataset matching the active stage / ``train.dataset.type``.

    Selection (via :func:`~sgdjscc_lab.training.stages.resolve_dataset_type`):
      ``image`` → ImageOnlyDataset, ``text_image`` → TextImageDataset,
      ``text_image_edge`` → TextImageEdgeDataset.
    """
    if stage is None:
        stage = resolve_stage(cfg)
    ds_type = resolve_dataset_type(cfg, stage)
    transform = build_transform(cfg, training=training)
    files = _resolve_input_files(cfg, training=training)   # None in folder mode

    if ds_type == "image":
        return ImageOnlyDataset(input_path, transform=transform, files=files)

    if ds_type == "edge":
        edge_source = OmegaConf.select(cfg, "train.dataset.edge_source", default="canny")
        edge_dir = OmegaConf.select(cfg, "train.dataset.edge_dir", default=None)
        return EdgeOnlyDataset(
            input_path, edge_source=str(edge_source), edge_dir=edge_dir,
            transform=transform, files=files,
        )

    resolver = _make_caption_resolver(cfg, training=training)

    if ds_type == "text_image":
        return TextImageDataset(input_path, resolver, transform=transform, files=files)

    # text_image_edge
    edge_source = OmegaConf.select(cfg, "train.dataset.edge_source", default="canny")
    edge_dir = OmegaConf.select(cfg, "train.dataset.edge_dir", default=None)
    return TextImageEdgeDataset(
        input_path, resolver,
        edge_source=str(edge_source), edge_dir=edge_dir, transform=transform,
        files=files,
    )


def build_dataloader_for_stage(
    input_path: str,
    cfg,
    *,
    shuffle: bool = True,
    training: bool = True,
    stage: Optional[str] = None,
) -> DataLoader:
    """Build a stage-appropriate dataset + DataLoader from a training config."""
    ds = build_dataset_for_stage(input_path, cfg, training=training, stage=stage)
    batch_size = int(OmegaConf.select(cfg, "train.batch_size", default=4))
    num_workers = int(OmegaConf.select(cfg, "train.num_workers", default=2))
    drop_last = shuffle and len(ds) >= batch_size
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=collate_stage_batch,
    )
