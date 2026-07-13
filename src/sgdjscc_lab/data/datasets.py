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


def muge_reduce(data: torch.Tensor) -> torch.Tensor:
    """Reduce MuGE multi-channel soft-edge output to a single ``[1,H,W]`` map.

    MuGE's ``generate_canny`` returns ``[N,11,H,W]`` soft-edge data; the training
    edge condition is single-channel, so we take the channel-mean intensity and
    min-max normalise to ``[0,1]``. This 11→1 reduction is **paper-like** (the
    inference pipeline consumes the full multi-channel MuGE output; the training
    ControlNet condition here is reduced). Documented in docs/paper_gap_closure.md.
    """
    if data.dim() == 4:
        data = data[0]
    if data.dim() == 3 and data.shape[0] > 1:
        m = data.float().mean(dim=0, keepdim=True)
    else:
        m = data.float().reshape(1, *data.shape[-2:])
    mn, mx = m.amin(), m.amax()
    return ((m - mn) / (mx - mn + 1e-8)).clamp(0, 1)


# MuGE edge representations for the training edge condition (see
# docs/paper_gap_closure.md "Stage-3 edge path alignment"):
#   reduced          1ch mean edge — LEGACY default; matches the inference edge
#                    CHANNEL COUNT (inference also means 11→1) but DROPS uncertainty.
#   edge_uncertainty 2ch [mean-edge, mean-uncertainty] — the inference path carries
#                    BOTH (canny_transmission_net input is edge+uncertainty), so this
#                    is the CLOSEST-to-inference representation (recommended).
#   multi            11ch all MuGE edge channels (per-channel min-max) — preserves
#                    the full map; OPT-IN. Inference collapses these, so multi is
#                    information-preserving but NOT "more inference-aligned".
_MUGE_REPRS = ("reduced", "edge_uncertainty", "multi")
_MUGE_REPR_CHANNELS = {"reduced": 1, "edge_uncertainty": 2, "multi": 11}


def muge_repr_channels(repr_name: str) -> int:
    """Edge channel count for a MuGE representation (1 | 2 | 11)."""
    r = str(repr_name).lower()
    if r not in _MUGE_REPR_CHANNELS:
        raise StageConfigError(
            f"train.dataset.muge_repr must be one of {_MUGE_REPRS}, got {repr_name!r}")
    return _MUGE_REPR_CHANNELS[r]


def muge_channels(data: torch.Tensor, uncertainty: Optional[torch.Tensor],
                  repr_name: str = "reduced") -> torch.Tensor:
    """MuGE output → the chosen training edge representation ``[C,H,W]`` in [0,1].

    See ``_MUGE_REPRS``. ``reduced`` → 1ch; ``edge_uncertainty`` → 2ch (edge +
    uncertainty, the closest match to what the inference path carries); ``multi``
    → 11ch (full map, opt-in).
    """
    r = str(repr_name).lower()
    edge = muge_reduce(data)                                   # [1,H,W]
    if r == "reduced":
        return edge
    if r == "edge_uncertainty":
        unc = muge_reduce(uncertainty) if uncertainty is not None else torch.zeros_like(edge)
        return torch.cat([edge, unc], dim=0)                  # [2,H,W]
    if r == "multi":
        d = data[0] if data.dim() == 4 else data              # [11,H,W]
        d = d.float()
        flat = d.reshape(d.shape[0], -1)
        mn = flat.amin(dim=1, keepdim=True)
        mx = flat.amax(dim=1, keepdim=True)
        return ((flat - mn) / (mx - mn + 1e-8)).reshape_as(d).clamp(0, 1)  # [11,H,W]
    raise StageConfigError(
        f"train.dataset.muge_repr must be one of {_MUGE_REPRS}, got {repr_name!r}")


class LazyMugeExtractor:
    """Lazily build + cache a MuGE :class:`EdgeExtractor` for ``muge_runtime``.

    Reuses ``guidance/edge_extractor.build_edge_extractor`` (the same network the
    *inference* path uses) so training and inference share the MuGE structure.
    Runs on CPU by default (safe inside DataLoader workers). For speed/large
    datasets, **precompute** edges with ``scripts/prepare_muge_edges.py`` and use
    ``edge_source: muge_sidecar`` instead — runtime MuGE in every worker is slow.
    """

    def __init__(self, model_root, device: str = "cpu",
                 repr_name: str = "reduced") -> None:
        self.model_root = model_root
        self.device = torch.device(device)
        self.repr_name = str(repr_name).lower()
        self._ex = None

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self._ex is None:
            from sgdjscc_lab.guidance.edge_extractor import build_edge_extractor
            self._ex = build_edge_extractor(self.model_root, self.device)
        data, unc = self._ex.extract(img.unsqueeze(0).to(self.device), self.device)
        return muge_channels(data, unc, self.repr_name).to(img.device)


def _resize_edge_tensor(edge: torch.Tensor, target_hw) -> torch.Tensor:
    """Resize a C-channel edge tensor to ``target_hw`` with bilinear sampling."""
    if tuple(edge.shape[-2:]) == tuple(target_hw):
        return edge
    import torch.nn.functional as F
    return F.interpolate(
        edge.unsqueeze(0), size=tuple(target_hw),
        mode="bilinear", align_corners=False,
    ).squeeze(0)


def _load_edge_map(
    fpath: Path,
    img: torch.Tensor,
    edge_source: str,
    edge_dir: Optional[Path],
    transform: Optional[ImageTransform],
    muge_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    muge_repr: str = "reduced",
) -> torch.Tensor:
    """Resolve an edge tensor ``[C,H,W]`` for *fpath*.

    Shared by :class:`TextImageEdgeDataset` (stage 3) and
    :class:`EdgeOnlyDataset` (edge_codec). Sources:

    * ``canny``        compute a Canny edge on the fly from *img* (paper-like).
    * ``sidecar``      read a generic precomputed map ``<stem>_edge.<ext>``.
    * ``muge_sidecar`` read a precomputed MuGE soft edge ``<stem>_muge.<ext>``
                       (``.png`` reduced 1ch, or ``.npy`` multi-channel repr).
    * ``muge_runtime`` run *muge_extractor* on *img* (reuses the inference MuGE).
    """
    if edge_source == "canny":
        return _canny_edge(img)
    if edge_source == "muge_runtime":
        if muge_extractor is None:
            raise StageConfigError(
                "edge_source='muge_runtime' needs a MuGE extractor but none was "
                "provided (model_root missing?). Either set model_root so the "
                "MuGE checkpoint can be loaded, or precompute edges with "
                "scripts/prepare_muge_edges.py and use edge_source='muge_sidecar'."
            )
        return muge_extractor(img)

    # File-based sources: sidecar (<stem>_edge) | muge_sidecar (<stem>_muge).
    suffix = "_muge" if edge_source == "muge_sidecar" else "_edge"
    # muge_sidecar may carry a MULTI-channel representation saved as .npy
    # (edge_uncertainty=2ch / multi=11ch) — prefer it; it is returned as-is
    # (NOT mean-collapsed), so multi-channel MuGE conditioning flows to the codec.
    muge_repr = str(muge_repr).lower()
    if edge_source == "muge_sidecar":
        npy_cands = []
        if edge_dir is not None:
            npy_cands.append(edge_dir / f"{fpath.stem}_muge.npy")
        npy_cands.append(fpath.with_name(f"{fpath.stem}_muge.npy"))
        for cand in npy_cands:
            if cand.exists():
                import numpy as _np
                arr = _np.load(cand)
                t = torch.from_numpy(arr).float()
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                return _resize_edge_tensor(t, img.shape[-2:])  # [C,H,W] in [0,1]
        if muge_repr != "reduced":
            raise FileNotFoundError(
                "edge_source='muge_sidecar' with "
                f"train.dataset.muge_repr={muge_repr!r} requires a multi-channel "
                f"NumPy sidecar <stem>_muge.npy for {fpath.name}, but none was "
                "found. Re-run scripts/prepare_muge_edges.py with "
                f"--repr {muge_repr}."
            )
    candidates = []
    if edge_dir is not None:
        for ext in _IMG_EXTS:
            candidates.append(edge_dir / f"{fpath.stem}{ext}")
            candidates.append(edge_dir / f"{fpath.stem}{suffix}{ext}")
    candidates.append(fpath.with_name(f"{fpath.stem}{suffix}.png"))
    for cand in candidates:
        if cand.exists():
            from sgdjscc_lab.io import load_image_as_tensor
            edge = load_image_as_tensor(cand)[0]
            if transform is not None:
                edge = transform(edge)
            return edge.mean(dim=0, keepdim=True)  # → single channel
    raise FileNotFoundError(
        f"edge_source={edge_source!r} but no edge map found for {fpath.name} "
        f"(looked in edge_dir={edge_dir} and <stem>{suffix}.png). "
        + ("Precompute MuGE edges with scripts/prepare_muge_edges.py."
           if edge_source == "muge_sidecar" else "")
    )


class TextImageEdgeDataset(TextImageDataset):
    """Stage 3 (ControlNet): text-image-edge tuple dataset.

    Returns ``{"image", "caption", "edge": Tensor[C,H,W], "path"}``.

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
        muge_repr: str = "reduced",
        transform: Optional[ImageTransform] = None,
        files: Optional[List[Path]] = None,
        muge_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(input_path, caption_resolver, transform, files=files)
        edge_source = str(edge_source).lower()
        if edge_source not in VALID_EDGE_SOURCES:
            raise StageConfigError(
                f"edge_source must be one of {VALID_EDGE_SOURCES}, got {edge_source!r}"
            )
        self.edge_source = edge_source
        self.edge_dir = Path(edge_dir) if edge_dir else None
        self.muge_repr = str(muge_repr).lower()
        self.muge_extractor = muge_extractor

    def _load_edge(self, fpath: Path, img: torch.Tensor) -> torch.Tensor:
        return _load_edge_map(fpath, img, self.edge_source, self.edge_dir,
                              self.transform, self.muge_extractor, self.muge_repr)

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

    Returns ``{"edge": Tensor[C,H,W], "path": str}``.  The edge map is the
    codec's input *and* its reconstruction target (self-supervised BCE+Dice).
    Edge source matches :class:`TextImageEdgeDataset` (``canny`` | ``sidecar`` |
    ``muge_sidecar`` | ``muge_runtime``).
    """

    def __init__(
        self,
        input_path,
        edge_source: str = "canny",
        edge_dir: Optional[str] = None,
        muge_repr: str = "reduced",
        transform: Optional[ImageTransform] = None,
        files: Optional[List[Path]] = None,
        muge_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(input_path, transform, files=files)
        edge_source = str(edge_source).lower()
        if edge_source not in VALID_EDGE_SOURCES:
            raise StageConfigError(
                f"edge_source must be one of {VALID_EDGE_SOURCES}, got {edge_source!r}"
            )
        self.edge_source = edge_source
        self.edge_dir = Path(edge_dir) if edge_dir else None
        self.muge_repr = str(muge_repr).lower()
        self.muge_extractor = muge_extractor

    def __getitem__(self, idx: int) -> Dict:
        fpath = self.files[idx]
        img = self._load_image(fpath)
        edge = _load_edge_map(fpath, img, self.edge_source, self.edge_dir,
                              self.transform, self.muge_extractor, self.muge_repr)
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
    root_cache: Dict[str, Optional[Path]] = {}
    files: List[Path] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        files.append(_resolve_list_entry(line, base, root_cache))
    if not files:
        raise FileNotFoundError(f"file list {p} contains no image paths.")
    return files


def _resolve_list_entry(
    entry: str,
    base: Path,
    root_cache: Dict[str, Optional[Path]],
) -> Path:
    """Resolve one file-list entry.

    Relative paths are primarily interpreted relative to the list file,
    preserving the original file-list contract.  For multi-part paths, the first
    segment is resolved once against the list directory and its parents, then
    cached.  This supports generated lists that store repo-root-relative entries
    such as ``data/...`` while the list itself lives under ``data/_lists/...``.
    """
    fp = Path(entry)
    if fp.is_absolute():
        return fp.resolve()

    parts = fp.parts
    if len(parts) <= 1:
        return (base / fp).resolve()

    first = parts[0]
    if first not in root_cache:
        root_cache[first] = None
        for root in (base, *base.parents):
            if (root / first).exists():
                root_cache[first] = root
                break

    root = root_cache[first] or base
    return (root / fp).resolve()


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
        muge_repr = OmegaConf.select(cfg, "train.dataset.muge_repr", default="reduced")
        return EdgeOnlyDataset(
            input_path, edge_source=str(edge_source), edge_dir=edge_dir,
            muge_repr=str(muge_repr),
            transform=transform, files=files,
            muge_extractor=_maybe_muge_extractor(cfg, str(edge_source)),
        )

    resolver = _make_caption_resolver(cfg, training=training)

    if ds_type == "text_image":
        return TextImageDataset(input_path, resolver, transform=transform, files=files)

    # text_image_edge
    edge_source = OmegaConf.select(cfg, "train.dataset.edge_source", default="canny")
    edge_dir = OmegaConf.select(cfg, "train.dataset.edge_dir", default=None)
    muge_repr = OmegaConf.select(cfg, "train.dataset.muge_repr", default="reduced")
    return TextImageEdgeDataset(
        input_path, resolver,
        edge_source=str(edge_source), edge_dir=edge_dir, muge_repr=str(muge_repr),
        transform=transform,
        files=files,
        muge_extractor=_maybe_muge_extractor(cfg, str(edge_source)),
    )


def _maybe_muge_extractor(cfg, edge_source: str):
    """Build a :class:`LazyMugeExtractor` when edge_source is ``muge_runtime``.

    Resolves ``model_root`` (where ``muge-epoch-19-checkpoint.pth`` lives). Runs
    on CPU so it is safe inside DataLoader workers (precompute via
    scripts/prepare_muge_edges.py + ``muge_sidecar`` is faster for large data).
    """
    if str(edge_source).lower() != "muge_runtime":
        return None
    model_root = OmegaConf.select(cfg, "model_root", default="../checkpoints")
    repr_name = str(OmegaConf.select(cfg, "train.dataset.muge_repr", default="reduced"))
    logger.warning(
        "edge_source='muge_runtime' (repr=%s): MuGE runs in DataLoader workers on "
        "CPU (slow). For large datasets precompute edges with "
        "scripts/prepare_muge_edges.py and use edge_source='muge_sidecar'.", repr_name)
    return LazyMugeExtractor(model_root, device="cpu", repr_name=repr_name)


def build_dataloader_for_stage(
    input_path: str,
    cfg,
    *,
    shuffle: bool = True,
    training: bool = True,
    stage: Optional[str] = None,
) -> DataLoader:
    """Build a stage-appropriate dataset + DataLoader from a training config.

    DDP-aware: under ``torchrun`` (distributed) the loader uses a
    :class:`~torch.utils.data.distributed.DistributedSampler` so each rank sees a
    disjoint shard (the sampler owns shuffling; ``set_epoch`` is driven by the
    train pipeline). Single-process runs keep the original ``shuffle=`` path
    (byte-for-byte unchanged). ``batch_size`` is PER-RANK (see docs).
    """
    from sgdjscc_lab import distributed as ddp

    ds = build_dataset_for_stage(input_path, cfg, training=training, stage=stage)
    batch_size = int(OmegaConf.select(cfg, "train.batch_size", default=4))
    num_workers = int(OmegaConf.select(cfg, "train.num_workers", default=2))
    drop_last = shuffle and len(ds) >= batch_size

    sampler = None
    if ddp.is_distributed():
        from torch.utils.data import DistributedSampler
        # train: drop_last so every rank gets equal-size batches; val: keep all
        # samples (a small padding duplication is corrected by the sum+count
        # all-reduce of the metric — see distributed.reduce_metric_sums).
        sampler = DistributedSampler(
            ds, shuffle=shuffle, drop_last=(drop_last if training else False))

    return DataLoader(
        ds,
        batch_size=batch_size,
        # With a sampler, DataLoader requires shuffle=False (sampler shuffles).
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(drop_last if sampler is None else False),
        collate_fn=collate_stage_batch,
    )
