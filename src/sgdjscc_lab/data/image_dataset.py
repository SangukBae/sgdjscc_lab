"""data/image_dataset.py – Folder-based image dataset for training.

Designed for easy extension toward semantic-packet and channel-simulation
training.  The base class loads raw images; subclass or compose transforms
to add patch tiling, augmentation, or paired-data generation.

Usage
-----
>>> ds = ImageFolderDataset("/data/kodak/train", patch_size=128)
>>> loader = build_dataloader(ds, batch_size=4, num_workers=2)
>>> for batch, fpaths in loader:
...     pass   # batch: [B, 3, 128, 128]  fpaths: list[str]
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# Image extensions accepted by load_image_as_tensor.
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images(root: str | Path) -> List[Path]:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found under {root}")
    return files


class ImageFolderDataset(Dataset):
    """Flat-folder image dataset that returns (tensor, filepath) pairs.

    Parameters
    ----------
    input_path:
        Root directory to search recursively for images.
    patch_size:
        If set, each image is randomly cropped (or tiled) to ``patch_size ×
        patch_size`` before being returned.  None → return full image (padded
        to multiples of 128 by default so patch-based models work).
    transform:
        Optional callable applied to each image tensor ``[3, H, W]`` after
        loading and before returning.  Use this hook to add augmentation or
        normalization without subclassing.
    require_multiples_of:
        When *patch_size* is None, resize so H and W are multiples of this
        value (default 128, matching the SGDJSCC patch grid).  Set to 1 to
        disable.
    """

    def __init__(
        self,
        input_path: str | Path,
        patch_size: Optional[int] = 128,
        transform: Optional[Callable] = None,
        require_multiples_of: int = 128,
    ) -> None:
        self.files = _list_images(input_path)
        self.patch_size = patch_size
        self.transform = transform
        self.require_multiples_of = require_multiples_of

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        fpath = self.files[idx]
        img = self._load(fpath)       # [3, H, W]  float32 in [0, 1]

        if self.patch_size is not None:
            img = self._random_crop(img, self.patch_size)
        elif self.require_multiples_of > 1:
            img = self._pad_to_multiple(img, self.require_multiples_of)

        if self.transform is not None:
            img = self.transform(img)

        return img, str(fpath)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load(fpath: Path) -> torch.Tensor:
        from sgdjscc_lab.io import load_image_as_tensor
        return load_image_as_tensor(fpath)[0]   # drop the batch dim → [3, H, W]

    @staticmethod
    def _random_crop(img: torch.Tensor, size: int) -> torch.Tensor:
        _, H, W = img.shape
        if H < size or W < size:
            # Pad then crop if the image is smaller than the requested patch.
            pad_h = max(0, size - H)
            pad_w = max(0, size - W)
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
            _, H, W = img.shape
        top  = random.randint(0, H - size)
        left = random.randint(0, W - size)
        return img[:, top:top + size, left:left + size]

    @staticmethod
    def _pad_to_multiple(img: torch.Tensor, m: int) -> torch.Tensor:
        _, H, W = img.shape
        ph = (m - H % m) % m
        pw = (m - W % m) % m
        if ph or pw:
            img = torch.nn.functional.pad(img, (0, pw, 0, ph), mode="reflect")
        return img


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Wrap *dataset* in a DataLoader with sensible training defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )


def build_dataloader_from_cfg(input_path: str, cfg, shuffle: bool = True) -> DataLoader:
    """Build an ImageFolderDataset + DataLoader from a training config node.

    Parameters
    ----------
    input_path:
        Resolved absolute path to the image directory.
    cfg:
        OmegaConf DictConfig with ``train.*`` fields (batch_size, num_workers, etc.)
    shuffle:
        True for the training loader, False for validation.
    """
    from omegaconf import OmegaConf

    train_cfg  = OmegaConf.select(cfg, "train", default={}) or {}
    batch_size  = int(OmegaConf.select(cfg, "train.batch_size",  default=4))
    num_workers = int(OmegaConf.select(cfg, "train.num_workers", default=2))
    patch_size  = OmegaConf.select(cfg, "train.patch_size", default=128)
    patch_size  = int(patch_size) if patch_size is not None else None

    ds = ImageFolderDataset(input_path, patch_size=patch_size)
    # drop_last only makes sense when the dataset is larger than one batch;
    # disable it so small datasets (dry-run, quick test) still yield batches.
    drop_last = shuffle and len(ds) >= batch_size
    return build_dataloader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
    )
