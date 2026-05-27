"""io.py – Image file discovery, loading, and saving utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image as tv_save_image

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_image_files(path: str | Path) -> List[Path]:
    """Return a sorted list of image file paths.

    *path* may be:
    - A single image file  → returns ``[path]``
    - A directory          → returns all images found inside (non-recursive)
    """
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in _IMAGE_EXTENSIONS:
            return [p]
        raise ValueError(f"File {p} does not look like a supported image ({_IMAGE_EXTENSIONS}).")

    if p.is_dir():
        files = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not files:
            raise FileNotFoundError(f"No images found in directory: {p}")
        return files

    raise FileNotFoundError(f"Path does not exist: {p}")


def load_image_as_tensor(path: str | Path) -> torch.Tensor:
    """Load an image file and return a ``[1, 3, H, W]`` float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img)        # [3, H, W]  in [0, 1]
    return tensor.unsqueeze(0)                 # [1, 3, H, W]


def save_tensor_as_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a ``[1, 3, H, W]`` or ``[3, H, W]`` float tensor (values in [0,1]) as PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # tv_save_image clamps to [0,1] and writes PNG/JPEG based on extension
    tv_save_image(tensor.cpu().float(), str(path))
    logger.info("Saved → %s", path)
