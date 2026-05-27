"""utils/preprocessing.py – Image resize and 128×128 patch split/merge.

All heavy algorithms (split_image_v2, merge_image_v2, CropLongSide) are
imported from the original SGDJSCC package to preserve exact behaviour.

Patch-explosion warning  ⚠️
--------------------------
split_image_v2() uses a stride-based sliding window, NOT a simple ceil grid.
The stride formula (from SGDJSCC/utils/utils.py line 180–181) is:

    stride_h = max(128 - (H % 128), 1)   if H > 128  else  H
    stride_w = max(128 - (W % 128), 1)   if W > 128  else  W

When H is a *multiple* of 128 the stride equals 128 (non-overlapping, safe).
When H is *not* a multiple of 128 the stride shrinks, creating *overlapping*
patches. The closer H % 128 is to 127, the more severe the explosion:

    H=256  (256 % 128 = 0)   → stride=128 →   2 patches   ✓
    H=384  (384 % 128 = 0)   → stride=128 →   3 patches   ✓
    H=192  (192 % 128 = 64)  → stride=64  →   3 patches (64 px overlap)
    H=400  (400 % 128 = 16)  → stride=112 →   4 patches (16 px overlap)
    H=255  (255 % 128 = 127) → stride=1   → 129 patches  ⚠ extreme
    H=383  (383 % 128 = 127) → stride=1   → 257 patches  ⚠ extreme

For a 255×255 image this produces 129×129 = 16 641 patches, each going
through the full diffusion pipeline — unusable in practice.

Rule: always resize input images so that both H and W are multiples of 128
before calling prepare_patches().  The safest choices are 128, 256, 384, 512.

Images whose shorter side is < 128 px are resized to 128×128 (CropLongSide +
Resize) before splitting so the model always sees 128×128 inputs.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import torch
from torchvision import transforms

from sgdjscc_lab._sgdjscc import ensure_sgdjscc_on_path

ensure_sgdjscc_on_path()

from utils.utils import split_image_v2, merge_image_v2, CropLongSide  # noqa: E402

logger = logging.getLogger(__name__)

_TARGET_SIZE = 128


def preprocess_image(tensor: torch.Tensor) -> torch.Tensor:
    """Resize an image tensor to 128×128 using the SGDJSCC pipeline.

    Applies CropLongSide (centre-crop to square) then Resize(128).
    Input:  ``[1, 3, H, W]`` float in [0, 1]
    Output: ``[1, 3, 128, 128]`` float in [0, 1]
    """
    img_pil = transforms.ToPILImage()(tensor.squeeze(0))
    pipeline = transforms.Compose([
        CropLongSide(),
        transforms.Resize((_TARGET_SIZE, _TARGET_SIZE)),
        transforms.ToTensor(),
    ])
    return pipeline(img_pil).unsqueeze(0)


def prepare_patches(tensor: torch.Tensor):
    """Split an arbitrary-resolution image into 128×128 patches.

    For images where both H ≥ 128 and W ≥ 128: calls split_image_v2 directly.
    For images smaller than 128 in either dimension: applies CropLongSide +
    Resize(128) first, then splits — the model always sees 128×128 inputs.

    Input:  ``[1, 3, H, W]`` float in [0, 1]
    Output: (patches ``[N, 3, 128, 128]``, meta)

    **Warning:** patch count depends on H % 128 and W % 128, not simply on
    image size.  An image of 255×255 produces 129×129 = 16 641 patches.
    See module docstring for the full stride formula and safe input sizes.
    """
    _, _, H, W = tensor.shape
    if H >= _TARGET_SIZE and W >= _TARGET_SIZE:
        return split_image_v2(tensor)
    return split_image_v2(preprocess_image(tensor))


def split_patches(img: torch.Tensor):
    """Raw wrapper around split_image_v2 (no pre-processing).

    Input:  ``[1, 3, H, W]`` tensor
    Output: (patches ``[N, 3, 128, 128]``, meta)
    """
    return split_image_v2(img)


def merge_patches(patches: torch.Tensor, meta) -> torch.Tensor:
    """Wrapper around merge_image_v2.

    Input:  patches ``[N, 3, 128, 128]`` + meta from split_patches
    Output: ``[1, 3, H_orig, W_orig]`` tensor
    """
    return merge_image_v2(patches, meta)
