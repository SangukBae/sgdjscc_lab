"""data/transforms.py – Common image transform for stage-aware training.

A single config-driven transform shared by all three dataset types so the
paper's preprocessing ("All the images are center-cropped and resized to
128×128", Sec. VI) is expressed once.

Config schema (train/default.yaml → train.transforms)
-----------------------------------------------------
train:
  transforms:
    resize_to: 128          # int → square side, or [H, W]
    crop_mode: center       # "center" | "random" | "none"
    normalize: false        # false → keep [0, 1]; true → map to [-1, 1]

Notes
-----
- Operates on a CHW float tensor in ``[0, 1]`` and returns the same layout.
- ``normalize`` should stay **false** for the JSCC stage: the JSCC VAE encode
  path applies its own ``x*2-1`` internally (algorithm-preservation invariant).
  It exists for symmetry / future use.
"""

from __future__ import annotations

import random
from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


def _as_hw(resize_to: Union[int, Sequence[int], None]) -> Union[Tuple[int, int], None]:
    if resize_to is None:
        return None
    if isinstance(resize_to, (list, tuple)):
        h, w = int(resize_to[0]), int(resize_to[1])
        return h, w
    s = int(resize_to)
    return s, s


class ImageTransform:
    """Resize (shorter-side preserving) → crop → optional normalize.

    Parameters
    ----------
    resize_to:
        Target square side (int) or ``(H, W)``.  None disables resizing.
    crop_mode:
        ``"center"`` (default), ``"random"`` or ``"none"``.
    normalize:
        If True, map ``[0, 1] → [-1, 1]`` after cropping.
    training:
        When False, ``crop_mode="random"`` is downgraded to ``"center"`` so
        validation is deterministic.
    """

    def __init__(
        self,
        resize_to: Union[int, Sequence[int], None] = 128,
        crop_mode: str = "center",
        normalize: bool = False,
        training: bool = True,
    ) -> None:
        self.hw = _as_hw(resize_to)
        self.crop_mode = str(crop_mode).lower()
        self.normalize = bool(normalize)
        self.training = bool(training)
        if self.crop_mode not in ("center", "random", "none"):
            raise ValueError(
                f"crop_mode must be center|random|none, got {crop_mode!r}"
            )

    # ── public API ────────────────────────────────────────────────────────────
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.hw is not None:
            img = self._resize_and_crop(img, self.hw)
        if self.normalize:
            img = img * 2.0 - 1.0
        return img

    # ── helpers ───────────────────────────────────────────────────────────────
    def _resize_and_crop(self, img: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        target_h, target_w = hw
        _, H, W = img.shape

        # Resize so the shorter side covers the target (preserve aspect ratio),
        # then crop to the exact target — matches "center-cropped and resized".
        scale = max(target_h / H, target_w / W)
        new_h, new_w = max(target_h, int(round(H * scale))), max(target_w, int(round(W * scale)))
        if (new_h, new_w) != (H, W):
            img = F.interpolate(
                img.unsqueeze(0), size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        _, H, W = img.shape
        mode = self.crop_mode
        if mode == "none":
            return img
        if mode == "random" and self.training:
            top = random.randint(0, H - target_h)
            left = random.randint(0, W - target_w)
        else:  # center (or random in eval)
            top = (H - target_h) // 2
            left = (W - target_w) // 2
        return img[:, top:top + target_h, left:left + target_w]


def build_transform(cfg: DictConfig, *, training: bool = True) -> ImageTransform:
    """Build an :class:`ImageTransform` from ``train.transforms`` in *cfg*.

    Paper-reproduction defaults: ``resize_to=128``, ``crop_mode="center"``,
    ``normalize=false``.
    """
    t_cfg = OmegaConf.select(cfg, "train.transforms", default=None)
    resize_to = 128
    crop_mode = "center"
    normalize = False
    if t_cfg is not None:
        resize_to = OmegaConf.select(t_cfg, "resize_to", default=128)
        crop_mode = OmegaConf.select(t_cfg, "crop_mode", default="center")
        normalize = bool(OmegaConf.select(t_cfg, "normalize", default=False))
    return ImageTransform(resize_to, crop_mode, normalize, training=training)
