"""jscc_model.py – JSCC encoder/decoder model.

Extracted from the inline _JSCCModel class that was defined inside
runtime.build_models().  The AWGN channel is now delegated to AWGNChannel
so it can be replaced independently (e.g. Phase 5 Rayleigh).

All model construction arguments mirror inference_one.py exactly.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgdjscc_lab._sgdjscc import ensure_sgdjscc_on_path
from sgdjscc_lab.channels.awgn import AWGNChannel

logger = logging.getLogger(__name__)

# ── SGDJSCC model config (identical to inference_one.py) ─────────────────────
DDCONFIG = {
    "double_z": True,
    "z_channels": 16,
    "resolution": 128,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
}

# Canny JSCC model constructor arguments (from inference_one.py line 39)
_CANNY_FILTERS = [256, 2, 8, [128, 192, 256, 320], [2, 2, 6, 2], [4, 6, 8, 10]]


class JSCCModel(nn.Module):
    """JSCC model: VAE encoder/decoder + blind SNR predictor + canny TX net.

    Mirrors the inline _JSCCModel from SGDJSCC/inference_one.py.
    AWGN channel injection is delegated to AWGNChannel.transmit().
    """

    def __init__(self) -> None:
        super().__init__()
        ensure_sgdjscc_on_path()
        from models.test_advanced_network.autoencoderkl import AutoencoderKL
        from models.test_advanced_network.snr_prediction_net import Prediction_Model
        from models.model_canny import Semantic_Communication_Model as CannySCModel

        self.vae = AutoencoderKL(DDCONFIG, 16)
        self.snr: float = 10.0
        self.snr_prediction_net = Prediction_Model()
        self.canny_transmission_net = CannySCModel(
            filters=_CANNY_FILTERS,
            snrdB=-5,
            channel="Dynamic_AWGN",
            channel_coding=False,
            modulating=False,
            in_feature=8192,
            size1=640,
            size2=320,
            model_type="vit_witt_adaln_radepth_4_uncertain",
            task_name="reconstruction",
        )
        self._awgn_channel = AWGNChannel()
        # Phase 5-A: optional channel override. When None, behaviour is identical
        # to the original AWGN path (algorithm-preservation invariant). Set this
        # to a Rayleigh/fast-fading/packet-drop channel to transmit over it.
        self.channel_model = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalise per sample and rescale to original energy."""
        batch_size, c, h, w = x.shape
        x = x.reshape(batch_size, -1)
        x = F.normalize(x, p=2, dim=1) * math.sqrt(x.shape[1])
        return x.reshape(batch_size, c, h, w)

    def channel(self, encode_features: torch.Tensor) -> torch.Tensor:
        """Apply channel noise at self.snr dB.

        Uses ``self.channel_model`` when set (Phase 5-A Rayleigh / fading /
        packet-drop), otherwise the original AWGN channel — so the default path
        is numerically identical to ``inference_one.py``.
        """
        ch = self.channel_model if self.channel_model is not None else self._awgn_channel
        return ch.transmit(encode_features, self.snr)


def build_jscc_model(
    model_root: Union[Path, str],
    device: torch.device,
) -> JSCCModel:
    """Instantiate JSCCModel, load JSCC_model.pth checkpoint, return in eval mode."""
    model = JSCCModel()
    model.to(device)
    ckpt_path = Path(model_root) / "JSCC_model.pth"
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded JSCC_model.pth from %s", ckpt_path)
    return model
