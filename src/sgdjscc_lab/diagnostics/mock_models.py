"""diagnostics/mock_models.py – CPU-only synthetic ModelBundle for tests/dry-run.

NOT used for any real measurement — only exercises the harness's routing,
tensor-instrumentation, comparison, and CSV/report-writing code paths on CPU
without real checkpoints. See ``scripts/diagnose_float32_digital_quality.py``'s
``--no-models`` flag and ``tests/test_float32_digital_diagnostics.py``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from sgdjscc_lab.models.model_bundle import ModelBundle

_LATENT_CH = 16
_LATENT_HW = 16
_SCALING_FACTOR = 15.45


class _MockLatentDist:
    def __init__(self, mean: torch.Tensor):
        self.mean = mean
        self.std = torch.zeros_like(mean)


class _MockVaeEncodeOut:
    """Mirrors diffusers' ``AutoencoderKLOutput``: ``.latent_dist`` AND
    ``[0]`` both give the latent distribution (``_encode_latent`` uses the
    attribute form, ``_encode_canny_latent`` uses the subscript form)."""

    def __init__(self, mean: torch.Tensor):
        self.latent_dist = _MockLatentDist(mean)

    def __getitem__(self, index: int) -> _MockLatentDist:
        if index != 0:
            raise IndexError(index)
        return self.latent_dist


class _MockVae:
    """Deterministic, invertible-ish stand-in for the real VAE.

    ``encode(...).mean`` is a pure function of ``x`` (no sampling), so
    repeated calls are bit-identical — required for the harness's redundant
    pre/post-normalization latent capture to be meaningful.
    """

    def encode(self, x: torch.Tensor) -> _MockVaeEncodeOut:
        pooled = F.adaptive_avg_pool2d(x, (_LATENT_HW, _LATENT_HW))
        mean = pooled.mean(dim=1, keepdim=True).repeat(1, _LATENT_CH, 1, 1) * _SCALING_FACTOR
        return _MockVaeEncodeOut(mean)

    def decode(self, z: torch.Tensor):
        img = z.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        img = F.interpolate(img, size=(128, 128), mode="nearest")
        return (img,)


def _mock_canny_transmission_net(cat: torch.Tensor, gt_snr=None, snr=None, cr=None) -> torch.Tensor:
    return cat[:, :1].contiguous()


class _MockJscc:
    def __init__(self, snr_db: float = 10.0):
        self.snr = float(snr_db)
        self.channel_model = None
        self.vae = _MockVae()
        self.snr_prediction_net = lambda z: torch.full((z.shape[0], 1), 0.3)
        self.canny_transmission_net = _mock_canny_transmission_net

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def channel(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_model is not None:
            return self.channel_model.transmit(x, self.snr)
        return x + 0.01 * torch.randn_like(x)


class _MockPipe:
    alphas_cumprod = torch.linspace(0.999, 0.001, 1000)

    def generate(self, **kwargs: Any):
        latent = kwargs["latent"]
        return None, latent + 0.001 * torch.ones_like(latent)


def build_diagnostic_cfg(**overrides: Any):
    """Minimal cfg with every key the instrumented paths access, for
    ``--no-models``/CPU tests. Real (non-mock) runs use the project's actual
    composed config instead (see ``diagnose_float32_digital_quality.py::_make_cfg``)."""
    base = {
        "use_semantic": True, "use_text": True, "use_controlnet": True,
        "use_jscc_feature": True, "use_gt_csi": False, "mask_method": "none",
        "step_style": "continuous", "canny_cr": "1", "diffusion_step": 2,
        "guidance_scale": 3.0, "controlnet_scale": 1.0, "cfg_method": "constant",
        "th": 0.5, "snr_db": 10.0, "use_phase4": True,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def build_mock_models(device: "str | torch.device" = "cpu", snr_db: float = 10.0) -> ModelBundle:
    dev = torch.device(device)
    return ModelBundle(
        jscc_model=_MockJscc(snr_db=snr_db), sem_pipeline=_MockPipe(),
        text_extractor=None, edge_extractor=None, device=dev,
    )
