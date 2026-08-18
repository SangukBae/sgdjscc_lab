from __future__ import annotations

import inspect

import pytest
import torch
from omegaconf import OmegaConf

from sgdjscc_lab.transmission.packet_bundle import build_frame_bundle, serialize_bundle
from sgdjscc_lab.transmission.receiver_runtime import reconstruct_frame_from_bundle_bytes


class _Jscc:
    snr = 10.0

    @staticmethod
    def normalize(x):
        return x


class _Models:
    device = torch.device("cpu")
    jscc_model = _Jscc()
    sem_pipeline = object()


def _cfg(mask_method="none"):
    return OmegaConf.create({
        "mask_method": mask_method,
        "step_style": "continuous",
        "use_jscc_feature": True,
        "use_gt_csi": False,
        "use_semantic": True,
    })


def _data(analog=False):
    visual = None if analog else torch.ones(1, 16, 16, 16)
    edge = torch.zeros(1, 11, 128, 128)
    bundle = build_frame_bundle(
        visual_latent_patches=visual, visual_is_analog=analog,
        visual_bit_depth=None if analog else 8, visual_granularity="per_tensor",
        visual_channel_dim=1, visual_channel_symbols=4096,
        caption=["receiver only"], edge_tensor=edge,
        edge_uncertainty_tensor=torch.zeros_like(edge), edge_bit_depth=8,
        keyframe_index=0,
        manifest={
            "patch_layout": {"height": 128, "width": 128, "positions": [[0, 0]]},
            "n_patches": 1, "selected_keyframes": [0],
        },
    )
    return serialize_bundle(bundle)


def test_receiver_api_has_no_source_frame_parameter():
    assert list(inspect.signature(reconstruct_frame_from_bundle_bytes).parameters) == [
        "data", "models", "cfg"
    ]


def test_receiver_decodes_from_bytes_and_patch_manifest(monkeypatch):
    import sgdjscc_lab.pipelines.infer_pipeline as infer

    monkeypatch.setattr(infer, "_compute_step", lambda **kwargs: (0.5, 10.0))
    monkeypatch.setattr(
        infer, "_decode_diffusion",
        lambda artifacts, jscc, pipe, gt_text, cfg, device, original_image=None:
            torch.full((1, 3, 128, 128), 0.25),
    )
    result = reconstruct_frame_from_bundle_bytes(_data(), _Models(), _cfg())
    assert result.shape == (1, 3, 128, 128)
    assert torch.allclose(result, torch.full_like(result, 0.25))


def test_receiver_rejects_analog_descriptor_without_visual_samples():
    with pytest.raises(ValueError, match="analog visual samples"):
        reconstruct_frame_from_bundle_bytes(_data(analog=True), _Models(), _cfg())


def test_receiver_rejects_sender_dependent_mask_mode():
    with pytest.raises(ValueError, match="mask_method=none"):
        reconstruct_frame_from_bundle_bytes(_data(), _Models(), _cfg("topk"))
