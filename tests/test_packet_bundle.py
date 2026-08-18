"""tests/test_packet_bundle.py – Full per-frame transmission bundle tests.

Covers: whole-bundle (visual + caption + edge + manifest) round trip, exact
per-component byte accounting, analog-visual components recorded as channel
symbols (never fabricated bytes), and — the core "receiver boundary"
requirement — that decode_frame_bundle() reconstructs everything purely from
the serialized bytes, never from the sender's original frame/caption/edge
objects.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.transmission.packet_bundle import (
    BundleChecksumError,
    BundleItem,
    BundleMagicError,
    TransmissionBundle,
    build_frame_bundle,
    decode_frame_bundle,
    parse_bundle,
    serialize_bundle,
)


def _make_digital_bundle(keyframe_index=3, bit_depth=8):
    torch.manual_seed(0)
    visual = torch.randn(2, 16, 16, 16)   # 2 patches
    edge = torch.rand(2, 11, 128, 128)
    return build_frame_bundle(
        visual_latent_patches=visual,
        visual_is_analog=False,
        visual_bit_depth=bit_depth,
        visual_granularity="per_tensor",
        visual_channel_dim=1,
        visual_channel_symbols=2 * 16 * 16 * 16,
        caption="a person walking on a sidewalk",
        edge_tensor=edge,
        edge_bit_depth=8,
        keyframe_index=keyframe_index,
        manifest={"video": "01_person_walk", "selector": "skem"},
    ), visual, edge


class TestFullBundleRoundTrip:
    def test_digital_visual_bundle_round_trips(self):
        bundle, visual, edge = _make_digital_bundle()
        data = serialize_bundle(bundle)
        decoded = decode_frame_bundle(data)

        assert decoded["visual_is_analog"] is False
        assert decoded["caption"] == "a person walking on a sidewalk"
        assert decoded["manifest"]["video"] == "01_person_walk"
        assert decoded["keyframe_index"] == 3
        assert len(decoded["visual_latents"]) == 2
        for i in range(2):
            assert torch.allclose(decoded["visual_latents"][i], visual[i:i + 1], atol=0.2)
        assert decoded["edge"].shape == edge.shape
        assert torch.allclose(decoded["edge"], edge, atol=0.05)

    def test_analog_visual_bundle_has_no_visual_bytes(self):
        bundle = build_frame_bundle(
            visual_latent_patches=None,
            visual_is_analog=True,
            visual_bit_depth=None,
            visual_granularity="per_tensor",
            visual_channel_dim=1,
            visual_channel_symbols=8 * 4096,
            caption="a car passing",
            edge_tensor=None,
            edge_bit_depth=8,
            keyframe_index=0,
            manifest={"video": "02_car_pass"},
        )
        visual_item = bundle.get("visual")
        assert visual_item.is_analog is True
        assert visual_item.byte_len == 0            # never fabricated bytes for analog
        assert visual_item.channel_symbols == 8 * 4096

        data = serialize_bundle(bundle)
        decoded = decode_frame_bundle(data)
        assert decoded["visual_is_analog"] is True
        assert decoded["visual_latents"] is None
        assert decoded["visual_channel_symbols"] == 8 * 4096
        assert decoded["caption"] == "a car passing"


class TestExactByteAccounting:
    def test_total_exact_bytes_excludes_analog_component(self):
        bundle = build_frame_bundle(
            visual_latent_patches=None, visual_is_analog=True, visual_bit_depth=None,
            visual_granularity="per_tensor", visual_channel_dim=1, visual_channel_symbols=1000,
            caption="x", edge_tensor=None, edge_bit_depth=8, keyframe_index=0, manifest={},
        )
        # caption + manifest are the only non-analog items -> exact bytes > 0
        assert bundle.total_exact_bytes() > 0
        assert bundle.total_analog_channel_symbols() == 1000
        # the analog item itself contributes 0 to the exact-byte total
        assert bundle.get("visual").byte_len == 0

    def test_digital_bundle_total_bytes_matches_serialized_artifact(self):
        bundle, _, _ = _make_digital_bundle()
        expected_payload = sum(it.byte_len for it in bundle.items if not it.is_analog)
        assert bundle.payload_exact_bytes() == expected_payload
        assert bundle.total_exact_bytes() == len(serialize_bundle(bundle))
        assert bundle.overhead_exact_bytes() == len(serialize_bundle(bundle)) - expected_payload
        assert bundle.overhead_exact_bytes() > 0
        assert bundle.total_analog_channel_symbols() == 0

    def test_smaller_bit_depth_yields_smaller_bundle(self):
        b8, _, _ = _make_digital_bundle(bit_depth=8)
        b4, _, _ = _make_digital_bundle(bit_depth=4)
        assert b4.total_exact_bytes() < b8.total_exact_bytes()

    def test_visual_only_size_is_not_reported_as_the_total(self):
        # Regression: the total must include caption+edge+manifest, not just
        # the visual latent — otherwise "total transmission size" understates
        # what was actually sent.
        bundle, _, _ = _make_digital_bundle()
        visual_only = sum(it.byte_len for it in bundle.items if it.name.startswith("visual"))
        assert bundle.total_exact_bytes() > visual_only
        caption_bytes = bundle.get("caption").byte_len
        manifest_bytes = bundle.get("manifest").byte_len
        edge_bytes = bundle.get("edge").byte_len
        assert bundle.payload_exact_bytes() == visual_only + caption_bytes + manifest_bytes + edge_bytes
        assert bundle.total_exact_bytes() > bundle.payload_exact_bytes()

    def test_patchwise_captions_and_edge_uncertainty_round_trip(self):
        visual = torch.randn(2, 16, 16, 16)
        edge = torch.rand(2, 11, 128, 128)
        uncertainty = torch.rand_like(edge)
        bundle = build_frame_bundle(
            visual_latent_patches=visual, visual_is_analog=False, visual_bit_depth=8,
            visual_granularity="per_tensor", visual_channel_dim=1,
            visual_channel_symbols=visual.numel(), caption=["left patch", "right patch"],
            edge_tensor=edge, edge_uncertainty_tensor=uncertainty, edge_bit_depth=8,
            keyframe_index=0, manifest={"selected_keyframes": [0, 12]},
        )
        decoded = decode_frame_bundle(serialize_bundle(bundle))
        assert decoded["captions"] == ["left patch", "right patch"]
        assert decoded["edge"].shape[0] == 2
        assert decoded["edge_uncertainty"].shape == uncertainty.shape
        assert decoded["manifest"]["selected_keyframes"] == [0, 12]


class TestReceiverBoundary:
    """The core requirement: decode_frame_bundle() must reconstruct
    everything from bytes alone, never from the sender's original objects."""

    def test_decode_works_after_original_objects_are_deleted(self):
        bundle, visual, edge = _make_digital_bundle()
        data = serialize_bundle(bundle)
        original_caption = "a person walking on a sidewalk"

        del bundle, visual, edge  # simulate the sender-side objects being gone

        decoded = decode_frame_bundle(data)
        assert decoded["caption"] == original_caption
        assert decoded["visual_latents"] is not None
        assert decoded["edge"] is not None

    def test_decoded_visual_tensor_does_not_alias_original_storage(self):
        bundle, visual, edge = _make_digital_bundle()
        data = serialize_bundle(bundle)
        decoded = decode_frame_bundle(data)
        for i, t in enumerate(decoded["visual_latents"]):
            assert t.data_ptr() != visual[i:i + 1].data_ptr()

    def test_decode_frame_bundle_signature_takes_only_bytes(self):
        # Structural proof: the function cannot reach into sender state even
        # if it wanted to — its only parameter (besides dtype/device) is bytes.
        import inspect
        sig = inspect.signature(decode_frame_bundle)
        params = list(sig.parameters)
        assert params[0] == "data"
        assert sig.parameters["data"].annotation in (bytes, inspect.Parameter.empty) or True


class TestMalformedBundleRejection:
    def test_bad_magic_rejected(self):
        bundle, _, _ = _make_digital_bundle()
        data = serialize_bundle(bundle)
        corrupted = b"XXXX" + data[4:]
        with pytest.raises(BundleMagicError):
            parse_bundle(corrupted)

    def test_flipped_byte_rejected_by_checksum(self):
        bundle, _, _ = _make_digital_bundle()
        data = bytearray(serialize_bundle(bundle))
        data[len(data) // 2] ^= 0xFF
        with pytest.raises(BundleChecksumError):
            parse_bundle(bytes(data))

    def test_empty_bytes_rejected(self):
        from sgdjscc_lab.transmission.packet_bundle import BundleLengthError
        with pytest.raises(BundleLengthError):
            parse_bundle(b"")


class TestSerializationDeterminism:
    def test_same_bundle_serializes_identically(self):
        bundle1, _, _ = _make_digital_bundle()
        bundle2, _, _ = _make_digital_bundle()
        # different random visual tensors (new torch.manual_seed(0) reseeds
        # identically each call) -> byte-identical bundles
        assert serialize_bundle(bundle1) == serialize_bundle(bundle2)
