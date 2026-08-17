"""tests/test_transmission_reduction_temporal_integration.py – Small, CPU-only,
mock-reconstruction integration test proving the transmission-reduction
driver's TemporalPipeline wiring (keyframe_extractor + reconstruct_fn/
packet_fn signatures + the TRANSMITTING_DECISIONS filter) behaves correctly,
without needing real checkpoints/GPU. The real-model, real-GPU version of
this same wiring is exercised by the (separately run, not part of this suite)
1-video smoke test — see scripts/run_transmission_reduction_eval.py's module
docstring.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor
from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector
from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline


def _load_driver_module():
    spec = importlib.util.spec_from_file_location(
        "_txred_integration_mod", _REPO_ROOT / "scripts" / "run_transmission_reduction_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_driver_module()


def _mock_reconstruct_fn(frame, run_cfg):
    """Identity-like mock reconstruction (no models needed) — matches the
    exact (frame, run_cfg) -> Tensor signature
    pipelines/eval_pipeline.py::_reconstruct_with_cfg has, so this test
    proves the wiring pattern the real driver uses, not a different one."""
    return frame.clone() * 0.99  # trivially distinguishable from the input


def _mock_packet_fn(frame, frame_id):
    return {"frame_id": str(frame_id)}  # same minimal stub the real driver uses


class TestTransmittingDecisionsFilter:
    def test_mock_pipeline_produces_only_documented_decisions(self):
        torch.manual_seed(0)
        frames = [torch.rand(1, 3, 32, 32) for _ in range(12)]
        keyframe_extractor = KeyframeExtractor(SceneChangeDetector(), max_gop=4)

        pipeline = TemporalPipeline(
            reconstruct_fn=_mock_reconstruct_fn, packet_fn=_mock_packet_fn,
            keyframe_extractor=keyframe_extractor,
        )
        result = pipeline.run(frames)
        records = sorted(result["records"], key=lambda r: r.index)

        assert len(records) == len(frames)
        decisions = {r.decision for r in records}
        assert decisions <= {"keyframe", "reuse", "recompute_semantic", "recompute_motion", "generate"}
        # max_gop=4 with an empty (no objects/relations) packet_fn -> every
        # inter-frame's semantic delta is "no change" -> reuse, except the
        # keyframe_extractor's own forced boundaries every 4 frames.
        assert records[0].decision == "keyframe"

    def test_transmitting_decisions_constant_matches_frame_record_decision_values(self):
        # Regression: mod.TRANSMITTING_DECISIONS must be a subset of the real
        # decision vocabulary TemporalPipeline actually produces, or the
        # driver's shadow-accounting loop would silently never fire (or crash
        # trying to filter on an unknown value).
        torch.manual_seed(0)
        frames = [torch.rand(1, 3, 32, 32) for _ in range(8)]
        keyframe_extractor = KeyframeExtractor(SceneChangeDetector(), max_gop=3)
        pipeline = TemporalPipeline(
            reconstruct_fn=_mock_reconstruct_fn, packet_fn=_mock_packet_fn,
            keyframe_extractor=keyframe_extractor,
        )
        result = pipeline.run(frames)
        real_decisions = {r.decision for r in result["records"]}
        assert real_decisions.issubset(
            {"keyframe", "reuse", "recompute_semantic", "recompute_motion", "generate"}
        )
        assert set(mod.TRANSMITTING_DECISIONS).issubset(
            {"keyframe", "reuse", "recompute_semantic", "recompute_motion", "generate"}
        )
        # at least the forced keyframes must be classified as transmitting
        transmitting = [r for r in result["records"] if r.decision in mod.TRANSMITTING_DECISIONS]
        assert any(r.decision == "keyframe" for r in transmitting)
        assert len(transmitting) >= 1

    def test_reused_frames_never_counted_as_transmitting(self):
        torch.manual_seed(0)
        frames = [torch.rand(1, 3, 32, 32) for _ in range(6)]
        # max_gop larger than n_frames -> only frame 0 is ever a keyframe,
        # every other frame reuses it (empty packet_fn -> zero delta always).
        keyframe_extractor = KeyframeExtractor(SceneChangeDetector(), max_gop=100)
        pipeline = TemporalPipeline(
            reconstruct_fn=_mock_reconstruct_fn, packet_fn=_mock_packet_fn,
            keyframe_extractor=keyframe_extractor,
        )
        result = pipeline.run(frames)
        records = sorted(result["records"], key=lambda r: r.index)
        assert records[0].decision == "keyframe"
        reused = [r for r in records[1:] if r.decision == "reuse"]
        assert len(reused) == len(records) - 1
        transmitting = [r for r in records if r.decision in mod.TRANSMITTING_DECISIONS]
        assert transmitting == [records[0]]

    def test_every_frame_gets_a_recon_tensor_for_full_video_saving(self):
        # The driver saves EVERY record's .recon (not just keyframes) so
        # recon_videos/ holds the full reconstructed video, not a keyframe-only
        # subset — this is the "point 2" full-video-evaluation fix.
        torch.manual_seed(0)
        frames = [torch.rand(1, 3, 32, 32) for _ in range(5)]
        keyframe_extractor = KeyframeExtractor(SceneChangeDetector(), max_gop=2)
        pipeline = TemporalPipeline(
            reconstruct_fn=_mock_reconstruct_fn, packet_fn=_mock_packet_fn,
            keyframe_extractor=keyframe_extractor,
        )
        result = pipeline.run(frames)
        records = sorted(result["records"], key=lambda r: r.index)
        assert all(r.recon is not None for r in records)
        assert all(r.recon.shape == frames[0].shape for r in records)
