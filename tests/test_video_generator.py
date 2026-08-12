"""tests/test_video_generator.py – video/video_generator.py backend tests (offline).

Covers the ETRI 3차 start-only generate-branch backend interface in isolation
(GenerationRequest/Result/Metadata, CopyGenerator, InterpolationGenerator,
build_generator registry, the Rx-legal ground-truth-reference boundary, the
bidirectional-conditioning NotImplementedError guard, and save_generated_frames).
TemporalPipeline-level 3-way decision wiring is covered separately in
tests/test_video.py::TestGenerateBranch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ─────────────────────────────────────────────────────────────────────────────
# CopyGenerator
# ─────────────────────────────────────────────────────────────────────────────

class TestCopyGenerator:
    def test_returns_clone_of_keyframe(self):
        from sgdjscc_lab.video.video_generator import CopyGenerator, GenerationRequest
        kf = torch.rand(1, 3, 8, 8)
        req = GenerationRequest(start_keyframe_recon=kf, start_keyframe_index=0, target_index=3)
        result = CopyGenerator().generate(req)
        assert torch.equal(result.frame, kf)
        assert result.frame is not kf   # clone, not the same tensor object

    def test_metadata_fields(self):
        from sgdjscc_lab.video.video_generator import CopyGenerator, GenerationRequest
        req = GenerationRequest(
            start_keyframe_recon=torch.zeros(1, 3, 4, 4), start_keyframe_index=2, target_index=5,
            caption="a red car", packet={"objects": ["car"]}, side_info={"delta": 0.3},
        )
        meta = CopyGenerator().generate(req).metadata.to_dict()
        assert meta["backend"] == "copy"
        assert meta["conditioning_mode"] == "start_only"
        assert meta["source_keyframe_index"] == 2
        assert meta["target_indices"] == [5]
        assert meta["used_caption"] is True
        assert meta["used_side_info"] is True
        assert meta["mock"] is True

    def test_metadata_flags_false_without_caption_or_side_info(self):
        from sgdjscc_lab.video.video_generator import CopyGenerator, GenerationRequest
        req = GenerationRequest(start_keyframe_recon=torch.zeros(1, 3, 4, 4), start_keyframe_index=0, target_index=1)
        meta = CopyGenerator().generate(req).metadata.to_dict()
        assert meta["used_caption"] is False
        assert meta["used_side_info"] is False

    def test_metadata_is_json_serialisable(self):
        import json
        from sgdjscc_lab.video.video_generator import CopyGenerator, GenerationRequest
        req = GenerationRequest(start_keyframe_recon=torch.zeros(1, 3, 4, 4), start_keyframe_index=0, target_index=1)
        json.dumps(CopyGenerator().generate(req).metadata.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# InterpolationGenerator + Rx-legal ground-truth-reference boundary
# ─────────────────────────────────────────────────────────────────────────────

class TestInterpolationGenerator:
    def test_blends_with_prev_recon(self):
        from sgdjscc_lab.video.video_generator import InterpolationGenerator, GenerationRequest
        kf = torch.zeros(1, 3, 4, 4)
        prev = torch.ones(1, 3, 4, 4)
        req = GenerationRequest(
            start_keyframe_recon=kf, start_keyframe_index=0, target_index=1,
            reference_prev_recon=prev,
        )
        result = InterpolationGenerator(alpha=0.5).generate(req)
        assert torch.allclose(result.frame, torch.full((1, 3, 4, 4), 0.5))
        assert "ground-truth" not in result.metadata.notes

    def test_alpha_zero_is_pure_keyframe(self):
        from sgdjscc_lab.video.video_generator import InterpolationGenerator, GenerationRequest
        kf = torch.zeros(1, 3, 4, 4)
        prev = torch.ones(1, 3, 4, 4)
        req = GenerationRequest(
            start_keyframe_recon=kf, start_keyframe_index=0, target_index=1,
            reference_prev_recon=prev,
        )
        result = InterpolationGenerator(alpha=0.0).generate(req)
        assert torch.allclose(result.frame, kf)

    def test_no_reference_degenerates_to_copy(self):
        from sgdjscc_lab.video.video_generator import InterpolationGenerator, GenerationRequest
        kf = torch.rand(1, 3, 4, 4)
        req = GenerationRequest(start_keyframe_recon=kf, start_keyframe_index=0, target_index=1)
        result = InterpolationGenerator(alpha=0.5).generate(req)
        assert torch.equal(result.frame, kf)

    def test_ground_truth_reference_ignored_by_default(self):
        """Rx-legal boundary: reference_target_frame must NOT be used unless the
        backend was explicitly constructed with allow_ground_truth_reference=True."""
        from sgdjscc_lab.video.video_generator import InterpolationGenerator, GenerationRequest
        kf = torch.zeros(1, 3, 4, 4)
        target = torch.ones(1, 3, 4, 4)
        req = GenerationRequest(
            start_keyframe_recon=kf, start_keyframe_index=0, target_index=1,
            reference_target_frame=target,   # only legal when allow_ground_truth_reference=True
        )
        result = InterpolationGenerator(alpha=0.5)  # allow_ground_truth_reference defaults False
        out = result.generate(req)
        assert torch.equal(out.frame, kf)   # target frame ignored → degenerates to copy
        assert "ground-truth" not in out.metadata.notes

    def test_ground_truth_reference_used_only_when_explicitly_enabled(self):
        from sgdjscc_lab.video.video_generator import InterpolationGenerator, GenerationRequest
        kf = torch.zeros(1, 3, 4, 4)
        target = torch.ones(1, 3, 4, 4)
        req = GenerationRequest(
            start_keyframe_recon=kf, start_keyframe_index=0, target_index=1,
            reference_target_frame=target,
        )
        gen = InterpolationGenerator(alpha=0.5, allow_ground_truth_reference=True)
        out = gen.generate(req)
        assert torch.allclose(out.frame, torch.full((1, 3, 4, 4), 0.5))
        assert "ground-truth" in out.metadata.notes
        assert out.metadata.mock is True   # always tagged mock regardless of reference source

    def test_prev_recon_preferred_over_ground_truth_even_when_allowed(self):
        from sgdjscc_lab.video.video_generator import InterpolationGenerator, GenerationRequest
        kf = torch.zeros(1, 3, 4, 4)
        prev = torch.full((1, 3, 4, 4), 0.2)
        target = torch.ones(1, 3, 4, 4)
        req = GenerationRequest(
            start_keyframe_recon=kf, start_keyframe_index=0, target_index=1,
            reference_prev_recon=prev, reference_target_frame=target,
        )
        gen = InterpolationGenerator(alpha=0.5, allow_ground_truth_reference=True)
        out = gen.generate(req)
        assert torch.allclose(out.frame, torch.full((1, 3, 4, 4), 0.1))  # blend with prev, not target
        assert "ground-truth" not in out.metadata.notes


# ─────────────────────────────────────────────────────────────────────────────
# Reserved 4차 bidirectional extension point — must stay unimplemented in 3차
# ─────────────────────────────────────────────────────────────────────────────

class TestStartOnlyBackendsRejectEndKeyframe:
    """Start-only backends must reject end_keyframe_recon — it is
    start-only-illegal, not merely unused (ETRI 4차 makes bidirectional a real
    mode elsewhere, but these two backends never support it)."""

    def test_end_keyframe_recon_raises_in_copy_generator(self):
        from sgdjscc_lab.video.video_generator import CopyGenerator, GenerationRequest
        req = GenerationRequest(
            start_keyframe_recon=torch.zeros(1, 3, 4, 4), start_keyframe_index=0, target_index=1,
            end_keyframe_recon=torch.zeros(1, 3, 4, 4),
        )
        with pytest.raises(NotImplementedError):
            CopyGenerator().generate(req)

    def test_end_keyframe_recon_raises_in_interpolation_generator(self):
        from sgdjscc_lab.video.video_generator import InterpolationGenerator, GenerationRequest
        req = GenerationRequest(
            start_keyframe_recon=torch.zeros(1, 3, 4, 4), start_keyframe_index=0, target_index=1,
            end_keyframe_recon=torch.zeros(1, 3, 4, 4),
        )
        with pytest.raises(NotImplementedError):
            InterpolationGenerator().generate(req)


class TestBidirectionalConditioningModeBuild:
    """ETRI 4차: conditioning_mode='bidirectional' is now implemented (mock
    backend only) — see TestBidirectionalInterpolationGenerator below for the
    backend's own behaviour."""

    def test_build_generator_returns_bidirectional_backend(self):
        from sgdjscc_lab.video.video_generator import build_generator, BidirectionalInterpolationGenerator
        cfg = OmegaConf.create({"video_generator": {"conditioning_mode": "bidirectional"}})
        gen = build_generator(cfg)
        assert isinstance(gen, BidirectionalInterpolationGenerator)
        assert gen.missing_end_policy == "error"   # default

    def test_auto_backend_selects_mode_canonical_backend(self):
        from sgdjscc_lab.video.video_generator import (
            build_generator, CopyGenerator, BidirectionalInterpolationGenerator,
        )
        start_cfg = OmegaConf.create({"video_generator": {
            "conditioning_mode": "start_only", "backend": "auto",
        }})
        bidi_cfg = OmegaConf.create({"video_generator": {
            "conditioning_mode": "bidirectional", "backend": "auto",
        }})
        assert isinstance(build_generator(start_cfg), CopyGenerator)
        assert isinstance(build_generator(bidi_cfg), BidirectionalInterpolationGenerator)

    def test_build_generator_reads_missing_end_policy(self):
        from sgdjscc_lab.video.video_generator import build_generator
        cfg = OmegaConf.create({
            "video_generator": {
                "conditioning_mode": "bidirectional",
                "bidirectional_missing_end_policy": "fallback_start_only",
            }
        })
        gen = build_generator(cfg)
        assert gen.missing_end_policy == "fallback_start_only"

    def test_incompatible_backend_under_bidirectional_mode_raises(self):
        from sgdjscc_lab.video.video_generator import build_generator
        cfg = OmegaConf.create({
            "video_generator": {"conditioning_mode": "bidirectional", "backend": "interpolation"}
        })
        with pytest.raises(NotImplementedError):
            build_generator(cfg)

    def test_unknown_conditioning_mode_raises(self):
        from sgdjscc_lab.video.video_generator import build_generator
        cfg = OmegaConf.create({"video_generator": {"conditioning_mode": "something_else"}})
        with pytest.raises(NotImplementedError):
            build_generator(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# BidirectionalInterpolationGenerator (ETRI 4차, step 6)
# ─────────────────────────────────────────────────────────────────────────────

def _bidi_request(target_index, start_idx=0, end_idx=10, start_val=0.0, end_val=1.0, **kw):
    from sgdjscc_lab.video.video_generator import GenerationRequest
    return GenerationRequest(
        start_keyframe_recon=torch.full((1, 3, 4, 4), start_val),
        start_keyframe_index=start_idx,
        target_index=target_index,
        end_keyframe_recon=torch.full((1, 3, 4, 4), end_val),
        end_keyframe_index=end_idx,
        **kw,
    )


class TestBidirectionalInterpolationGenerator:
    def test_midpoint_is_even_blend(self):
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator
        req = _bidi_request(target_index=5, start_idx=0, end_idx=10)
        result = BidirectionalInterpolationGenerator().generate(req)
        assert result.metadata.relative_position == pytest.approx(0.5)
        assert torch.allclose(result.frame, torch.full((1, 3, 4, 4), 0.5))

    def test_target_at_start_is_pure_start_keyframe(self):
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator
        req = _bidi_request(target_index=0, start_idx=0, end_idx=10)
        result = BidirectionalInterpolationGenerator().generate(req)
        assert result.metadata.relative_position == pytest.approx(0.0)
        assert torch.allclose(result.frame, torch.zeros(1, 3, 4, 4))

    def test_target_at_end_is_pure_end_keyframe(self):
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator
        req = _bidi_request(target_index=10, start_idx=0, end_idx=10)
        result = BidirectionalInterpolationGenerator().generate(req)
        assert result.metadata.relative_position == pytest.approx(1.0)
        assert torch.allclose(result.frame, torch.ones(1, 3, 4, 4))

    def test_relative_position_scales_with_target_position(self):
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator
        gen = BidirectionalInterpolationGenerator()
        r_near = gen.generate(_bidi_request(target_index=2, start_idx=0, end_idx=10)).metadata.relative_position
        r_far = gen.generate(_bidi_request(target_index=8, start_idx=0, end_idx=10)).metadata.relative_position
        assert r_near == pytest.approx(0.2)
        assert r_far == pytest.approx(0.8)
        assert r_near < r_far

    def test_metadata_fields(self):
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator
        req = _bidi_request(target_index=3, start_idx=0, end_idx=6, caption="a caption", side_info={"delta": 1})
        meta = BidirectionalInterpolationGenerator().generate(req).metadata.to_dict()
        assert meta["backend"] == "bidirectional_interpolation"
        assert meta["conditioning_mode"] == "bidirectional"
        assert meta["source_keyframe_index"] == 0
        assert meta["end_keyframe_index"] == 6
        assert meta["target_indices"] == [3]
        assert meta["relative_position"] == pytest.approx(0.5)
        assert meta["used_caption"] is True
        assert meta["used_side_info"] is True
        assert meta["mock"] is True

    def test_missing_end_keyframe_raises_by_default(self):
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator, GenerationRequest
        req = GenerationRequest(
            start_keyframe_recon=torch.zeros(1, 3, 4, 4), start_keyframe_index=0, target_index=3,
            end_keyframe_recon=None, end_keyframe_index=None,
        )
        with pytest.raises(ValueError):
            BidirectionalInterpolationGenerator().generate(req)

    def test_missing_end_keyframe_falls_back_when_configured(self):
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator, GenerationRequest
        kf = torch.zeros(1, 3, 4, 4)
        req = GenerationRequest(
            start_keyframe_recon=kf, start_keyframe_index=0, target_index=3,
            end_keyframe_recon=None, end_keyframe_index=None,
        )
        gen = BidirectionalInterpolationGenerator(missing_end_policy="fallback_start_only")
        result = gen.generate(req)
        assert torch.equal(result.frame, kf)
        assert result.metadata.conditioning_mode == "start_only"
        assert result.metadata.relative_position is None
        assert "fallback_start_only" in result.metadata.notes

    def test_target_out_of_range_raises_by_default(self):
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator
        req = _bidi_request(target_index=15, start_idx=0, end_idx=10)   # 15 is past end_idx=10
        with pytest.raises(ValueError):
            BidirectionalInterpolationGenerator().generate(req)

    def test_target_out_of_range_falls_back_when_configured(self):
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator
        req = _bidi_request(target_index=15, start_idx=0, end_idx=10, start_val=0.3)
        gen = BidirectionalInterpolationGenerator(missing_end_policy="fallback_start_only")
        result = gen.generate(req)
        assert torch.allclose(result.frame, torch.full((1, 3, 4, 4), 0.3))
        assert result.metadata.conditioning_mode == "start_only"

    def test_invalid_missing_end_policy_rejected_at_construction(self):
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator
        with pytest.raises(ValueError):
            BidirectionalInterpolationGenerator(missing_end_policy="not_a_real_policy")

    def test_metadata_is_json_serialisable(self):
        import json
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator
        req = _bidi_request(target_index=5)
        json.dumps(BidirectionalInterpolationGenerator().generate(req).metadata.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# build_generator registry
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildGenerator:
    def test_default_backend_is_copy(self):
        from sgdjscc_lab.video.video_generator import build_generator, CopyGenerator
        gen = build_generator(OmegaConf.create({}))
        assert isinstance(gen, CopyGenerator)

    def test_explicit_copy_backend(self):
        from sgdjscc_lab.video.video_generator import build_generator, CopyGenerator
        cfg = OmegaConf.create({"video_generator": {"backend": "copy"}})
        assert isinstance(build_generator(cfg), CopyGenerator)

    def test_interpolation_backend_reads_alpha_and_gt_flag(self):
        from sgdjscc_lab.video.video_generator import build_generator, InterpolationGenerator
        cfg = OmegaConf.create({
            "video_generator": {
                "backend": "interpolation",
                "interpolation_alpha": 0.3,
                "allow_ground_truth_reference": True,
            }
        })
        gen = build_generator(cfg)
        assert isinstance(gen, InterpolationGenerator)
        assert gen.alpha == pytest.approx(0.3)
        assert gen.allow_ground_truth_reference is True

    def test_unknown_backend_raises_not_implemented(self):
        from sgdjscc_lab.video.video_generator import build_generator
        cfg = OmegaConf.create({"video_generator": {"backend": "svd"}})
        with pytest.raises(NotImplementedError):
            build_generator(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Segment-level generation contract (ETRI 후속 1단계 step 1A)
#
# SegmentGenerationRequest/Result are the GOP/segment-level counterpart of
# GenerationRequest/Result — see the module docstring's "Segment-level
# contract" section. These tests cover the contract itself (validation,
# JSON-serialisable metadata, the Rx-legal no-ground-truth-field boundary) and
# the base VideoGenerator.generate_segment() default fallback (which every
# backend in this module — Copy/Interpolation/BidirectionalInterpolation —
# inherits unmodified). TemporalPipeline-level batching (one call per GOP,
# mixed decisions, malformed-backend detection) is covered separately in
# tests/test_video.py.
# ─────────────────────────────────────────────────────────────────────────────

def _segment_request(target_indices, start_idx=0, end_idx=6, **kw):
    from sgdjscc_lab.video.video_generator import SegmentGenerationRequest
    kw.setdefault("start_keyframe_recon", torch.zeros(1, 3, 4, 4))
    kw.setdefault("start_keyframe_index", start_idx)
    return SegmentGenerationRequest(
        segment_id=0,
        start_frame_index=start_idx,
        end_frame_index=end_idx,
        target_indices=list(target_indices),
        **kw,
    )


class TestSegmentGenerationRequestRxLegal:
    """No ground-truth/original-frame field exists on the segment contract at
    all — unlike GenerationRequest's guarded reference_target_frame, there is
    no opt-in escape hatch to (mis)use here (see module docstring)."""

    def test_no_ground_truth_or_original_frame_field(self):
        import dataclasses
        from sgdjscc_lab.video.video_generator import SegmentGenerationRequest
        field_names = {f.name for f in dataclasses.fields(SegmentGenerationRequest)}
        assert "reference_target_frame" not in field_names
        assert not any("target_frame" in name for name in field_names)
        assert not any("ground_truth" in name for name in field_names)

    def test_segment_length_property(self):
        req = _segment_request([1, 2], start_idx=0, end_idx=6)
        assert req.segment_length == 7

    def test_segment_length_single_frame_gop(self):
        req = _segment_request([], start_idx=3, end_idx=3)
        assert req.segment_length == 1


class TestValidateSegmentRequest:
    def test_valid_request_passes(self):
        from sgdjscc_lab.video.video_generator import validate_segment_request
        validate_segment_request(_segment_request([1, 2, 3], end_idx=6))

    def test_empty_targets_is_valid(self):
        from sgdjscc_lab.video.video_generator import validate_segment_request
        validate_segment_request(_segment_request([], end_idx=6))

    def test_unsorted_targets_rejected(self):
        from sgdjscc_lab.video.video_generator import validate_segment_request
        with pytest.raises(ValueError):
            validate_segment_request(_segment_request([3, 1], end_idx=6))

    def test_duplicate_targets_rejected(self):
        from sgdjscc_lab.video.video_generator import validate_segment_request
        with pytest.raises(ValueError):
            validate_segment_request(_segment_request([1, 1, 2], end_idx=6))

    def test_target_outside_span_rejected(self):
        from sgdjscc_lab.video.video_generator import validate_segment_request
        with pytest.raises(ValueError):
            validate_segment_request(_segment_request([10], start_idx=0, end_idx=6))

    def test_target_equal_to_start_is_valid(self):
        from sgdjscc_lab.video.video_generator import validate_segment_request
        validate_segment_request(_segment_request([0], start_idx=0, end_idx=6))

    def test_mismatched_caption_length_rejected(self):
        from sgdjscc_lab.video.video_generator import validate_segment_request
        with pytest.raises(ValueError):
            validate_segment_request(_segment_request([1, 2], end_idx=6, captions=["a"]))

    def test_mismatched_packets_length_rejected(self):
        from sgdjscc_lab.video.video_generator import validate_segment_request
        with pytest.raises(ValueError):
            validate_segment_request(_segment_request([1, 2], end_idx=6, packets=[{"objects": []}]))

    def test_mismatched_side_infos_length_rejected(self):
        from sgdjscc_lab.video.video_generator import validate_segment_request
        with pytest.raises(ValueError):
            validate_segment_request(_segment_request([1, 2], end_idx=6, side_infos=[{"delta": 0.1}]))

    def test_end_keyframe_recon_without_index_rejected(self):
        from sgdjscc_lab.video.video_generator import validate_segment_request
        with pytest.raises(ValueError):
            validate_segment_request(
                _segment_request([1, 2], end_idx=6, end_keyframe_recon=torch.zeros(1, 3, 4, 4))
            )

    def test_end_keyframe_index_without_recon_rejected(self):
        from sgdjscc_lab.video.video_generator import validate_segment_request
        with pytest.raises(ValueError):
            validate_segment_request(_segment_request([1, 2], end_idx=6, end_keyframe_index=6))

    def test_end_frame_before_start_rejected(self):
        from sgdjscc_lab.video.video_generator import validate_segment_request
        with pytest.raises(ValueError):
            validate_segment_request(_segment_request([], start_idx=6, end_idx=0))


class TestSegmentGenerationResult:
    def test_frame_for_and_metadata_for_lookup_by_index(self):
        from sgdjscc_lab.video.video_generator import SegmentGenerationResult, GenerationMetadata
        f1, f2 = torch.zeros(1, 3, 4, 4), torch.ones(1, 3, 4, 4)
        m1 = GenerationMetadata(backend="copy", conditioning_mode="start_only", source_keyframe_index=0,
                                 target_indices=[1], used_caption=False, used_side_info=False, mock=True)
        m2 = GenerationMetadata(backend="copy", conditioning_mode="start_only", source_keyframe_index=0,
                                 target_indices=[2], used_caption=False, used_side_info=False, mock=True)
        result = SegmentGenerationResult(segment_id=0, target_indices=[1, 2], frames=[f1, f2], metadata=[m1, m2])
        assert torch.equal(result.frame_for(2), f2)
        assert result.metadata_for(1) is m1


class TestValidateSegmentResult:
    def _result(self, target_indices, frames=None, metadata=None):
        from sgdjscc_lab.video.video_generator import SegmentGenerationResult, GenerationMetadata
        frames = frames if frames is not None else [torch.zeros(1, 3, 4, 4) for _ in target_indices]
        metadata = metadata if metadata is not None else [
            GenerationMetadata(backend="copy", conditioning_mode="start_only", source_keyframe_index=0,
                                target_indices=[i], used_caption=False, used_side_info=False, mock=True)
            for i in target_indices
        ]
        return SegmentGenerationResult(segment_id=0, target_indices=list(target_indices),
                                        frames=frames, metadata=metadata)

    def test_matching_result_passes(self):
        from sgdjscc_lab.video.video_generator import validate_segment_result
        req = _segment_request([1, 2], end_idx=6)
        validate_segment_result(req, self._result([1, 2]))

    def test_wrong_target_indices_raises(self):
        from sgdjscc_lab.video.video_generator import validate_segment_result
        req = _segment_request([1, 2], end_idx=6)
        with pytest.raises(ValueError):
            validate_segment_result(req, self._result([1, 3]))

    def test_reordered_target_indices_raises(self):
        from sgdjscc_lab.video.video_generator import validate_segment_result
        req = _segment_request([1, 2], end_idx=6)
        with pytest.raises(ValueError):
            validate_segment_result(req, self._result([2, 1]))

    def test_wrong_frame_count_raises(self):
        from sgdjscc_lab.video.video_generator import validate_segment_result
        req = _segment_request([1, 2], end_idx=6)
        bad = self._result([1, 2])
        bad.frames = bad.frames[:1]
        with pytest.raises(ValueError):
            validate_segment_result(req, bad)

    def test_wrong_shape_raises(self):
        from sgdjscc_lab.video.video_generator import validate_segment_result
        req = _segment_request(
            [1, 2], end_idx=6, start_keyframe_recon=torch.zeros(1, 3, 4, 4),
        )
        bad = self._result([1, 2], frames=[torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 4, 4)])
        with pytest.raises(ValueError):
            validate_segment_result(req, bad)

    def test_wrong_segment_id_raises(self):
        from sgdjscc_lab.video.video_generator import validate_segment_result
        req = _segment_request([1, 2], end_idx=6)
        assert req.segment_id == 0
        bad = self._result([1, 2])
        bad.segment_id = 999
        with pytest.raises(ValueError):
            validate_segment_result(req, bad)

    def test_metadata_target_indices_mismatch_raises(self):
        from sgdjscc_lab.video.video_generator import validate_segment_result, GenerationMetadata
        req = _segment_request([1, 2], end_idx=6)
        bad = self._result([1, 2])
        # Second frame's metadata claims to describe frame 999, not frame 2.
        bad.metadata[1] = GenerationMetadata(
            backend="copy", conditioning_mode="start_only", source_keyframe_index=0,
            target_indices=[999], used_caption=False, used_side_info=False, mock=True,
        )
        with pytest.raises(ValueError):
            validate_segment_result(req, bad)

    def test_metadata_source_keyframe_index_mismatch_raises(self):
        from sgdjscc_lab.video.video_generator import validate_segment_result, GenerationMetadata
        req = _segment_request([1, 2], end_idx=6, start_keyframe_index=0)
        bad = self._result([1, 2])
        bad.metadata[0] = GenerationMetadata(
            backend="copy", conditioning_mode="start_only", source_keyframe_index=999,
            target_indices=[1], used_caption=False, used_side_info=False, mock=True,
        )
        with pytest.raises(ValueError):
            validate_segment_result(req, bad)

    def test_metadata_wrong_type_raises(self):
        from sgdjscc_lab.video.video_generator import validate_segment_result
        req = _segment_request([1, 2], end_idx=6)
        bad = self._result([1, 2])
        bad.metadata[0] = {"backend": "copy"}   # plain dict, not GenerationMetadata
        with pytest.raises(ValueError):
            validate_segment_result(req, bad)

    def test_metadata_unknown_conditioning_mode_raises(self):
        from sgdjscc_lab.video.video_generator import validate_segment_result, GenerationMetadata
        req = _segment_request([1, 2], end_idx=6)
        bad = self._result([1, 2])
        bad.metadata[0] = GenerationMetadata(
            backend="copy", conditioning_mode="sideways", source_keyframe_index=0,
            target_indices=[1], used_caption=False, used_side_info=False, mock=True,
        )
        with pytest.raises(ValueError):
            validate_segment_result(req, bad)


class TestDefaultGenerateSegmentFallback:
    """VideoGenerator.generate_segment()'s default implementation just loops
    over generate() once per target — proves every existing mock backend
    keeps working unmodified under the new contract (task requirement #3)."""

    def test_copy_generator_segment_matches_per_frame_semantics(self):
        from sgdjscc_lab.video.video_generator import CopyGenerator
        kf = torch.rand(1, 3, 4, 4)
        req = _segment_request([1, 2, 3], end_idx=4, start_keyframe_recon=kf, start_keyframe_index=0)
        result = CopyGenerator().generate_segment(req)
        assert result.target_indices == [1, 2, 3]
        assert len(result.frames) == 3
        for frame in result.frames:
            assert torch.equal(frame, kf)
        for meta in result.metadata:
            assert meta.backend == "copy"
            assert meta.conditioning_mode == "start_only"
            assert meta.mock is True

    def test_empty_targets_returns_empty_result_without_calling_generate(self):
        from sgdjscc_lab.video.video_generator import CopyGenerator, GenerationResult, GenerationMetadata
        calls = []

        class _SpyCopy(CopyGenerator):
            def generate(self, request):
                calls.append(request)
                return super().generate(request)

        req = _segment_request([], end_idx=4)
        result = _SpyCopy().generate_segment(req)
        assert result.frames == []
        assert result.target_indices == []
        assert calls == []

    def test_bidirectional_segment_blends_per_target_relative_position(self):
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator
        start = torch.zeros(1, 3, 4, 4)
        end = torch.ones(1, 3, 4, 4)
        req = _segment_request(
            [2, 5, 8], start_idx=0, end_idx=10,
            start_keyframe_recon=start, start_keyframe_index=0,
            end_keyframe_recon=end, end_keyframe_index=10,
        )
        result = BidirectionalInterpolationGenerator().generate_segment(req)
        assert result.target_indices == [2, 5, 8]
        assert torch.allclose(result.frames[0], torch.full((1, 3, 4, 4), 0.2))
        assert torch.allclose(result.frames[1], torch.full((1, 3, 4, 4), 0.5))
        assert torch.allclose(result.frames[2], torch.full((1, 3, 4, 4), 0.8))
        assert [m.relative_position for m in result.metadata] == pytest.approx([0.2, 0.5, 0.8])

    def test_start_only_backend_rejects_end_keyframe_in_segment_request(self):
        from sgdjscc_lab.video.video_generator import CopyGenerator
        req = _segment_request(
            [1], end_idx=6, end_keyframe_recon=torch.zeros(1, 3, 4, 4), end_keyframe_index=6,
        )
        with pytest.raises(NotImplementedError):
            CopyGenerator().generate_segment(req)

    def test_per_target_captions_packets_side_info_forwarded(self):
        from sgdjscc_lab.video.video_generator import CopyGenerator
        req = _segment_request(
            [1, 2], end_idx=4,
            captions=["a caption", None],
            packets=[{"objects": ["car"]}, None],
            side_infos=[{"delta": 0.1}, None],
        )
        result = CopyGenerator().generate_segment(req)
        assert result.metadata[0].used_caption is True
        assert result.metadata[1].used_caption is False
        assert result.metadata[0].used_side_info is True
        assert result.metadata[1].used_side_info is False

    def test_interpolation_backend_chains_prev_recon_within_one_call(self):
        from sgdjscc_lab.video.video_generator import InterpolationGenerator
        kf = torch.zeros(1, 3, 4, 4)
        prev = torch.full((1, 3, 4, 4), 1.0)
        req = _segment_request(
            [1, 2], end_idx=4, start_keyframe_recon=kf, reference_prev_recon=prev,
        )
        result = InterpolationGenerator(alpha=0.5).generate_segment(req)
        # target 1: blend(kf=0, prev=1.0) = 0.5
        assert torch.allclose(result.frames[0], torch.full((1, 3, 4, 4), 0.5))
        # target 2: blend(kf=0, prev=result[0]=0.5) = 0.25 — chained within the call.
        assert torch.allclose(result.frames[1], torch.full((1, 3, 4, 4), 0.25))

    def test_metadata_list_is_json_serialisable(self):
        import json
        from sgdjscc_lab.video.video_generator import CopyGenerator
        req = _segment_request([1, 2], end_idx=4, captions=["a", None])
        result = CopyGenerator().generate_segment(req)
        json.dumps([m.to_dict() for m in result.metadata])


# ─────────────────────────────────────────────────────────────────────────────
# save_generated_frames
# ─────────────────────────────────────────────────────────────────────────────

class _FakeRecord:
    def __init__(self, index, decision, recon):
        self.index = index
        self.decision = decision
        self.recon = recon


class TestSaveGeneratedFrames:
    def test_saves_only_generate_decision_frames(self, tmp_path):
        from sgdjscc_lab.video.video_generator import save_generated_frames
        records = [
            _FakeRecord(0, "keyframe", torch.rand(1, 3, 4, 4)),
            _FakeRecord(1, "generate", torch.rand(1, 3, 4, 4)),
            _FakeRecord(2, "reuse", torch.rand(1, 3, 4, 4)),
            _FakeRecord(3, "generate", torch.rand(1, 3, 4, 4)),
        ]
        saved = save_generated_frames(records, tmp_path)
        assert [p.name for p in saved] == ["generated_00001.png", "generated_00003.png"]
        assert all(p.exists() for p in saved)

    def test_skips_records_with_no_recon(self, tmp_path):
        from sgdjscc_lab.video.video_generator import save_generated_frames
        records = [_FakeRecord(0, "generate", None)]
        saved = save_generated_frames(records, tmp_path)
        assert saved == []

    def test_clears_stale_files_from_previous_run(self, tmp_path):
        from sgdjscc_lab.video.video_generator import save_generated_frames
        first = [_FakeRecord(i, "generate", torch.rand(1, 3, 4, 4)) for i in range(3)]
        save_generated_frames(first, tmp_path)
        assert len(list(tmp_path.glob("generated_*.png"))) == 3

        second = [_FakeRecord(0, "generate", torch.rand(1, 3, 4, 4))]
        saved = save_generated_frames(second, tmp_path)
        remaining = sorted(tmp_path.glob("generated_*.png"))
        assert len(remaining) == 1
        assert saved == remaining


# ─────────────────────────────────────────────────────────────────────────────
# ExternalSegmentWorkerGenerator (ETRI 후속 1단계 step 1B)
#
# All tests use the real scripts/lgvsc_generate_worker.py's `mock` backend
# (dependency-free — no GPU, no diffusers/Open-Sora/Wan) run as an ACTUAL
# subprocess via sys.executable (the same ptest interpreter these tests
# themselves run under — correct here only because the mock backend needs
# nothing beyond torch/torchvision/PIL, all already ptest dependencies).
# Error-path tests use tiny hand-written fake worker scripts (not the real
# one) so ExternalSegmentWorkerGenerator's OWN defensive checks — as opposed
# to the real worker's — are what's actually being exercised.
# ─────────────────────────────────────────────────────────────────────────────

def _worker_request(target_indices, start_idx=0, end_idx=None, with_end=False, **kw):
    from sgdjscc_lab.video.video_generator import SegmentGenerationRequest
    end_recon = torch.full((1, 3, 8, 8), 1.0) if with_end else None
    return SegmentGenerationRequest(
        segment_id=kw.pop("segment_id", 3),
        start_frame_index=start_idx,
        end_frame_index=(end_idx if end_idx is not None else max(target_indices)),
        target_indices=list(target_indices),
        start_keyframe_recon=torch.zeros(1, 3, 8, 8),
        start_keyframe_index=start_idx,
        end_keyframe_recon=end_recon,
        end_keyframe_index=(end_idx if with_end else None),
        **kw,
    )


class TestExternalSegmentWorkerGeneratorRoundTrip:
    def test_start_only_round_trip(self):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, validate_segment_result
        req = _worker_request([1, 2], end_idx=4, captions=["a red car", None],
                               packets=[{"objects": ["car"]}, None],
                               side_infos=[{"delta": 0.2}, None])
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, backend="mock", device="cpu")
        result = gen.generate_segment(req)
        validate_segment_result(req, result)
        assert result.target_indices == [1, 2]
        assert result.metadata_for(1).conditioning_mode == "start_only"
        assert result.metadata_for(1).used_caption is True
        assert result.metadata_for(2).used_caption is False

    def test_bidirectional_round_trip_matches_expected_blend(self):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, validate_segment_result
        req = _worker_request([2, 5, 8], start_idx=0, end_idx=10, with_end=True)
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, backend="mock", device="cpu")
        result = gen.generate_segment(req)
        validate_segment_result(req, result)
        assert result.metadata_for(2).relative_position == pytest.approx(0.2)
        assert result.metadata_for(5).relative_position == pytest.approx(0.5)
        assert result.metadata_for(8).relative_position == pytest.approx(0.8)
        # start=0.0, end=1.0 → target 5 (relative 0.5) should be ~0.5
        assert torch.allclose(result.frame_for(5), torch.full((1, 3, 8, 8), 0.5), atol=0.02)

    def test_seed_and_fps_forwarded_into_manifest(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator
        req = _worker_request([1], fps=29.97)
        gen = ExternalSegmentWorkerGenerator(
            python_bin=sys.executable, backend="mock", device="cpu", seed=4242,
            work_dir=str(tmp_path), cleanup_on_success=False,
        )
        gen.generate_segment(req)
        work_dirs = list(tmp_path.glob("lgvsc_seg*"))
        assert len(work_dirs) == 1
        manifest = json.loads((work_dirs[0] / "manifest.json").read_text())
        assert manifest["fps"] == pytest.approx(29.97)
        assert manifest["run_config"]["seed"] == 4242

    def test_cleanup_on_success_removes_work_dir(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator
        req = _worker_request([1])
        gen = ExternalSegmentWorkerGenerator(
            python_bin=sys.executable, backend="mock", work_dir=str(tmp_path), cleanup_on_success=True,
        )
        gen.generate_segment(req)
        assert list(tmp_path.glob("lgvsc_seg*")) == []

    def test_keep_work_dir_when_cleanup_disabled(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator
        req = _worker_request([1])
        gen = ExternalSegmentWorkerGenerator(
            python_bin=sys.executable, backend="mock", work_dir=str(tmp_path), cleanup_on_success=False,
        )
        gen.generate_segment(req)
        dirs = list(tmp_path.glob("lgvsc_seg*"))
        assert len(dirs) == 1
        assert (dirs[0] / "result.json").exists()

    def test_generate_raises_not_implemented(self):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, GenerationRequest
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable)
        req = GenerationRequest(start_keyframe_recon=torch.zeros(1, 3, 4, 4), start_keyframe_index=0, target_index=1)
        with pytest.raises(NotImplementedError):
            gen.generate(req)

    def test_requires_python_bin(self):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator
        with pytest.raises(ValueError):
            ExternalSegmentWorkerGenerator(python_bin="")


class TestExternalSegmentWorkerGeneratorManifestRxLegal:
    def test_manifest_has_no_original_frame_field_and_matches_request(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator
        req = _worker_request(
            [1, 2], start_idx=0, end_idx=6, with_end=True, fps=24.0,
            captions=["cap1", "cap2"], packets=[{"objects": ["car"]}, {"objects": ["bus"]}],
            side_infos=[{"delta": 0.1}, {"delta": 0.2}],
        )
        gen = ExternalSegmentWorkerGenerator(
            python_bin=sys.executable, backend="mock", seed=99,
            work_dir=str(tmp_path), cleanup_on_success=False,
        )
        gen.generate_segment(req)
        work_dir = next(tmp_path.glob("lgvsc_seg*"))
        manifest = json.loads((work_dir / "manifest.json").read_text())

        # No key anywhere in the manifest could carry an original/un-transmitted
        # target frame — SegmentGenerationRequest structurally has no such field.
        assert "reference_target_frame" not in manifest
        assert not any("target_frame" in k for k in manifest)
        assert not any("original" in k for k in manifest)

        assert manifest["segment_id"] == req.segment_id
        assert manifest["start_frame_index"] == req.start_frame_index
        assert manifest["end_frame_index"] == req.end_frame_index
        assert manifest["segment_length"] == req.segment_length
        assert manifest["target_indices"] == [1, 2]
        assert manifest["start_keyframe_index"] == req.start_keyframe_index
        assert manifest["end_keyframe_index"] == req.end_keyframe_index
        assert manifest["captions"] == ["cap1", "cap2"]
        assert manifest["packets"] == [{"objects": ["car"]}, {"objects": ["bus"]}]
        assert manifest["side_infos"] == [{"delta": 0.1}, {"delta": 0.2}]
        assert manifest["run_config"]["seed"] == 99
        assert (work_dir / manifest["start_keyframe_image"]).exists()
        assert (work_dir / manifest["end_keyframe_image"]).exists()

    def test_manifest_omits_end_keyframe_image_in_start_only_mode(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator
        req = _worker_request([1], start_idx=0, end_idx=3, with_end=False)
        gen = ExternalSegmentWorkerGenerator(
            python_bin=sys.executable, backend="mock",
            work_dir=str(tmp_path), cleanup_on_success=False,
        )
        gen.generate_segment(req)
        work_dir = next(tmp_path.glob("lgvsc_seg*"))
        manifest = json.loads((work_dir / "manifest.json").read_text())
        assert manifest["end_keyframe_image"] is None
        assert manifest["end_keyframe_index"] is None


class TestExternalSegmentWorkerGeneratorErrors:
    def _fake_worker(self, tmp_path, body: str) -> str:
        """Write a minimal standalone fake worker script (NOT the real
        lgvsc_generate_worker.py) so ExternalSegmentWorkerGenerator's own
        defensive checks in _read_result()/_run_worker() are what's under
        test, independent of the real worker's behaviour."""
        script = tmp_path / "fake_worker.py"
        script.write_text(
            "import argparse, json, sys, time\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--manifest')\n"
            "p.add_argument('--output-dir')\n"
            "args, _unknown = p.parse_known_args()\n"
            + body,
            encoding="utf-8",
        )
        return str(script)

    def test_timeout_raises_segment_worker_error(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentWorkerError
        script = self._fake_worker(tmp_path, "time.sleep(2.0)\nsys.exit(0)\n")
        req = _worker_request([1])
        gen = ExternalSegmentWorkerGenerator(
            python_bin=sys.executable, worker_script=script, timeout_sec=0.2,
        )
        with pytest.raises(SegmentWorkerError, match="timed out"):
            gen.generate_segment(req)

    def test_nonzero_exit_with_error_json_raises_with_message(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentWorkerError
        script = self._fake_worker(
            tmp_path,
            "import pathlib\n"
            "pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)\n"
            "(pathlib.Path(args.output_dir) / 'error.json').write_text(json.dumps("
            "{'status': 'error', 'error_type': 'FakeModelUnavailableError', "
            "'message': 'pretend weights missing', 'traceback': ''}))\n"
            "sys.exit(1)\n",
        )
        req = _worker_request([1])
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, worker_script=script)
        with pytest.raises(SegmentWorkerError, match="pretend weights missing"):
            gen.generate_segment(req)

    def test_nonzero_exit_without_error_json_raises_generic_message(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentWorkerError
        script = self._fake_worker(tmp_path, "sys.exit(1)\n")
        req = _worker_request([1])
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, worker_script=script)
        with pytest.raises(SegmentWorkerError, match="no error.json"):
            gen.generate_segment(req)

    def test_missing_result_json_raises(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentWorkerError
        script = self._fake_worker(
            tmp_path,
            "import pathlib\n"
            "pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)\n"
            "sys.exit(0)\n",
        )
        req = _worker_request([1])
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, worker_script=script)
        with pytest.raises(SegmentWorkerError, match="wrote no result.json"):
            gen.generate_segment(req)

    def test_wrong_frame_count_in_result_raises(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentWorkerError
        script = self._fake_worker(
            tmp_path,
            "import pathlib\n"
            "out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'result.json').write_text(json.dumps({'status': 'ok', 'segment_id': 3, "
            "'backend': 'fake', 'model_id': None, 'device': 'cpu', 'seed': None, "
            "'duration_sec': 0.0, 'target_indices': [1, 2], 'frames': {}, 'metadata': {}}))\n"
            "sys.exit(0)\n",
        )
        req = _worker_request([1, 2])
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, worker_script=script)
        with pytest.raises(SegmentWorkerError, match="returned frames for"):
            gen.generate_segment(req)

    def test_wrong_shape_in_result_raises(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentWorkerError
        script = self._fake_worker(
            tmp_path,
            "import pathlib\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)\n"
            "Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(out / 'frame_00001.png')\n"
            "(out / 'result.json').write_text(json.dumps({'status': 'ok', 'segment_id': 3, "
            "'backend': 'fake', 'model_id': None, 'device': 'cpu', 'seed': None, "
            "'duration_sec': 0.0, 'target_indices': [1], "
            "'frames': {'1': 'frame_00001.png'}, "
            "'metadata': {'1': {'backend': 'fake', 'conditioning_mode': 'start_only', "
            "'source_keyframe_index': 0, 'target_indices': [1], 'used_caption': False, "
            "'used_side_info': False, 'mock': True}}}))\n"
            "sys.exit(0)\n",
        )
        req = _worker_request([1])  # start_keyframe_recon is 8x8, fake worker returns 2x2
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, worker_script=script)
        with pytest.raises(SegmentWorkerError, match="with shape"):
            gen.generate_segment(req)

    def test_missing_frame_file_raises(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentWorkerError
        script = self._fake_worker(
            tmp_path,
            "import pathlib\n"
            "out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'result.json').write_text(json.dumps({'status': 'ok', 'segment_id': 3, "
            "'backend': 'fake', 'model_id': None, 'device': 'cpu', 'seed': None, "
            "'duration_sec': 0.0, 'target_indices': [1], "
            "'frames': {'1': 'does_not_exist.png'}, "
            "'metadata': {'1': {'backend': 'fake', 'conditioning_mode': 'start_only', "
            "'source_keyframe_index': 0, 'target_indices': [1], 'used_caption': False, "
            "'used_side_info': False, 'mock': True}}}))\n"
            "sys.exit(0)\n",
        )
        req = _worker_request([1])
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, worker_script=script)
        with pytest.raises(SegmentWorkerError, match="does not exist"):
            gen.generate_segment(req)

    def test_status_not_ok_raises(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentWorkerError
        script = self._fake_worker(
            tmp_path,
            "import pathlib\n"
            "out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'result.json').write_text(json.dumps({'status': 'weird'}))\n"
            "sys.exit(0)\n",
        )
        req = _worker_request([1])
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, worker_script=script)
        with pytest.raises(SegmentWorkerError, match="expected 'ok'"):
            gen.generate_segment(req)

    def test_malformed_metadata_raises_segment_worker_error(self, tmp_path):
        """A worker whose frame count/indices/shape all look fine but whose
        per-frame metadata lies about which frame it describes (wrong
        conditioning_mode/source_keyframe_index/target_indices) must still be
        rejected — generate_segment() must run the full 1A contract check
        itself, not rely on a caller invoking validate_segment_result()
        separately (TemporalPipeline does, but a direct call must not
        silently accept a bad result)."""
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentWorkerError
        script = self._fake_worker(
            tmp_path,
            "import pathlib\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)\n"
            "Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(out / 'frame_00001.png')\n"
            "(out / 'result.json').write_text(json.dumps({'status': 'ok', 'segment_id': 3, "
            "'backend': 'fake', 'model_id': None, 'device': 'cpu', 'seed': None, "
            "'duration_sec': 0.0, 'target_indices': [1], "
            "'frames': {'1': 'frame_00001.png'}, "
            "'metadata': {'1': {'backend': 'fake', 'conditioning_mode': 'sideways', "
            "'source_keyframe_index': 999, 'target_indices': [999], 'used_caption': False, "
            "'used_side_info': False, 'mock': True}}}))\n"
            "sys.exit(0)\n",
        )
        req = _worker_request([1])  # segment_id=3, start_keyframe_index=0
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, worker_script=script)
        with pytest.raises(SegmentWorkerError, match="failed contract validation"):
            gen.generate_segment(req)

    def test_wrong_segment_id_in_result_raises_segment_worker_error(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentWorkerError
        script = self._fake_worker(
            tmp_path,
            "import pathlib\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)\n"
            "Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(out / 'frame_00001.png')\n"
            "(out / 'result.json').write_text(json.dumps({'status': 'ok', 'segment_id': 999, "
            "'backend': 'fake', 'model_id': None, 'device': 'cpu', 'seed': None, "
            "'duration_sec': 0.0, 'target_indices': [1], "
            "'frames': {'1': 'frame_00001.png'}, "
            "'metadata': {'1': {'backend': 'fake', 'conditioning_mode': 'start_only', "
            "'source_keyframe_index': 0, 'target_indices': [1], 'used_caption': False, "
            "'used_side_info': False, 'mock': True}}}))\n"
            "sys.exit(0)\n",
        )
        req = _worker_request([1])  # segment_id=3, but result.json claims segment_id=999
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, worker_script=script)
        with pytest.raises(SegmentWorkerError, match="failed contract validation"):
            gen.generate_segment(req)

    def test_work_dir_kept_on_failure_regardless_of_cleanup_flag(self, tmp_path):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentWorkerError
        script = self._fake_worker(tmp_path, "sys.exit(1)\n")
        req = _worker_request([1])
        base = tmp_path / "base"
        base.mkdir()
        gen = ExternalSegmentWorkerGenerator(
            python_bin=sys.executable, worker_script=script,
            work_dir=str(base), cleanup_on_success=True,   # cleanup=True must NOT apply on failure
        )
        with pytest.raises(SegmentWorkerError):
            gen.generate_segment(req)
        assert list(base.glob("lgvsc_seg*"))   # work dir survives despite cleanup_on_success=True

    def test_bad_python_bin_raises_clear_launch_error(self):
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentWorkerError
        gen = ExternalSegmentWorkerGenerator(python_bin="/no/such/interpreter/here", backend="mock")
        with pytest.raises(SegmentWorkerError, match="Could not launch"):
            gen.generate_segment(_worker_request([1]))


class TestBuildGeneratorExternalSegmentWorker:
    def test_build_generator_constructs_worker_backend(self):
        from sgdjscc_lab.video.video_generator import build_generator, ExternalSegmentWorkerGenerator
        cfg = OmegaConf.create({
            "video_generator": {
                "backend": "external_segment_worker",
                "conditioning_mode": "bidirectional",  # ignored for this backend — no dispatch on it
                "worker": {
                    "python_bin": sys.executable,
                    "backend": "mock",
                    "device": "cpu",
                    "seed": 5,
                    "extra_env": {"PYTHONNOUSERSITE": "1"},
                },
            }
        })
        gen = build_generator(cfg)
        assert isinstance(gen, ExternalSegmentWorkerGenerator)
        assert gen.python_bin == sys.executable
        assert gen.backend == "mock"
        assert gen.seed == 5
        assert gen.extra_env == {"PYTHONNOUSERSITE": "1"}

    def test_build_generator_requires_python_bin_in_config(self):
        from sgdjscc_lab.video.video_generator import build_generator
        cfg = OmegaConf.create({"video_generator": {"backend": "external_segment_worker", "worker": {}}})
        with pytest.raises(ValueError, match="python_bin"):
            build_generator(cfg)

    def test_build_generator_constructs_wan_backend(self):
        """The worker.backend value is a plain passthrough string — this
        proves 'wan' specifically routes through config → build_generator()
        → ExternalSegmentWorkerGenerator → subprocess argv, matching what
        configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_start_only.yaml relies on."""
        from sgdjscc_lab.video.video_generator import build_generator, ExternalSegmentWorkerGenerator
        cfg = OmegaConf.create({
            "video_generator": {
                "backend": "external_segment_worker",
                "worker": {
                    "python_bin": "/some/env/bin/python",
                    "backend": "wan",
                    "model_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                    "device": "cuda:0",
                    "dtype": "bf16",
                    "extra_json": '{"offload_mode": "sequential"}',
                },
            }
        })
        gen = build_generator(cfg)
        assert isinstance(gen, ExternalSegmentWorkerGenerator)
        assert gen.backend == "wan"
        assert gen.model_id == "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        assert gen.dtype == "bf16"
        assert gen.extra_json == '{"offload_mode": "sequential"}'

        cmd = gen._build_command(Path("/tmp/manifest.json"), Path("/tmp/work"))
        assert "--backend" in cmd and cmd[cmd.index("--backend") + 1] == "wan"
        assert "--model-id" in cmd and cmd[cmd.index("--model-id") + 1] == gen.model_id
        assert "--extra-json" in cmd and cmd[cmd.index("--extra-json") + 1] == gen.extra_json
