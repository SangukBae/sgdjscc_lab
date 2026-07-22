"""tests/test_video_generator.py – video/video_generator.py backend tests (offline).

Covers the ETRI 3차 start-only generate-branch backend interface in isolation
(GenerationRequest/Result/Metadata, CopyGenerator, InterpolationGenerator,
build_generator registry, the Rx-legal ground-truth-reference boundary, the
bidirectional-conditioning NotImplementedError guard, and save_generated_frames).
TemporalPipeline-level 3-way decision wiring is covered separately in
tests/test_video.py::TestGenerateBranch.
"""

from __future__ import annotations

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
