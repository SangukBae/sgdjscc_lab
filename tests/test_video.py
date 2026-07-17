"""tests/test_video.py – Phase 4-B keyframe/temporal tests (offline).

Uses synthetic frame tensors and mock reconstruct/packet functions; no CLIP,
BLIP2 or SGD-JSCC checkpoints required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.guidance.semantic_packet_extractor import build_packet  # noqa: E402


def _two_scene_frames(per_scene=3):
    """Return 2×per_scene frames: a reddish scene then a bluish scene."""
    torch.manual_seed(0)
    red = torch.zeros(1, 3, 32, 32); red[:, 0] = 0.8
    blue = torch.zeros(1, 3, 32, 32); blue[:, 2] = 0.8
    frames = [(red + 0.01 * torch.randn_like(red)).clamp(0, 1) for _ in range(per_scene)]
    frames += [(blue + 0.01 * torch.randn_like(blue)).clamp(0, 1) for _ in range(per_scene)]
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# scene change detection
# ─────────────────────────────────────────────────────────────────────────────

class TestSceneChangeDetector:
    def test_histogram_distance_range(self):
        from sgdjscc_lab.video.scene_change_detector import histogram_distance
        red = torch.zeros(1, 3, 16, 16); red[:, 0] = 0.8
        blue = torch.zeros(1, 3, 16, 16); blue[:, 2] = 0.8
        assert histogram_distance(red, red) == pytest.approx(0.0, abs=1e-6)
        assert histogram_distance(red, blue) > 0.5

    def test_detect_marks_boundaries(self):
        from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector, SceneChangeConfig
        frames = _two_scene_frames(3)
        det = SceneChangeDetector(SceneChangeConfig(threshold=0.35))
        out = det.detect(frames)
        assert out["boundaries"][0] is True       # first frame always a boundary
        assert out["boundaries"][3] is True       # cut between scenes
        assert out["boundaries"][1] is False

    def test_empty_sequence(self):
        from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector
        out = SceneChangeDetector().detect([])
        assert out["boundaries"] == []


# ─────────────────────────────────────────────────────────────────────────────
# keyframe extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestKeyframeExtractor:
    def test_extract_from_boundaries(self):
        from sgdjscc_lab.video.keyframe_extractor import extract_keyframes
        boundaries = [True, False, False, True, False]
        out = extract_keyframes(boundaries, max_gop=None)
        assert out["keyframes"] == [0, 3]
        assert out["frame_roles"] == ["keyframe", "inter", "inter", "keyframe", "inter"]

    def test_max_gop_forces_refresh(self):
        from sgdjscc_lab.video.keyframe_extractor import extract_keyframes
        boundaries = [True, False, False, False, False]
        out = extract_keyframes(boundaries, max_gop=2)
        # keyframe at 0, then forced every 2 frames: 0, 2, 4
        assert out["keyframes"] == [0, 2, 4]

    def test_gops_cover_all_frames(self):
        from sgdjscc_lab.video.keyframe_extractor import extract_keyframes
        out = extract_keyframes([True, False, True, False], max_gop=None)
        covered = []
        for g in out["gops"]:
            covered.append(g["keyframe"])
            covered.extend(g["inter_frames"])
        assert sorted(covered) == [0, 1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# semantic delta
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticDelta:
    def test_new_object(self):
        from sgdjscc_lab.video.semantic_delta import compute_delta
        ref = build_packet(objects=["car"], scene="street scene")
        cur = build_packet(objects=["car", "dog"], scene="street scene")
        d = compute_delta(ref, cur)
        assert "dog" in d["new_objects"]
        assert d["removed_objects"] == []
        assert d["magnitude"] > 0.0

    def test_identical_zero_magnitude(self):
        from sgdjscc_lab.video.semantic_delta import compute_delta
        p = build_packet(objects=["car"], scene="street scene")
        d = compute_delta(p, p)
        assert d["magnitude"] == pytest.approx(0.0)
        assert d["is_empty"] is True

    def test_scene_change_flag(self):
        from sgdjscc_lab.video.semantic_delta import compute_delta
        ref = build_packet(objects=["car"], scene="street scene")
        cur = build_packet(objects=["car"], scene="beach")
        assert compute_delta(ref, cur)["scene_changed"] is True


# ─────────────────────────────────────────────────────────────────────────────
# motion residual
# ─────────────────────────────────────────────────────────────────────────────

class TestMotionResidual:
    def test_zero_for_identical(self):
        from sgdjscc_lab.video.motion_residual import residual_energy
        f = torch.rand(1, 3, 16, 16)
        assert residual_energy(f, f) == pytest.approx(0.0, abs=1e-6)

    def test_estimate_keys(self):
        from sgdjscc_lab.video.motion_residual import estimate
        a = torch.zeros(1, 3, 16, 16)
        b = torch.ones(1, 3, 16, 16)
        out = estimate(a, b)
        assert set(out.keys()) == {"residual_energy", "block_mean", "block_max", "block_map"}
        assert out["residual_energy"] == pytest.approx(1.0, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# staged schedule
# ─────────────────────────────────────────────────────────────────────────────

class TestStagedSchedule:
    def test_three_stages_cumulative_prompt(self):
        from sgdjscc_lab.video.temporal_pipeline import build_staged_schedule
        p = build_packet(
            caption="a red car next to a dog",
            objects=["car", "dog"], scene="street scene",
            relations=[{"subject": "car", "predicate": "next to", "object": "dog"}],
            attributes={"car": ["red"]},
        )
        sched = build_staged_schedule(p, diffusion_step=50)
        names = [s["name"] for s in sched["stages"]]
        assert names == ["early", "middle", "late"]
        # cumulative: late prompt is the longest / contains attributes
        assert len(sched["stages"][2]["prompt"]) >= len(sched["stages"][0]["prompt"])
        assert "red car" in sched["final_prompt"]


# ─────────────────────────────────────────────────────────────────────────────
# temporal pipeline I/O + overhead reduction
# ─────────────────────────────────────────────────────────────────────────────

def _packet_fn(frame, fid):
    red = frame[:, 0].mean() > frame[:, 2].mean()
    return build_packet(
        caption="a red car" if red else "a blue boat",
        objects=["car"] if red else ["boat"],
        scene="street scene" if red else "beach",
    )


class TestTemporalPipeline:
    def _pipeline(self, reuse_threshold=0.2):
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
        from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector, SceneChangeConfig
        return TemporalPipeline(
            reconstruct_fn=lambda frame, cfg: frame.clone(),
            packet_fn=_packet_fn,
            scene_detector=SceneChangeDetector(SceneChangeConfig(threshold=0.35)),
            reuse_threshold=reuse_threshold,
        )

    def test_run_returns_expected_structure(self):
        frames = _two_scene_frames(3)
        res = self._pipeline().run(frames)
        assert set(res.keys()) == {
            "frame_records", "keyframe_structure", "records",
            "segments", "segment_records", "summary",
        }
        assert len(res["frame_records"]) == 6

    def test_two_keyframes_detected(self):
        frames = _two_scene_frames(3)
        res = self._pipeline().run(frames)
        assert res["summary"]["n_keyframes"] == 2

    def test_overhead_reduction_positive(self):
        frames = _two_scene_frames(3)
        res = self._pipeline().run(frames)
        # keyframe + delta transmits fewer units than naive per-frame full packets
        assert res["summary"]["overhead_reduction"] > 0.0
        assert res["summary"]["transmitted_units"] < res["summary"]["naive_units"]

    def test_roles_consistent_with_structure(self):
        frames = _two_scene_frames(3)
        res = self._pipeline().run(frames)
        roles = [r["role"] for r in res["frame_records"]]
        assert roles.count("keyframe") == res["summary"]["n_keyframes"]


class _StubDetector:
    """Scene detector returning a fixed boundary list."""

    def __init__(self, boundaries):
        self.boundaries = boundaries

    def detect(self, frames):
        return {"boundaries": self.boundaries, "distances": [0.0] * len(frames)}


class TestTemporalReferenceConsistency:
    """Regression: inter-frame reuse must reference the keyframe, not a previously
    recomputed inter-frame (consistent packet + pixel reference)."""

    def _run(self):
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
        from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor

        # 4 frames, single keyframe at index 0.
        obj_map = {
            0: ["car"],                                  # keyframe
            1: ["car", "dog", "cat", "tree", "bus"],     # big change → recompute
            2: ["car"],                                  # back to keyframe → reuse
            3: ["car"],                                  # reuse
        }
        frames = [torch.full((1, 3, 8, 8), 0.1 * (i + 1)) for i in range(4)]

        def packet_fn(frame, fid):
            if str(fid).startswith("frame_"):
                idx = int(str(fid).split("_")[1])
                objs = obj_map[idx]
            else:
                objs = ["car"]
            return build_packet(objects=objs, scene="s")

        def recon_fn(frame, cfg):
            return frame * 10.0   # content-deterministic reconstruction

        kfx = KeyframeExtractor(_StubDetector([True, False, False, False]), max_gop=None)
        pipe = TemporalPipeline(
            reconstruct_fn=recon_fn, packet_fn=packet_fn,
            keyframe_extractor=kfx, reuse_threshold=0.2,
        )
        return pipe.run(frames)

    def test_reused_frame_references_keyframe(self):
        res = self._run()
        recs = res["records"]
        assert recs[1].role == "inter" and recs[1].reused is False   # recomputed
        assert recs[2].reused is True                                 # reused
        # The reused recon must be the KEYFRAME reconstruction…
        assert torch.equal(recs[2].recon, recs[0].recon)
        # …and NOT the recomputed inter-frame's reconstruction.
        assert not torch.equal(recs[2].recon, recs[1].recon)

    def test_delta_reference_is_keyframe(self):
        res = self._run()
        recs = res["records"]
        # Frame 3 delta is computed vs the keyframe packet (objects ['car']),
        # so a frame identical to the keyframe yields an empty delta.
        assert recs[3].delta["is_empty"] is True


class TestStagedPromptWiring:
    """The staged schedule must actually condition reconstruction via prompt_override."""

    def test_prompt_override_passed_to_reconstruct(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
        from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor

        seen = {}

        def recon_fn(frame, cfg):
            seen["prompt_override"] = cfg.get("prompt_override")
            seen["staged_prompts"] = cfg.get("staged_prompts")
            return frame.clone()

        def packet_fn(frame, fid):
            return build_packet(
                caption="a red car", objects=["car"], scene="street scene",
                attributes={"car": ["red"]},
            )

        cfg = OmegaConf.create({"guidance_scale": 4.0, "use_text": True, "diffusion_step": 30})
        kfx = KeyframeExtractor(_StubDetector([True]), max_gop=None)
        pipe = TemporalPipeline(
            reconstruct_fn=recon_fn, packet_fn=packet_fn,
            keyframe_extractor=kfx, cfg=cfg,
        )
        pipe.run([torch.rand(1, 3, 8, 8)])
        assert seen["prompt_override"]                    # non-empty staged prompt
        assert "car" in seen["prompt_override"]
        assert len(seen["staged_prompts"]) == 3           # early / middle / late

    def test_diffusion_step_synced_between_schedule_and_cfg(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
        from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor

        seen = {}

        def recon_fn(frame, cfg):
            seen["diffusion_step"] = cfg.get("diffusion_step")
            return frame.clone()

        def packet_fn(frame, fid):
            return build_packet(caption="a red car", objects=["car"], scene="street scene")

        # cfg says 50, but the constructor explicitly overrides to 20.
        cfg = OmegaConf.create({"guidance_scale": 4.0, "use_text": True, "diffusion_step": 50})
        kfx = KeyframeExtractor(_StubDetector([True]), max_gop=None)
        pipe = TemporalPipeline(
            reconstruct_fn=recon_fn, packet_fn=packet_fn,
            keyframe_extractor=kfx, cfg=cfg, diffusion_step=20,
        )
        res = pipe.run([torch.rand(1, 3, 8, 8)])
        # Reconstruction cfg uses the authoritative step count…
        assert seen["diffusion_step"] == 20
        # …and the staged schedule's stage boundaries are built for the same 20.
        last_stage = res["records"][0].staged_schedule["stages"][-1]
        assert last_stage["step_range"][1] == 20

    def test_diffusion_step_defaults_to_cfg(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
        from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor

        cfg = OmegaConf.create({"guidance_scale": 4.0, "use_text": True, "diffusion_step": 35})
        kfx = KeyframeExtractor(_StubDetector([True]), max_gop=None)
        pipe = TemporalPipeline(
            reconstruct_fn=lambda f, c: f.clone(),
            packet_fn=lambda f, i: build_packet(objects=["car"], scene="s"),
            keyframe_extractor=kfx, cfg=cfg,   # no explicit diffusion_step
        )
        assert pipe.diffusion_step == 35


# ─────────────────────────────────────────────────────────────────────────────
# temporal consistency metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalConsistency:
    def test_evaluate_sequence_keys(self):
        from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
        records = [
            {"srs": 0.8, "orig_packet": build_packet(objects=["car"]),
             "recon_packet": build_packet(objects=["car"])},
            {"srs": 0.7, "orig_packet": build_packet(objects=["car"]),
             "recon_packet": build_packet(objects=["car", "dog"])},
        ]
        out = evaluate_sequence(records)
        for k in ("n_frames", "temporal_srs", "srs_flicker",
                  "object_identity_consistency", "temporal_hallucination_rate"):
            assert k in out

    def test_hallucination_detected(self):
        from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
        records = [
            {"orig_packet": build_packet(objects=["car"]),
             "recon_packet": build_packet(objects=["car", "dog"])},
        ]
        out = evaluate_sequence(records)
        assert out["temporal_hallucination_rate"] > 0.0

    def test_accepts_frame_record_objects(self):
        from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
        from sgdjscc_lab.video.temporal_pipeline import FrameRecord
        recs = [
            FrameRecord(index=0, role="keyframe", srs=0.9,
                        orig_packet=build_packet(objects=["car"]),
                        recon_packet=build_packet(objects=["car"])),
        ]
        out = evaluate_sequence(recs)
        assert out["temporal_srs"] == pytest.approx(0.9)


# ─────────────────────────────────────────────────────────────────────────────
# ETRI 1차 time-axis metrics: PTC / SFR / SDI (provisional, packet-based)
# ─────────────────────────────────────────────────────────────────────────────

class TestPTCSFRSDI:
    def _rec(self, role, orig_objs, recon_objs):
        return {
            "role": role,
            "orig_packet": build_packet(objects=orig_objs, scene="s"),
            "recon_packet": build_packet(objects=recon_objs, scene="s"),
        }

    def test_keys_present(self):
        from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
        out = evaluate_sequence([self._rec("keyframe", ["car"], ["car"])])
        for k in ("ptc", "sfr", "sdi"):
            assert k in out

    def test_ptc_perfect_match_is_one(self):
        from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
        recs = [
            self._rec("keyframe", ["car"], ["car"]),
            self._rec("inter", ["car"], ["car"]),
        ]
        out = evaluate_sequence(recs)
        assert out["ptc"] == pytest.approx(1.0)
        assert out["sfr"] == pytest.approx(0.0)

    def test_ptc_drops_on_packet_mismatch(self):
        from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
        recs = [
            self._rec("keyframe", ["car", "tree"], ["car", "tree"]),
            self._rec("inter", ["car", "tree"], ["car"]),          # missing tree
        ]
        out = evaluate_sequence(recs)
        assert out["ptc"] < 1.0

    def test_sfr_counts_spurious_birth(self):
        from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
        # Original object set is constant, but the recon makes a dog appear then
        # vanish → pure flicker.
        recs = [
            self._rec("keyframe", ["car"], ["car"]),
            self._rec("inter", ["car"], ["car", "dog"]),
            self._rec("inter", ["car"], ["car"]),
        ]
        out = evaluate_sequence(recs)
        assert out["sfr"] > 0.0

    def test_sfr_ignores_genuine_scene_change(self):
        from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
        # The dog genuinely appears in the original too → not flicker.
        recs = [
            self._rec("keyframe", ["car"], ["car"]),
            self._rec("inter", ["car", "dog"], ["car", "dog"]),
        ]
        out = evaluate_sequence(recs)
        assert out["sfr"] == pytest.approx(0.0)

    def test_sdi_positive_when_drifting_from_keyframe(self):
        from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
        # Recon packet loses more objects the further the frame is from its
        # keyframe → drift grows with keyframe distance → positive slope.
        recs = [
            self._rec("keyframe", ["car", "tree", "dog", "bus"], ["car", "tree", "dog", "bus"]),
            self._rec("inter", ["car", "tree", "dog", "bus"], ["car", "tree", "dog"]),
            self._rec("inter", ["car", "tree", "dog", "bus"], ["car", "tree"]),
            self._rec("inter", ["car", "tree", "dog", "bus"], ["car"]),
        ]
        out = evaluate_sequence(recs)
        assert out["sdi"] is not None and out["sdi"] > 0.0

    def test_sdi_zero_when_no_drift(self):
        from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
        recs = [
            self._rec("keyframe", ["car"], ["car"]),
            self._rec("inter", ["car"], ["car"]),
            self._rec("inter", ["car"], ["car"]),
        ]
        out = evaluate_sequence(recs)
        assert out["sdi"] == pytest.approx(0.0)

    def test_sdi_none_without_roles(self):
        from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
        recs = [
            {"orig_packet": build_packet(objects=["car"]),
             "recon_packet": build_packet(objects=["car"])},
        ]
        assert evaluate_sequence(recs)["sdi"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Motion gate (semantic delta + motion dual gate)
# ─────────────────────────────────────────────────────────────────────────────

class TestMotionGate:
    """Inter-frames with an unchanged packet but large pixel motion must not be
    reused when the motion gate is enabled (docs/etri_strategy.md 순서 3)."""

    def _run(self, motion_threshold):
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
        from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor

        # Frame 1 has identical semantics (same packet) but big pixel change.
        frames = [torch.zeros(1, 3, 16, 16), torch.full((1, 3, 16, 16), 0.9)]

        def packet_fn(frame, fid):
            return build_packet(objects=["car"], scene="street scene")

        kfx = KeyframeExtractor(_StubDetector([True, False]), max_gop=None)
        pipe = TemporalPipeline(
            reconstruct_fn=lambda f, c: f.clone(),
            packet_fn=packet_fn,
            keyframe_extractor=kfx,
            reuse_threshold=0.2,
            motion_threshold=motion_threshold,
        )
        return pipe.run(frames)

    def test_default_no_motion_gate_reuses(self):
        res = self._run(motion_threshold=None)
        rec = res["records"][1]
        assert rec.reused is True
        assert rec.decision == "reuse"
        assert rec.motion_score is None            # gate off → no motion computed

    def test_high_motion_not_reused(self):
        res = self._run(motion_threshold=0.1)
        rec = res["records"][1]
        assert rec.reused is False
        assert rec.decision == "recompute_motion"
        assert rec.motion_score is not None and rec.motion_score >= 0.1
        assert res["summary"]["n_recompute_motion"] == 1

    def test_low_motion_still_reuses(self):
        # Threshold above the actual motion (≈0.9) → gate passes → reuse.
        res = self._run(motion_threshold=5.0)
        rec = res["records"][1]
        assert rec.reused is True
        assert rec.decision == "reuse"
        assert rec.motion_score is not None        # gate on → motion logged

    def test_motion_fields_in_frame_log(self):
        res = self._run(motion_threshold=0.1)
        log = res["frame_records"][1]
        for k in ("decision", "motion_score", "motion_residual", "motion_block_max"):
            assert k in log


# ─────────────────────────────────────────────────────────────────────────────
# Segment abstraction (GOP/segment records)
# ─────────────────────────────────────────────────────────────────────────────

class TestSegmentRecords:
    def _run(self):
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
        from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector, SceneChangeConfig
        pipe = TemporalPipeline(
            reconstruct_fn=lambda frame, cfg: frame.clone(),
            packet_fn=_packet_fn,
            scene_detector=SceneChangeDetector(SceneChangeConfig(threshold=0.35)),
            reuse_threshold=0.2,
        )
        return pipe.run(_two_scene_frames(3))

    def test_one_segment_per_keyframe(self):
        res = self._run()
        assert len(res["segment_records"]) == res["summary"]["n_keyframes"]

    def test_segments_cover_all_frames(self):
        res = self._run()
        covered = []
        for seg in res["segment_records"]:
            covered.append(seg["keyframe_index"])
            covered.extend(seg["inter_frame_indices"])
        assert sorted(covered) == list(range(res["summary"]["n_frames"]))

    def test_segment_record_schema(self):
        res = self._run()
        seg = res["segment_records"][0]
        for k in ("segment_id", "keyframe_index", "inter_frame_indices",
                  "frame_decisions", "transmitted_units", "semantic_delta",
                  "motion", "temporal_metrics", "generation"):
            assert k in seg
        # Generate branch is a reserved interface in the 1차 scope.
        assert seg["generation"] is None
        # Decisions align with the segment's frames.
        assert [d["index"] for d in seg["frame_decisions"]] == \
            [seg["keyframe_index"]] + seg["inter_frame_indices"]

    def test_segment_units_sum_to_summary(self):
        res = self._run()
        total = sum(s["transmitted_units"] for s in res["segment_records"])
        assert total == res["summary"]["transmitted_units"]

    def test_segment_json_serialisable(self):
        import json
        res = self._run()
        json.dumps(res["segment_records"])   # must not raise
