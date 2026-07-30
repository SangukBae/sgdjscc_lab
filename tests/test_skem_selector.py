"""tests/test_skem_selector.py – src/sgdjscc_lab/video/skem_selector.py
(PsssKeyframeSelector) + its wiring into keyframe_extractor.py's factory and
TemporalPipeline/segment.py.

No real model weights anywhere: PSSS scores come from MockPsssBackend or a
tiny deterministic fake backend built for this file.
"""

from __future__ import annotations

import pytest
import torch

from sgdjscc_lab.video.psss import MockPsssBackend, PsssScoreResult
from sgdjscc_lab.video.skem_selector import PsssKeyframeSelector


class _ScriptedPsssBackend:
    """Returns S_rel values from a pre-scripted list, one per call, in order —
    lets a test dictate the EXACT sequence of keyframe-insertion decisions
    without depending on caption content at all."""

    backend_kind = "mock"
    backend_name = "scripted"
    model_id = None

    def __init__(self, s_rel_sequence):
        self._seq = list(s_rel_sequence)
        self._i = 0

    def score(self, info_a, info_b, semantic_focus="x") -> PsssScoreResult:
        s_rel = self._seq[self._i]
        self._i += 1
        p_no = (1.0 + s_rel) / 2.0
        p_yes = 1.0 - p_no
        return PsssScoreResult(
            info_a=info_a, info_b=info_b, semantic_focus=semantic_focus,
            p_yes=p_yes, p_no=p_no, p_yes_norm=p_yes, p_no_norm=p_no,
            s_abs=p_yes, s_rel=s_rel, backend=self.backend_name, backend_kind=self.backend_kind,
        )


def _caption_by_index(captions):
    def fn(frame, idx):
        return captions[idx]
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Basic PSSS-threshold behaviour + variable-length segments
# ─────────────────────────────────────────────────────────────────────────────

class TestVariableLengthKeyframeSelection:
    def test_variable_length_segments_from_scripted_scores(self):
        # 6 frames: keyframe 0, then scores evaluated at 1..5 (min_segment=1).
        # Scripted so keyframe 2 and keyframe 5 are inserted, giving segments
        # of length 2, 3, 1 — NOT all equal (proves variable-length, unlike a
        # fixed-interval selector).
        backend = _ScriptedPsssBackend([-1.0, 0.5, -1.0, -1.0, 0.5])
        sel = PsssKeyframeSelector(
            caption_fn=_caption_by_index(["x"] * 6), psss_backend=backend,
            threshold=0.3, min_segment_length=1, max_segment_length=None,
        )
        out = sel.extract([torch.zeros(1) for _ in range(6)])
        assert out["keyframes"] == [0, 2, 5]
        lengths = [1 + len(g["inter_frames"]) for g in out["gops"]]
        assert lengths == [2, 3, 1]
        assert len(set(lengths)) > 1  # genuinely variable, not fixed

    def test_threshold_boundary_is_strict_greater_than(self):
        backend = _ScriptedPsssBackend([0.35])  # exactly == threshold
        sel = PsssKeyframeSelector(
            caption_fn=_caption_by_index(["x", "y"]), psss_backend=backend,
            threshold=0.35, min_segment_length=1,
        )
        out = sel.extract([torch.zeros(1), torch.zeros(1)])
        assert out["keyframes"] == [0]  # NOT inserted: S_rel > threshold is strict
        assert out["psss_scores"][0]["decision"] == "continue_segment"

    def test_threshold_boundary_just_above_triggers_insertion(self):
        backend = _ScriptedPsssBackend([0.350001])
        sel = PsssKeyframeSelector(
            caption_fn=_caption_by_index(["x", "y"]), psss_backend=backend,
            threshold=0.35, min_segment_length=1,
        )
        out = sel.extract([torch.zeros(1), torch.zeros(1)])
        assert out["keyframes"] == [0, 1]
        assert out["psss_scores"][0]["decision"] == "new_keyframe"

    def test_first_frame_always_a_keyframe_no_psss_call(self):
        calls = []

        class _CountingBackend(_ScriptedPsssBackend):
            def score(self, info_a, info_b, semantic_focus="x"):
                calls.append((info_a, info_b))
                return super().score(info_a, info_b, semantic_focus)

        backend = _CountingBackend([-1.0])
        sel = PsssKeyframeSelector(caption_fn=_caption_by_index(["a", "b"]), psss_backend=backend, threshold=0.3)
        out = sel.extract([torch.zeros(1), torch.zeros(1)])
        assert out["keyframes"][0] == 0
        assert len(calls) == 1  # only frame 1 vs keyframe 0 was ever scored


# ─────────────────────────────────────────────────────────────────────────────
# min/max segment length enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestMinMaxSegmentLength:
    def test_min_segment_length_suppresses_early_new_keyframe(self):
        # Score sequence would insert a keyframe at frame 1 (S_rel=0.9 >
        # 0.3), but min_segment_length=3 forbids evaluating before span>=3.
        backend = _ScriptedPsssBackend([0.9])  # only consumed once span check passes
        sel = PsssKeyframeSelector(
            caption_fn=_caption_by_index(["x"] * 4), psss_backend=backend,
            threshold=0.3, min_segment_length=3,
        )
        out = sel.extract([torch.zeros(1) for _ in range(4)])
        # frame 1, 2 skipped (span 1, 2 < 3); frame 3 evaluated (span 3 >= 3)
        # and scores 0.9 > 0.3 -> new keyframe.
        assert out["keyframes"] == [0, 3]
        assert out["psss_scores"] == [{
            "index": 3, "compared_to_keyframe": 0, "s_abs": pytest.approx(0.05),
            "s_rel": pytest.approx(0.9), "p_yes": pytest.approx(0.05), "p_no": pytest.approx(0.95),
            "p_yes_norm": pytest.approx(0.05), "p_no_norm": pytest.approx(0.95),
            "threshold": 0.3, "backend": "scripted", "backend_kind": "mock",
            "clipped": False, "decision": "new_keyframe",
            # From PsssScoreResult.to_dict() (full provenance now preserved —
            # see skem_selector.py's record = score.to_dict()):
            "info_a": "x", "info_b": "x", "semantic_focus": "the main subject and action",
            "model_id": None, "proxy_of": None, "raw_logits": {}, "evidence": {}, "notes": "",
        }]

    def test_max_segment_length_forces_keyframe_without_psss_call(self):
        calls = []

        class _CountingBackend(_ScriptedPsssBackend):
            def score(self, info_a, info_b, semantic_focus="x"):
                calls.append(1)
                return super().score(info_a, info_b, semantic_focus)

        # Every score is far below threshold, so ONLY max_segment_length can
        # explain any keyframe after frame 0.
        backend = _CountingBackend([-1.0] * 10)
        sel = PsssKeyframeSelector(
            caption_fn=_caption_by_index(["x"] * 6), psss_backend=backend,
            threshold=0.9, min_segment_length=1, max_segment_length=3,
        )
        out = sel.extract([torch.zeros(1) for _ in range(6)])
        assert out["keyframes"] == [0, 3]
        lengths = [1 + len(g["inter_frames"]) for g in out["gops"]]
        assert lengths == [3, 3]
        # Frame 3 was a FORCED keyframe (max_segment_length) — no PSSS call
        # for it; frames 1,2,4,5 were evaluated (4 calls total).
        assert len(calls) == 4
        assert "max_segment_length" in out["keyframe_reasons"]["3"]
        assert "not a PSSS decision" in out["keyframe_reasons"]["3"] or \
               "NOT a PSSS decision" in out["keyframe_reasons"]["3"]

    def test_min_segment_length_must_be_at_least_1(self):
        with pytest.raises(ValueError):
            PsssKeyframeSelector(caption_fn=lambda f, i: "x", psss_backend=MockPsssBackend(), min_segment_length=0)

    def test_max_must_be_at_least_min(self):
        with pytest.raises(ValueError):
            PsssKeyframeSelector(
                caption_fn=lambda f, i: "x", psss_backend=MockPsssBackend(),
                min_segment_length=5, max_segment_length=2,
            )

    def test_min_segment_length_one_allows_adjacent_keyframes(self):
        backend = _ScriptedPsssBackend([0.9, 0.9, 0.9])
        sel = PsssKeyframeSelector(
            caption_fn=_caption_by_index(["x"] * 4), psss_backend=backend,
            threshold=0.3, min_segment_length=1,
        )
        out = sel.extract([torch.zeros(1) for _ in range(4)])
        assert out["keyframes"] == [0, 1, 2, 3]  # every frame becomes its own keyframe


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases: empty/tiny/truncated input
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_frame_list(self):
        sel = PsssKeyframeSelector(caption_fn=lambda f, i: "x", psss_backend=MockPsssBackend())
        out = sel.extract([])
        assert out["keyframes"] == [] and out["gops"] == [] and out["frame_roles"] == []
        assert out["selector"] == "psss"

    def test_single_frame(self):
        sel = PsssKeyframeSelector(caption_fn=lambda f, i: "x", psss_backend=MockPsssBackend())
        out = sel.extract([torch.zeros(1)])
        assert out["keyframes"] == [0]
        assert out["gops"] == [{"keyframe": 0, "inter_frames": [], "range": [0, 0]}]
        assert out["frame_roles"] == ["keyframe"]

    def test_truncated_clip_last_segment_shorter_than_min(self):
        # min_segment_length=5 but only 3 frames total after the keyframe —
        # the tail segment simply runs to the end, no crash, no forced pad.
        sel = PsssKeyframeSelector(
            caption_fn=_caption_by_index(["x"] * 3), psss_backend=MockPsssBackend(),
            threshold=0.3, min_segment_length=5,
        )
        out = sel.extract([torch.zeros(1) for _ in range(3)])
        assert out["keyframes"] == [0]
        assert out["gops"][0]["inter_frames"] == [1, 2]

    def test_adjacent_forced_keyframes_when_max_equals_min(self):
        sel = PsssKeyframeSelector(
            caption_fn=_caption_by_index(["x"] * 5), psss_backend=MockPsssBackend(),
            threshold=0.9999, min_segment_length=2, max_segment_length=2,
        )
        out = sel.extract([torch.zeros(1) for _ in range(5)])
        # Every segment forced to length 2 (except a possible short tail).
        lengths = [1 + len(g["inter_frames"]) for g in out["gops"]]
        assert lengths == [2, 2, 1]


# ─────────────────────────────────────────────────────────────────────────────
# FixedIntervalKeyframeSelector — literal SKIM (no scene-change signal at all)
# ─────────────────────────────────────────────────────────────────────────────

class TestFixedIntervalKeyframeSelector:
    def test_equal_length_segments_no_content_dependence(self):
        from sgdjscc_lab.video.keyframe_extractor import FixedIntervalKeyframeSelector

        sel = FixedIntervalKeyframeSelector(interval=4)
        # Frames are wildly different (wouldn't matter to a scene-change
        # extractor either way here) — selector must ignore content entirely.
        frames = [torch.full((1, 3, 4, 4), float(i)) for i in range(10)]
        out = sel.extract(frames)
        assert out["keyframes"] == [0, 4, 8]
        lengths = [1 + len(g["inter_frames"]) for g in out["gops"]]
        assert lengths == [4, 4, 2]  # last segment truncated, not padded
        assert out["selector"] == "fixed_interval"
        assert out["psss_backend_kind"] == "not_applicable"

    def test_identical_content_does_not_change_segmentation(self):
        """Sanity check that this really is content-independent (unlike
        KeyframeExtractor, which would react differently to identical vs.
        varying frames via its scene-change signal)."""
        from sgdjscc_lab.video.keyframe_extractor import FixedIntervalKeyframeSelector

        sel = FixedIntervalKeyframeSelector(interval=3)
        varying = [torch.full((1, 3, 4, 4), float(i)) for i in range(7)]
        identical = [torch.zeros(1, 3, 4, 4) for _ in range(7)]
        assert sel.extract(varying)["keyframes"] == sel.extract(identical)["keyframes"] == [0, 3, 6]

    def test_interval_must_be_positive(self):
        from sgdjscc_lab.video.keyframe_extractor import FixedIntervalKeyframeSelector
        with pytest.raises(ValueError):
            FixedIntervalKeyframeSelector(interval=0)

    def test_empty_and_single_frame(self):
        from sgdjscc_lab.video.keyframe_extractor import FixedIntervalKeyframeSelector
        sel = FixedIntervalKeyframeSelector(interval=5)
        assert sel.extract([])["keyframes"] == []
        out = sel.extract([torch.zeros(1)])
        assert out["keyframes"] == [0]
        assert out["gops"] == [{"keyframe": 0, "inter_frames": [], "range": [0, 0]}]

    def test_exact_multiple_has_no_short_tail(self):
        from sgdjscc_lab.video.keyframe_extractor import FixedIntervalKeyframeSelector
        sel = FixedIntervalKeyframeSelector(interval=2)
        out = sel.extract([torch.zeros(1) for _ in range(6)])
        assert out["keyframes"] == [0, 2, 4]
        lengths = [1 + len(g["inter_frames"]) for g in out["gops"]]
        assert lengths == [2, 2, 2]


class TestFixedCountKeyframeSelector:
    def test_exact_count_for_interval_unrepresentable_pair(self):
        from sgdjscc_lab.video.keyframe_extractor import FixedCountKeyframeSelector

        out = FixedCountKeyframeSelector(count=6).extract(
            [torch.zeros(1) for _ in range(10)]
        )
        assert out["keyframes"] == [0, 1, 3, 5, 6, 8]
        assert len(out["keyframes"]) == 6
        lengths = [1 + len(g["inter_frames"]) for g in out["gops"]]
        assert max(lengths) - min(lengths) <= 1

    def test_every_valid_frame_count_pair_is_exact(self):
        from sgdjscc_lab.video.keyframe_extractor import FixedCountKeyframeSelector

        for n_frames in range(1, 31):
            for count in range(1, n_frames + 1):
                out = FixedCountKeyframeSelector(count).extract([None] * n_frames)
                assert len(out["keyframes"]) == count
                assert len(set(out["keyframes"])) == count

    def test_rejects_more_keyframes_than_frames(self):
        from sgdjscc_lab.video.keyframe_extractor import FixedCountKeyframeSelector

        with pytest.raises(ValueError):
            FixedCountKeyframeSelector(6).extract([None] * 5)

    def test_factory_requires_and_uses_count(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.video.keyframe_extractor import (
            FixedCountKeyframeSelector,
            build_keyframe_extractor,
        )

        cfg = OmegaConf.create({
            "keyframe": {"selector": "fixed_count", "fixed_count": {"count": 6}},
        })
        selector = build_keyframe_extractor(cfg)
        assert isinstance(selector, FixedCountKeyframeSelector)
        assert selector.count == 6

        with pytest.raises(ValueError):
            build_keyframe_extractor(OmegaConf.create({
                "keyframe": {"selector": "fixed_count"},
            }))


# ─────────────────────────────────────────────────────────────────────────────
# keyframe_extractor.py factory wiring (build_keyframe_extractor / build_caption_fn)
# ─────────────────────────────────────────────────────────────────────────────

class TestKeyframeExtractorFactory:
    def test_default_selector_is_fixed_and_unaffected(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor, build_keyframe_extractor
        from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector

        cfg = OmegaConf.create({"keyframe": {"max_gop": 7}})
        extractor = build_keyframe_extractor(cfg, scene_detector=SceneChangeDetector())
        assert isinstance(extractor, KeyframeExtractor)

    def test_fixed_interval_selector_built_from_config(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.video.keyframe_extractor import FixedIntervalKeyframeSelector, build_keyframe_extractor

        cfg = OmegaConf.create({"keyframe": {"selector": "fixed_interval", "fixed_interval": {"interval": 5}}})
        extractor = build_keyframe_extractor(cfg)
        assert isinstance(extractor, FixedIntervalKeyframeSelector)
        assert extractor.interval == 5

    def test_fixed_interval_selector_defaults_to_max_gop(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.video.keyframe_extractor import build_keyframe_extractor

        cfg = OmegaConf.create({"keyframe": {"selector": "fixed_interval", "max_gop": 9}})
        extractor = build_keyframe_extractor(cfg)
        assert extractor.interval == 9

    def test_fixed_selector_without_scene_detector_raises(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.video.keyframe_extractor import build_keyframe_extractor
        with pytest.raises(ValueError):
            build_keyframe_extractor(OmegaConf.create({}))

    def test_psss_selector_built_from_config(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.video.keyframe_extractor import build_keyframe_extractor

        cfg = OmegaConf.create({
            "keyframe": {"selector": "psss", "psss": {
                "backend": "mock", "threshold": 0.4,
                "min_segment_length": 2, "max_segment_length": 6,
            }},
        })
        sel = build_keyframe_extractor(cfg, caption_fn=lambda f, i: "x")
        assert isinstance(sel, PsssKeyframeSelector)
        assert sel.threshold == 0.4
        assert sel.min_segment_length == 2
        assert sel.max_segment_length == 6
        assert sel.psss_backend.backend_kind == "mock"

    def test_psss_selector_without_caption_fn_raises(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.video.keyframe_extractor import build_keyframe_extractor
        cfg = OmegaConf.create({"keyframe": {"selector": "psss", "psss": {"backend": "mock"}}})
        with pytest.raises(ValueError):
            build_keyframe_extractor(cfg)

    def test_unknown_selector_raises(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.video.keyframe_extractor import build_keyframe_extractor
        cfg = OmegaConf.create({"keyframe": {"selector": "not-a-real-selector"}})
        with pytest.raises(NotImplementedError):
            build_keyframe_extractor(cfg)

    def test_build_caption_fn_captions_file(self):
        from sgdjscc_lab.video.keyframe_extractor import build_caption_fn
        fn = build_caption_fn("captions_file", captions=["a", "b", "c"])
        assert fn(torch.zeros(1), 1) == "b"
        assert fn(torch.zeros(1), 99) == ""  # out of range -> empty, no crash

    def test_build_caption_fn_captions_file_requires_captions(self):
        from sgdjscc_lab.video.keyframe_extractor import build_caption_fn
        with pytest.raises(ValueError):
            build_caption_fn("captions_file", captions=None)

    def test_build_caption_fn_model_requires_text_extractor(self):
        from sgdjscc_lab.video.keyframe_extractor import build_caption_fn
        with pytest.raises(ValueError):
            build_caption_fn("model", text_extractor=None)

    def test_build_caption_fn_mock_varies_with_frame_content(self):
        from sgdjscc_lab.video.keyframe_extractor import build_caption_fn
        fn = build_caption_fn("mock")
        a = fn(torch.zeros(3, 4, 4), 0)
        b = fn(torch.ones(3, 4, 4), 1)
        assert a != b  # genuinely reflects different frame content

    def test_build_caption_fn_unknown_source_raises(self):
        from sgdjscc_lab.video.keyframe_extractor import build_caption_fn
        with pytest.raises(NotImplementedError):
            build_caption_fn("not-a-real-source")


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: PsssKeyframeSelector driving TemporalPipeline, variable-length
# segments flowing through segment.py's keyframe_selection provenance, and
# the last (open) GOP's bidirectional-generate fallback.
# ─────────────────────────────────────────────────────────────────────────────

def _packet_fn(frame, fid):
    return {"caption": "x", "objects": [], "scene": None}


class TestTemporalPipelineWithPsssSelector:
    def _build_pipeline(self, backend, threshold=0.3, **kwargs):
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline

        captions = kwargs.pop("captions")
        reuse_threshold = kwargs.pop("reuse_threshold", 0.9)
        sel = PsssKeyframeSelector(
            caption_fn=_caption_by_index(captions), psss_backend=backend,
            threshold=threshold, min_segment_length=kwargs.pop("min_segment_length", 1),
            max_segment_length=kwargs.pop("max_segment_length", None),
        )
        return TemporalPipeline(
            reconstruct_fn=lambda frame, cfg: frame.clone(), packet_fn=_packet_fn,
            keyframe_extractor=sel, reuse_threshold=reuse_threshold, **kwargs,
        )

    def test_segments_carry_keyframe_selection_provenance(self):
        backend = _ScriptedPsssBackend([-1.0, 0.6, -1.0, -1.0])
        frames = [torch.full((1, 3, 4, 4), 0.1 * i) for i in range(5)]
        pipe = self._build_pipeline(backend, captions=["c"] * 5)
        res = pipe.run(frames)
        segs = res["segment_records"]
        assert [s["keyframe_index"] for s in segs] == [0, 2]
        assert segs[0]["keyframe_selection"]["reason"].startswith("first frame")
        assert segs[1]["keyframe_selection"]["psss_score"]["s_rel"] == pytest.approx(0.6)
        assert segs[1]["keyframe_selection"]["backend_kind"] == "mock"
        lengths = [1 + len(s["inter_frame_indices"]) for s in segs]
        assert lengths == [2, 3]
        # Full PsssScoreResult provenance (raw_logits/evidence/notes/model_id)
        # must survive all the way from PsssBackend.score() through
        # keyframe_extractor's psss_scores -> segment.py's keyframe_selection
        # -> segments.json, not just the handful of scalar fields.
        triggering_score = segs[1]["keyframe_selection"]["psss_score"]
        for key in ("raw_logits", "evidence", "notes", "model_id", "proxy_of"):
            assert key in triggering_score

    def test_fixed_selector_segments_have_no_keyframe_selection(self):
        from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor
        from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector, SceneChangeConfig
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline

        frames = [torch.full((1, 3, 4, 4), 0.1 * i) for i in range(3)]
        kfx = KeyframeExtractor(SceneChangeDetector(SceneChangeConfig(threshold=10.0)), max_gop=None)
        pipe = TemporalPipeline(
            reconstruct_fn=lambda f, c: f.clone(), packet_fn=_packet_fn,
            keyframe_extractor=kfx, reuse_threshold=0.9,
        )
        res = pipe.run(frames)
        # The key is OMITTED entirely (not present-with-null) — a
        # fixed-selector run's segments.json has the exact same key set as
        # before keyframe_selection existed.
        assert all("keyframe_selection" not in s for s in res["segment_records"])
        # The underlying SegmentRecord object still has the attribute
        # (always None) — only serialisation to_dict() omits it.
        assert all(seg.keyframe_selection is None for seg in res["segments"])

    def test_fixed_interval_selector_segments_DO_have_keyframe_selection(self):
        """Unlike the scene-change `fixed` selector above, `fixed_interval`
        DOES populate `keyframe_selection` — it sets `structure["selector"]`
        just like the PSSS selector does, so segment.py's
        `_keyframe_selection_summary` attaches provenance (with
        `backend_kind="not_applicable"` and no PSSS score/reason, since this
        selector never runs PSSS at all). This is intentional, documented
        behaviour, not something `to_dict()` should omit."""
        from sgdjscc_lab.video.keyframe_extractor import FixedIntervalKeyframeSelector
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline

        frames = [torch.full((1, 3, 4, 4), 0.1 * i) for i in range(6)]
        sel = FixedIntervalKeyframeSelector(interval=3)
        pipe = TemporalPipeline(
            reconstruct_fn=lambda f, c: f.clone(), packet_fn=_packet_fn,
            keyframe_extractor=sel, reuse_threshold=0.9,
        )
        res = pipe.run(frames)
        segs = res["segment_records"]
        assert all("keyframe_selection" in s for s in segs)
        for s in segs:
            ks = s["keyframe_selection"]
            assert ks["selector"] == "fixed_interval"
            assert ks["backend_kind"] == "not_applicable"
            assert ks["psss_score"] is None
            assert ks["reason"] is None

    def test_variable_length_generate_branch_and_last_gop_fallback(self):
        """Wires the PSSS selector's genuinely variable-length GOPs into the
        bidirectional generate branch, confirming: (1) each GOP's segment
        request reflects its own (variable) length/target_indices, (2) the
        LAST (open) GOP has no following keyframe and its request correctly
        carries end_keyframe_recon=None, and (3)
        BidirectionalInterpolationGenerator's fallback_start_only policy
        degrades that GOP's generation to conditioning_mode='start_only' —
        exactly the 'last open GOP -> start-only fallback' behaviour the
        skem_dsa_psss config relies on."""
        from sgdjscc_lab.video.video_generator import BidirectionalInterpolationGenerator

        backend = _ScriptedPsssBackend([-1.0, 0.6, -1.0, -1.0])   # keyframes at 0, 2 (of 5 frames)
        frames = [torch.full((1, 3, 4, 4), 0.1 * i) for i in range(5)]
        pipe = self._build_pipeline(
            backend, captions=["c"] * 5, reuse_threshold=0.0,   # force everything to generate
            enable_generate=True, conditioning_mode="bidirectional",
            video_generator=BidirectionalInterpolationGenerator(missing_end_policy="fallback_start_only"),
            generate_delta_min=0.0, generate_delta_max=1.0,
        )
        res = pipe.run(frames)
        segs = res["segment_records"]
        assert [s["keyframe_index"] for s in segs] == [0, 2]
        assert 1 + len(segs[0]["inter_frame_indices"]) == 2  # first GOP: variable length 2
        assert 1 + len(segs[1]["inter_frame_indices"]) == 3  # second (last, open) GOP: length 3

        first_gen = segs[0]["generation"]
        last_gen = segs[1]["generation"]
        assert first_gen["conditioning_mode"] == "bidirectional"
        assert first_gen["end_keyframe_index"] == 2
        # Last GOP has no following keyframe -> fell back to start_only, and
        # says so honestly rather than reporting bidirectional.
        assert last_gen["conditioning_mode"] == "start_only"
        assert last_gen["end_keyframe_index"] is None

    def test_rx_legal_segment_request_has_no_ground_truth_field(self):
        """SegmentGenerationRequest (the contract driving ANY selector's
        GOPs, including the PSSS one) structurally has no field a backend
        could use to read the un-transmitted original target frame."""
        from sgdjscc_lab.video.video_generator import SegmentGenerationRequest
        field_names = set(SegmentGenerationRequest.__dataclass_fields__.keys())
        assert not any("target_frame" in f or "ground_truth" in f or "original" in f for f in field_names)
