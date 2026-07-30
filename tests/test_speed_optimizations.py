"""tests/test_speed_optimizations.py – ETRI video-eval speed-up features (offline).

Covers the additions made to diagnose/reduce real-model video evaluation
runtime: utils/profiling.py (call-count + progress instrumentation),
video/temporal_pipeline.py's force_interframe_reuse (keyframe-only real-model
mode), guidance/packet_cache.py (original-frame packet disk cache), the CLIP
text-embedding memoization in evaluators/clip_score.py, and
utils/gpu_logger.py. No GPU / SGD-JSCC checkpoints required; the CLIP test
uses the real (small) ViT-B/32 weights already cached locally by prior runs
and runs on CPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.guidance.semantic_packet_extractor import build_packet  # noqa: E402
from sgdjscc_lab.utils import profiling  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# utils/profiling.py
# ─────────────────────────────────────────────────────────────────────────────

class TestProfilingCounters:
    def teardown_method(self):
        profiling.set_active(None)

    def test_record_functions_are_noop_without_active_profiler(self):
        profiling.set_active(None)
        # Must not raise, must not create anything to assert on — the point is
        # every existing call site (infer_pipeline / text_extractor / clip_score)
        # stays a true no-op unless a profiler was explicitly installed.
        profiling.record_diffusion_call(steps=50)
        profiling.record_blip2_call()
        profiling.record_clip_call(kind="image")
        assert profiling.get_active() is None

    def test_set_active_get_active_roundtrip(self):
        prof = profiling.RunProfiler(video="v", total_frames=1)
        profiling.set_active(prof)
        assert profiling.get_active() is prof
        profiling.set_active(None)
        assert profiling.get_active() is None

    def test_record_accumulates_counters(self):
        prof = profiling.RunProfiler(video="v", total_frames=1)
        profiling.set_active(prof)
        profiling.record_diffusion_call(steps=10)
        profiling.record_diffusion_call(steps=10)
        profiling.record_blip2_call()
        profiling.record_clip_call(kind="image", n=2)
        profiling.record_clip_call(kind="text", n=3)
        assert prof.counters["diffusion_calls"] == 2
        assert prof.counters["diffusion_steps"] == 20
        assert prof.counters["blip2_calls"] == 1
        assert prof.counters["blip2_images"] == 1
        assert prof.counters["clip_image_calls"] == 1
        assert prof.counters["clip_image_n"] == 2
        assert prof.counters["clip_text_calls"] == 1
        assert prof.counters["clip_text_n"] == 3


class TestProfilingFrameContext:
    def test_frame_context_tallies_deltas_and_elapsed(self):
        prof = profiling.RunProfiler(video="v", total_frames=2)
        profiling.set_active(prof)
        with prof.frame(0) as ctx:
            profiling.record_diffusion_call(steps=50)
            profiling.record_blip2_call()
            ctx.decision = "keyframe"
        with prof.frame(1) as ctx:
            profiling.record_diffusion_call(steps=50)
            profiling.record_diffusion_call(steps=50)
            ctx.decision = "recompute_semantic"

        assert len(prof.frame_records) == 2
        r0, r1 = prof.frame_records
        assert r0.index == 0 and r0.decision == "keyframe"
        assert r0.diffusion_calls == 1 and r0.blip2_calls == 1
        assert r1.index == 1 and r1.decision == "recompute_semantic"
        assert r1.diffusion_calls == 2 and r1.blip2_calls == 0
        assert r0.elapsed_sec >= 0.0 and r1.elapsed_sec >= 0.0
        profiling.set_active(None)

    def test_frame_context_isolates_manual_enter_exit_driving(self, tmp_path):
        """video/temporal_pipeline.py drives the context manager via explicit
        __enter__/__exit__ (not a `with` block) to avoid re-indenting its
        per-frame decision tree — verify that usage pattern works identically."""
        prof = profiling.RunProfiler(video="v", total_frames=1,
                                     progress_path=str(tmp_path / "progress.json"))
        profiling.set_active(prof)
        ctx = prof.frame(0)
        ctx.__enter__()
        profiling.record_clip_call(kind="text")
        ctx.decision = "reuse"
        ctx.__exit__(None, None, None)
        profiling.set_active(None)

        assert prof.frame_records[0].decision == "reuse"
        assert prof.frame_records[0].clip_calls == 1
        assert Path(tmp_path / "progress.json").exists()


class TestProfilingProgressAndSummary:
    def test_write_progress_is_valid_json_with_expected_fields(self, tmp_path):
        prof = profiling.RunProfiler(video="01_person_walk", total_frames=4)
        profiling.set_active(prof)
        for i in range(2):
            with prof.frame(i) as ctx:
                profiling.record_diffusion_call(steps=10)
                ctx.decision = "keyframe"
        progress_path = tmp_path / "progress.json"
        prof.write_progress(progress_path)
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        assert data["video"] == "01_person_walk"
        assert data["frames_done"] == 2
        assert data["total_frames"] == 4
        assert data["counters"]["diffusion_calls"] == 2
        assert data["counters"]["diffusion_steps"] == 20
        assert data["last_frame"]["decision"] == "keyframe"
        assert "eta_sec" in data
        profiling.set_active(None)

    def test_write_summary_includes_full_frame_table(self, tmp_path):
        prof = profiling.RunProfiler(video="v", total_frames=2)
        profiling.set_active(prof)
        with prof.frame(0) as ctx:
            ctx.decision = "keyframe"
        with prof.frame(1) as ctx:
            ctx.decision = "reuse"
        profiling.set_active(None)
        summary_path = tmp_path / "profiling_summary.json"
        prof.write_summary(summary_path)
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert len(data["frame_records"]) == 2
        assert [r["decision"] for r in data["frame_records"]] == ["keyframe", "reuse"]

    def test_progress_write_is_atomic_replace(self, tmp_path):
        """write_progress must never leave a half-written file readable at
        the target path (long runs are tailed mid-write)."""
        prof = profiling.RunProfiler(video="v", total_frames=1)
        path = tmp_path / "progress.json"
        prof.write_progress(path)
        first = json.loads(path.read_text(encoding="utf-8"))
        assert first["frames_done"] == 0
        # No stray .tmp* files left behind.
        assert list(tmp_path.glob("*.tmp*")) == []


# ─────────────────────────────────────────────────────────────────────────────
# video/temporal_pipeline.py – force_interframe_reuse (keyframe-only real-model
# validation mode)
# ─────────────────────────────────────────────────────────────────────────────

class _StubDetector:
    """Scene detector returning a fixed boundary list (mirrors tests/test_video.py)."""

    def __init__(self, boundaries):
        self.boundaries = boundaries

    def detect(self, frames):
        return {"boundaries": self.boundaries, "distances": [0.0] * len(frames)}


class TestForceInterframeReuse:
    def _run(self, force_interframe_reuse: bool):
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
        from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor

        obj_map = {
            0: ["car"],                                   # keyframe
            1: ["car", "dog", "cat", "tree", "bus"],      # big change → would recompute
            2: ["car"],
        }
        frames = [torch.full((1, 3, 8, 8), 0.1 * (i + 1)) for i in range(3)]

        calls = {"n": 0}

        def packet_fn(frame, fid):
            if str(fid).startswith("frame_"):
                idx = int(str(fid).split("_")[1])
                objs = obj_map[idx]
            else:
                objs = ["car"]
            return build_packet(objects=objs, scene="s")

        def recon_fn(frame, cfg):
            calls["n"] += 1
            return frame * 10.0

        kfx = KeyframeExtractor(_StubDetector([True, False, False]), max_gop=None)
        pipe = TemporalPipeline(
            reconstruct_fn=recon_fn, packet_fn=packet_fn,
            keyframe_extractor=kfx, reuse_threshold=0.2,
            force_interframe_reuse=force_interframe_reuse,
        )
        return pipe.run(frames), calls

    def test_default_behaviour_unchanged(self):
        """force_interframe_reuse defaults False — byte-identical to the
        pre-existing recompute-on-big-delta behaviour."""
        res, calls = self._run(force_interframe_reuse=False)
        recs = res["records"]
        assert recs[1].decision == "recompute_semantic"
        assert calls["n"] == 2  # keyframe + the one recompute (frame 2 reuses)

    def test_force_reuse_skips_diffusion_on_every_interframe(self):
        res, calls = self._run(force_interframe_reuse=True)
        recs = res["records"]
        # Every inter-frame is forced to reuse — only the keyframe ever calls
        # reconstruct_fn (the stand-in for a real diffusion call).
        assert recs[1].decision == "reuse" and recs[1].reused is True
        assert recs[2].decision == "reuse" and recs[2].reused is True
        assert calls["n"] == 1
        # The reused recon really is the keyframe's, not a fresh reconstruction.
        assert torch.equal(recs[1].recon, recs[0].recon)
        assert torch.equal(recs[2].recon, recs[0].recon)

    def test_force_reuse_still_computes_and_logs_delta(self):
        """The decision is overridden, but the delta that WOULD have driven a
        normal recompute/reuse choice is still computed and available in the
        log — force_interframe_reuse must not blind the per-frame diagnostics."""
        res, _ = self._run(force_interframe_reuse=True)
        recs = res["records"]
        assert recs[1].delta is not None
        assert recs[1].delta["is_empty"] is False  # big object-set change, correctly detected

    def test_profiler_frame_decision_reflects_forced_reuse(self):
        prof = profiling.RunProfiler(video="v", total_frames=3)
        profiling.set_active(prof)
        try:
            self._run(force_interframe_reuse=True)
        finally:
            profiling.set_active(None)
        decisions = [r.decision for r in prof.frame_records]
        assert decisions == ["keyframe", "reuse", "reuse"]


# ─────────────────────────────────────────────────────────────────────────────
# ETRI 1A follow-up: generate_segment() wall-clock cost must be attributed to
# the generate-decision frames it covers (video/temporal_pipeline.py's
# _flush_pending_generate + utils/profiling.py's RunProfiler.record_frame()),
# not lost (a deferred frame's placeholder timer never actually ran the
# backend) or misattributed to whichever keyframe is processed next (the
# flush for a middle GOP runs before that keyframe's own timer starts).
# ─────────────────────────────────────────────────────────────────────────────

class TestSegmentGenerationProfilingAttribution:
    def _run(self, total_frames, boundaries, obj_map, sleep_sec):
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
        from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor
        from sgdjscc_lab.video.video_generator import (
            VideoGenerator, SegmentGenerationResult, GenerationMetadata,
        )

        class _SlowSegmentGenerator(VideoGenerator):
            """Mock backend whose generate_segment() takes real wall-clock
            time (via time.sleep) — stands in for a slow real (1B) backend so
            the profiler's attribution can be checked against a known cost."""

            backend_name = "slow_mock"

            def generate(self, request):
                raise NotImplementedError("only generate_segment is exercised here")

            def generate_segment(self, request):
                time.sleep(sleep_sec)
                frames = [request.start_keyframe_recon.clone() for _ in request.target_indices]
                metadata = [
                    GenerationMetadata(
                        backend=self.backend_name, conditioning_mode="start_only",
                        source_keyframe_index=request.start_keyframe_index,
                        target_indices=[i], used_caption=False, used_side_info=False, mock=True,
                    )
                    for i in request.target_indices
                ]
                return SegmentGenerationResult(
                    segment_id=request.segment_id, target_indices=list(request.target_indices),
                    frames=frames, metadata=metadata,
                )

        frames = [torch.full((1, 3, 4, 4), 0.1 * (i + 1)) for i in range(total_frames)]

        def packet_fn(frame, fid):
            fid = str(fid)
            idx = int(fid.split("_")[1]) if fid.startswith(("frame_", "recon_")) else 0
            return build_packet(objects=obj_map.get(idx, ["car"]), scene="s")

        def recon_fn(frame, cfg):
            return frame * 10.0

        kfx = KeyframeExtractor(_StubDetector(boundaries), max_gop=None)
        pipe = TemporalPipeline(
            reconstruct_fn=recon_fn, packet_fn=packet_fn, keyframe_extractor=kfx,
            reuse_threshold=0.2, enable_generate=True, video_generator=_SlowSegmentGenerator(),
        )
        prof = profiling.RunProfiler(video="v", total_frames=total_frames)
        profiling.set_active(prof)
        try:
            res = pipe.run(frames)
        finally:
            profiling.set_active(None)
        return res, prof

    def test_middle_gop_generation_time_attributed_to_generate_frame_not_next_keyframe(self):
        # 2 GOPs: keyframe 0 / inter 1 (generate) / keyframe 2 / inter 3 (reuse).
        obj_map = {0: ["car"], 1: ["car", "dog"], 2: ["boat"], 3: ["boat"]}
        res, prof = self._run(
            total_frames=4, boundaries=[True, False, True, False], obj_map=obj_map, sleep_sec=0.05,
        )
        assert res["records"][1].decision == "generate"
        by_index = {r.index: r for r in prof.frame_records}
        assert 1 in by_index
        # The ~50ms backend cost must land on frame 1 (the generate frame
        # whose call it actually was) — not on keyframe 2, which is processed
        # immediately afterward and used to swallow this time before the fix.
        assert by_index[1].elapsed_sec >= 0.03
        assert by_index[2].elapsed_sec < 0.03

    def test_last_gop_generation_time_is_not_lost(self):
        # Single GOP: keyframe 0 / inter 1 (generate) is the LAST frame in the
        # sequence — its generate_segment() call only flushes after the main
        # loop ends, with no next-keyframe iteration to (mis)attribute it to.
        obj_map = {0: ["car"], 1: ["car", "dog"]}
        res, prof = self._run(
            total_frames=2, boundaries=[True, False], obj_map=obj_map, sleep_sec=0.05,
        )
        assert res["records"][1].decision == "generate"
        by_index = {r.index: r for r in prof.frame_records}
        assert 1 in by_index
        assert by_index[1].elapsed_sec >= 0.03

    def test_multiple_generate_frames_in_one_segment_share_the_total_elapsed(self):
        # Single GOP: keyframe 0 / inter 1 (generate) / inter 2 (reuse) /
        # inter 3 (generate) — ONE generate_segment() call covers frames 1
        # and 3; its cost should be split between them, not doubled or lost.
        obj_map = {0: ["car"], 1: ["car", "dog"], 2: ["car"], 3: ["car", "dog"]}
        res, prof = self._run(
            total_frames=4, boundaries=[True, False, False, False], obj_map=obj_map, sleep_sec=0.06,
        )
        decisions = {r.index: r.decision for r in res["records"]}
        assert decisions[1] == "generate"
        assert decisions[2] == "reuse"
        assert decisions[3] == "generate"
        by_index = {r.index: r for r in prof.frame_records}
        assert 1 in by_index and 3 in by_index
        # ~60ms split evenly across 2 frames ≈ 30ms each — well above zero
        # (not lost) and well below the full 60ms (not double-counted).
        assert 0.01 < by_index[1].elapsed_sec < 0.05
        assert 0.01 < by_index[3].elapsed_sec < 0.05

    def test_generate_segment_call_counts_are_distributed_across_covered_frames(self):
        """A real (1B+) backend that calls profiling.record_diffusion_call()/
        record_blip2_call()/record_clip_call() inside generate_segment() must
        have those counts show up on the generate frames it covered, not just
        in the run-wide totals — otherwise a per-frame call-count breakdown
        (not just elapsed time) would silently read 0 for every generate
        frame forever, regardless of what the backend actually did."""
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
        from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor
        from sgdjscc_lab.video.video_generator import (
            VideoGenerator, SegmentGenerationResult, GenerationMetadata,
        )

        class _InstrumentedSegmentGenerator(VideoGenerator):
            """Mock backend that stands in for a real 1B backend which
            instruments its own model calls via the profiling module."""

            backend_name = "instrumented_mock"

            def generate(self, request):
                raise NotImplementedError("only generate_segment is exercised here")

            def generate_segment(self, request):
                # Simulates one diffusion call per target frame + one shared
                # BLIP2/CLIP call for the whole segment — a plausible real
                # backend shape, deliberately uneven so the distribution is
                # exercised (not just a trivial 1-per-frame case).
                for _ in request.target_indices:
                    profiling.record_diffusion_call(steps=10)
                profiling.record_blip2_call()
                profiling.record_clip_call(kind="image", n=1)
                frames = [request.start_keyframe_recon.clone() for _ in request.target_indices]
                metadata = [
                    GenerationMetadata(
                        backend=self.backend_name, conditioning_mode="start_only",
                        source_keyframe_index=request.start_keyframe_index,
                        target_indices=[i], used_caption=False, used_side_info=False, mock=True,
                    )
                    for i in request.target_indices
                ]
                return SegmentGenerationResult(
                    segment_id=request.segment_id, target_indices=list(request.target_indices),
                    frames=frames, metadata=metadata,
                )

        # Single GOP: keyframe 0 / inter 1 (generate) / inter 2 (reuse) /
        # inter 3 (generate) — same shape as the elapsed-time sharing test
        # above, so ONE generate_segment() call covers frames 1 and 3.
        obj_map = {0: ["car"], 1: ["car", "dog"], 2: ["car"], 3: ["car", "dog"]}
        frames = [torch.full((1, 3, 4, 4), 0.1 * (i + 1)) for i in range(4)]

        def packet_fn(frame, fid):
            fid = str(fid)
            idx = int(fid.split("_")[1]) if fid.startswith(("frame_", "recon_")) else 0
            return build_packet(objects=obj_map.get(idx, ["car"]), scene="s")

        def recon_fn(frame, cfg):
            return frame * 10.0

        kfx = KeyframeExtractor(_StubDetector([True, False, False, False]), max_gop=None)
        pipe = TemporalPipeline(
            reconstruct_fn=recon_fn, packet_fn=packet_fn, keyframe_extractor=kfx,
            reuse_threshold=0.2, enable_generate=True,
            video_generator=_InstrumentedSegmentGenerator(),
        )
        prof = profiling.RunProfiler(video="v", total_frames=4)
        profiling.set_active(prof)
        try:
            res = pipe.run(frames)
        finally:
            profiling.set_active(None)

        decisions = {r.index: r.decision for r in res["records"]}
        assert decisions[1] == "generate"
        assert decisions[3] == "generate"

        # Run-wide totals reflect the backend's real call counts regardless
        # (this part already worked before the fix).
        assert prof.counters["diffusion_calls"] == 2
        assert prof.counters["blip2_calls"] == 1
        assert prof.counters["clip_image_calls"] == 1

        by_index = {r.index: r for r in prof.frame_records}
        assert 1 in by_index and 3 in by_index
        # 2 diffusion_calls split across 2 frames → 1 each. 1 blip2_call and
        # 1 clip_call split across 2 frames → 1 to the first frame, 0 to the
        # second (deterministic remainder tie-break) — the key regression
        # check is that these are NOT all zero.
        assert by_index[1].diffusion_calls + by_index[3].diffusion_calls == 2
        assert by_index[1].diffusion_calls == 1 and by_index[3].diffusion_calls == 1
        assert by_index[1].blip2_calls + by_index[3].blip2_calls == 1
        assert by_index[1].clip_calls + by_index[3].clip_calls == 1
        # Not every covered frame is silently left at 0 for every counter.
        assert any(
            (r.diffusion_calls, r.blip2_calls, r.clip_calls) != (0, 0, 0)
            for r in (by_index[1], by_index[3])
        )

    def test_no_active_profiler_generate_segment_call_still_works(self):
        """Sanity: with no profiler installed, the generate branch must not
        break just because it now snapshots/distributes call-count deltas."""
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
        from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor
        from sgdjscc_lab.video.video_generator import CopyGenerator

        frames = [torch.full((1, 3, 4, 4), 0.1 * (i + 1)) for i in range(2)]

        def packet_fn(frame, fid):
            fid = str(fid)
            idx = int(fid.split("_")[1]) if fid.startswith(("frame_", "recon_")) else 0
            objs = {0: ["car"], 1: ["car", "dog"]}.get(idx, ["car"])
            return build_packet(objects=objs, scene="s")

        kfx = KeyframeExtractor(_StubDetector([True, False]), max_gop=None)
        pipe = TemporalPipeline(
            reconstruct_fn=lambda f, c: f * 10.0, packet_fn=packet_fn,
            keyframe_extractor=kfx, reuse_threshold=0.2,
            enable_generate=True, video_generator=CopyGenerator(),
        )
        assert profiling.get_active() is None
        out = pipe.run(frames)   # must not raise despite no active profiler
        assert out["records"][1].decision == "generate"


# ─────────────────────────────────────────────────────────────────────────────
# ETRI 1B regression: the profiler elapsed/call-count attribution 1A's
# segment batching added (see TestSegmentGenerationProfilingAttribution
# above) must hold for a REAL out-of-process backend
# (ExternalSegmentWorkerGenerator, mock worker subprocess), not just
# in-process mocks — a real subprocess call has genuine, non-negligible
# wall-clock cost that a broken attribution could just as easily lose or
# misattribute to the wrong frame.
# ─────────────────────────────────────────────────────────────────────────────

class TestSegmentGenerationProfilingAttributionExternalWorker:
    def test_external_worker_generation_time_attributed_to_generate_frame(self):
        from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
        from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor
        from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator

        # 2 GOPs: keyframe 0 / inter 1 (generate) / keyframe 2 / inter 3 (reuse) —
        # same shape as the mock-backend regression test above.
        obj_map = {0: ["car"], 1: ["car", "dog"], 2: ["boat"], 3: ["boat"]}
        frames = [torch.full((1, 3, 8, 8), 0.1 * (i + 1)) for i in range(4)]

        def packet_fn(frame, fid):
            fid = str(fid)
            idx = int(fid.split("_")[1]) if fid.startswith(("frame_", "recon_")) else 0
            return build_packet(objects=obj_map.get(idx, ["car"]), scene="s")

        kfx = KeyframeExtractor(_StubDetector([True, False, True, False]), max_gop=None)
        gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, backend="mock", device="cpu")
        pipe = TemporalPipeline(
            reconstruct_fn=lambda f, c: f * 10.0, packet_fn=packet_fn,
            keyframe_extractor=kfx, reuse_threshold=0.2,
            enable_generate=True, video_generator=gen,
        )
        prof = profiling.RunProfiler(video="v", total_frames=4)
        profiling.set_active(prof)
        try:
            res = pipe.run(frames)
        finally:
            profiling.set_active(None)

        assert res["records"][1].decision == "generate"
        by_index = {r.index: r for r in prof.frame_records}
        assert 1 in by_index
        # Launching a real python subprocess takes measurable time (interpreter
        # startup alone is several ms) — it must land on frame 1's record, not
        # be zero, and must not be swallowed by keyframe 2 (processed right
        # after in the main loop — see _flush_pending_generate's docstring).
        assert by_index[1].elapsed_sec > 0.0
        assert by_index[2].elapsed_sec < by_index[1].elapsed_sec


# ─────────────────────────────────────────────────────────────────────────────
# guidance/packet_cache.py
# ─────────────────────────────────────────────────────────────────────────────

class TestPacketCache:
    def _video(self, tmp_path, name="clip.mp4", content=b"fake video bytes"):
        p = tmp_path / name
        p.write_bytes(content)
        return p

    def test_build_meta_captures_key_fields(self, tmp_path):
        from sgdjscc_lab.guidance.packet_cache import build_meta
        from sgdjscc_lab.utils.packet_io import PACKET_VERSION

        video = self._video(tmp_path)
        meta = build_meta(video, caption_source="blip2", clip_model_name="ViT-B/32",
                          packet_caption_objects=True)
        assert meta["packet_version"] == PACKET_VERSION
        assert meta["video_size"] == video.stat().st_size
        assert meta["caption_source"] == "blip2"
        assert meta["clip_model_name"] == "ViT-B/32"
        assert meta["packet_caption_objects"] is True

    def test_put_get_roundtrip_without_save(self, tmp_path):
        from sgdjscc_lab.guidance.packet_cache import PacketCache, build_meta

        video = self._video(tmp_path)
        meta = build_meta(video, caption_source="blip2", clip_model_name=None,
                          packet_caption_objects=True)
        cache = PacketCache(tmp_path / "cache", video, meta)
        assert cache.get("frame_00000") is None
        pkt = build_packet(caption="a dog", objects=["dog"], scene="park")
        cache.put("frame_00000", pkt)
        assert cache.get("frame_00000") == pkt
        assert cache.stats["hits"] == 1 and cache.stats["misses"] == 1

    def test_save_then_reload_hits_cache(self, tmp_path):
        from sgdjscc_lab.guidance.packet_cache import PacketCache, build_meta

        video = self._video(tmp_path)
        meta = build_meta(video, caption_source="blip2", clip_model_name=None,
                          packet_caption_objects=True)
        cache_dir = tmp_path / "cache"
        cache1 = PacketCache(cache_dir, video, meta)
        pkt = build_packet(caption="a cat", objects=["cat"], scene="kitchen")
        cache1.put("frame_00000", pkt)
        cache1.save()

        cache2 = PacketCache(cache_dir, video, meta)
        assert cache2.get("frame_00000") == pkt
        assert cache2.stats["misses"] == 0

    def test_different_meta_invalidates_whole_cache(self, tmp_path):
        """A cache built with captions must never be silently reused by a
        BLIP2-caption run (or vice versa) — the WHOLE file is ignored on any
        meta mismatch, never patched entry-by-entry."""
        from sgdjscc_lab.guidance.packet_cache import PacketCache, build_meta

        video = self._video(tmp_path)
        cache_dir = tmp_path / "cache"
        meta_a = build_meta(video, caption_source="blip2", clip_model_name=None,
                            packet_caption_objects=True)
        cache1 = PacketCache(cache_dir, video, meta_a)
        cache1.put("frame_00000", build_packet(caption="from blip2", objects=["x"]))
        cache1.save()

        meta_b = build_meta(video, caption_source="captions:foo.txt", clip_model_name=None,
                            packet_caption_objects=True)
        cache2 = PacketCache(cache_dir, video, meta_b)
        assert cache2.get("frame_00000") is None  # old cache ignored, not mixed in

    def test_video_mtime_change_invalidates_cache(self, tmp_path):
        from sgdjscc_lab.guidance.packet_cache import PacketCache, build_meta

        video = self._video(tmp_path)
        cache_dir = tmp_path / "cache"
        meta1 = build_meta(video, caption_source="blip2", clip_model_name=None,
                           packet_caption_objects=True)
        cache1 = PacketCache(cache_dir, video, meta1)
        cache1.put("frame_00000", build_packet(caption="old", objects=["x"]))
        cache1.save()

        time.sleep(0.01)
        video.write_bytes(b"re-exported, different content")  # mtime + size change
        meta2 = build_meta(video, caption_source="blip2", clip_model_name=None,
                           packet_caption_objects=True)
        cache2 = PacketCache(cache_dir, video, meta2)
        assert cache2.get("frame_00000") is None

    def test_save_uses_pid_unique_tmp_file_and_leaves_none_behind(self, tmp_path, monkeypatch):
        """The same video can legitimately be processed by several concurrent
        processes (different modes/thresholds all reading the same ORIGINAL
        frames) sharing one --packet-cache-dir. A fixed ".json.tmp" name would
        let two processes' writes interleave on the same temp file before
        either renames it; the temp file must be pid-suffixed instead."""
        from sgdjscc_lab.guidance.packet_cache import PacketCache, build_meta

        video = self._video(tmp_path)
        meta = build_meta(video, caption_source="blip2", clip_model_name=None,
                          packet_caption_objects=True)
        cache_dir = tmp_path / "cache"
        cache = PacketCache(cache_dir, video, meta)
        cache.put("frame_00000", build_packet(caption="a cat", objects=["cat"]))

        seen_tmp_names = []
        real_replace = __import__("os").replace

        def _spy_replace(src, dst):
            seen_tmp_names.append(Path(src).name)
            return real_replace(src, dst)

        monkeypatch.setattr("sgdjscc_lab.guidance.packet_cache.os.replace", _spy_replace)
        monkeypatch.setattr("sgdjscc_lab.guidance.packet_cache.os.getpid", lambda: 424242)
        cache.save()

        assert seen_tmp_names == ["clip.json.tmp424242"]
        # No leftover temp file — os.replace() consumed it — and no stray
        # non-pid-suffixed ".json.tmp" from an older naming scheme either.
        leftovers = list(cache_dir.glob("*.tmp*"))
        assert leftovers == []
        assert (cache_dir / "clip.json.tmp").exists() is False

    def test_stale_tmp_file_from_another_pid_does_not_interfere_with_save(self, tmp_path, monkeypatch):
        """A leftover tmp file from an earlier/other process (crashed mid-write,
        or a concurrently running one) must not be touched or collided with
        by this process's own pid-suffixed save."""
        from sgdjscc_lab.guidance.packet_cache import PacketCache, build_meta

        video = self._video(tmp_path)
        meta = build_meta(video, caption_source="blip2", clip_model_name=None,
                          packet_caption_objects=True)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        stale = cache_dir / "clip.json.tmp999"
        stale.write_text('{"stale": true}', encoding="utf-8")

        cache = PacketCache(cache_dir, video, meta)
        cache.put("frame_00000", build_packet(caption="a cat", objects=["cat"]))
        monkeypatch.setattr("sgdjscc_lab.guidance.packet_cache.os.getpid", lambda: 424242)
        cache.save()

        # This process's own (different-pid) tmp file was consumed by the
        # rename; the other pid's stale tmp file is left completely alone.
        assert stale.exists()
        assert stale.read_text(encoding="utf-8") == '{"stale": true}'
        assert (cache_dir / "clip.json.tmp424242").exists() is False
        assert json.loads(cache.path.read_text(encoding="utf-8"))["packets"]["frame_00000"]["caption"] == "a cat"


# ─────────────────────────────────────────────────────────────────────────────
# evaluators/clip_score.py – text-embedding memoization
# ─────────────────────────────────────────────────────────────────────────────

class TestClipTextEmbeddingCache:
    def test_repeated_text_encode_hits_cache_and_skips_profiling_record(self):
        clip = pytest.importorskip("clip")
        from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator

        ev = CLIPScoreEvaluator(model_name="ViT-B/32", device=torch.device("cpu"))
        prof = profiling.RunProfiler(video="v", total_frames=1)
        profiling.set_active(prof)
        try:
            texts = ["a photo of a cat", "a photo of a dog"]
            feats1 = ev._encode_texts(texts)
            feats2 = ev._encode_texts(list(texts))  # same content, new list object
        finally:
            profiling.set_active(None)

        # Only the first call actually hit the model — recorded once, not twice.
        assert prof.counters.get("clip_text_calls", 0) == 1
        assert prof.counters.get("clip_text_n", 0) == len(texts)
        assert torch.equal(feats1, feats2)
        # Cache must return an independent tensor (mutating one must not
        # corrupt the cached copy for the next caller).
        feats1[0, 0] = 999.0
        feats3 = ev._encode_texts(texts)
        assert not torch.equal(feats3, feats1)

    def test_different_text_lists_are_not_conflated(self):
        pytest.importorskip("clip")
        from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator

        ev = CLIPScoreEvaluator(model_name="ViT-B/32", device=torch.device("cpu"))
        feats_a = ev._encode_texts(["a cat"])
        feats_b = ev._encode_texts(["a dog"])
        assert not torch.equal(feats_a, feats_b)


# ─────────────────────────────────────────────────────────────────────────────
# utils/gpu_logger.py
# ─────────────────────────────────────────────────────────────────────────────

class TestGPULogger:
    def test_disabled_when_nvidia_smi_missing(self, tmp_path, monkeypatch):
        from sgdjscc_lab.utils import gpu_logger

        monkeypatch.setattr(gpu_logger.shutil, "which", lambda name: None)
        gl = gpu_logger.GPULogger(tmp_path / "gpu_util.csv", interval_sec=0.05)
        gl.start()
        gl.stop()
        assert not (tmp_path / "gpu_util.csv").exists()

    def test_query_once_parses_csv_output(self, monkeypatch):
        from sgdjscc_lab.utils import gpu_logger

        sample = "0, NVIDIA GeForce RTX 4090, 42, 1000, 24564\n1, NVIDIA GeForce RTX 4090, 0, 11, 24564\n"

        class _FakeProc:
            returncode = 0
            stdout = sample

        monkeypatch.setattr(gpu_logger.subprocess, "run", lambda *a, **k: _FakeProc())
        rows = gpu_logger._query_once()
        assert len(rows) == 2
        assert rows[0]["index"] == "0"
        assert rows[0]["utilization_gpu_pct"] == "42"
        assert rows[1]["memory_used_mib"] == "11"

    def test_query_once_returns_none_when_nvidia_smi_absent(self, monkeypatch):
        from sgdjscc_lab.utils import gpu_logger

        def _raise(*a, **k):
            raise FileNotFoundError()

        monkeypatch.setattr(gpu_logger.subprocess, "run", _raise)
        assert gpu_logger._query_once() is None

    def test_start_stop_writes_rows_when_available(self, tmp_path, monkeypatch):
        from sgdjscc_lab.utils import gpu_logger

        sample = "0, GPU, 10, 100, 24564\n"

        class _FakeProc:
            returncode = 0
            stdout = sample

        monkeypatch.setattr(gpu_logger.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        monkeypatch.setattr(gpu_logger.subprocess, "run", lambda *a, **k: _FakeProc())

        csv_path = tmp_path / "gpu_util.csv"
        gl = gpu_logger.GPULogger(csv_path, interval_sec=0.02)
        gl.start()
        time.sleep(0.15)
        gl.stop()

        assert csv_path.exists()
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0].startswith("timestamp")
        assert len(lines) >= 2  # header + at least one sample
