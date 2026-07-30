"""tests/test_lgvsc_generate_worker.py – scripts/lgvsc_generate_worker.py tests
(ETRI 후속 1단계 step 1B).

All tests use ``--backend mock`` (dependency-free — PIL + numpy only) or a
deliberately broken/slow callable adapter written to a temp module for this
test file — no GPU, no ``diffusers``/Open-Sora/Wan, no network. Covers both
the importable functions directly (fast — most tests) and at least one real
subprocess invocation of the actual CLI entrypoint (proving the script itself
works, not just its internals).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
_WORKER_SCRIPT = _REPO / "scripts" / "lgvsc_generate_worker.py"


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("lgvsc_generate_worker", _WORKER_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


worker = _load_worker_module()


def _make_keyframe_png(path, gray_value: int) -> None:
    arr = np.full((8, 8, 3), gray_value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _write_manifest(tmp_path, target_indices, start_idx=0, end_idx=None, end_image=False,
                     captions=None, packets=None, side_infos=None):
    start_path = tmp_path / "start_keyframe.png"
    _make_keyframe_png(start_path, 0)
    end_rel = None
    if end_image:
        end_path = tmp_path / "end_keyframe.png"
        _make_keyframe_png(end_path, 255)
        end_rel = end_path.name

    span_end = end_idx if end_idx is not None else max(target_indices)
    n = len(target_indices)
    manifest = {
        "segment_id": 0,
        "start_frame_index": start_idx,
        "end_frame_index": span_end,
        "segment_length": span_end - start_idx + 1,
        "target_indices": list(target_indices),
        "start_keyframe_index": start_idx,
        "end_keyframe_index": end_idx,
        "start_keyframe_image": start_path.name,
        "end_keyframe_image": end_rel,
        "fps": 24.0,
        "captions": captions if captions is not None else [None] * n,
        "packets": packets if packets is not None else [None] * n,
        "side_infos": side_infos if side_infos is not None else [None] * n,
        "run_config": {"seed": 42, "model_id": None, "device": "cpu", "dtype": "fp32",
                       "height": 8, "width": 8},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, manifest


# ─────────────────────────────────────────────────────────────────────────────
# image IO
# ─────────────────────────────────────────────────────────────────────────────

class TestImageIO:
    def test_round_trip_float_array(self, tmp_path):
        arr = np.random.rand(4, 4, 3).astype(np.float32)
        p = tmp_path / "img.png"
        worker.save_frame_image(arr, p)
        loaded = worker.load_keyframe_image(p)
        assert loaded.shape == (4, 4, 3)
        assert np.allclose(loaded, arr, atol=1.0 / 255)

    def test_save_uint8_array(self, tmp_path):
        arr = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)
        p = tmp_path / "img.png"
        worker.save_frame_image(arr, p)
        assert p.exists()
        loaded = worker.load_keyframe_image(p)
        assert np.allclose(loaded * 255, arr, atol=1)

    def test_save_pil_image(self, tmp_path):
        img = Image.new("RGB", (4, 4), color=(10, 20, 30))
        p = tmp_path / "img.png"
        worker.save_frame_image(img, p)
        assert p.exists()

    def test_load_manifest(self, tmp_path):
        manifest_path, manifest = _write_manifest(tmp_path, [1, 2])
        loaded = worker.load_manifest(manifest_path)
        assert loaded == manifest


# ─────────────────────────────────────────────────────────────────────────────
# mock backend
# ─────────────────────────────────────────────────────────────────────────────

class TestRunMockBackend:
    def test_start_only_returns_copy_of_start_keyframe(self, tmp_path):
        _, manifest = _write_manifest(tmp_path, [1, 2])
        result = worker.run_mock_backend(manifest, tmp_path, argparse.Namespace(seed=None))
        assert set(result["frames"]) == {1, 2}
        for idx in (1, 2):
            assert np.allclose(result["frames"][idx], 0.0)
            assert result["metadata"][idx]["conditioning_mode"] == "start_only"
            assert result["metadata"][idx]["end_keyframe_index"] is None
            assert result["metadata"][idx]["mock"] is True

    def test_bidirectional_blends_by_relative_position(self, tmp_path):
        _, manifest = _write_manifest(tmp_path, [2, 5, 8], end_idx=10, end_image=True)
        result = worker.run_mock_backend(manifest, tmp_path, argparse.Namespace(seed=None))
        assert result["metadata"][2]["conditioning_mode"] == "bidirectional"
        assert result["metadata"][2]["relative_position"] == pytest.approx(0.2)
        assert result["metadata"][5]["relative_position"] == pytest.approx(0.5)
        assert result["metadata"][8]["relative_position"] == pytest.approx(0.8)
        assert result["metadata"][5]["end_keyframe_index"] == 10
        # target 5 (relative 0.5) should be roughly the average of start(0) and end(255)
        assert np.allclose(result["frames"][5], 0.5, atol=0.02)

    def test_caption_and_side_info_flags(self, tmp_path):
        _, manifest = _write_manifest(
            tmp_path, [1, 2], captions=["a caption", None],
            side_infos=[{"delta": 0.1}, None],
        )
        result = worker.run_mock_backend(manifest, tmp_path, argparse.Namespace(seed=None))
        assert result["metadata"][1]["used_caption"] is True
        assert result["metadata"][2]["used_caption"] is False
        assert result["metadata"][1]["used_side_info"] is True
        assert result["metadata"][2]["used_side_info"] is False

    def test_seed_recorded_but_not_used_for_pixels(self, tmp_path):
        _, manifest = _write_manifest(tmp_path, [1])
        r1 = worker.run_mock_backend(manifest, tmp_path, argparse.Namespace(seed=111))
        r2 = worker.run_mock_backend(manifest, tmp_path, argparse.Namespace(seed=222))
        assert "111" in r1["metadata"][1]["notes"]
        assert "222" in r2["metadata"][1]["notes"]
        assert np.allclose(r1["frames"][1], r2["frames"][1])  # deterministic regardless of seed

    def test_rx_legal_ignores_unreferenced_original_frame_on_disk(self, tmp_path):
        """Even if an original/un-transmitted target frame happens to exist
        on disk (e.g. left over from some other tool), run_mock_backend()
        only ever reads the two keyframe image paths named in the manifest —
        it has no field/mechanism to discover or read anything else."""
        _make_keyframe_png(tmp_path / "original_target_should_never_be_read.png", 128)
        _, manifest = _write_manifest(tmp_path, [1, 2], end_idx=5, end_image=True)
        result = worker.run_mock_backend(manifest, tmp_path, argparse.Namespace(seed=None))
        for idx in (1, 2):
            assert not np.allclose(result["frames"][idx], 128.0 / 255.0, atol=0.01)


class TestRunSvdBackendReferenceWiring:
    def test_one_target_smoke_uses_model_clip_length_and_passes_size(self, tmp_path, monkeypatch):
        """Regression: a one-frame smoke test must not call SVD with
        num_frames=1. The real SVD/SVD-XT pipeline expects its configured clip
        length (14/25) even if the caller only needs one target frame back."""
        calls = []

        class _FakePipeline:
            def __init__(self):
                self.unet = types.SimpleNamespace(config=types.SimpleNamespace(num_frames=14))

            @classmethod
            def from_pretrained(cls, model_id, torch_dtype=None):
                return cls()

            def to(self, device):
                self.device = device
                return self

            def __call__(self, image, **kwargs):
                calls.append(kwargs)
                frames = [Image.new("RGB", (kwargs["width"], kwargs["height"]), color=(i, i, i))
                          for i in range(kwargs["num_frames"])]
                return types.SimpleNamespace(frames=[frames])

        monkeypatch.setitem(
            sys.modules,
            "diffusers",
            types.SimpleNamespace(StableVideoDiffusionPipeline=_FakePipeline),
        )
        _, manifest = _write_manifest(tmp_path, [1])
        args = argparse.Namespace(
            model_id="fake/svd", dtype="fp16", device="cpu", seed=123,
            height=16, width=24, num_inference_steps=5, decode_chunk_size=1,
        )

        result = worker.run_svd_backend(manifest, tmp_path, args)

        assert calls
        assert calls[0]["num_frames"] == 14
        assert calls[0]["height"] == 16
        assert calls[0]["width"] == 24
        assert calls[0]["decode_chunk_size"] == 1
        assert set(result["frames"]) == {1}
        assert result["frames"][1].size == (8, 8)


class TestRunWanBackendReferenceWiring:
    """Wan (diffusers.WanImageToVideoPipeline) genuinely supports the fuller
    LGVSC segment contract SVD cannot: image (start keyframe) + last_image
    (end keyframe, when present — real bidirectional conditioning) + prompt
    (caption — real text conditioning). side_infos are still accepted but
    intentionally NOT used (see run_wan_backend's docstring) — verified here
    too, so that limitation stays enforced by a test, not just documentation.
    """

    @staticmethod
    def _fake_diffusers_module(calls, model_ids_loaded=None):
        """Fake WanImageToVideoPipeline whose `transformer.config.pos_embed_seq_len`
        depends on the model_id passed to from_pretrained — 514 for any id
        containing "FLF2V" (simulating the real Wan2.1-FLF2V-14B-720P
        checkpoint's transformer config), None otherwise (simulating the
        plain Wan2.1-I2V-14B-480P checkpoint). This lets tests exercise the
        real per-segment checkpoint-selection/preflight-check logic in
        run_wan_backend, not just the pipeline __call__ arguments."""

        class _FakePipeline:
            @classmethod
            def from_pretrained(cls, model_id, torch_dtype=None):
                if model_ids_loaded is not None:
                    model_ids_loaded.append(model_id)
                self = cls()
                self.model_id = model_id
                pos_embed_seq_len = 514 if "FLF2V" in model_id else None
                self.transformer = types.SimpleNamespace(
                    config=types.SimpleNamespace(pos_embed_seq_len=pos_embed_seq_len)
                )
                return self

            def to(self, device):
                self.device = device
                return self

            def __call__(self, image, **kwargs):
                calls.append(kwargs)
                n = kwargs["num_frames"]
                frames = [
                    Image.new("RGB", (kwargs["width"], kwargs["height"]), color=(i, i, i))
                    for i in range(n)
                ]
                return types.SimpleNamespace(frames=[frames])

        return types.SimpleNamespace(WanImageToVideoPipeline=_FakePipeline)

    @staticmethod
    def _args(**overrides):
        base = dict(
            model_id="fake/wan", dtype="bf16", device="cpu", seed=None,
            height=None, width=None, num_inference_steps=None,
            decode_chunk_size=None, extra_json=None,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_start_only_conditioning_and_caption_becomes_prompt(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setitem(sys.modules, "diffusers", self._fake_diffusers_module(calls))
        _, manifest = _write_manifest(
            tmp_path, [1, 2], captions=["a red car", None], side_infos=[{"delta": 0.4}, None],
        )
        result = worker.run_wan_backend(manifest, tmp_path, self._args(
            seed=7, height=16, width=24, num_inference_steps=8,
        ))

        # ONE batched pipeline call for the whole segment, not one per frame.
        assert len(calls) == 1
        kw = calls[0]
        assert kw["prompt"] == "a red car"
        assert "last_image" not in kw
        assert kw["height"] == 16 and kw["width"] == 24
        assert kw["num_inference_steps"] == 8

        assert set(result["frames"]) == {1, 2}
        # Resized back to the ORIGINAL keyframe size (8x8 in _write_manifest),
        # not the generation height/width (16x24) — shape must always match
        # start_keyframe_recon, enforced upstream by ExternalSegmentWorkerGenerator
        # too, but the backend itself must already hand back the right size.
        assert result["frames"][1].size == (8, 8)
        assert result["frames"][2].size == (8, 8)

        for idx in (1, 2):
            meta = result["metadata"][idx]
            assert meta["backend"] == "external_segment_worker:wan:fake/wan"
            assert meta["conditioning_mode"] == "start_only"
            assert meta["source_keyframe_index"] == 0
            assert meta["end_keyframe_index"] is None
            assert meta["target_indices"] == [idx]
            assert meta["used_caption"] is True          # a caption was present in the segment
            assert meta["used_side_info"] is False        # documented limitation — never used
            assert meta["mock"] is False
            assert "not verified" in meta["notes"].lower() or "NOT verified" in meta["notes"]

    def test_bidirectional_conditioning_uses_last_image(self, tmp_path, monkeypatch):
        calls = []
        model_ids_loaded = []
        monkeypatch.setitem(
            sys.modules, "diffusers", self._fake_diffusers_module(calls, model_ids_loaded),
        )
        _, manifest = _write_manifest(tmp_path, [2, 5, 8], end_idx=10, end_image=True)
        result = worker.run_wan_backend(manifest, tmp_path, self._args(
            extra_json='{"bidirectional_model_id": "fake/wan-FLF2V"}',
        ))

        assert len(calls) == 1
        assert "last_image" in calls[0]
        # Bidirectional segment must load the FLF2V checkpoint (the one whose
        # transformer config actually has pos_embed_seq_len), NOT the default
        # start-only model_id — see run_wan_backend's docstring for why.
        assert model_ids_loaded == ["fake/wan-FLF2V"]
        # The dict run_wan_backend returns to main() must report the checkpoint
        # it ACTUALLY loaded, not args.model_id — main() uses this for
        # result.json's top-level model_id field (see main()'s
        # result.get("model_id", args.model_id)).
        assert result["model_id"] == "fake/wan-FLF2V"
        assert set(result["frames"]) == {2, 5, 8}
        for idx in (2, 5, 8):
            assert result["metadata"][idx]["conditioning_mode"] == "bidirectional"
            assert result["metadata"][idx]["end_keyframe_index"] == 10
            assert "fake/wan-FLF2V" in result["metadata"][idx]["backend"]
            assert "pos_embed_seq_len=514" in result["metadata"][idx]["notes"]

    def test_no_caption_available_leaves_used_caption_false(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setitem(sys.modules, "diffusers", self._fake_diffusers_module(calls))
        _, manifest = _write_manifest(tmp_path, [1])  # no captions given
        result = worker.run_wan_backend(manifest, tmp_path, self._args())
        assert calls[0]["prompt"] == ""
        assert result["metadata"][1]["used_caption"] is False

    def test_frame_count_rounds_up_to_valid_wan_length(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setitem(sys.modules, "diffusers", self._fake_diffusers_module(calls))
        # Segment span is start_frame_index=0 .. end_frame_index=6 (7 frames);
        # (7 - 1) % 4 == 2 != 0, so it must round up to the next valid Wan
        # length (9), even though only 2 of those 7 positions are targets.
        _, manifest = _write_manifest(tmp_path, [3, 6], end_idx=6)
        worker.run_wan_backend(manifest, tmp_path, self._args())
        assert calls[0]["num_frames"] == 9

    def test_single_target_still_requests_minimum_valid_length(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setitem(sys.modules, "diffusers", self._fake_diffusers_module(calls))
        _, manifest = _write_manifest(tmp_path, [1])
        result = worker.run_wan_backend(manifest, tmp_path, self._args())
        assert calls[0]["num_frames"] == 5
        assert set(result["frames"]) == {1}   # only the requested target is returned

    def test_non_contiguous_targets_map_to_correct_temporal_position(self, tmp_path, monkeypatch):
        """Regression for the frame-mapping bug: target_indices=[1, 5, 8] are
        non-contiguous within the segment. The generated clip's frame N
        corresponds to segment offset N (frame 0 == start keyframe), so target
        1 must come from clip position 1, target 5 from position 5, and target
        8 from position 8 — NOT from clip positions 0, 1, 2 (their positions
        in the target_indices list), which is what the old enumerate()-based
        mapping produced."""
        calls = []
        monkeypatch.setitem(sys.modules, "diffusers", self._fake_diffusers_module(calls))
        _, manifest = _write_manifest(tmp_path, [1, 5, 8])  # start-only, span_end=8
        result = worker.run_wan_backend(manifest, tmp_path, self._args())

        n_frames = calls[0]["num_frames"]
        assert n_frames > 8  # must be able to reach clip position 8

        def _gray(idx):
            return result["frames"][idx].getpixel((0, 0))[0]

        assert _gray(1) == 1
        assert _gray(5) == 5
        assert _gray(8) == 8

    def test_bidirectional_targets_map_to_relative_temporal_position(self, tmp_path, monkeypatch):
        """Same regression, but for bidirectional mode: each target's clip
        position must be proportional to its fraction of the way from the
        start keyframe to the end keyframe, even after num_frames is padded
        up to a valid Wan length."""
        calls = []
        monkeypatch.setitem(sys.modules, "diffusers", self._fake_diffusers_module(calls))
        _, manifest = _write_manifest(tmp_path, [2, 5, 8], end_idx=10, end_image=True)
        result = worker.run_wan_backend(manifest, tmp_path, self._args(
            extra_json='{"bidirectional_model_id": "fake/wan-FLF2V"}',
        ))

        n_frames = calls[0]["num_frames"]

        def _gray(idx):
            return result["frames"][idx].getpixel((0, 0))[0]

        assert _gray(2) == round(0.2 * (n_frames - 1))
        assert _gray(5) == round(0.5 * (n_frames - 1))
        assert _gray(8) == round(0.8 * (n_frames - 1))
        assert result["metadata"][2]["relative_position"] == pytest.approx(0.2)
        assert result["metadata"][5]["relative_position"] == pytest.approx(0.5)
        assert result["metadata"][8]["relative_position"] == pytest.approx(0.8)

    def test_start_only_segment_uses_default_model_even_with_bidirectional_id_set(
        self, tmp_path, monkeypatch,
    ):
        """A segment with NO end keyframe (e.g. a video's last GOP) must load
        args.model_id (the plain start-only checkpoint), never
        extra_json.bidirectional_model_id — even when the latter is
        configured for OTHER segments in the same run. This is what lets one
        config safely serve a video where only some GOPs have an end
        keyframe."""
        calls = []
        model_ids_loaded = []
        monkeypatch.setitem(
            sys.modules, "diffusers", self._fake_diffusers_module(calls, model_ids_loaded),
        )
        _, manifest = _write_manifest(tmp_path, [1, 2])  # no end_image -> start-only
        worker.run_wan_backend(manifest, tmp_path, self._args(
            model_id="fake/wan-I2V", extra_json='{"bidirectional_model_id": "fake/wan-FLF2V"}',
        ))
        assert model_ids_loaded == ["fake/wan-I2V"]

    def test_bidirectional_segment_with_incapable_checkpoint_raises_clear_error(
        self, tmp_path, monkeypatch,
    ):
        """Regression for the real-GPU failure this fix addresses: asking a
        checkpoint with no pos_embed_seq_len (e.g. the plain Wan2.1-I2V-14B-
        480P) to do last_image/bidirectional conditioning must fail with a
        clear, actionable WorkerBackendUnavailableError BEFORE the pipeline
        is even called — not with diffusers' cryptic downstream tensor-size
        RuntimeError."""
        calls = []
        monkeypatch.setitem(sys.modules, "diffusers", self._fake_diffusers_module(calls))
        _, manifest = _write_manifest(tmp_path, [2, 5, 8], end_idx=10, end_image=True)
        # No bidirectional_model_id configured -> falls back to args.model_id
        # ("fake/wan", which the fake module treats as pos_embed_seq_len=None).
        with pytest.raises(worker.WorkerBackendUnavailableError, match="pos_embed_seq_len"):
            worker.run_wan_backend(manifest, tmp_path, self._args())
        assert calls == []  # must fail BEFORE calling the pipeline

    def test_start_only_segment_with_flf2v_only_checkpoint_raises_clear_error(
        self, tmp_path, monkeypatch,
    ):
        """The inverse misconfiguration: pointing model_id directly at an
        FLF2V (two-image-only) checkpoint for a segment with no end keyframe
        must also fail clearly, not silently reshape-crash inside the
        pipeline."""
        calls = []
        monkeypatch.setitem(sys.modules, "diffusers", self._fake_diffusers_module(calls))
        _, manifest = _write_manifest(tmp_path, [1])  # no end_image -> start-only
        with pytest.raises(worker.WorkerBackendUnavailableError, match="pos_embed_seq_len"):
            worker.run_wan_backend(manifest, tmp_path, self._args(model_id="fake/wan-FLF2V"))
        assert calls == []

    def test_offload_mode_from_extra_json(self, tmp_path, monkeypatch):
        offload_calls = []

        class _FakePipeline:
            @classmethod
            def from_pretrained(cls, model_id, torch_dtype=None):
                self = cls()
                self.transformer = types.SimpleNamespace(
                    config=types.SimpleNamespace(pos_embed_seq_len=None)
                )
                return self

            def enable_sequential_cpu_offload(self, device=None):
                offload_calls.append(("sequential", device))

            def enable_model_cpu_offload(self, device=None):
                offload_calls.append(("model", device))

            def to(self, device):
                offload_calls.append(("to", device))
                return self

            def __call__(self, image, **kwargs):
                n = kwargs["num_frames"]
                frames = [
                    Image.new("RGB", (kwargs["width"], kwargs["height"]), color=(0, 0, 0))
                    for _ in range(n)
                ]
                return types.SimpleNamespace(frames=[frames])

        monkeypatch.setitem(
            sys.modules, "diffusers", types.SimpleNamespace(WanImageToVideoPipeline=_FakePipeline),
        )
        _, manifest = _write_manifest(tmp_path, [1])
        result = worker.run_wan_backend(manifest, tmp_path, self._args(
            device="cuda:0", extra_json='{"offload_mode": "sequential"}',
        ))
        assert offload_calls == [("sequential", "cuda:0")]
        assert set(result["frames"]) == {1}

    def test_missing_wan_pipeline_class_raises_worker_backend_unavailable(self, tmp_path):
        """ptest's own diffusers (0.26.3, needed by SGD-JSCC's own diffusion
        reconstruction — NOT installed for this feature) has no
        WanImageToVideoPipeline — this naturally exercises the
        dependency-unavailable path with no monkeypatching required."""
        _, manifest = _write_manifest(tmp_path, [1])
        with pytest.raises(worker.WorkerBackendUnavailableError):
            worker.run_wan_backend(manifest, tmp_path, self._args())


# ─────────────────────────────────────────────────────────────────────────────
# generate() dispatch + validation
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateDispatch:
    def test_unknown_backend_raises(self, tmp_path):
        _, manifest = _write_manifest(tmp_path, [1])
        with pytest.raises(ValueError, match="Unknown --backend"):
            worker.generate(manifest, tmp_path, argparse.Namespace(backend="nope"))

    def test_backend_returning_wrong_indices_raises(self, tmp_path):
        _, manifest = _write_manifest(tmp_path, [1, 2])

        def _broken(manifest, manifest_dir, args):
            return {"frames": {1: np.zeros((4, 4, 3))}, "metadata": {1: {}}}

        worker._BACKENDS["_broken_for_test"] = _broken
        try:
            with pytest.raises(ValueError, match="returned frames for"):
                worker.generate(manifest, tmp_path, argparse.Namespace(backend="_broken_for_test"))
        finally:
            del worker._BACKENDS["_broken_for_test"]


# ─────────────────────────────────────────────────────────────────────────────
# main() end-to-end (direct function call — fast)
# ─────────────────────────────────────────────────────────────────────────────

class TestMainEndToEnd:
    def _argv(self, manifest_path, output_dir, **overrides):
        base = dict(manifest=str(manifest_path), output_dir=str(output_dir), backend="mock")
        base.update(overrides)
        argv = []
        for k, v in base.items():
            if v is None:
                continue
            argv += [f"--{k.replace('_', '-')}", str(v)]
        return argv

    def test_success_writes_result_json_and_frames(self, tmp_path):
        manifest_path, _ = _write_manifest(tmp_path, [1, 2])
        out_dir = tmp_path / "out"
        rc = worker.main(self._argv(manifest_path, out_dir, seed=7))
        assert rc == 0
        assert not (out_dir / "error.json").exists()
        result = json.loads((out_dir / "result.json").read_text())
        assert result["status"] == "ok"
        assert result["backend"] == "mock"
        assert result["seed"] == 7
        assert sorted(int(k) for k in result["frames"]) == [1, 2]
        for idx in (1, 2):
            assert (out_dir / result["frames"][str(idx)]).exists()
            assert result["metadata"][str(idx)]["target_indices"] == [idx]

    def test_missing_manifest_writes_error_json_nonzero_exit(self, tmp_path):
        out_dir = tmp_path / "out"
        rc = worker.main(["--manifest", str(tmp_path / "nope.json"), "--output-dir", str(out_dir)])
        assert rc != 0
        err = json.loads((out_dir / "error.json").read_text())
        assert err["status"] == "error"
        assert "traceback" in err

    def test_callable_backend_missing_entrypoint_flag_errors_clearly(self, tmp_path):
        manifest_path, _ = _write_manifest(tmp_path, [1])
        out_dir = tmp_path / "out"
        rc = worker.main(self._argv(manifest_path, out_dir, backend="callable"))
        assert rc != 0
        err = json.loads((out_dir / "error.json").read_text())
        assert "backend-entrypoint" in err["message"]

    def test_callable_backend_via_example_template(self, tmp_path):
        manifest_path, _ = _write_manifest(tmp_path, [1, 2])
        out_dir = tmp_path / "out"
        rc = worker.main(self._argv(
            manifest_path, out_dir, backend="callable",
            backend_entrypoint="lgvsc_example_callable_backend:generate_segment",
        ))
        assert rc == 0
        result = json.loads((out_dir / "result.json").read_text())
        assert sorted(int(k) for k in result["frames"]) == [1, 2]


# ─────────────────────────────────────────────────────────────────────────────
# real subprocess invocation (the actual CLI entrypoint, not just functions)
# ─────────────────────────────────────────────────────────────────────────────

class TestSubprocessInvocation:
    def test_real_subprocess_round_trip(self, tmp_path):
        manifest_path, _ = _write_manifest(tmp_path, [1, 2], end_idx=5, end_image=True)
        out_dir = tmp_path / "out"
        cmd = [
            sys.executable, str(_WORKER_SCRIPT),
            "--manifest", str(manifest_path), "--output-dir", str(out_dir),
            "--backend", "mock", "--seed", "7",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
        result = json.loads((out_dir / "result.json").read_text())
        assert result["backend"] == "mock"
        assert result["seed"] == 7
        assert sorted(int(k) for k in result["frames"]) == [1, 2]

    def test_real_subprocess_nonzero_exit_on_bad_manifest(self, tmp_path):
        out_dir = tmp_path / "out"
        cmd = [
            sys.executable, str(_WORKER_SCRIPT),
            "--manifest", str(tmp_path / "does_not_exist.json"),
            "--output-dir", str(out_dir), "--backend", "mock",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert proc.returncode != 0
        assert (out_dir / "error.json").exists()

    def test_real_subprocess_callable_backend_example_template(self, tmp_path):
        manifest_path, _ = _write_manifest(tmp_path, [1])
        out_dir = tmp_path / "out"
        cmd = [
            sys.executable, str(_WORKER_SCRIPT),
            "--manifest", str(manifest_path), "--output-dir", str(out_dir),
            "--backend", "callable",
            "--backend-entrypoint", "lgvsc_example_callable_backend:generate_segment",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(_REPO))
        assert proc.returncode == 0, proc.stderr
        result = json.loads((out_dir / "result.json").read_text())
        assert sorted(int(k) for k in result["frames"]) == [1]

    def test_never_imports_diffusers_for_mock_backend(self, tmp_path):
        """Guard against accidentally making the mock path depend on heavy
        model packages — run with a PYTHONPATH hack that fails loudly if
        `diffusers` is ever imported during a mock run (it isn't installed in
        ptest anyway, but this makes the invariant explicit/regression-proof
        rather than relying on an ImportError happening to occur elsewhere)."""
        manifest_path, _ = _write_manifest(tmp_path, [1])
        out_dir = tmp_path / "out"
        sitecustomize = tmp_path / "sitecustomize.py"
        sitecustomize.write_text(
            "import sys\n"
            "class _Blocker:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name == 'diffusers':\n"
            "            raise ImportError('diffusers must not be imported for backend=mock')\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Blocker())\n",
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [
            sys.executable, str(_WORKER_SCRIPT),
            "--manifest", str(manifest_path), "--output-dir", str(out_dir),
            "--backend", "mock",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
        assert proc.returncode == 0, proc.stderr
