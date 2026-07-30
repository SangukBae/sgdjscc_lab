"""tests/test_remeasure_video_metrics_cli.py – scripts/remeasure_video_metrics.py
CLI-level tests for the --from-recon-frames mode (ETRI 5차 OWLv2/VQA
readiness preparation).

No CLIP/OWLv2/VQA weights are a test dependency here: CLIPScoreEvaluator and
SemanticPacketExtractor construction are monkeypatched with lightweight
stand-ins so these tests stay fast and offline-safe, mirroring
tests/test_presence_backends.py's stated philosophy. The underlying
pipelines.heldout_remeasurement.items_from_recon_frame_dirs() itself is
covered directly (with the same stub-extractor approach) in
tests/test_heldout_remeasurement.py.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_script():
    path = _REPO / "scripts" / "remeasure_video_metrics.py"
    spec = importlib.util.spec_from_file_location("remeasure_video_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cli = _load_script()


class _StubPacketExtractor:
    """Duck-typed stand-in for SemanticPacketExtractor — no CLIP weights."""

    def extract(self, image, frame_id=None, caption=None):
        return {"objects": ["car"], "caption": caption or "", "meta": {"frame_id": frame_id}}


def _write_run_dir(tmp_path, n_frames=2, nested_extracted=True, with_temporal_csv=True):
    run_dir = tmp_path / "run"
    recon_dir = run_dir / "recon_frames"
    recon_dir.mkdir(parents=True)
    for i in range(n_frames):
        Image.new("RGB", (4, 4), color=(0, i * 10, 0)).save(recon_dir / f"recon_{i:05d}.png")

    extracted_root = run_dir / "extracted_frames"
    extracted_leaf = (extracted_root / "01_toy") if nested_extracted else extracted_root
    extracted_leaf.mkdir(parents=True)
    for i in range(n_frames):
        Image.new("RGB", (4, 4), color=(i * 10, 0, 0)).save(extracted_leaf / f"frame_{i:05d}.png")

    if with_temporal_csv:
        with open(run_dir / "temporal_frames.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["index", "role"])
            w.writeheader()
            for i in range(n_frames):
                w.writerow({"index": i, "role": "keyframe" if i == 0 else "inter"})

    return run_dir


class _Args:
    def __init__(self, captions=None, device=None):
        self.captions = captions
        self.device = device


# ─────────────────────────────────────────────────────────────────────────────
# _load_captions
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadCaptions:
    def test_none_when_not_given(self, tmp_path):
        files = [tmp_path / "a.png"]
        assert cli._load_captions(None, files) is None

    def test_from_txt_file(self, tmp_path):
        cap_file = tmp_path / "caps.txt"
        cap_file.write_text("first\nsecond\n", encoding="utf-8")
        files = [tmp_path / f"frame_{i:05d}.png" for i in range(3)]
        captions = cli._load_captions(str(cap_file), files)
        assert captions == ["first", "second", ""]

    def test_from_directory(self, tmp_path):
        cap_dir = tmp_path / "caps"
        cap_dir.mkdir()
        (cap_dir / "frame_00000.txt").write_text("hello", encoding="utf-8")
        files = [tmp_path / "frame_00000.png", tmp_path / "frame_00001.png"]
        captions = cli._load_captions(str(cap_dir), files)
        assert captions == ["hello", ""]

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cli._load_captions(str(tmp_path / "does_not_exist"), [])


# ─────────────────────────────────────────────────────────────────────────────
# _load_gt_metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadGtMetadata:
    def test_none_when_not_given(self):
        assert cli._load_gt_metadata(None) is None
        assert cli._load_gt_metadata("") is None

    def test_already_converted_mapping_passes_through_unchanged(self, tmp_path):
        gt_file = tmp_path / "gt_presence.json"
        gt_file.write_text('{"frame_00000": {"person": true}}', encoding="utf-8")
        assert cli._load_gt_metadata(str(gt_file)) == {"frame_00000": {"person": True}}

    def test_raw_segment_level_gt_auto_converts(self, tmp_path):
        """The exact bug this fixes: pointing --gt-metadata straight at
        data/etri_video_eval/gt/<video>.json (segment-level format, NOT
        {item_id: {...}}) must not silently produce something useless — it
        must be auto-detected and converted."""
        import json as _json
        gt_file = tmp_path / "01_toy.json"
        gt_file.write_text(_json.dumps({
            "video_id": "01_toy", "n_frames": 2,
            "segments": [
                {"start_frame": 0, "end_frame": 1,
                 "objects": [{"label": "person", "count": 1, "presence": "visible"}]},
            ],
        }), encoding="utf-8")

        result = cli._load_gt_metadata(str(gt_file))
        assert result == {
            "frame_00000": {"person": True},
            "frame_00001": {"person": True},
        }


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_recon_frame_device
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveReconFrameDevice:
    def test_explicit_device_wins(self):
        assert str(cli._resolve_recon_frame_device("cpu")) == "cpu"

    def test_defaults_to_cuda_when_available(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert str(cli._resolve_recon_frame_device(None)) == "cuda:0"

    def test_defaults_to_cpu_when_cuda_unavailable(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert str(cli._resolve_recon_frame_device(None)) == "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# _build_items_from_recon_frames
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildItemsFromReconFrames:
    def _patch_light_extractor(self, monkeypatch):
        """Replace CLIPScoreEvaluator/SemanticPacketExtractor with weight-free
        stand-ins so this test never loads real CLIP."""
        import sgdjscc_lab.evaluators.clip_score as clip_score_mod
        import sgdjscc_lab.guidance.semantic_packet_extractor as pe_mod

        class _StubClip:
            def __init__(self, *a, **kw):
                pass

        monkeypatch.setattr(clip_score_mod, "CLIPScoreEvaluator", _StubClip)
        monkeypatch.setattr(pe_mod, "SemanticPacketExtractor", lambda **kw: _StubPacketExtractor())

    def test_missing_recon_frames_dir_exits(self, tmp_path, monkeypatch):
        self._patch_light_extractor(monkeypatch)
        with pytest.raises(SystemExit):
            cli._build_items_from_recon_frames(str(tmp_path / "no_such_run"), _Args())

    def test_missing_extracted_frames_dir_exits(self, tmp_path, monkeypatch):
        self._patch_light_extractor(monkeypatch)
        run_dir = tmp_path / "run"
        (run_dir / "recon_frames").mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(run_dir / "recon_frames" / "recon_00000.png")
        with pytest.raises(SystemExit):
            cli._build_items_from_recon_frames(str(run_dir), _Args())

    def test_builds_items_with_nested_extracted_dir_and_roles(self, tmp_path, monkeypatch):
        """Reproduces outputs/etri_video_eval_real_full_step50's actual layout:
        extracted_frames/<video_stem>/frame_*.png (nested) + a temporal_frames.csv
        with a role column — both must be auto-discovered."""
        self._patch_light_extractor(monkeypatch)
        run_dir = _write_run_dir(tmp_path, n_frames=2, nested_extracted=True, with_temporal_csv=True)

        items = cli._build_items_from_recon_frames(str(run_dir), _Args())
        assert len(items) == 2
        assert items[0].role == "keyframe"
        assert items[1].role == "inter"
        assert items[0].reconstructed_image is not None
        assert items[0].reference_packet["objects"] == ["car"]

    def test_flat_extracted_dir_also_works(self, tmp_path, monkeypatch):
        self._patch_light_extractor(monkeypatch)
        run_dir = _write_run_dir(tmp_path, n_frames=2, nested_extracted=False, with_temporal_csv=False)
        items = cli._build_items_from_recon_frames(str(run_dir), _Args())
        assert len(items) == 2
        # No temporal_frames.csv → role stays None (SDI degrades gracefully, no crash).
        assert all(it.role is None for it in items)

    def test_mismatched_temporal_csv_row_count_skips_roles_without_crashing(self, tmp_path, monkeypatch):
        self._patch_light_extractor(monkeypatch)
        run_dir = _write_run_dir(tmp_path, n_frames=2, nested_extracted=False, with_temporal_csv=False)
        # A temporal_frames.csv with a different row count than the actual frames.
        with open(run_dir / "temporal_frames.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["index", "role"])
            w.writeheader()
            w.writerow({"index": 0, "role": "keyframe"})  # only 1 row, but 2 frames exist

        items = cli._build_items_from_recon_frames(str(run_dir), _Args())
        assert len(items) == 2
        assert all(it.role is None for it in items)

    def test_captions_reach_reference_packet_only(self, tmp_path, monkeypatch):
        self._patch_light_extractor(monkeypatch)
        run_dir = _write_run_dir(tmp_path, n_frames=1, nested_extracted=False, with_temporal_csv=False)
        cap_file = tmp_path / "caps.txt"
        cap_file.write_text("a red car\n", encoding="utf-8")

        items = cli._build_items_from_recon_frames(str(run_dir), _Args(captions=str(cap_file)))
        assert items[0].reference_packet["caption"] == "a red car"
        # Reconstructed-frame packet always gets caption=None in this mode
        # (see items_from_recon_frame_dirs' fidelity note).
        assert items[0].reconstructed_packet["caption"] == ""
