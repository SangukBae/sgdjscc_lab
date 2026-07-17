"""tests/test_video_io.py – ETRI 1차 video IO + evaluate_video CLI tests.

Covers:
- mp4 ↔ frame-folder round trips through utils/video_io (skipped when neither
  cv2 nor the ffmpeg CLI is available),
- the evaluate_video.py --no-models dry run end-to-end via subprocess, for both
  a frame-folder input and an mp4 input, checking the 1차 deliverables:
  temporal_metrics.csv (with ptc/sfr/sdi), temporal_frames.csv (with the gate
  decision columns), keyframes.json, segments.json, the reconstructed frame
  folder and the optional reconstructed mp4.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "evaluate_video.py"

from sgdjscc_lab.utils import video_io  # noqa: E402

_HAS_BACKEND = video_io.get_backend() is not None
needs_backend = pytest.mark.skipif(
    not _HAS_BACKEND, reason="no video backend (cv2 or ffmpeg/ffprobe) available"
)


def _write_frames(folder: Path, n: int = 6, size: int = 64):
    """Write *n* deterministic PNG frames (two colour scenes) into *folder*."""
    from torchvision.utils import save_image
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        t = torch.zeros(3, size, size)
        if i < n // 2:
            t[0] = 0.2 + 0.1 * i          # reddish scene
        else:
            t[2] = 0.2 + 0.1 * i          # bluish scene
        f = folder / f"frame_{i:05d}.png"
        save_image(t, str(f))
        files.append(f)
    return files


# ─────────────────────────────────────────────────────────────────────────────
# video_io round trip
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoIO:
    def test_is_video_file(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"x")
        assert video_io.is_video_file(f) is True
        assert video_io.is_video_file(tmp_path / "img.png") is False
        assert video_io.is_video_file(tmp_path) is False

    @needs_backend
    def test_mp4_roundtrip_preserves_structure(self, tmp_path):
        frames = _write_frames(tmp_path / "frames", n=6, size=64)
        mp4 = video_io.write_video(frames, tmp_path / "clip.mp4", fps=5.0)
        assert mp4.exists() and mp4.stat().st_size > 0

        info = video_io.video_info(mp4)
        assert info["width"] == 64 and info["height"] == 64
        assert info["fps"] == pytest.approx(5.0, rel=0.05)

        out = video_io.extract_frames(mp4, tmp_path / "extracted")
        assert out["n_frames"] == 6
        assert out["fps"] == pytest.approx(5.0, rel=0.05)
        # Ordered, contiguous naming aligned with io.list_image_files sorting.
        names = [f.name for f in out["files"]]
        assert names == [f"frame_{i:05d}.png" for i in range(6)]

    @needs_backend
    def test_write_video_empty_list_raises(self, tmp_path):
        with pytest.raises(ValueError):
            video_io.write_video([], tmp_path / "x.mp4", fps=5.0)

    @needs_backend
    def test_reextraction_clears_stale_frames(self, tmp_path):
        """Extracting a shorter mp4 into the same out_dir must not leave frames
        from a previous, longer extraction behind."""
        long_clip = video_io.write_video(
            _write_frames(tmp_path / "long", n=6), tmp_path / "long.mp4", fps=5.0)
        short_clip = video_io.write_video(
            _write_frames(tmp_path / "short", n=3), tmp_path / "short.mp4", fps=5.0)

        out_dir = tmp_path / "extracted"
        first = video_io.extract_frames(long_clip, out_dir)
        assert first["n_frames"] == 6

        # A non-frame file in the same dir must survive the cleanup.
        keep = out_dir / "notes.txt"
        keep.write_text("keep me", encoding="utf-8")

        second = video_io.extract_frames(short_clip, out_dir)
        assert second["n_frames"] == 3
        remaining = sorted(out_dir.glob("frame_*.png"))
        assert [f.name for f in remaining] == [f"frame_{i:05d}.png" for i in range(3)]
        assert keep.read_text(encoding="utf-8") == "keep me"


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_video.py dry runs (subprocess, --no-models)
# ─────────────────────────────────────────────────────────────────────────────

def _write_cfg(tmp_path: Path, motion_threshold=None) -> Path:
    out = tmp_path / "out"
    cfg = f"""
use_phase4: true
device: cpu
scene_change:
  threshold: 0.35
keyframe:
  max_gop: 12
temporal:
  reuse_threshold: 0.2
  motion_threshold: {"null" if motion_threshold is None else motion_threshold}
keyframe_json: "{out}/keyframes.json"
segment_json: "{out}/segments.json"
temporal_csv: "{out}/temporal_metrics.csv"
frame_log_csv: "{out}/temporal_frames.csv"
video_io:
  extracted_frames_dir: "{out}/video_frames"
  recon_frames_dir: "{out}/recon_frames"
  save_recon_frames: true
  save_recon_video: false
  recon_video: "{out}/recon.mp4"
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(cfg, encoding="utf-8")
    return p


def _run_script(cfg: Path, input_path: Path, *extra):
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--config", str(cfg),
         "--input", str(input_path), "--no-models", *extra],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    return proc


class TestEvaluateVideoDryRun:
    def test_frame_folder_dry_run_produces_1cha_artifacts(self, tmp_path):
        frames = _write_frames(tmp_path / "frames", n=6)
        cfg = _write_cfg(tmp_path)
        _run_script(cfg, tmp_path / "frames")
        out = tmp_path / "out"

        # temporal_metrics.csv exists and carries the provisional ptc/sfr/sdi.
        with open(out / "temporal_metrics.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        for col in ("ptc", "sfr", "sdi", "temporal_srs", "n_frames"):
            assert col in rows[0]

        # Per-frame log has the gate decision + motion columns.
        with open(out / "temporal_frames.csv", newline="", encoding="utf-8") as fh:
            flog = list(csv.DictReader(fh))
        assert len(flog) == 6
        for col in ("decision", "motion_score"):
            assert col in flog[0]

        # Keyframe + segment structure JSONs.
        structure = json.loads((out / "keyframes.json").read_text(encoding="utf-8"))
        assert structure["keyframes"]
        segments = json.loads((out / "segments.json").read_text(encoding="utf-8"))
        assert len(segments) == len(structure["keyframes"])
        assert all(seg["generation"] is None for seg in segments)

        # Reconstructed frame folder (identity recon in the dry run).
        recon = sorted((out / "recon_frames").glob("recon_*.png"))
        assert len(recon) == 6

    def test_stale_recon_frames_cleared_before_run(self, tmp_path):
        """recon_frames_dir may hold recon_*.png from an earlier (longer) run;
        after a dry run only the current input's frames must remain."""
        _write_frames(tmp_path / "frames", n=4)
        cfg = _write_cfg(tmp_path)

        recon_dir = tmp_path / "out" / "recon_frames"
        recon_dir.mkdir(parents=True)
        # Simulate a previous 10-frame run + an unrelated file.
        for i in range(10):
            (recon_dir / f"recon_{i:05d}.png").write_bytes(b"stale")
        keep = recon_dir / "notes.txt"
        keep.write_text("keep me", encoding="utf-8")

        _run_script(cfg, tmp_path / "frames")

        remaining = sorted(recon_dir.glob("recon_*.png"))
        assert [f.name for f in remaining] == [f"recon_{i:05d}.png" for i in range(4)]
        # Current-run frames are real PNGs, not the stale placeholders.
        assert all(f.stat().st_size > len(b"stale") for f in remaining)
        assert keep.read_text(encoding="utf-8") == "keep me"

    @needs_backend
    def test_mp4_input_dry_run_with_video_output(self, tmp_path):
        frames = _write_frames(tmp_path / "frames", n=6)
        clip = video_io.write_video(frames, tmp_path / "clip.mp4", fps=5.0)
        cfg = _write_cfg(tmp_path)
        _run_script(cfg, clip, "--save-video")
        out = tmp_path / "out"

        # mp4 was unpacked into ordered frames and processed frame-wise.
        extracted = sorted((out / "video_frames" / "clip").glob("frame_*.png"))
        assert len(extracted) == 6
        with open(out / "temporal_frames.csv", newline="", encoding="utf-8") as fh:
            assert len(list(csv.DictReader(fh))) == 6

        # Reconstruction was re-assembled into an mp4 at the source fps.
        recon_mp4 = out / "recon.mp4"
        assert recon_mp4.exists() and recon_mp4.stat().st_size > 0
        info = video_io.video_info(recon_mp4)
        assert info["fps"] == pytest.approx(5.0, rel=0.05)
