"""utils/video_io.py – mp4 ↔ frame-folder conversion (ETRI 1차 step 1).

Bridges video files and the frame-folder representation the temporal pipeline
consumes: an mp4 input is unpacked into an ordered PNG frame folder, and
reconstructed frames can be re-assembled into an mp4 at the source fps.

Backends (auto-selected, no new hard dependency)
------------------------------------------------
1. ``cv2`` (OpenCV) when importable.
2. The system ``ffmpeg`` / ``ffprobe`` CLI otherwise (present on the dev/GPU
   hosts; encoder falls back libx264 → mpeg4 depending on the ffmpeg build).

``get_backend()`` reports which one is active; callers should degrade
gracefully (skip video IO with a warning) when it returns ``None`` so the
frame-folder path keeps working everywhere.

Note: mp4 is a lossy container — round-trips preserve frame count / fps /
resolution / order, not exact pixel values (docs/video_extension_lgvsc.md
§6.3 1단계 판정 기준).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_FRAME_PATTERN = "frame_%05d.png"


def is_video_file(path) -> bool:
    """True when *path* is an existing file with a known video extension."""
    p = Path(path)
    return p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def get_backend() -> Optional[str]:
    """Return the active video backend name: ``"cv2"``, ``"ffmpeg"`` or None."""
    if _has_cv2():
        return "cv2"
    if _has_ffmpeg():
        return "ffmpeg"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Probing
# ─────────────────────────────────────────────────────────────────────────────

def video_info(video_path) -> Dict:
    """Return ``{"fps", "width", "height"}`` for *video_path* (fps may be None)."""
    video_path = Path(video_path)
    backend = get_backend()
    if backend == "cv2":
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        return {"fps": float(fps) if fps else None, "width": width, "height": height}
    if backend == "ffmpeg":
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,width,height",
             "-of", "default=noprint_wrappers=1", str(video_path)],
            capture_output=True, text=True, check=True,
        ).stdout
        info: Dict = {"fps": None, "width": None, "height": None}
        for line in out.splitlines():
            key, _, val = line.partition("=")
            if key == "avg_frame_rate" and "/" in val:
                num, den = val.split("/")
                if float(den) > 0:
                    info["fps"] = float(num) / float(den)
            elif key in ("width", "height") and val.isdigit():
                info[key] = int(val)
        return info
    raise RuntimeError("No video backend available (need cv2 or ffmpeg/ffprobe).")


# ─────────────────────────────────────────────────────────────────────────────
# mp4 → frames
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path, out_dir) -> Dict:
    """Unpack *video_path* into ordered PNG frames under *out_dir*.

    Frames are written as ``frame_00000.png``, ``frame_00001.png``, … so their
    sorted order equals playback order (matching ``io.list_image_files``).

    Any ``frame_*.png`` left in *out_dir* by a previous extraction is removed
    first, so re-extracting a shorter video into the same directory cannot mix
    stale frames into the sequence.  Only files matching that pattern are
    touched — other files in *out_dir* are left alone.

    Returns ``{"files": [Path, ...], "fps": float|None, "n_frames": int}``.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = get_backend()
    if backend is None:
        raise RuntimeError("No video backend available (need cv2 or ffmpeg/ffprobe).")

    stale = sorted(out_dir.glob("frame_*.png"))
    for f in stale:
        f.unlink()
    if stale:
        logger.info("Removed %d stale frame_*.png from %s", len(stale), out_dir)

    info = video_info(video_path)

    if backend == "cv2":
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                cv2.imwrite(str(out_dir / (_FRAME_PATTERN % idx)), frame)
                idx += 1
        finally:
            cap.release()
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
             "-start_number", "0", str(out_dir / _FRAME_PATTERN)],
            check=True,
        )

    files = sorted(out_dir.glob("frame_*.png"))
    logger.info("Extracted %d frames from %s → %s (backend=%s, fps=%s)",
                len(files), video_path.name, out_dir, backend, info.get("fps"))
    return {"files": files, "fps": info.get("fps"), "n_frames": len(files)}


# ─────────────────────────────────────────────────────────────────────────────
# frames → mp4
# ─────────────────────────────────────────────────────────────────────────────

def write_video(frame_files: List, out_path, fps: float = 24.0) -> Path:
    """Assemble ordered image files into an mp4 at *out_path*.

    Parameters
    ----------
    frame_files:
        Ordered list of image file paths (any names; order is preserved).
    out_path:
        Target video path (parent directories are created).
    fps:
        Output frame rate (use the source fps for round-trips).
    """
    frame_files = [Path(f) for f in frame_files]
    if not frame_files:
        raise ValueError("write_video: empty frame list.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backend = get_backend()
    if backend is None:
        raise RuntimeError("No video backend available (need cv2 or ffmpeg/ffprobe).")

    if backend == "cv2":
        import cv2
        first = cv2.imread(str(frame_files[0]))
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h),
        )
        try:
            for f in frame_files:
                img = cv2.imread(str(f))
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                writer.write(img)
        finally:
            writer.release()
    else:
        # ffmpeg's image2 demuxer needs a contiguous numeric pattern; stage the
        # (arbitrarily named) frames into a temp dir as seq_%05d.png first.
        with tempfile.TemporaryDirectory(prefix="video_io_") as tmp:
            tmp_dir = Path(tmp)
            for i, f in enumerate(frame_files):
                shutil.copy(f, tmp_dir / (f"seq_{i:05d}" + f.suffix.lower()))
            suffix = frame_files[0].suffix.lower()
            pattern = str(tmp_dir / ("seq_%05d" + suffix))
            base = ["ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", f"{float(fps):g}", "-i", pattern]
            # Encoder preference: libx264 (widely playable) → mpeg4 (always in
            # stock ffmpeg builds without libx264).
            for encoder in ("libx264", "mpeg4"):
                cmd = base + ["-c:v", encoder, "-pix_fmt", "yuv420p", str(out_path)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode == 0:
                    break
            else:
                raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.strip()}")

    logger.info("Wrote video → %s (%d frames @ %.3g fps, backend=%s)",
                out_path, len(frame_files), fps, backend)
    return out_path
