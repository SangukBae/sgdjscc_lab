#!/usr/bin/env python3
"""Benchmark ETRI semantic-video payloads against conventional codecs.

This script answers four questions for the ten videos in
``data/etri_video_eval/manifest.csv``:

1. How large are the raw source and the normalized experiment MP4?
2. What receiver inputs did a completed semantic-generation run require?
3. How large are H.264, H.265, and AV1 encodes of the same 100 frames?
4. What PSNR, SSIM, and LPIPS quality does each decoded video achieve?

Important accounting boundary
-----------------------------
The completed SGD-JSCC/Wan runs did not persist an on-wire bitstream.  An AWGN
JSCC channel transmits real-valued channel symbols, so an exact network byte
count cannot be recovered from the saved PNG/JSON results.  The script keeps
three quantities separate so they cannot be confused:

``reference_payload_bytes``
    Exact size of a deterministic reference payload built from the selected
    source keyframe PNGs, per-segment UTF-8 captions, optional persisted side
    information, and a compact manifest.  This is a reproducible file payload,
    not the original JSCC wire format.

``jscc_visual_channel_symbols_proxy``
    Visual latent symbol count inferred from the fixed repository architecture
    (4096 symbols per 128x128 patch) and the actual selected keyframes.

``estimated_wire_bytes``
    Only populated when the caller supplies ``--bits-per-channel-symbol``.
    It applies that explicit assumption to the symbol proxy and adds the exact
    non-visual payload bytes.  It remains an estimate, never an exact bitstream.

Conventional codec sizes are exact encoded file sizes.  Codec quality and the
semantic reconstruction quality are measured with the same reference video.

Example
-------
python scripts/benchmark_etri_video_rate.py \
  --semantic-run-root outputs/remote_hq_4090_20260816/generation/wan_skem_dsa \
  --output-root outputs/remote_hq_4090_20260816/rate_benchmark \
  --lpips-device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

REFERENCE_PAYLOAD_MAGIC = b"SGDPAY1\0"
PATCH_SIZE = 128
LATENT_SYMBOLS_PER_PATCH = 4096


class CodecSpec(NamedTuple):
    name: str
    encoder: str
    extension: str
    default_crfs: Tuple[int, ...]
    encode_args: Tuple[str, ...]


CODECS: Dict[str, CodecSpec] = {
    "h264": CodecSpec(
        "h264", "libx264", ".mp4", (18, 23, 28, 33),
        ("-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart"),
    ),
    "h265": CodecSpec(
        "h265", "libx265", ".mp4", (18, 23, 28, 33),
        ("-preset", "medium", "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
         "-x265-params", "log-level=error"),
    ),
    "av1": CodecSpec(
        "av1", "libaom-av1", ".mkv", (20, 30, 40, 50),
        ("-b:v", "0", "-cpu-used", "6", "-row-mt", "1", "-pix_fmt", "yuv420p"),
    ),
}


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure 10-video semantic payloads and H.264/H.265/AV1 rate-quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", default=str(_REPO_ROOT / "data/etri_video_eval"))
    parser.add_argument(
        "--semantic-run-root", required=True,
        help="Completed mode root containing <video_id>/{keyframes.json,segments.json,recon.mp4}.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--video-ids", default=None,
                        help="Comma-separated subset such as 01_person_walk,02_car_pass.")
    parser.add_argument("--codecs", default="h264,h265,av1")
    parser.add_argument(
        "--crf", action="append", default=[], metavar="CODEC=LIST",
        help="Override CRFs, e.g. --crf h264=18,23,28 --crf av1=20,30,40.",
    )
    parser.add_argument("--side-info-root", default=None,
                        help="Optional root containing <video_id>/ files actually sent as side info.")
    parser.add_argument("--bits-per-channel-symbol", type=float, default=None,
                        help="Optional explicit assumption used only for estimated_wire_bytes.")
    parser.add_argument("--lpips-device", default="cuda:0")
    parser.add_argument("--lpips-net", choices=("vgg", "alex"), default="vgg")
    parser.add_argument("--lpips-batch-size", type=int, default=8)
    parser.add_argument("--lpips-stride", type=int, default=1,
                        help="Measure every Nth frame; 1 uses all 100 frames.")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode codec files even when the destination already exists.")
    parser.add_argument("--skip-unavailable-codecs", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and write the plan without encoding or metrics.")
    parser.add_argument(
        "--ffmpeg", default="auto",
        help="FFmpeg executable or 'auto'. Auto skips Conda builds missing requested encoders.",
    )
    parser.add_argument(
        "--ffprobe", default="auto",
        help="FFprobe executable or 'auto' (paired with the selected FFmpeg when possible).",
    )
    return parser.parse_args(argv)


def _compact_json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_manifest(dataset_root: Path, selected_ids: Optional[Sequence[str]] = None) -> List[dict]:
    dataset_root = Path(dataset_root).resolve()
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    selected = set(selected_ids or [])
    rows: List[dict] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            video_id = f"{source['id']}_{source['name']}"
            if selected and video_id not in selected:
                continue
            row = dict(source)
            row.update({
                "video_id": video_id,
                "raw_path": (dataset_root / source["raw_file"]).resolve(),
                "processed_path": (dataset_root / source["processed_file"]).resolve(),
                "frames_path": (dataset_root / source["frames_dir"]).resolve(),
                "caption_path": (dataset_root / "captions" / f"{video_id}.txt").resolve(),
            })
            for required in ("raw_path", "processed_path", "frames_path", "caption_path"):
                if not Path(row[required]).exists():
                    raise FileNotFoundError(f"{video_id}: missing {required}: {row[required]}")
            rows.append(row)
    if selected:
        missing = selected - {row["video_id"] for row in rows}
        if missing:
            raise ValueError(f"Unknown --video-ids: {', '.join(sorted(missing))}")
    if not rows:
        raise ValueError("Manifest selection is empty")
    return rows


def parse_crf_overrides(values: Sequence[str]) -> Dict[str, Tuple[int, ...]]:
    result = {name: spec.default_crfs for name, spec in CODECS.items()}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --crf {value!r}; expected CODEC=18,23,28")
        name, raw = value.split("=", 1)
        name = name.strip().lower()
        if name not in CODECS:
            raise ValueError(f"Unknown codec in --crf: {name}")
        try:
            crfs = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid CRF list: {value}") from exc
        if not crfs:
            raise ValueError(f"Empty CRF list: {value}")
        result[name] = crfs
    return result


def ffprobe_video(path: Path, ffprobe: str = "ffprobe") -> dict:
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,bit_rate,duration:format=duration,bit_rate,size",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream: {path}")
    stream = streams[0]
    fmt = payload.get("format") or {}
    duration = _float_or_none(stream.get("duration")) or _float_or_none(fmt.get("duration"))
    n_frames = _int_or_none(stream.get("nb_frames"))
    fps = _fraction_or_none(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    if n_frames is None and duration is not None and fps is not None:
        n_frames = int(round(duration * fps))
    size_bytes = path.stat().st_size
    bitrate = (size_bytes * 8.0 / duration) if duration else _float_or_none(fmt.get("bit_rate"))
    width = int(stream["width"])
    height = int(stream["height"])
    bpp = (size_bytes * 8.0 / (width * height * n_frames)) if n_frames else None
    return {
        "codec_name": stream.get("codec_name"),
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": duration,
        "n_frames": n_frames,
        "size_bytes": size_bytes,
        "bitrate_bps": bitrate,
        "bits_per_pixel": bpp,
    }


def _float_or_none(value) -> Optional[float]:
    try:
        return float(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value) -> Optional[int]:
    try:
        return int(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def _fraction_or_none(value) -> Optional[float]:
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        if "/" in str(value):
            a, b = str(value).split("/", 1)
            return float(a) / float(b)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def available_encoders(ffmpeg: str = "ffmpeg") -> set:
    proc = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], check=True,
                          capture_output=True, text=True)
    names = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return names


def resolve_ffmpeg_tools(
    ffmpeg: str,
    ffprobe: str,
    required_encoders: Sequence[str],
) -> Tuple[str, str, set]:
    """Select an FFmpeg/FFprobe pair that provides all requested encoders.

    The research container prepends ``/opt/ptest/bin`` to ``PATH``.  Its Conda
    FFmpeg build is non-GPL and therefore lacks libx264/libx265, while the
    Ubuntu system build under ``/usr/bin`` provides both.  ``auto`` tests the
    PATH pair first and then common system/Conda locations instead of failing
    on the first executable found.
    """
    required = set(required_encoders)
    candidates: List[Tuple[str, str]] = []

    if ffmpeg != "auto":
        if ffprobe == "auto":
            executable = Path(ffmpeg)
            sibling = executable.with_name("ffprobe") if executable.parent != Path(".") else None
            selected_probe = str(sibling) if sibling is not None and sibling.is_file() else "ffprobe"
        else:
            selected_probe = ffprobe
        candidates.append((ffmpeg, selected_probe))
    else:
        path_ffmpeg = shutil.which("ffmpeg")
        path_ffprobe = shutil.which("ffprobe")
        if path_ffmpeg and path_ffprobe:
            candidates.append((path_ffmpeg, path_ffprobe))
        for base in (Path("/usr/bin"), Path("/usr/local/bin"), Path("/opt/ptest/bin")):
            executable = base / "ffmpeg"
            probe = base / "ffprobe"
            if executable.is_file() and probe.is_file():
                candidates.append((str(executable), str(probe)))

    unique: List[Tuple[str, str]] = []
    seen = set()
    for pair in candidates:
        resolved = tuple(str(Path(value).resolve()) if Path(value).exists() else value for value in pair)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(pair)

    inspected = []
    for executable, probe in unique:
        try:
            encoders = available_encoders(executable)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            inspected.append(f"{executable}: unavailable ({exc})")
            continue
        missing = sorted(required - encoders)
        inspected.append(
            f"{executable}: " + ("compatible" if not missing else f"missing {','.join(missing)}")
        )
        if not missing:
            if shutil.which(probe) is None and not Path(probe).is_file():
                inspected[-1] += f"; ffprobe unavailable: {probe}"
                continue
            return executable, probe, encoders

    details = "; ".join(inspected) if inspected else "no FFmpeg candidates found"
    raise RuntimeError(
        "No FFmpeg build provides all requested encoders "
        f"({', '.join(sorted(required))}). Inspected: {details}. "
        "On Ubuntu install the GPL-enabled system build with: apt-get update && "
        "apt-get install -y ffmpeg"
    )


def build_encode_command(
    input_path: Path, output_path: Path, spec: CodecSpec, crf: int,
    ffmpeg: str = "ffmpeg",
) -> List[str]:
    return [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path),
        "-map", "0:v:0", "-an", "-c:v", spec.encoder, "-crf", str(crf),
        *spec.encode_args, str(output_path),
    ]


def encode_video(
    input_path: Path, output_path: Path, spec: CodecSpec, crf: int,
    *, ffmpeg: str = "ffmpeg", overwrite: bool = False,
) -> None:
    if output_path.is_file() and output_path.stat().st_size > 0 and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_encode_command(input_path, output_path, spec, crf, ffmpeg), check=True)


_PSNR_RE = re.compile(r"average:([-+0-9.eEinfINF]+)")
_SSIM_RE = re.compile(r"All:([-+0-9.eE]+)")


def _run_filter_metric(
    reference_path: Path, candidate_path: Path, metric: str, ffmpeg: str,
) -> float:
    cmd = [
        ffmpeg, "-hide_banner", "-nostats", "-i", str(candidate_path),
        "-i", str(reference_path), "-lavfi", f"[0:v][1:v]{metric}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    pattern = _PSNR_RE if metric == "psnr" else _SSIM_RE
    matches = pattern.findall(proc.stderr)
    if not matches:
        raise RuntimeError(f"Could not parse {metric} from ffmpeg output for {candidate_path}")
    token = matches[-1]
    return float("inf") if token.lower() in ("inf", "+inf") else float(token)


def measure_psnr_ssim(
    reference_path: Path, candidate_path: Path, ffmpeg: str = "ffmpeg",
) -> Tuple[float, float]:
    return (
        _run_filter_metric(reference_path, candidate_path, "psnr", ffmpeg),
        _run_filter_metric(reference_path, candidate_path, "ssim", ffmpeg),
    )


class LpipsEvaluator:
    """Persistent batched LPIPS evaluator; imports GPU dependencies lazily."""

    def __init__(self, device: str, net: str, batch_size: int, stride: int) -> None:
        if batch_size <= 0 or stride <= 0:
            raise ValueError("LPIPS batch size and stride must be positive")
        import lpips
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.stride = stride
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    @staticmethod
    def _load_tensor(path: Path):
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def evaluate(self, reference_frames: Sequence[Path], candidate_frames: Sequence[Path]) -> dict:
        if len(reference_frames) != len(candidate_frames):
            raise ValueError(
                f"LPIPS frame count mismatch: {len(reference_frames)} vs {len(candidate_frames)}"
            )
        pairs = list(zip(reference_frames, candidate_frames))[::self.stride]
        values: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            ref = self.torch.stack([self._load_tensor(p) for p, _ in batch]).to(self.device)
            cand = self.torch.stack([self._load_tensor(p) for _, p in batch]).to(self.device)
            if ref.shape != cand.shape:
                raise ValueError(f"LPIPS shape mismatch: {tuple(ref.shape)} vs {tuple(cand.shape)}")
            with self.torch.no_grad():
                distances = self.model(ref * 2.0 - 1.0, cand * 2.0 - 1.0)
            values.extend(float(v) for v in distances.detach().cpu().reshape(-1).tolist())
        return {
            "lpips_mean": (sum(values) / len(values)) if values else None,
            "lpips_n_frames": len(values),
            "lpips_stride": self.stride,
        }


def decode_frames(video_path: Path, output_dir: Path, ffmpeg: str = "ffmpeg") -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vsync", "0", str(output_dir / "frame_%05d.png"),
    ], check=True)
    files = sorted(output_dir.glob("frame_*.png"))
    if not files:
        raise RuntimeError(f"No decoded frames from {video_path}")
    return files


def measure_lpips(
    reference_frames_dir: Path, candidate_path: Path, evaluator: LpipsEvaluator,
    ffmpeg: str = "ffmpeg",
) -> dict:
    reference_frames = sorted(Path(reference_frames_dir).glob("frame_*.png"))
    if not reference_frames:
        raise FileNotFoundError(f"No reference frames: {reference_frames_dir}")
    with tempfile.TemporaryDirectory(prefix="etri-rate-lpips-") as tmp:
        candidate_frames = decode_frames(candidate_path, Path(tmp), ffmpeg)
        return evaluator.evaluate(reference_frames, candidate_frames)


def _generation_flags(segments: Sequence[dict]) -> Tuple[bool, bool]:
    used_caption = False
    used_side_info = False
    for segment in segments:
        generation = segment.get("generation") or {}
        for frame in generation.get("frames") or []:
            used_caption = used_caption or bool(frame.get("used_caption"))
            used_side_info = used_side_info or bool(frame.get("used_side_info"))
    return used_caption, used_side_info


def _caption_records(caption_path: Path, segments: Sequence[dict], used_caption: bool) -> List[dict]:
    if not used_caption:
        return []
    captions = caption_path.read_text(encoding="utf-8").splitlines()
    records = []
    for segment in segments:
        frame_index = int(segment.get("keyframe_index", 0))
        caption = captions[frame_index].strip() if frame_index < len(captions) else ""
        records.append({
            "segment_id": int(segment.get("segment_id", len(records))),
            "frame_index": frame_index,
            "caption": caption,
        })
    return records


def _side_info_files(side_info_root: Optional[Path], video_id: str) -> List[Path]:
    if side_info_root is None:
        return []
    root = Path(side_info_root) / video_id
    return sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []


def write_reference_payload(path: Path, items: Sequence[Tuple[str, bytes]]) -> int:
    """Write deterministic length-prefixed payload and return its exact size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(REFERENCE_PAYLOAD_MAGIC)
        handle.write(struct.pack(">I", len(items)))
        for name, data in items:
            encoded_name = name.encode("utf-8")
            handle.write(struct.pack(">I", len(encoded_name)))
            handle.write(encoded_name)
            handle.write(struct.pack(">Q", len(data)))
            handle.write(data)
    return path.stat().st_size


def build_payload(
    entry: dict,
    run_dir: Path,
    payload_path: Path,
    *,
    side_info_root: Optional[Path] = None,
    bits_per_channel_symbol: Optional[float] = None,
) -> dict:
    keyframes_path = run_dir / "keyframes.json"
    segments_path = run_dir / "segments.json"
    recon_path = run_dir / "recon.mp4"
    for path in (keyframes_path, segments_path, recon_path):
        if not path.is_file():
            raise FileNotFoundError(f"{entry['video_id']}: missing semantic result {path}")

    keyframe_data = json.loads(keyframes_path.read_text(encoding="utf-8"))
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    keyframe_indices = sorted({int(v) for v in keyframe_data.get("keyframes") or []})
    if not keyframe_indices:
        raise ValueError(f"{entry['video_id']}: keyframes.json has no keyframes")
    used_caption, used_side_info = _generation_flags(segments)
    captions = _caption_records(entry["caption_path"], segments, used_caption)
    caption_text_bytes = sum(len(row["caption"].encode("utf-8")) for row in captions)
    captions_blob = _compact_json_bytes(captions)

    items: List[Tuple[str, bytes]] = []
    keyframe_bytes = 0
    components = []
    for index in keyframe_indices:
        source_path = Path(entry["frames_path"]) / f"frame_{index:05d}.png"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing selected keyframe: {source_path}")
        data = source_path.read_bytes()
        name = f"keyframes/frame_{index:05d}.png"
        items.append((name, data))
        keyframe_bytes += len(data)
        components.append({"name": name, "size_bytes": len(data), "sha256": _sha256(data)})

    side_files = _side_info_files(side_info_root, entry["video_id"])
    side_info_bytes = 0
    for source_path in side_files:
        base = Path(side_info_root) / entry["video_id"]
        name = f"side_info/{source_path.relative_to(base).as_posix()}"
        data = source_path.read_bytes()
        items.append((name, data))
        side_info_bytes += len(data)
        components.append({"name": name, "size_bytes": len(data), "sha256": _sha256(data)})

    items.append(("captions.json", captions_blob))
    components.append({
        "name": "captions.json", "size_bytes": len(captions_blob),
        "sha256": _sha256(captions_blob),
    })
    metadata = {
        "format": "sgdjscc_reference_payload_v1",
        "video_id": entry["video_id"],
        "source": "selected source PNG keyframes + per-segment captions + optional side-info files",
        "not_wire_format": True,
        "width": int(entry["width"]),
        "height": int(entry["height"]),
        "fps": float(entry["fps"]),
        "n_frames": int(entry["n_frames"]),
        "keyframe_indices": keyframe_indices,
        "used_caption": used_caption,
        "used_side_info_reported_by_generator": used_side_info,
        "side_info_files_supplied": bool(side_files),
        "components": components,
    }
    manifest_blob = _compact_json_bytes(metadata)
    items.insert(0, ("manifest.json", manifest_blob))
    payload_bytes = write_reference_payload(payload_path, items)

    width = int(entry["width"])
    height = int(entry["height"])
    patches_per_frame = math.ceil(width / PATCH_SIZE) * math.ceil(height / PATCH_SIZE)
    symbols_per_frame = patches_per_frame * LATENT_SYMBOLS_PER_PATCH
    visual_symbols = symbols_per_frame * len(keyframe_indices)
    exact_non_visual_bytes = len(captions_blob) + side_info_bytes + len(manifest_blob)
    estimated_wire_bytes = None
    if bits_per_channel_symbol is not None:
        estimated_wire_bytes = math.ceil(
            (visual_symbols * bits_per_channel_symbol) / 8.0
        ) + exact_non_visual_bytes

    return {
        "video_id": entry["video_id"],
        "semantic_mode": run_dir.parent.name,
        "n_frames": int(entry["n_frames"]),
        "n_keyframes": len(keyframe_indices),
        "keyframe_indices": json.dumps(keyframe_indices, separators=(",", ":")),
        "n_segments": len(segments),
        "used_caption": used_caption,
        "used_side_info_reported": used_side_info,
        "side_info_status": (
            "persisted_and_counted" if side_files else
            "required_but_missing" if used_side_info else
            "not_used"
        ),
        "keyframe_png_bytes": keyframe_bytes,
        "caption_text_utf8_bytes": caption_text_bytes,
        "caption_json_bytes": len(captions_blob),
        "side_info_bytes": side_info_bytes,
        "payload_manifest_bytes": len(manifest_blob),
        "reference_payload_bytes": payload_bytes,
        "reference_payload_path": str(payload_path.resolve()),
        "jscc_symbols_per_keyframe_proxy": symbols_per_frame,
        "jscc_visual_channel_symbols_proxy": visual_symbols,
        "jscc_float32_storage_bytes_proxy": visual_symbols * 4 + exact_non_visual_bytes,
        "bits_per_channel_symbol_assumption": bits_per_channel_symbol,
        "estimated_wire_bytes": estimated_wire_bytes,
        "exact_network_bitstream_available": False,
        "recon_video_path": str(recon_path.resolve()),
        "recon_video_file_bytes_not_payload": recon_path.stat().st_size,
    }


def _quality_row(
    reference_path: Path,
    candidate_path: Path,
    reference_frames: Path,
    *,
    ffmpeg: str,
    lpips_evaluator: Optional[LpipsEvaluator],
) -> dict:
    psnr, ssim = measure_psnr_ssim(reference_path, candidate_path, ffmpeg)
    result = {"psnr": psnr, "ssim": ssim, "lpips": None,
              "lpips_n_frames": 0, "lpips_stride": None}
    if lpips_evaluator is not None:
        result.update(measure_lpips(reference_frames, candidate_path, lpips_evaluator, ffmpeg))
        result["lpips"] = result.pop("lpips_mean")
    return result


def _base_rate_row(video_id: str, method: str, setting: str, path: Path, probe: dict) -> dict:
    return {
        "video_id": video_id,
        "method": method,
        "setting": setting,
        "status": "ok",
        "size_kind": "encoded_file",
        "size_bytes": probe["size_bytes"],
        "bitrate_bps": probe["bitrate_bps"],
        "bits_per_pixel": probe["bits_per_pixel"],
        "width": probe["width"],
        "height": probe["height"],
        "fps": probe["fps"],
        "duration_sec": probe["duration_sec"],
        "n_frames": probe["n_frames"],
        "codec_name": probe["codec_name"],
        "path": str(path.resolve()),
    }


def summarize_comparison(rows: Sequence[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row.get("method")), str(row.get("setting")), str(row.get("size_kind")))
        groups.setdefault(key, []).append(row)
    result = []
    for (method, setting, size_kind), items in sorted(groups.items()):
        out = {
            "method": method,
            "setting": setting,
            "size_kind": size_kind,
            "n_videos": len(items),
            "total_size_bytes": sum(float(row["size_bytes"]) for row in items),
        }
        for field in ("size_bytes", "bitrate_bps", "bits_per_pixel", "psnr", "ssim", "lpips"):
            values = [_float_or_none(row.get(field)) for row in items]
            finite = [v for v in values if v is not None and math.isfinite(v)]
            out[f"mean_{field}"] = (sum(finite) / len(finite)) if finite else None
        result.append(out)
    return result


def match_codec_quality(rows: Sequence[dict], metric: str = "psnr") -> List[dict]:
    """Pick the smallest codec file meeting semantic quality, else nearest point."""
    semantic = {
        row["video_id"]: row for row in rows
        if row.get("method", "").startswith("semantic_") and row.get("status") == "ok"
    }
    output = []
    codec_names = sorted({row.get("method") for row in rows if row.get("method") in CODECS})
    for video_id, target in sorted(semantic.items()):
        target_value = _float_or_none(target.get(metric))
        if target_value is None:
            continue
        for codec in codec_names:
            candidates = [row for row in rows if row.get("video_id") == video_id
                          and row.get("method") == codec and row.get("status") == "ok"
                          and _float_or_none(row.get(metric)) is not None]
            meets = [row for row in candidates if float(row[metric]) >= target_value]
            if meets:
                chosen = min(meets, key=lambda row: float(row["size_bytes"]))
                match_status = "quality_at_least_target"
            elif candidates:
                chosen = min(candidates, key=lambda row: abs(float(row[metric]) - target_value))
                match_status = "closest_available"
            else:
                continue
            semantic_size = float(target["size_bytes"])
            codec_size = float(chosen["size_bytes"])
            output.append({
                "video_id": video_id,
                "metric": metric,
                "semantic_method": target["method"],
                "semantic_quality": target_value,
                "semantic_reference_payload_bytes": semantic_size,
                "codec": codec,
                "codec_setting": chosen["setting"],
                "codec_quality": float(chosen[metric]),
                "codec_size_bytes": codec_size,
                "codec_over_semantic_payload_ratio": codec_size / semantic_size if semantic_size else None,
                "match_status": match_status,
            })
    return output


def _write_methodology(path: Path, args: argparse.Namespace, codec_crfs: Dict[str, Tuple[int, ...]]) -> None:
    text = f"""# ETRI 10-video rate/quality benchmark

Generated by `scripts/benchmark_etri_video_rate.py`.

## Inputs

- Dataset: `{Path(args.dataset_root).resolve()}`
- Semantic runs: `{Path(args.semantic_run_root).resolve()}`
- Codecs/CRFs: `{json.dumps({k: list(v) for k, v in codec_crfs.items()}, sort_keys=True)}`
- LPIPS: `{'disabled' if args.no_lpips else args.lpips_net + ' on ' + args.lpips_device}`
- LPIPS stride: `{args.lpips_stride}`

## Interpretation

- `source_sizes.csv`: exact raw and normalized input file sizes.
- `payloads.csv`: exact deterministic reference-payload size plus separate JSCC symbol proxies.
- `codec_results.csv`: exact codec file sizes and decoded PSNR/SSIM/LPIPS.
- `comparison.csv`: original, semantic reconstruction, and codec rows in one schema.
- `aggregate.csv`: ten-video method averages/totals.
- `quality_matched.csv`: smallest sampled codec point whose PSNR is at least the semantic result;
  if none exists, the nearest sampled point is explicitly marked `closest_available`.

`reference_payload_bytes` is not an original JSCC network bitstream. It is the exact size of
a documented, deterministic container holding selected source keyframe PNGs, per-segment
captions, optional persisted side information, and a manifest. `jscc_visual_channel_symbols_proxy`
is reported separately. `estimated_wire_bytes` is only present when an explicit
`--bits-per-channel-symbol` assumption is supplied.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    dataset_root = Path(args.dataset_root).resolve()
    semantic_root = Path(args.semantic_run_root).resolve()
    output_root = Path(args.output_root).resolve()
    selected_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()] if args.video_ids else None
    entries = read_manifest(dataset_root, selected_ids)

    codec_names = [name.strip().lower() for name in args.codecs.split(",") if name.strip()]
    unknown = set(codec_names) - set(CODECS)
    if unknown:
        raise ValueError(f"Unknown codecs: {', '.join(sorted(unknown))}")
    all_crfs = parse_crf_overrides(args.crf)
    codec_crfs = {name: all_crfs[name] for name in codec_names}
    output_root.mkdir(parents=True, exist_ok=True)

    plan = {
        "videos": [entry["video_id"] for entry in entries],
        "codecs": {name: list(codec_crfs[name]) for name in codec_names},
        "n_encodes": len(entries) * sum(len(v) for v in codec_crfs.values()),
        "lpips_enabled": not args.no_lpips,
        "dry_run": args.dry_run,
    }
    (output_root / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.dry_run:
        _write_methodology(output_root / "README.md", args, codec_crfs)
        return plan

    required_encoders = (
        [] if args.skip_unavailable_codecs
        else [CODECS[name].encoder for name in codec_names]
    )
    selected_ffmpeg, selected_ffprobe, encoders = resolve_ffmpeg_tools(
        args.ffmpeg, args.ffprobe, required_encoders,
    )
    args.ffmpeg = selected_ffmpeg
    args.ffprobe = selected_ffprobe
    plan["ffmpeg"] = selected_ffmpeg
    plan["ffprobe"] = selected_ffprobe
    (output_root / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    _write_methodology(output_root / "README.md", args, codec_crfs)

    unavailable = [name for name in codec_names if CODECS[name].encoder not in encoders]
    if unavailable and not args.skip_unavailable_codecs:
        details = ", ".join(f"{name}({CODECS[name].encoder})" for name in unavailable)
        raise RuntimeError(f"Required ffmpeg encoders unavailable: {details}")
    active_codecs = [name for name in codec_names if name not in unavailable]

    lpips_evaluator = None
    if not args.no_lpips:
        try:
            lpips_evaluator = LpipsEvaluator(
                args.lpips_device, args.lpips_net, args.lpips_batch_size, args.lpips_stride,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "LPIPS dependencies are unavailable. Run inside the research container "
                "with torch/lpips installed, or use --no-lpips for a codec-only smoke test."
            ) from exc

    source_rows: List[dict] = []
    payload_rows: List[dict] = []
    codec_rows: List[dict] = []
    comparison_rows: List[dict] = []

    for entry in entries:
        video_id = entry["video_id"]
        raw_probe = ffprobe_video(entry["raw_path"], args.ffprobe)
        processed_probe = ffprobe_video(entry["processed_path"], args.ffprobe)
        source_rows.append({
            "video_id": video_id,
            "raw_path": str(entry["raw_path"]),
            "raw_size_bytes": raw_probe["size_bytes"],
            "raw_codec": raw_probe["codec_name"],
            "processed_path": str(entry["processed_path"]),
            "processed_size_bytes": processed_probe["size_bytes"],
            "processed_codec": processed_probe["codec_name"],
            "width": processed_probe["width"], "height": processed_probe["height"],
            "fps": processed_probe["fps"], "n_frames": processed_probe["n_frames"],
            "duration_sec": processed_probe["duration_sec"],
        })
        original_row = _base_rate_row(
            video_id, "original_processed", "existing_h264", entry["processed_path"], processed_probe,
        )
        original_row.update({"psnr": None, "ssim": None, "lpips": None,
                             "note": "Reference input; quality against itself is omitted."})
        comparison_rows.append(original_row)

        run_dir = semantic_root / video_id
        payload = build_payload(
            entry, run_dir, output_root / "payloads" / f"{video_id}.sgdref",
            side_info_root=(Path(args.side_info_root).resolve() if args.side_info_root else None),
            bits_per_channel_symbol=args.bits_per_channel_symbol,
        )
        payload_rows.append(payload)
        semantic_probe = ffprobe_video(run_dir / "recon.mp4", args.ffprobe)
        semantic_quality = _quality_row(
            entry["processed_path"], run_dir / "recon.mp4", entry["frames_path"],
            ffmpeg=args.ffmpeg, lpips_evaluator=lpips_evaluator,
        )
        semantic_row = {
            "video_id": video_id,
            "method": f"semantic_{payload['semantic_mode']}",
            "setting": "reference_payload_v1",
            "status": "ok",
            "size_kind": "reference_serialized_payload_not_jscc_wire",
            "size_bytes": payload["reference_payload_bytes"],
            "bitrate_bps": payload["reference_payload_bytes"] * 8.0 / processed_probe["duration_sec"],
            "bits_per_pixel": payload["reference_payload_bytes"] * 8.0 /
                              (processed_probe["width"] * processed_probe["height"] * processed_probe["n_frames"]),
            "width": processed_probe["width"], "height": processed_probe["height"],
            "fps": processed_probe["fps"], "duration_sec": processed_probe["duration_sec"],
            "n_frames": processed_probe["n_frames"], "codec_name": "semantic_generation",
            "path": payload["reference_payload_path"],
            "decoded_output_path": payload["recon_video_path"],
            "decoded_output_file_bytes_not_payload": semantic_probe["size_bytes"],
            "jscc_visual_channel_symbols_proxy": payload["jscc_visual_channel_symbols_proxy"],
            "jscc_float32_storage_bytes_proxy": payload["jscc_float32_storage_bytes_proxy"],
            "estimated_wire_bytes": payload["estimated_wire_bytes"],
            "exact_network_bitstream_available": False,
            **semantic_quality,
        }
        comparison_rows.append(semantic_row)

        for codec_name in active_codecs:
            spec = CODECS[codec_name]
            for crf in codec_crfs[codec_name]:
                output_path = output_root / "encoded" / codec_name / f"crf_{crf}" / f"{video_id}{spec.extension}"
                encode_video(entry["processed_path"], output_path, spec, crf,
                             ffmpeg=args.ffmpeg, overwrite=args.overwrite)
                probe = ffprobe_video(output_path, args.ffprobe)
                quality = _quality_row(
                    entry["processed_path"], output_path, entry["frames_path"],
                    ffmpeg=args.ffmpeg, lpips_evaluator=lpips_evaluator,
                )
                row = _base_rate_row(video_id, codec_name, f"crf_{crf}", output_path, probe)
                row.update({"crf": crf, **quality})
                codec_rows.append(row)
                comparison_rows.append(dict(row))

    for codec_name in unavailable:
        for entry in entries:
            codec_rows.append({
                "video_id": entry["video_id"], "method": codec_name, "setting": None,
                "status": "encoder_unavailable", "encoder": CODECS[codec_name].encoder,
            })

    aggregate_rows = summarize_comparison(comparison_rows)
    matched_rows = match_codec_quality(comparison_rows, metric="psnr")
    _write_csv(output_root / "source_sizes.csv", source_rows)
    _write_csv(output_root / "payloads.csv", payload_rows)
    _write_csv(output_root / "codec_results.csv", codec_rows)
    _write_csv(output_root / "comparison.csv", comparison_rows)
    _write_csv(output_root / "aggregate.csv", aggregate_rows)
    _write_csv(output_root / "quality_matched.csv", matched_rows)
    summary = {
        "plan": plan,
        "outputs": {
            "source_rows": len(source_rows), "payload_rows": len(payload_rows),
            "codec_rows": len(codec_rows), "comparison_rows": len(comparison_rows),
            "aggregate_rows": len(aggregate_rows), "quality_matched_rows": len(matched_rows),
        },
        "accounting_warning": (
            "Semantic reference payload bytes are deterministic serialized artifacts, not an exact "
            "JSCC network bitstream. Channel-symbol proxy fields remain separate."
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return summary


def main(argv=None) -> int:
    args = _parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
