"""CPU-only tests for scripts/benchmark_etri_video_rate.py."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parent.parent


def _load_script():
    path = _REPO / "scripts" / "benchmark_etri_video_rate.py"
    spec = importlib.util.spec_from_file_location("benchmark_etri_video_rate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_script()


def test_parse_crf_overrides_and_encode_command(tmp_path):
    crfs = bench.parse_crf_overrides(["h264=19,27", "av1=31"])
    assert crfs["h264"] == (19, 27)
    assert crfs["h265"] == bench.CODECS["h265"].default_crfs
    assert crfs["av1"] == (31,)

    cmd = bench.build_encode_command(
        tmp_path / "in.mp4", tmp_path / "out.mp4", bench.CODECS["h264"], 27,
    )
    assert cmd[0] == "ffmpeg"
    assert "libx264" in cmd
    assert cmd[cmd.index("-crf") + 1] == "27"
    assert cmd[-1].endswith("out.mp4")

    h265_cmd = bench.build_encode_command(
        tmp_path / "in.mp4", tmp_path / "out.mp4", bench.CODECS["h265"], 28,
    )
    assert "log-level=error:numa-pools=0" in h265_cmd


def test_explicit_ffmpeg_pair_resolution(monkeypatch):
    monkeypatch.setattr(bench, "available_encoders", lambda executable: {"libx264", "libx265"})
    monkeypatch.setattr(bench.shutil, "which", lambda executable: executable)
    ffmpeg, ffprobe, encoders = bench.resolve_ffmpeg_tools(
        "custom-ffmpeg", "custom-ffprobe", ["libx264", "libx265"],
    )
    assert ffmpeg == "custom-ffmpeg"
    assert ffprobe == "custom-ffprobe"
    assert {"libx264", "libx265"} <= encoders


def test_reference_payload_and_symbol_proxy(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_00000.png").write_bytes(b"png-zero")
    (frames / "frame_00002.png").write_bytes(b"png-two-longer")
    captions = tmp_path / "captions.txt"
    captions.write_text("first caption\nmiddle caption\nlast caption\n", encoding="utf-8")

    run_dir = tmp_path / "generation" / "wan_skem_dsa" / "01_toy"
    run_dir.mkdir(parents=True)
    (run_dir / "recon.mp4").write_bytes(b"not-probed-in-this-unit-test")
    (run_dir / "keyframes.json").write_text(json.dumps({"keyframes": [0, 2]}))
    generation = {"frames": [{"used_caption": True, "used_side_info": False}]}
    (run_dir / "segments.json").write_text(json.dumps([
        {"segment_id": 0, "keyframe_index": 0, "generation": generation},
        {"segment_id": 1, "keyframe_index": 2, "generation": generation},
    ]))
    entry = {
        "video_id": "01_toy", "width": "130", "height": "129", "fps": "10",
        "n_frames": "3", "frames_path": frames, "caption_path": captions,
    }
    output = tmp_path / "payload.sgdref"
    row = bench.build_payload(entry, run_dir, output, bits_per_channel_symbol=2.0)

    # ceil(130/128) * ceil(129/128) * 4096 symbols * two keyframes.
    assert row["jscc_visual_channel_symbols_proxy"] == 2 * 2 * 4096 * 2
    assert row["n_keyframes"] == 2
    assert row["n_segments"] == 2
    assert row["side_info_status"] == "not_used"
    assert row["reference_payload_bytes"] == output.stat().st_size
    assert row["estimated_wire_bytes"] is not None
    assert output.read_bytes().startswith(bench.REFERENCE_PAYLOAD_MAGIC)


def test_required_side_info_is_marked_missing(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_00000.png").write_bytes(b"frame")
    captions = tmp_path / "captions.txt"
    captions.write_text("caption\n", encoding="utf-8")
    run_dir = tmp_path / "mode" / "01_toy"
    run_dir.mkdir(parents=True)
    (run_dir / "recon.mp4").write_bytes(b"video")
    (run_dir / "keyframes.json").write_text(json.dumps({"keyframes": [0]}))
    (run_dir / "segments.json").write_text(json.dumps([{
        "segment_id": 0, "keyframe_index": 0,
        "generation": {"frames": [{"used_caption": False, "used_side_info": True}]},
    }]))
    entry = {
        "video_id": "01_toy", "width": "128", "height": "128", "fps": "10",
        "n_frames": "1", "frames_path": frames, "caption_path": captions,
    }
    row = bench.build_payload(entry, run_dir, tmp_path / "payload.sgdref")
    assert row["side_info_status"] == "required_but_missing"
    assert row["exact_network_bitstream_available"] is False


def test_quality_matching_prefers_smallest_point_at_or_above_target():
    rows = [
        {"video_id": "01", "method": "semantic_mode", "setting": "ref",
         "status": "ok", "size_bytes": 100, "psnr": 30.0},
        {"video_id": "01", "method": "h264", "setting": "crf_18",
         "status": "ok", "size_bytes": 200, "psnr": 34.0},
        {"video_id": "01", "method": "h264", "setting": "crf_23",
         "status": "ok", "size_bytes": 140, "psnr": 31.0},
        {"video_id": "01", "method": "h264", "setting": "crf_28",
         "status": "ok", "size_bytes": 90, "psnr": 28.0},
    ]
    matched = bench.match_codec_quality(rows)
    assert len(matched) == 1
    assert matched[0]["codec_setting"] == "crf_23"
    assert matched[0]["match_status"] == "quality_at_least_target"
    assert matched[0]["codec_over_semantic_payload_ratio"] == pytest.approx(1.4)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_ffmpeg_metric_parser_on_identical_video(tmp_path):
    video = tmp_path / "reference.mkv"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=5:duration=1",
        "-c:v", "ffv1", str(video),
    ], check=True)
    psnr, ssim = bench.measure_psnr_ssim(video, video)
    assert psnr == float("inf") or psnr > 100.0
    assert ssim == pytest.approx(1.0)
