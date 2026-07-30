"""tests/test_batch_lgvsc_1c_reproduce.py – scripts/batch_lgvsc_1c_reproduce.py
tests (ETRI 후속 1단계 step 1C).

All tests run in `ptest` with no GPU/network/real subprocess: subprocess.run
is monkeypatched wherever a "job" is dispatched, and dry-run's whole point is
that it never touches subprocess.run at all (asserted directly by making the
patched subprocess.run raise if it's ever called). No real evaluate_video.py
invocation happens here — that is the user's job on real hardware, per the
1C task's explicit scope.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "batch_lgvsc_1c_reproduce.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("batch_lgvsc_1c_reproduce", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


driver = _load_module()


def _make_dataset(data_root: Path, videos=("01_toy", "02_toy")):
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "processed").mkdir(exist_ok=True)
    header = ("id,name,raw_file,processed_file,frames_dir,width,height,fps,"
              "duration_sec,n_frames,primary_objects,event\n")
    rows = []
    for i, key in enumerate(videos, start=1):
        video_path = data_root / "processed" / f"{key}.mp4"
        video_path.write_bytes(b"fake")
        rows.append(f"{i:02d},{key},raw/{key}.mp4,processed/{key}.mp4,"
                    f"frames/{key},64,64,10,1,10,x,x\n")
    (data_root / "manifest.csv").write_text(header + "".join(rows), encoding="utf-8")


class _FakeCompletedProcess:
    def __init__(self, returncode=0):
        self.returncode = returncode


def _write_complete_source_run(
    output_root: Path,
    mode: str,
    video: str,
    *,
    n_frames: int,
    keyframes: list,
) -> Path:
    out_dir = driver.out_dir_for(output_root, mode, video)
    out_dir.mkdir(parents=True, exist_ok=True)
    boundaries = [False] * n_frames
    for index in keyframes:
        boundaries[index] = True
    (out_dir / "keyframes.json").write_text(
        json.dumps({"keyframes": keyframes, "boundaries": boundaries}),
        encoding="utf-8",
    )
    (out_dir / "temporal_metrics.csv").write_text(
        f"n_frames\n{n_frames}\n", encoding="utf-8",
    )
    return out_dir


# ──────────────────────────────────────────────────────────────────────────────
# 1) mode -> config selection
# ──────────────────────────────────────────────────────────────────────────────

class TestModeConfigSelection:
    def test_each_mode_selects_its_own_config_file(self):
        for mode in driver.MODES:
            path = driver.mode_config_path(mode)
            assert path.exists(), f"{mode} -> {path} does not exist"
            assert path.name == f"etri_lgvsc_1c_{mode}.yaml"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown 1C mode"):
            driver.mode_config_path("not_a_real_mode")

    def test_modes_tuple_matches_task_brief(self):
        assert driver.MODES == (
            "mock_baseline", "svd_start_only", "wan_skim_sfa", "wan_skem_dsa",
            "skim_sfa_fixed", "skem_dsa_psss", "skem_dsa_mock_psss", "skem_dsa_proxy_psss",
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2) generated config output-path isolation
# ──────────────────────────────────────────────────────────────────────────────

class TestOutputPathIsolation:
    def test_paths_isolated_across_modes_and_videos(self, tmp_path):
        output_root = tmp_path / "lgvsc_1c_reproduce"
        seen_paths = set()
        for mode in ("wan_skim_sfa", "wan_skem_dsa"):
            for video in ("01_person_walk", "05_camera_pan_person"):
                out_dir = driver.out_dir_for(output_root, mode, video)
                cfg = driver.build_run_config(mode, out_dir)
                for p in (cfg["keyframe_json"], cfg["segment_json"], cfg["temporal_csv"],
                          cfg["frame_log_csv"], cfg["video_io"]["recon_video"],
                          cfg["video_generator"]["generated_frames_dir"]):
                    assert p not in seen_paths, f"path collision: {p}"
                    seen_paths.add(p)
                    # Every output path must actually live under this (mode, video)'s
                    # own out_dir, not under a shared _generated_configs/ directory.
                    assert str(out_dir) in p

    def test_generated_config_file_path_is_per_mode_per_video(self, tmp_path):
        output_root = tmp_path / "lgvsc_1c_reproduce"
        p1 = driver.generated_config_path(output_root, "wan_skim_sfa", "01_person_walk")
        p2 = driver.generated_config_path(output_root, "wan_skim_sfa", "05_camera_pan_person")
        p3 = driver.generated_config_path(output_root, "wan_skem_dsa", "01_person_walk")
        assert len({p1, p2, p3}) == 3


# ──────────────────────────────────────────────────────────────────────────────
# 3) dry-run never touches subprocess
# ──────────────────────────────────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_never_calls_subprocess(self, tmp_path, monkeypatch):
        def _boom(*a, **kw):
            raise AssertionError("subprocess.run must not be called in --dry-run")

        monkeypatch.setattr(driver.subprocess, "run", _boom)
        entry = {"key": "01_toy", "processed": Path("/data/01_toy.mp4"), "captions": None}
        status = driver.run_job(
            "wan_skim_sfa", entry, tmp_path / "out",
            device="cuda:0", max_frames=14, dry_run=True,
        )
        assert status["status"] == "dry_run"
        assert status["cmd"] is not None

    def test_dry_run_via_main_never_calls_subprocess(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        _make_dataset(data_root)

        def _boom(*a, **kw):
            raise AssertionError("subprocess.run must not be called in --dry-run")

        monkeypatch.setattr(driver.subprocess, "run", _boom)
        rc = driver.main([
            "--data-root", str(data_root), "--output-root", str(tmp_path / "out"),
            "--modes", "mock_baseline,svd_start_only", "--dry-run",
        ])
        assert rc == 0


# ──────────────────────────────────────────────────────────────────────────────
# 4) command generation reflects flags
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildCommand:
    def test_max_frames_device_no_models_and_captions_reflected(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cmd = driver.build_command(
            cfg_path, Path("/data/01_toy.mp4"), Path("/data/01_toy.txt"),
            device="cuda:0", max_frames=14, no_models=True, save_video=True,
        )
        assert "--config" in cmd and str(cfg_path) in cmd
        assert "--input" in cmd and "/data/01_toy.mp4" in cmd
        assert "--captions" in cmd and "/data/01_toy.txt" in cmd
        idx = cmd.index("--device")
        assert cmd[idx + 1] == "cuda:0"
        idx = cmd.index("--max-frames")
        assert cmd[idx + 1] == "14"
        assert "--no-models" in cmd
        assert "--save-video" in cmd

    def test_no_models_absent_when_false(self, tmp_path):
        cmd = driver.build_command(tmp_path / "c.yaml", Path("/d/v.mp4"), no_models=False)
        assert "--no-models" not in cmd

    def test_captions_omitted_when_none(self, tmp_path):
        cmd = driver.build_command(tmp_path / "c.yaml", Path("/d/v.mp4"), None)
        assert "--captions" not in cmd


# ──────────────────────────────────────────────────────────────────────────────
# 5) --summary-only regenerates from existing metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestSummaryOnly:
    def _write_fake_run(self, out_dir: Path, *, n_generate=11, backend="external_segment_worker:wan:x",
                         conditioning_mode="bidirectional", end_keyframe_index=12):
        out_dir.mkdir(parents=True, exist_ok=True)
        fields = ("n_frames", "temporal_srs", "srs_flicker", "object_identity_consistency",
                  "temporal_segmentation_iou", "temporal_hallucination_rate", "ptc", "sfr", "sdi",
                  "n_keyframes", "n_interframes", "n_reused", "n_generate",
                  "n_recompute_semantic", "n_recompute_motion", "transmitted_units",
                  "naive_units", "overhead_reduction")
        row = {f: 0 for f in fields}
        row.update(n_frames=14, n_keyframes=2, n_interframes=12, n_generate=n_generate,
                   temporal_srs=1.0, ptc=0.99, sfr=0.0, sdi=0.0,
                   temporal_hallucination_rate=0.0, transmitted_units=10, naive_units=70,
                   overhead_reduction=0.857143)
        with open(out_dir / "temporal_metrics.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerow(row)
        segments = [{"segment_id": 0, "generation": {
            "conditioning_mode": conditioning_mode, "backend": backend,
            "end_keyframe_index": end_keyframe_index,
        }}]
        (out_dir / "segments.json").write_text(json.dumps(segments), encoding="utf-8")
        gen_dir = out_dir / "generated_frames"
        gen_dir.mkdir(exist_ok=True)
        for i in range(1, n_generate + 1):
            (gen_dir / f"generated_{i:05d}.png").write_bytes(b"")

    def test_summary_only_regenerates_table_from_disk(self, tmp_path):
        data_root = tmp_path / "data"
        _make_dataset(data_root, videos=("01_toy",))
        output_root = tmp_path / "out"
        self._write_fake_run(output_root / "wan_skem_dsa" / "01_toy")

        rc = driver.main([
            "--data-root", str(data_root), "--output-root", str(output_root),
            "--modes", "wan_skem_dsa", "--summary-only",
        ])
        assert rc == 0
        assert (output_root / "summary_metrics.csv").exists()
        with open(output_root / "summary_metrics.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        row = rows[0]
        assert row["mode"] == "wan_skem_dsa"
        assert row["video_id"] == "01_toy"
        assert row["status"] == "ok"
        assert row["n_generate"] == "11"
        assert row["generated_frame_count"] == "11"
        assert row["conditioning_modes_observed"] == "bidirectional"
        assert row["has_end_keyframe"] == "True"

    def test_summary_only_marks_missing_run_without_touching_subprocess(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        _make_dataset(data_root, videos=("01_toy",))
        output_root = tmp_path / "out"

        def _boom(*a, **kw):
            raise AssertionError("summary-only must never dispatch a job")

        monkeypatch.setattr(driver.subprocess, "run", _boom)
        rc = driver.main([
            "--data-root", str(data_root), "--output-root", str(output_root),
            "--modes", "mock_baseline", "--summary-only",
        ])
        assert rc == 0
        with open(output_root / "summary_metrics.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["status"] == "missing"


# ──────────────────────────────────────────────────────────────────────────────
# 6) continue-on-error
# ──────────────────────────────────────────────────────────────────────────────

class TestContinueOnError:
    def test_failed_job_recorded_and_next_job_still_runs(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        _make_dataset(data_root, videos=("01_fail", "02_ok"))
        output_root = tmp_path / "out"
        calls = []

        def fake_run(cmd, stdout=None, stderr=None, cwd=None):
            video = "01_fail" if "01_fail" in " ".join(cmd) else "02_ok"
            calls.append(video)
            if video == "01_fail":
                stdout.write("boom\n")
                return _FakeCompletedProcess(returncode=1)
            (Path(cmd[cmd.index("--config") + 1]).parent / "temporal_metrics.csv").write_text(
                "n_frames\n1\n", encoding="utf-8")
            stdout.write("ok\n")
            return _FakeCompletedProcess(returncode=0)

        monkeypatch.setattr(driver.subprocess, "run", fake_run)
        rc = driver.main([
            "--data-root", str(data_root), "--output-root", str(output_root),
            "--modes", "mock_baseline", "--continue-on-error",
        ])
        assert rc == 1  # overall batch still reports a failure occurred
        assert calls == ["01_fail", "02_ok"]  # both jobs were attempted

        status = json.loads((output_root / "batch_status.json").read_text())
        by_video = {r["video"]: r for r in status}
        assert by_video["01_fail"]["status"] == "failed"
        assert by_video["02_ok"]["status"] == "ok"

    def test_without_continue_on_error_stops_after_first_failure(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        _make_dataset(data_root, videos=("01_fail", "02_ok"))
        output_root = tmp_path / "out"
        calls = []

        def fake_run(cmd, stdout=None, stderr=None, cwd=None):
            calls.append(cmd)
            stdout.write("boom\n")
            return _FakeCompletedProcess(returncode=1)

        monkeypatch.setattr(driver.subprocess, "run", fake_run)
        rc = driver.main([
            "--data-root", str(data_root), "--output-root", str(output_root),
            "--modes", "mock_baseline",
        ])
        assert rc == 1
        assert len(calls) == 1  # stopped after the first (failing) job


# ──────────────────────────────────────────────────────────────────────────────
# 7/8) wan_skim_sfa / wan_skem_dsa configs are based on the verified 1B configs
# ──────────────────────────────────────────────────────────────────────────────

class TestBasedOnVerifiedConfigs:
    @staticmethod
    def _load_yaml(path: Path):
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(path), resolve=False)

    def test_wan_skim_sfa_worker_matches_wan_start_only(self):
        c1c = self._load_yaml(_REPO / "configs" / "etri_lgvsc_1c_wan_skim_sfa.yaml")
        source = self._load_yaml(_REPO / "configs" / "etri_video_eval_lgvsc_worker_wan_start_only.yaml")
        assert c1c["video_generator"]["worker"] == source["video_generator"]["worker"]
        assert c1c["video_generator"]["conditioning_mode"] == source["video_generator"]["conditioning_mode"]
        assert c1c["video_generator"]["conditioning_mode"] == "start_only"

    def test_wan_skem_dsa_worker_matches_wan_bidirectional_fixed(self):
        c1c = self._load_yaml(_REPO / "configs" / "etri_lgvsc_1c_wan_skem_dsa.yaml")
        source = self._load_yaml(_REPO / "configs" / "etri_video_eval_lgvsc_worker_wan_bidirectional_fixed.yaml")
        assert c1c["video_generator"]["worker"] == source["video_generator"]["worker"]
        assert c1c["video_generator"]["conditioning_mode"] == source["video_generator"]["conditioning_mode"]
        assert c1c["video_generator"]["conditioning_mode"] == "bidirectional"
        assert "bidirectional_model_id" in c1c["video_generator"]["worker"]["extra_json"]

    def test_svd_start_only_worker_matches_verified_svd_config(self):
        c1c = self._load_yaml(_REPO / "configs" / "etri_lgvsc_1c_svd_start_only.yaml")
        source = self._load_yaml(_REPO / "configs" / "etri_video_eval_lgvsc_worker_svd.yaml")
        assert c1c["video_generator"]["worker"] == source["video_generator"]["worker"]


# ──────────────────────────────────────────────────────────────────────────────
# Every mode's static template must load through the real config loader
# ──────────────────────────────────────────────────────────────────────────────

class TestStaticConfigsLoad:
    def test_all_mode_configs_load(self):
        sys.path.insert(0, str(_REPO / "src"))
        from sgdjscc_lab.config import load_config
        for mode in driver.MODES:
            cfg = load_config(str(driver.mode_config_path(mode)))
            assert cfg.video_generator.enabled is True


# ──────────────────────────────────────────────────────────────────────────────
# PSSS/SKEM readiness step: new modes' config provenance + extended summary
# ──────────────────────────────────────────────────────────────────────────────

class TestPsssSkemModeConfigs:
    @staticmethod
    def _load_yaml(path: Path):
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(path), resolve=False)

    def test_skim_sfa_fixed_worker_matches_wan_start_only_and_is_explicitly_fixed(self):
        c1c = self._load_yaml(_REPO / "configs" / "etri_lgvsc_1c_skim_sfa_fixed.yaml")
        source = self._load_yaml(_REPO / "configs" / "etri_video_eval_lgvsc_worker_wan_start_only.yaml")
        assert c1c["video_generator"]["worker"] == source["video_generator"]["worker"]
        assert c1c["video_generator"]["conditioning_mode"] == "start_only"
        # Literal SKIM (zero scene-change signal), not the scene-change-
        # reactive "fixed" default — see FixedIntervalKeyframeSelector.
        assert c1c["keyframe"]["selector"] == "fixed_interval"
        assert c1c["keyframe"]["fixed_interval"]["interval"] == 12

    def test_skem_dsa_psss_worker_matches_wan_bidirectional_fixed_and_uses_psss_selector(self):
        c1c = self._load_yaml(_REPO / "configs" / "etri_lgvsc_1c_skem_dsa_psss.yaml")
        source = self._load_yaml(_REPO / "configs" / "etri_video_eval_lgvsc_worker_wan_bidirectional_fixed.yaml")
        assert c1c["video_generator"]["worker"] == source["video_generator"]["worker"]
        assert c1c["video_generator"]["conditioning_mode"] == "bidirectional"
        assert c1c["keyframe"]["selector"] == "psss"
        assert c1c["keyframe"]["psss"]["backend"] == "real"

    def test_mock_and_proxy_psss_modes_use_mock_decoder_and_their_named_backend(self):
        for mode, expected_backend in (
            ("skem_dsa_mock_psss", "mock"), ("skem_dsa_proxy_psss", "proxy"),
        ):
            cfg = self._load_yaml(driver.mode_config_path(mode))
            assert cfg["keyframe"]["selector"] == "psss"
            assert cfg["keyframe"]["psss"]["backend"] == expected_backend
            assert cfg["video_generator"]["backend"] == "auto"
            assert cfg["video_generator"]["conditioning_mode"] == "bidirectional"

    def test_selector_family_mapping_covers_new_modes(self):
        for mode in ("wan_skim_sfa", "skim_sfa_fixed"):
            assert driver._SKIM_SKEM_FAMILY[mode] == "skim_sfa"
        for mode in ("wan_skem_dsa", "skem_dsa_psss", "skem_dsa_mock_psss", "skem_dsa_proxy_psss"):
            assert driver._SKIM_SKEM_FAMILY[mode] == "skem_dsa"


class TestExtendedSummaryFields:
    def _write_fake_psss_run(self, out_dir: Path, output_root: Path, mode: str, video_id: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        fields = ("n_frames", "temporal_srs", "srs_flicker", "ptc", "sfr", "sdi",
                  "temporal_hallucination_rate", "n_keyframes", "n_interframes", "n_reused",
                  "n_generate", "n_recompute_semantic", "n_recompute_motion",
                  "transmitted_units", "naive_units", "overhead_reduction")
        row = {f: 0 for f in fields}
        row.update(n_frames=6, n_keyframes=3, temporal_srs=0.9, ptc=0.95, overhead_reduction=0.5)
        with open(out_dir / "temporal_metrics.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerow(row)

        segments = [
            {
                "segment_id": 0, "keyframe_index": 0, "inter_frame_indices": [1],
                "keyframe_selection": {
                    "selector": "psss", "backend_kind": "mock", "threshold": 0.3,
                    "segment_length": 2, "psss_score": None,
                    "reason": "first frame (K_1 = 1) — always a keyframe, no PSSS evaluated.",
                },
                "generation": {
                    "conditioning_mode": "bidirectional", "backend": "bidirectional_interpolation",
                    "end_keyframe_index": 2,
                },
            },
            {
                "segment_id": 1, "keyframe_index": 2, "inter_frame_indices": [3, 4],
                "keyframe_selection": {
                    "selector": "psss", "backend_kind": "mock", "threshold": 0.3,
                    "segment_length": 3,
                    "psss_score": {"index": 2, "s_rel": 0.72, "decision": "new_keyframe"},
                    "reason": "S_rel=0.7200 > threshold=0.3 vs keyframe 0 → semantic divergence, new keyframe.",
                },
                "generation": {
                    "conditioning_mode": "bidirectional", "backend": "bidirectional_interpolation",
                    "end_keyframe_index": 5,
                },
            },
            {
                "segment_id": 2, "keyframe_index": 5, "inter_frame_indices": [],
                "keyframe_selection": {
                    "selector": "psss", "backend_kind": "mock", "threshold": 0.3,
                    "segment_length": 1,
                    "psss_score": {"index": 5, "s_rel": 0.81, "decision": "new_keyframe"},
                    "reason": "S_rel=0.8100 > threshold=0.3 vs keyframe 2 → semantic divergence, new keyframe.",
                },
                # Last (open) GOP: bidirectional requested but no end keyframe → the
                # mock decoder fell back to start_only (its own metadata says so).
                "generation": {
                    "conditioning_mode": "start_only", "backend": "bidirectional_interpolation",
                    "end_keyframe_index": None,
                },
            },
        ]
        (out_dir / "segments.json").write_text(json.dumps(segments), encoding="utf-8")

        # Full evaluated-score population (video/skem_selector.py's
        # keyframes.json output) — includes "continue_segment" evaluations
        # that segments.json's keyframe_selection never sees (only the
        # TRIGGERING score is attached to a segment).
        keyframes = {
            "keyframes": [0, 2, 5],
            "psss_scores": [
                {"index": 1, "s_rel": -0.2, "decision": "continue_segment"},
                {"index": 2, "s_rel": 0.72, "decision": "new_keyframe"},
                {"index": 3, "s_rel": -0.5, "decision": "continue_segment"},
                {"index": 4, "s_rel": 0.1, "decision": "continue_segment"},
                {"index": 5, "s_rel": 0.81, "decision": "new_keyframe"},
            ],
        }
        (out_dir / "keyframes.json").write_text(json.dumps(keyframes), encoding="utf-8")

        cfg_path = driver.generated_config_path(output_root, mode, video_id)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        driver._write_yaml({
            "keyframe": {"selector": "psss", "psss": {"backend": "mock"}},
            "video_generator": {"conditioning_mode": "bidirectional"},
        }, cfg_path)

    def test_psss_run_summary_row_has_selector_and_segment_stats(self, tmp_path):
        output_root = tmp_path / "out"
        out_dir = driver.out_dir_for(output_root, "skem_dsa_mock_psss", "01_toy")
        self._write_fake_psss_run(out_dir, output_root, "skem_dsa_mock_psss", "01_toy")

        row = driver.collect_run_metrics(output_root, "skem_dsa_mock_psss", "01_toy")
        assert row["selector_backend"] == "psss"
        assert row["psss_backend_kind"] == "mock"
        assert row["n_segments"] == 3
        assert row["segment_length_min"] == 1
        assert row["segment_length_max"] == 3
        assert row["segment_length_mean"] == pytest.approx(2.0)
        # Population-wide (ALL evaluated scores, from keyframes.json) — must
        # NOT equal the trigger-only mean; a biased "trigger scores only"
        # aggregate would silently omit the 3 continue_segment evaluations.
        all_scores = [-0.2, 0.72, -0.5, 0.1, 0.81]
        assert row["psss_score_mean"] == pytest.approx(sum(all_scores) / len(all_scores))
        assert row["psss_score_min"] == pytest.approx(-0.5)
        assert row["psss_score_max"] == pytest.approx(0.81)
        assert row["psss_score_n"] == 5
        # Trigger-only stats (separately named, not conflated with the above).
        assert row["trigger_psss_score_mean"] == pytest.approx((0.72 + 0.81) / 2)
        assert row["trigger_psss_score_min"] == pytest.approx(0.72)
        assert row["trigger_psss_score_max"] == pytest.approx(0.81)
        # segment 2's bidirectional request degraded to start_only (no end
        # keyframe on the last/open GOP) — counted as a fallback, not a
        # legitimate start_only-by-design segment.
        assert row["n_bidirectional_segments"] == 2
        assert row["n_start_only_segments"] == 1
        assert row["n_fallback_segments"] == 1
        assert row["segments_json_path"].endswith("segments.json")

    def test_fixed_selector_run_reports_not_applicable_psss_kind(self, tmp_path):
        output_root = tmp_path / "out"
        out_dir = driver.out_dir_for(output_root, "mock_baseline", "01_toy")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "segments.json").write_text("[]", encoding="utf-8")
        row = driver.collect_run_metrics(output_root, "mock_baseline", "01_toy")
        assert row["selector_backend"] == "fixed"
        assert row["psss_backend_kind"] == "not_applicable"
        assert row["n_segments"] == 0
        assert row["segment_length_mean"] is None


class TestAggregateComparison:
    def test_per_video_and_mean_rows_built(self):
        rows = [
            {"mode": "skim_sfa_fixed", "video_id": "01", "n_segments": 5, "segment_length_mean": 2.0,
             "segment_length_std": 0.0, "segment_length_min": 2, "segment_length_max": 2,
             "psss_score_mean": None, "temporal_srs": 0.8, "ptc": 0.9, "sfr": 0.1, "sdi": 0.05,
             "overhead_reduction": 0.5, "n_start_only_segments": 5, "n_bidirectional_segments": 0,
             "n_fallback_segments": 0},
            {"mode": "skem_dsa_psss", "video_id": "01", "n_segments": 3, "segment_length_mean": 3.33,
             "segment_length_std": 1.2, "segment_length_min": 1, "segment_length_max": 5,
             "psss_score_mean": 0.6, "temporal_srs": 0.85, "ptc": 0.95, "sfr": 0.05, "sdi": 0.02,
             "overhead_reduction": 0.6, "n_start_only_segments": 1, "n_bidirectional_segments": 2,
             "n_fallback_segments": 1},
        ]
        out = driver.build_aggregate_comparison(rows)
        assert len(out) == 2  # one per-video row + one MEAN row
        video_row = out[0]
        assert video_row["video_id"] == "01"
        assert video_row["skim_sfa_fixed.n_segments"] == 5
        assert video_row["skem_dsa_psss.n_segments"] == 3
        assert video_row["skem_dsa_psss.psss_score_mean"] == 0.6
        mean_row = out[-1]
        assert mean_row["video_id"] == "MEAN"
        assert mean_row["skim_sfa_fixed.n_segments"] == pytest.approx(5.0)
        assert mean_row["skem_dsa_psss.n_segments"] == pytest.approx(3.0)

    def test_missing_mode_leaves_none_without_dropping_video(self):
        rows = [{"mode": "skim_sfa_fixed", "video_id": "01", "n_segments": 4}]
        out = driver.build_aggregate_comparison(rows)
        assert out[0]["video_id"] == "01"
        assert out[0]["skim_sfa_fixed.n_segments"] == 4
        assert out[0]["skem_dsa_psss.n_segments"] is None

    def test_empty_rows_produce_empty_table(self):
        assert driver.build_aggregate_comparison([]) == []


class TestBatchDryRunAndDiscoveryForNewModes:
    def test_dry_run_across_new_modes_never_calls_subprocess(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        _make_dataset(data_root, videos=("01_toy", "02_toy"))

        def _boom(*a, **kw):
            raise AssertionError("subprocess.run must not be called in --dry-run")

        monkeypatch.setattr(driver.subprocess, "run", _boom)
        rc = driver.main([
            "--data-root", str(data_root), "--output-root", str(tmp_path / "out"),
            "--modes", "skim_sfa_fixed,skem_dsa_psss,skem_dsa_mock_psss,skem_dsa_proxy_psss",
            "--max-frames", "14", "--dry-run",
        ])
        assert rc == 0

    def test_two_video_two_mode_summary_isolated_by_mode_and_video(self, tmp_path):
        data_root = tmp_path / "data"
        _make_dataset(data_root, videos=("01_toy", "02_toy"))
        output_root = tmp_path / "out"
        for mode in ("skim_sfa_fixed", "skem_dsa_mock_psss"):
            for video in ("01_toy", "02_toy"):
                out_dir = driver.out_dir_for(output_root, mode, video)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "temporal_metrics.csv").write_text("n_frames\n6\n", encoding="utf-8")
                (out_dir / "segments.json").write_text("[]", encoding="utf-8")

        rc = driver.main([
            "--data-root", str(data_root), "--output-root", str(output_root),
            "--modes", "skim_sfa_fixed,skem_dsa_mock_psss", "--summary-only",
        ])
        assert rc == 0
        with open(output_root / "summary_metrics.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4  # 2 modes x 2 videos, none dropped/merged
        seen = {(r["mode"], r["video_id"]) for r in rows}
        assert seen == {
            ("skim_sfa_fixed", "01_toy"), ("skim_sfa_fixed", "02_toy"),
            ("skem_dsa_mock_psss", "01_toy"), ("skem_dsa_mock_psss", "02_toy"),
        }
        assert (output_root / "summary_aggregate_comparison.csv").exists()


# ──────────────────────────────────────────────────────────────────────────────
# CBR-matching (--cbr-match-from): calibrate skim_sfa_fixed's fixed_interval
# to a SKEM run's ACTUAL keyframe count, not just share max_segment_length as
# an upper bound.
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeCbrMatchedInterval:
    def test_exact_division_matches_keyframe_count(self):
        # 100 frames, want exactly 8 keyframes -> interval=13 -> ceil(100/13)=8.
        interval = driver.compute_cbr_matched_interval(100, 8)
        assert interval == 13
        import math
        assert math.ceil(100 / interval) == 8

    def test_unrepresentable_interval_raises(self):
        with pytest.raises(ValueError, match="No integer fixed interval"):
            driver.compute_cbr_matched_interval(10, 6)

    def test_more_keyframes_than_frames_rejected(self):
        with pytest.raises(ValueError):
            driver.compute_cbr_matched_interval(5, 20)

    def test_single_keyframe(self):
        assert driver.compute_cbr_matched_interval(50, 1) == 50

    def test_rejects_non_positive_inputs(self):
        with pytest.raises(ValueError):
            driver.compute_cbr_matched_interval(0, 5)
        with pytest.raises(ValueError):
            driver.compute_cbr_matched_interval(50, 0)


class TestResolveCbrMatch:
    def test_matched_when_source_keyframes_json_exists(self, tmp_path):
        output_root = tmp_path / "out"
        _write_complete_source_run(
            output_root, "skem_dsa_psss", "01_toy",
            n_frames=14, keyframes=[0, 3, 7, 9],
        )

        result = driver.resolve_cbr_match(
            output_root, "skem_dsa_psss", "01_toy", target_n_frames=14,
        )
        assert result["status"] == "count_planned"
        assert result["source_mode"] == "skem_dsa_psss"
        assert result["source_n_keyframes"] == 4
        assert result["source_n_frames"] == 14
        assert result["requested_keyframes"] == 4
        assert result["source_sha256"]
        assert result["matched_interval"] is None

    def test_source_missing_reports_status_without_raising(self, tmp_path):
        output_root = tmp_path / "out"
        result = driver.resolve_cbr_match(output_root, "skem_dsa_psss", "never_ran")
        assert result["status"] == "source_missing"
        assert result["matched_interval"] is None

    def test_source_empty_reports_status_without_raising(self, tmp_path):
        output_root = tmp_path / "out"
        src_dir = driver.out_dir_for(output_root, "skem_dsa_psss", "01_toy")
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "keyframes.json").write_text(json.dumps({"keyframes": [], "boundaries": []}), encoding="utf-8")
        result = driver.resolve_cbr_match(output_root, "skem_dsa_psss", "01_toy")
        assert result["status"] == "source_empty"
        assert result["matched_interval"] is None

    def test_source_without_temporal_metrics_is_incomplete(self, tmp_path):
        output_root = tmp_path / "out"
        src_dir = driver.out_dir_for(output_root, "skem_dsa_psss", "01_toy")
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "keyframes.json").write_text(
            json.dumps({"keyframes": [0, 5], "boundaries": [True] * 10}),
            encoding="utf-8",
        )
        result = driver.resolve_cbr_match(
            output_root, "skem_dsa_psss", "01_toy", target_n_frames=10,
        )
        assert result["status"] == "source_incomplete"

    def test_source_artifacts_must_agree_on_frame_count(self, tmp_path):
        output_root = tmp_path / "out"
        src_dir = _write_complete_source_run(
            output_root, "skem_dsa_psss", "01_toy",
            n_frames=10, keyframes=[0, 5],
        )
        (src_dir / "temporal_metrics.csv").write_text(
            "n_frames\n14\n", encoding="utf-8",
        )
        result = driver.resolve_cbr_match(
            output_root, "skem_dsa_psss", "01_toy", target_n_frames=10,
        )
        assert result["status"] == "source_inconsistent"

    def test_source_and_target_frame_count_must_match(self, tmp_path):
        output_root = tmp_path / "out"
        _write_complete_source_run(
            output_root, "skem_dsa_psss", "01_toy",
            n_frames=14, keyframes=[0, 5],
        )
        result = driver.resolve_cbr_match(
            output_root, "skem_dsa_psss", "01_toy", target_n_frames=10,
        )
        assert result["status"] == "frame_count_mismatch"


class TestBuildRunConfigCbrMatch:
    def test_match_plan_switches_to_exact_fixed_count_selector(self, tmp_path):
        cbr_match = {
            "source_mode": "skem_dsa_psss", "status": "count_planned",
            "source_n_frames": 14, "source_n_keyframes": 4,
            "target_n_frames": 14, "requested_keyframes": 4,
            "source_sha256": "abc", "matched_interval": None,
        }
        cfg = driver.build_run_config(
            "skim_sfa_fixed", tmp_path / "skim_sfa_fixed" / "01_toy", cbr_match=cbr_match,
        )
        assert cfg["keyframe"]["selector"] == "fixed_count"
        assert cfg["keyframe"]["fixed_count"]["count"] == 4
        assert cfg["_keyframe_count_match"] == cbr_match

    def test_source_missing_is_recorded_as_failed_precondition(self, tmp_path):
        cbr_match = {
            "source_mode": "skem_dsa_psss", "status": "source_missing",
            "source_n_frames": None, "source_n_keyframes": None,
            "target_n_frames": 14, "requested_keyframes": None,
            "source_sha256": None, "matched_interval": None,
        }
        cfg = driver.build_run_config(
            "skim_sfa_fixed", tmp_path / "skim_sfa_fixed" / "01_toy", cbr_match=cbr_match,
        )
        assert cfg["keyframe"]["fixed_interval"]["interval"] == 12
        assert cfg["_keyframe_count_match"]["status"] == "source_missing"

    def test_ignored_for_non_fixed_interval_modes(self, tmp_path):
        cbr_match = {
            "source_mode": "skem_dsa_psss", "status": "count_planned",
            "source_n_frames": 14, "source_n_keyframes": 4,
            "target_n_frames": 14, "requested_keyframes": 4,
        }
        cfg = driver.build_run_config(
            "skem_dsa_psss", tmp_path / "skem_dsa_psss" / "01_toy", cbr_match=cbr_match,
        )
        assert "_keyframe_count_match" not in cfg

    def test_no_cbr_match_leaves_config_untouched(self, tmp_path):
        cfg = driver.build_run_config("skim_sfa_fixed", tmp_path / "skim_sfa_fixed" / "01_toy")
        assert "_keyframe_count_match" not in cfg
        assert cfg["keyframe"]["fixed_interval"]["interval"] == 12


class TestCbrMatchEndToEnd:
    def test_run_job_records_cbr_match_and_summary_reflects_it(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        _make_dataset(data_root, videos=("01_toy",))
        output_root = tmp_path / "out"

        # Pre-seed a "SKEM already ran" keyframes.json for this video.
        _write_complete_source_run(
            output_root, "skem_dsa_psss", "01_toy",
            n_frames=10, keyframes=[0, 5],
        )

        def fake_run(cmd, stdout=None, stderr=None, cwd=None):
            out_dir = driver.out_dir_for(output_root, "skim_sfa_fixed", "01_toy")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "temporal_metrics.csv").write_text("n_frames\n10\n", encoding="utf-8")
            (out_dir / "keyframes.json").write_text(
                json.dumps({
                    "keyframes": [0, 5],
                    "boundaries": [True] + [False] * 9,
                }),
                encoding="utf-8",
            )
            stdout.write("ok\n")
            return _FakeCompletedProcess(returncode=0)

        monkeypatch.setattr(driver.subprocess, "run", fake_run)
        entries = driver.select_videos(driver.read_manifest(data_root))
        status = driver.run_job(
            "skim_sfa_fixed", entries[0], output_root,
            cbr_match_from="skem_dsa_psss",
        )
        assert status["status"] == "ok"

        gen_cfg_path = driver.generated_config_path(output_root, "skim_sfa_fixed", "01_toy")
        from omegaconf import OmegaConf
        gen_cfg = OmegaConf.to_container(OmegaConf.load(gen_cfg_path), resolve=False)
        assert gen_cfg["_keyframe_count_match"]["status"] == "count_planned"
        assert gen_cfg["keyframe"]["selector"] == "fixed_count"
        assert gen_cfg["keyframe"]["fixed_count"]["count"] == 2

        row = driver.collect_run_metrics(output_root, "skim_sfa_fixed", "01_toy")
        assert row["keyframe_match_status"] == "keyframe_count_verified"
        assert row["cbr_match_status"] == "count_only"
        assert row["cbr_match_source"] == "skem_dsa_psss"
        assert row["requested_keyframes"] == 2
        assert row["actual_fixed_keyframes"] == 2
        assert row["keyframe_count_delta"] == 0
        assert row["fixed_count_value"] == 2
        assert row["cbr_matched_interval"] is None

    def test_no_cbr_match_from_leaves_static_interval_and_not_requested_status(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        _make_dataset(data_root, videos=("01_toy",))
        output_root = tmp_path / "out"

        def fake_run(cmd, stdout=None, stderr=None, cwd=None):
            out_dir = driver.out_dir_for(output_root, "skim_sfa_fixed", "01_toy")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "temporal_metrics.csv").write_text("n_frames\n1\n", encoding="utf-8")
            stdout.write("ok\n")
            return _FakeCompletedProcess(returncode=0)

        monkeypatch.setattr(driver.subprocess, "run", fake_run)
        entries = driver.select_videos(driver.read_manifest(data_root))
        driver.run_job("skim_sfa_fixed", entries[0], output_root)

        row = driver.collect_run_metrics(output_root, "skim_sfa_fixed", "01_toy")
        assert row["cbr_match_status"] == "not_requested"
        assert row["fixed_interval_value"] == 12

    def test_post_run_count_mismatch_marks_job_failed(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        _make_dataset(data_root, videos=("01_toy",))
        output_root = tmp_path / "out"
        _write_complete_source_run(
            output_root, "skem_dsa_psss", "01_toy",
            n_frames=10, keyframes=[0, 5],
        )

        def fake_run(cmd, stdout=None, stderr=None, cwd=None):
            out_dir = driver.out_dir_for(output_root, "skim_sfa_fixed", "01_toy")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "temporal_metrics.csv").write_text("n_frames\n10\n", encoding="utf-8")
            (out_dir / "keyframes.json").write_text(
                json.dumps({"keyframes": [0], "boundaries": [True] * 10}),
                encoding="utf-8",
            )
            return _FakeCompletedProcess(returncode=0)

        monkeypatch.setattr(driver.subprocess, "run", fake_run)
        entry = driver.read_manifest(data_root)[0]
        status = driver.run_job(
            "skim_sfa_fixed", entry, output_root,
            cbr_match_from="skem_dsa_psss",
        )
        assert status["status"] == "failed"
        assert status["keyframe_count_match"]["status"] == "keyframe_count_mismatch"

    def test_source_artifact_change_invalidates_plan(self, tmp_path):
        output_root = tmp_path / "out"
        src_dir = _write_complete_source_run(
            output_root, "skem_dsa_psss", "01_toy",
            n_frames=10, keyframes=[0, 5],
        )
        plan = driver.resolve_cbr_match(
            output_root, "skem_dsa_psss", "01_toy", target_n_frames=10,
        )
        assert plan["status"] == "count_planned"

        target_dir = driver.out_dir_for(output_root, "skim_sfa_fixed", "01_toy")
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "keyframes.json").write_text(
            json.dumps({"keyframes": [0, 5], "boundaries": [True] * 10}),
            encoding="utf-8",
        )
        (src_dir / "temporal_metrics.csv").write_text(
            "n_frames\n10\n# changed\n", encoding="utf-8",
        )
        result = driver.verify_keyframe_count_match(target_dir, plan)
        assert result["status"] == "source_metrics_changed"

    def test_source_generated_config_change_invalidates_plan(self, tmp_path):
        output_root = tmp_path / "out"
        _write_complete_source_run(
            output_root, "skem_dsa_psss", "01_toy",
            n_frames=10, keyframes=[0, 5],
        )
        source_cfg = driver.generated_config_path(
            output_root, "skem_dsa_psss", "01_toy",
        )
        source_cfg.parent.mkdir(parents=True, exist_ok=True)
        source_cfg.write_text("keyframe:\n  selector: psss\n", encoding="utf-8")
        plan = driver.resolve_cbr_match(
            output_root, "skem_dsa_psss", "01_toy", target_n_frames=10,
        )

        target_dir = driver.out_dir_for(output_root, "skim_sfa_fixed", "01_toy")
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "keyframes.json").write_text(
            json.dumps({"keyframes": [0, 5], "boundaries": [True] * 10}),
            encoding="utf-8",
        )
        source_cfg.write_text(
            "keyframe:\n  selector: psss\n  changed: true\n", encoding="utf-8",
        )
        result = driver.verify_keyframe_count_match(target_dir, plan)
        assert result["status"] == "source_config_changed"

    def test_summary_distinguishes_count_only_and_measured_cbr(self, tmp_path):
        output_root = tmp_path / "out"
        source_dir = _write_complete_source_run(
            output_root, "skem_dsa_psss", "01_toy",
            n_frames=10, keyframes=[0, 5],
        )
        plan = driver.resolve_cbr_match(
            output_root, "skem_dsa_psss", "01_toy", target_n_frames=10,
        )
        target_dir = driver.out_dir_for(output_root, "skim_sfa_fixed", "01_toy")
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "keyframes.json").write_text(
            json.dumps({"keyframes": [0, 5], "boundaries": [True] * 10}),
            encoding="utf-8",
        )
        (target_dir / "temporal_metrics.csv").write_text(
            "n_frames\n10\n", encoding="utf-8",
        )
        verification = driver.verify_keyframe_count_match(target_dir, plan)
        (target_dir / "keyframe_count_match.json").write_text(
            json.dumps(verification), encoding="utf-8",
        )
        cfg = driver.build_run_config(
            "skim_sfa_fixed", target_dir, cbr_match=plan,
        )
        driver._write_yaml(
            cfg,
            driver.generated_config_path(
                output_root, "skim_sfa_fixed", "01_toy",
            ),
        )

        row = driver.collect_run_metrics(
            output_root, "skim_sfa_fixed", "01_toy",
        )
        assert row["cbr_match_status"] == "count_only"

        for out_dir, symbols in ((source_dir, 1200), (target_dir, 1200)):
            accounting_dir = out_dir / "accounting"
            accounting_dir.mkdir()
            (accounting_dir / "accounting_summary.json").write_text(
                json.dumps({
                    "total_channel_symbols": symbols,
                    "proxy_fraction": 0,
                }),
                encoding="utf-8",
            )
        row = driver.collect_run_metrics(
            output_root, "skim_sfa_fixed", "01_toy",
        )
        assert row["cbr_match_status"] == "verified"
        assert row["cbr_accounting_kind"] == "exact"
        assert row["measured_cbr_delta"] == 0

        (target_dir / "accounting" / "accounting_summary.json").write_text(
            json.dumps({
                "total_channel_symbols": 1250,
                "proxy_fraction": 0,
            }),
            encoding="utf-8",
        )
        row = driver.collect_run_metrics(
            output_root, "skim_sfa_fixed", "01_toy",
        )
        assert row["cbr_match_status"] == "mismatch"
        assert row["measured_cbr_delta"] == 50

    def test_equal_proxy_accounting_does_not_claim_verified_cbr(self, tmp_path):
        output_root = tmp_path / "out"
        source_dir = _write_complete_source_run(
            output_root, "skem_dsa_psss", "01_toy",
            n_frames=10, keyframes=[0, 5],
        )
        plan = driver.resolve_cbr_match(
            output_root, "skem_dsa_psss", "01_toy", target_n_frames=10,
        )
        target_dir = driver.out_dir_for(output_root, "skim_sfa_fixed", "01_toy")
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "keyframes.json").write_text(
            json.dumps({"keyframes": [0, 5], "boundaries": [True] * 10}),
            encoding="utf-8",
        )
        (target_dir / "temporal_metrics.csv").write_text(
            "n_frames\n10\n", encoding="utf-8",
        )
        verification = driver.verify_keyframe_count_match(target_dir, plan)
        (target_dir / "keyframe_count_match.json").write_text(
            json.dumps(verification), encoding="utf-8",
        )
        driver._write_yaml(
            driver.build_run_config(
                "skim_sfa_fixed", target_dir, cbr_match=plan,
            ),
            driver.generated_config_path(
                output_root, "skim_sfa_fixed", "01_toy",
            ),
        )
        for out_dir in (source_dir, target_dir):
            accounting_dir = out_dir / "accounting"
            accounting_dir.mkdir()
            (accounting_dir / "accounting_summary.json").write_text(
                json.dumps({
                    "total_channel_symbols": 1200,
                    "proxy_fraction": 0.5,
                }),
                encoding="utf-8",
            )

        row = driver.collect_run_metrics(
            output_root, "skim_sfa_fixed", "01_toy",
        )
        assert row["measured_cbr_delta"] == 0
        assert row["cbr_accounting_kind"] == "proxy_or_unknown"
        assert row["cbr_match_status"] == "count_only"
