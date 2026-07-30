"""tests/test_batch_remeasure_owlv2_vqa_10videos.py –
scripts/batch_remeasure_owlv2_vqa_10videos.py tests (ETRI 5차 follow-up:
10-video OWLv2/VQA/ensemble batch driver preparation).

No real weights/GPU/subprocess execution is a test dependency: every test
that would otherwise invoke ``scripts/remeasure_video_metrics.py`` either
uses ``--dry-run`` or monkeypatches ``subprocess.run`` to fail loudly if
called, mirroring tests/test_remeasure_video_metrics_cli.py's philosophy.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_script():
    path = _REPO / "scripts" / "batch_remeasure_owlv2_vqa_10videos.py"
    name = "batch_remeasure_owlv2_vqa_10videos"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec: the module defines @dataclass
    # classes, and Python 3.9's dataclasses._is_type() looks the module up via
    # sys.modules[cls.__module__] — without this it raises
    # AttributeError: 'NoneType' object has no attribute '__dict__'.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cli = _load_script()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: fake baseline/captions/gt/configs trees
# ─────────────────────────────────────────────────────────────────────────────

def _make_baseline(tmp_path, video_ids, with_recon=True):
    root = tmp_path / "baseline"
    for vid in video_ids:
        d = root / vid
        (d / "extracted_frames").mkdir(parents=True)
        if with_recon:
            (d / "recon_frames").mkdir(parents=True)
    return root


def _make_captions_gt(tmp_path, video_ids, skip_captions=(), skip_gt=()):
    cap_dir = tmp_path / "captions"
    gt_dir = tmp_path / "gt"
    cap_dir.mkdir()
    gt_dir.mkdir()
    for vid in video_ids:
        if vid not in skip_captions:
            (cap_dir / f"{vid}.txt").write_text("a caption\n", encoding="utf-8")
        if vid not in skip_gt:
            (gt_dir / f"{vid}.json").write_text(
                json.dumps({"n_frames": 1, "segments": []}), encoding="utf-8")
    return cap_dir, gt_dir


_FAKE_BASE_CONFIG_BODIES = {
    "etri_video_eval_owlv2.yaml": {
        "use_phase4": True,
        "heldout": {"clip_only_json": "manual_owlv2/heldout/clip_only_metrics.json"},
        "verifier": {"presence_mode": "owlv2_only", "presence_backends": ["owlv2"]},
    },
    "etri_video_eval_vqa.yaml": {
        "use_phase4": True,
        "heldout": {"clip_only_json": "manual_vqa/heldout/clip_only_metrics.json"},
        "verifier": {"presence_mode": "vqa_only", "presence_backends": ["vqa"]},
    },
    "etri_video_eval_ensemble.yaml": {
        "use_phase4": True,
        "heldout": {"clip_only_json": "manual_ensemble/heldout/clip_only_metrics.json"},
        "verifier": {
            "presence_mode": "ensemble_weighted",
            "presence_backends": ["clip", "owlv2", "vqa"],
            "object_vocabulary_filter": {"enabled": True, "use_gt_vocabulary": True},
        },
    },
}


def _make_fake_configs_dir(tmp_path):
    import yaml
    d = tmp_path / "configs"
    d.mkdir()
    for name, body in _FAKE_BASE_CONFIG_BODIES.items():
        (d / name).write_text(yaml.safe_dump(body), encoding="utf-8")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# discover_videos
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscoverVideos:
    def test_finds_videos_with_recon_frames(self, tmp_path):
        root = _make_baseline(tmp_path, ["01_a", "02_b", "03_c"])
        assert cli.discover_videos(root) == ["01_a", "02_b", "03_c"]

    def test_excludes_dirs_without_recon_frames(self, tmp_path):
        root = _make_baseline(tmp_path, ["01_a"])
        (root / "not_a_video_incomplete").mkdir()   # no recon_frames/
        assert cli.discover_videos(root) == ["01_a"]

    def test_missing_root_returns_empty(self, tmp_path):
        assert cli.discover_videos(tmp_path / "does_not_exist") == []

    def test_filter_subset_preserves_requested_order(self, tmp_path):
        root = _make_baseline(tmp_path, ["01_a", "02_b", "03_c"])
        assert cli.discover_videos(root, videos_filter=["03_c", "01_a"]) == ["03_c", "01_a"]

    def test_filter_unknown_video_raises(self, tmp_path):
        root = _make_baseline(tmp_path, ["01_a"])
        with pytest.raises(ValueError, match="not found"):
            cli.discover_videos(root, videos_filter=["01_a", "99_missing"])


# ─────────────────────────────────────────────────────────────────────────────
# Per-mode/video output paths
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputPaths:
    def test_mode_output_dir_layout(self):
        out = cli.mode_output_dir("/root", "owlv2", "01_person_walk")
        assert str(out) == "/root/owlv2/01_person_walk"

    def test_metric_delta_path_matches_task_spec_example(self):
        # exact example path from the task: .../owlv2/01_person_walk/heldout/metric_delta.json
        p = cli.metric_delta_path("outputs/etri_video_eval/remeasure_10videos", "owlv2", "01_person_walk")
        assert str(p) == "outputs/etri_video_eval/remeasure_10videos/owlv2/01_person_walk/heldout/metric_delta.json"

    def test_different_modes_never_collide_for_same_video(self):
        paths = {cli.metric_delta_path("/root", mode, "01_x") for mode in cli.MODE_SPECS}
        assert len(paths) == len(cli.MODE_SPECS)

    def test_different_videos_never_collide_for_same_mode(self):
        paths = {cli.metric_delta_path("/root", "owlv2", vid) for vid in ("01_x", "02_y", "03_z")}
        assert len(paths) == 3

    def test_generated_config_path_under_generated_configs_dir(self):
        p = cli.generated_config_path("/root", "vqa", "01_x")
        assert str(p) == f"/root/{cli.GENERATED_CONFIGS_DIRNAME}/vqa/01_x.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# build_mode_config — generated config content + overrides
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildModeConfig:
    def test_unknown_mode_raises(self, tmp_path):
        configs_dir = _make_fake_configs_dir(tmp_path)
        with pytest.raises(ValueError, match="Unknown mode"):
            cli.build_mode_config("nonexistent_mode", "01_x", tmp_path / "out", configs_dir=configs_dir)

    def test_missing_base_config_file_raises(self, tmp_path):
        empty_configs_dir = tmp_path / "empty_configs"
        empty_configs_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            cli.build_mode_config("owlv2", "01_x", tmp_path / "out", configs_dir=empty_configs_dir)

    def test_owlv2_mode_preserves_base_presence_mode_no_vocab_filter(self, tmp_path):
        from omegaconf import OmegaConf
        configs_dir = _make_fake_configs_dir(tmp_path)
        cfg = cli.build_mode_config("owlv2", "01_x", tmp_path / "out", configs_dir=configs_dir)
        assert OmegaConf.select(cfg, "verifier.presence_mode") == "owlv2_only"
        assert OmegaConf.select(cfg, "verifier.object_vocabulary_filter") is None

    def test_vqa_mode_preserves_base_presence_mode(self, tmp_path):
        from omegaconf import OmegaConf
        configs_dir = _make_fake_configs_dir(tmp_path)
        cfg = cli.build_mode_config("vqa", "01_x", tmp_path / "out", configs_dir=configs_dir)
        assert OmegaConf.select(cfg, "verifier.presence_mode") == "vqa_only"

    def test_ensemble_nofilter_disables_filter(self, tmp_path):
        from omegaconf import OmegaConf
        configs_dir = _make_fake_configs_dir(tmp_path)
        cfg = cli.build_mode_config("ensemble_nofilter", "01_x", tmp_path / "out", configs_dir=configs_dir)
        assert OmegaConf.select(cfg, "verifier.object_vocabulary_filter.enabled") is False

    def test_ensemble_gt_filter_enables_gt_vocabulary(self, tmp_path):
        from omegaconf import OmegaConf
        configs_dir = _make_fake_configs_dir(tmp_path)
        cfg = cli.build_mode_config("ensemble_gt_filter", "01_x", tmp_path / "out", configs_dir=configs_dir)
        assert OmegaConf.select(cfg, "verifier.object_vocabulary_filter.enabled") is True
        assert OmegaConf.select(cfg, "verifier.object_vocabulary_filter.use_gt_vocabulary") is True

    def test_ensemble_openworld_filter_disables_gt_vocabulary(self, tmp_path):
        from omegaconf import OmegaConf
        configs_dir = _make_fake_configs_dir(tmp_path)
        cfg = cli.build_mode_config("ensemble_openworld_filter", "01_x", tmp_path / "out", configs_dir=configs_dir)
        assert OmegaConf.select(cfg, "verifier.object_vocabulary_filter.enabled") is True
        assert OmegaConf.select(cfg, "verifier.object_vocabulary_filter.use_gt_vocabulary") is False

    def test_heldout_paths_redirected_under_mode_video_output_dir(self, tmp_path):
        from omegaconf import OmegaConf
        configs_dir = _make_fake_configs_dir(tmp_path)
        output_root = tmp_path / "out"
        cfg = cli.build_mode_config("owlv2", "01_person_walk", output_root, configs_dir=configs_dir)
        expected = str(output_root / "owlv2" / "01_person_walk" / "heldout" / "metric_delta.json")
        assert OmegaConf.select(cfg, "heldout.output_json") == expected
        assert OmegaConf.select(cfg, "heldout.clip_only_json") == str(
            output_root / "owlv2" / "01_person_walk" / "heldout" / "clip_only_metrics.json")

    def test_two_videos_same_mode_get_distinct_generated_paths(self, tmp_path):
        from omegaconf import OmegaConf
        configs_dir = _make_fake_configs_dir(tmp_path)
        output_root = tmp_path / "out"
        cfg_a = cli.build_mode_config("owlv2", "01_a", output_root, configs_dir=configs_dir)
        cfg_b = cli.build_mode_config("owlv2", "02_b", output_root, configs_dir=configs_dir)
        assert (OmegaConf.select(cfg_a, "heldout.output_json")
                != OmegaConf.select(cfg_b, "heldout.output_json"))


class TestWriteGeneratedConfig:
    def test_write_and_reload_round_trip(self, tmp_path):
        from omegaconf import OmegaConf
        configs_dir = _make_fake_configs_dir(tmp_path)
        cfg = cli.build_mode_config("ensemble_gt_filter", "01_x", tmp_path / "out", configs_dir=configs_dir)
        path = cli.write_generated_config(cfg, tmp_path / "generated" / "01_x.yaml")
        assert path.is_file()
        reloaded = OmegaConf.load(path)
        assert OmegaConf.select(reloaded, "verifier.object_vocabulary_filter.use_gt_vocabulary") is True
        # No _defaults_ leftover — already fully composed before writing.
        assert "_defaults_" not in reloaded


# ─────────────────────────────────────────────────────────────────────────────
# build_command
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCommand:
    def test_includes_captions_and_gt_when_present(self, tmp_path):
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x"])
        cmd = cli.build_command("01_x", "cfg.yaml", tmp_path / "baseline", "cuda:0",
                                captions_dir=cap_dir, gt_dir=gt_dir)
        assert "--captions" in cmd
        assert str(cap_dir / "01_x.txt") in cmd
        assert "--gt-metadata" in cmd
        assert str(gt_dir / "01_x.json") in cmd
        assert "--from-recon-frames" in cmd
        assert str(Path(tmp_path / "baseline" / "01_x")) in cmd

    def test_omits_captions_flag_when_file_missing(self, tmp_path):
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x"], skip_captions=["01_x"])
        cmd = cli.build_command("01_x", "cfg.yaml", tmp_path / "baseline", "cuda:0",
                                captions_dir=cap_dir, gt_dir=gt_dir)
        assert "--captions" not in cmd
        assert "--gt-metadata" in cmd

    def test_omits_gt_flag_when_file_missing(self, tmp_path):
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x"], skip_gt=["01_x"])
        cmd = cli.build_command("01_x", "cfg.yaml", tmp_path / "baseline", "cuda:0",
                                captions_dir=cap_dir, gt_dir=gt_dir)
        assert "--gt-metadata" not in cmd

    def test_device_and_config_forwarded(self, tmp_path):
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x"])
        cmd = cli.build_command("01_x", "/some/cfg.yaml", tmp_path / "baseline", "cpu",
                                captions_dir=cap_dir, gt_dir=gt_dir)
        assert cmd[cmd.index("--config") + 1] == "/some/cfg.yaml"
        assert cmd[cmd.index("--device") + 1] == "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# run_job / run_batch — dry-run, skip-existing, continue-on-error
# ─────────────────────────────────────────────────────────────────────────────

class TestRunJobDryRun:
    def test_dry_run_never_calls_subprocess(self, tmp_path, monkeypatch):
        def _boom(*a, **kw):
            raise AssertionError("subprocess.run must not be called in --dry-run")
        monkeypatch.setattr(cli.subprocess, "run", _boom)

        configs_dir = _make_fake_configs_dir(tmp_path)
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x"])
        baseline = _make_baseline(tmp_path, ["01_x"])
        output_root = tmp_path / "out"

        result = cli.run_job("owlv2", "01_x", output_root, baseline, "cuda:0",
                             captions_dir=cap_dir, gt_dir=gt_dir, configs_dir=configs_dir, dry_run=True)
        assert result.status == "dry_run"
        assert Path(result.config_path).is_file()          # config IS generated for inspection
        assert not Path(result.metric_delta_path).exists()  # but nothing was actually run

    def test_dry_run_command_field_matches_what_would_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(AssertionError()))
        configs_dir = _make_fake_configs_dir(tmp_path)
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x"])
        baseline = _make_baseline(tmp_path, ["01_x"])
        result = cli.run_job("vqa", "01_x", tmp_path / "out", baseline, "cuda:0",
                             captions_dir=cap_dir, gt_dir=gt_dir, configs_dir=configs_dir, dry_run=True)
        assert result.command[0:2] == [sys.executable, str(cli.REMEASURE_SCRIPT)]
        assert "--from-recon-frames" in result.command


class TestSkipExisting:
    def test_skips_when_metric_delta_already_exists(self, tmp_path, monkeypatch):
        def _boom(*a, **kw):
            raise AssertionError("subprocess.run must not be called when skip_existing hits")
        monkeypatch.setattr(cli.subprocess, "run", _boom)

        configs_dir = _make_fake_configs_dir(tmp_path)
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x"])
        baseline = _make_baseline(tmp_path, ["01_x"])
        output_root = tmp_path / "out"

        existing = cli.metric_delta_path(output_root, "owlv2", "01_x")
        existing.parent.mkdir(parents=True)
        existing.write_text("{}", encoding="utf-8")

        result = cli.run_job("owlv2", "01_x", output_root, baseline, "cuda:0",
                             captions_dir=cap_dir, gt_dir=gt_dir, configs_dir=configs_dir,
                             skip_existing=True)
        assert result.status == "skipped"
        assert existing.read_text(encoding="utf-8") == "{}"   # untouched

    def test_does_not_skip_when_flag_unset(self, tmp_path, monkeypatch):
        calls = []
        class _FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""
        monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: (calls.append(1), _FakeCompleted())[1])

        configs_dir = _make_fake_configs_dir(tmp_path)
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x"])
        baseline = _make_baseline(tmp_path, ["01_x"])
        output_root = tmp_path / "out"
        existing = cli.metric_delta_path(output_root, "owlv2", "01_x")
        existing.parent.mkdir(parents=True)
        existing.write_text("{}", encoding="utf-8")

        result = cli.run_job("owlv2", "01_x", output_root, baseline, "cuda:0",
                             captions_dir=cap_dir, gt_dir=gt_dir, configs_dir=configs_dir,
                             skip_existing=False)
        assert result.status == "ok"
        assert len(calls) == 1


class TestRunBatchFailureHandling:
    def _patch_subprocess_sequence(self, monkeypatch, returncodes):
        calls = []
        class _FakeCompleted:
            def __init__(self, rc):
                self.returncode = rc
                self.stdout = ""
                self.stderr = "boom" if rc else ""
        def _fake_run(*a, **kw):
            idx = len(calls)
            calls.append(1)
            return _FakeCompleted(returncodes[idx])
        monkeypatch.setattr(cli.subprocess, "run", _fake_run)
        return calls

    def test_stops_after_first_failure_by_default(self, tmp_path, monkeypatch):
        calls = self._patch_subprocess_sequence(monkeypatch, [1, 0, 0])
        configs_dir = _make_fake_configs_dir(tmp_path)
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x", "02_y", "03_z"])
        baseline = _make_baseline(tmp_path, ["01_x", "02_y", "03_z"])

        results = cli.run_batch(["owlv2"], ["01_x", "02_y", "03_z"], tmp_path / "out", baseline, "cuda:0",
                                captions_dir=cap_dir, gt_dir=gt_dir, configs_dir=configs_dir,
                                continue_on_error=False)
        assert len(results) == 1
        assert results[0].status == "failed"
        assert len(calls) == 1

    def test_continues_after_failure_when_flag_set(self, tmp_path, monkeypatch):
        calls = self._patch_subprocess_sequence(monkeypatch, [1, 0, 0])
        configs_dir = _make_fake_configs_dir(tmp_path)
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x", "02_y", "03_z"])
        baseline = _make_baseline(tmp_path, ["01_x", "02_y", "03_z"])

        results = cli.run_batch(["owlv2"], ["01_x", "02_y", "03_z"], tmp_path / "out", baseline, "cuda:0",
                                captions_dir=cap_dir, gt_dir=gt_dir, configs_dir=configs_dir,
                                continue_on_error=True)
        assert [r.status for r in results] == ["failed", "ok", "ok"]
        assert len(calls) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV/Markdown generation
# ─────────────────────────────────────────────────────────────────────────────

_FAKE_DELTA_TEMPLATE = {
    "n_items_clip_only": 100, "n_items_calibrated": 100, "n_items_diff": 0,
    "mean_severity_clip_only": 0.54, "mean_severity_calibrated": 0.24, "mean_severity_diff": -0.30,
    "ptc_clip_only": 0.26, "ptc_calibrated": 0.76, "ptc_diff": 0.50,
    "sfr_clip_only": 0.13, "sfr_calibrated": 0.05, "sfr_diff": -0.08,
    "sdi_clip_only": 0.001, "sdi_calibrated": -0.0008, "sdi_diff": -0.0018,
    "total_missing_objects_clip_only": 40, "total_missing_objects_calibrated": 10,
    "total_missing_objects_diff": -30,
    "total_additional_objects_clip_only": 20, "total_additional_objects_calibrated": 5,
    "total_additional_objects_diff": -15,
    "temporal_hallucination_rate_clip_only": 0.2, "temporal_hallucination_rate_calibrated": 0.05,
    "temporal_hallucination_rate_diff": -0.15,
    "note": "structural diff only",
}


def _write_fake_delta(output_root, mode, video_id, overrides=None):
    p = Path(cli.metric_delta_path(output_root, mode, video_id))
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(_FAKE_DELTA_TEMPLATE)
    payload.update(overrides or {})
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


class TestSummarizeBatch:
    def test_empty_output_root_returns_no_rows(self, tmp_path):
        assert cli.summarize_batch(tmp_path / "out") == []

    def test_collects_one_row_per_mode_video(self, tmp_path):
        output_root = tmp_path / "out"
        _write_fake_delta(output_root, "owlv2", "01_x")
        _write_fake_delta(output_root, "vqa", "01_x")
        _write_fake_delta(output_root, "owlv2", "02_y")
        rows = cli.summarize_batch(output_root)
        pairs = {(r["mode"], r["video_id"]) for r in rows}
        assert pairs == {("owlv2", "01_x"), ("vqa", "01_x"), ("owlv2", "02_y")}

    def test_row_contains_exact_requested_columns(self, tmp_path):
        output_root = tmp_path / "out"
        _write_fake_delta(output_root, "owlv2", "01_x")
        rows = cli.summarize_batch(output_root)
        expected_cols = {
            "video_id", "mode", "n_items",
            "mean_severity_clip_only", "mean_severity_calibrated", "mean_severity_diff",
            "ptc_clip_only", "ptc_calibrated", "ptc_diff",
            "sfr_clip_only", "sfr_calibrated", "sfr_diff",
            "sdi_clip_only", "sdi_calibrated", "sdi_diff",
            "total_missing_objects_clip_only", "total_missing_objects_calibrated",
            "total_missing_objects_diff",
            "total_additional_objects_clip_only", "total_additional_objects_calibrated",
            "total_additional_objects_diff",
            "temporal_hallucination_rate_clip_only", "temporal_hallucination_rate_calibrated",
            "temporal_hallucination_rate_diff",
        }
        assert set(rows[0].keys()) == expected_cols
        assert rows[0]["video_id"] == "01_x"
        assert rows[0]["mode"] == "owlv2"
        assert rows[0]["n_items"] == 100
        assert rows[0]["mean_severity_diff"] == -0.30

    def test_generated_configs_dir_never_treated_as_a_mode(self, tmp_path):
        output_root = tmp_path / "out"
        _write_fake_delta(output_root, "owlv2", "01_x")
        (output_root / cli.GENERATED_CONFIGS_DIRNAME / "owlv2").mkdir(parents=True)
        (output_root / cli.GENERATED_CONFIGS_DIRNAME / "owlv2" / "01_x.yaml").write_text("x", encoding="utf-8")
        rows = cli.summarize_batch(output_root)
        assert {r["mode"] for r in rows} == {"owlv2"}

    def test_modes_filter_restricts_rows(self, tmp_path):
        output_root = tmp_path / "out"
        _write_fake_delta(output_root, "owlv2", "01_x")
        _write_fake_delta(output_root, "vqa", "01_x")
        rows = cli.summarize_batch(output_root, modes=["owlv2"])
        assert {r["mode"] for r in rows} == {"owlv2"}

    def test_videos_filter_restricts_rows(self, tmp_path):
        output_root = tmp_path / "out"
        _write_fake_delta(output_root, "owlv2", "01_x")
        _write_fake_delta(output_root, "owlv2", "02_y")
        rows = cli.summarize_batch(output_root, videos=["01_x"])
        assert {r["video_id"] for r in rows} == {"01_x"}


class TestWriteSummaryFiles:
    def test_writes_csv_and_md(self, tmp_path):
        output_root = tmp_path / "out"
        _write_fake_delta(output_root, "owlv2", "01_x")
        _write_fake_delta(output_root, "ensemble_gt_filter", "01_x")
        rows = cli.summarize_batch(output_root)
        csv_path, md_path = cli.write_summary_files(rows, output_root)

        assert csv_path == output_root / "summary_metrics.csv"
        assert md_path == output_root / "summary_metrics.md"
        assert csv_path.is_file() and md_path.is_file()

        with open(csv_path, newline="", encoding="utf-8") as fh:
            csv_rows = list(csv.DictReader(fh))
        assert len(csv_rows) == 2
        assert {r["mode"] for r in csv_rows} == {"owlv2", "ensemble_gt_filter"}

        md_text = md_path.read_text(encoding="utf-8")
        assert "ensemble_gt_filter" in md_text
        assert "object-preservation" in md_text.lower() or "GT-object-only" in md_text

    def test_empty_rows_writes_nothing(self, tmp_path):
        csv_path, md_path = cli.write_summary_files([], tmp_path / "out")
        assert csv_path is None and md_path is None
        assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []


# ─────────────────────────────────────────────────────────────────────────────
# CLI main() — --dry-run / --skip-existing / --summary-only end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestMainDryRun:
    def test_dry_run_end_to_end_generates_configs_but_no_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli.subprocess, "run",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no subprocess in dry-run")))
        configs_dir = _make_fake_configs_dir(tmp_path)
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x", "02_y"])
        baseline = _make_baseline(tmp_path, ["01_x", "02_y"])
        output_root = tmp_path / "out"

        cli.main([
            "--baseline-root", str(baseline), "--captions-dir", str(cap_dir), "--gt-dir", str(gt_dir),
            "--configs-dir", str(configs_dir), "--output-root", str(output_root),
            "--modes", "owlv2,vqa", "--dry-run",
        ])

        for mode in ("owlv2", "vqa"):
            for vid in ("01_x", "02_y"):
                assert cli.generated_config_path(output_root, mode, vid).is_file()
                assert not cli.metric_delta_path(output_root, mode, vid).exists()
        assert not (output_root / "summary_metrics.csv").exists()


class TestMainRelativeOutputRoot:
    """Regression: a relative --output-root must not get double-nested.

    build_mode_config() writes output-artefact paths into the generated
    per-(mode, video) config, which remeasure_video_metrics.py later re-loads
    via sgdjscc_lab.config.load_config() — any path still relative at that
    point is re-resolved relative to the GENERATED config's own directory
    (<output-root>/_generated_configs/<mode>/), not the CWD main() was
    invoked from. main() must resolve --output-root (and the other path
    flags) to absolute before anything derives paths from it, or a relative
    --output-root silently produces a doubled-up path like
    "rel_out/_generated_configs/owlv2/.../rel_out/owlv2/.../metric_delta.json"
    instead of the intended "<cwd>/rel_out/owlv2/.../metric_delta.json".
    """

    def test_relative_output_root_resolves_against_cwd_not_doubled(self, tmp_path, monkeypatch):
        from omegaconf import OmegaConf

        monkeypatch.setattr(cli.subprocess, "run",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no subprocess in dry-run")))
        configs_dir = _make_fake_configs_dir(tmp_path)
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x"])
        baseline = _make_baseline(tmp_path, ["01_x"])
        monkeypatch.chdir(tmp_path)

        cli.main([
            "--baseline-root", str(baseline), "--captions-dir", str(cap_dir), "--gt-dir", str(gt_dir),
            "--configs-dir", str(configs_dir), "--output-root", "rel_out",
            "--modes", "owlv2", "--dry-run",
        ])

        expected_root = (tmp_path / "rel_out").resolve()
        generated = expected_root / cli.GENERATED_CONFIGS_DIRNAME / "owlv2" / "01_x.yaml"
        assert generated.is_file()

        cfg = OmegaConf.load(generated)
        output_json = Path(OmegaConf.select(cfg, "heldout.output_json"))
        assert output_json.is_absolute()
        assert output_json == expected_root / "owlv2" / "01_x" / "heldout" / "metric_delta.json"
        # The double-nesting bug this guards against: "rel_out" must appear
        # exactly once in the resolved path, not twice (once for the
        # generated-config location, once re-resolved from a leftover
        # relative fragment).
        assert str(output_json).count("rel_out") == 1

    def test_relative_baseline_captions_gt_configs_dirs_also_resolved(self, tmp_path, monkeypatch):
        """The same class of bug for --baseline-root/--captions-dir/--gt-dir/
        --configs-dir: build_command()'s --from-recon-frames/--captions/
        --gt-metadata values must be absolute regardless of what directory
        main() was invoked from, since remeasure_video_metrics.py (run as a
        separate subprocess) has its own CWD assumptions."""
        monkeypatch.setattr(cli.subprocess, "run",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no subprocess in dry-run")))
        _make_fake_configs_dir(tmp_path)
        _make_captions_gt(tmp_path, ["01_x"])
        _make_baseline(tmp_path, ["01_x"])
        monkeypatch.chdir(tmp_path)

        result_holder = {}
        orig_run_job = cli.run_job

        def _capturing_run_job(mode, video_id, *a, **kw):
            r = orig_run_job(mode, video_id, *a, **kw)
            result_holder[(mode, video_id)] = r
            return r
        monkeypatch.setattr(cli, "run_job", _capturing_run_job)

        cli.main([
            "--baseline-root", "baseline", "--captions-dir", "captions", "--gt-dir", "gt",
            "--configs-dir", "configs", "--output-root", "out",
            "--modes", "owlv2", "--dry-run",
        ])

        cmd = result_holder[("owlv2", "01_x")].command
        from_recon_idx = cmd.index("--from-recon-frames") + 1
        captions_idx = cmd.index("--captions") + 1
        gt_idx = cmd.index("--gt-metadata") + 1
        assert Path(cmd[from_recon_idx]).is_absolute()
        assert Path(cmd[captions_idx]).is_absolute()
        assert Path(cmd[gt_idx]).is_absolute()


class TestMainSkipExisting:
    def test_skip_existing_avoids_subprocess_for_completed_job(self, tmp_path, monkeypatch):
        calls = []
        class _FakeCompleted:
            returncode = 0
            stdout = ""
            stderr = ""
        monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: (calls.append(1), _FakeCompleted())[1])

        configs_dir = _make_fake_configs_dir(tmp_path)
        cap_dir, gt_dir = _make_captions_gt(tmp_path, ["01_x"])
        baseline = _make_baseline(tmp_path, ["01_x"])
        output_root = tmp_path / "out"
        _write_fake_delta(output_root, "owlv2", "01_x")

        cli.main([
            "--baseline-root", str(baseline), "--captions-dir", str(cap_dir), "--gt-dir", str(gt_dir),
            "--configs-dir", str(configs_dir), "--output-root", str(output_root),
            "--modes", "owlv2", "--videos", "01_x", "--skip-existing",
        ])
        assert len(calls) == 0


class TestMainSummaryOnly:
    def test_summary_only_never_touches_baseline_or_subprocess(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli.subprocess, "run",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no subprocess in --summary-only")))
        output_root = tmp_path / "out"
        _write_fake_delta(output_root, "owlv2", "01_x")
        _write_fake_delta(output_root, "vqa", "01_x")

        # A deliberately-nonexistent baseline root — --summary-only must never
        # call discover_videos()/read it at all.
        cli.main([
            "--baseline-root", str(tmp_path / "no_such_baseline_dir"),
            "--output-root", str(output_root), "--summary-only",
        ])

        assert (output_root / "summary_metrics.csv").is_file()
        assert (output_root / "summary_metrics.md").is_file()

    def test_summary_only_with_no_results_exits_nonzero(self, tmp_path):
        with pytest.raises(SystemExit):
            cli.main(["--output-root", str(tmp_path / "empty_out"), "--summary-only"])


class TestMainErrorHandling:
    def test_unknown_mode_exits(self, tmp_path):
        with pytest.raises(SystemExit, match="unknown mode"):
            cli.main(["--output-root", str(tmp_path / "out"), "--modes", "not_a_real_mode", "--dry-run"])

    def test_no_videos_found_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            cli.main([
                "--baseline-root", str(tmp_path / "empty_baseline"),
                "--output-root", str(tmp_path / "out"), "--dry-run",
            ])


# ─────────────────────────────────────────────────────────────────────────────
# MODE_SPECS sanity (documents the 5 required modes + their caveats)
# ─────────────────────────────────────────────────────────────────────────────

class TestModeSpecs:
    def test_all_five_required_modes_present(self):
        assert set(cli.MODE_SPECS) == {
            "owlv2", "vqa", "ensemble_nofilter", "ensemble_gt_filter", "ensemble_openworld_filter",
        }

    def test_ensemble_nofilter_has_a_caveat(self):
        assert cli.MODE_SPECS["ensemble_nofilter"].caveat

    def test_owlv2_and_vqa_base_configs_match_existing_repo_configs(self):
        assert cli.MODE_SPECS["owlv2"].base_config == "etri_video_eval_owlv2.yaml"
        assert cli.MODE_SPECS["vqa"].base_config == "etri_video_eval_vqa.yaml"
        assert cli.MODE_SPECS["ensemble_nofilter"].base_config == "etri_video_eval_ensemble.yaml"
        assert cli.MODE_SPECS["ensemble_gt_filter"].base_config == "etri_video_eval_ensemble.yaml"
        assert cli.MODE_SPECS["ensemble_openworld_filter"].base_config == "etri_video_eval_ensemble.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Real repo configs — lightweight structural check (no torch/GPU needed;
# sgdjscc_lab.config.load_config only needs omegaconf). Confirms this driver
# actually works against the real configs/ directory, not just the fakes above.
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildModeConfigAgainstRealRepoConfigs:
    def test_all_modes_build_against_real_configs_dir(self, tmp_path):
        for mode in cli.MODE_SPECS:
            cfg = cli.build_mode_config(mode, "01_person_walk", tmp_path / "out")
            assert cfg is not None

    def test_ensemble_modes_apply_over_real_ensemble_config_defaults(self, tmp_path):
        from omegaconf import OmegaConf
        cfg = cli.build_mode_config("ensemble_openworld_filter", "01_person_walk", tmp_path / "out")
        # Real configs/etri_video_eval_ensemble.yaml defaults use_gt_vocabulary=True —
        # the openworld mode's override must actually flip it to False, not just
        # inherit the base config's default.
        assert OmegaConf.select(cfg, "verifier.object_vocabulary_filter.use_gt_vocabulary") is False
