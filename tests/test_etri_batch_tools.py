"""tests/test_etri_batch_tools.py – ETRI 10-video batch tooling tests.

Covers the three batch scripts added for the ETRI 10-video evaluation set
(scripts/run_etri_video_eval.py, scripts/summarize_etri_video_eval.py,
scripts/generate_etri_final_report.py):

- manifest parsing + GT segment-JSON → presence-backend input conversion,
- per-stage config generation (stage toggles, isolated output dirs,
  motion-sweep thresholds, accounting/rate-reliability wiring),
- subprocess command assembly (evaluate_video vs remeasure entry points),
- stage summarizers against a synthetic on-disk output tree,
- final-report generation from the synthetic summaries.

No GPU, no checkpoints, no video backend needed — everything here runs on
synthetic files. The end-to-end --no-models batch run over real ETRI videos is
a separate manual smoke (see docs/etri_strategy.md conventions).
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


def _load(script_name: str):
    path = _REPO / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("run_etri_video_eval.py")
summarizer = _load("summarize_etri_video_eval.py")
reporter = _load("generate_etri_final_report.py")


# ──────────────────────────────────────────────────────────────────────────────
# Manifest / GT helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestManifestAndGt:
    def _write_dataset(self, root: Path):
        (root / "processed").mkdir(parents=True)
        (root / "captions").mkdir()
        (root / "gt").mkdir()
        (root / "processed" / "01_toy.mp4").write_bytes(b"")
        (root / "captions" / "01_toy.txt").write_text("a person\n" * 4)
        (root / "gt" / "01_toy.json").write_text(json.dumps({
            "video_id": "01_toy", "n_frames": 4,
            "segments": [
                {"start_frame": 0, "end_frame": 1, "objects": [
                    {"label": "person", "count": 1, "presence": "visible"}]},
                {"start_frame": 2, "end_frame": 3, "objects": [
                    {"label": "person", "count": 1, "presence": "absent"},
                    {"label": "car", "count": 1, "presence": "visible"}]},
            ],
        }))
        with open(root / "manifest.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "name", "processed_file"])
            w.writeheader()
            w.writerow({"id": "01", "name": "toy", "processed_file": "processed/01_toy.mp4"})

    def test_read_manifest(self, tmp_path):
        self._write_dataset(tmp_path)
        entries = runner.read_manifest(tmp_path)
        assert len(entries) == 1
        e = entries[0]
        assert e["key"] == "01_toy"
        assert e["processed"].exists()
        assert e["captions"] is not None and e["captions"].name == "01_toy.txt"
        assert e["gt"] is not None
        assert e["row"]["id"] == "01"

    def test_convert_gt_to_presence(self, tmp_path):
        self._write_dataset(tmp_path)
        gt = json.loads((tmp_path / "gt" / "01_toy.json").read_text())
        presence = runner.convert_gt_to_presence(gt)
        assert set(presence) == {f"frame_{i:05d}" for i in range(4)}
        assert presence["frame_00000"] == {"car": False, "person": True}
        # segment 2: person marked absent, car visible
        assert presence["frame_00003"] == {"car": True, "person": False}


# ──────────────────────────────────────────────────────────────────────────────
# Config generation / command assembly
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildRunConfig:
    def _cfg(self, stage, **kw):
        return runner.build_run_config(_REPO, Path("/data/vid.mp4"), stage, **kw)

    def test_baseline_defaults(self):
        cfg = self._cfg("baseline")
        assert cfg["use_phase4"] is True and cfg["use_packet_eval"] is True
        # Relative output paths → resolved against the run dir by load_config.
        assert cfg["temporal_csv"] == "temporal_metrics.csv"
        assert cfg["video_io"]["recon_video"] == "recon.mp4"
        # Per-run (not a path shared across stages/videos) — see
        # build_run_config's docstring on the extract_frames() race this avoids.
        assert cfg["video_io"]["extracted_frames_dir"] == "extracted_frames"
        # Baseline must not switch on any gated feature.
        for key in ("use_packet_verifier", "use_video_gen", "accounting", "heldout"):
            assert key not in cfg
        # _defaults_ entries are absolute and must exist.
        for frag in cfg["_defaults_"]:
            assert Path(f"{frag}.yaml").exists(), frag

    def test_model_root_is_absolute_and_survives_nested_run_dir(self, tmp_path):
        """Regression: model/sgdjscc.yaml's model_root ("../checkpoints/") is
        relative to a config file living directly under configs/ — but this
        driver writes config.yaml two levels below output_root
        (<output_root>/<stage>/<video>/), where "../checkpoints" resolves to
        <output_root>/<stage>/checkpoints (wrong) instead of the real
        checkpoints/ dir. Only --no-models runs ever existed before this was
        caught, since build_models() (the only reader of model_root) is
        skipped entirely in that mode. build_run_config must override
        model_root absolutely so a real-model run's config.yaml — written
        anywhere under output_root — still finds the real checkpoints."""
        cfg = self._cfg("baseline")
        assert Path(cfg["model_root"]).is_absolute()
        assert cfg["model_root"] == str((_REPO / "checkpoints").resolve())

        # Simulate load_config()'s path resolution from a nested run dir and
        # confirm it does NOT get corrupted back into a relative-looking join.
        from sgdjscc_lab.config import load_config
        run_dir = tmp_path / "real_all_frames_step10" / "01_person_walk"
        run_dir.mkdir(parents=True)
        runner._write_yaml(cfg, run_dir / "config.yaml")
        loaded = load_config(run_dir / "config.yaml")
        assert loaded.model_root == str((_REPO / "checkpoints").resolve())

    def test_motion_sweep_threshold(self):
        cfg = self._cfg("motion_sweep", motion_threshold=0.08)
        assert cfg["temporal"]["motion_threshold"] == 0.08
        cfg_off = self._cfg("motion_sweep", motion_threshold=None)
        assert "temporal" not in cfg_off  # gate off = fragment default (null)

    def test_verifier_stage(self):
        cfg = self._cfg("verifier")
        assert cfg["use_packet_verifier"] is True
        assert cfg["verifier"]["enabled"] is True
        assert cfg["verifier"]["report_json"] == "packet_match_report.json"

    def test_generate_stage(self):
        cfg = self._cfg("generate")
        assert cfg["use_video_gen"] is True
        vg = cfg["video_generator"]
        assert vg["enabled"] is True and vg["conditioning_mode"] == "start_only"
        assert vg["generate_delta_min"] == 0.0
        assert cfg["temporal"]["motion_threshold"] == pytest.approx(0.05)

    def test_bidirectional_stage(self):
        cfg = self._cfg("bidirectional")
        vg = cfg["video_generator"]
        assert vg["conditioning_mode"] == "bidirectional"
        assert vg["bidirectional_missing_end_policy"] == "fallback_start_only"
        assert vg["comparison_enabled"] is True
        assert vg["comparison_output"] == "generation_mode_comparison.json"

    def test_accounting_stage(self):
        cfg = self._cfg("accounting", rate_reliability_label="01_toy")
        assert cfg["accounting"]["enabled"] is True
        assert cfg["accounting"]["output_dir"] == "accounting"
        assert cfg["rate_reliability"]["label"] == "01_toy"
        # Always a per-run local file (never a shared path) so concurrent
        # accounting runs across videos/processes cannot race on one CSV —
        # summarize_etri_video_eval.py merges these post-hoc.
        assert cfg["rate_reliability"]["curve_csv"] == "accounting/rate_reliability_curve.csv"
        # accounting stage also enables the verifier for mean_severity.
        assert cfg["use_packet_verifier"] is True

    def test_heldout_stage(self):
        cfg = self._cfg("heldout")
        assert cfg["heldout"]["enabled"] is True
        assert cfg["heldout"]["output_json"] == "heldout/metric_delta.json"

    def test_generated_config_loads_through_load_config(self, tmp_path):
        """The generated YAML must compose with the real fragment set."""
        from omegaconf import OmegaConf
        from sgdjscc_lab.config import load_config

        cfg_dict = runner.build_run_config(
            _REPO, Path("/data/vid.mp4"), "accounting", rate_reliability_label="x")
        p = tmp_path / "config.yaml"
        OmegaConf.save(OmegaConf.create(cfg_dict), str(p))
        cfg = load_config(p)
        # Relative outputs resolved against the run dir:
        assert cfg.temporal_csv == str(tmp_path / "temporal_metrics.csv")
        assert cfg.accounting.output_dir == str(tmp_path / "accounting")
        # extracted_frames_dir resolves per-run too (not some shared cache):
        assert OmegaConf.select(cfg, "video_io.extracted_frames_dir") == \
            str(tmp_path / "extracted_frames")
        assert OmegaConf.select(cfg, "rate_reliability.curve_csv") == \
            str(tmp_path / "accounting" / "rate_reliability_curve.csv")
        # ${accounting.output_dir} interpolation must follow the override:
        assert OmegaConf.select(cfg, "accounting.summary_json") == \
            str(tmp_path / "accounting" / "accounting_summary.json")
        # Fragment values still present (video/default.yaml):
        assert OmegaConf.select(cfg, "keyframe.max_gop") == 12

    def test_stage_out_dir_isolated(self):
        base = runner.stage_out_dir(Path("/o"), "baseline", "01_toy")
        sweep = runner.stage_out_dir(Path("/o"), "motion_sweep", "01_toy", threshold=0.05)
        sweep_off = runner.stage_out_dir(Path("/o"), "motion_sweep", "01_toy", threshold=None)
        assert base == Path("/o/baseline/01_toy")
        assert sweep == Path("/o/motion_sweep/th_0.05/01_toy")
        assert sweep_off == Path("/o/motion_sweep/th_off/01_toy")
        assert len({base, sweep, sweep_off}) == 3

    def test_build_command(self):
        cmd = runner.build_command(_REPO, "baseline", Path("/o/c.yaml"),
                                   captions=Path("/d/c.txt"), no_models=True)
        assert "evaluate_video.py" in cmd[1]
        assert "--save-video" in cmd and "--no-models" in cmd
        assert "--captions" in cmd

        cmd_h = runner.build_command(_REPO, "heldout", Path("/o/c.yaml"), no_models=True)
        assert "remeasure_video_metrics.py" in cmd_h[1]
        assert "--save-video" not in cmd_h


class TestRemapDeviceForCudaVisible:
    """SGDJSCC's DiffusionGenerator.encode_text() hardcodes .cuda() (always the
    process's default CUDA device) instead of honouring the requested device —
    see remap_device_for_cuda_visible's docstring. This must be applied to
    ANY cuda:N device (including cuda:0), and it must work whether or not a
    base env dict is already available (parallel path passes a thread-limited
    one; sequential path has none)."""

    def test_noop_for_non_cuda_device(self):
        env, dev = runner.remap_device_for_cuda_visible("cpu", {"X": "1"})
        assert dev == "cpu"
        assert env == {"X": "1"}

    def test_noop_for_none_device(self):
        env, dev = runner.remap_device_for_cuda_visible(None, None)
        assert dev is None and env is None

    def test_remaps_cuda_n_with_given_base_env(self):
        env, dev = runner.remap_device_for_cuda_visible("cuda:2", {"BASE": "x"})
        assert dev == "cuda:0"
        assert env["CUDA_VISIBLE_DEVICES"] == "2"
        assert env["BASE"] == "x"

    def test_remaps_cuda_n_building_its_own_env_when_none_given(self):
        """The sequential (--parallel 1) dispatch path has no pre-built env —
        remap must still work by falling back to a fresh os.environ copy,
        not silently skip the fix because env was None."""
        env, dev = runner.remap_device_for_cuda_visible("cuda:1", None)
        assert dev == "cuda:0"
        assert env is not None
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
        # A real os.environ snapshot, not an empty dict.
        assert len(env) > 1

    def test_remaps_cuda_zero_too(self):
        """cuda:0 is remapped as well (harmless CUDA_VISIBLE_DEVICES=0), so
        there is no special-cased "only non-zero indices are fixed" gap."""
        env, dev = runner.remap_device_for_cuda_visible("cuda:0", None)
        assert dev == "cuda:0"
        assert env["CUDA_VISIBLE_DEVICES"] == "0"


class TestSequentialDeviceRemapAppliedInMain:
    """Regression: the CUDA_VISIBLE_DEVICES remap used to only be wired into
    the --parallel > 1 dispatch branch, so `--parallel 1 --device cuda:1`
    (the default, most common single-GPU invocation style) still hit
    SGDJSCC's .cuda() hardcoding crash. main() must remap for every dispatch,
    parallel or not."""

    def _make_dataset(self, data_root: Path):
        data_root.mkdir(parents=True)
        (data_root / "processed").mkdir()
        video = data_root / "processed" / "01_toy.mp4"
        video.write_bytes(b"fake")
        (data_root / "manifest.csv").write_text(
            "id,name,raw_file,processed_file,frames_dir,width,height,fps,duration_sec,"
            "n_frames,primary_objects,event\n"
            "01,toy,raw/01_toy.mp4,processed/01_toy.mp4,frames/01_toy,64,64,10,1,10,x,x\n",
            encoding="utf-8",
        )

    def test_sequential_run_with_cuda_device_gets_env_remap(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        self._make_dataset(data_root)
        output_root = tmp_path / "out"
        seen = {}

        def fake_run(cmd, stdout, stderr, cwd, env=None):
            seen["cmd"] = cmd
            seen["env"] = env
            (Path(cmd[cmd.index("--config") + 1]).parent / "temporal_metrics.csv").write_text("n_frames\n1\n")
            stdout.write("ok\n")
            class _P:
                returncode = 0
            return _P()

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        argv = ["run_etri_video_eval.py", "--data-root", str(data_root),
               "--output-root", str(output_root), "--stages", "baseline",
               "--device", "cuda:1", "--gpu-log-interval", "0"]
        monkeypatch.setattr(sys, "argv", argv)
        runner.main()

        # The subprocess must have been launched with --device cuda:0 (the
        # remapped, in-process value) and CUDA_VISIBLE_DEVICES=1 in its env —
        # not the raw "--device cuda:1" that would crash inside SGDJSCC.
        assert "--device" in seen["cmd"]
        idx = seen["cmd"].index("--device")
        assert seen["cmd"][idx + 1] == "cuda:0"
        assert seen["env"] is not None
        assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "1"

        # batch_status.json still records the PHYSICAL device for traceability.
        status = json.loads((output_root / "batch_status.json").read_text())
        assert status[0]["device"] == "cuda:1"

    def test_sequential_run_without_cuda_device_env_stays_none(self, tmp_path, monkeypatch):
        """No --device given (e.g. --no-models runs) → no remap needed, and
        subprocess.run must be called exactly as before (no env kwarg) —
        this is also what keeps the pre-existing --no-models test fixtures
        (which use a stricter (cmd, stdout, stderr, cwd) fake signature) passing."""
        data_root = tmp_path / "data"
        self._make_dataset(data_root)
        output_root = tmp_path / "out"
        seen = {}

        def fake_run(cmd, stdout, stderr, cwd):
            seen["called_without_env_kwarg"] = True
            (Path(cmd[cmd.index("--config") + 1]).parent / "temporal_metrics.csv").write_text("n_frames\n1\n")
            stdout.write("ok\n")
            class _P:
                returncode = 0
            return _P()

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        argv = ["run_etri_video_eval.py", "--data-root", str(data_root),
               "--output-root", str(output_root), "--stages", "baseline",
               "--no-models", "--gpu-log-interval", "0"]
        monkeypatch.setattr(sys, "argv", argv)
        runner.main()
        assert seen.get("called_without_env_kwarg") is True


class TestRunOneMetadata:
    """batch_status.json rows must record which config (no_models/snr/device)
    produced them, since a run's output directory is deterministic from
    (stage, video, motion_threshold) alone and a rerun with a different
    config silently overwrites the same files."""

    def _entry(self):
        return {"key": "01_toy", "processed": Path("/data/01_toy.mp4"),
                "captions": None, "gt": None}

    def test_ok_run_records_metadata(self, tmp_path, monkeypatch):
        def fake_run(cmd, stdout, stderr, cwd):
            stdout.write("ok\n")
            class _P:
                returncode = 0
            return _P()

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        cfg = runner.build_run_config(_REPO, Path("/data/01_toy.mp4"), "baseline")
        out_dir = tmp_path / "baseline" / "01_toy"
        status = runner.run_one(_REPO, "baseline", self._entry(), out_dir, cfg,
                                no_models=True, save_video=True, device="cuda:0", snr=5.0)
        assert status["status"] == "ok"
        assert status["no_models"] is True
        assert status["snr"] == 5.0
        assert status["device"] == "cuda:0"

    def test_skipped_run_without_prior_has_unknown_provenance(self, tmp_path):
        """No prior batch_status.json record → produced-by config must be
        reported as unknown (None), never fabricated from this invocation's
        args — the marker file could have been produced by any config."""
        out_dir = tmp_path / "baseline" / "01_toy"
        out_dir.mkdir(parents=True)
        (out_dir / "temporal_metrics.csv").write_text("x")
        cfg = runner.build_run_config(_REPO, Path("/data/01_toy.mp4"), "baseline")
        status = runner.run_one(_REPO, "baseline", self._entry(), out_dir, cfg,
                                no_models=False, save_video=True, device="cuda:0",
                                snr=5.0, skip_existing=True)
        assert status["status"] == "skipped"
        assert status["no_models"] is None
        assert status["snr"] is None and status["device"] is None
        assert status["note"]
        # What THIS invocation asked for is still recorded, separately:
        assert status["requested_no_models"] is False
        assert status["requested_snr"] == 5.0
        assert status["requested_device"] == "cuda:0"

    def test_skipped_run_with_prior_preserves_produced_provenance(self, tmp_path):
        """The bug this guards against: a later --skip-existing invocation
        with different args must not overwrite the record of what actually
        produced the on-disk files (cmd/duration_sec/no_models/snr/device)."""
        out_dir = tmp_path / "baseline" / "01_toy"
        out_dir.mkdir(parents=True)
        (out_dir / "temporal_metrics.csv").write_text("x")
        cfg = runner.build_run_config(_REPO, Path("/data/01_toy.mp4"), "baseline")
        prior = {"stage": "baseline", "video": "01_toy", "out_dir": str(out_dir),
                 "status": "ok", "returncode": 0, "duration_sec": 12.3,
                 "cmd": "python evaluate_video.py --no-models",
                 "no_models": True, "snr": None, "device": None,
                 "requested_no_models": True, "requested_snr": None, "requested_device": None}
        status = runner.run_one(_REPO, "baseline", self._entry(), out_dir, cfg,
                                no_models=False, save_video=True, device="cuda:0",
                                snr=5.0, skip_existing=True, prior=prior)
        assert status["status"] == "skipped"
        # Produced-by config comes from prior, NOT this invocation's args:
        assert status["no_models"] is True
        assert status["snr"] is None
        assert status["device"] is None
        assert status["cmd"] == "python evaluate_video.py --no-models"
        assert status["duration_sec"] == 12.3
        # Requested (this invocation's) args recorded separately:
        assert status["requested_no_models"] is False
        assert status["requested_snr"] == 5.0
        assert status["requested_device"] == "cuda:0"

    def test_skipped_run_with_failed_prior_flags_uncertain_provenance(self, tmp_path):
        """A marker file can exist even though the last recorded run for this
        output 'failed' (partial output before a crash, or a stale marker an
        earlier ok run left and the failed rerun didn't clean up). The
        carried-forward no_models/snr/device are only the LAST ATTEMPTED
        config in that case, not a confirmed produced-by one — the row must
        say so rather than look identical to a clean 'ok' prior."""
        out_dir = tmp_path / "baseline" / "01_toy"
        out_dir.mkdir(parents=True)
        (out_dir / "temporal_metrics.csv").write_text("x")  # partial/stale artefact
        cfg = runner.build_run_config(_REPO, Path("/data/01_toy.mp4"), "baseline")
        prior = {"stage": "baseline", "video": "01_toy", "out_dir": str(out_dir),
                 "status": "failed", "returncode": 1, "duration_sec": 3.1,
                 "cmd": "python evaluate_video.py --no-models",
                 "no_models": True, "snr": None, "device": None,
                 "requested_no_models": True, "requested_snr": None, "requested_device": None}
        status = runner.run_one(_REPO, "baseline", self._entry(), out_dir, cfg,
                                no_models=False, save_video=True, device="cuda:0",
                                snr=5.0, skip_existing=True, prior=prior)
        assert status["status"] == "skipped"
        assert status["prior_status"] == "failed"
        assert "failed" in status["note"]
        # Last-attempted config is still carried through (better than nothing):
        assert status["no_models"] is True
        assert status["cmd"] == "python evaluate_video.py --no-models"
        # This invocation's request is still recorded separately:
        assert status["requested_device"] == "cuda:0"

    def test_skipped_run_with_ok_prior_has_no_failed_flag(self, tmp_path):
        """Sanity check: an ok prior must NOT get the failed-provenance flag."""
        out_dir = tmp_path / "baseline" / "01_toy"
        out_dir.mkdir(parents=True)
        (out_dir / "temporal_metrics.csv").write_text("x")
        cfg = runner.build_run_config(_REPO, Path("/data/01_toy.mp4"), "baseline")
        prior = {"stage": "baseline", "video": "01_toy", "out_dir": str(out_dir),
                 "status": "ok", "returncode": 0, "duration_sec": 5.0,
                 "cmd": "python evaluate_video.py --no-models",
                 "no_models": True, "snr": None, "device": None,
                 "requested_no_models": True, "requested_snr": None, "requested_device": None}
        status = runner.run_one(_REPO, "baseline", self._entry(), out_dir, cfg,
                                no_models=True, save_video=True, skip_existing=True, prior=prior)
        assert "prior_status" not in status
        assert "note" not in status


class TestMainSkipExistingProvenance:
    """End-to-end: two real main() invocations against a synthetic 1-video
    dataset, with subprocess.run faked so no actual pipeline runs. Reproduces
    the exact scenario the bug fix targets: build with --no-models, then
    rerun with different args + --skip-existing, and check batch_status.json
    still describes the ORIGINAL run, not the second invocation's args."""

    def _make_dataset(self, root: Path):
        (root / "processed").mkdir(parents=True)
        (root / "processed" / "01_toy.mp4").write_bytes(b"")
        with open(root / "manifest.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "name", "processed_file"])
            w.writeheader()
            w.writerow({"id": "01", "name": "toy", "processed_file": "processed/01_toy.mp4"})

    @staticmethod
    def _fake_run_writes_marker(cmd, stdout, stderr, cwd):
        cfg_path = Path(cmd[cmd.index("--config") + 1])
        (cfg_path.parent / "temporal_metrics.csv").write_text("n_frames\n1\n")
        stdout.write("ok\n")
        class _P:
            returncode = 0
        return _P()

    def test_skip_existing_does_not_clobber_original_run_provenance(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        self._make_dataset(data_root)
        output_root = tmp_path / "out"
        argv_base = ["run_etri_video_eval.py", "--data-root", str(data_root),
                    "--output-root", str(output_root), "--stages", "baseline"]

        monkeypatch.setattr(runner.subprocess, "run", self._fake_run_writes_marker)
        monkeypatch.setattr(sys, "argv", argv_base + ["--no-models"])
        runner.main()

        row1 = json.loads((output_root / "batch_status.json").read_text())[0]
        assert row1["status"] == "ok" and row1["no_models"] is True
        orig_cmd, orig_duration = row1["cmd"], row1["duration_sec"]

        def _fail_if_called(*a, **kw):
            raise AssertionError("subprocess.run must not run when skip_existing finds the marker")
        monkeypatch.setattr(runner.subprocess, "run", _fail_if_called)
        monkeypatch.setattr(sys, "argv", argv_base + ["--device", "cuda:0", "--skip-existing"])
        runner.main()

        row2 = json.loads((output_root / "batch_status.json").read_text())[0]
        assert row2["status"] == "skipped"
        # Produced-by provenance must still reflect the ORIGINAL --no-models run:
        assert row2["no_models"] is True
        assert row2["device"] is None
        assert row2["cmd"] == orig_cmd
        assert row2["duration_sec"] == orig_duration
        # This invocation's (different) request is recorded separately:
        assert row2["requested_no_models"] is False
        assert row2["requested_device"] == "cuda:0"


# ──────────────────────────────────────────────────────────────────────────────
# Summarizers (synthetic output tree)
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def _tm_row(**over):
    row = {
        "n_frames": 10, "temporal_srs": 1.0, "ptc": 0.9, "sfr": 0.1, "sdi": 0.05,
        "n_keyframes": 2, "n_interframes": 8, "n_reused": 6, "n_generate": 0,
        "n_recompute_semantic": 1, "n_recompute_motion": 1,
        "transmitted_units": 12, "naive_units": 50, "overhead_reduction": 0.76,
    }
    row.update(over)
    return row


@pytest.fixture()
def synthetic_root(tmp_path):
    root = tmp_path / "etri_out"
    for video in ("01_toy", "05_camera_pan_person"):
        # baseline
        vdir = root / "baseline" / video
        _write_csv(vdir / "temporal_metrics.csv", _tm_row())
        (vdir / "segments.json").write_text(json.dumps(
            [{"segment_id": 0, "keyframe_index": 0}, {"segment_id": 1, "keyframe_index": 5}]))
        (vdir / "keyframes.json").write_text(json.dumps({"keyframes": [0, 5]}))
        # motion sweep: two thresholds
        for th, reused, motion in (("off", 8, 0), ("0.05", 4, 4)):
            _write_csv(root / "motion_sweep" / f"th_{th}" / video / "temporal_metrics.csv",
                       _tm_row(n_reused=reused, n_recompute_motion=motion,
                               n_recompute_semantic=0))
        # verifier
        report = [
            {"frame_index": 0, "severity": 0.0, "controller_decision": "accept",
             "missing_object_count": 0, "additional_object_count": 0,
             "relation_error_count": 0, "attribute_error_count": 0},
            {"frame_index": 1, "severity": 0.8, "controller_decision": "fallback_recompute",
             "missing_object_count": 1, "additional_object_count": 2,
             "relation_error_count": 1, "attribute_error_count": 0},
        ]
        (root / "verifier" / video).mkdir(parents=True)
        (root / "verifier" / video / "packet_match_report.json").write_text(json.dumps(report))
        # generate
        gdir = root / "generate" / video
        _write_csv(gdir / "temporal_metrics.csv", _tm_row(n_generate=3, n_reused=4))
        (gdir / "generated_frames").mkdir(parents=True)
        for i in range(3):
            (gdir / "generated_frames" / f"generated_{i:05d}.png").write_bytes(b"")
        (gdir / "segments.json").write_text(json.dumps(
            [{"segment_id": 0, "generation": {"n_generated": 3}}, {"segment_id": 1, "generation": None}]))
        # bidirectional
        bdir = root / "bidirectional" / video
        bdir.mkdir(parents=True)
        (bdir / "generation_mode_comparison.json").write_text(json.dumps({
            "start_only": {"ptc": 0.9, "sfr": 0.1, "sdi": 0.05, "n_generate": 3,
                           "n_reused": 4, "n_recompute_semantic": 1, "n_recompute_motion": 0},
            "bidirectional": {"ptc": 0.92, "sfr": 0.08, "sdi": 0.04, "n_generate": 3,
                              "n_reused": 4, "n_recompute_semantic": 1, "n_recompute_motion": 0},
            "comparison": {"ptc_start_only": 0.9, "ptc_bidirectional": 0.92,
                           "ptc_diff": 0.02, "sfr_diff": -0.02, "sdi_diff": -0.01,
                           "n_generate_diff": 0, "n_reused_diff": 0,
                           "n_recompute_semantic_diff": 0, "n_recompute_motion_diff": 0},
        }))
        # heldout
        hdir = root / "heldout" / video / "heldout"
        hdir.mkdir(parents=True)
        metrics = {"metrics": {"n_items": 10, "ptc": 0.9, "sfr": 0.1, "sdi": 0.05,
                               "mean_severity": 0.2}}
        (hdir / "clip_only_metrics.json").write_text(json.dumps(metrics))
        (hdir / "calibrated_metrics.json").write_text(json.dumps(metrics))
        (hdir / "metric_delta.json").write_text(json.dumps({"items": []}))
        # accounting
        adir = root / "accounting" / video / "accounting"
        adir.mkdir(parents=True)
        (adir / "accounting_summary.json").write_text(json.dumps({
            "n_frames": 10, "total_bits": 1000, "total_channel_symbols": 500,
            "total_semantic_units": 12, "baseline": "naive_full_frame_packet",
            "bit_reduction": 0.4, "symbol_reduction": 0.35,
            "semantic_unit_reduction": 0.76, "proxy_fraction": 0.6,
        }))
        (adir / "rate_reliability_summary.json").write_text(json.dumps({
            "bits_per_frame": 100.0, "symbols_per_frame": 50.0,
            "ptc": 0.9, "sfr": 0.1, "sdi": 0.05, "mean_severity": 0.2,
        }))
        # Per-run rate/reliability curve row (what the accounting stage
        # actually writes on disk — see build_run_config's curve_csv default;
        # matches pipelines/rate_reliability_report.py's _CURVE_FIELDS schema).
        _write_csv(adir / "rate_reliability_curve.csv", {
            "label": video, "bits_per_frame": 100.0, "symbols_per_frame": 50.0,
            "bit_reduction": 0.4, "symbol_reduction": 0.35,
            "semantic_unit_reduction": 0.76, "ptc": 0.9, "sfr": 0.1, "sdi": 0.05,
            "mean_severity": 0.2, "n_generate": 0, "n_reused": 6, "n_recompute": 2,
            "baseline": "naive_full_frame_packet", "proxy_fraction": 0.6,
        })
    return root


class TestSummarizers:
    def test_videos_summary(self, synthetic_root):
        rows = summarizer.summarize_videos(synthetic_root)
        assert len(rows) == 2
        r = rows[0]
        assert r["n_segments"] == 2 and r["n_keyframes"] == 2
        assert r["reuse_ratio"] == pytest.approx(6 / 8)
        # accounting values merged in:
        assert r["bit_reduction"] == 0.4

    def test_motion_sweep_summary_and_focus(self, synthetic_root):
        rows = summarizer.summarize_motion_sweep(synthetic_root)
        assert len(rows) == 4  # 2 videos × 2 thresholds
        by = {(r["video"], r["motion_threshold"]): r for r in rows}
        assert by[("05_camera_pan_person", "off")]["n_recompute_motion"] == 0
        assert by[("05_camera_pan_person", "0.05")]["n_recompute_motion"] == 4
        focus = summarizer.motion_sweep_focus_md(rows)
        assert "05_camera_pan_person" in focus

    def test_verifier_summary(self, synthetic_root):
        rows = summarizer.summarize_verifier(synthetic_root)
        r = rows[0]
        assert r["status"] == "ok"
        assert r["missing_objects_total"] == 1
        assert r["additional_objects_total"] == 2
        assert r["structural_errors_total"] == 1
        assert r["decision_accept"] == 1 and r["decision_fallback_recompute"] == 1
        assert r["mean_severity"] == pytest.approx(0.4)

    def test_verifier_missing_artifact_does_not_crash(self, tmp_path):
        """A video subdirectory exists under verifier/ (the stage was attempted
        for it) but packet_match_report.json is absent (failed/never written).
        This must surface as an explicit missing_artifact row — not a crash,
        not a silently-dropped video — while a sibling video with a real
        report still summarizes normally in the same call."""
        root = tmp_path / "out"
        (root / "verifier" / "01_ok").mkdir(parents=True)
        (root / "verifier" / "01_ok" / "packet_match_report.json").write_text(json.dumps([
            {"frame_index": 0, "severity": 0.1, "controller_decision": "accept",
             "missing_object_count": 0, "additional_object_count": 0,
             "relation_error_count": 0, "attribute_error_count": 0},
        ]))
        (root / "verifier" / "02_missing_report").mkdir(parents=True)
        # no packet_match_report.json written for 02_missing_report

        rows = summarizer.summarize_verifier(root)
        by_video = {r["video"]: r for r in rows}
        assert set(by_video) == {"01_ok", "02_missing_report"}

        ok_row = by_video["01_ok"]
        assert ok_row["status"] == "ok"
        assert ok_row["n_frames_verified"] == 1

        missing_row = by_video["02_missing_report"]
        assert missing_row["status"] == "missing_artifact"
        for field in summarizer._VERIFIER_MISSING_ROW_FIELDS:
            assert missing_row[field] is None

        # Every row must share the same field set (write_summary derives the
        # CSV header from rows[0]) so mixed ok/missing rows still write a
        # single well-formed CSV/JSON/MD.
        assert set(ok_row) == set(missing_row)
        summarizer.write_summary(rows, root / "summary" / "verifier_summary", "t")
        with open((root / "summary" / "verifier_summary.csv"), newline="", encoding="utf-8") as fh:
            csv_rows = list(csv.DictReader(fh))
        assert {r["video"] for r in csv_rows} == {"01_ok", "02_missing_report"}

    def test_generate_summary(self, synthetic_root):
        rows = summarizer.summarize_generate(synthetic_root)
        r = rows[0]
        assert r["n_generate"] == 3 and r["generated_frames_saved"] == 3
        assert r["saved_matches_decisions"] is True
        assert r["segments_with_generation"] == 1

    def test_bidirectional_summary(self, synthetic_root):
        rows = summarizer.summarize_bidirectional(synthetic_root)
        r = rows[0]
        assert r["start_ptc"] == 0.9 and r["bidi_ptc"] == 0.92
        assert r["diff_ptc"] == pytest.approx(0.02)

    def test_heldout_summary(self, synthetic_root):
        rows = summarizer.summarize_heldout(synthetic_root)
        r = rows[0]
        assert r["metric_delta_exists"] is True
        assert r["calibrated_equals_clip_only"] is True

    def test_accounting_summary(self, synthetic_root):
        rows = summarizer.summarize_accounting(synthetic_root)
        r = rows[0]
        assert r["bit_reduction"] == 0.4 and r["symbols_per_frame"] == 50.0

    def test_merge_accounting_curve(self, synthetic_root):
        """Per-video curve CSVs (never a shared file — see build_run_config)
        get combined post-hoc into one summary/rate_reliability_curve.csv."""
        n = summarizer.merge_accounting_curve(synthetic_root)
        assert n == 2  # one row per video in the fixture
        out = synthetic_root / "summary" / "rate_reliability_curve.csv"
        assert out.exists()
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert {r["label"] for r in rows} == {"01_toy", "05_camera_pan_person"}

    def test_merge_accounting_curve_no_accounting_stage(self, tmp_path):
        assert summarizer.merge_accounting_curve(tmp_path / "empty") == 0

    def test_summarize_all_writes_files(self, synthetic_root):
        written = summarizer.summarize_all(synthetic_root)
        assert set(written) == {
            "videos_summary", "motion_sweep_summary", "verifier_summary",
            "generate_summary", "bidirectional_summary", "heldout_summary",
            "accounting_summary_all", "rate_reliability_curve"}
        sdir = synthetic_root / "summary"
        for name in written:
            exts = (".csv",) if name == "rate_reliability_curve" else (".csv", ".json", ".md")
            for ext in exts:
                assert (sdir / f"{name}{ext}").exists(), f"{name}{ext}"

    def test_summarize_all_partial_tree(self, tmp_path):
        """Only-baseline tree → only videos_summary written, no crash."""
        root = tmp_path / "partial"
        _write_csv(root / "baseline" / "01_toy" / "temporal_metrics.csv", _tm_row())
        (root / "baseline" / "01_toy" / "segments.json").write_text("[]")
        written = summarizer.summarize_all(root)
        assert set(written) == {"videos_summary"}


class TestFinalReport:
    def test_build_report(self, synthetic_root):
        summarizer.summarize_all(synthetic_root)
        (synthetic_root / "batch_status.json").write_text(json.dumps([
            {"stage": "baseline", "video": "01_toy", "status": "ok",
             "out_dir": "x", "returncode": 0, "duration_sec": 1.0},
            {"stage": "verifier", "video": "01_toy", "status": "failed",
             "out_dir": "y", "returncode": 1, "duration_sec": 1.0},
        ]))
        report = reporter.build_report(synthetic_root)
        assert "코드/PoC로 완료된 항목" in report
        assert "외부 모델/GT/판정 데이터가 필요한 항목" in report
        assert "1~6차 ↔ 산출물 대응표" in report
        assert "OWLv2/VQA" in report
        assert "실패한 run" in report and "`y/run.log`" in report
        # Every stage summary section present:
        for _, title in reporter.SUMMARIES:
            assert title in report

    def test_batch_overview_flags_uncertain_provenance_separately(self, synthetic_root):
        summarizer.summarize_all(synthetic_root)
        overview = reporter._batch_overview([
            {"stage": "baseline", "video": "01_toy", "status": "ok", "out_dir": "a",
             "returncode": 0, "duration_sec": 1.0,
             "no_models": True, "snr": None, "device": None},
            {"stage": "baseline", "video": "02_toy", "status": "skipped", "out_dir": "b",
             "returncode": 1, "duration_sec": 3.1, "cmd": "…", "prior_status": "failed",
             "note": "…failed…", "no_models": True, "snr": None, "device": None},
            {"stage": "baseline", "video": "03_toy", "status": "skipped", "out_dir": "c",
             "returncode": None, "duration_sec": 0.0,
             "no_models": None, "snr": None, "device": None},
        ])
        assert "(+1 skipped, prior run failed — provenance uncertain)" in overview
        assert "(+1 skipped, provenance unknown)" in overview
        # The confident config set must come only from the "ok" row, not the
        # failed-prior skip (even though it also has no_models=True):
        assert "no_models=True,snr=None,device=None" in overview
        assert overview.count("no_models=True,snr=None,device=None") == 1
