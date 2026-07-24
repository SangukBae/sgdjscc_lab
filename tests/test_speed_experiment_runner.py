"""tests/test_speed_experiment_runner.py – scripts/run_speed_experiment.py tests.

Covers the 4-mode speed/quality comparison driver: per-run result collection
from synthetic profiling_summary.json / temporal_metrics.csv files, markdown
report generation, and the results.json merge-across-concurrent-invocations
logic (this driver is meant to be launched once per mode per GPU against the
SAME --output-root — see this task's remote verification, where two modes ran
in parallel on cuda:0/cuda:1 and a naive overwrite would have raced).

No GPU, no checkpoints, no subprocess — run_mode_video is monkeypatched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
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


runner = _load("run_speed_experiment.py")


class TestCollect:
    def test_collect_reads_profiling_and_temporal_csv(self, tmp_path):
        out_dir = tmp_path / "real_all_frames_step10" / "01_person_walk"
        out_dir.mkdir(parents=True)
        (out_dir / "profiling_summary.json").write_text(json.dumps({
            "frames_done": 100, "avg_frame_sec": 1.7,
            "counters": {"diffusion_calls": 72, "blip2_calls": 9, "clip_image_calls": 40, "clip_text_calls": 40},
        }), encoding="utf-8")
        (out_dir / "temporal_metrics.csv").write_text(
            "n_frames,n_keyframes,n_reused,n_recompute_semantic,n_recompute_motion,ptc,sfr,sdi\n"
            "100,9,88,3,0,0.87,0.02,0.01\n",
            encoding="utf-8",
        )
        r = runner._collect("real_all_frames_step10", "01_person_walk", out_dir,
                            returncode=0, duration_sec=253.9)
        assert r["status"] == "ok"
        assert r["frames_done"] == 100
        assert r["avg_frame_sec"] == 1.7
        assert r["counters"]["diffusion_calls"] == 72
        assert r["n_keyframes"] == "9"
        assert r["ptc"] == "0.87"

    def test_collect_handles_missing_files(self, tmp_path):
        out_dir = tmp_path / "mode" / "video"
        out_dir.mkdir(parents=True)
        r = runner._collect("mode", "video", out_dir, returncode=1, duration_sec=5.0)
        assert r["status"] == "failed"
        assert "frames_done" not in r


class TestRunModeVideoDeviceRemap:
    """run_mode_video must apply the same CUDA_VISIBLE_DEVICES remap the batch
    driver uses (see run_etri_video_eval.remap_device_for_cuda_visible) — this
    tool routinely launches different modes pinned to different GPUs (exactly
    how this task's own remote verification ran real_keyframe_only_step10 on
    cuda:0 and real_all_frames_step10 on cuda:1 simultaneously), and without
    the remap any non-cuda:0 --device crashes inside SGDJSCC's hardcoded
    .cuda() call."""

    def _entry(self, tmp_path):
        video = tmp_path / "01_toy.mp4"
        video.write_bytes(b"fake")
        return {"key": "01_toy", "processed": video, "captions": None}

    def test_cuda_n_device_gets_remapped_command_and_env(self, tmp_path, monkeypatch):
        seen = {}

        def fake_run(cmd, stdout, stderr, cwd, env=None):
            seen["cmd"] = cmd
            seen["env"] = env
            class _P:
                returncode = 0
            return _P()

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        entry = self._entry(tmp_path)
        r = runner.run_mode_video(
            "real_all_frames_step10", runner.MODES["real_all_frames_step10"], entry,
            tmp_path / "out", device="cuda:1", snr=None, max_frames=None,
            no_clip=False, recon_caption_mode=None, skip_existing=False,
        )
        assert r["status"] == "ok"
        idx = seen["cmd"].index("--device")
        assert seen["cmd"][idx + 1] == "cuda:0"
        assert seen["env"] is not None
        assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "1"

    def test_cpu_device_env_stays_none(self, tmp_path, monkeypatch):
        seen = {}

        def fake_run(cmd, stdout, stderr, cwd):
            seen["called_without_env_kwarg"] = True
            class _P:
                returncode = 0
            return _P()

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        entry = self._entry(tmp_path)
        runner.run_mode_video(
            "no_models_captions", runner.MODES["no_models_captions"], entry,
            tmp_path / "out", device="cpu", snr=None, max_frames=None,
            no_clip=False, recon_caption_mode=None, skip_existing=False,
        )
        assert seen.get("called_without_env_kwarg") is True


class TestMergeWriteResultsLocked:
    def test_merges_new_rows_with_existing(self, tmp_path):
        results_path = tmp_path / "results.json"
        first = runner._merge_write_results_locked(
            results_path, [{"mode": "a", "video": "v1", "duration_sec": 1.0}])
        assert first == [{"mode": "a", "video": "v1", "duration_sec": 1.0}]

        second = runner._merge_write_results_locked(
            results_path, [{"mode": "b", "video": "v1", "duration_sec": 2.0}])
        modes = {(r["mode"], r["video"]) for r in second}
        assert modes == {("a", "v1"), ("b", "v1")}
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        assert on_disk == second

    def test_replaces_same_mode_video_not_duplicates(self, tmp_path):
        results_path = tmp_path / "results.json"
        runner._merge_write_results_locked(
            results_path, [{"mode": "a", "video": "v1", "duration_sec": 1.0}])
        merged = runner._merge_write_results_locked(
            results_path, [{"mode": "a", "video": "v1", "duration_sec": 9.0}])
        assert len(merged) == 1
        assert merged[0]["duration_sec"] == 9.0

    def test_tolerates_corrupt_existing_file(self, tmp_path):
        results_path = tmp_path / "results.json"
        results_path.write_text("not valid json{{{", encoding="utf-8")
        merged = runner._merge_write_results_locked(
            results_path, [{"mode": "a", "video": "v1", "duration_sec": 1.0}])
        assert merged == [{"mode": "a", "video": "v1", "duration_sec": 1.0}]

    def test_concurrent_threads_do_not_lose_updates(self, tmp_path):
        """Real concurrency test (not just sequential-call simulation): N
        threads each write a distinct (mode, video) row to the SAME
        results.json at the same time. Before the flock-based fix, a plain
        read-then-write race could let a later writer's read (taken before an
        earlier writer's write completed) clobber that earlier writer's row
        when it, in turn, writes back its own merged snapshot. With the lock,
        every row must survive regardless of interleaving."""
        results_path = tmp_path / "results.json"
        n = 12
        barrier = threading.Barrier(n)

        def _write(i):
            barrier.wait()  # maximise actual overlap of the read-merge-write sections
            runner._merge_write_results_locked(
                results_path, [{"mode": f"mode{i}", "video": "v1", "duration_sec": float(i)}])

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = json.loads(results_path.read_text(encoding="utf-8"))
        modes_present = {r["mode"] for r in final}
        assert modes_present == {f"mode{i}" for i in range(n)}
        assert len(final) == n  # no row lost, no duplicate


class TestWriteReport:
    def test_report_has_table_and_truncation_notice(self, tmp_path):
        results = [
            {"mode": "no_models_captions", "video": "01_person_walk", "status": "ok",
             "duration_sec": 2.5, "frames_done": 5, "avg_frame_sec": 0.005, "counters": {}},
            {"mode": "real_all_frames_step10", "video": "01_person_walk", "status": "ok",
             "duration_sec": 253.9, "frames_done": 100, "avg_frame_sec": 1.7,
             "counters": {"diffusion_calls": 72, "blip2_calls": 9, "clip_image_calls": 40, "clip_text_calls": 40},
             "n_keyframes": "9", "n_reused": "88", "ptc": "0.87", "sfr": "0.02", "sdi": "0.01"},
        ]
        path = tmp_path / "report.md"
        runner.write_report(results, path, truncated=True, modes=["no_models_captions", "real_all_frames_step10"])
        text = path.read_text(encoding="utf-8")
        assert "THROUGHPUT ESTIMATE" in text
        assert "no_models_captions" in text and "real_all_frames_step10" in text
        assert "| mode | video | status" in text


class TestMainResultsMerge:
    def _entry_fixture(self, tmp_path):
        data_root = tmp_path / "data"
        (data_root / "processed").mkdir(parents=True)
        (data_root / "captions").mkdir(parents=True)
        video = data_root / "processed" / "01_person_walk.mp4"
        video.write_bytes(b"fake")
        (data_root / "captions" / "01_person_walk.txt").write_text("a person walking\n" * 5, encoding="utf-8")
        (data_root / "manifest.csv").write_text(
            "id,name,raw_file,processed_file,frames_dir,width,height,fps,duration_sec,n_frames,primary_objects,event\n"
            "01,person_walk,raw/01_person_walk.mp4,processed/01_person_walk.mp4,frames/01_person_walk,"
            "512,256,10,10,100,person,one person walking\n",
            encoding="utf-8",
        )
        return data_root

    def test_two_invocations_merge_instead_of_clobbering(self, tmp_path, monkeypatch):
        data_root = self._entry_fixture(tmp_path)
        output_root = tmp_path / "out"

        def _fake_run_mode_video(mode, mode_cfg, entry, out_root, **kw):
            out_dir = Path(out_root) / mode / entry["key"]
            out_dir.mkdir(parents=True, exist_ok=True)
            return {"mode": mode, "video": entry["key"], "status": "ok",
                   "duration_sec": 1.0, "out_dir": str(out_dir)}

        monkeypatch.setattr(runner, "run_mode_video", _fake_run_mode_video)

        argv_a = ["run_speed_experiment.py", "--data-root", str(data_root),
                 "--output-root", str(output_root), "--videos", "01_person_walk",
                 "--modes", "real_keyframe_only_step10", "--device", "cuda:0"]
        monkeypatch.setattr(sys, "argv", argv_a)
        runner.main()

        results_after_a = json.loads((output_root / "results.json").read_text())
        assert len(results_after_a) == 1
        assert results_after_a[0]["mode"] == "real_keyframe_only_step10"

        argv_b = ["run_speed_experiment.py", "--data-root", str(data_root),
                 "--output-root", str(output_root), "--videos", "01_person_walk",
                 "--modes", "real_all_frames_step10", "--device", "cuda:1"]
        monkeypatch.setattr(sys, "argv", argv_b)
        runner.main()

        results_after_b = json.loads((output_root / "results.json").read_text())
        modes_present = {r["mode"] for r in results_after_b}
        # The second invocation's write must NOT have clobbered the first's
        # result — this is exactly the race hit during this task's remote
        # verification when two modes ran in parallel against one --output-root.
        assert modes_present == {"real_keyframe_only_step10", "real_all_frames_step10"}
        assert len(results_after_b) == 2

        report_text = (output_root / "speed_experiment_report.md").read_text(encoding="utf-8")
        assert "real_keyframe_only_step10" in report_text
        assert "real_all_frames_step10" in report_text

    def test_rerunning_same_mode_video_replaces_not_duplicates(self, tmp_path, monkeypatch):
        data_root = self._entry_fixture(tmp_path)
        output_root = tmp_path / "out"
        calls = {"n": 0}

        def _fake_run_mode_video(mode, mode_cfg, entry, out_root, **kw):
            calls["n"] += 1
            return {"mode": mode, "video": entry["key"], "status": "ok", "duration_sec": float(calls["n"])}

        monkeypatch.setattr(runner, "run_mode_video", _fake_run_mode_video)
        argv = ["run_speed_experiment.py", "--data-root", str(data_root),
               "--output-root", str(output_root), "--videos", "01_person_walk",
               "--modes", "no_models_captions"]
        monkeypatch.setattr(sys, "argv", argv)
        runner.main()
        runner.main()

        results = json.loads((output_root / "results.json").read_text())
        assert len(results) == 1
        assert results[0]["duration_sec"] == 2.0  # the SECOND run's value, not stale
