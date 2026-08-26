import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.utils import run_manifest as rm


def test_sha256_file_matches_known_digest(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello world")
    assert rm.sha256_file(f) == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe"
        "37a5380ee9088f7ace2efcde9"
    )


def test_sha256_file_missing_is_unknown(tmp_path):
    assert rm.sha256_file(tmp_path / "does_not_exist.txt") == rm.UNKNOWN
    assert rm.sha256_file(None) == rm.UNKNOWN


def test_get_git_state_nonexistent_root_is_unknown(tmp_path):
    state = rm.get_git_state(tmp_path / "no_such_dir")
    assert state == {"commit": rm.UNKNOWN, "dirty": rm.UNKNOWN, "branch": rm.UNKNOWN}


def test_get_git_state_non_repo_dir_is_unknown(tmp_path):
    # tmp_path is a real directory but not a git repo.
    state = rm.get_git_state(tmp_path)
    assert state["commit"] == rm.UNKNOWN
    assert state["dirty"] == rm.UNKNOWN
    assert state["branch"] == rm.UNKNOWN


@pytest.mark.skipif(shutil.which("git") is None, reason="git binary unavailable")
def test_get_git_state_detects_clean_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    state = rm.get_git_state(tmp_path)
    assert state["commit"] != rm.UNKNOWN
    assert len(state["commit"]) == 40
    assert state["dirty"] is False

    (tmp_path / "f.txt").write_text("y")
    dirty_state = rm.get_git_state(tmp_path)
    assert dirty_state["dirty"] is True
    assert dirty_state["commit"] == state["commit"]


def test_get_python_env_always_populated():
    env = rm.get_python_env()
    assert env["python_version"] != rm.UNKNOWN
    assert env["platform"] != rm.UNKNOWN


def test_get_cuda_env_never_raises_and_has_expected_keys():
    env = rm.get_cuda_env()
    assert set(env.keys()) == {"torch_version", "cuda_available", "cuda_version", "gpu_name"}


def test_get_cuda_env_missing_torch_is_fully_unknown(monkeypatch):
    monkeypatch.setattr(rm, "_try_import_torch", lambda: None)
    env = rm.get_cuda_env()
    assert env == {
        "torch_version": rm.UNKNOWN,
        "cuda_available": rm.UNKNOWN,
        "cuda_version": rm.UNKNOWN,
        "gpu_name": rm.UNKNOWN,
    }


def test_get_cuda_env_broken_cuda_runtime_degrades_to_unknown(monkeypatch):
    """A torch install that imports fine but whose CUDA runtime is broken
    (e.g. mismatched driver) should surface torch_version but not crash or
    fabricate cuda_available/cuda_version/gpu_name."""

    class FakeCudaModule:
        @staticmethod
        def is_available():
            raise OSError("libcudart.so.11.0: cannot open shared object file")

    class FakeTorch:
        __version__ = "2.1.0+broken"
        cuda = FakeCudaModule()

    monkeypatch.setattr(rm, "_try_import_torch", lambda: FakeTorch())
    env = rm.get_cuda_env()
    assert env["torch_version"] == "2.1.0+broken"
    assert env["cuda_available"] == rm.UNKNOWN
    assert env["cuda_version"] == rm.UNKNOWN
    assert env["gpu_name"] == rm.UNKNOWN


def test_get_cuda_env_gpu_name_lookup_failure_is_unknown(monkeypatch):
    class FakeVersion:
        cuda = "11.8"

    class FakeCudaModule:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(_index):
            raise RuntimeError("CUDA error: invalid device ordinal")

    class FakeTorch:
        __version__ = "2.1.0"
        cuda = FakeCudaModule()
        version = FakeVersion()

    monkeypatch.setattr(rm, "_try_import_torch", lambda: FakeTorch())
    env = rm.get_cuda_env()
    assert env["torch_version"] == "2.1.0"
    assert env["cuda_available"] is True
    assert env["cuda_version"] == "11.8"
    assert env["gpu_name"] == rm.UNKNOWN


def test_try_import_torch_survives_broken_shared_library(monkeypatch):
    """Simulate a real import-time failure (OSError from a broken .so, not
    just ImportError from a missing package) using a poisoned meta_path
    finder, to prove the broad `except Exception` in `_try_import_torch`
    actually covers this case end-to-end, not just via a mocked seam."""

    class PoisonedFinder:
        def find_spec(self, name, path, target=None):
            if name == "torch":
                raise OSError("simulated broken shared library")
            return None

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    finder = PoisonedFinder()
    sys.meta_path.insert(0, finder)
    try:
        assert rm._try_import_torch() is None
        env = rm.get_cuda_env()
        assert env["torch_version"] == rm.UNKNOWN
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop("torch", None)


def test_build_run_manifest_defaults_to_unknown_not_guessed():
    manifest = rm.build_run_manifest(run_id="test_run", include_environment=False)
    assert manifest["run_id"] == "test_run"
    assert manifest["command"] == {
        "text": rm.UNKNOWN,
        "argv": rm.UNKNOWN,
        "source": rm.UNKNOWN,
    }
    assert manifest["seed"] == rm.UNKNOWN
    assert manifest["dataset"] == {"ref": rm.UNKNOWN, "hash": rm.UNKNOWN}
    assert manifest["checkpoints"] == {"status": rm.UNKNOWN, "items": {}}
    assert manifest["evaluator_versions"] == {}
    assert manifest["original_artifact_paths"] == {}
    assert manifest["artifacts"] == {}
    assert manifest["accounting"] == {"exact_fields": [], "proxy_fields": []}
    assert manifest["nan_or_failure_counts"] == {}
    assert manifest["environment"]["python_version"] == rm.UNKNOWN
    assert manifest["resolved_config"] == {
        "status": "unknown",
        "resolved": rm.UNKNOWN,
        "config_source": None,
    }


@pytest.mark.skipif(shutil.which("git") is None, reason="git binary unavailable")
def test_build_run_manifest_include_git_false_never_probes_repo(tmp_path):
    # tmp_path is a real git repo with real commits — if include_git=False
    # actually skipped the probe, we'd see "unknown" here despite that.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    manifest = rm.build_run_manifest(
        run_id="r", repo_root=tmp_path, include_git=False, include_environment=False,
    )
    assert manifest["git"] == {"commit": rm.UNKNOWN, "dirty": rm.UNKNOWN, "branch": rm.UNKNOWN}

    # Sanity: with include_git=True (default) the same repo_root DOES resolve.
    live = rm.build_run_manifest(run_id="r", repo_root=tmp_path, include_environment=False)
    assert live["git"]["commit"] != rm.UNKNOWN


def test_get_git_state_reads_head_without_git_binary(tmp_path, monkeypatch):
    commit = "a" * 40
    git_dir = tmp_path / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "refs" / "heads" / "main").write_text(commit + "\n")
    monkeypatch.setattr(rm, "_run_git", lambda *_args, **_kwargs: None)
    state = rm.get_git_state(tmp_path)
    assert state == {"commit": commit, "dirty": rm.UNKNOWN, "branch": "main"}


def test_get_git_state_accepts_verified_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SGDJSCC_GIT_COMMIT", "b" * 40)
    monkeypatch.setenv("SGDJSCC_GIT_DIRTY", "false")
    monkeypatch.setenv("SGDJSCC_GIT_BRANCH", "main")
    assert rm.get_git_state(tmp_path) == {
        "commit": "b" * 40, "dirty": False, "branch": "main",
    }


def test_build_run_manifest_seed_none_is_rejected():
    with pytest.raises(ValueError, match="seed"):
        rm.build_run_manifest(run_id="r", seed=None, include_environment=False)


def test_build_run_manifest_rejects_empty_run_id():
    with pytest.raises(ValueError, match="run_id"):
        rm.build_run_manifest(run_id="  ", include_environment=False)


def test_build_run_manifest_seed_not_set_vs_unknown_are_distinct():
    not_set = rm.build_run_manifest(run_id="r", seed=rm.NOT_SET, include_environment=False)
    unknown = rm.build_run_manifest(run_id="r", seed=rm.UNKNOWN, include_environment=False)
    assert not_set["seed"] == "not_set"
    assert unknown["seed"] == "unknown"
    assert not_set["seed"] != unknown["seed"]


def test_build_run_manifest_command_requires_valid_source():
    with pytest.raises(ValueError, match="command_source"):
        rm.build_run_manifest(
            run_id="r", command="python foo.py", command_source="captured_bogus",
            include_environment=False,
        )


def test_build_run_manifest_command_reconstructed_is_labeled_not_captured():
    manifest = rm.build_run_manifest(
        run_id="r",
        command="python scripts/run_transmission_reduction_eval.py --configs fixed_awgn",
        command_source="reconstructed",
        include_environment=False,
    )
    assert manifest["command"]["source"] == "reconstructed"
    assert manifest["command"]["argv"] == rm.UNKNOWN
    assert "reduction_eval" in manifest["command"]["text"]


def test_build_run_manifest_command_argv_is_lossless_and_shell_quoted():
    argv = ["python", "script.py", "--output", "path with spaces/result.json"]
    manifest = rm.build_run_manifest(
        run_id="r",
        command_argv=argv,
        command_source="captured",
        include_environment=False,
    )
    assert manifest["command"]["argv"] == argv
    assert manifest["command"]["text"] == "python script.py --output 'path with spaces/result.json'"


def test_build_run_manifest_rejects_command_source_without_command():
    with pytest.raises(ValueError, match="command_source"):
        rm.build_run_manifest(
            run_id="r", command_source="captured", include_environment=False,
        )


def test_build_run_manifest_rejects_multiple_command_forms():
    with pytest.raises(ValueError, match="at most one"):
        rm.build_run_manifest(
            run_id="r",
            command="python script.py",
            command_argv=["python", "script.py"],
            command_source="captured",
            include_environment=False,
        )


def test_build_run_manifest_resolved_config_path_status_is_resolved(tmp_path):
    cfg_path = tmp_path / "resolved.yaml"
    cfg_path.write_text("key: value\n")
    manifest = rm.build_run_manifest(
        run_id="r", resolved_config_path=cfg_path, include_environment=False,
    )
    record = manifest["resolved_config"]
    assert record["status"] == "resolved"
    assert record["resolved"]["path"] == str(cfg_path)
    assert record["resolved"]["sha256"] == rm.sha256_file(cfg_path)
    assert record["config_source"] is None


def test_build_run_manifest_rejects_missing_resolved_config_path(tmp_path):
    with pytest.raises(ValueError, match="resolved_config_path"):
        rm.build_run_manifest(
            run_id="r",
            resolved_config_path=tmp_path / "missing.yaml",
            include_environment=False,
        )


def test_build_run_manifest_config_source_only_does_not_claim_resolved(tmp_path):
    cfg_path = tmp_path / "composed.yaml"
    cfg_path.write_text("_defaults_: [a, b]\n")
    manifest = rm.build_run_manifest(
        run_id="r", config_source_path=cfg_path, include_environment=False,
    )
    record = manifest["resolved_config"]
    assert record["status"] == "config_source_only"
    assert record["resolved"] == rm.UNKNOWN
    assert record["config_source"]["path"] == str(cfg_path)
    assert record["config_source"]["sha256"] == rm.sha256_file(cfg_path)
    assert "note" in record["config_source"]


def test_build_run_manifest_rejects_missing_config_source_path(tmp_path):
    with pytest.raises(ValueError, match="config_source_path"):
        rm.build_run_manifest(
            run_id="r",
            config_source_path=tmp_path / "missing.yaml",
            include_environment=False,
        )


def test_build_run_manifest_rejects_multiple_config_kinds(tmp_path):
    cfg_path = tmp_path / "composed.yaml"
    cfg_path.write_text("k: v\n")
    with pytest.raises(ValueError, match="at most one"):
        rm.build_run_manifest(
            run_id="r",
            resolved_config_path=cfg_path,
            config_source_path=cfg_path,
            include_environment=False,
        )


def test_build_run_manifest_records_explicit_values(tmp_path):
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"fake-weights")
    cfg_path = tmp_path / "composed.yaml"
    cfg_path.write_text("key: value\n")

    manifest = rm.build_run_manifest(
        run_id="transmission_20260818",
        command="python scripts/run_transmission_reduction_eval.py --configs fixed_awgn",
        command_source="reconstructed",
        seed=rm.NOT_SET,
        resolved_config_path=cfg_path,
        dataset_ref="data/etri_video_eval",
        checkpoints={"jscc": ckpt},
        evaluator_versions={"psnr": "unknown"},
        original_artifact_paths={"output_root": "outputs/transmission_reduction_full_20260818_043425"},
        exact_fields=["packet_components.csv:total_bundle_bytes"],
        proxy_fields=["keyframe_selection.csv:estimated_wire_bytes"],
        nan_or_failure_counts={"total_nan_or_inf_frames": 0},
        include_environment=False,
    )
    assert manifest["seed"] == "not_set"
    assert manifest["checkpoints"]["status"] == "recorded"
    assert manifest["checkpoints"]["items"]["jscc"]["sha256"] == rm.sha256_file(ckpt)
    assert manifest["resolved_config"]["resolved"]["path"] == str(cfg_path)
    assert manifest["resolved_config"]["resolved"]["sha256"] == rm.sha256_file(cfg_path)
    assert manifest["accounting"]["exact_fields"] == ["packet_components.csv:total_bundle_bytes"]
    assert manifest["accounting"]["proxy_fields"] == ["keyframe_selection.csv:estimated_wire_bytes"]
    assert manifest["nan_or_failure_counts"] == {"total_nan_or_inf_frames": 0}


def test_build_run_manifest_empty_checkpoints_means_confirmed_not_set():
    manifest = rm.build_run_manifest(
        run_id="r", checkpoints={}, include_environment=False,
    )
    assert manifest["checkpoints"] == {"status": rm.NOT_SET, "items": {}}


def test_build_run_manifest_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="checkpoints"):
        rm.build_run_manifest(
            run_id="r",
            checkpoints={"jscc": tmp_path / "missing.pth"},
            include_environment=False,
        )


def test_hash_artifact_pair_matching_files(tmp_path):
    original = tmp_path / "outputs" / "aggregate.csv"
    original.parent.mkdir(parents=True)
    original.write_text("a,b\n1,2\n")
    copied = tmp_path / "results" / "aggregate.csv"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(original.read_bytes())

    record = rm.hash_artifact_pair(copied, original)
    assert record["matches"] is True
    assert record["copied_sha256"] == record["original_sha256"]


def test_hash_artifact_pair_mismatched_files(tmp_path):
    original = tmp_path / "orig.csv"
    original.write_text("a,b\n1,2\n")
    copied = tmp_path / "copy.csv"
    copied.write_text("a,b\n1,3\n")

    record = rm.hash_artifact_pair(copied, original)
    assert record["matches"] is False


def test_hash_artifact_pair_missing_original_is_unknown_match(tmp_path):
    copied = tmp_path / "copy.csv"
    copied.write_text("a,b\n1,2\n")
    record = rm.hash_artifact_pair(copied, tmp_path / "does_not_exist.csv")
    assert record["matches"] == rm.UNKNOWN
    assert record["original_sha256"] == rm.UNKNOWN


def test_hash_artifact_pair_no_original_path_given(tmp_path):
    copied = tmp_path / "copy.csv"
    copied.write_text("a,b\n1,2\n")
    record = rm.hash_artifact_pair(copied)
    assert record["original_path"] == rm.UNKNOWN
    assert record["matches"] == rm.UNKNOWN


def test_build_run_manifest_artifacts_field(tmp_path):
    original = tmp_path / "outputs" / "aggregate.csv"
    original.parent.mkdir(parents=True)
    original.write_text("a,b\n1,2\n")
    copied = tmp_path / "results" / "aggregate.csv"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(original.read_bytes())

    manifest = rm.build_run_manifest(
        run_id="r",
        artifacts={"aggregate.csv": {"copied_path": copied, "original_path": original}},
        include_environment=False,
    )
    record = manifest["artifacts"]["aggregate.csv"]
    assert record["matches"] is True
    assert record["copied_path"] == str(copied)
    assert record["original_path"] == str(original)


def test_write_run_manifest_round_trips(tmp_path):
    manifest = rm.build_run_manifest(run_id="roundtrip", include_environment=False)
    out_path = tmp_path / "nested" / "manifest.json"
    written = rm.write_run_manifest(out_path, manifest)
    assert written == out_path
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "roundtrip"
    assert loaded["schema_version"] == rm.SCHEMA_VERSION
