from pathlib import Path

from omegaconf import OmegaConf

from sgdjscc_lab import paths


def test_workspace_roots_fall_back_to_repo(monkeypatch):
    for name in (
        "SGDJSCC_DATA_ROOT", "SGDJSCC_MODEL_ROOT", "SGDJSCC_RUN_ROOT",
        "SGDJSCC_CACHE_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    root = paths.lab_repo_root()
    assert paths.data_root() == root / "data"
    assert paths.model_root() == root / "checkpoints"
    assert paths.run_root() == root / "outputs"


def test_workspace_roots_use_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SGDJSCC_DATA_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setenv("SGDJSCC_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("SGDJSCC_RUN_ROOT", str(tmp_path / "runs"))
    assert paths.data_root() == (tmp_path / "datasets").resolve()
    assert paths.model_root() == (tmp_path / "models" / "checkpoints").resolve()
    assert paths.checkpoints_custom_root() == (tmp_path / "models" / "checkpoints_custom").resolve()
    assert paths.remote_weights_root() == (tmp_path / "models" / "remote_weights").resolve()
    assert paths.run_root() == (tmp_path / "runs").resolve()


def test_omegaconf_resolver(monkeypatch, tmp_path):
    monkeypatch.setenv("SGDJSCC_DATA_ROOT", str(tmp_path / "datasets"))
    paths.register_omegaconf_resolver()
    cfg = OmegaConf.create({"path": "${sgdjscc:data}/coco"})
    assert Path(cfg.path) == (tmp_path / "datasets" / "coco").resolve()


def test_cache_environment_does_not_override_explicit_values(monkeypatch, tmp_path):
    monkeypatch.setenv("SGDJSCC_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("TORCH_HOME", "/explicit/torch")
    paths.configure_external_cache_env()
    assert Path(__import__("os").environ["HF_HOME"]) == tmp_path / "cache" / "huggingface"
    assert __import__("os").environ["TORCH_HOME"] == "/explicit/torch"
