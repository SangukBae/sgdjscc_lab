from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from sgdjscc_lab.data.saver_dataset import (
    SaverTensorManifestDataset,
    collate_saver_tensor_sequences,
    make_saver_sequence_batches,
)
from sgdjscc_lab.models.saver.factory import build_saver_video_pipeline


ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(sequence=2):
    slots = 3
    entity_ids = torch.tensor([[1, -1, -1], [1, -1, -1]])[:sequence]
    versions = torch.tensor([[0, -1, -1], [1, -1, -1]])[:sequence]
    actions = torch.tensor([[1, 0, 0], [2, 0, 0]])[:sequence]
    return {
        "source_features": torch.randn(sequence, 5, 6),
        "visual_tokens": torch.randn(sequence, 4, 8),
        "diffusion_hidden": torch.randn(sequence, 7, 12),
        "diffusion_target": torch.randn(sequence, 7, 12),
        "timestep_embedding": torch.randn(sequence, 12),
        "channel_features": torch.randn(sequence, 3),
        "channel_tokens": torch.randn(sequence, 4, 4),
        "entity_ids": entity_ids,
        "semantic_states": torch.tensor([[0, 2, 2], [0, 2, 2]])[:sequence],
        "versions": versions,
        "valid_mask": torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.bool)[:sequence],
        "teacher_actions": actions,
        "teacher_allocated_symbols": torch.tensor([[4, 0, 0], [4, 0, 0]])[:sequence],
        "event_target_features": torch.randn(sequence, slots, 8),
        "scene_epochs": torch.zeros(sequence, dtype=torch.long),
        "scene_reset_mask": torch.zeros(sequence, dtype=torch.bool),
    }


def _write_manifest(tmp_path, *, provenance=None, mutate=None):
    tensor_path = tmp_path / "sample.pt"
    value = _sample()
    if mutate:
        mutate(value)
    torch.save(value, tensor_path)
    row = {
        "schema_version": 1,
        "sample_id": "clip-1",
        "tensor_path": tensor_path.name,
        "sha256": _sha(tensor_path),
        "split": "development",
        "provenance": provenance or {
            "source_only": True,
            "contains_receiver_reconstruction": False,
            "contains_future_context": False,
            "label_source": "official_track_annotation",
        },
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    return manifest


def _config():
    return {
        "use_saver_jscc": True,
        "saver": {
            "architecture_version": "saver-jscc-v1",
            "packet_schema_version": 1,
            "source_feature_dim": 6,
            "slot_count": 3,
            "memory_dim": 8,
            "num_heads": 2,
            "router": {
                "channel_feature_dim": 3,
                "hidden_dim": 12,
                "rate_options": [0, 2, 4],
                "visual_rate_options": [0, 4, 8],
                "protection_extra_symbols": [0, 2, 1, 4],
            },
            "codec": {
                "max_symbols": 4,
                "hidden_dim": 16,
                "wireless_profile": {"snr_db": 10.0, "channel": "awgn"},
            },
            "dit": {
                "hidden_dim": 12,
                "timestep_dim": 12,
                "channel_token_dim": 4,
                "bottleneck_dim": 8,
                "injection_layers": [1, 3],
            },
            "training": {"stage": "sv1", "learning_rate": 1e-4},
        },
    }


def test_tensor_manifest_checks_hash_provenance_and_builds_sequence(tmp_path):
    manifest = _write_manifest(tmp_path)
    dataset = SaverTensorManifestDataset(manifest)
    sample = dataset[0]
    assert sample["tensors"]["source_features"].shape[:2] == (2, 5)
    collated = collate_saver_tensor_sequences([sample])

    from omegaconf import OmegaConf
    pipeline = build_saver_video_pipeline(OmegaConf.create(_config()))
    batches = make_saver_sequence_batches(
        collated, pipeline, device="cpu", total_budget=16
    )
    assert len(batches) == 2
    assert batches[0]["pipeline_inputs"]["versions"].tolist() == [[0, -1, -1]]
    assert batches[1]["pipeline_inputs"]["versions"].tolist() == [[1, -1, -1]]

    tensor_path = tmp_path / "sample.pt"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        dataset[0]


def test_tensor_manifest_rejects_receiver_or_future_leakage(tmp_path):
    bad = {
        "source_only": True,
        "contains_receiver_reconstruction": True,
        "contains_future_context": False,
    }
    with pytest.raises(ValueError, match="receiver reconstructions"):
        SaverTensorManifestDataset(_write_manifest(tmp_path, provenance=bad))


def test_tensor_sample_rejects_forbidden_payload_keys(tmp_path):
    manifest = _write_manifest(
        tmp_path, mutate=lambda value: value.update({"future_features": torch.ones(1)})
    )
    with pytest.raises(ValueError, match="forbidden keys"):
        SaverTensorManifestDataset(manifest)[0]


def test_training_cli_runs_cpu_sequence_and_writes_versioned_checkpoint(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path)
    config_path = tmp_path / "config.yaml"
    from omegaconf import OmegaConf
    OmegaConf.save(OmegaConf.create(_config()), config_path)
    output = tmp_path / "run"

    spec = importlib.util.spec_from_file_location("train_saver_jscc", ROOT / "scripts/train_saver_jscc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_git_state", lambda: {"commit": "test", "dirty": False})
    assert module.run([
        "--config", str(config_path),
        "--manifest", str(manifest),
        "--output-dir", str(output),
        "--device", "cpu",
        "--epochs", "1",
        "--batch-size", "1",
        "--total-budget", "16",
    ]) == 0
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "COMPLETED_NOT_VALIDATED"
    assert status["global_step"] == 1
    checkpoint = torch.load(output / "checkpoints/epoch_0000.pt", map_location="cpu")
    assert checkpoint["kind"] == "saver_jscc_checkpoint"
    assert checkpoint["contract"]["slot_count"] == 3

