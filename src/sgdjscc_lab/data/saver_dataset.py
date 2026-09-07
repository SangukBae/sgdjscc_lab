"""Provenance-checked tensor-sequence dataset for staged SAVER training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

from sgdjscc_lab.pipelines.saver_video_pipeline import SaverVideoPipeline


SAVER_TENSOR_DATASET_SCHEMA = 1

_REQUIRED_TENSORS = {
    "source_features": 3,               # [S,L,d_source]
    "visual_tokens": 3,                 # [S,L_visual,d_m]
    "diffusion_hidden": 3,              # [S,L_diff,d_hidden]
    "diffusion_target": 3,
    "timestep_embedding": 2,            # [S,d_t]
    "channel_features": 2,              # [S,d_channel]
    "channel_tokens": 3,                # [S,T,d_channel_token]
    "entity_ids": 2,                    # [S,K]
    "semantic_states": 2,
    "versions": 2,
    "valid_mask": 2,
    "teacher_actions": 2,
    "teacher_allocated_symbols": 2,
    "event_target_features": 3,         # [S,K,d_m]
}

_FORBIDDEN_KEYS = {
    "receiver_reconstruction",
    "future_features",
    "final_evaluator_scores",
    "heldout_labels",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SaverTensorManifestDataset(Dataset):
    """Load fixed-shape GOP sequences declared by a checksummed JSONL manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        allowed_splits: Sequence[str] = ("development",),
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.root = self.manifest_path.parent
        self.allowed_splits = frozenset(str(x) for x in allowed_splits)
        rows = []
        seen = set()
        for line_number, line in enumerate(self.manifest_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at manifest line {line_number}") from exc
            self._validate_row(row, line_number)
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id={sample_id!r}")
            seen.add(sample_id)
            rows.append(row)
        if not rows:
            raise ValueError("SAVER tensor manifest is empty")
        self.rows = rows
        self.manifest_sha256 = _sha256(self.manifest_path)

    def _resolve_tensor_path(self, raw: str) -> Path:
        path = (self.root / raw).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"tensor_path escapes manifest root: {raw!r}") from exc
        return path

    def _validate_row(self, row: Mapping, line_number: int) -> None:
        required = {"schema_version", "sample_id", "tensor_path", "sha256", "split", "provenance"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"manifest line {line_number} missing {sorted(missing)}")
        if int(row["schema_version"]) != SAVER_TENSOR_DATASET_SCHEMA:
            raise ValueError(f"unsupported tensor dataset schema at line {line_number}")
        if row["split"] not in self.allowed_splits:
            raise ValueError(
                f"split={row['split']!r} is not allowed; allowed={sorted(self.allowed_splits)}"
            )
        provenance = row["provenance"]
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be an object")
        if provenance.get("source_only") is not True:
            raise ValueError("SAVER training samples must declare provenance.source_only=true")
        if provenance.get("contains_receiver_reconstruction") is not False:
            raise ValueError("receiver reconstructions are prohibited in SAVER source targets")
        if provenance.get("contains_future_context") is not False:
            raise ValueError("future context is prohibited in online SAVER targets")
        checksum = str(row["sha256"])
        if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum.lower()):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        path = self._resolve_tensor_path(str(row["tensor_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict:
        row = self.rows[index]
        path = self._resolve_tensor_path(str(row["tensor_path"]))
        actual = _sha256(path)
        if actual != str(row["sha256"]).lower():
            raise RuntimeError(
                f"tensor checksum mismatch for sample={row['sample_id']}: "
                f"declared={row['sha256']} actual={actual}"
            )
        value = torch.load(path, map_location="cpu")
        if not isinstance(value, Mapping):
            raise ValueError(f"sample={row['sample_id']} must contain a tensor mapping")
        forbidden = _FORBIDDEN_KEYS & set(value)
        if forbidden:
            raise ValueError(f"sample={row['sample_id']} contains forbidden keys {sorted(forbidden)}")
        missing = set(_REQUIRED_TENSORS) - set(value)
        if missing:
            raise ValueError(f"sample={row['sample_id']} missing tensors {sorted(missing)}")
        sequence_length = None
        tensors: Dict[str, torch.Tensor] = {}
        for name, rank in _REQUIRED_TENSORS.items():
            tensor = value[name]
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != rank:
                raise ValueError(f"{name} must be a rank-{rank} tensor")
            if sequence_length is None:
                sequence_length = tensor.shape[0]
            if tensor.shape[0] != sequence_length:
                raise ValueError("all sample tensors must share sequence dimension S")
            tensors[name] = tensor
        slots = tensors["entity_ids"].shape[1]
        for name in ("semantic_states", "versions", "valid_mask", "teacher_actions", "teacher_allocated_symbols"):
            if tuple(tensors[name].shape) != (sequence_length, slots):
                raise ValueError(f"{name} must match entity_ids [S,K]")
        if tensors["event_target_features"].shape[:2] != (sequence_length, slots):
            raise ValueError("event_target_features must match entity_ids [S,K]")
        for optional in ("memory_positive", "memory_negative", "entity_presence_logits", "scene_epochs", "scene_reset_mask"):
            if optional in value:
                tensor = value[optional]
                if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != sequence_length:
                    raise ValueError(f"optional tensor {optional} must start with sequence dimension S")
                tensors[optional] = tensor
        return {
            "sample_id": str(row["sample_id"]),
            "split": str(row["split"]),
            "provenance": dict(row["provenance"]),
            "tensors": tensors,
        }


def collate_saver_tensor_sequences(samples: Sequence[Mapping]) -> Dict:
    if not samples:
        raise ValueError("cannot collate an empty SAVER batch")
    names = set(samples[0]["tensors"])
    if any(set(sample["tensors"]) != names for sample in samples):
        raise ValueError("all SAVER samples in a batch must have identical tensor keys")
    tensors = {}
    for name in sorted(names):
        shapes = {tuple(sample["tensors"][name].shape) for sample in samples}
        if len(shapes) != 1:
            raise ValueError(f"fixed-shape collator found inconsistent {name} shapes: {sorted(shapes)}")
        tensors[name] = torch.stack([sample["tensors"][name] for sample in samples], dim=0)
    return {
        "sample_ids": [sample["sample_id"] for sample in samples],
        "splits": [sample["split"] for sample in samples],
        "tensors": tensors,
    }


def make_saver_sequence_batches(
    collated: Mapping,
    pipeline: SaverVideoPipeline,
    *,
    device: torch.device | str,
    total_budget: int,
    snr_db: float = 10.0,
    channel: str = "awgn",
    packet_loss_probability: float = 0.0,
) -> List[Dict]:
    """Convert ``[B,S,...]`` tensors into state-carrying timestep batches."""

    values = {name: value.to(device) for name, value in collated["tensors"].items()}
    batch, sequence_length = values["entity_ids"].shape[:2]
    memory_state = pipeline.memory.initial_state(
        batch, device=device, dtype=values["source_features"].dtype
    )
    batches: List[Dict] = []
    for step in range(sequence_length):
        scene_epochs = values.get("scene_epochs")
        scene_reset = values.get("scene_reset_mask")
        pipeline_inputs = {
            "source_features": values["source_features"][:, step],
            "visual_tokens": values["visual_tokens"][:, step],
            "diffusion_hidden": values["diffusion_hidden"][:, step],
            "timestep_embedding": values["timestep_embedding"][:, step],
            "channel_features": values["channel_features"][:, step],
            "channel_tokens": values["channel_tokens"][:, step],
            "memory_state": memory_state,
            "entity_ids": values["entity_ids"][:, step],
            "semantic_states": values["semantic_states"][:, step],
            "versions": values["versions"][:, step],
            "scene_epochs": (
                scene_epochs[:, step].long()
                if scene_epochs is not None
                else torch.zeros(batch, device=device, dtype=torch.long)
            ),
            "total_budget": total_budget,
            "valid_mask": values["valid_mask"][:, step].bool(),
            "scene_reset_mask": (
                scene_reset[:, step].bool() if scene_reset is not None else None
            ),
            "teacher_actions": values["teacher_actions"][:, step].long(),
            "teacher_allocated_symbols": values["teacher_allocated_symbols"][:, step].long(),
            "snr_db": snr_db,
            "channel": channel,
            "packet_loss_probability": packet_loss_probability,
        }
        targets = {
            "diffusion_target": values["diffusion_target"][:, step],
            "event_target_features": values["event_target_features"][:, step],
        }
        for name in ("memory_positive", "memory_negative", "entity_presence_logits"):
            if name in values:
                targets[name] = values[name][:, step]
        batches.append({
            "pipeline_inputs": pipeline_inputs,
            "targets": targets,
            "sample_ids": list(collated["sample_ids"]),
            "sequence_step": step,
        })
    return batches

