#!/usr/bin/env python3
"""Train an opt-in SAVER stage from a checksummed source-only tensor manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sgdjscc_lab.data.saver_dataset import (  # noqa: E402
    SaverTensorManifestDataset,
    collate_saver_tensor_sequences,
    make_saver_sequence_batches,
)
from sgdjscc_lab.models.saver.factory import (  # noqa: E402
    build_saver_video_pipeline,
    checkpoint_contract_from_config,
)
from sgdjscc_lab.training.saver_losses import SaverLoss, SaverLossWeights  # noqa: E402
from sgdjscc_lab.training.saver_stage_runner import SAVER_STAGES, SaverStageRunner  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip())
    return {"commit": commit, "dirty": dirty}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _weights(cfg) -> SaverLossWeights:
    values = OmegaConf.select(cfg, "saver.training.loss_weights", default={})
    values = OmegaConf.to_container(values, resolve=True) if values else {}
    return SaverLossWeights(**values)


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/saver_jscc/model_v1.yaml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=SAVER_STAGES)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--total-budget", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--allow-validation", action="store_true")
    parser.add_argument("--allow-dirty-development", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.total_budget < 0:
        raise SystemExit("epochs/batch-size must be positive and total-budget non-negative")

    git = _git_state()
    if git["dirty"] and not args.allow_dirty_development:
        raise SystemExit("tracked-dirty checkout; commit changes or use --allow-dirty-development")
    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    cfg = OmegaConf.load(config_path)
    if not bool(OmegaConf.select(cfg, "use_saver_jscc", default=False)):
        raise SystemExit("training config must explicitly set use_saver_jscc=true")
    stage = args.stage or str(OmegaConf.select(cfg, "saver.training.stage", default="sv1"))
    allowed_splits = ("development", "validation") if args.allow_validation else ("development",)
    dataset = SaverTensorManifestDataset(manifest_path, allowed_splits=allowed_splits)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_saver_tensor_sequences,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    pipeline = build_saver_video_pipeline(cfg).to(device)
    contract = checkpoint_contract_from_config(cfg)
    runner = SaverStageRunner(
        pipeline,
        SaverLoss(_weights(cfg)),
        contract,
        stage=stage,
        learning_rate=float(OmegaConf.select(cfg, "saver.training.learning_rate", default=1e-4)),
        weight_decay=float(OmegaConf.select(cfg, "saver.training.weight_decay", default=1e-5)),
    )
    if args.resume is not None:
        runner.load_checkpoint(args.resume.resolve())

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_spec = {
        "schema_version": 1,
        "kind": "saver_training",
        "stage": stage,
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "total_budget": args.total_budget,
        "git": git,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": dataset.manifest_sha256,
        "allowed_splits": list(allowed_splits),
        "checkpoint_contract": contract.to_dict(),
    }
    spec_path = output / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != run_spec:
        raise SystemExit("run_spec mismatch; choose a new output directory")
    _atomic_json(spec_path, run_spec)

    metrics_path = output / "metrics.jsonl"
    for epoch in range(args.epochs):
        for batch_index, collated in enumerate(loader):
            sequence = make_saver_sequence_batches(
                collated,
                pipeline,
                device=device,
                total_budget=args.total_budget,
                snr_db=float(OmegaConf.select(cfg, "saver.codec.wireless_profile.snr_db", default=10.0)),
                channel=str(OmegaConf.select(cfg, "saver.codec.wireless_profile.channel", default="awgn")),
            )
            metrics = runner.training_sequence_step(sequence)
            record = {
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": runner.global_step,
                "sample_ids": collated["sample_ids"],
                **metrics,
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        runner.save_checkpoint(
            output / "checkpoints" / f"epoch_{epoch:04d}.pt",
            {"epoch": epoch, "seed": args.seed, "manifest_sha256": dataset.manifest_sha256},
        )
    _atomic_json(output / "status.json", {
        "status": "COMPLETED_NOT_VALIDATED",
        "global_step": runner.global_step,
        "stage": stage,
        "heldout_accessed": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

