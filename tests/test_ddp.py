"""tests/test_ddp.py – DDP infrastructure tests (CPU / Gloo, no GPU needed).

Covers:
  * distributed helpers degrade to no-ops in single-process runs;
  * a real **world_size=2 Gloo** smoke that drives the Stage-2 (text_dm) runner
    with tiny stubs and asserts the DDP plumbing is correct: denoiser params and
    the learned CFG null token are gradient-SYNCED across ranks, the loader uses
    a DistributedSampler, and only rank 0 writes checkpoints.

The distributed test spawns 2 CPU processes (Gloo), so it runs anywhere — the
real GPU/NCCL run is exercised separately via torchrun (see docs).
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ── single-process helper behaviour ───────────────────────────────────────────

def test_helpers_single_process_are_noops():
    from sgdjscc_lab import distributed as ddp
    assert ddp.is_distributed() is False
    assert ddp.is_rank0() is True
    assert ddp.get_world_size() == 1
    m = nn.Linear(3, 3)
    assert ddp.maybe_wrap_ddp(m) is m            # pass-through (no wrap)
    assert ddp.unwrap_module(m) is m
    # reduce_metric_sums → local mean
    out = ddp.reduce_metric_sums({"loss": 4.0, "x": 2.0}, count=2)
    assert out == {"loss": 2.0, "x": 1.0}
    ddp.barrier()                                # no-op
    ddp.all_reduce_grads([m])                    # no-op


# ── Stage-2 distributed smoke (world_size=2, Gloo, CPU) ───────────────────────

class _StubDenoiser(nn.Module):
    """Minimal denoiser: depends on BOTH ft and labels (so the CFG null token
    participates in the graph), output shaped like f0 ``[B,C,h,w]``."""

    def __init__(self, ch: int = 4, label_dim: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.lab = nn.Linear(label_dim, ch)

    def forward(self, ft, noise_level, labels, enable_mask: bool = False, **kw):
        b, c, h, w = ft.shape
        bias = self.lab(labels).mean(dim=0).view(1, c, 1, 1)   # uses labels (→ null token)
        return self.conv(ft) + bias


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _ddp_worker(rank: int, world: int, tmp: str, port: int):
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from sgdjscc_lab import distributed as ddp
    from sgdjscc_lab.training.stage_runners import TextDMStageRunner
    from sgdjscc_lab.pipelines.train_pipeline import save_checkpoint

    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        assert ddp.is_distributed() and ddp.get_world_size() == world
        assert ddp.is_rank0() == (rank == 0)

        cfg = OmegaConf.create({"train": {"dm": {"cfg_null_mode": "learned",
                                                 "cfg_dropout_prob": 0.5,
                                                 "use_masked_branch": True},
                                          "grad_accum_steps": 1, "lr": 0.1}})
        torch.manual_seed(0)                       # identical init; DDP also broadcasts
        denoiser = _StubDenoiser(ch=4, label_dim=8)
        runner = TextDMStageRunner(
            denoiser,
            encode_latent_fn=lambda x: x,                       # image already latent-shaped
            encode_text_fn=lambda caps: torch.ones(len(caps), 8),
            cfg=cfg, device=torch.device("cpu"),
            param_groups=[{"params": list(denoiser.parameters())}],
        )
        # the denoiser is DDP-wrapped and registered for grad sync
        from torch.nn.parallel import DistributedDataParallel as DDP
        assert isinstance(runner.denoiser, DDP)
        assert len(runner._ddp_modules) == 2       # denoiser + null token

        # one training step on a DIFFERENT batch per rank → grads differ pre-sync
        batch = {"image": torch.randn(2, 4, 8, 8) + rank, "caption": ["a", "b"]}
        runner.training_step(batch)

        # After the synced step, denoiser params + null token are IDENTICAL on all ranks.
        def _all_same(t: torch.Tensor) -> bool:
            g = [torch.zeros_like(t) for _ in range(world)]
            dist.all_gather(g, t.contiguous())
            return all(torch.allclose(g[0], gi, atol=1e-6) for gi in g)

        core = ddp.unwrap_module(runner.denoiser)
        assert _all_same(next(core.parameters()).detach())
        assert _all_same(runner._null_core.token.detach())

        # DistributedSampler is used when building a loader under DDP.
        import tempfile
        from PIL import Image
        from sgdjscc_lab.data.datasets import build_dataloader_for_stage
        from torch.utils.data import DistributedSampler
        dd = Path(tmp) / "imgs"
        if rank == 0:
            dd.mkdir(parents=True, exist_ok=True)
            for i in range(6):
                Image.new("RGB", (16, 16), (10 * i, 20, 30)).save(dd / f"{i}.png")
        dist.barrier()
        lcfg = OmegaConf.create({"train": {"stage": "jscc", "batch_size": 2,
                                           "num_workers": 0, "image_size": 16,
                                           "dataset": {"type": "image"}}})
        loader = build_dataloader_for_stage(str(dd), lcfg, shuffle=True,
                                            training=True, stage="jscc")
        assert isinstance(loader.sampler, DistributedSampler)

        # rank0-only checkpoint write.
        ckpt_dir = Path(tmp) / "ck"
        save_checkpoint({"epoch": 1, "x": 1}, ckpt_dir, epoch=1)
        dist.barrier()
        if rank == 0:
            assert (ckpt_dir / "latest.pth").exists()
    finally:
        dist.barrier()
        dist.destroy_process_group()


def test_ddp_stage2_gloo_smoke(tmp_path):
    world = 2
    mp.spawn(_ddp_worker, args=(world, str(tmp_path), _free_port()),
             nprocs=world, join=True)
