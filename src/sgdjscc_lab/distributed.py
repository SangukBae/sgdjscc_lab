"""distributed.py – minimal, correctness-first DDP helpers (torchrun-based).

These helpers let the training stack run under
``torchrun --standalone --nproc_per_node=N scripts/train.py …`` while keeping the
**single-process / CPU path byte-for-byte unchanged** (every helper degrades to a
no-op / local computation when not launched distributed).

Scope (see docs/paper_gap_closure.md "DDP"): Stage 2 (text_dm) is the validated
target; Stage 3 (controlnet) shares the same plumbing. Nothing here is GPU- or
NCCL-specific — the smoke tests run on the Gloo CPU backend.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def torchrun_launched() -> bool:
    """True if the process was started by torchrun (RANK/WORLD_SIZE present)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank() -> int:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _env_int("RANK", 0)


def get_world_size() -> int:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return _env_int("WORLD_SIZE", 1)


def get_local_rank() -> int:
    return _env_int("LOCAL_RANK", 0)


def is_distributed() -> bool:
    """True only when a process group with world_size > 1 is initialised."""
    import torch.distributed as dist
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def is_rank0() -> bool:
    """True on the main rank (always True when not distributed)."""
    return get_rank() == 0


def configure_worker_logging(non_root_level: int = logging.ERROR) -> None:
    """Suppress duplicate console logs on non-zero ranks.

    ``torchrun`` launches one Python process per rank. Without a filter, every
    informational / warning log line is emitted N times before and after DDP
    initialisation. The training UI already routes user-facing progress to rank
    0, so worker ranks keep only ``ERROR``+ console output here.
    """
    if is_rank0():
        return
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setLevel(non_root_level)


def setup_distributed():
    """Initialise the process group when launched under torchrun (WORLD_SIZE>1).

    Idempotent and safe to call always. Returns ``(rank, world_size, local_rank,
    device_or_None)``; *device* is ``cuda:{local_rank}`` when CUDA + distributed,
    else None (the caller keeps its configured device — single-process path).
    """
    import torch.distributed as dist

    world = _env_int("WORLD_SIZE", 1)
    local = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    if world > 1 and not (dist.is_available() and dist.is_initialized()):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        # rank-0-only: every rank inits, but one summary line is enough on the
        # console (each rank's device is deterministic from its local_rank).
        if rank == 0:
            logger.info("DDP init: backend=%s world_size=%d", backend, world)
    device = None
    if world > 1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{local}")
    return rank, world, local, device


def cleanup_distributed() -> None:
    """Barrier + destroy the process group (no-op when not initialised)."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()


def barrier() -> None:
    import torch.distributed as dist
    if is_distributed():
        dist.barrier()


def unwrap_module(m):
    """Return the underlying module behind a DDP wrapper (else *m* unchanged)."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    return m.module if isinstance(m, DDP) else m


def maybe_wrap_ddp(
    module,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
):
    """Wrap *module* in DDP when distributed; return it unchanged otherwise.

    Backward compatible: in single-process / CPU runs this is a pure pass-through,
    so the runner's forward calls and checkpoints are identical to before.
    """
    if not is_distributed():
        return module
    from torch.nn.parallel import DistributedDataParallel as DDP
    # device_ids only apply to CUDA modules — decide from the MODULE's actual
    # device (a CUDA-available host can still train CPU modules, e.g. the Gloo
    # CPU smoke test), otherwise DDP rejects device_ids for a CPU module.
    on_cuda = any(p.is_cuda for p in module.parameters())
    if on_cuda:
        lr = get_local_rank()
        return DDP(module, device_ids=[lr], output_device=lr,
                   find_unused_parameters=find_unused_parameters,
                   broadcast_buffers=broadcast_buffers)
    return DDP(module, find_unused_parameters=find_unused_parameters,
               broadcast_buffers=broadcast_buffers)  # CPU module


def reduce_metric_sums(sums: Dict[str, float], count: int,
                       device: Optional[torch.device] = None) -> Dict[str, float]:
    """All-reduce metric **sums + count** across ranks → averaged dict.

    This is the correct way to average a validation metric over a distributed run
    (a plain mean-of-means is wrong when ranks see different sample counts). When
    not distributed it just returns ``sum / count`` locally.
    """
    if not sums:
        return {}
    if not is_distributed():
        return {k: v / max(count, 1) for k, v in sums.items()}
    import torch.distributed as dist
    keys = sorted(sums.keys())
    dev = device or (torch.device(f"cuda:{get_local_rank()}")
                     if torch.cuda.is_available() else torch.device("cpu"))
    t = torch.tensor([sums[k] for k in keys] + [float(count)],
                     dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total = float(t[-1].item())
    return {k: (float(t[i].item()) / max(total, 1.0)) for i, k in enumerate(keys)}


def all_reduce_grads(modules: List, average: bool = True) -> None:
    """Manually all-reduce the gradients of *modules* across ranks.

    Used at a grad-accumulation FLUSH (epoch boundary): the trailing micro-steps
    were accumulated under ``no_sync`` so their grads are rank-local; this syncs
    them before the optimizer step so ranks stay identical (correctness-first).
    """
    if not is_distributed():
        return
    import torch.distributed as dist
    world = get_world_size()
    for m in modules:
        for p in unwrap_module(m).parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                if average:
                    p.grad /= world


def maybe_set_epoch(loader, epoch: int) -> None:
    """Call ``sampler.set_epoch`` when the loader uses a DistributedSampler."""
    from torch.utils.data import DistributedSampler
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)
