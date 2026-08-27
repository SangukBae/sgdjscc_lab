"""diagnostics/tensor_recorder.py – Per-stage tensor instrumentation.

Records shape/dtype/finite/NaN-Inf/min/max/mean/std/norm/fingerprint for a
named tensor at a named pipeline stage, for one (video, frame, seed,
ablation, path) run. Deliberately data-only (no torch.Tensor kept in the
written record) so ``tensor_stage_stats.jsonl`` stays small; the harness
keeps live tensors only transiently in memory (one frame's worth) for
pairwise comparison — see ``tensor_compare.py``.

``TensorRecorder(enabled=False)`` makes every ``.record()`` call an
immediate no-op (skips fingerprinting/stat computation entirely) — the large
multi-frame server runs (stage 5-6 of ``run_float32_digital_diagnostics.sh``)
disable tensor instrumentation and only keep path-level metrics, without a
second code path to maintain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """SHA-256 of the tensor's raw bytes in canonical (contiguous, CPU) layout.

    Two tensors with the same fingerprint are bitwise identical regardless of
    dtype view — used for the float32 wire round-trip exactness check.
    """
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def tensor_stats(
    tensor: Any,
    *,
    video: str,
    frame: int,
    seed: int,
    ablation: str,
    path: str,
    stage: str,
) -> Dict[str, Any]:
    """Build one JSONL-ready record for *tensor* at *stage*.

    ``tensor=None`` (stage not applicable to this path/ablation combination,
    e.g. ControlNet-off has no ``controlnet_input_latent``) records
    ``present: False`` rather than being silently omitted, so the absence
    itself is visible in ``tensor_stage_stats.jsonl``.
    """
    row: Dict[str, Any] = {
        "video": video, "frame": int(frame), "seed": int(seed),
        "ablation": ablation, "path": path, "stage": stage,
    }
    if tensor is None:
        row["present"] = False
        return row
    if not torch.is_tensor(tensor):
        row.update({"present": True, "is_tensor": False, "value": repr(tensor)})
        return row

    t = tensor.detach()
    finite_mask = torch.isfinite(t)
    n_nan = int(torch.isnan(t).sum().item())
    n_inf = int(torch.isinf(t).sum().item())
    numel = int(t.numel())
    finite_t = t[finite_mask].float()
    all_finite = bool(finite_mask.all().item()) if numel else True

    row.update({
        "present": True,
        "is_tensor": True,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": numel,
        "finite": all_finite,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "min": float(finite_t.min().item()) if finite_t.numel() else None,
        "max": float(finite_t.max().item()) if finite_t.numel() else None,
        "mean": float(finite_t.mean().item()) if finite_t.numel() else None,
        "std": float(finite_t.std().item()) if finite_t.numel() > 1 else None,
        "norm": float(torch.linalg.norm(finite_t).item()) if finite_t.numel() else None,
        "fingerprint": tensor_fingerprint(t) if all_finite else None,
    })
    return row


@dataclass
class TensorRecorder:
    """Accumulates stage records for one run and (optionally) live tensors
    for the CURRENT (video, frame, ablation) iteration only — the caller is
    responsible for clearing ``live`` between iterations (see
    ``diagnose_float32_digital_quality.py``'s main loop) so memory does not
    grow across a multi-frame run.
    """

    enabled: bool = True
    save_tensor_files: bool = False
    tensor_dir: Optional[Path] = None
    rows: List[Dict[str, Any]] = field(default_factory=list)
    live: Dict[tuple, torch.Tensor] = field(default_factory=dict)

    def record(
        self,
        tensor: Any,
        *,
        video: str,
        frame: int,
        seed: int,
        ablation: str,
        path: str,
        stage: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        row = tensor_stats(
            tensor, video=video, frame=frame, seed=seed,
            ablation=ablation, path=path, stage=stage,
        )
        self.rows.append(row)
        key = (video, frame, seed, ablation, path, stage)
        if torch.is_tensor(tensor):
            self.live[key] = tensor.detach().cpu().clone()
            if self.save_tensor_files and self.tensor_dir is not None:
                self._save_tensor_file(key, tensor)
        return row

    def _save_tensor_file(self, key: tuple, tensor: torch.Tensor) -> None:
        video, frame, seed, ablation, path, stage = key
        out_dir = self.tensor_dir / video / str(frame) / ablation / path
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.detach().cpu(), out_dir / f"{stage}.pt")

    def clear_live(self) -> None:
        self.live.clear()

    def write_jsonl(self, path: "str | Path") -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
