#!/usr/bin/env python
"""benchmark_latency.py – Phase 5-B latency benchmark CLI.

Measures end-to-end and decoder latency of the SGD-JSCC reconstruction at a given
denoising step budget, using ``acceleration.latency_profiler``.

Usage
-----
python scripts/benchmark_latency.py --config configs/composed.yaml \
    --input ../inputs/test_1.png --steps 50 --runs 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sgdjscc_lab.config import load_config, merge_cli_overrides

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("benchmark_latency")


def _args():
    p = argparse.ArgumentParser(description="SGD-JSCC latency benchmark (Phase 5-B)")
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--input", "-i", default=None, help="Single image to benchmark")
    p.add_argument("--steps", type=int, default=None, help="Diffusion step budget")
    p.add_argument("--device", default=None)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    return p.parse_args()


def main():
    args = _args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, input_path=args.input, device=args.device)

    from sgdjscc_lab.phase_gates import phase5_enabled
    if not phase5_enabled(cfg):
        sys.exit(
            "Error: use_phase5 is false — Phase 5-B latency benchmarks require "
            "'use_phase5: true' in your config.\n"
            "Add it to configs/eval/default.yaml or use configs/composed_phase5.yaml."
        )

    if args.steps is not None:
        cfg.diffusion_step = int(args.steps)

    from sgdjscc_lab.runtime import resolve_device, build_models
    from sgdjscc_lab.io import list_image_files, load_image_as_tensor
    from sgdjscc_lab.pipelines.eval_pipeline import _reconstruct_with_cfg
    from sgdjscc_lab.acceleration.latency_profiler import profile_callable

    device = resolve_device(cfg.device)
    models = build_models(cfg, device)
    fpath = list_image_files(cfg.input_path)[0]
    frame = load_image_as_tensor(fpath)

    steps = int(cfg.diffusion_step)
    logger.info("Benchmarking %s at %d steps (%d runs)…", fpath.name, steps, args.runs)
    out = profile_callable(
        lambda: _reconstruct_with_cfg(frame, models, cfg),
        n_warmup=args.warmup, n_runs=args.runs, steps=steps,
    )
    out.pop("result", None)
    print("\n=== Latency benchmark ===")
    for k, v in out.items():
        print(f"  {k:<18} {v}")


if __name__ == "__main__":
    main()
