#!/usr/bin/env python
"""benchmark_sampling.py – Phase 5-B quality-vs-latency sweep CLI.

Runs the SGD-JSCC reconstruction at several denoising step budgets and reports the
latency and (optional) SRS for each, producing the quality-vs-latency / SRS-vs-
latency data described in the Phase 5-B plan.

Usage
-----
python scripts/benchmark_sampling.py --config configs/composed.yaml \
    --input ../inputs/test_1.png --steps 50,20,10,5
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sgdjscc_lab.config import load_config, merge_cli_overrides

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("benchmark_sampling")


def _args():
    p = argparse.ArgumentParser(description="SGD-JSCC sampling quality/latency sweep")
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--input", "-i", default=None)
    p.add_argument("--steps", default="50,20,10,5", help="Comma-separated step budgets")
    p.add_argument("--device", default=None)
    p.add_argument("--no-srs", action="store_true", help="Skip SRS (latency only)")
    p.add_argument("--out-csv", default="../outputs/sampling_benchmark.csv")
    return p.parse_args()


def main():
    args = _args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, input_path=args.input, device=args.device)

    from sgdjscc_lab.phase_gates import phase5_enabled
    if not phase5_enabled(cfg):
        sys.exit(
            "Error: use_phase5 is false — Phase 5-B sampling benchmarks require "
            "'use_phase5: true' in your config.\n"
            "Add it to configs/eval/default.yaml or use configs/composed_phase5.yaml."
        )

    budgets = [int(s) for s in str(args.steps).split(",")]

    from omegaconf import OmegaConf
    from sgdjscc_lab.runtime import resolve_device, build_models
    from sgdjscc_lab.io import list_image_files, load_image_as_tensor
    from sgdjscc_lab.pipelines.eval_pipeline import _reconstruct_with_cfg
    from sgdjscc_lab.acceleration.latency_profiler import profile_callable

    device = resolve_device(cfg.device)
    models = build_models(cfg, device)
    fpath = list_image_files(cfg.input_path)[0]
    frame = load_image_as_tensor(fpath)

    srs_eval = None
    if not args.no_srs:
        from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
        srs_eval = SemanticReliabilityEvaluator()

    rows = []
    for steps in budgets:
        run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        run_cfg.diffusion_step = steps
        prof = profile_callable(lambda: _reconstruct_with_cfg(frame, models, run_cfg),
                                n_warmup=1, n_runs=2, steps=steps)
        recon = prof.pop("result")
        srs = None
        if srs_eval is not None:
            srs = srs_eval.evaluate(frame, recon).get("semantic_reliability_score")
        row = {"steps": steps, "mean_s": prof["mean_s"],
               "per_step_s": prof.get("per_step_s"), "srs": srs}
        rows.append(row)
        logger.info("steps=%d  latency=%.3fs  srs=%s", steps, prof["mean_s"], srs)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["steps", "mean_s", "per_step_s", "srs"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved sweep → {out_csv}")


if __name__ == "__main__":
    main()
