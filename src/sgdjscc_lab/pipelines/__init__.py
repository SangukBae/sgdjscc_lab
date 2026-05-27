"""sgdjscc_lab.pipelines – Inference and evaluation pipeline runners."""

from .infer_pipeline import run_batch, run_single_image

__all__ = ["run_batch", "run_single_image"]
