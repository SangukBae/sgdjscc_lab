"""sgdjscc_lab.pipelines – Inference and evaluation pipeline runners."""

from sgdjscc_lab._sgdjscc import ensure_sgdjscc_on_path

ensure_sgdjscc_on_path()

from .infer_pipeline import run_batch, run_single_image
from .saver_video_pipeline import SaverForwardOutput, SaverVideoPipeline

__all__ = [
    "run_batch",
    "run_single_image",
    "SaverForwardOutput",
    "SaverVideoPipeline",
]
