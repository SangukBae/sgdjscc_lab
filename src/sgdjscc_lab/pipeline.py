"""pipeline.py – Phase 2 compatibility shim.

The inference logic has moved to:
  pipelines/infer_pipeline.py

This module re-exports the public API so that any code still importing from
``sgdjscc_lab.pipeline`` continues to work without modification.
"""

from sgdjscc_lab.pipelines.infer_pipeline import (  # noqa: F401
    run_batch,
    run_single_image,
    _process_patches,
    _jscc_forward,
    _compute_step,
)
