"""scripts/lgvsc_example_callable_backend.py – Template adapter for
``scripts/lgvsc_generate_worker.py --backend callable`` (ETRI 후속 1단계 step 1B).

This is a **template**, not a real Open-Sora/Wan integration — it deliberately
falls back to the exact same deterministic keyframe blend as
``lgvsc_generate_worker.py``'s ``mock`` backend, so it can be used as a smoke
test for the ``--backend callable`` dispatch path itself (proving the dynamic
import + function-call plumbing works) without requiring any model weights.

To wire up a real backend (Open-Sora, Wan, or anything else): copy this file
into the conda environment that has your model's package installed (e.g.
``lgvsc_gen``), replace the body of :func:`generate_segment` with your actual
model call, and point the worker at it::

    python scripts/lgvsc_generate_worker.py \\
        --manifest <manifest.json> --output-dir <out/> \\
        --backend callable \\
        --backend-entrypoint lgvsc_example_callable_backend:generate_segment

The function signature below is the entire contract — see
``lgvsc_generate_worker.py``'s module docstring ("callable backend contract")
for the authoritative description.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict


def generate_segment(manifest: dict, manifest_dir: Path, args: argparse.Namespace) -> Dict:
    """Return ``{"frames": {target_index: image, ...}, "metadata": {target_index: dict, ...}}``.

    *manifest*: the parsed JSON request (see ``lgvsc_generate_worker.py``'s
    module docstring for the exact schema) — ``manifest["target_indices"]``
    are the frame indices you must produce, one image each.
    *manifest_dir*: directory the manifest lives in — keyframe image paths in
    the manifest (``start_keyframe_image``/``end_keyframe_image``) are
    relative to this.
    *args*: the parsed CLI namespace (``--model-id``/``--device``/``--dtype``/
    ``--seed``/``--height``/``--width``/``--num-inference-steps``/
    ``--decode-chunk-size``/``--extra-json``), so your adapter can honour the
    same run-config knobs the built-in backends do.

    Each value in ``"frames"`` may be a ``PIL.Image.Image`` or an HWC numpy
    array (float in [0, 1] or uint8 in [0, 255]). Each value in ``"metadata"``
    should be a plain JSON-serialisable dict — see
    ``src/sgdjscc_lab/video/video_generator.py``'s ``GenerationMetadata`` for
    the fields ``ExternalSegmentWorkerGenerator`` expects
    (``backend``/``conditioning_mode``/``source_keyframe_index``/
    ``end_keyframe_index``/``target_indices``/``relative_position``/
    ``used_caption``/``used_side_info``/``mock``/``notes``) — missing optional
    fields default sensibly, but ``target_indices`` must equal ``[that one
    index]`` for every entry (this is enforced by
    ``video_generator.validate_segment_result()`` after the round-trip).
    """
    # Reuse the worker's own mock helpers so this template is a genuine,
    # runnable smoke test — replace this whole block with your real model call.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lgvsc_generate_worker import run_mock_backend  # noqa: E402

    return run_mock_backend(manifest, manifest_dir, args)
