#!/usr/bin/env python
"""scripts/lgvsc_generate_worker.py – Out-of-process segment generation worker
(ETRI 후속 1단계 step 1B).

Scope note (read before touching this file)
--------------------------------------------
1A (``src/sgdjscc_lab/video/video_generator.py``) built the Rx-legal
GOP/segment-level generation *contract* (``SegmentGenerationRequest`` /
``SegmentGenerationResult`` / ``generate_segment()``) with mock in-process
backends only. 1B's job is to let a **real** video generation model (Open-Sora,
Wan, Stable Video Diffusion via ``diffusers``, or anything else) sit behind
that contract — without ever installing that model's (large, version-fragile)
dependencies into the ``ptest`` conda environment this repo's tests run in.

The split is: ``sgdjscc_lab.video.video_generator.ExternalSegmentWorkerGenerator``
(runs in ``ptest``, imports only stdlib + torch/torchvision/PIL, all of which
``ptest`` already has) launches *this* script as a **subprocess in a different
Python interpreter** (e.g. a separate ``lgvsc_gen`` conda env with
``diffusers``/``torch``/model weights installed) and exchanges data with it
purely through files: a JSON *manifest* + PNG keyframe images in, PNG frame
images + a JSON *result* (or *error*) out. Neither side needs to import the
other's package — this script does not import ``sgdjscc_lab`` at all, and can
be copied into (or referenced from) a completely separate environment.

Only the ``mock`` backend below is exercised by this repo's own ``ptest``
suite (no model weights, no GPU, no network — just PIL + numpy). The ``svd``
and ``wan`` backends are **reference wiring** for real diffusers pipelines —
see docs/lgvsc_1b_worker_readiness.md for exactly what has and hasn't been
verified against real weights/GPU in this repo before using either for a real
evaluation run:

- ``svd`` — ``diffusers.StableVideoDiffusionPipeline``. Image-only
  conditioning; the standard SVD pipeline has no text-prompt or second-image
  input, so captions/side-info/end-keyframe are accepted by the manifest but
  not used.
- ``wan`` — ``diffusers.WanImageToVideoPipeline`` (Wan2.1/2.2 image-to-video).
  The closer match to LGVSC's segment-decoder contract: genuinely conditions
  on the start keyframe (``image``), the end keyframe when present
  (``last_image`` — real bidirectional conditioning), and the caption
  (``prompt`` — real text conditioning). ``side_infos`` are still accepted
  but not used (no established way to turn those numeric dicts into a useful
  condition for this pipeline) — a documented limitation, not silently
  dropped. Bidirectional (``last_image``) conditioning requires a Wan
  first-last-frame (FLF2V) checkpoint, NOT the plain start-only I2V
  checkpoint — see ``run_wan_backend``'s docstring and
  ``extra_json.bidirectional_model_id`` for how this script picks the right
  one per segment.

The ``callable`` backend is the recommended extension point for an actual
Open-Sora integration (or a Wan variant needing the experimental "modular
pipeline" API): point it at your own adapter function (see "callable backend
contract" below) written against whatever exact package version you install —
this script deliberately does not guess Open-Sora's API, since it isn't
installed anywhere in this repo and its API is not stable enough to hard-code
confidently.

IPC contract
------------
**Manifest (in, JSON)** — written by ``ExternalSegmentWorkerGenerator``,
mirrors ``SegmentGenerationRequest`` minus its tensors (which become PNG files
alongside the manifest, referenced by filename)::

    {
      "segment_id": int,
      "start_frame_index": int, "end_frame_index": int, "segment_length": int,
      "target_indices": [int, ...],
      "start_keyframe_index": int, "end_keyframe_index": int | null,
      "start_keyframe_image": "start_keyframe.png",
      "end_keyframe_image": "end_keyframe.png" | null,
      "fps": float | null,
      "captions": [str | null, ...],       # aligned with target_indices
      "packets": [dict | null, ...],       # aligned with target_indices
      "side_infos": [dict | null, ...],    # aligned with target_indices
      "run_config": {                      # PROVENANCE ONLY — CLI flags win when both are given
        "seed": int | null, "model_id": str | null, "device": str | null,
        "dtype": str | null, "height": int | null, "width": int | null
      }
    }

All pixel evidence is Rx-legal by construction: only reconstructed keyframes
(never the un-transmitted original target frame — ``SegmentGenerationRequest``
has no such field to leak in the first place; see
``video/video_generator.py``'s module docstring).

**Result (out, JSON, on success)** — ``<output-dir>/result.json``::

    {
      "status": "ok", "segment_id": int, "backend": "mock"|"svd"|"wan"|"callable",
      "model_id": str | null, "device": str, "seed": int | null,
      "duration_sec": float, "target_indices": [int, ...],
      "frames": {"<index>": "frame_00001.png", ...},     # relative to output-dir
      "metadata": {"<index>": {...GenerationMetadata-shaped dict...}, ...}
    }

**Error (out, JSON, on failure)** — ``<output-dir>/error.json``, and the
process exits non-zero::

    {"status": "error", "error_type": str, "message": str, "traceback": str}

``main()`` always writes exactly one of these two files before returning,
even when a backend raises — a caller (``ExternalSegmentWorkerGenerator``)
that sees a non-zero exit code with no ``error.json`` at all knows the worker
crashed too hard to self-report (segfault, OOM-killed, etc.) and should surface
the raw exit code + captured stdout/stderr instead.

callable backend contract
--------------------------
``--backend callable --backend-entrypoint "your.module:generate_segment"``
dynamically imports ``your.module`` (in *this* process — i.e. in whatever
environment ``python scripts/lgvsc_generate_worker.py`` is actually running
under) and calls::

    def generate_segment(manifest: dict, manifest_dir: pathlib.Path,
                          args: argparse.Namespace) -> dict:
        \"\"\"Return {"frames": {target_index: PIL.Image.Image | HWC numpy array
        in [0, 1] or [0, 255], ...}, "metadata": {target_index: dict, ...}}.
        Every target_index in manifest["target_indices"] must be a key in both.
        \"\"\"

This is the intended plug-in point for a real Open-Sora/Wan backend: write a
small adapter module in the environment that has that package installed,
implementing exactly this function, and pass its dotted path here. See
``scripts/lgvsc_example_callable_backend.py`` for a documented
template (also runnable as a smoke test via ``--backend callable`` — it just
falls back to the same deterministic blend as ``mock``, to prove the IPC path
works end-to-end without any real model).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional


class WorkerBackendUnavailableError(RuntimeError):
    """Raised when a backend's dependency/weights/entrypoint cannot be used —
    always caught by ``main()`` and reported via ``error.json`` with a
    human-readable message, never left as a bare traceback with no context.
    """


# ── manifest / image IO (PIL + numpy only — no torch here on purpose, see
#    module docstring: this keeps the mock/callable paths runnable in any
#    Python environment that merely has Pillow + numpy installed) ────────────

def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_keyframe_image(path):
    """Return an HWC float32 numpy array in [0, 1]."""
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def save_frame_image(image, path) -> None:
    """*image*: a PIL.Image, or an HWC numpy array (float in [0, 1], or uint8
    in [0, 255] — both accepted so a callable backend can return whichever is
    more natural for it)."""
    from PIL import Image
    import numpy as np
    if isinstance(image, Image.Image):
        image.save(path)
        return
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


# ── backends ──────────────────────────────────────────────────────────────────

def run_mock_backend(manifest: dict, manifest_dir: Path, args: argparse.Namespace) -> Dict:
    """Deterministic, dependency-free backend — proves the IPC round-trip
    (manifest → subprocess → result files) works with no model at all. This
    is the ONLY backend this repo's own test suite (``ptest``) exercises.

    Behaviour mirrors ``video/video_generator.py``'s
    ``BidirectionalInterpolationGenerator``/``CopyGenerator`` mocks: linear
    blend of the start/end keyframe images by relative position when an end
    keyframe is given, else a plain copy of the start keyframe. Captions/
    side-info are accepted and reflected in ``used_caption``/``used_side_info``
    metadata but do not affect the pixels (this is a structural stand-in, not
    a claim about generation quality — see the module docstring).
    """
    start_img = load_keyframe_image(manifest_dir / manifest["start_keyframe_image"])
    end_rel = manifest.get("end_keyframe_image")
    end_img = load_keyframe_image(manifest_dir / end_rel) if end_rel else None

    start_idx = int(manifest["start_keyframe_index"])
    end_idx = manifest.get("end_keyframe_index")
    end_idx = int(end_idx) if end_idx is not None else None

    target_indices = [int(i) for i in manifest["target_indices"]]
    n = len(target_indices)
    captions = manifest.get("captions") or [None] * n
    side_infos = manifest.get("side_infos") or [None] * n

    frames: Dict[int, object] = {}
    metadata: Dict[int, Dict] = {}
    bidirectional = end_img is not None and end_idx is not None and end_idx != start_idx
    for i, idx in enumerate(target_indices):
        caption = captions[i] if i < len(captions) else None
        side_info = side_infos[i] if i < len(side_infos) else None
        if bidirectional:
            a = min(max((idx - start_idx) / (end_idx - start_idx), 0.0), 1.0)
            frame = (1.0 - a) * start_img + a * end_img
            conditioning_mode = "bidirectional"
            relative_position = float(a)
        else:
            frame = start_img.copy()
            conditioning_mode = "start_only"
            relative_position = None
        frames[idx] = frame
        metadata[idx] = {
            "backend": "external_segment_worker:mock",
            "conditioning_mode": conditioning_mode,
            "source_keyframe_index": start_idx,
            "end_keyframe_index": end_idx if conditioning_mode == "bidirectional" else None,
            "target_indices": [idx],
            "relative_position": relative_position,
            "used_caption": bool(caption),
            "used_side_info": bool(side_info),
            "mock": True,
            "notes": (
                "external_segment_worker mock backend (ETRI 1B): deterministic "
                "blend of start/end keyframe images, produced via an actual "
                "out-of-process subprocess round-trip; not learned generation. "
                f"seed={args.seed!r} recorded for provenance only (mock output "
                "does not depend on it)."
            ),
        }
    return {"frames": frames, "metadata": metadata}


def run_svd_backend(manifest: dict, manifest_dir: Path, args: argparse.Namespace) -> Dict:
    """Best-effort reference wiring for ``diffusers``'
    ``StableVideoDiffusionPipeline`` — image-to-video, start-keyframe-only
    conditioning (the standard SVD pipeline has no text-prompt or
    second-image input, so captions/side-info/end-keyframe are accepted by
    the manifest but NOT used here).

    NOT verified against real weights/GPU by this repo (no GPU / diffusers
    install in this session) — read docs/lgvsc_1b_worker_readiness.md before
    trusting this path; confirm the exact call shape against your installed
    ``diffusers`` version first. Prefer ``--backend callable`` for a real
    Open-Sora/Wan integration that needs the full contract (captions,
    side-info, bidirectional conditioning).
    """
    try:
        import torch
        from diffusers import StableVideoDiffusionPipeline
        from PIL import Image
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise WorkerBackendUnavailableError(
            "backend=svd requires `torch` + `diffusers` (+ `transformers`/"
            "`accelerate`) installed in THIS worker's Python environment — "
            "these must NOT be installed into `ptest`; use a separate conda env "
            "(e.g. `lgvsc_gen`). Install with `pip install diffusers transformers "
            f"accelerate`. Original import error: {exc}"
        ) from exc

    model_id = args.model_id or "stabilityai/stable-video-diffusion-img2vid-xt"
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.dtype, torch.float16)

    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        pipe = pipe.to(args.device)
    except Exception as exc:  # noqa: BLE001
        raise WorkerBackendUnavailableError(
            f"Could not load SVD pipeline {model_id!r} onto device {args.device!r}: "
            f"{exc}. Check: (1) Hugging Face auth — `huggingface-cli login` — and "
            "license acceptance for this gated model on huggingface.co; (2) enough "
            "free VRAM for the model + your requested resolution/frame count; "
            "(3) network access to download weights on first use."
        ) from exc

    start_arr = load_keyframe_image(manifest_dir / manifest["start_keyframe_image"])
    start_img = Image.fromarray((start_arr * 255.0).round().astype(np.uint8))
    output_size = start_img.size
    if args.height or args.width:
        h = args.height or start_img.height
        w = args.width or start_img.width
        start_img = start_img.resize((w, h))
    height, width = start_img.height, start_img.width

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(int(args.seed))

    target_indices = [int(i) for i in manifest["target_indices"]]
    # SVD checkpoints are trained/configured for a clip length (14 for SVD,
    # 25 for SVD-XT). Passing num_frames=1 for a one-target smoke test can fail
    # inside the pipeline's latent/image_latent concatenation. Generate at
    # least the model's native clip length, then map the requested target
    # indices onto the first generated frames below.
    model_num_frames = int(getattr(pipe.unet.config, "num_frames", 14))
    n_frames = max(len(target_indices), model_num_frames)
    try:
        out = pipe(
            start_img,
            height=height,
            width=width,
            num_frames=n_frames,
            num_inference_steps=args.num_inference_steps or 25,
            decode_chunk_size=args.decode_chunk_size or min(8, n_frames),
            generator=generator,
        )
        generated = out.frames[0]  # list[PIL.Image], expected length == n_frames
    except Exception as exc:  # noqa: BLE001
        raise WorkerBackendUnavailableError(
            f"SVD pipeline call failed on device {args.device!r} (model={model_id!r}, "
            f"num_frames={n_frames}): {exc}. Common causes: out-of-memory (reduce "
            "--height/--width or num_frames), or a diffusers version whose "
            "StableVideoDiffusionPipeline signature differs from what this script "
            "assumes — check the installed `diffusers` version's own example code."
        ) from exc

    frames, metadata = {}, {}
    for i, idx in enumerate(target_indices):
        frame = generated[min(i, len(generated) - 1)]
        if frame.size != output_size:
            frame = frame.resize(output_size, Image.BICUBIC)
        frames[idx] = frame
        metadata[idx] = {
            "backend": f"external_segment_worker:svd:{model_id}",
            "conditioning_mode": "start_only",
            "source_keyframe_index": int(manifest["start_keyframe_index"]),
            "end_keyframe_index": None,
            "target_indices": [idx],
            "relative_position": None,
            "used_caption": False,
            "used_side_info": False,
            "mock": False,
            "notes": (
                "diffusers StableVideoDiffusionPipeline (image-conditioned only — "
                "no caption/side-info/end-keyframe support in the standard pipeline). "
                f"Generated an internal {n_frames}-frame SVD clip and mapped "
                f"target position {i} to this requested frame. "
                "Reference wiring, NOT verified against real GPU output by this repo "
                "— see docs/lgvsc_1b_worker_readiness.md."
            ),
        }
    return {"frames": frames, "metadata": metadata}


def run_wan_backend(manifest: dict, manifest_dir: Path, args: argparse.Namespace) -> Dict:
    """Reference wiring for ``diffusers``' ``WanImageToVideoPipeline`` (Wan2.1/
    2.2 image-to-video) — the closest off-the-shelf match to LGVSC's
    SKEM+DSA-style segment decoder available via a stable, documented
    diffusers pipeline class (as opposed to Wan2.2's still-experimental
    "modular pipeline" API, which this script deliberately does not use).

    Unlike ``run_svd_backend`` (image-only), Wan's I2V pipeline genuinely
    supports the fuller LGVSC segment contract:

    - ``image`` — the start keyframe (always used, Rx-legal).
    - ``last_image`` — the end keyframe, when the manifest has one
      (``conditioning_mode="bidirectional"``; real bidirectional
      conditioning, not a linear blend — this is what
      ``BidirectionalInterpolationGenerator``/``run_mock_backend`` only
      *simulate*).
    - ``prompt`` — the target frame's caption, when available (real text
      conditioning; ``used_caption`` reflects whether one was actually found
      and passed).

    ``side_infos`` (semantic-delta/motion dicts) are accepted by the manifest
    but **not** folded into the prompt or any other conditioning signal here
    — there is no established, verified way to turn those numeric dicts into
    a useful text/latent condition for this pipeline, so ``used_side_info``
    is always ``False``. Documented as a known limitation rather than
    overclaimed; see docs/lgvsc_1b_worker_readiness.md.

    NOT verified against real weights/GPU in every configuration by this
    repo — read docs/lgvsc_1b_worker_readiness.md before trusting this path
    for a real evaluation run. The smallest official Wan I2V checkpoint
    (Wan2.1-I2V-14B-480P) is ~90GB and VRAM-hungry; set
    ``video_generator.worker.extra_json`` to
    ``'{"offload_mode": "sequential"}'`` for a much lower (but much slower)
    VRAM footprint via ``enable_sequential_cpu_offload()``, or
    ``'{"offload_mode": "model"}'`` for the faster ``enable_model_cpu_offload()``.

    Checkpoint choice for bidirectional (``last_image``) conditioning
    -------------------------------------------------------------------
    ``WanImageToVideoPipeline``'s ``last_image`` path only works with a
    checkpoint whose ``transformer/config.json`` has ``pos_embed_seq_len``
    set — that value drives a learned positional-embedding parameter
    (``WanImageEmbedding.pos_embed``) that reshapes the two-image (start+end)
    CLIP embedding batch into a single doubled-length sequence so it can be
    concatenated with the (batch-1) text embedding. The default
    ``Wan2.1-I2V-14B-480P`` checkpoint has **no** ``pos_embed_seq_len`` (it
    was only ever trained for single-image start-only conditioning) — asking
    it to use ``last_image`` skips that reshape and crashes downstream in
    ``transformer_wan.py`` with a batch-size-mismatch ``RuntimeError`` on
    ``torch.concat([encoder_hidden_states_image, encoder_hidden_states],
    dim=1)`` (confirmed by direct inspection of the diffusers source, not a
    diffusers version bug). Wan's official **first-last-frame** checkpoint,
    ``Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers``, does ship
    ``pos_embed_seq_len: 514`` and is the correct checkpoint for real
    bidirectional conditioning — but by the same reshape math it can ONLY do
    two-image conditioning (a single-image call reshapes a 257-token
    embedding into a 514-token buffer, which fails), so it cannot serve
    start-only segments (e.g. a video's last GOP, which has no end keyframe).

    Because a manifest's ``end_keyframe_image`` presence is decided per
    segment (by ``TemporalPipeline``, not by this script), a single worker
    invocation may be asked for either start-only or bidirectional
    generation depending on which GOP it's serving. This function therefore
    picks the model per-call: ``args.model_id`` (default
    ``Wan2.1-I2V-14B-480P``) for start-only segments, and
    ``extra_json["bidirectional_model_id"]`` (if set) for segments that do
    have an end keyframe — e.g. ``video_generator.worker.extra_json:
    '{"bidirectional_model_id": "Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers"}'``.
    If a checkpoint is loaded that doesn't match what the segment actually
    needs (missing ``pos_embed_seq_len`` for a bidirectional segment, or
    present for a start-only one), this raises ``WorkerBackendUnavailableError``
    immediately with that exact explanation, instead of letting the cryptic
    downstream tensor-shape error surface.
    """
    try:
        import torch
        from diffusers import WanImageToVideoPipeline
        from PIL import Image
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise WorkerBackendUnavailableError(
            "backend=wan requires `torch` + `diffusers` (+ `transformers`/"
            "`accelerate`) installed in THIS worker's Python environment — "
            "these must NOT be installed into `ptest`; use a separate conda env "
            "(e.g. `lgvsc_gen`, or this machine's `semantic-diffusers`). Install "
            "with `pip install diffusers transformers accelerate`. Original "
            f"import error: {exc}"
        ) from exc

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    extra: Dict = {}
    if args.extra_json:
        try:
            extra = json.loads(args.extra_json)
        except (json.JSONDecodeError, TypeError):
            extra = {}
    offload_mode = extra.get("offload_mode")
    bidirectional_model_id = extra.get("bidirectional_model_id")

    start_arr = load_keyframe_image(manifest_dir / manifest["start_keyframe_image"])
    start_img = Image.fromarray((start_arr * 255.0).round().astype(np.uint8))
    output_size = start_img.size

    end_rel = manifest.get("end_keyframe_image")
    end_img = None
    if end_rel:
        end_arr = load_keyframe_image(manifest_dir / end_rel)
        end_img = Image.fromarray((end_arr * 255.0).round().astype(np.uint8))

    # Pick the checkpoint BEFORE loading it: a segment with an end keyframe
    # needs a checkpoint with pos_embed_seq_len (see docstring); one without
    # needs the plain start-only checkpoint. Falls back to args.model_id if
    # no bidirectional_model_id is configured (matches the pre-existing
    # single-model_id behavior for configs that never set it).
    default_model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    if end_img is not None and bidirectional_model_id:
        model_id = bidirectional_model_id
    else:
        model_id = args.model_id or default_model_id

    try:
        pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        if offload_mode == "sequential":
            pipe.enable_sequential_cpu_offload(device=args.device)
        elif offload_mode == "model":
            pipe.enable_model_cpu_offload(device=args.device)
        else:
            pipe = pipe.to(args.device)
    except Exception as exc:  # noqa: BLE001
        raise WorkerBackendUnavailableError(
            f"Could not load Wan I2V pipeline {model_id!r} onto device "
            f"{args.device!r}: {exc}. Check: (1) enough disk space + network "
            "access to download weights on first use (the smallest official "
            "Wan I2V checkpoint is ~90GB); (2) enough VRAM — set "
            "video_generator.worker.extra_json='{\"offload_mode\": \"sequential\"}' "
            "for a much lower-VRAM (but much slower) CPU-offloaded run; "
            "(3) Hugging Face access if using a gated mirror."
        ) from exc

    pos_embed_seq_len = getattr(pipe.transformer.config, "pos_embed_seq_len", None)
    if end_img is not None and pos_embed_seq_len is None:
        raise WorkerBackendUnavailableError(
            f"Segment has an end keyframe (bidirectional conditioning requested) "
            f"but the loaded checkpoint {model_id!r} has no `pos_embed_seq_len` "
            "in its transformer config — it was only trained for single-image "
            "(start-only) conditioning and will crash inside "
            "transformer_wan.py's image/text embedding concat if used with "
            "`last_image`. Set video_generator.worker.extra_json to include "
            "'\"bidirectional_model_id\": \"Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers\"' "
            "(Wan's official first-last-frame checkpoint, which does have "
            "pos_embed_seq_len) so bidirectional segments load the right model."
        )
    if end_img is None and pos_embed_seq_len is not None:
        raise WorkerBackendUnavailableError(
            f"Segment has NO end keyframe (start-only conditioning requested) "
            f"but the loaded checkpoint {model_id!r} has `pos_embed_seq_len="
            f"{pos_embed_seq_len}` in its transformer config — this checkpoint "
            "(e.g. a Wan FLF2V first-last-frame variant) was trained to require "
            "TWO conditioning images and will fail to reshape a single-image "
            "embedding. Point --model-id / video_generator.worker.model_id at "
            "a plain start-only checkpoint (e.g. Wan-AI/Wan2.1-I2V-14B-480P-"
            "Diffusers) for segments without an end keyframe, and use "
            "extra_json.bidirectional_model_id for the FLF2V checkpoint instead "
            "of setting it as the primary model_id."
        )

    height = args.height or start_img.height
    width = args.width or start_img.width

    target_indices = [int(i) for i in manifest["target_indices"]]
    start_frame_index = int(manifest["start_frame_index"])
    end_frame_index = int(manifest["end_frame_index"])

    captions = manifest.get("captions") or [None] * len(target_indices)
    side_infos = manifest.get("side_infos") or [None] * len(target_indices)
    prompt = next((c for c in captions if c), None)
    used_caption = prompt is not None
    used_side_info = False  # accepted by the manifest, not used — see docstring
    _ = side_infos  # explicitly unused; kept for readability at the call site

    conditioning_mode = "start_only"
    end_keyframe_index_raw = manifest.get("end_keyframe_index")
    bidirectional_end = None
    if end_img is not None:
        conditioning_mode = "bidirectional"
        if end_keyframe_index_raw is not None:
            bidirectional_end = int(end_keyframe_index_raw)

    # The generated clip's frame 0 is the start keyframe (segment offset 0).
    # In bidirectional mode, `last_image` conditions the clip's LAST frame to
    # be the end keyframe, so the clip spans exactly
    # [start_frame_index, end_keyframe_index] (inclusive). In start-only mode
    # there is no defined end, so the clip only needs to reach far enough to
    # cover every requested target's offset from the start keyframe.
    span_end = bidirectional_end if bidirectional_end is not None else end_frame_index
    span_end = max([span_end, start_frame_index] + target_indices)
    n_requested = max(span_end - start_frame_index + 1, 1)
    # Wan's temporal VAE compresses (num_frames - 1) by 4x internally — valid
    # frame counts satisfy (num_frames - 1) % 4 == 0. Round the request up to
    # the nearest valid count (and at least 5) so a short smoke test doesn't
    # get silently rejected/mis-shaped by the pipeline.
    remainder = (n_requested - 1) % 4
    n_frames = n_requested if remainder == 0 else n_requested + (4 - remainder)
    n_frames = max(n_frames, 5)

    def _clip_position(target_index: int) -> int:
        """Map an absolute target frame index to its position within the
        generated clip, using the target's actual temporal offset from the
        segment start (not its position in the target_indices list) — a
        non-contiguous target set like [1, 5, 8] must land on the clip
        frames at those relative offsets, not on generated[0]/[1]/[2]."""
        offset = target_index - start_frame_index
        if bidirectional_end is not None and bidirectional_end > start_frame_index:
            # Rescale into the (possibly-padded) generated clip so the
            # fraction of the way from start to end keyframe is preserved
            # even when n_frames was rounded up past the true span.
            frac = offset / (bidirectional_end - start_frame_index)
            pos = round(frac * (n_frames - 1))
        else:
            pos = offset
        return max(0, min(int(pos), n_frames - 1))

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed))

    call_kwargs = dict(
        image=start_img,
        prompt=prompt or "",
        height=height,
        width=width,
        num_frames=n_frames,
        num_inference_steps=args.num_inference_steps or 30,
        generator=generator,
    )
    if end_img is not None:
        call_kwargs["last_image"] = end_img

    try:
        out = pipe(**call_kwargs)
        generated = out.frames[0]  # list/array of frames, length == n_frames
    except Exception as exc:  # noqa: BLE001
        raise WorkerBackendUnavailableError(
            f"Wan I2V pipeline call failed on device {args.device!r} (model="
            f"{model_id!r}, num_frames={n_frames}, height={height}, "
            f"width={width}): {exc}. Common causes: out-of-memory (reduce "
            "--height/--width, request fewer target frames, or set "
            "video_generator.worker.extra_json='{\"offload_mode\": \"sequential\"}'), "
            "or a diffusers version whose WanImageToVideoPipeline signature "
            "differs from what this script assumes."
        ) from exc

    end_keyframe_index = end_keyframe_index_raw if conditioning_mode == "bidirectional" else None
    notes_conditioning = "start-keyframe conditioned"
    if conditioning_mode == "bidirectional":
        notes_conditioning += (
            f" + end-keyframe (last_image) bidirectional conditioning — checkpoint "
            f"{model_id!r} genuinely used (pos_embed_seq_len={pos_embed_seq_len}, "
            "confirms this checkpoint's transformer was actually trained for "
            "two-image start+end conditioning, not a simulated/interpolated blend)"
        )
    notes_caption = f", prompt=caption ({prompt[:60]!r})" if used_caption else ", no caption available (empty prompt)"

    frames_out, metadata = {}, {}
    for idx in target_indices:
        pos = _clip_position(idx)
        frame = generated[pos]
        if not isinstance(frame, Image.Image):
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            frame = Image.fromarray(arr)
        if frame.size != output_size:
            frame = frame.resize(output_size, Image.BICUBIC)
        frames_out[idx] = frame
        relative_position = (
            (idx - start_frame_index) / (bidirectional_end - start_frame_index)
            if bidirectional_end is not None and bidirectional_end > start_frame_index
            else None
        )
        metadata[idx] = {
            "backend": f"external_segment_worker:wan:{model_id}",
            "conditioning_mode": conditioning_mode,
            "source_keyframe_index": int(manifest["start_keyframe_index"]),
            "end_keyframe_index": end_keyframe_index,
            "target_indices": [idx],
            "relative_position": relative_position,
            "used_caption": used_caption,
            "used_side_info": used_side_info,
            "mock": False,
            "notes": (
                f"diffusers WanImageToVideoPipeline — {notes_conditioning}{notes_caption}. "
                f"Generated an internal {n_frames}-frame Wan clip (segment span "
                f"[{start_frame_index}, {span_end}]) and mapped target index {idx} "
                f"(segment offset {idx - start_frame_index}) to clip position "
                f"{pos}/{n_frames - 1} by its actual temporal position, not its "
                "position in target_indices. side_infos accepted but NOT used "
                "for conditioning (known limitation — see module docstring). Reference "
                "wiring, NOT verified against real GPU output by this repo in every "
                "configuration — see docs/lgvsc_1b_worker_readiness.md."
            ),
        }
    # Report the checkpoint ACTUALLY loaded (may differ from args.model_id —
    # see the per-segment selection above) so main()'s result.json envelope
    # doesn't echo a stale/wrong model_id for bidirectional segments.
    return {"frames": frames_out, "metadata": metadata, "model_id": model_id}


def run_callable_backend(manifest: dict, manifest_dir: Path, args: argparse.Namespace) -> Dict:
    """Dynamically import and call a user-supplied adapter function — the
    recommended extension point for a real Open-Sora/Wan backend (see module
    docstring's "callable backend contract")."""
    if not args.backend_entrypoint:
        raise ValueError(
            "backend=callable requires --backend-entrypoint 'module.path:function_name'."
        )
    module_name, sep, func_name = args.backend_entrypoint.partition(":")
    if not sep or not module_name or not func_name:
        raise ValueError(
            f"--backend-entrypoint={args.backend_entrypoint!r} must be in "
            "'module.path:function_name' form."
        )
    import importlib
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        raise WorkerBackendUnavailableError(
            f"Could not import --backend-entrypoint module {module_name!r} in this "
            f"worker's Python environment: {exc}. Make sure the module (and whatever "
            "it itself imports, e.g. your real backend's package) is importable here."
        ) from exc
    try:
        fn = getattr(module, func_name)
    except AttributeError as exc:
        raise ValueError(
            f"Module {module_name!r} has no attribute {func_name!r} "
            f"(--backend-entrypoint={args.backend_entrypoint!r})."
        ) from exc

    raw = fn(manifest, manifest_dir, args)
    if not isinstance(raw, dict) or "frames" not in raw or "metadata" not in raw:
        got = type(raw).__name__ if not isinstance(raw, dict) else sorted(raw.keys())
        raise ValueError(
            f"--backend-entrypoint {args.backend_entrypoint!r} must return a dict "
            f"with 'frames' and 'metadata' keys (see this script's module docstring's "
            f"\"callable backend contract\") — got: {got!r}."
        )
    # Normalise keys to int (a user adapter may return str keys from JSON-ish code).
    return {
        "frames": {int(k): v for k, v in raw["frames"].items()},
        "metadata": {int(k): v for k, v in raw["metadata"].items()},
    }


_BACKENDS = {
    "mock": run_mock_backend,
    "svd": run_svd_backend,
    "wan": run_wan_backend,
    "callable": run_callable_backend,
}


def generate(manifest: dict, manifest_dir: Path, args: argparse.Namespace) -> Dict:
    if args.backend not in _BACKENDS:
        raise ValueError(f"Unknown --backend={args.backend!r}; expected one of {sorted(_BACKENDS)}.")
    result = _BACKENDS[args.backend](manifest, manifest_dir, args)
    target_indices = {int(i) for i in manifest["target_indices"]}
    got_frames = set(result.get("frames", {}))
    got_metadata = set(result.get("metadata", {}))
    if got_frames != target_indices or got_metadata != target_indices:
        raise ValueError(
            f"backend={args.backend!r} returned frames for {sorted(got_frames)!r} and "
            f"metadata for {sorted(got_metadata)!r}, but manifest target_indices are "
            f"{sorted(target_indices)!r} — a backend must return exactly the requested "
            "frames, no more, no fewer."
        )
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LGVSC/ETRI 1B out-of-process segment generation worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True, help="Path to the JSON request manifest.")
    p.add_argument("--output-dir", required=True, help="Where to write frame PNGs + result.json/error.json.")
    p.add_argument("--backend", choices=sorted(_BACKENDS), default="mock")
    p.add_argument("--backend-entrypoint", default=None,
                    help="'module.path:function_name' — required for --backend callable.")
    p.add_argument("--model-id", default=None, help="HF model id / local path (backend-specific).")
    p.add_argument("--device", default="cpu", help="e.g. cpu, cuda:0.")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--height", type=int, default=None, help="Override output frame height.")
    p.add_argument("--width", type=int, default=None, help="Override output frame width.")
    p.add_argument("--num-inference-steps", type=int, default=None, help="Backend-specific (e.g. svd).")
    p.add_argument("--decode-chunk-size", type=int, default=None, help="Backend-specific (svd VRAM knob).")
    p.add_argument("--extra-json", default=None,
                    help="JSON object string of extra backend-specific kwargs (forwarded as-is; "
                         "a callable backend can read it off `args.extra_json`).")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        manifest_path = Path(args.manifest)
        manifest = load_manifest(manifest_path)
        result = generate(manifest, manifest_path.parent, args)

        frame_files: Dict[str, str] = {}
        metadata_out: Dict[str, Dict] = {}
        for idx in sorted(result["frames"]):
            fp = output_dir / f"frame_{idx:05d}.png"
            save_frame_image(result["frames"][idx], fp)
            frame_files[str(idx)] = fp.name
            metadata_out[str(idx)] = result["metadata"][idx]

        result_json = {
            "status": "ok",
            "segment_id": manifest["segment_id"],
            "backend": args.backend,
            # Prefer the backend's own report of which checkpoint it actually
            # loaded (e.g. run_wan_backend may pick a different one than
            # args.model_id per segment) — fall back to the CLI arg for
            # backends that don't report this (mock/svd/callable).
            "model_id": result.get("model_id", args.model_id),
            "device": args.device,
            "seed": args.seed,
            "duration_sec": round(time.time() - t0, 3),
            "target_indices": [int(i) for i in manifest["target_indices"]],
            "frames": frame_files,
            "metadata": metadata_out,
        }
        (output_dir / "result.json").write_text(json.dumps(result_json, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:  # noqa: BLE001 — always report via error.json, never a bare crash
        error_json = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "error.json").write_text(json.dumps(error_json, indent=2), encoding="utf-8")
        print(f"lgvsc_generate_worker: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
