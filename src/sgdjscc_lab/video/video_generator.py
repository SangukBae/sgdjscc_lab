"""video/video_generator.py – Start-only / bidirectional video generation backend
interface (ETRI 3차 step 5 + 4차 step 6), extended with a Rx-legal
**segment-level** generation contract (ETRI 후속 1단계 step 1A) and an
**out-of-process worker backend** for real model integration (step 1B).

Scope note (read before touching this file)
--------------------------------------------
This module's job is **structural**: give ``TemporalPipeline`` a real third
branch (``reuse`` / ``recompute`` / ``generate``) that a future strong video
generator (SVD, Open-Sora, …) can plug into, and prove the branch — in both
its start-only (3차) and bidirectional (4차) conditioning modes — works
end-to-end with cheap placeholder backends. It is explicitly **not** trying to
produce good-looking generated frames or to demonstrate that bidirectional
conditioning reduces drift/flicker; that quality/comparison claim needs a real
generator and real GPU validation. Every mock backend here
(:class:`CopyGenerator`, :class:`InterpolationGenerator`,
:class:`BidirectionalInterpolationGenerator`) is verbatim copy or linear
blend, never learned generation, always ``mock=True``. Real backend
integration now has a concrete path — :class:`ExternalSegmentWorkerGenerator`
(ETRI 1B, below) — but that class only builds the plumbing (subprocess IPC,
manifest/result contract, error handling); it does not itself install
Open-Sora/Wan/diffusers, download weights, run real GPU generation, or claim
anything about generation quality — that is the explicit scope of the actual
GPU validation run, done separately (see docs/lgvsc_1b_worker_readiness.md).
See docs/etri_strategy.md "후속 딥러닝 4단계" and docs/video_extension_lgvsc.md
6.0-a/6.2 for the full staging (1A/1B/1C).

Segment-level contract (ETRI 1A)
---------------------------------
docs/video_extension_lgvsc.md 6.2 "주의 1" observes that a real Rx-side world
model (LGVSC's SKIM+SFA / SKEM+DSA) is not a per-frame function — it is a
**GOP/segment-level** contract: ``(start keyframe, [end keyframe], segment
captions/packets/side-info, segment length) → frames``. 1A introduces that
contract as :class:`SegmentGenerationRequest` / :class:`SegmentGenerationResult`
/ :meth:`VideoGenerator.generate_segment`, alongside (not replacing) the
existing per-frame :class:`GenerationRequest` / :meth:`VideoGenerator.generate`.
The base class's :meth:`VideoGenerator.generate_segment` default implementation
simply loops over ``generate()`` once per target frame — this keeps
:class:`CopyGenerator` / :class:`InterpolationGenerator` /
:class:`BidirectionalInterpolationGenerator` working unmodified under the new
contract (they never override ``generate_segment``). ``TemporalPipeline`` now
calls ``generate_segment()`` **once per GOP** (batching its own call site) for
every GOP that has at least one ``generate``-decision frame; a future 1B
backend that can truly process a whole segment in one model call overrides
``generate_segment`` directly without ``TemporalPipeline`` needing to change
again. 1A does not implement or claim any such real batched generation —
only the contract and its mock-compatible default fallback.

External worker backend (ETRI 1B)
-----------------------------------
:class:`ExternalSegmentWorkerGenerator` is that 1B backend: it overrides
``generate_segment`` directly (no per-frame looping) and delegates the actual
generation to ``scripts/lgvsc_generate_worker.py`` running as a **subprocess
in a different Python environment** (e.g. a separate ``lgvsc_gen`` conda env
with ``diffusers``/Open-Sora/Wan installed) — this process/environment
(``ptest``) never needs those heavy, version-fragile dependencies. Request →
response crosses the process boundary purely as files (JSON manifest + PNG
keyframe images in; PNG frame images + JSON metadata out), so the contract
this class honours is identical for a `mock` worker backend (dependency-free,
what this repo's tests exercise) and a real one — only the subprocess's own
``--backend`` flag differs. See the worker script's module docstring for the
manifest/result/error JSON schema, and docs/lgvsc_1b_worker_readiness.md for
how to point this at a real model, the conda environment split, and what
GPU-side validation is still the user's responsibility (1B builds the
plumbing; it does not itself verify real generation quality).

Rx-legal boundary (important)
------------------------------
A real Rx-side generator may only condition on what the receiver actually has:
keyframe reconstructions (start, and — in bidirectional mode — end), the
semantic packet/caption side info, and previously reconstructed frames. It
must **never** condition on the original target frame — that would be
evaluation-time cheating (the receiver does not have the un-transmitted
ground truth in a real deployment). ``GenerationRequest`` keeps
``reference_target_frame`` as a separate, explicitly-named field for this
reason: :class:`InterpolationGenerator` only reads it when
``allow_ground_truth_reference=True`` is passed in (default False), and every
result produced through that path is tagged ``mock=True`` with a note in its
metadata making the leakage explicit. Treat any report built with that flag on
as a structural/test artifact, never as an evaluation number. Note that
``end_keyframe_recon`` is *not* subject to this boundary — a bidirectional
receiver genuinely has already decoded the end keyframe too (it is a real
Rx-side reconstruction, just like the start keyframe), so bidirectional
conditioning on it is Rx-legal by construction.

``SegmentGenerationRequest`` deliberately has **no** ground-truth/original-frame
field at all — not even an opt-in test-only one. Only the frame-level
``GenerationRequest`` (a pre-1A, still-supported API) carries the guarded
``reference_target_frame`` escape hatch for offline mock/test use. The
segment-level contract is the one a real 1B backend will actually be driven
through, so it never gets a leakage path to inherit in the first place — see
``tests/test_video_generator.py::TestSegmentGenerationRequestRxLegal``.

Conditioning mode
-----------------
- **start_only** (3차, still the default): one start keyframe reconstruction
  conditions every generated inter-frame in its GOP. Implemented by
  :class:`CopyGenerator` / :class:`InterpolationGenerator`, both of which
  still reject any request carrying ``end_keyframe_recon`` (via
  :func:`_check_request`) — that field is start-only-backend-illegal, not
  merely unused.
- **bidirectional** (4차): both the start keyframe and the *next* GOP's
  keyframe condition the generated frame, blended by the target frame's
  relative position between them. Implemented by
  :class:`BidirectionalInterpolationGenerator`. Only reachable when
  ``TemporalPipeline`` is explicitly run in bidirectional mode (see
  ``video/temporal_pipeline.py``'s ``conditioning_mode`` parameter) — the
  default remains start_only, so existing 3차 results are unaffected.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

CONDITIONING_MODE_START_ONLY = "start_only"
# ETRI 4차 implemented mock bidirectional conditioning via
# BidirectionalInterpolationGenerator. Real learned bidirectional backends
# remain a future extension point.
CONDITIONING_MODE_BIDIRECTIONAL = "bidirectional"


@dataclass
class GenerationRequest:
    """Input to a :class:`VideoGenerator` backend's ``generate()``.

    Parameters
    ----------
    start_keyframe_recon:
        The GOP's start-keyframe reconstruction (already produced by the
        normal SGD-JSCC forward pass) — the only *pixel* evidence a real
        Rx-side generator may condition on in 3차's start-only mode.
    start_keyframe_index / target_index:
        Frame indices (into the full sequence) of the conditioning keyframe
        and the frame being generated.
    segment_context:
        Optional caller-supplied context about the enclosing GOP/segment
        (e.g. ``{"segment_id": ..., "frame_indices": [...]}}``). Not
        interpreted by the backends here; kept for a future segment-level
        (batched) generator.
    caption:
        The target frame's packet caption, if any (side info, not pixels).
    packet:
        The target frame's semantic packet (side info: objects/relations/
        attributes/scene — never pixels of the un-transmitted frame).
    side_info:
        Free-form extra side info (e.g. the semantic-delta / motion dicts
        already computed by ``TemporalPipeline``).
    reference_prev_recon:
        The immediately-previous frame's reconstruction (any decision type).
        Rx-legal: it is something the receiver has already produced.
    reference_target_frame:
        The *original* target frame. **Test/mock only** — see the module
        docstring's Rx-legal boundary note. A real backend must ignore this
        field; only reads it when the caller explicitly opts in
        (``allow_ground_truth_reference=True``).
    end_keyframe_recon / end_keyframe_index:
        The *next* GOP's keyframe reconstruction/index (ETRI 4차 bidirectional
        conditioning) — Rx-legal (see the module docstring). ``None`` when
        conditioning is start-only (3차 default), or when bidirectional mode
        has no following keyframe available (last GOP in the sequence; see
        ``BidirectionalInterpolationGenerator``'s ``missing_end_policy``).
        Start-only backends (:class:`CopyGenerator`, :class:`InterpolationGenerator`)
        still reject any request with ``end_keyframe_recon`` set — it is
        start-only-illegal, not merely unused, so a caller misconfiguration is
        caught immediately rather than silently ignored.
    """

    start_keyframe_recon: torch.Tensor
    start_keyframe_index: int
    target_index: int
    segment_context: Optional[Dict] = None
    caption: Optional[str] = None
    packet: Optional[Dict] = None
    side_info: Optional[Dict] = None
    reference_prev_recon: Optional[torch.Tensor] = None
    reference_target_frame: Optional[torch.Tensor] = None
    # ETRI 4차 bidirectional extension — see module docstring's Rx-legal note.
    end_keyframe_recon: Optional[torch.Tensor] = None
    end_keyframe_index: Optional[int] = None


@dataclass
class GenerationMetadata:
    """JSON/dict-serialisable record of how one frame was generated.

    ``end_keyframe_index`` and ``relative_position`` are always present in
    ``to_dict()`` (defaulting to ``None``) even for start-only results, so a
    consumer (CSV writer, comparison pipeline) can treat every generation
    record with one uniform schema regardless of conditioning mode — the same
    convention already used for the motion-gate columns in
    ``FrameRecord.to_log()``.
    """

    backend: str
    conditioning_mode: str
    source_keyframe_index: int
    target_indices: List[int]
    used_caption: bool
    used_side_info: bool
    mock: bool
    notes: str = ""
    # ETRI 4차 bidirectional fields — None for start-only (3차) results.
    end_keyframe_index: Optional[int] = None
    relative_position: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "backend": self.backend,
            "conditioning_mode": self.conditioning_mode,
            "source_keyframe_index": self.source_keyframe_index,
            "end_keyframe_index": self.end_keyframe_index,
            "target_indices": list(self.target_indices),
            "relative_position": self.relative_position,
            "used_caption": self.used_caption,
            "used_side_info": self.used_side_info,
            "mock": self.mock,
            "notes": self.notes,
        }


@dataclass
class GenerationResult:
    frame: torch.Tensor
    metadata: GenerationMetadata


def _check_request(request: GenerationRequest) -> None:
    """Reject 4차-only request shapes so a caller finds out immediately."""
    if request.end_keyframe_recon is not None:
        raise NotImplementedError(
            "Bidirectional (start+end keyframe) conditioning is reserved for "
            "ETRI 4차 — see docs/etri_strategy.md 순서 6. 3차 only implements "
            "start-only conditioning; GenerationRequest.end_keyframe_recon must "
            "stay None."
        )


def _build_metadata(backend: str, request: GenerationRequest, mock: bool, notes: str) -> GenerationMetadata:
    return GenerationMetadata(
        backend=backend,
        conditioning_mode=CONDITIONING_MODE_START_ONLY,
        source_keyframe_index=int(request.start_keyframe_index),
        target_indices=[int(request.target_index)],
        used_caption=bool(request.caption),
        used_side_info=bool(request.side_info),
        mock=mock,
        notes=notes,
    )


@dataclass
class SegmentGenerationRequest:
    """Input to a :class:`VideoGenerator` backend's ``generate_segment()``
    (ETRI 후속 1단계 step 1A) — the GOP/segment-level counterpart of
    :class:`GenerationRequest`.

    A "segment" here is exactly one ``TemporalPipeline`` GOP (the same unit
    ``video/segment.py::SegmentRecord`` groups), so ``segment_id`` matches the
    corresponding ``SegmentRecord.segment_id`` for the same run. Only the
    frames the pipeline actually decided to *generate* are listed in
    ``target_indices`` — a GOP may (and often will) mix ``reuse``/
    ``recompute_*``/``generate`` decisions; the ones already resolved by reuse
    or recompute are simply not part of this request.

    Parameters
    ----------
    segment_id:
        GOP/segment index (0-based, in GOP order — matches
        ``SegmentRecord.segment_id`` built from the same ``TemporalPipeline``
        run).
    start_frame_index / end_frame_index:
        The segment's full frame span: this GOP's keyframe index and the
        index of its last frame (last inter-frame, or the keyframe itself if
        the GOP has no inter-frames). This is the segment *layout*, not the
        generation target list — ``end_frame_index`` may exceed the last
        ``target_indices`` entry when the GOP mixes decisions.
    target_indices:
        Ascending, unique frame indices *within* ``[start_frame_index,
        end_frame_index]`` that this backend call must generate — in the
        order results are expected back in ``SegmentGenerationResult``.
    start_keyframe_recon / start_keyframe_index:
        This segment's start-keyframe reconstruction/index — the same
        Rx-legal pixel evidence :class:`GenerationRequest` uses.
    end_keyframe_recon / end_keyframe_index:
        The *next* GOP's keyframe reconstruction/index (ETRI 4차 bidirectional
        conditioning carried over to the segment contract) — Rx-legal (see the
        module docstring), ``None`` in start-only mode or when there is no
        following keyframe (last GOP in the sequence).
    fps:
        Source video frame rate, when known (``None`` for image/synthetic
        sequences with no natural playback rate). Combined with
        ``start_frame_index``/``end_frame_index`` this gives a real backend
        the segment's wall-clock duration, not just its frame count.
    captions / packets / side_infos:
        Optional per-target-frame side info, one entry per ``target_indices``
        position (lengths must match when given): ``captions[i]`` is the
        target frame's packet caption (or ``None``); ``packets[i]`` is its
        full semantic packet dict; ``side_infos[i]`` is the free-form extra
        side info ``TemporalPipeline`` already computed (semantic-delta /
        motion dicts) for that frame. All side info — never pixels of the
        un-transmitted target frame.
    reference_prev_recon:
        The reconstruction of the frame immediately before ``target_indices[0]``
        at the time this segment request was built (Rx-legal: whatever the
        receiver had actually already reconstructed there — a keyframe, a
        reused frame, or a recomputed one). Kept as a single "segment-entry"
        value for backends that only want one prior-context frame and do not
        care about per-target precision; see ``reference_prev_recons`` for the
        precise per-target version. When ``reference_prev_recons`` is also
        given, its first entry takes precedence for ``target_indices[0]`` and
        this field is redundant with it.
    reference_prev_recons:
        Per-target overrides, one entry per ``target_indices`` position
        (length must match when given): ``reference_prev_recons[i]`` is the
        *actual* reconstruction of the frame immediately before
        ``target_indices[i]`` — Rx-legal, whatever the receiver had really
        already reconstructed there — whenever that frame was resolved
        independently of this segment call (a keyframe, a reused frame, a
        recomputed frame, or a frame outside this GOP entirely). An entry is
        ``None`` exactly when ``target_indices[i] - 1 == target_indices[i-1]``
        — i.e. the immediately-preceding frame is *itself* one of this
        segment's own generate targets, so its reconstruction is not known
        until this very call produces it. The base class's default
        ``generate_segment`` fallback resolves a ``None`` entry by chaining to
        the previously generated target's own output within the same call
        (see :meth:`VideoGenerator.generate_segment`) — this is what makes a
        GOP that mixes ``generate`` with ``reuse``/``recompute`` frames
        reference the *correct* prior reconstruction at every target, not
        just the first one. ``None`` (the whole list, not just an entry) when
        the caller does not track per-target precision — callers should
        prefer supplying this over relying on ``reference_prev_recon`` alone
        whenever more than one target is requested. Real segment-level
        backends (1B+) are free to ignore both fields entirely and condition
        on the two keyframes only.

    Rx-legal boundary
    -----------------
    Deliberately has **no** original/ground-truth target-frame field — see the
    module docstring. A caller must never construct one from un-transmitted
    pixels.
    """

    segment_id: int
    start_frame_index: int
    end_frame_index: int
    target_indices: List[int]
    start_keyframe_recon: torch.Tensor
    start_keyframe_index: int
    end_keyframe_recon: Optional[torch.Tensor] = None
    end_keyframe_index: Optional[int] = None
    fps: Optional[float] = None
    captions: Optional[List[Optional[str]]] = None
    packets: Optional[List[Optional[Dict]]] = None
    side_infos: Optional[List[Optional[Dict]]] = None
    reference_prev_recon: Optional[torch.Tensor] = None
    reference_prev_recons: Optional[List[Optional[torch.Tensor]]] = None

    @property
    def segment_length(self) -> int:
        """Number of frame slots spanned by this segment (inclusive)."""
        return int(self.end_frame_index) - int(self.start_frame_index) + 1


@dataclass
class SegmentGenerationResult:
    """Output of a :class:`VideoGenerator` backend's ``generate_segment()``.

    ``target_indices``/``frames``/``metadata`` are parallel lists in the same
    order as the originating request's ``target_indices`` — use
    :meth:`frame_for` / :meth:`metadata_for` rather than assuming a caller
    keeps the ordering invariant itself. Each ``metadata`` entry is a plain
    per-frame :class:`GenerationMetadata` (the same type
    :class:`GenerationResult` uses) so existing consumers
    (``FrameRecord.generation``, ``segment.py``'s generation summary) need no
    schema change to accept segment-produced frames.
    """

    segment_id: int
    target_indices: List[int]
    frames: List[torch.Tensor]
    metadata: List[GenerationMetadata]

    def frame_for(self, index: int) -> torch.Tensor:
        return self.frames[self.target_indices.index(index)]

    def metadata_for(self, index: int) -> GenerationMetadata:
        return self.metadata[self.target_indices.index(index)]


def validate_segment_request(request: SegmentGenerationRequest) -> None:
    """Raise ``ValueError`` on a structurally invalid
    :class:`SegmentGenerationRequest` (bad ordering/bounds/list lengths).

    Deliberately does not import ``torch`` for shape checks here — the
    request's own tensors are trusted at construction time; :func:`
    validate_segment_result` is what catches a misbehaving backend's output.
    """
    start, end = int(request.start_frame_index), int(request.end_frame_index)
    if end < start:
        raise ValueError(
            f"SegmentGenerationRequest.end_frame_index ({end}) must be >= "
            f"start_frame_index ({start})."
        )
    targets = list(request.target_indices)
    if targets != sorted(targets):
        raise ValueError(f"target_indices must be ascending; got {targets!r}.")
    if len(set(targets)) != len(targets):
        raise ValueError(f"target_indices must be unique; got {targets!r}.")
    for t in targets:
        if not (start <= t <= end):
            raise ValueError(
                f"target_index {t} is outside the segment span [{start}, {end}] "
                f"(segment_id={request.segment_id})."
            )
    n = len(targets)
    for name, seq in (
        ("captions", request.captions),
        ("packets", request.packets),
        ("side_infos", request.side_infos),
        ("reference_prev_recons", request.reference_prev_recons),
    ):
        if seq is not None and len(seq) != n:
            raise ValueError(
                f"{name} has length {len(seq)} but target_indices has length "
                f"{n} (segment_id={request.segment_id})."
            )
    if (request.end_keyframe_recon is None) != (request.end_keyframe_index is None):
        raise ValueError(
            "end_keyframe_recon and end_keyframe_index must both be set or "
            f"both be None (segment_id={request.segment_id})."
        )


def validate_segment_result(request: SegmentGenerationRequest, result: SegmentGenerationResult) -> None:
    """Raise ``ValueError`` when a backend's :class:`SegmentGenerationResult`
    does not honour its :class:`SegmentGenerationRequest` — wrong
    ``segment_id``, wrong frame count, wrong/reordered indices, a mismatched
    tensor shape, or metadata that doesn't actually describe the frame it's
    attached to (wrong ``target_indices``/``source_keyframe_index``, wrong
    type, or an unknown ``conditioning_mode``). Every real (1B+) or mock
    backend result must pass this before ``TemporalPipeline`` projects it onto
    ``FrameRecord``/``SegmentRecord`` — a backend that silently returns
    frames/metadata for the wrong request must fail loudly here rather than
    have its mislabelled output pass through unnoticed.
    """
    if result.segment_id != request.segment_id:
        raise ValueError(
            f"SegmentGenerationResult.segment_id={result.segment_id!r} does not "
            f"match the request's segment_id={request.segment_id!r}. A backend "
            "must return the result for the segment it was asked to generate."
        )
    expected = list(request.target_indices)
    got = list(result.target_indices)
    if got != expected:
        raise ValueError(
            f"SegmentGenerationResult.target_indices={got!r} does not match "
            f"the request's target_indices={expected!r} (segment_id="
            f"{request.segment_id}). A backend must return exactly the "
            "requested frames, in the requested order."
        )
    if len(result.frames) != len(expected) or len(result.metadata) != len(expected):
        raise ValueError(
            f"SegmentGenerationResult for segment_id={request.segment_id} has "
            f"{len(result.frames)} frames / {len(result.metadata)} metadata "
            f"entries but {len(expected)} target_indices were requested."
        )
    expected_shape = tuple(request.start_keyframe_recon.shape)
    for idx, frame, meta in zip(expected, result.frames, result.metadata):
        if tuple(frame.shape) != expected_shape:
            raise ValueError(
                f"SegmentGenerationResult frame for index {idx} has shape "
                f"{tuple(frame.shape)}, expected {expected_shape} (matching "
                f"start_keyframe_recon; segment_id={request.segment_id})."
            )
        if not isinstance(meta, GenerationMetadata):
            raise ValueError(
                f"SegmentGenerationResult metadata for index {idx} is a "
                f"{type(meta).__name__}, expected GenerationMetadata "
                f"(segment_id={request.segment_id})."
            )
        if list(meta.target_indices) != [idx]:
            raise ValueError(
                f"SegmentGenerationResult metadata for index {idx} has "
                f"target_indices={list(meta.target_indices)!r}, expected [{idx}] "
                f"(segment_id={request.segment_id}). Each per-frame metadata "
                "entry must describe exactly the one frame it's attached to."
            )
        if int(meta.source_keyframe_index) != int(request.start_keyframe_index):
            raise ValueError(
                f"SegmentGenerationResult metadata for index {idx} has "
                f"source_keyframe_index={meta.source_keyframe_index!r}, expected "
                f"{request.start_keyframe_index!r} (segment_id={request.segment_id})."
            )
        if meta.conditioning_mode not in (CONDITIONING_MODE_START_ONLY, CONDITIONING_MODE_BIDIRECTIONAL):
            raise ValueError(
                f"SegmentGenerationResult metadata for index {idx} has an unknown "
                f"conditioning_mode={meta.conditioning_mode!r} (segment_id="
                f"{request.segment_id}); expected one of "
                f"{(CONDITIONING_MODE_START_ONLY, CONDITIONING_MODE_BIDIRECTIONAL)!r}."
            )


class VideoGenerator:
    """Backend interface for the start-only generate branch.

    Subclasses implement ``generate(request) -> GenerationResult``. The base
    class only enforces the Rx-legal / conditioning-mode guard so every
    backend gets it for free.

    ``generate_segment(request) -> SegmentGenerationResult`` (ETRI 1A) is the
    GOP/segment-level counterpart — the base class implementation below simply
    calls ``generate()`` once per target frame in order. Each target's
    ``reference_prev_recon`` is resolved as: the corresponding
    ``request.reference_prev_recons[i]`` entry when the caller supplied one
    (the true immediately-preceding reconstruction, precise even for
    non-contiguous targets — see :class:`SegmentGenerationRequest`'s
    docstring); otherwise the previous target's own just-generated frame,
    chained within this call (falling back to ``request.reference_prev_recon``
    for the first target). This keeps every backend defined in this module
    working under the new contract with zero changes. A future 1B backend
    whose underlying model can process a whole segment in a single call
    should override ``generate_segment`` directly instead of relying on this
    per-frame fallback loop.
    """

    backend_name = "base"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError

    def generate_segment(self, request: SegmentGenerationRequest) -> SegmentGenerationResult:
        validate_segment_request(request)
        frames: List[torch.Tensor] = []
        metadata: List[GenerationMetadata] = []
        chained_prev_recon = request.reference_prev_recon
        for i, target_index in enumerate(request.target_indices):
            override = request.reference_prev_recons[i] if request.reference_prev_recons is not None else None
            reference_prev_recon = override if override is not None else chained_prev_recon
            frame_request = GenerationRequest(
                start_keyframe_recon=request.start_keyframe_recon,
                start_keyframe_index=request.start_keyframe_index,
                target_index=target_index,
                segment_context={
                    "segment_id": request.segment_id,
                    "start_frame_index": request.start_frame_index,
                    "end_frame_index": request.end_frame_index,
                    "target_indices": list(request.target_indices),
                },
                caption=(request.captions[i] if request.captions is not None else None),
                packet=(request.packets[i] if request.packets is not None else None),
                side_info=(request.side_infos[i] if request.side_infos is not None else None),
                reference_prev_recon=reference_prev_recon,
                end_keyframe_recon=request.end_keyframe_recon,
                end_keyframe_index=request.end_keyframe_index,
            )
            result = self.generate(frame_request)
            frames.append(result.frame)
            metadata.append(result.metadata)
            chained_prev_recon = result.frame
        return SegmentGenerationResult(
            segment_id=request.segment_id,
            target_indices=list(request.target_indices),
            frames=frames,
            metadata=metadata,
        )


class CopyGenerator(VideoGenerator):
    """Mock backend: returns the start keyframe reconstruction unchanged.

    This is the simplest possible placeholder for a real generator — it proves
    the 3-way branch wiring (decision routing, metadata, storage) without
    claiming any generation quality. ``mock=True`` always.
    """

    backend_name = "copy"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        _check_request(request)
        frame = request.start_keyframe_recon.clone()
        meta = _build_metadata(
            self.backend_name, request, mock=True,
            notes="copy backend: returns the start keyframe reconstruction "
                  "unchanged (placeholder for a real generator, not learned generation).",
        )
        return GenerationResult(frame=frame, metadata=meta)


class InterpolationGenerator(VideoGenerator):
    """Mock backend: linear blend of the start keyframe recon with a reference.

    The reference is, in priority order:
    1. ``request.reference_prev_recon`` (Rx-legal: the previous reconstruction).
    2. ``request.reference_target_frame``, but **only** when
       ``allow_ground_truth_reference=True`` was passed to the constructor —
       this is a test/mock-only path (see the module docstring's Rx-legal
       boundary note) and is always tagged in the result's ``notes``.
    3. The start keyframe reconstruction itself (degenerates to a copy) when
       neither reference is available.

    Parameters
    ----------
    alpha:
        Blend weight for the reference frame; ``0`` = pure keyframe copy,
        ``1`` = pure reference. Default 0.5.
    allow_ground_truth_reference:
        Enables the test/mock-only ground-truth reference path. Default False
        (Rx-legal by default).
    """

    backend_name = "interpolation"

    def __init__(self, alpha: float = 0.5, allow_ground_truth_reference: bool = False) -> None:
        self.alpha = float(alpha)
        self.allow_ground_truth_reference = bool(allow_ground_truth_reference)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        _check_request(request)
        used_ground_truth = False
        ref = request.reference_prev_recon
        if ref is None and self.allow_ground_truth_reference and request.reference_target_frame is not None:
            ref = request.reference_target_frame
            used_ground_truth = True

        notes = (
            "interpolation backend (TEST/MOCK ONLY): linear blend of the start "
            "keyframe reconstruction with a reference frame; not learned generation."
        )
        if ref is None:
            frame = request.start_keyframe_recon.clone()
            notes += " No reference available — degenerated to a keyframe copy."
        else:
            a = min(max(self.alpha, 0.0), 1.0)
            frame = (1.0 - a) * request.start_keyframe_recon + a * ref
            if used_ground_truth:
                notes += (
                    " reference=ground-truth target frame — NEVER legal for a real "
                    "Rx-side evaluation; only enabled via allow_ground_truth_reference=True."
                )
            else:
                notes += " reference=previous reconstruction (Rx-legal)."

        meta = _build_metadata(self.backend_name, request, mock=True, notes=notes)
        return GenerationResult(frame=frame, metadata=meta)


MISSING_END_POLICY_ERROR = "error"
MISSING_END_POLICY_FALLBACK_START_ONLY = "fallback_start_only"
_MISSING_END_POLICIES = (MISSING_END_POLICY_ERROR, MISSING_END_POLICY_FALLBACK_START_ONLY)


class BidirectionalInterpolationGenerator(VideoGenerator):
    """Mock backend (ETRI 4차): linear blend of the start and end keyframe
    reconstructions, weighted by the target frame's relative position between
    them.

    Both references are Rx-legal (see the module docstring) — this backend
    conditions only on two keyframe reconstructions the receiver has actually
    decoded, never on the un-transmitted target frame. It is still a mock:
    the blend is not learned generation, so ``mock=True`` always.

    ``relative_position = (target_index - start_keyframe_index) /
    (end_keyframe_index - start_keyframe_index)``, clamped to ``[0, 1]``.

    Parameters
    ----------
    missing_end_policy:
        ``"error"`` (default) — raise ``ValueError`` when the request has no
        usable end keyframe (``end_keyframe_recon``/``end_keyframe_index`` is
        ``None``, e.g. the last GOP in a sequence has no following keyframe)
        or when ``target_index`` falls outside ``[start_keyframe_index,
        end_keyframe_index]``.
        ``"fallback_start_only"`` — degrade to a plain start-keyframe copy
        instead of raising; the returned metadata reports
        ``conditioning_mode="start_only"`` and the fallback is recorded in
        ``notes`` so it is never silently indistinguishable from a real
        bidirectional result.
    """

    backend_name = "bidirectional_interpolation"

    def __init__(self, missing_end_policy: str = MISSING_END_POLICY_ERROR) -> None:
        if missing_end_policy not in _MISSING_END_POLICIES:
            raise ValueError(
                f"Unknown missing_end_policy={missing_end_policy!r}; expected one of "
                f"{_MISSING_END_POLICIES!r}."
            )
        self.missing_end_policy = missing_end_policy

    def generate(self, request: GenerationRequest) -> GenerationResult:
        missing_end = request.end_keyframe_recon is None or request.end_keyframe_index is None
        out_of_range = False
        relative_position: Optional[float] = None

        if not missing_end:
            span = int(request.end_keyframe_index) - int(request.start_keyframe_index)
            if span <= 0 or not (
                request.start_keyframe_index <= request.target_index <= request.end_keyframe_index
            ):
                out_of_range = True
            else:
                relative_position = (request.target_index - request.start_keyframe_index) / span

        if missing_end or out_of_range:
            reason = "end keyframe unavailable" if missing_end else "target_index outside [start, end] range"
            if self.missing_end_policy == MISSING_END_POLICY_ERROR:
                raise ValueError(
                    f"BidirectionalInterpolationGenerator: {reason} for target_index="
                    f"{request.target_index} (start_keyframe_index={request.start_keyframe_index}, "
                    f"end_keyframe_index={request.end_keyframe_index}). Set "
                    "video_generator.bidirectional_missing_end_policy: fallback_start_only "
                    "to degrade instead of raising."
                )
            # fallback_start_only: degenerate to a plain start-keyframe copy,
            # reporting the ACTUAL conditioning used (start_only), not bidirectional.
            frame = request.start_keyframe_recon.clone()
            notes = (
                "bidirectional_interpolation backend (TEST/MOCK ONLY): "
                f"{reason}; bidirectional_missing_end_policy=fallback_start_only "
                "→ degenerated to a start-keyframe copy."
            )
            meta = GenerationMetadata(
                backend=self.backend_name,
                conditioning_mode=CONDITIONING_MODE_START_ONLY,
                source_keyframe_index=int(request.start_keyframe_index),
                end_keyframe_index=(
                    None if request.end_keyframe_index is None else int(request.end_keyframe_index)
                ),
                target_indices=[int(request.target_index)],
                relative_position=None,
                used_caption=bool(request.caption),
                used_side_info=bool(request.side_info),
                mock=True,
                notes=notes,
            )
            return GenerationResult(frame=frame, metadata=meta)

        a = min(max(relative_position, 0.0), 1.0)
        frame = (1.0 - a) * request.start_keyframe_recon + a * request.end_keyframe_recon
        notes = (
            "bidirectional_interpolation backend (TEST/MOCK ONLY): linear blend of "
            "the start and end keyframe reconstructions, weighted by the target "
            "frame's relative position between them; not learned generation."
        )
        meta = GenerationMetadata(
            backend=self.backend_name,
            conditioning_mode=CONDITIONING_MODE_BIDIRECTIONAL,
            source_keyframe_index=int(request.start_keyframe_index),
            end_keyframe_index=int(request.end_keyframe_index),
            target_indices=[int(request.target_index)],
            relative_position=float(relative_position),
            used_caption=bool(request.caption),
            used_side_info=bool(request.side_info),
            mock=True,
            notes=notes,
        )
        return GenerationResult(frame=frame, metadata=meta)


class SegmentWorkerError(RuntimeError):
    """Raised by :class:`ExternalSegmentWorkerGenerator` when the external
    worker subprocess fails — timeout, non-zero exit, missing/malformed
    ``result.json``, a frame-count/index mismatch, or a returned frame whose
    shape doesn't match ``start_keyframe_recon``. Always includes the work
    directory (and, when available, the worker's own ``error.json``/
    ``run.log``) so a failure is debuggable without re-running anything.
    """


def _default_worker_script() -> Path:
    """``src/sgdjscc_lab/video/video_generator.py`` → ``<repo_root>/scripts/
    lgvsc_generate_worker.py`` (ETRI 1B's out-of-process worker script)."""
    return Path(__file__).resolve().parents[3] / "scripts" / "lgvsc_generate_worker.py"


def _json_safe(value):
    """Recursively coerce *value* (packet/side-info dicts, which may contain
    numpy/torch scalars from ``motion_residual``/``semantic_delta``) into
    plain JSON-serialisable Python types."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes, int, float, bool)):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return value
    return value


def _metadata_from_dict(d: Dict) -> GenerationMetadata:
    """Reconstruct a :class:`GenerationMetadata` from the worker's per-frame
    JSON metadata dict (``lgvsc_generate_worker.py``'s ``result.json``),
    defaulting any optional field a minimal backend adapter omitted."""
    end_kf = d.get("end_keyframe_index")
    rel_pos = d.get("relative_position")
    return GenerationMetadata(
        backend=str(d.get("backend", "external_segment_worker")),
        conditioning_mode=str(d.get("conditioning_mode", CONDITIONING_MODE_START_ONLY)),
        source_keyframe_index=int(d.get("source_keyframe_index", 0)),
        target_indices=[int(i) for i in (d.get("target_indices") or [])],
        used_caption=bool(d.get("used_caption", False)),
        used_side_info=bool(d.get("used_side_info", False)),
        mock=bool(d.get("mock", False)),
        notes=str(d.get("notes", "")),
        end_keyframe_index=(int(end_kf) if end_kf is not None else None),
        relative_position=(float(rel_pos) if rel_pos is not None else None),
    )


class ExternalSegmentWorkerGenerator(VideoGenerator):
    """Segment-level backend that delegates real generation to an **external
    process**, typically in a different conda environment (ETRI 후속 1단계
    step 1B) — see ``scripts/lgvsc_generate_worker.py`` and
    ``docs/lgvsc_1b_worker_readiness.md``.

    Uses the exact same Rx-legal :class:`SegmentGenerationRequest` /
    :class:`SegmentGenerationResult` contract 1A defined; only
    ``generate_segment()`` is overridden (``generate()`` — the pre-1A
    per-frame API — is not supported and raises ``NotImplementedError``,
    since a real segment-level model is the entire point of this backend).

    Per call: writes the request to a temporary work directory as a JSON
    manifest + PNG keyframe image(s) (never the original/un-transmitted
    target frame — ``SegmentGenerationRequest`` has no such field to write in
    the first place), invokes ``worker_script`` as a subprocess using
    ``python_bin`` (so the heavy model dependencies never need to be
    importable from *this* process/environment), then reads back the
    generated frame PNGs + JSON metadata the worker wrote, then — since an
    external process is a trust boundary, not something to take on faith —
    runs the full 1A contract check (:func:`validate_segment_result`) on the
    parsed result before ever returning it. Every failure mode — timeout,
    non-zero exit, missing ``result.json``, wrong frame count/indices, wrong
    frame shape, or a contract violation only ``validate_segment_result``
    catches (wrong ``segment_id``, a metadata entry whose
    ``target_indices``/``source_keyframe_index`` doesn't match the frame it's
    attached to, an unknown ``conditioning_mode``) — raises
    :class:`SegmentWorkerError` with the work directory path so the
    run.log/error.json are easy to find. This happens unconditionally inside
    ``generate_segment()`` itself — a caller does not need to separately
    invoke ``validate_segment_result()`` for this backend to be safe.

    Parameters
    ----------
    python_bin:
        Path to the **other** environment's Python interpreter (e.g.
        ``/home/user/anaconda3/envs/lgvsc_gen/bin/python``) — never the
        interpreter this code itself is running under. Required; this class
        does not fall back to ``sys.executable``, since running a real
        backend's heavy deps under ``ptest`` is exactly what must not happen.
    worker_script:
        Path to ``lgvsc_generate_worker.py``. Defaults to the copy shipped in
        this repo (``scripts/lgvsc_generate_worker.py``) — override only if
        driving a modified/relocated copy.
    backend:
        Forwarded to the worker's ``--backend`` (``"mock"`` default —
        dependency-free; ``"svd"`` — best-effort ``diffusers`` reference
        wiring, unverified; ``"callable"`` — dynamic import of your own
        adapter via ``backend_entrypoint``). See the worker script's module
        docstring for the full contract.
    backend_entrypoint:
        ``"module.path:function_name"`` — required when ``backend="callable"``.
    model_id, device, dtype, seed, height, width, num_inference_steps,
    decode_chunk_size, extra_json:
        Forwarded verbatim to the worker's matching CLI flags. These are run
        configuration, not Rx-legal segment content, so — deliberately — they
        live on this backend class, not on ``SegmentGenerationRequest`` (which
        stays backend-neutral; see its docstring).
    timeout_sec:
        Optional subprocess timeout. ``None`` (default) = no timeout — a real
        video diffusion model can legitimately take minutes per segment.
    work_dir:
        Optional base directory for per-call temp directories (default:
        system temp dir). Each call gets its own
        ``lgvsc_seg<NNNNN>_<random>/`` subdirectory — never shared across
        calls, so concurrent segments never collide.
    cleanup_on_success:
        Delete the per-call work directory after a **successful** round-trip
        (default ``True``). Always kept on failure, regardless of this flag,
        so a ``SegmentWorkerError`` is always debuggable.
    extra_env:
        Optional extra environment variables merged into the subprocess's
        environment (e.g. ``{"HF_HOME": "...", "CUDA_VISIBLE_DEVICES": "0"}``).
    """

    backend_name = "external_segment_worker"

    def __init__(
        self,
        python_bin: str,
        worker_script: Optional[str] = None,
        backend: str = "mock",
        backend_entrypoint: Optional[str] = None,
        model_id: Optional[str] = None,
        device: str = "cpu",
        dtype: str = "fp32",
        seed: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        decode_chunk_size: Optional[int] = None,
        extra_json: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        work_dir: Optional[str] = None,
        cleanup_on_success: bool = True,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        if not python_bin:
            raise ValueError(
                "ExternalSegmentWorkerGenerator requires python_bin (the OTHER "
                "environment's interpreter, e.g. a lgvsc_gen conda env) — it must "
                "never default to this process's own interpreter."
            )
        self.python_bin = str(python_bin)
        self.worker_script = str(worker_script) if worker_script is not None else str(_default_worker_script())
        self.backend = str(backend)
        self.backend_entrypoint = backend_entrypoint
        self.model_id = model_id
        self.device = str(device)
        self.dtype = str(dtype)
        self.seed = seed
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.decode_chunk_size = decode_chunk_size
        self.extra_json = extra_json
        self.timeout_sec = timeout_sec
        self.work_dir = work_dir
        self.cleanup_on_success = bool(cleanup_on_success)
        self.extra_env = extra_env

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError(
            "ExternalSegmentWorkerGenerator only implements the segment-level "
            "generate_segment() contract (ETRI 1B) — it does not support the "
            "pre-1A per-frame generate() API."
        )

    def generate_segment(self, request: SegmentGenerationRequest) -> SegmentGenerationResult:
        validate_segment_request(request)
        work_dir = self._make_work_dir(request)
        try:
            manifest_path = self._write_manifest(request, work_dir)
            cmd = self._build_command(manifest_path, work_dir)
            self._run_worker(cmd, work_dir)
            result = self._read_result(request, work_dir)
            # _read_result() already catches frame-count/order/shape/missing-
            # metadata problems with a friendly SegmentWorkerError, but an
            # external process is a trust boundary — run the full 1A contract
            # check (segment_id, per-frame target_indices/source_keyframe_index,
            # metadata type, known conditioning_mode) here too, rather than
            # relying on whatever caller happens to invoke
            # validate_segment_result() afterwards (TemporalPipeline does, but
            # a direct generate_segment() call must not silently skip it).
            try:
                validate_segment_result(request, result)
            except ValueError as exc:
                raise SegmentWorkerError(
                    f"External segment worker returned a result that failed "
                    f"contract validation (segment_id={request.segment_id}): "
                    f"{exc} work_dir={work_dir}."
                ) from exc
        except Exception:
            logger.warning(
                "ExternalSegmentWorkerGenerator: segment_id=%s failed — work_dir kept "
                "at %s for inspection (run.log / error.json).",
                request.segment_id, work_dir,
            )
            raise
        if self.cleanup_on_success:
            shutil.rmtree(work_dir, ignore_errors=True)
        return result

    def _make_work_dir(self, request: SegmentGenerationRequest) -> Path:
        base = Path(self.work_dir) if self.work_dir else Path(tempfile.gettempdir())
        base.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f"lgvsc_seg{int(request.segment_id):05d}_", dir=str(base)))

    def _write_manifest(self, request: SegmentGenerationRequest, work_dir: Path) -> Path:
        from torchvision.utils import save_image as tv_save_image

        start_path = work_dir / "start_keyframe.png"
        tv_save_image(request.start_keyframe_recon.detach().cpu().float().clamp(0, 1), str(start_path))

        end_rel = None
        if request.end_keyframe_recon is not None:
            end_path = work_dir / "end_keyframe.png"
            tv_save_image(request.end_keyframe_recon.detach().cpu().float().clamp(0, 1), str(end_path))
            end_rel = end_path.name

        shape = tuple(request.start_keyframe_recon.shape)  # (1, C, H, W) or (C, H, W)
        n = len(request.target_indices)

        manifest = {
            "segment_id": int(request.segment_id),
            "start_frame_index": int(request.start_frame_index),
            "end_frame_index": int(request.end_frame_index),
            "segment_length": int(request.segment_length),
            "target_indices": [int(i) for i in request.target_indices],
            "start_keyframe_index": int(request.start_keyframe_index),
            "end_keyframe_index": (
                int(request.end_keyframe_index) if request.end_keyframe_index is not None else None
            ),
            "start_keyframe_image": start_path.name,
            "end_keyframe_image": end_rel,
            "fps": (float(request.fps) if request.fps is not None else None),
            "captions": (
                list(request.captions) if request.captions is not None else [None] * n
            ),
            "packets": (
                _json_safe(list(request.packets)) if request.packets is not None else [None] * n
            ),
            "side_infos": (
                _json_safe(list(request.side_infos)) if request.side_infos is not None else [None] * n
            ),
            "run_config": {
                "seed": self.seed,
                "model_id": self.model_id,
                "device": self.device,
                "dtype": self.dtype,
                "height": self.height if self.height is not None else shape[-2],
                "width": self.width if self.width is not None else shape[-1],
            },
        }
        manifest_path = work_dir / "manifest.json"
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except (TypeError, ValueError) as exc:
            raise SegmentWorkerError(
                f"Could not JSON-serialise the manifest for segment_id="
                f"{request.segment_id} (a packet/side_info value is not "
                f"JSON-safe): {exc}. work_dir={work_dir}."
            ) from exc
        return manifest_path

    def _build_command(self, manifest_path: Path, work_dir: Path) -> List[str]:
        cmd = [
            self.python_bin, self.worker_script,
            "--manifest", str(manifest_path),
            "--output-dir", str(work_dir),
            "--backend", self.backend,
            "--device", self.device,
            "--dtype", self.dtype,
        ]
        if self.model_id is not None:
            cmd += ["--model-id", str(self.model_id)]
        if self.backend_entrypoint is not None:
            cmd += ["--backend-entrypoint", str(self.backend_entrypoint)]
        if self.seed is not None:
            cmd += ["--seed", str(int(self.seed))]
        if self.height is not None:
            cmd += ["--height", str(int(self.height))]
        if self.width is not None:
            cmd += ["--width", str(int(self.width))]
        if self.num_inference_steps is not None:
            cmd += ["--num-inference-steps", str(int(self.num_inference_steps))]
        if self.decode_chunk_size is not None:
            cmd += ["--decode-chunk-size", str(int(self.decode_chunk_size))]
        if self.extra_json is not None:
            cmd += ["--extra-json", str(self.extra_json)]
        return cmd

    def _run_worker(self, cmd: List[str], work_dir: Path) -> None:
        log_path = work_dir / "run.log"
        env = None
        if self.extra_env:
            env = dict(os.environ)
            env.update(self.extra_env)
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(f"# cmd: {' '.join(cmd)}\n\n")
            log.flush()
            try:
                proc = subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT,
                    timeout=self.timeout_sec, env=env,
                )
            except subprocess.TimeoutExpired as exc:
                raise SegmentWorkerError(
                    f"External segment worker timed out after {self.timeout_sec}s. "
                    f"work_dir={work_dir} (see run.log for partial output)."
                ) from exc
            except OSError as exc:
                raise SegmentWorkerError(
                    f"Could not launch external segment worker (python_bin="
                    f"{self.python_bin!r}, worker_script={self.worker_script!r}): "
                    f"{exc}. Check that python_bin points at a real interpreter in "
                    "the intended (e.g. lgvsc_gen) conda environment."
                ) from exc

        if proc.returncode != 0:
            error_path = work_dir / "error.json"
            detail = None
            if error_path.exists():
                try:
                    detail = json.loads(error_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    detail = None
            if detail is not None:
                raise SegmentWorkerError(
                    f"External segment worker failed (exit code {proc.returncode}): "
                    f"{detail.get('error_type')}: {detail.get('message')}. "
                    f"work_dir={work_dir} (full traceback in error.json, stdout/stderr in run.log)."
                )
            raise SegmentWorkerError(
                f"External segment worker exited with code {proc.returncode} and wrote "
                f"no error.json (it may have crashed too hard to self-report — e.g. "
                f"segfault, OOM-killed). See {log_path} for captured stdout/stderr."
            )

    def _read_result(self, request: SegmentGenerationRequest, work_dir: Path) -> SegmentGenerationResult:
        result_path = work_dir / "result.json"
        if not result_path.exists():
            raise SegmentWorkerError(
                f"External segment worker exited 0 but wrote no result.json in "
                f"{work_dir} — see run.log."
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SegmentWorkerError(f"result.json in {work_dir} is not valid JSON: {exc}") from exc

        if payload.get("status") != "ok":
            raise SegmentWorkerError(
                f"result.json in {work_dir} has status={payload.get('status')!r}, expected 'ok'."
            )

        expected = [int(i) for i in request.target_indices]
        raw_frames = payload.get("frames") or {}
        got_indices = sorted(int(k) for k in raw_frames)
        if got_indices != sorted(expected):
            raise SegmentWorkerError(
                f"External segment worker returned frames for indices {got_indices!r} "
                f"but the request asked for {sorted(expected)!r} (segment_id="
                f"{request.segment_id}, work_dir={work_dir})."
            )

        from torchvision.io import read_image

        expected_shape = tuple(request.start_keyframe_recon.shape)
        frames: List[torch.Tensor] = []
        metadata_list: List[GenerationMetadata] = []
        raw_metadata = payload.get("metadata") or {}
        for idx in expected:
            rel = raw_frames.get(str(idx))
            fp = work_dir / rel
            if not fp.exists():
                raise SegmentWorkerError(
                    f"result.json references frame file {rel!r} for index {idx} but it "
                    f"does not exist in {work_dir}."
                )
            try:
                img = read_image(str(fp))  # (C, H, W) uint8
            except Exception as exc:  # noqa: BLE001
                raise SegmentWorkerError(
                    f"Could not read frame image {fp} for index {idx}: {exc}"
                ) from exc
            tensor = img.float().div(255.0)
            if len(expected_shape) == 4:
                tensor = tensor.unsqueeze(0)
            if tuple(tensor.shape) != expected_shape:
                raise SegmentWorkerError(
                    f"External segment worker returned frame {idx} with shape "
                    f"{tuple(tensor.shape)}, expected {expected_shape} (matching "
                    f"start_keyframe_recon; segment_id={request.segment_id}, "
                    f"work_dir={work_dir})."
                )
            frames.append(tensor)

            meta_dict = raw_metadata.get(str(idx))
            if meta_dict is None:
                raise SegmentWorkerError(
                    f"result.json missing metadata entry for index {idx} "
                    f"(segment_id={request.segment_id}, work_dir={work_dir})."
                )
            metadata_list.append(_metadata_from_dict(meta_dict))

        return SegmentGenerationResult(
            segment_id=payload.get("segment_id", request.segment_id),
            target_indices=expected,
            frames=frames,
            metadata=metadata_list,
        )


# Backends implemented so far (3차: copy/interpolation; 4차:
# bidirectional_interpolation; 1B: external_segment_worker). Real backends
# (SVD / Open-Sora / Wan / …) are a reserved extension point: add a
# VideoGenerator subclass + a registry entry here, without touching
# TemporalPipeline (it only depends on the VideoGenerator.generate()/
# generate_segment() contract).
_BACKENDS = {
    "copy": CopyGenerator,
    "interpolation": InterpolationGenerator,
    "bidirectional_interpolation": BidirectionalInterpolationGenerator,
    "external_segment_worker": ExternalSegmentWorkerGenerator,
}


def _build_external_segment_worker_generator(cfg) -> "ExternalSegmentWorkerGenerator":
    """Build an :class:`ExternalSegmentWorkerGenerator` from
    ``video_generator.worker.*`` cfg keys (ETRI 1B). Conditioning-mode-agnostic
    by design — see :func:`build_generator`."""
    from omegaconf import OmegaConf

    w = "video_generator.worker."
    python_bin = OmegaConf.select(cfg, w + "python_bin", default=None)
    if not python_bin:
        raise ValueError(
            "video_generator.backend='external_segment_worker' requires "
            "video_generator.worker.python_bin (the path to the OTHER conda "
            "environment's python interpreter, e.g. "
            "'/home/<user>/anaconda3/envs/lgvsc_gen/bin/python') — see "
            "docs/lgvsc_1b_worker_readiness.md."
        )
    extra_env_cfg = OmegaConf.select(cfg, w + "extra_env", default=None)
    extra_env = (
        OmegaConf.to_container(extra_env_cfg, resolve=True)
        if extra_env_cfg is not None else None
    )
    return ExternalSegmentWorkerGenerator(
        python_bin=str(python_bin),
        worker_script=OmegaConf.select(cfg, w + "worker_script", default=None),
        backend=str(OmegaConf.select(cfg, w + "backend", default="mock")),
        backend_entrypoint=OmegaConf.select(cfg, w + "backend_entrypoint", default=None),
        model_id=OmegaConf.select(cfg, w + "model_id", default=None),
        device=str(OmegaConf.select(cfg, w + "device", default="cpu")),
        dtype=str(OmegaConf.select(cfg, w + "dtype", default="fp32")),
        seed=OmegaConf.select(cfg, w + "seed", default=None),
        height=OmegaConf.select(cfg, w + "height", default=None),
        width=OmegaConf.select(cfg, w + "width", default=None),
        num_inference_steps=OmegaConf.select(cfg, w + "num_inference_steps", default=None),
        decode_chunk_size=OmegaConf.select(cfg, w + "decode_chunk_size", default=None),
        extra_json=OmegaConf.select(cfg, w + "extra_json", default=None),
        timeout_sec=OmegaConf.select(cfg, w + "timeout_sec", default=None),
        work_dir=OmegaConf.select(cfg, w + "work_dir", default=None),
        cleanup_on_success=bool(OmegaConf.select(cfg, w + "cleanup_on_success", default=True)),
        extra_env=extra_env,
    )


def build_generator(cfg) -> VideoGenerator:
    """Build a :class:`VideoGenerator` from ``video_generator.*`` cfg keys.

    ``video_generator.backend: external_segment_worker`` (ETRI 1B) is checked
    **first**, before ``conditioning_mode`` dispatch — the external worker is
    conditioning-mode-agnostic (it just uses whichever keyframes the request
    actually carries), unlike the mock backends below which are hard-wired to
    one mode each. See :func:`_build_external_segment_worker_generator` and
    ``video_generator.worker.*`` in ``configs/video/default.yaml``.

    Otherwise reads ``video_generator.conditioning_mode`` (default ``"start_only"``):

    - ``"start_only"``: dispatches on ``video_generator.backend`` (default
      ``"copy"``) to :class:`CopyGenerator` / :class:`InterpolationGenerator`.
    - ``"bidirectional"`` (ETRI 4차): builds a
      :class:`BidirectionalInterpolationGenerator`. ``video_generator.backend``
      may be unset / ``"auto"`` (canonical default) or explicitly
      ``"bidirectional_interpolation"`` — any other backend name raises
      ``NotImplementedError`` (it does not support bidirectional conditioning).
      ``video_generator.bidirectional_missing_end_policy`` (default ``"error"``)
      is forwarded to the backend.
    - any other conditioning mode raises ``NotImplementedError``.
    """
    from omegaconf import OmegaConf

    backend_probe = OmegaConf.select(cfg, "video_generator.backend", default="auto")
    if backend_probe == "external_segment_worker":
        return _build_external_segment_worker_generator(cfg)

    conditioning_mode = str(
        OmegaConf.select(cfg, "video_generator.conditioning_mode", default=CONDITIONING_MODE_START_ONLY)
    )

    if conditioning_mode == CONDITIONING_MODE_BIDIRECTIONAL:
        backend = OmegaConf.select(cfg, "video_generator.backend", default="auto")
        backend = "auto" if backend is None else str(backend)
        if backend == "auto":
            backend = "bidirectional_interpolation"
        if backend != "bidirectional_interpolation":
            raise NotImplementedError(
                f"video_generator.backend={backend!r} does not support "
                "conditioning_mode='bidirectional' in ETRI 4차 — only "
                "'bidirectional_interpolation' is implemented. Real bidirectional "
                "backends (SVD/Open-Sora) are a reserved extension point."
            )
        policy = str(
            OmegaConf.select(
                cfg, "video_generator.bidirectional_missing_end_policy", default=MISSING_END_POLICY_ERROR
            )
        )
        return BidirectionalInterpolationGenerator(missing_end_policy=policy)

    if conditioning_mode != CONDITIONING_MODE_START_ONLY:
        raise NotImplementedError(
            f"video_generator.conditioning_mode={conditioning_mode!r} is not "
            "implemented — only 'start_only' and 'bidirectional' exist "
            "(docs/etri_strategy.md 순서 5/6)."
        )

    backend = OmegaConf.select(cfg, "video_generator.backend", default="auto")
    backend = "auto" if backend is None else str(backend)
    if backend == "auto":
        backend = "copy"
    if backend == "interpolation":
        alpha = float(OmegaConf.select(cfg, "video_generator.interpolation_alpha", default=0.5))
        allow_gt = bool(OmegaConf.select(cfg, "video_generator.allow_ground_truth_reference", default=False))
        return InterpolationGenerator(alpha=alpha, allow_ground_truth_reference=allow_gt)
    if backend == "copy":
        return CopyGenerator()

    raise NotImplementedError(
        f"video_generator.backend={backend!r} is not implemented for "
        "conditioning_mode='start_only'. Real backends (e.g. SVD/Open-Sora) are "
        "a reserved extension point — add a VideoGenerator subclass and "
        "register it in video/video_generator.py::_BACKENDS. See "
        "docs/etri_strategy.md 순서 5/6."
    )


def save_generated_frames(records, output_dir) -> "List[Path]":
    """Save every ``decision == "generate"`` frame record's reconstruction to
    *output_dir* as ``generated_{index:05d}.png``.

    Kept as a standalone function (rather than inlined in
    ``scripts/evaluate_video.py``) so the "did the generate branch actually
    produce and persist frames" check is unit-testable without a full CLI/model
    run. Clears any ``generated_*.png`` left by a previous run first, mirroring
    the ``video_io``/recon-frame convention. Returns the list of saved paths.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(output_dir.glob("generated_*.png")):
        stale.unlink()

    from torchvision.utils import save_image as tv_save_image

    saved: List[Path] = []
    for rec in records:
        if getattr(rec, "decision", None) != "generate" or getattr(rec, "recon", None) is None:
            continue
        fp = output_dir / f"generated_{rec.index:05d}.png"
        tv_save_image(rec.recon.cpu().float().clamp(0, 1), str(fp))
        saved.append(fp)
    return saved
