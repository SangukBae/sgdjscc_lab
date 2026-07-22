"""video/video_generator.py – Start-only / bidirectional video generation backend
interface (ETRI 3차 step 5 + 4차 step 6).

Scope note (read before touching this file)
--------------------------------------------
This module's job is **structural**: give ``TemporalPipeline`` a real third
branch (``reuse`` / ``recompute`` / ``generate``) that a future strong video
generator (SVD, Open-Sora, …) can plug into, and prove the branch — in both
its start-only (3차) and bidirectional (4차) conditioning modes — works
end-to-end with cheap placeholder backends. It is explicitly **not** trying to
produce good-looking generated frames or to demonstrate that bidirectional
conditioning reduces drift/flicker; that quality/comparison claim needs a real
generator and is 5차+ work. Every backend here
(:class:`CopyGenerator`, :class:`InterpolationGenerator`,
:class:`BidirectionalInterpolationGenerator`) is a mock: verbatim copy or
linear blend, never learned generation, always ``mock=True``. Real backend
integration (SVD/Open-Sora) is a reserved extension point (see
:func:`build_generator`), not implemented here.

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

import logging
from dataclasses import dataclass, field
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


class VideoGenerator:
    """Backend interface for the start-only generate branch.

    Subclasses implement ``generate(request) -> GenerationResult``. The base
    class only enforces the Rx-legal / conditioning-mode guard so every
    backend gets it for free.
    """

    backend_name = "base"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError


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


# Backends implemented so far (3차: copy/interpolation; 4차:
# bidirectional_interpolation). Real backends (SVD / Open-Sora / …) are a
# reserved extension point: add a VideoGenerator subclass + a registry entry
# here, without touching TemporalPipeline (it only depends on the
# VideoGenerator.generate() contract).
_BACKENDS = {
    "copy": CopyGenerator,
    "interpolation": InterpolationGenerator,
    "bidirectional_interpolation": BidirectionalInterpolationGenerator,
}


def build_generator(cfg) -> VideoGenerator:
    """Build a :class:`VideoGenerator` from ``video_generator.*`` cfg keys.

    Reads ``video_generator.conditioning_mode`` (default ``"start_only"``):

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
