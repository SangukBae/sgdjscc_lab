"""video/temporal_pipeline.py – Keyframe/inter-frame temporal orchestration (Phase 4-B).

Drives an ordered frame sequence through the keyframe-oriented semantic pipeline:

- **keyframe**     : run the full image pipeline + build a full semantic packet
                     (this is the reference for the GOP).
- **inter-frame**  : compute the semantic delta vs the latest keyframe packet and,
                     depending on its magnitude (and, when enabled, keyframe-anchored
                     motion), pick one of three branches: *reuse* the keyframe
                     reconstruction (cheap, no diffusion), *generate* a start-only
                     conditioned frame (ETRI 3차, gated off by default — see
                     ``video/video_generator.py``), or *recompute* with guidance
                     attenuated in proportion to the change.

It also approximates FAST-GSC's `sequential conditional denoising`
(``paper/FAST-GSC/FAST_GSC.tex``).  :func:`build_staged_schedule` splits the
packet into semantic groups ordered over the denoising schedule (early: scene +
major objects → middle: relations + structure → late: attributes + fine
corrections) and composes a cumulative text prompt per stage.

What is actually wired vs. emulated (kept explicit, per review):

- **Wired**: the schedule's cumulative ``final_prompt`` is injected into the real
  reconstruction as ``cfg.prompt_override`` (see ``infer_pipeline``), so the
  packet-derived staged composition genuinely conditions the diffusion text
  prompt for both keyframes and recomputed inter-frames.  The per-stage prompts
  are also exposed on the cfg as ``staged_prompts`` for a future denoiser-aware
  consumer.
- **Not done (by design)**: true per-denoising-step prompt switching *inside* the
  DPM-Solver loop.  That would require modifying the SGD-JSCC sampler, which the
  algorithm-preservation invariant forbids; so injection happens at the
  prompt/schedule level, not inside the sampler.

Everything model-touching is injected (``reconstruct_fn`` / ``packet_fn``), so the
orchestration is unit-testable without checkpoints, and the SGD-JSCC numerics are
never modified — keyframes reuse the existing per-frame forward pass and the GOP
keyframe is the single, consistent packet+pixel reference for its inter-frames.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from sgdjscc_lab.utils import profiling

logger = logging.getLogger(__name__)


# ── FAST-GSC-inspired staged semantic injection ──────────────────────────────

def build_staged_schedule(
    packet: Dict,
    diffusion_step: int = 50,
    splits=(0.4, 0.8),
) -> Dict:
    """Split a packet into early/middle/late semantic groups over the schedule.

    Parameters
    ----------
    packet:
        A semantic packet dict.
    diffusion_step:
        Total denoising steps; stage boundaries are derived from *splits*.
    splits:
        Two fractions ``(a, b)`` in (0, 1): early = [0, a), middle = [a, b),
        late = [b, 1].

    Returns
    -------
    dict with ``stages`` (list) and ``final_prompt`` (cumulative caption text).
    Each stage has ``name``, ``fraction``, ``step_range``, ``units`` and the
    cumulative ``prompt`` injected up to and including that stage.
    """
    objects = list(packet.get("objects") or [])
    importance = (packet.get("importance") or {}).get("order") or objects
    relations = packet.get("relations") or []
    attributes = packet.get("attributes") or {}
    scene = packet.get("scene")

    # "Major" objects = top half by importance (at least one).
    n_major = max(1, (len(importance) + 1) // 2) if importance else 0
    major_objects = importance[:n_major]
    minor_objects = importance[n_major:]

    a, b = splits
    s = int(diffusion_step)
    bounds = [(0.0, a), (a, b), (b, 1.0)]
    step_bounds = [(int(lo * s), int(hi * s)) for lo, hi in bounds]

    stage_units = [
        {"name": "early", "scene": scene, "objects": major_objects},
        {"name": "middle", "relations": relations, "objects": minor_objects},
        {"name": "late", "attributes": attributes},
    ]

    stages: List[Dict] = []
    for i, units in enumerate(stage_units):
        prompt = _compose_prompt(packet, up_to=i)
        stages.append({
            "name": units["name"],
            "fraction": list(bounds[i]),
            "step_range": list(step_bounds[i]),
            "units": units,
            "prompt": prompt,
        })

    return {"stages": stages, "final_prompt": _compose_prompt(packet, up_to=2)}


def _compose_prompt(packet: Dict, up_to: int) -> str:
    """Build a cumulative text prompt from packet groups up to *up_to* stage idx."""
    parts: List[str] = []
    scene = packet.get("scene")
    importance = (packet.get("importance") or {}).get("order") or list(packet.get("objects") or [])
    n_major = max(1, (len(importance) + 1) // 2) if importance else 0

    if up_to >= 0:  # early: scene + major objects
        if scene:
            parts.append(str(scene))
        parts.extend(importance[:n_major])
    if up_to >= 1:  # middle: minor objects + relations
        parts.extend(importance[n_major:])
        for rel in (packet.get("relations") or []):
            parts.append(f"{rel.get('subject')} {rel.get('predicate')} {rel.get('object')}")
    if up_to >= 2:  # late: attributes
        for obj, adjs in (packet.get("attributes") or {}).items():
            if adjs:
                parts.append(f"{' '.join(adjs)} {obj}")
    return ", ".join(str(p) for p in parts if p)


# ── Per-frame record + pipeline ───────────────────────────────────────────────

@dataclass
class FrameRecord:
    index: int
    role: str                       # "keyframe" | "inter"
    reused: bool = False
    delta: Optional[Dict] = None
    guidance_scale: Optional[float] = None
    transmitted_units: int = 0
    srs: Optional[float] = None
    orig_packet: Optional[Dict] = None
    recon_packet: Optional[Dict] = None
    staged_schedule: Optional[Dict] = None
    # Motion gate (Phase 4-B dual gate): keyframe-anchored motion estimate and
    # the resulting decision. decision ∈ {"keyframe", "reuse", "generate",
    # "recompute_semantic", "recompute_motion"}; motion fields stay None when
    # motion gating is disabled (default). "generate" only occurs when the
    # ETRI 3차 generate branch is enabled (default off) — see
    # ``TemporalPipeline.enable_generate`` / ``video/video_generator.py``.
    motion: Optional[Dict] = None
    motion_score: Optional[float] = None
    decision: Optional[str] = None
    # Start-only generation metadata (ETRI 3차; see video/video_generator.py's
    # GenerationMetadata.to_dict()). None unless decision == "generate".
    generation: Optional[Dict] = None
    recon: Optional[object] = None   # reconstructed frame tensor (not logged)

    def to_log(self) -> Dict:
        return {
            "index": self.index,
            "role": self.role,
            "reused": self.reused,
            "decision": self.decision,
            "guidance_scale": self.guidance_scale,
            "transmitted_units": self.transmitted_units,
            "magnitude": (self.delta or {}).get("magnitude") if self.delta else None,
            "num_changes": (self.delta or {}).get("num_changes") if self.delta else None,
            "motion_score": self.motion_score,
            "motion_residual": (self.motion or {}).get("residual_energy") if self.motion else None,
            "motion_block_max": (self.motion or {}).get("block_max") if self.motion else None,
            "srs": self.srs,
            # ETRI 4차: which conditioning mode actually produced this frame
            # ("start_only" | "bidirectional"), None unless decision == "generate".
            # Note a bidirectional REQUEST can still report "start_only" here —
            # see BidirectionalInterpolationGenerator's missing-end fallback.
            "generation_conditioning_mode": (self.generation or {}).get("conditioning_mode"),
        }


def count_units(packet: Dict) -> int:
    """Count transmittable semantic units in a packet (objects + relations + attrs)."""
    objs = len(packet.get("objects") or [])
    rels = len(packet.get("relations") or [])
    attrs = sum(len(v) for v in (packet.get("attributes") or {}).values())
    scene = 1 if packet.get("scene") else 0
    return objs + rels + attrs + scene


def _split_evenly(total: int, n: int) -> List[int]:
    """Split an integer *total* into *n* non-negative shares as evenly as
    possible, summing back to exactly *total* (ETRI 1A: distributing one
    ``generate_segment()`` call's profiler call-count deltas across the
    frames it covered — see ``TemporalPipeline._flush_pending_generate``).

    Each share is ``total // n``, with the remainder (``total % n``) added one
    at a time to the first few frames — an arbitrary but deterministic tie-
    break, since a single batched call genuinely cannot say which specific
    frame "caused" which call.
    """
    base, remainder = divmod(total, n)
    return [base + (1 if i < remainder else 0) for i in range(n)]


class TemporalPipeline:
    """Keyframe/inter-frame orchestrator for a frame sequence.

    Parameters
    ----------
    reconstruct_fn:
        ``(frame_tensor, cfg) -> reconstructed_tensor``.  Runs the per-frame
        SGD-JSCC forward pass (injected so the pipeline is model-agnostic).
    packet_fn:
        ``(frame_tensor, frame_id) -> packet`` building a semantic packet.
    keyframe_extractor:
        A ``KeyframeExtractor``.  If None one is built from ``scene_detector``.
    scene_detector:
        Used to build a default keyframe extractor when one is not supplied.
    delta:
        A ``SemanticDelta`` (defaults to a fresh instance).
    cfg:
        Optional base run config passed to ``reconstruct_fn`` (copied per frame).
    reuse_threshold:
        Semantic-delta magnitude below which an inter-frame is a *candidate* for
        reusing the keyframe reconstruction (config alias:
        ``temporal.semantic_delta_threshold``).
    motion_threshold:
        Optional keyframe-anchored motion-score threshold (Phase 4-B dual gate).
        Default None = motion gating DISABLED → behaviour identical to the
        semantic-delta-only pipeline.  When set, an inter-frame whose semantic
        delta is small but whose motion score vs the keyframe is >= this value
        is NOT reused; it is recomputed (decision "recompute_motion").  This
        catches camera pan/zoom where the object inventory is unchanged but the
        pixels moved (docs/etri_strategy.md 한계 1).
    motion_weight:
        Blend between global and localised motion in the score:
        ``score = (1-w)*residual_energy + w*block_max`` (default 0.5).
    motion_grid:
        Block grid size for ``motion_residual.block_motion`` (default 8).
    diffusion_step:
        Single source of truth for the denoising-step count used by **both** the
        staged schedule (stage boundaries) and the per-frame reconstruction
        config (``_frame_cfg`` forces ``cfg.diffusion_step`` to this value).  If
        None, it is taken from ``cfg.diffusion_step`` (default 50), so the stage
        plan can never silently disagree with the actual denoising.
    srs_fn:
        Optional ``(orig_packet, recon_packet) -> srs`` to fill per-frame SRS
        (e.g. ``SemanticReliabilityEvaluator.score_packet`` wrapper).
    enable_generate:
        ETRI 3차 start-only generate branch (default ``False`` → behaviour
        identical to the pre-3차 reuse/recompute-only pipeline). When True, an
        inter-frame that fails the reuse dual-gate but whose semantic delta and
        motion are still moderate (see ``generate_delta_min/max`` and
        ``generate_motion_max``) is routed to ``video_generator`` instead of a
        full diffusion recompute. See ``video/video_generator.py`` for the
        scope note on what "generate" does and does not mean in 3차.
    video_generator:
        A ``video.video_generator.VideoGenerator`` instance. If None and
        ``enable_generate`` is True, a ``CopyGenerator`` (mock) is used.
        Ignored when ``enable_generate`` is False.
    generate_delta_min / generate_delta_max:
        Semantic-delta magnitude band (inclusive) that makes an inter-frame a
        *generate candidate* once it has already failed the reuse dual-gate.
        Defaults: ``generate_delta_min = reuse_threshold`` (right where reuse
        stops), ``generate_delta_max = min(1.0, 3 * reuse_threshold)`` (a
        heuristic "moderate change" ceiling — tune per dataset). Frames whose
        delta exceeds ``generate_delta_max`` fall through to recompute.
    generate_motion_max:
        Optional keyframe-anchored motion-score ceiling for generate
        candidacy. Defaults to ``motion_threshold`` (the same bound the reuse
        gate uses) when motion gating is enabled, else unconstrained (motion
        is never computed when motion gating is off, so this has no effect).
        A frame whose motion exceeds this still falls through to recompute —
        generate is not meant to paper over real camera motion.
    allow_ground_truth_reference:
        Kept for backward-compatible construction only. Pre-1A, this was
        forwarded to backends that support a test/mock-only ground-truth
        reference path (e.g. ``InterpolationGenerator`` via
        ``GenerationRequest.reference_target_frame``). Since ETRI 1A the
        generate branch always goes through the segment-level
        ``SegmentGenerationRequest`` contract, which has **no**
        ground-truth-frame field at all (see ``video/video_generator.py``'s
        Rx-legal boundary note) — so this flag no longer has any effect on
        ``TemporalPipeline``'s own behaviour. The backend-level ground-truth
        path is still fully testable by constructing a ``GenerationRequest``
        directly (see ``tests/test_video_generator.py``); it is simply never
        reachable through this pipeline anymore.
    conditioning_mode:
        ETRI 4차: ``"start_only"`` (default) or ``"bidirectional"``. Only
        consulted when ``enable_generate`` is True; ignored otherwise. In
        bidirectional mode, ``run()`` performs a lightweight *prepass* that
        reconstructs every keyframe up front (a GOP's inter-frames need the
        *next* GOP's keyframe reconstruction as the "end" condition, which a
        single forward pass hasn't reached yet) — see ``run()``'s docstring.
        The start-only / disabled path is completely unaffected and stays
        single-pass, so 1~3차 results are unchanged regardless of this flag's
        value when ``enable_generate`` is False.
    force_interframe_reuse:
        Speed/validation knob (default ``False`` = unchanged behaviour): when
        True, every inter-frame unconditionally takes the *reuse* branch
        (the keyframe reconstruction, no diffusion call) regardless of its
        semantic delta / motion score. Delta and motion are still computed and
        logged as usual (cheap, no model call) so the per-frame CSV keeps
        showing what a normal recompute/generate decision *would* have been —
        only the actual reconstruction/decision is forced to reuse. This is
        what makes a "keyframe-only real-model" validation run possible: only
        keyframes ever invoke the diffusion model, so wall-clock scales with
        ``n_keyframes`` instead of ``n_frames``. It is a WEAKER validation
        than the full per-frame pipeline — see docs/etri_strategy.md and the
        speed-experiment report before treating its temporal metrics as
        equivalent to a real_all_frames run.
    fps:
        Optional source frame rate, forwarded verbatim into every
        ``SegmentGenerationRequest.fps`` (ETRI 1A) built for the generate
        branch. ``None`` (default) when the caller has no natural playback
        rate (e.g. synthetic/image sequences) — has no effect on any
        reconstruction/decision, only on what a segment-level backend is told
        about the segment's duration.

    Generate-branch batching (ETRI 후속 1단계 step 1A)
    ----------------------------------------------------
    Previously (3~4차) every ``generate``-decision inter-frame invoked
    ``video_generator.generate()`` immediately, one call per frame. 1A
    replaces that with GOP-level batching: all of a GOP's ``generate``-decision
    frames (which may be a non-contiguous subset of its inter-frames — a GOP
    can freely mix ``reuse``/``recompute_*``/``generate``) are collected while
    the GOP is being processed, and resolved with exactly one
    ``video_generator.generate_segment(SegmentGenerationRequest(...))`` call
    once the GOP is complete (at the next keyframe, or at the end of the
    sequence for the last GOP). A GOP with zero ``generate`` decisions never
    calls the backend at all. See ``video/video_generator.py`` for the
    request/result contract and its Rx-legal boundary; ``VideoGenerator``'s
    default ``generate_segment`` implementation still loops over ``generate()``
    per target frame internally, so 3~4차's mock backends (and their exact
    numeric output) are unaffected by this change — only the pipeline's own
    call site was batched, not backend behaviour.
    """

    def __init__(
        self,
        reconstruct_fn: Callable,
        packet_fn: Callable,
        keyframe_extractor=None,
        scene_detector=None,
        delta=None,
        cfg=None,
        reuse_threshold: float = 0.2,
        motion_threshold: Optional[float] = None,
        motion_weight: float = 0.5,
        motion_grid: int = 8,
        diffusion_step: Optional[int] = None,
        srs_fn: Optional[Callable] = None,
        enable_generate: bool = False,
        video_generator=None,
        generate_delta_min: Optional[float] = None,
        generate_delta_max: Optional[float] = None,
        generate_motion_max: Optional[float] = None,
        allow_ground_truth_reference: bool = False,
        conditioning_mode: str = "start_only",
        force_interframe_reuse: bool = False,
        fps: Optional[float] = None,
    ) -> None:
        self.reconstruct_fn = reconstruct_fn
        self.packet_fn = packet_fn
        self.cfg = cfg
        self.fps = None if fps is None else float(fps)
        self.reuse_threshold = reuse_threshold
        self.motion_threshold = None if motion_threshold is None else float(motion_threshold)
        self.motion_weight = float(motion_weight)
        self.motion_grid = int(motion_grid)
        # Resolve the authoritative step count: explicit arg wins, else cfg, else 50.
        if diffusion_step is not None:
            self.diffusion_step = int(diffusion_step)
        elif cfg is not None:
            self.diffusion_step = int(cfg.get("diffusion_step", 50))
        else:
            self.diffusion_step = 50
        self.srs_fn = srs_fn

        # ETRI 3차 start-only generate branch (default off; see class docstring).
        self.enable_generate = bool(enable_generate)
        self.generate_delta_min = (
            float(generate_delta_min) if generate_delta_min is not None else float(self.reuse_threshold)
        )
        self.generate_delta_max = (
            float(generate_delta_max) if generate_delta_max is not None
            else float(min(1.0, 3.0 * self.reuse_threshold))
        )
        if generate_motion_max is not None:
            self.generate_motion_max: Optional[float] = float(generate_motion_max)
        else:
            self.generate_motion_max = self.motion_threshold  # may itself be None
        self.allow_ground_truth_reference = bool(allow_ground_truth_reference)
        self.conditioning_mode = str(conditioning_mode)
        self.force_interframe_reuse = bool(force_interframe_reuse)
        if self.enable_generate and video_generator is None:
            from sgdjscc_lab.video.video_generator import CopyGenerator
            video_generator = CopyGenerator()
        self.video_generator = video_generator

        if keyframe_extractor is None:
            from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor
            from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector
            keyframe_extractor = KeyframeExtractor(
                scene_detector or SceneChangeDetector()
            )
        self.keyframe_extractor = keyframe_extractor

        if delta is None:
            from sgdjscc_lab.video.semantic_delta import SemanticDelta
            delta = SemanticDelta()
        self.delta = delta

    def _frame_cfg(self, schedule: Optional[Dict], magnitude: Optional[float] = None):
        """Build a per-frame run config with the staged prompt actually wired in.

        The schedule's cumulative ``final_prompt`` is set as ``prompt_override`` so
        the packet-derived (staged) semantics genuinely condition the diffusion
        reconstruction (see ``infer_pipeline`` ``prompt_override``).  When
        *magnitude* is given (inter-frame recompute), guidance is attenuated in
        proportion to the change so small changes lean less on the prior.

        Returns ``(cfg, guidance_scale)``; ``(None, None)`` when no base cfg is
        configured (e.g. unit tests that inject a config-free reconstruct_fn).
        """
        if self.cfg is None:
            return None, None
        from omegaconf import OmegaConf
        out = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        # Keep the actual denoising step count in lock-step with the schedule's
        # stage boundaries (both derive from self.diffusion_step) so a stage plan
        # built for N steps can never run against a different reconstruction N.
        out.diffusion_step = int(self.diffusion_step)
        if schedule is not None:
            out.prompt_override = schedule.get("final_prompt", "")
            # Stage plan kept on the cfg for downstream/denoiser-aware consumers.
            out.staged_prompts = [s["prompt"] for s in schedule.get("stages", [])]
        base_gs = float(self.cfg.get("guidance_scale", 4.0))
        if magnitude is None:
            gs = base_gs
        else:
            gs = round(base_gs * (0.5 + 0.5 * float(magnitude)), 6)  # 0.5×..1.0× of base
            out.guidance_scale = gs
        return out, gs

    def _motion_vs_keyframe(self, keyframe_frame, frame):
        """Keyframe-anchored motion estimate → ``(motion_dict, motion_score)``.

        Only computed when motion gating is enabled (``motion_threshold`` set)
        and a keyframe pixel reference exists; returns ``(None, None)``
        otherwise, keeping the default path identical to the legacy pipeline.
        """
        if self.motion_threshold is None or keyframe_frame is None:
            return None, None
        from sgdjscc_lab.video.motion_residual import estimate
        motion = estimate(keyframe_frame, frame, grid=self.motion_grid)
        w = min(max(self.motion_weight, 0.0), 1.0)
        score = float((1.0 - w) * motion["residual_energy"] + w * motion["block_max"])
        return motion, score

    def _is_generate_candidate(self, magnitude: float, motion_score: Optional[float]) -> bool:
        """Return True when a reuse-ineligible inter-frame should be *generated*
        instead of recomputed (ETRI 3차; only ever True when ``enable_generate``).

        Only reached after the reuse dual-gate has already failed, so this
        picks out the "moderate change" middle ground: semantic delta in
        ``[generate_delta_min, generate_delta_max]`` AND motion (when measured)
        within ``generate_motion_max``. Deltas/motion beyond those bounds fall
        through to the existing recompute branch unchanged.
        """
        if not self.enable_generate:
            return False
        if magnitude < self.generate_delta_min or magnitude > self.generate_delta_max:
            return False
        if motion_score is not None and self.generate_motion_max is not None \
                and motion_score > self.generate_motion_max:
            return False
        return True

    def _flush_pending_generate(self, pending: List[Dict]) -> object:
        """Resolve one GOP's deferred ``generate``-decision frames with a
        single ``video_generator.generate_segment()`` call (ETRI 1A).

        *pending* is a list of dicts (one per deferred frame, in ascending
        index order) built by :meth:`run` — see its per-frame ``continue``
        branch for the ``generate`` decision. Mutates each entry's
        ``rec`` (a :class:`FrameRecord`) in place: fills in ``recon``,
        ``generation`` and ``recon_packet``/``srs`` exactly as the pre-1A
        per-frame path did, so every downstream consumer (``to_log()``,
        ``segment.py::build_segments``, CSV/JSON writers) sees an identical
        schema. Returns the last resolved frame's reconstruction (the new
        ``prev_recon`` for whatever comes after this GOP), or ``None`` if
        *pending* is empty (no-op — the backend is never called for a GOP
        with zero generate decisions).

        Profiling: the ``generate_segment()`` call's wall-clock cost is timed
        here directly (not via the per-index ``RunProfiler.frame()`` context —
        see :meth:`run`, which deliberately never exits that context for a
        deferred frame) and split evenly across the frames this call covers
        via ``RunProfiler.record_frame()``. This keeps the real backend
        latency attributed to the *generate* frames themselves, instead of
        being invisible (a deferred frame's placeholder timer never actually
        ran the backend) or wrongly blamed on whichever keyframe happens to
        be processed next (this method runs before that keyframe's own timer
        starts, since it flushes the previous GOP first — see :meth:`run`).
        The same treatment applies to the profiler's *call-count* deltas
        (``diffusion_calls``/``blip2_calls``/``clip_calls``): a real (1B+)
        backend that calls ``profiling.record_diffusion_call()`` etc. inside
        ``generate_segment()`` has those counted in the run-wide totals
        regardless, but without an explicit snapshot here they would never
        show up on any individual frame's record either — this method
        snapshots ``RunProfiler.counters`` immediately before the call and
        distributes the post-call deltas across the covered frames the same
        way as the elapsed time (as evenly as an integer count allows; see
        :func:`_split_evenly`).
        """
        if not pending:
            return None

        from sgdjscc_lab.video.video_generator import SegmentGenerationRequest, validate_segment_result

        first = pending[0]
        target_indices = [p["index"] for p in pending]
        captions = [(p["packet"] or {}).get("caption") or None for p in pending]
        packets = [p["packet"] for p in pending]
        side_infos = [
            {"delta": p["delta"], "motion": p["motion"], "motion_score": p["motion_score"]}
            for p in pending
        ]
        reference_prev_recons = [p["reference_prev_recon"] for p in pending]
        request = SegmentGenerationRequest(
            segment_id=first["segment_id"],
            start_frame_index=first["gop_keyframe_index"],
            end_frame_index=first["gop_end_index"],
            target_indices=target_indices,
            start_keyframe_recon=first["keyframe_recon"],
            start_keyframe_index=first["gop_keyframe_index"],
            end_keyframe_recon=first["end_keyframe_recon"],
            end_keyframe_index=first["end_keyframe_index"],
            fps=self.fps,
            captions=captions,
            packets=packets,
            side_infos=side_infos,
            reference_prev_recon=reference_prev_recons[0],
            reference_prev_recons=reference_prev_recons,
        )
        _prof = profiling.get_active()
        _counters_before = dict(_prof.counters) if _prof is not None else None

        _t0 = time.monotonic()
        result = self.video_generator.generate_segment(request)
        _elapsed = time.monotonic() - _t0
        validate_segment_result(request, result)

        n = len(pending)
        _share = _elapsed / n
        if _prof is not None:
            def _delta(key: str) -> int:
                return _prof.counters.get(key, 0) - _counters_before.get(key, 0)

            _diffusion_shares = _split_evenly(_delta("diffusion_calls"), n)
            _blip2_shares = _split_evenly(_delta("blip2_calls"), n)
            _clip_shares = _split_evenly(
                _delta("clip_image_calls") + _delta("clip_text_calls"), n
            )

        last_recon = None
        for j, p in enumerate(pending):
            idx = p["index"]
            frame = result.frame_for(idx)
            meta = result.metadata_for(idx)
            rec = p["rec"]
            rec.recon = frame
            rec.generation = meta.to_dict()
            rec.recon_packet = self.packet_fn(frame, f"recon_{idx:05d}")
            if self.srs_fn is not None and rec.orig_packet and rec.recon_packet:
                try:
                    rec.srs = float(self.srs_fn(rec.orig_packet, rec.recon_packet))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("srs_fn failed on frame %d: %s", idx, exc)
            if _prof is not None:
                _prof.record_frame(
                    index=idx, decision="generate", elapsed_sec=_share,
                    diffusion_calls=_diffusion_shares[j],
                    blip2_calls=_blip2_shares[j],
                    clip_calls=_clip_shares[j],
                )
            last_recon = frame
        return last_recon

    def _prepass_keyframe_recons(self, frames: List, structure: Dict):
        """Precompute every keyframe's packet/reconstruction/guidance-scale
        before the main per-frame loop (ETRI 4차 bidirectional mode only).

        A GOP's inter-frames need the *next* GOP's keyframe reconstruction as
        the bidirectional "end" condition, but the main loop is a single
        forward pass that only reaches that keyframe later. Bidirectional mode
        precomputes every keyframe up front instead, so the main loop then
        looks the result up rather than recomputing it — no keyframe is
        reconstructed twice, and the reconstruction itself is identical to the
        single-pass path (same ``reconstruct_fn``/``packet_fn`` calls, just
        reordered). Never called when ``conditioning_mode == "start_only"``
        (the default) or when ``enable_generate`` is False.

        Returns
        -------
        ``(recon_cache, recon_packet_cache, gs_cache, next_keyframe_of)`` —
        dicts keyed by keyframe frame index; ``next_keyframe_of[k]`` is the
        following GOP's keyframe index, or ``None`` for the last GOP.
        """
        gop_keyframes = [int(g["keyframe"]) for g in (structure.get("gops") or [])]
        recon_cache: Dict[int, object] = {}
        recon_packet_cache: Dict[int, Dict] = {}
        gs_cache: Dict[int, Optional[float]] = {}
        next_keyframe_of: Dict[int, Optional[int]] = {}

        for pos, kf in enumerate(gop_keyframes):
            next_keyframe_of[kf] = gop_keyframes[pos + 1] if pos + 1 < len(gop_keyframes) else None
            kf_frame = frames[kf]
            kf_packet = self.packet_fn(kf_frame, f"frame_{kf:05d}")
            kf_schedule = build_staged_schedule(kf_packet, self.diffusion_step)
            cfg_i, gs = self._frame_cfg(kf_schedule, magnitude=None)
            kf_recon = self.reconstruct_fn(kf_frame, cfg_i)
            recon_cache[kf] = kf_recon
            recon_packet_cache[kf] = self.packet_fn(kf_recon, f"recon_{kf:05d}")
            gs_cache[kf] = gs

        return recon_cache, recon_packet_cache, gs_cache, next_keyframe_of

    def run(self, frames: List) -> Dict:
        """Process an ordered list of frame tensors.

        Returns
        -------
        dict with:
            ``frame_records`` – list[dict] per-frame logs (see FrameRecord.to_log).
            ``keyframe_structure`` – GOP layout from the keyframe extractor.
            ``records`` – the raw FrameRecord objects (with packets) for metric use.
            ``summary`` – overhead statistics including the semantic-unit reduction
                          vs naive per-frame full transmission.
        """
        n = len(frames)
        structure = self.keyframe_extractor.extract(frames)
        roles = structure["frame_roles"]

        # ETRI 1A: per-GOP span/segment-id lookup, used to build the batched
        # SegmentGenerationRequest when a GOP's pending generate frames are
        # flushed (see _flush_pending_generate). segment_id matches
        # video/segment.py::build_segments's enumerate() order over the same
        # structure["gops"], so a generate branch's segment_id lines up with
        # its SegmentRecord.segment_id.
        gop_end_index: Dict[int, int] = {}
        gop_segment_id: Dict[int, int] = {}
        for sid, g in enumerate(structure.get("gops") or []):
            kf = int(g["keyframe"])
            inter = [int(x) for x in (g.get("inter_frames") or [])]
            gop_end_index[kf] = inter[-1] if inter else kf
            gop_segment_id[kf] = sid

        # ETRI 4차: bidirectional generate needs each GOP's *next* keyframe
        # reconstruction as the "end" condition for the CURRENT GOP's
        # inter-frames — see _prepass_keyframe_recons. Only active in
        # bidirectional mode; start_only/disabled stays single-pass and
        # numerically unchanged (kf_recon_cache stays empty → every branch
        # below falls through to the original single-pass code path).
        bidirectional = self.enable_generate and self.conditioning_mode == "bidirectional"
        kf_recon_cache: Dict[int, object] = {}
        kf_recon_packet_cache: Dict[int, Dict] = {}
        kf_gs_cache: Dict[int, Optional[float]] = {}
        next_keyframe_of: Dict[int, Optional[int]] = {}
        if bidirectional:
            kf_recon_cache, kf_recon_packet_cache, kf_gs_cache, next_keyframe_of = \
                self._prepass_keyframe_recons(frames, structure)

        records: List[FrameRecord] = []
        # GOP anchor — the packet reference (for delta), the pixel reference (for
        # reuse) and the raw keyframe pixels (for the motion gate) are the current
        # keyframe, kept consistent.  These are NOT advanced by inter-frame
        # recomputations.
        keyframe_packet: Optional[Dict] = None
        keyframe_frame = None
        keyframe_recon = None
        keyframe_recon_packet: Optional[Dict] = None
        keyframe_index: Optional[int] = None
        # Tracks the immediately-previous frame's reconstruction (any decision
        # type). Only consumed by the ETRI 3차 generate branch's mock/test
        # interpolation reference path (Rx-legal: it's a real Rx-side artifact).
        prev_recon = None

        # ETRI 1A: deferred generate-decision frames for the GOP currently
        # being processed. Flushed (one video_generator.generate_segment()
        # call) at the next keyframe or at the end of the sequence — see
        # _flush_pending_generate(). Each entry's own "reference_prev_recon"
        # override is computed at append time (see below): the TRUE
        # immediately-preceding reconstruction whenever that frame was
        # resolved independently of this segment call, or None when the
        # immediate predecessor is itself a still-pending generate target
        # (in which case the default generate_segment() fallback chains to
        # that target's own output — see SegmentGenerationRequest's
        # reference_prev_recons docstring).
        pending: List[Dict] = []

        # Optional per-frame profiling (utils/profiling.py). No-op unless a
        # RunProfiler has been installed via profiling.set_active() — see
        # scripts/evaluate_video.py. Manually driving the context manager's
        # enter/exit (instead of a `with` block) avoids re-indenting the whole
        # per-frame decision tree below.
        _prof = profiling.get_active()

        for i in range(n):
            # ETRI 1A: flush the PREVIOUS GOP's deferred generate frames
            # before this iteration overwrites the keyframe_* anchor vars
            # they depend on (keyframe_recon / keyframe_index / end keyframe
            # lookup) — and, importantly, BEFORE this frame's own profiler
            # context starts (see below). No-op (and no backend call) when
            # that GOP had zero generate-decision frames. Doing this ahead of
            # `_prof.frame(i)` keeps the flush's real backend cost off frame
            # i's timer entirely — it is independently timed and attributed
            # inside _flush_pending_generate() itself, so counting it here
            # too would double-attribute it to the next keyframe.
            if roles[i] == "keyframe" and pending:
                last_recon = self._flush_pending_generate(pending)
                pending = []
                if last_recon is not None:
                    prev_recon = last_recon

            _pctx = _prof.frame(i) if _prof is not None else None
            if _pctx is not None:
                _pctx.__enter__()
            frame = frames[i]
            curr_packet = self.packet_fn(frame, f"frame_{i:05d}")
            schedule = build_staged_schedule(curr_packet, self.diffusion_step)

            if roles[i] == "keyframe":
                if i in kf_recon_cache:
                    # Bidirectional prepass already computed this keyframe —
                    # reuse it (avoids a duplicate reconstruct_fn call).
                    recon = kf_recon_cache[i]
                    recon_packet = kf_recon_packet_cache[i]
                    gs = kf_gs_cache[i]
                else:
                    cfg_i, gs = self._frame_cfg(schedule, magnitude=None)
                    recon = self.reconstruct_fn(frame, cfg_i)
                    recon_packet = self.packet_fn(recon, f"recon_{i:05d}")
                # Set the GOP anchor (packet + pixel + motion reference together).
                keyframe_packet = curr_packet
                keyframe_frame = frame
                keyframe_recon = recon
                keyframe_recon_packet = recon_packet
                keyframe_index = i
                rec = FrameRecord(
                    index=i, role="keyframe", reused=False, delta=None,
                    guidance_scale=gs,
                    transmitted_units=count_units(curr_packet),
                    orig_packet=curr_packet, recon_packet=recon_packet,
                    staged_schedule=schedule, decision="keyframe",
                )
            else:
                # Reference = keyframe packet (consistent with the pixel reference).
                d = self.delta.compute(keyframe_packet or {}, curr_packet)
                # Dual gate (Phase 4-B): semantic delta AND keyframe-anchored
                # motion must both be small to reuse.  Motion gating is off by
                # default (motion_threshold=None) → legacy semantic-only gate.
                motion, motion_score = self._motion_vs_keyframe(keyframe_frame, frame)
                semantic_ok = d["magnitude"] < self.reuse_threshold
                motion_ok = motion_score is None or motion_score < self.motion_threshold
                if self.force_interframe_reuse or (semantic_ok and motion_ok):
                    # Reuse the keyframe reconstruction AND its packet — the pixel
                    # and packet references are the same keyframe, so reuse
                    # semantics stay consistent.  Only the delta is "transmitted".
                    recon = keyframe_recon
                    recon_packet = keyframe_recon_packet if keyframe_recon_packet is not None else curr_packet
                    rec = FrameRecord(
                        index=i, role="inter", reused=True, delta=d,
                        guidance_scale=0.0,
                        transmitted_units=d["num_changes"],
                        orig_packet=curr_packet, recon_packet=recon_packet,
                        staged_schedule=schedule,
                        motion=motion, motion_score=motion_score,
                        decision="reuse",
                    )
                elif self._is_generate_candidate(d["magnitude"], motion_score):
                    # ETRI 3차/4차/1A: moderate delta/motion → generate branch
                    # instead of a full diffusion recompute. Structural only — see
                    # video/video_generator.py's scope note (mock backends, not
                    # learned generation). The GOP anchor is left unchanged, same
                    # as the recompute branch. In bidirectional mode, the current
                    # GOP's "end" keyframe is the NEXT GOP's keyframe (already
                    # cached by the prepass); None for the last GOP (no following
                    # keyframe) or in start-only mode.
                    #
                    # ETRI 1A: this frame is not generated immediately — it is
                    # deferred and resolved together with the rest of this
                    # GOP's generate-decision frames in one
                    # video_generator.generate_segment() call (see
                    # _flush_pending_generate(), invoked at the next keyframe
                    # or at the end of run()). recon/recon_packet/generation
                    # stay unset on this FrameRecord until that flush.
                    end_kf_idx = next_keyframe_of.get(keyframe_index) if bidirectional else None
                    end_kf_recon = kf_recon_cache.get(end_kf_idx) if end_kf_idx is not None else None
                    rec = FrameRecord(
                        index=i, role="inter", reused=False, delta=d,
                        guidance_scale=0.0,   # no diffusion invoked, same convention as reuse
                        transmitted_units=d["num_changes"],
                        orig_packet=curr_packet, recon_packet=None,
                        staged_schedule=schedule,
                        motion=motion, motion_score=motion_score,
                        decision="generate",
                        generation=None,
                    )
                    # The true previous reconstruction is only knowable now
                    # when frame i-1 was NOT itself deferred into this same
                    # pending batch (i.e. it was a keyframe/reuse/recompute,
                    # already resolved, and prev_recon reflects it exactly).
                    # When frame i-1 IS the previous pending entry, its
                    # reconstruction only exists once this GOP's
                    # generate_segment() call actually runs — leave this
                    # entry's override None so the default fallback chains to
                    # that target's own (future) output instead of a stale
                    # prev_recon snapshot.
                    if pending and pending[-1]["index"] == i - 1:
                        reference_prev_recon_override = None
                    else:
                        reference_prev_recon_override = prev_recon
                    pending.append({
                        "rec": rec,
                        "index": i,
                        "packet": curr_packet,
                        "delta": d,
                        "motion": motion,
                        "motion_score": motion_score,
                        "gop_keyframe_index": keyframe_index,
                        "gop_end_index": gop_end_index[keyframe_index],
                        "segment_id": gop_segment_id[keyframe_index],
                        "keyframe_recon": keyframe_recon,
                        "end_keyframe_index": end_kf_idx,
                        "end_keyframe_recon": end_kf_recon,
                        "reference_prev_recon": reference_prev_recon_override,
                    })
                    records.append(rec)
                    # Deliberately do NOT exit _pctx here: the real backend
                    # work for this frame hasn't happened yet (it's deferred
                    # to _flush_pending_generate), so timing it now would
                    # record a near-zero elapsed_sec and permanently lose the
                    # actual generate_segment() cost. _flush_pending_generate
                    # times the real call itself and attributes it to this
                    # frame's index via RunProfiler.record_frame() instead —
                    # this placeholder context is simply discarded unexited.
                    continue
                else:
                    # Recompute this frame with the staged prompt + attenuated
                    # guidance.  The GOP anchor is left unchanged so subsequent
                    # inter-frames keep referencing the keyframe.  When the
                    # recompute was triggered by motion (semantic delta small),
                    # fold the motion score into the guidance magnitude so large
                    # motion leans more on the current frame's own evidence.
                    decision = "recompute_semantic" if not semantic_ok else "recompute_motion"
                    mag = float(d["magnitude"])
                    if not motion_ok:
                        mag = max(mag, min(1.0, float(motion_score)))
                    cfg_i, gs = self._frame_cfg(schedule, magnitude=mag)
                    recon = self.reconstruct_fn(frame, cfg_i)
                    recon_packet = self.packet_fn(recon, f"recon_{i:05d}")
                    rec = FrameRecord(
                        index=i, role="inter", reused=False, delta=d,
                        guidance_scale=gs,
                        transmitted_units=d["num_changes"],
                        orig_packet=curr_packet, recon_packet=recon_packet,
                        staged_schedule=schedule,
                        motion=motion, motion_score=motion_score,
                        decision=decision,
                    )

            rec.recon = recon
            if self.srs_fn is not None and rec.orig_packet and rec.recon_packet:
                try:
                    rec.srs = float(self.srs_fn(rec.orig_packet, rec.recon_packet))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("srs_fn failed on frame %d: %s", i, exc)

            records.append(rec)
            prev_recon = recon
            if _pctx is not None:
                _pctx.decision = rec.decision
                _pctx.__exit__(None, None, None)

        # ETRI 1A: flush the LAST GOP's pending generate frames (no following
        # keyframe iteration will do it). No-op if that GOP had none.
        if pending:
            self._flush_pending_generate(pending)
            pending = []

        summary = self._summarize(records)

        # GOP/segment abstraction (ETRI 1차 step 4): frame records grouped per
        # GOP with delta/motion/temporal summaries.  Pure aggregation over the
        # already-computed records — no numeric effect on the frame path.
        # SegmentRecord.generation is filled in (ETRI 3차) when any frame in the
        # segment was generated; it stays None otherwise (default, off).
        from sgdjscc_lab.video.segment import build_segments
        segments = build_segments(records, structure)

        return {
            "frame_records": [r.to_log() for r in records],
            "keyframe_structure": structure,
            "records": records,
            "segments": segments,
            "segment_records": [s.to_dict() for s in segments],
            "summary": summary,
        }

    @staticmethod
    def _summarize(records: List[FrameRecord]) -> Dict:
        n = len(records)
        n_key = sum(1 for r in records if r.role == "keyframe")
        n_reused = sum(1 for r in records if r.reused)
        transmitted = sum(r.transmitted_units for r in records)
        # Naive baseline: every frame transmits its full packet's units.
        naive = sum(count_units(r.orig_packet or {}) for r in records)
        reduction = (1.0 - transmitted / naive) if naive > 0 else 0.0
        return {
            "n_frames": n,
            "n_keyframes": n_key,
            "n_interframes": n - n_key,
            "n_reused": n_reused,
            "n_generate": sum(1 for r in records if r.decision == "generate"),
            "n_recompute_semantic": sum(1 for r in records if r.decision == "recompute_semantic"),
            "n_recompute_motion": sum(1 for r in records if r.decision == "recompute_motion"),
            "transmitted_units": transmitted,
            "naive_units": naive,
            "overhead_reduction": float(round(reduction, 6)),
        }
