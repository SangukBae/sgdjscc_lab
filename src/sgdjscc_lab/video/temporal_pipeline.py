"""video/temporal_pipeline.py – Keyframe/inter-frame temporal orchestration (Phase 4-B).

Drives an ordered frame sequence through the keyframe-oriented semantic pipeline:

- **keyframe**     : run the full image pipeline + build a full semantic packet
                     (this is the reference for the GOP).
- **inter-frame**  : compute the semantic delta vs the latest keyframe packet and,
                     depending on its magnitude, either *reuse* the keyframe
                     reconstruction (cheap, no diffusion) or *recompute* with
                     guidance attenuated in proportion to the change.

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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

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
    recon: Optional[object] = None   # reconstructed frame tensor (not logged)

    def to_log(self) -> Dict:
        return {
            "index": self.index,
            "role": self.role,
            "reused": self.reused,
            "guidance_scale": self.guidance_scale,
            "transmitted_units": self.transmitted_units,
            "magnitude": (self.delta or {}).get("magnitude") if self.delta else None,
            "num_changes": (self.delta or {}).get("num_changes") if self.delta else None,
            "srs": self.srs,
        }


def count_units(packet: Dict) -> int:
    """Count transmittable semantic units in a packet (objects + relations + attrs)."""
    objs = len(packet.get("objects") or [])
    rels = len(packet.get("relations") or [])
    attrs = sum(len(v) for v in (packet.get("attributes") or {}).values())
    scene = 1 if packet.get("scene") else 0
    return objs + rels + attrs + scene


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
        Delta magnitude below which an inter-frame reuses the keyframe
        reconstruction instead of recomputing.
    diffusion_step:
        Single source of truth for the denoising-step count used by **both** the
        staged schedule (stage boundaries) and the per-frame reconstruction
        config (``_frame_cfg`` forces ``cfg.diffusion_step`` to this value).  If
        None, it is taken from ``cfg.diffusion_step`` (default 50), so the stage
        plan can never silently disagree with the actual denoising.
    srs_fn:
        Optional ``(orig_packet, recon_packet) -> srs`` to fill per-frame SRS
        (e.g. ``SemanticReliabilityEvaluator.score_packet`` wrapper).
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
        diffusion_step: Optional[int] = None,
        srs_fn: Optional[Callable] = None,
    ) -> None:
        self.reconstruct_fn = reconstruct_fn
        self.packet_fn = packet_fn
        self.cfg = cfg
        self.reuse_threshold = reuse_threshold
        # Resolve the authoritative step count: explicit arg wins, else cfg, else 50.
        if diffusion_step is not None:
            self.diffusion_step = int(diffusion_step)
        elif cfg is not None:
            self.diffusion_step = int(cfg.get("diffusion_step", 50))
        else:
            self.diffusion_step = 50
        self.srs_fn = srs_fn

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

        records: List[FrameRecord] = []
        # GOP anchor — both the packet reference (for delta) and the pixel
        # reference (for reuse) are the current keyframe, kept consistent.  These
        # are NOT advanced by inter-frame recomputations.
        keyframe_packet: Optional[Dict] = None
        keyframe_recon = None
        keyframe_recon_packet: Optional[Dict] = None

        for i in range(n):
            frame = frames[i]
            curr_packet = self.packet_fn(frame, f"frame_{i:05d}")
            schedule = build_staged_schedule(curr_packet, self.diffusion_step)

            if roles[i] == "keyframe":
                cfg_i, gs = self._frame_cfg(schedule, magnitude=None)
                recon = self.reconstruct_fn(frame, cfg_i)
                recon_packet = self.packet_fn(recon, f"recon_{i:05d}")
                # Set the GOP anchor (packet + pixel reference together).
                keyframe_packet = curr_packet
                keyframe_recon = recon
                keyframe_recon_packet = recon_packet
                rec = FrameRecord(
                    index=i, role="keyframe", reused=False, delta=None,
                    guidance_scale=gs,
                    transmitted_units=count_units(curr_packet),
                    orig_packet=curr_packet, recon_packet=recon_packet,
                    staged_schedule=schedule,
                )
            else:
                # Reference = keyframe packet (consistent with the pixel reference).
                d = self.delta.compute(keyframe_packet or {}, curr_packet)
                if d["magnitude"] < self.reuse_threshold:
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
                    )
                else:
                    # Recompute this frame with the staged prompt + attenuated
                    # guidance.  The GOP anchor is left unchanged so subsequent
                    # inter-frames keep referencing the keyframe.
                    cfg_i, gs = self._frame_cfg(schedule, magnitude=d["magnitude"])
                    recon = self.reconstruct_fn(frame, cfg_i)
                    recon_packet = self.packet_fn(recon, f"recon_{i:05d}")
                    rec = FrameRecord(
                        index=i, role="inter", reused=False, delta=d,
                        guidance_scale=gs,
                        transmitted_units=d["num_changes"],
                        orig_packet=curr_packet, recon_packet=recon_packet,
                        staged_schedule=schedule,
                    )

            rec.recon = recon
            if self.srs_fn is not None and rec.orig_packet and rec.recon_packet:
                try:
                    rec.srs = float(self.srs_fn(rec.orig_packet, rec.recon_packet))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("srs_fn failed on frame %d: %s", i, exc)

            records.append(rec)

        summary = self._summarize(records)
        return {
            "frame_records": [r.to_log() for r in records],
            "keyframe_structure": structure,
            "records": records,
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
            "transmitted_units": transmitted,
            "naive_units": naive,
            "overhead_reduction": float(round(reduction, 6)),
        }
