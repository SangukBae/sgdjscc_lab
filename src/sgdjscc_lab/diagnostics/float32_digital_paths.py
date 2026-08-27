"""diagnostics/float32_digital_paths.py – Three instrumented Tx/Rx paths.

Runs the SAME (video, frame, seed) through three paths using only real
production sender/receiver functions:

  awgn                – existing production AWGN path (baseline).
  digital_inprocess   – ``DigitalPacketChannel`` swapped into
                         ``jscc.channel_model``; this is the SAME call site
                         (``pipelines/infer_pipeline.py::_apply_channel`` via
                         ``_encode_and_transmit``) the AWGN path uses — see
                         ``transmission/receiver_runtime.py``'s own docstring,
                         which already calls this "the simple in-process
                         path" (no frame-level packet-bundle byte boundary).
  digital_wire        – the real ``transmission.receiver_runtime`` byte
                         boundary: ``encode_frame_to_bundle_bytes`` produces
                         actual serialized bytes, which are independently
                         parsed back via ``decode_frame_bundle``.

Every numeric operation is a direct call into real production code
(``_encode_and_transmit``, ``_compute_power_scalar``, ``_compute_step``,
``_retransmit_canny``, ``_encode_canny_latent``, ``_run_diffusion``,
``jscc.vae.decode``/``jscc.normalize``, ``decode_frame_bundle``,
``quantize_tensor``/``dequantize_tensor``). The only duplicated logic is the
control-flow glue :func:`instrumented_decode` re-implements from
``_decode_diffusion`` so stage tensors can be captured and ablations can
intercept individual steps — this module does not change
``pipelines/infer_pipeline.py`` at all, so default (non-diagnostic)
transmission behavior is provably unaffected. See
``tests/test_float32_digital_diagnostics.py::test_decode_parity_with_production``
for a bit-exact parity check against calling
``infer_pipeline._decode_diffusion`` directly under the baseline ablation —
it fails loudly if this duplicate orchestration ever drifts from the real one.

``mask_method`` is fixed to ``"none"`` throughout this harness (mirroring
``transmission/receiver_runtime.py``'s own constraint — sender-only mask
statistics cannot legitimately cross the digital_wire receiver boundary), so
all three paths are compared on equal footing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from sgdjscc_lab.diagnostics.ablations import AblationSpec, BASELINE_ABLATION
from sgdjscc_lab.diagnostics.tensor_recorder import TensorRecorder
from sgdjscc_lab.utils.finite_checks import NonFiniteError, assert_finite

PATH_NAMES = ("awgn", "digital_inprocess", "digital_wire")


# ─────────────────────────────────────────────────────────────────────────────
# Recording context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecordCtx:
    """Binds a :class:`TensorRecorder` to one (video, frame, seed, ablation,
    path) coordinate. ``record_enabled=False`` makes ``.rec()`` a no-op
    without touching the recorder (used for every patch except the
    instrumented representative patch — see module docstring's "patch 0"
    convention in the per-frame runners below)."""

    recorder: TensorRecorder
    video: str
    frame: int
    seed: int
    ablation: str
    path: str
    record_enabled: bool = True

    def rec(self, tensor: Any, stage: str) -> None:
        if not self.record_enabled:
            return
        self.recorder.record(
            tensor, video=self.video, frame=self.frame, seed=self.seed,
            ablation=self.ablation, path=self.path, stage=stage,
        )

    def get_live(self, stage: str) -> Optional[torch.Tensor]:
        key = (self.video, self.frame, self.seed, self.ablation, self.path, stage)
        return self.recorder.live.get(key)


def make_null_ctx(recorder: TensorRecorder, *, video: str, frame: int, seed: int,
                   ablation: str, path: str) -> RecordCtx:
    return RecordCtx(recorder=recorder, video=video, frame=frame, seed=seed,
                      ablation=ablation, path=path, record_enabled=False)


def _as_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    if value is None:
        return None
    try:
        return torch.tensor([float(value)])
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: sender encode + channel (AWGN / digital in-process share this)
# ─────────────────────────────────────────────────────────────────────────────

def instrumented_encode_and_transmit(x, jscc, pipe, canny_data, canny_uncertainty, cfg, device, ctx: RecordCtx):
    """Calls the real ``_encode_and_transmit`` unmodified; only ADDS stage
    capture around it (a redundant, deterministic recompute of the sender's
    VAE-encode latent before/after normalization — ``vae.encode(...).mean``
    is not sampled, so this always matches what ``_encode_and_transmit``
    computes internally; see
    ``test_sender_latent_capture_matches_production``)."""
    from sgdjscc_lab.pipelines.infer_pipeline import _SCALING_FACTOR, _encode_and_transmit

    with torch.inference_mode():
        latent_dist = jscc.vae.encode(x * 2 - 1).latent_dist
    pre_norm = latent_dist.mean / _SCALING_FACTOR
    ctx.rec(pre_norm, "sender_vae_latent_pre_norm")
    ctx.rec(jscc.normalize(pre_norm), "sender_vae_latent_post_norm")

    measurement_out: Dict[str, Any] = {}
    artifacts = _encode_and_transmit(
        x, jscc, pipe, canny_data, canny_uncertainty, cfg, device,
        measurement_out=measurement_out,
    )
    ctx.rec(getattr(artifacts, "soft_edge_image", None), "edge_mean")
    ctx.rec(getattr(artifacts, "soft_edge_uncertainty", None), "edge_uncertainty_mean")
    ctx.rec(artifacts.encode_features_hat, "channel_output")
    # Explicit alias, same tensor: awgn/digital_inprocess fold "apply channel"
    # and "normalize" into one call (_apply_channel), so there is no separate
    # deserialize step here -- but digital_wire DOES have one (see
    # run_frame_digital_wire's "receiver_post_norm_latent"), and both need the
    # same stage name to be directly comparable across paths.
    ctx.rec(artifacts.encode_features_hat, "receiver_post_norm_latent")
    ctx.rec(getattr(artifacts, "power_scalar", None), "power_scalar")
    ctx.rec(_as_tensor(getattr(artifacts, "cur_step", None)), "cur_step")
    ctx.rec(_as_tensor(getattr(artifacts, "cur_snr", None)), "cur_snr")
    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: decode (all three paths share this, with ablation hooks)
# ─────────────────────────────────────────────────────────────────────────────

def instrumented_decode(
    artifacts, jscc, pipe, gt_text, cfg, device, ctx: RecordCtx,
    *,
    edge_already_received: bool,
    ablation: AblationSpec = BASELINE_ABLATION,
    awgn_step_ref: Optional[Tuple[Any, Any]] = None,
    original_image: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    """Re-implements ``pipelines/infer_pipeline.py::_decode_diffusion``'s
    control flow (every actual tensor op still calls the real block
    functions it calls), adding stage capture + ablation interception
    points. Returns ``(reconstructed_patch, diffusion_steps_used)``.
    """
    from sgdjscc_lab.pipelines.infer_pipeline import (
        _build_not_control, _encode_canny_latent, _retransmit_canny, _run_diffusion,
    )

    if not artifacts.use_semantic:
        out = (jscc.vae.decode(jscc.normalize(artifacts.encode_features_hat))[0] + 1) / 2
        out = assert_finite(out, "final_reconstruction")
        ctx.rec(out, "final_reconstruction")
        return out, 0

    use_text = bool(cfg.use_text) if ablation.use_text is None else ablation.use_text
    use_controlnet = bool(cfg.use_controlnet) if ablation.use_controlnet is None else ablation.use_controlnet
    use_jscc_feat = bool(cfg.use_jscc_feature)
    canny_cr = str(cfg.canny_cr)
    step_style = str(cfg.step_style)
    diffusion_step = (
        int(ablation.diffusion_step_override)
        if ablation.diffusion_step_override is not None else int(cfg.diffusion_step)
    )
    guidance_scale = float(cfg.guidance_scale)
    ctrl_scale = float(cfg.controlnet_scale)
    cfg_method = str(cfg.cfg_method)
    th = float(cfg.th)

    encode_features_hat = artifacts.encode_features_hat
    mask_token = artifacts.mask_token
    power_scalar = artifacts.power_scalar
    cur_step = artifacts.cur_step
    cur_snr = artifacts.cur_snr
    bsz = artifacts.batch_size

    semantic_text = (
        list(gt_text[0]) if use_text and gt_text is not None else ["" for _ in range(bsz)]
    )

    if ablation.reuse_awgn_step and awgn_step_ref is not None:
        cur_step, cur_snr = awgn_step_ref
    if ablation.fixed_step is not None:
        fixed_step = float(ablation.fixed_step)
        # The production continuous diffusion implementation calls
        # ``cur_step.cpu().numpy()`` and expects one value per sample.  Keep
        # the computed step's tensor contract when possible; the fallback
        # covers configurations where _compute_step returned a scalar.
        if torch.is_tensor(cur_step):
            cur_step = torch.full_like(cur_step, fixed_step)
        else:
            cur_step = torch.full(
                (bsz, 1), fixed_step,
                dtype=encode_features_hat.dtype,
                device=encode_features_hat.device,
            )

    latent_init = (
        encode_features_hat / power_scalar if use_jscc_feat
        else torch.randn_like(encode_features_hat)
    )
    latent_init = assert_finite(latent_init, "diffusion_latent_init")
    ctx.rec(latent_init, "diffusion_latent_init")

    if ablation.bypass_diffusion:
        # True VAE-direct bypass (task requirement): skip Canny retransmission,
        # edge-latent encoding, AND ControlNet/diffusion entirely -- not just
        # the diffusion sampler call. This makes latency/VRAM measured under
        # this ablation reflect ONLY the VAE decode, and removes the edge
        # processing that this ablation's whole point is to rule out (so it
        # cannot itself OOM before diffusion would even have started).
        ctx.rec(None, "edge_post_retransmit")
        ctx.rec(None, "uncertainty_post_ablation")
        ctx.rec(None, "controlnet_input_latent")
        denoised_latent = latent_init
        steps_used = 0
    else:
        edge_off = ablation.use_edge is False
        thresholded = (
            torch.zeros_like(artifacts.soft_edge_image) if edge_off else artifacts.soft_edge_image
        )
        soft_uncertainty = (
            torch.zeros_like(artifacts.soft_edge_uncertainty)
            if (ablation.uncertainty_off or edge_off) else artifacts.soft_edge_uncertainty
        )
        edge_recv = (
            edge_already_received if ablation.force_edge_already_received is None
            else ablation.force_edge_already_received
        )
        if canny_cr != "none" and not edge_recv and not edge_off:
            thresholded = _retransmit_canny(
                jscc, thresholded, soft_uncertainty, cur_snr, canny_cr, th, bsz, device,
            )
        ctx.rec(thresholded, "edge_post_retransmit")
        ctx.rec(soft_uncertainty, "uncertainty_post_ablation")

        canny_latent = _encode_canny_latent(jscc, thresholded, device)
        ctx.rec(canny_latent, "controlnet_input_latent")

        not_control = _build_not_control(encode_features_hat, ctrl_scale, use_controlnet)
        denoised_latent = _run_diffusion(
            pipe=pipe, encode_features_hat=encode_features_hat, power_scalar=power_scalar,
            semantic_text=semantic_text, canny_latent=canny_latent, cur_step=cur_step,
            cfg_method=cfg_method, guidance_scale=guidance_scale, ctrl_scale=ctrl_scale,
            not_control=not_control, use_jscc_feat=use_jscc_feat, use_controlnet=use_controlnet,
            diffusion_step=diffusion_step, step_style=step_style, mask_token=mask_token,
            cfg=cfg,
        )
        steps_used = diffusion_step
    ctx.rec(denoised_latent, "diffusion_latent_final")

    vae_decode_input = jscc.normalize(denoised_latent)
    ctx.rec(vae_decode_input, "vae_decode_input")
    out = (jscc.vae.decode(vae_decode_input)[0] + 1) / 2
    out = assert_finite(out, "final_reconstruction")
    ctx.rec(out, "final_reconstruction")
    return out, steps_used


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame path runners
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PathOutcome:
    path: str
    reconstructed: Optional[torch.Tensor] = None   # [1, 3, H, W] cpu
    latency_ms: float = 0.0
    diffusion_steps: int = 0
    n_patches: int = 0
    wire_bytes: Optional[int] = None
    roundtrip_bitexact: Optional[bool] = None
    cur_step_ref: Optional[Tuple[Any, Any]] = None   # patch-0 (cur_step, cur_snr), for reuse_awgn_step
    failed: bool = False
    failure_stage: Optional[str] = None
    failure_message: Optional[str] = None


def _make_ctx(recorder, *, video, frame, seed, ablation, path, record_enabled):
    return RecordCtx(recorder=recorder, video=video, frame=frame, seed=seed,
                      ablation=ablation, path=path, record_enabled=record_enabled)


def _run_frame_common(
    frame_tensor: torch.Tensor, models, cfg, ablation: AblationSpec, *,
    channel_setup_fn, edge_already_received: bool, path_name: str,
    recorder: TensorRecorder, video: str, frame_index: int, seed: int,
    record_patch_index: Optional[int], awgn_step_ref: Optional[Tuple[Any, Any]],
) -> PathOutcome:
    from sgdjscc_lab.pipelines.infer_pipeline import _extract_semantic_guidance
    from sgdjscc_lab.utils.preprocessing import merge_patches, prepare_patches

    jscc = models.jscc_model
    device = models.device
    channel_setup_fn(jscc)

    patches, patch_meta = prepare_patches(frame_tensor)
    patches = patches.to(device)

    with torch.inference_mode():
        gt_text, canny_data, canny_uncertainty = _extract_semantic_guidance(patches, models, cfg, device)
        if ablation.use_edge is False:
            canny_data = None
            canny_uncertainty = None

    out_patches: List[torch.Tensor] = []
    diffusion_steps_used = 0
    cur_step_ref: Optional[Tuple[Any, Any]] = None
    t0 = time.perf_counter()

    for i in range(patches.shape[0]):
        patch = patches[i:i + 1]
        canny_i = canny_data[i:i + 1] if canny_data is not None else None
        unc_i = canny_uncertainty[i:i + 1] if canny_uncertainty is not None else None
        gt_text_i = [[gt_text[0][i]]] if (gt_text and gt_text[0]) else None
        record_this = record_patch_index is not None and i == record_patch_index
        ctx = _make_ctx(recorder, video=video, frame=frame_index, seed=seed,
                         ablation=ablation.name, path=path_name, record_enabled=record_this)

        with torch.inference_mode():
            artifacts = instrumented_encode_and_transmit(
                patch, jscc, models.sem_pipeline, canny_i, unc_i, cfg, device, ctx,
            )
            out, n_steps = instrumented_decode(
                artifacts, jscc, models.sem_pipeline, gt_text_i, cfg, device, ctx,
                edge_already_received=edge_already_received, ablation=ablation,
                awgn_step_ref=awgn_step_ref, original_image=patch,
            )
        if i == 0:
            cur_step_ref = (artifacts.cur_step, artifacts.cur_snr)
        out_patches.append(out.cpu())
        diffusion_steps_used = n_steps

    latency_ms = (time.perf_counter() - t0) * 1000.0
    recon = merge_patches(torch.cat(out_patches, dim=0), patch_meta)
    return PathOutcome(
        path=path_name, reconstructed=recon, latency_ms=latency_ms,
        diffusion_steps=diffusion_steps_used, n_patches=int(patches.shape[0]),
        cur_step_ref=cur_step_ref,
    )


def run_frame_awgn(
    frame_tensor, models, cfg, ablation: AblationSpec, *, recorder: TensorRecorder,
    video: str, frame_index: int, seed: int, record_patch_index: Optional[int] = 0,
) -> PathOutcome:
    def setup(jscc):
        jscc.channel_model = None

    return _run_frame_common(
        frame_tensor, models, cfg, ablation, channel_setup_fn=setup,
        edge_already_received=False, path_name="awgn", recorder=recorder,
        video=video, frame_index=frame_index, seed=seed,
        record_patch_index=record_patch_index, awgn_step_ref=None,
    )


def run_frame_digital_inprocess(
    frame_tensor, models, cfg, ablation: AblationSpec, *, bit_depth: int, granularity: str,
    recorder: TensorRecorder, video: str, frame_index: int, seed: int,
    record_patch_index: Optional[int] = 0, awgn_step_ref: Optional[Tuple[Any, Any]] = None,
) -> PathOutcome:
    from sgdjscc_lab.channels.digital_packet import DigitalPacketChannel

    def setup(jscc):
        jscc.channel_model = DigitalPacketChannel(bit_depth=bit_depth, granularity=granularity, channel_dim=1)

    return _run_frame_common(
        frame_tensor, models, cfg, ablation, channel_setup_fn=setup,
        edge_already_received=False, path_name="digital_inprocess", recorder=recorder,
        video=video, frame_index=frame_index, seed=seed,
        record_patch_index=record_patch_index, awgn_step_ref=awgn_step_ref,
    )


def run_frame_digital_wire(
    frame_tensor, models, cfg, ablation: AblationSpec, *, bit_depth: int, granularity: str,
    digital_step_policy: str, recorder: TensorRecorder, video: str, frame_index: int, seed: int,
    record_patch_index: Optional[int] = 0, awgn_step_ref: Optional[Tuple[Any, Any]] = None,
) -> PathOutcome:
    """Uses the real ``encode_frame_to_bundle_bytes`` / ``decode_frame_bundle``
    byte boundary. Duplicates only ``receiver_runtime.py::
    reconstruct_frame_from_bundle_bytes``'s per-patch orchestration (same
    imports it uses) so per-stage tensors can be captured; every tensor
    computation is still the real production function.
    """
    from sgdjscc_lab.pipelines.infer_pipeline import (
        ForwardArtifacts, _SCALING_FACTOR, _compute_power_scalar, _compute_step,
        _encode_latent, _preprocess_soft_edge,
    )
    from sgdjscc_lab.transmission.packet_bundle import decode_frame_bundle
    from sgdjscc_lab.transmission.receiver_runtime import encode_frame_to_bundle_bytes
    from sgdjscc_lab.utils.preprocessing import merge_patches, prepare_patches

    if str(cfg.get("mask_method", "none")) != "none":
        raise ValueError("digital_wire path requires mask_method=none (see receiver_runtime.py)")

    jscc = models.jscc_model
    device = models.device
    policy = ablation.digital_step_policy_override or digital_step_policy

    patches, patch_meta = prepare_patches(frame_tensor)
    patches = patches.to(device)

    with torch.inference_mode():
        encode_features, _std = _encode_latent(jscc, patches)

    data, _n_elements = encode_frame_to_bundle_bytes(
        frame_tensor, models, cfg, bit_depth=bit_depth, granularity=granularity,
        keyframe_index=int(frame_index), manifest={}, selected_keyframes=[int(frame_index)],
    )
    wire_bytes = len(data)

    decoded = decode_frame_bundle(data, dtype=torch.float32, device=str(device))
    if decoded["visual_is_analog"]:
        raise ValueError("digital_wire path received an analog bundle; expected digital visual latents")
    visual_latents = decoded["visual_latents"] or []
    visual_metadata = decoded.get("visual_metadata") or []
    captions = list(decoded.get("captions") or [])
    if len(captions) == 1 and len(visual_latents) > 1:
        captions *= len(visual_latents)
    edge = decoded.get("edge")
    edge_uncertainty = decoded.get("edge_uncertainty")
    if ablation.use_edge is False:
        edge = None
        edge_uncertainty = None

    record0 = record_patch_index is not None and len(visual_latents) > 0
    ctx0 = _make_ctx(recorder, video=video, frame=frame_index, seed=seed,
                      ablation=ablation.name, path="digital_wire", record_enabled=record0)
    if record0:
        p = min(record_patch_index, patches.shape[0] - 1)
        with torch.inference_mode():
            pre_norm = jscc.vae.encode(patches[p:p + 1] * 2 - 1).latent_dist.mean / _SCALING_FACTOR
        ctx0.rec(pre_norm, "sender_vae_latent_pre_norm")
        ctx0.rec(encode_features[p:p + 1], "sender_vae_latent_post_norm")
        # Explicit alias, same tensor: this is what actually gets handed to
        # encode_frame_to_bundle_bytes() for serialization (task requirement
        # for a named "immediately before serialize" stage, distinct from
        # the cross-path sender_vae_latent_post_norm comparability stage).
        ctx0.rec(encode_features[p:p + 1], "pre_serialize_latent")
        if p < len(visual_latents):
            ctx0.rec(visual_latents[p], "post_deserialize_latent_raw")

    roundtrip_bitexact: Optional[bool] = None
    if bit_depth == 32 and visual_latents:
        p = min(record_patch_index or 0, len(visual_latents) - 1, patches.shape[0] - 1)
        roundtrip_bitexact = torch.equal(
            encode_features[p:p + 1].detach().cpu().contiguous().float(),
            visual_latents[p].detach().cpu().contiguous().float(),
        )

    snr_scale = 10 ** (float(jscc.snr) / 10)
    out_patches: List[torch.Tensor] = []
    diffusion_steps_used = 0
    cur_step_ref: Optional[Tuple[Any, Any]] = None
    t0 = time.perf_counter()

    with torch.inference_mode():
        for i, received in enumerate(visual_latents):
            record_this = record_patch_index is not None and i == record_patch_index
            ctx_i = _make_ctx(recorder, video=video, frame=frame_index, seed=seed,
                               ablation=ablation.name, path="digital_wire", record_enabled=record_this)

            encode_features_hat = jscc.normalize(received.to(device))
            # Receiver-side normalization output -- the wire-path analogue of
            # what _apply_channel() computes in-line for awgn/digital_inprocess
            # (there it is folded into "channel_output"; here deserialization
            # and normalization are two separate steps, so this stage makes the
            # post-normalization point explicit and directly comparable).
            ctx_i.rec(encode_features_hat, "receiver_post_norm_latent")

            dummy_x = torch.empty(
                encode_features_hat.shape[0], 1, 1, 1, device=device, dtype=encode_features_hat.dtype,
            )
            edge_i = edge[i:i + 1].to(device) if edge is not None else None
            unc_i = (
                edge_uncertainty[i:i + 1].to(device)
                if edge_uncertainty is not None else (torch.zeros_like(edge_i) if edge_i is not None else None)
            )
            soft_edge, soft_unc = _preprocess_soft_edge(edge_i, unc_i, dummy_x, device)
            ctx_i.rec(soft_edge, "edge_mean")
            ctx_i.rec(soft_unc, "edge_uncertainty_mean")

            power_scalar = _compute_power_scalar(encode_features_hat, None, dummy_x)
            ctx_i.rec(power_scalar, "power_scalar")
            signal_scale = (snr_scale / (snr_scale + 1)) * torch.ones_like(encode_features_hat[:, 0:1, 0, 0])
            meta_i = visual_metadata[i] if i < len(visual_metadata) else {"bit_depth": bit_depth}
            cur_step, cur_snr = _compute_step(
                jscc=jscc, encode_features_hat=encode_features_hat, power_scalar=power_scalar,
                signal_scale=signal_scale, pipe=models.sem_pipeline, step_style=str(cfg.step_style),
                use_jscc_feat=bool(cfg.use_jscc_feature), use_gt_csi=bool(cfg.use_gt_csi), device=device,
                digital_bit_depth=meta_i.get("bit_depth", bit_depth), digital_policy=policy,
                digital_quant_snr_db=meta_i.get("quant_snr_db"),
            )
            ctx_i.rec(_as_tensor(cur_step), "cur_step")
            ctx_i.rec(_as_tensor(cur_snr), "cur_snr")
            artifacts = ForwardArtifacts(
                use_semantic=bool(cfg.use_semantic), encode_features_hat=encode_features_hat,
                signal_scale=signal_scale, device=device, batch_size=encode_features_hat.shape[0],
                mask_token=None, power_scalar=power_scalar, cur_step=cur_step, cur_snr=cur_snr,
                soft_edge_image=soft_edge, soft_edge_uncertainty=soft_unc,
            )
            gt_text_i = [[captions[i]]] if i < len(captions) else [[""]]
            out, n_steps = instrumented_decode(
                artifacts, jscc, models.sem_pipeline, gt_text_i, cfg, device, ctx_i,
                edge_already_received=True, ablation=ablation, awgn_step_ref=awgn_step_ref,
            )
            if i == 0:
                cur_step_ref = (cur_step, cur_snr)
            out_patches.append(out.cpu())
            diffusion_steps_used = n_steps

    latency_ms = (time.perf_counter() - t0) * 1000.0
    recon = merge_patches(torch.cat(out_patches, dim=0), patch_meta)
    return PathOutcome(
        path="digital_wire", reconstructed=recon, latency_ms=latency_ms,
        diffusion_steps=diffusion_steps_used, n_patches=int(patches.shape[0]),
        wire_bytes=wire_bytes, roundtrip_bitexact=roundtrip_bitexact,
        cur_step_ref=cur_step_ref,
    )


def run_path_with_failure_capture(run_fn, path_name: str, *args, **kwargs) -> PathOutcome:
    """Wraps a ``run_frame_*`` call, catching :class:`NonFiniteError`
    specifically (never a bare ``except Exception`` — see
    ``utils/finite_checks.py``'s own guidance) so a deliberately-degraded
    config (e.g. an aggressive ablation) produces a documented failed row
    instead of aborting the whole sweep; any other exception is a real bug
    and propagates.
    """
    try:
        return run_fn(*args, **kwargs)
    except NonFiniteError as exc:
        return PathOutcome(path=path_name, failed=True, failure_stage=exc.stage, failure_message=str(exc))
