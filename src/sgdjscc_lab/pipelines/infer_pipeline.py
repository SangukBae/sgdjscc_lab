"""pipelines/infer_pipeline.py – Core SGDJSCC AWGN inference logic.

Migrated from src/sgdjscc_lab/pipeline.py with the following structural
changes (Phase 2):

  • _extract_caption / _extract_canny replaced by TextExtractor / EdgeExtractor
  • _jscc_forward decomposed into focused block helpers
  • SGDJSCC path injection centralised via _sgdjscc.ensure_sgdjscc_on_path()
  • memory release delegated to utils.memory.release_cuda_memory()

Algorithm is identical to SGDJSCC/inference_one.py JSCC_model.forward():
  1. Soft edge pre-processing (canny_data → mean/uncertainty maps)
  2. VAE encode (x*2-1 → latent / scaling_factor → L2-normalise)
  3. AWGN channel + renormalise
  4. Non-semantic fallback: VAE decode
  5. Semantic path:
       a. Mask token + power scalar
       b. Blind SNR estimation / step matching
       c. Canny JSCC retransmission
       d. Canny latent encoding
       e. Diffusion denoising (DPM-Solver++ 2M)
  6. Final decode: (vae.decode(normalize(denoised)) + 1) / 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from sgdjscc_lab._sgdjscc import ensure_sgdjscc_on_path

ensure_sgdjscc_on_path()
from utils.utils import generate_mask  # noqa: E402

logger = logging.getLogger(__name__)

# VAE scaling factor – identical to inference_one.py
_SCALING_FACTOR = 15.45


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_single_image(
    img_tensor: torch.Tensor,
    models,
    cfg: DictConfig,
    measurement_out: Optional[dict] = None,
) -> torch.Tensor:
    """Run SGDJSCC AWGN inference on a single ``[N, 3, 128, 128]`` patch batch.

    Parameters
    ----------
    img_tensor:
        ``[N, 3, 128, 128]`` float in [0, 1], already on the correct device.
    models:
        ModelBundle from runtime.build_models().
    cfg:
        Loaded OmegaConf config.
    measurement_out:
        Optional dict; when provided it is populated (observation only, numerics
        unchanged) with real receiver evidence — ``encode_features_hat``,
        ``mask_token``, ``power_scalar``, ``signal_scale`` and the estimated SNR —
        for Phase 5-A channel conditioning (see ``MeasurementBundle``).

    Returns
    -------
    torch.Tensor
        ``[N, 3, 128, 128]`` float in [0, 1] – reconstructed patches.
    """
    device = models.device

    x = img_tensor.to(device)

    # ── Step 1: Extract semantic guidance ────────────────────────────────────
    gt_text, canny_data, canny_uncertainty = _extract_semantic_guidance(x, models, cfg, device)

    # ── Step 2: Forward pass ─────────────────────────────────────────────────
    with torch.inference_mode():
        result = _jscc_forward(
            x=x,
            jscc=models.jscc_model,
            pipe=models.sem_pipeline,
            gt_text=gt_text,
            canny_data=canny_data,
            canny_uncertainty=canny_uncertainty,
            cfg=cfg,
            device=device,
            measurement_out=measurement_out,
        )
    return result


def run_batch(
    input_path: str,
    output_dir: str,
    cfg: DictConfig,
    models,
) -> None:
    """Process all images at *input_path* and write results to *output_dir*.

    Handles both single-file and directory inputs.  Per-image errors are
    logged but do not abort the batch.
    """
    from sgdjscc_lab.io import list_image_files, load_image_as_tensor, save_tensor_as_image
    from sgdjscc_lab.utils.preprocessing import prepare_patches, merge_patches

    files = list_image_files(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Found %d image(s). SNR = %.1f dB", len(files), cfg.snr_db)

    for idx, fpath in enumerate(files, start=1):
        logger.info("[%d/%d] %s", idx, len(files), fpath.name)
        try:
            raw = load_image_as_tensor(fpath)          # [1, 3, H, W]
            patches, meta = prepare_patches(raw)        # [N, 3, 128, 128]
            patches = patches.to(models.device)

            out_patches = _process_patches(patches, models, cfg)

            reconstructed = merge_patches(out_patches.cpu(), meta)  # [1, 3, H, W]
            out_path = output_dir / (fpath.stem + ".png")
            save_tensor_as_image(reconstructed, out_path)

        except Exception as exc:
            logger.error("Failed on %s: %s", fpath.name, exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# Patch-loop helper
# ─────────────────────────────────────────────────────────────────────────────

def _process_patches(
    patches: torch.Tensor,
    models,
    cfg: DictConfig,
) -> torch.Tensor:
    """Run run_single_image on every 128×128 patch one at a time."""
    out_list = []
    for i in range(patches.shape[0]):
        patch = patches[i : i + 1]
        out = run_single_image(patch, models, cfg)
        out_list.append(out.cpu())
    return torch.cat(out_list, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic-guidance extraction (shared by the standard + one-pass paths)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_semantic_guidance(x: torch.Tensor, models, cfg: DictConfig, device):
    """Return ``(gt_text, canny_data, canny_uncertainty)`` for a patch batch.

    Honours the Phase 4 ``cfg.prompt_override`` contract: when set, it replaces
    the extracted caption as the diffusion text condition.
    """
    use_semantic = bool(cfg.use_semantic)
    use_text     = bool(cfg.use_text)
    gt_text = None
    canny_data = None
    canny_uncertainty = None
    prompt_override = cfg.get("prompt_override", None)

    if use_semantic:
        if use_text and prompt_override is not None:
            gt_text = [[str(prompt_override) for _ in range(x.shape[0])]]
        elif use_text and models.text_extractor is not None:
            gt_text = models.text_extractor.extract(
                x, device,
                offload_device=models.offload_device,
                offload_after=models.offload_caption,
            )
        if models.edge_extractor is not None:
            canny_data, canny_uncertainty = models.edge_extractor.extract(
                x, device,
                offload_device=models.offload_device,
                offload_after=models.offload_canny,
            )
        else:
            canny_data = torch.zeros(x.shape[0], 11, 128, 128, device=device)
            canny_uncertainty = torch.zeros_like(canny_data)
    return gt_text, canny_data, canny_uncertainty


# ─────────────────────────────────────────────────────────────────────────────
# One-pass channel-conditioned reconstruction (Phase 5-A/5-B)
# ─────────────────────────────────────────────────────────────────────────────

def run_image_channel_conditioned(patches, models, base_cfg, wrapper,
                                   csi="perfect", condition_mode="auto"):
    """Channel-conditioned reconstruction of a patched image in a single pass.

    Phase 1 encodes + transmits **every** patch once (capturing per-patch receiver
    evidence, no diffusion); the bundles are aggregated to an image-level
    measurement which the wrapper turns into a conditioned cfg; phase 2 runs the
    diffusion decode per patch **reusing** the phase-1 received latent.  The channel
    is therefore sampled once per patch (no extra measurement forward), the
    observed and decoded realisations are identical by construction, and
    conditioning reflects the whole image.

    Returns ``(out_patches[N,3,128,128] cpu, info)``.
    """
    from sgdjscc_lab.channels.measurement import aggregate_bundles

    jscc = models.jscc_model
    pipe = models.sem_pipeline
    device = models.device

    artifacts_list, gt_text_list, orig_list, bundles = [], [], [], []
    with torch.inference_mode():
        for i in range(patches.shape[0]):
            patch = patches[i:i + 1].to(device)
            gt_text, canny_data, canny_unc = _extract_semantic_guidance(patch, models, base_cfg, device)
            art = _encode_and_transmit(
                patch, jscc, pipe, canny_data, canny_unc, base_cfg, device,
                build_bundle=True,
            )
            artifacts_list.append(art)
            gt_text_list.append(gt_text)
            orig_list.append(patch)
            if art.bundle is not None:
                bundles.append(art.bundle)

    agg = aggregate_bundles(bundles)
    if agg is None:
        from sgdjscc_lab.channels.measurement import MeasurementBundle
        agg = MeasurementBundle(received=artifacts_list[0].encode_features_hat,
                                snr_db_true=float(getattr(jscc, "snr", 0.0)))
    cond_cfg, info = wrapper.build_conditioned_cfg(
        base_cfg, agg, mode=condition_mode, csi=csi,
    )
    # Expose the actually-applied conditioned cfg for logging / debugging.
    info["resolved_cfg"] = cond_cfg

    out_list = []
    with torch.inference_mode():
        for art, gt_text, patch in zip(artifacts_list, gt_text_list, orig_list):
            out = _decode_diffusion(art, jscc, pipe, gt_text, cond_cfg, device,
                                    original_image=patch)
            out_list.append(out.cpu())

    return torch.cat(out_list, dim=0), info


# ─────────────────────────────────────────────────────────────────────────────
# Core forward pass (mirrors JSCC_model.forward in inference_one.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForwardArtifacts:
    """Pre-diffusion state from one patch's encode+channel pass (Phase 5-B/one-pass).

    Produced by :func:`_encode_and_transmit` and consumed by
    :func:`_decode_diffusion`, so the channel is sampled **once** and the same
    received latent that the condition is derived from feeds the decoder.  This is
    what lets channel-conditioned inference run in a single forward (no throwaway
    measurement pass).
    """

    use_semantic: bool
    encode_features_hat: torch.Tensor
    signal_scale: torch.Tensor
    device: Any
    batch_size: int
    # Semantic-path fields (None on the non-semantic fallback path).
    mask_token: Any = None
    power_scalar: Any = None
    cur_step: Any = None
    cur_snr: Any = None
    soft_edge_image: Any = None
    soft_edge_uncertainty: Any = None
    bundle: Any = None   # per-patch MeasurementBundle (receiver evidence)


def _build_evidence_bundle(jscc, artifacts_fields: dict):
    """Build a per-patch MeasurementBundle whose condition source == decoder init.

    The channel-condition encoder reads ``bundle.best_estimate`` as the received
    feature, while the decoder uses ``decoder_init`` (the post-mask received latent,
    normalised by the power scalar) as its diffusion initialisation.  To keep the
    "condition source == diffusion init" contract, we set ``received``/``equalized``
    to ``decoder_init`` (so ``best_estimate`` returns exactly that), and copy the
    channel-level descriptors (gain / noise / mask / reliability / SNR) from the
    channel's realisation without mutating that shared object.
    """
    from sgdjscc_lab.channels.measurement import MeasurementBundle

    ef = artifacts_fields.get("encode_features_hat")
    decoder_init = artifacts_fields.get("decoder_init", ef)
    src = getattr(getattr(jscc, "channel_model", None), "last_bundle", None)

    if src is not None:
        bundle = MeasurementBundle(
            received=decoder_init,
            equalized=decoder_init,        # best_estimate == the decoder's init latent
            channel_gain=src.channel_gain,
            noise_var=src.noise_var,
            # Per-element noise level d (paper eq. 12) — REQUIRED so each patch's
            # fast-fading water-filling decode uses its OWN d, not the global
            # channel last_bundle (which is the last patch's realisation).
            noise_level=src.noise_level,
            mask=src.mask,
            reliability=src.reliability,
            phase_est=src.phase_est,
            snr_db_true=src.snr_db_true,
            snr_db_est=src.snr_db_est,
            meta=dict(src.meta),
        )
    else:
        bundle = MeasurementBundle(
            received=decoder_init, equalized=decoder_init,
            snr_db_true=float(getattr(jscc, "snr", 0.0)), meta={"channel": "awgn"},
        )

    bundle.encode_features_hat = ef
    bundle.mask_token = artifacts_fields.get("mask_token", bundle.mask_token)
    bundle.power_scalar = artifacts_fields.get("power_scalar", bundle.power_scalar)
    if artifacts_fields.get("snr_db_est") is not None:
        bundle.snr_db_est = artifacts_fields["snr_db_est"]
    return bundle


def _encode_and_transmit(
    x: torch.Tensor,
    jscc,
    pipe,
    canny_data: Optional[torch.Tensor],
    canny_uncertainty: Optional[torch.Tensor],
    cfg: DictConfig,
    device: torch.device,
    measurement_out: Optional[dict] = None,
    build_bundle: bool = False,
) -> ForwardArtifacts:
    """Stage 1: VAE encode → channel → mask/power → step match (no diffusion).

    Mirrors Blocks 1–3 and 5b–5c of the original ``_jscc_forward`` in the same
    order, so the standard path is numerically unchanged.  Returns the
    pre-diffusion :class:`ForwardArtifacts`.
    """
    use_semantic = bool(cfg.use_semantic)
    mask_method  = str(cfg.mask_method)
    step_style   = str(cfg.step_style)
    use_jscc_feat = bool(cfg.use_jscc_feature)
    use_gt_csi    = bool(cfg.use_gt_csi)

    # Block 1: soft edge pre-processing
    soft_edge_image, soft_edge_uncertainty = _preprocess_soft_edge(
        canny_data, canny_uncertainty, x, device
    )
    # Block 2: VAE encode + normalise
    encode_features, encode_features_std = _encode_latent(jscc, x)
    # Block 3: channel + renormalise
    encode_features_hat, signal_scale = _apply_channel(jscc, encode_features)

    if measurement_out is not None:
        measurement_out["encode_features_hat"] = encode_features_hat
        measurement_out["signal_scale"] = signal_scale

    if not use_semantic:
        art = ForwardArtifacts(
            use_semantic=False, encode_features_hat=encode_features_hat,
            signal_scale=signal_scale, device=device, batch_size=x.shape[0],
        )
        if build_bundle:
            art.bundle = _build_evidence_bundle(
                jscc, {"encode_features_hat": encode_features_hat})
        return art

    # Block 5b: mask token + power scalar
    mask_token = generate_mask(
        encode_features, encode_features_std, soft_edge_image, mask_method
    )
    encode_features_hat = _apply_mask_token(encode_features_hat, mask_token)
    power_scalar = _compute_power_scalar(encode_features_hat, mask_token, x)

    # Block 5c: step matching
    cur_step, cur_snr = _compute_step(
        jscc=jscc, encode_features_hat=encode_features_hat, power_scalar=power_scalar,
        signal_scale=signal_scale, pipe=pipe, step_style=step_style,
        use_jscc_feat=use_jscc_feat, use_gt_csi=use_gt_csi, device=device,
    )

    snr_est = None
    try:
        snr_est = float(cur_snr.mean().item() if hasattr(cur_snr, "mean") else cur_snr)
    except Exception:  # noqa: BLE001
        snr_est = None

    if measurement_out is not None:
        measurement_out["mask_token"] = mask_token
        measurement_out["power_scalar"] = power_scalar
        measurement_out["cur_step"] = cur_step
        measurement_out["snr_db_est"] = snr_est

    art = ForwardArtifacts(
        use_semantic=True, encode_features_hat=encode_features_hat,
        signal_scale=signal_scale, device=device, batch_size=x.shape[0],
        mask_token=mask_token, power_scalar=power_scalar,
        cur_step=cur_step, cur_snr=cur_snr,
        soft_edge_image=soft_edge_image, soft_edge_uncertainty=soft_edge_uncertainty,
    )
    if build_bundle:
        # decoder_init is exactly the diffusion init the decoder uses when
        # use_jscc_feature is on (which the channel-conditioned path forces):
        # encode_features_hat / power_scalar.  Making this the bundle's
        # best_estimate guarantees condition source == diffusion init.
        decoder_init = encode_features_hat / power_scalar
        art.bundle = _build_evidence_bundle(jscc, {
            "encode_features_hat": encode_features_hat, "decoder_init": decoder_init,
            "mask_token": mask_token, "power_scalar": power_scalar, "snr_db_est": snr_est,
        })
    return art


def _build_early_exit_score_fn(metric, decode_fn, original_image, verifier):
    """Build a verified in-loop score_fn for intra-sampler early-exit.

    Returns None for the ``heuristic`` metric (the wrapper then uses its built-in
    latent-convergence score) or when the verifier / original / decode are missing.
    Otherwise returns ``score_fn(state, i, total)`` that decodes the loop's current
    clean-latent prediction (``state["x0"]``) and scores it against the original
    patch with the configured verifier (SRS / SRS-v2).
    """
    if metric == "heuristic" or original_image is None or verifier is None or decode_fn is None:
        return None

    def _score(state, i, total):
        x0 = state.get("x0") if isinstance(state, dict) else None
        if x0 is None:
            return 0.0
        try:
            return float(verifier(original_image, decode_fn(x0)))
        except Exception:  # noqa: BLE001
            return 0.0

    return _score


def _get_srs_verifier(jscc, metric, device=None):
    """Lazily build + cache a reference SRS / SRS-v2 verifier on the jscc model.

    The evaluator is built on *device* (the diffusion model's device) so it does
    not spawn a second CLIP on CPU.  Returns ``verify(original, recon) -> float``
    or None when the evaluator (e.g. CLIP) is unavailable.
    """
    attr = "_early_exit_verifier_" + metric
    cached = getattr(jscc, attr, None)
    if cached is not None:
        return cached
    try:
        from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
        base = SemanticReliabilityEvaluator(device=device)
        if metric == "srs_v2":
            from sgdjscc_lab.evaluators.semantic_reliability_v2 import SemanticReliabilityV2Evaluator
            ev = SemanticReliabilityV2Evaluator(base_evaluator=base)
            fn = lambda o, r: (ev.evaluate(o, r).get("srs_v2") or -999.0)
        else:
            fn = lambda o, r: base.evaluate(o, r).get("semantic_reliability_score", -999.0)
        setattr(jscc, attr, fn)
        return fn
    except Exception as exc:  # noqa: BLE001
        logger.warning("early-exit %s verifier unavailable (%s); using heuristic.", metric, exc)
        return None


def _resolve_early_exit_score_fn(cfg, jscc, original_image, device=None):
    """Pick the in-loop early-exit score_fn from ``cfg.acceleration.early_exit_metric``."""
    acc = cfg.get("acceleration", None) if cfg is not None else None
    if acc is None or not bool(acc.get("early_exit", False)):
        return None
    if str(acc.get("early_exit_mode", "intra_sampler")) != "intra_sampler":
        return None
    metric = str(acc.get("early_exit_metric", "heuristic"))
    if metric == "heuristic":
        return None   # wrapper uses its built-in convergence heuristic
    if original_image is None:
        logger.warning("early_exit_metric=%s needs the original image; using heuristic.", metric)
        return None
    verifier = _get_srs_verifier(jscc, metric, device=device)
    decode_fn = lambda x0: (jscc.vae.decode(jscc.normalize(x0))[0] + 1) / 2
    score_fn = _build_early_exit_score_fn(metric, decode_fn, original_image, verifier)
    if score_fn is None:
        logger.warning("early_exit_metric=%s could not be wired; using heuristic.", metric)
    return score_fn


def _decode_diffusion(
    artifacts: ForwardArtifacts,
    jscc,
    pipe,
    gt_text,
    cfg: DictConfig,
    device: torch.device,
    original_image: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Stage 2: canny retransmit → ControlNet latent → diffusion → final decode.

    Mirrors Blocks 5a, 5d–5f and 6 of the original ``_jscc_forward``.  Reuses the
    received latent captured in *artifacts*, so the channel realisation is never
    re-sampled between conditioning and reconstruction.  ``original_image`` (the
    input patch) enables the verified intra-sampler early-exit metrics.
    """
    if not artifacts.use_semantic:
        return (jscc.vae.decode(jscc.normalize(artifacts.encode_features_hat))[0] + 1) / 2

    use_text       = bool(cfg.use_text)
    use_controlnet = bool(cfg.use_controlnet)
    use_jscc_feat  = bool(cfg.use_jscc_feature)
    canny_cr       = str(cfg.canny_cr)
    step_style     = str(cfg.step_style)
    diffusion_step = int(cfg.diffusion_step)
    guidance_scale = float(cfg.guidance_scale)
    ctrl_scale     = float(cfg.controlnet_scale)
    cfg_method     = str(cfg.cfg_method)
    th             = float(cfg.th)

    encode_features_hat = artifacts.encode_features_hat
    mask_token   = artifacts.mask_token
    power_scalar = artifacts.power_scalar
    cur_step     = artifacts.cur_step
    cur_snr      = artifacts.cur_snr
    thresholded  = artifacts.soft_edge_image
    bsz          = artifacts.batch_size

    # Block 5a: text guidance
    semantic_text = (
        list(gt_text[0]) if use_text and gt_text is not None
        else ["" for _ in range(bsz)]
    )

    # Block 5d: canny JSCC retransmission
    if canny_cr != "none":
        thresholded = _retransmit_canny(
            jscc, thresholded, artifacts.soft_edge_uncertainty,
            cur_snr, canny_cr, th, bsz, device,
        )

    # Block 5e: canny latent for ControlNet
    canny_latent = _encode_canny_latent(jscc, thresholded, device)

    # Verified intra-sampler early-exit metric (None → heuristic / disabled).
    early_exit_score_fn = _resolve_early_exit_score_fn(cfg, jscc, original_image, device=device)

    # Per-element noise level d for the fast-fading water-filling decode (Algorithm
    # 4). Prefers THIS patch's evidence bundle over the channel's global last_bundle
    # (see _water_filling_noise_level). None for AWGN / channels without it → the
    # standard step-matched decode runs. Only read when opted in.
    wf_noise_level = _water_filling_noise_level(artifacts, jscc, cfg)

    # Block 5f: diffusion denoising
    denoised_latent = _run_diffusion(
        pipe=pipe, encode_features_hat=encode_features_hat, power_scalar=power_scalar,
        semantic_text=semantic_text, canny_latent=canny_latent, cur_step=cur_step,
        cfg_method=cfg_method, guidance_scale=guidance_scale, ctrl_scale=ctrl_scale,
        not_control=_build_not_control(encode_features_hat, ctrl_scale, use_controlnet),
        use_jscc_feat=use_jscc_feat, use_controlnet=use_controlnet,
        diffusion_step=diffusion_step, step_style=step_style, mask_token=mask_token,
        cfg=cfg, early_exit_score_fn=early_exit_score_fn, noise_level=wf_noise_level,
    )

    # Block 6: final VAE decode (inference_one.py line 146)
    return (jscc.vae.decode(jscc.normalize(denoised_latent))[0] + 1) / 2


def _jscc_forward(
    x: torch.Tensor,
    jscc,
    pipe,
    gt_text,
    canny_data: Optional[torch.Tensor],
    canny_uncertainty: Optional[torch.Tensor],
    cfg: DictConfig,
    device: torch.device,
    measurement_out: Optional[dict] = None,
) -> torch.Tensor:
    """Full SGDJSCC forward = encode/transmit stage + diffusion-decode stage.

    Mirrors JSCC_model.forward() from inference_one.py; the standard single-call
    path is numerically identical to the original (both stages use the same cfg).
    """
    artifacts = _encode_and_transmit(
        x, jscc, pipe, canny_data, canny_uncertainty, cfg, device,
        measurement_out=measurement_out,
    )
    return _decode_diffusion(artifacts, jscc, pipe, gt_text, cfg, device, original_image=x)


# ─────────────────────────────────────────────────────────────────────────────
# Block helpers for _jscc_forward
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_soft_edge(
    canny_data: Optional[torch.Tensor],
    canny_uncertainty: Optional[torch.Tensor],
    x: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collapse 11-channel edge maps to 1-channel means."""
    if canny_data is not None:
        soft_edge_image       = torch.mean(canny_data,        axis=1, keepdim=True)
        soft_edge_uncertainty = torch.mean(canny_uncertainty, axis=1, keepdim=True)
    else:
        soft_edge_image       = torch.zeros(x.shape[0], 1, 128, 128, device=device)
        soft_edge_uncertainty = torch.zeros_like(soft_edge_image)
    return soft_edge_image, soft_edge_uncertainty


def _encode_latent(
    jscc,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VAE encode: x*2-1 → latent/scaling_factor → L2-normalise."""
    latent_dist = jscc.vae.encode(x * 2 - 1).latent_dist
    encode_features     = jscc.normalize(latent_dist.mean / _SCALING_FACTOR)
    encode_features_std = latent_dist.std
    return encode_features, encode_features_std


def _apply_channel(
    jscc,
    encode_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """AWGN channel injection and renormalisation.  Returns (hat, signal_scale)."""
    snr_scale    = 10 ** (jscc.snr / 10)
    signal_scale = (snr_scale / (snr_scale + 1)) * torch.ones_like(
        encode_features[:, 0:1, 0, 0]
    )
    encode_features_hat = jscc.normalize(jscc.channel(encode_features))
    return encode_features_hat, signal_scale


def _apply_mask_token(
    encode_features_hat: torch.Tensor,
    mask_token,
) -> torch.Tensor:
    """Zero out masked positions in encode_features_hat (if mask_token is not None)."""
    if mask_token is None:
        return encode_features_hat
    mask_expand = mask_token.reshape(
        [-1, 1, mask_token.shape[1], mask_token.shape[2]]
    ).repeat([1, encode_features_hat.shape[1], 1, 1])
    return encode_features_hat * mask_expand


def _compute_power_scalar(
    encode_features_hat: torch.Tensor,
    mask_token,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample power scalar (normalisation denominator)."""
    if mask_token is not None:
        numel = torch.sum(
            mask_token.reshape(
                [-1, 1, encode_features_hat.shape[2], encode_features_hat.shape[3]]
            )
            .repeat([1, encode_features_hat.shape[1], 1, 1])
            .reshape(x.shape[0], -1),
            axis=1,
        )
    else:
        numel = torch.tensor(
            encode_features_hat[0].numel(),
            dtype=torch.float32,
            device=encode_features_hat.device,
        )
    power_scalar = torch.sqrt(
        torch.linalg.norm(
            encode_features_hat.reshape([x.shape[0], -1]), ord=2, axis=1
        ) ** 2
        / numel
    ).reshape([-1, 1, 1, 1]).repeat(
        [1, encode_features_hat.shape[1],
         encode_features_hat.shape[2],
         encode_features_hat.shape[3]]
    )
    return power_scalar


def _retransmit_canny(
    jscc,
    thresholded: torch.Tensor,
    soft_edge_uncertainty: torch.Tensor,
    cur_snr,
    canny_cr: str,
    th: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Canny JSCC retransmission (inference_one.py lines 125–131)."""
    cr      = torch.ones(batch_size, 1, device=device) * round(float(canny_cr) * 64)
    snr_t   = torch.ones(batch_size, 1, device=device) * cur_snr
    gt_snr_t = torch.ones(batch_size, 1, device=device) * jscc.snr

    thresholded = jscc.canny_transmission_net(
        torch.cat([thresholded, soft_edge_uncertainty], dim=1).to(device),
        gt_snr=gt_snr_t,
        snr=snr_t,
        cr=cr,
    ).to(torch.float32)

    # Threshold guard for very low SNR (inference_one.py lines 129–130)
    snr_threshold = (
        (cur_snr <= -5)
        .reshape(-1, 1, 1, 1)
        .repeat(1, 1, thresholded.shape[2], thresholded.shape[3])
        .float()
    )
    thresholded = (
        thresholded * (thresholded > th) * snr_threshold
        + thresholded * (1 - snr_threshold)
    ).float()
    return thresholded


def _encode_canny_latent(
    jscc,
    thresholded: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Encode edge map into latent space for ControlNet conditioning.

    Mirrors inference_one.py line 133:
      canny_latent = vae.encode((thresholded*2-1).repeat([1,3,1,1]))[0].mean / scaling_factor
    """
    return (
        jscc.vae.encode(
            (thresholded * 2 - 1).repeat([1, 3, 1, 1]).to(device)
        )[0].mean
        / _SCALING_FACTOR
    )


def _build_not_control(
    encode_features_hat: torch.Tensor,
    ctrl_scale: float,
    use_controlnet: bool,
) -> list:
    """Build the not_control list for DiffusionGenerator.generate()."""
    n = encode_features_hat.size(0)
    if use_controlnet:
        return [ctrl_scale] * n + [0] * n
    return [0] * n + [0] * n


def _water_filling_noise_level(artifacts, jscc, cfg) -> Optional[torch.Tensor]:
    """Per-element noise level ``d`` for the fast-fading water-filling decode.

    Source priority (fixes per-patch reuse in the one-pass multi-patch path):
      1. ``artifacts.bundle.noise_level`` — THIS patch's evidence bundle, captured
         in phase 1 of ``run_image_channel_conditioned`` (each patch its own ``d``);
      2. else ``jscc.channel_model.last_bundle.noise_level`` — correct for the
         sequential single-image path where ``last_bundle`` IS this patch, but the
         WRONG (last) patch in the one-pass path, so it is only a fallback.

    Returns None when water-filling is off or no per-element ``d`` exists (AWGN /
    channels that don't populate it) → the standard step-matched decode runs.
    """
    if cfg is None or not bool(cfg.get("use_water_filling", False)):
        return None
    nl = getattr(getattr(artifacts, "bundle", None), "noise_level", None)
    if nl is not None:
        return nl
    lb = getattr(getattr(jscc, "channel_model", None), "last_bundle", None)
    return getattr(lb, "noise_level", None) if lb is not None else None


def _run_water_filling_diffusion(
    pipe,
    latent_init: torch.Tensor,
    noise_level: torch.Tensor,
    semantic_text: list,
    negative_prompt: list,
    guidance_scale: float,
    canny_latent: torch.Tensor,
    use_controlnet: bool,
    not_control: list,
    mask_token,
    steps: int,
) -> torch.Tensor:
    """Fast-fading water-filling decode (paper Algorithm 4) over the **real DM**.

    Builds an f0-predictor adapter over ``pipe.pred_image`` (the SGD-JSCC
    ``DiffusionGenerator`` f0 call) and runs the per-element water-filling loop on
    the equalized latent ``f̃`` (= *latent_init*) with per-element noise level ``d``
    (= *noise_level*). This is the runtime decode-swap that connects
    ``acceleration/water_filling.py`` to the inference path; it is exercised on CPU
    via a stub ``pipe`` (real numbers need the MDTv2 checkpoint).
    """
    import numpy as np
    from sgdjscc_lab.acceleration.water_filling import (
        build_mdt_f0_predictor, water_filling_denoise,
    )

    # Labels: encode text + concatenate the unconditional (negative) branch, exactly
    # as DiffusionGenerator.generate_fading() does before its loop.
    labels = pipe.encode_text(semantic_text, pipe.text_embed)
    neg = (pipe.encode_text(negative_prompt, pipe.text_embed)
           if negative_prompt is not None else torch.zeros_like(labels))
    labels = torch.cat([labels, neg])

    bsz = latent_init.shape[0]
    class_guidance = np.full(bsz, float(guidance_scale), dtype=np.float64)
    f0_fn = build_mdt_f0_predictor(
        pipe.pred_image, labels, class_guidance,
        c=canny_latent if use_controlnet else None,
        controlnet=use_controlnet, mask_token=mask_token, not_control=not_control,
    )
    return water_filling_denoise(latent_init, noise_level, f0_fn, steps=steps)


def _run_diffusion(
    pipe,
    encode_features_hat: torch.Tensor,
    power_scalar: torch.Tensor,
    semantic_text: list,
    canny_latent: torch.Tensor,
    cur_step,
    cfg_method: str,
    guidance_scale: float,
    ctrl_scale: float,
    not_control: list,
    use_jscc_feat: bool,
    use_controlnet: bool,
    diffusion_step: int,
    step_style: str,
    mask_token,
    cfg: Optional[DictConfig] = None,
    early_exit_score_fn=None,
    noise_level: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Diffusion denoising via pipe.generate() (DPM-Solver++ 2M).

    Returns denoised_latent.  When ``cfg`` enables intra-sampler early-exit
    (``cfg.acceleration.early_exit`` with ``early_exit_mode="intra_sampler"``),
    the interruptible wrapper loop is used instead (Phase 5-B); otherwise the
    original ``pipe.generate()`` path runs unchanged.

    Fast-fading water-filling (paper Algorithm 4): when ``cfg.use_water_filling``
    is set (Phase-5 gated) AND a per-element ``noise_level`` (``d`` from a
    fast-fading channel bundle) is supplied, the global step-matched decode is
    replaced by :func:`_run_water_filling_diffusion` (real-DM adapter over
    ``pipe.pred_image``). Default off → the original path is unchanged.
    """
    negative_prompt = [
        "distorted, discontinuous, ugly, blurry, low resolution, "
        "deformed, bad quality, deformed"
        for _ in range(len(semantic_text))
    ]

    latent_init = (
        encode_features_hat / power_scalar
        if use_jscc_feat
        else torch.randn_like(encode_features_hat)
    )

    # ── Fast-fading water-filling decode (paper Algorithm 4), opt-in ──────────
    # Replaces the global step-matched decode with the per-element water-filling
    # loop when (a) cfg.use_water_filling is set (Phase-5 gated), (b) a per-element
    # noise level d is supplied (from a fast-fading channel bundle) and (c) we are
    # decoding the actual received JSCC latent (use_jscc_feature). Default off.
    from sgdjscc_lab.phase_gates import phase5_enabled as _p5_on
    if (cfg is not None and bool(cfg.get("use_water_filling", False)) and _p5_on(cfg)
            and noise_level is not None and use_jscc_feat):
        wf = cfg.get("water_filling", None)
        steps = int(wf.get("steps", 50)) if wf is not None else 50
        logger.info("Fast-fading water-filling decode (Algorithm 4, %d steps).", steps)
        return _run_water_filling_diffusion(
            pipe, latent_init, noise_level, semantic_text, negative_prompt,
            guidance_scale, canny_latent, use_controlnet, not_control, mask_token, steps)

    # ── Phase 5-B: intra-sampler early-exit (opt-in via cfg.acceleration) ─────
    # use_phase5 is the master gate: if it is false the acceleration block is
    # skipped even when cfg.acceleration.early_exit is explicitly true.
    from sgdjscc_lab.phase_gates import phase5_enabled as _p5_on
    acc = cfg.get("acceleration", None) if cfg is not None else None
    if acc is not None and bool(acc.get("early_exit", False)) \
            and str(acc.get("early_exit_mode", "intra_sampler")) == "intra_sampler" \
            and _p5_on(cfg):
        from sgdjscc_lab.models.diffusion_wrapper import generate_interruptible
        from sgdjscc_lab.acceleration.early_exit import EarlyExitController
        controller = EarlyExitController(
            srs_threshold=float(acc.get("srs_threshold", 0.8)),
            improvement_delta=float(acc.get("improvement_delta", 0.01)),
            min_steps=int(acc.get("min_steps", 1)),
        )
        denoised_latent, _info = generate_interruptible(
            pipe, prompt=semantic_text, negative_prompt=negative_prompt,
            latent=latent_init, curr_step=cur_step, diffusion_step=diffusion_step,
            c=canny_latent, controlnet=use_controlnet, not_control=not_control,
            class_guidance=guidance_scale, cfg_weighting_method=cfg_method,
            mask_token=mask_token, mask_step=1, step_style=step_style,
            controller=controller, score_fn=early_exit_score_fn,
            check_interval=int(acc.get("early_exit_check_interval", 5)),
            min_steps=int(acc.get("min_steps", 1)),
            # fallback kwargs for the non-continuous / unsupported path:
            num_imgs=1, n_iter=40, scale_factor=1, img_channel=16, img_size=16,
            alphas_cumprod=pipe.alphas_cumprod,
        )
        return denoised_latent

    _image, denoised_latent = pipe.generate(
        prompt=semantic_text,
        num_imgs=1,
        class_guidance=guidance_scale,
        cfg_weighting_method=cfg_method,
        n_iter=40,
        not_control=not_control,
        scale_factor=1,
        latent=latent_init,
        negative_prompt=negative_prompt,
        return_latent=True,
        img_channel=16,
        img_size=16,
        alphas_cumprod=pipe.alphas_cumprod,
        curr_step=cur_step,
        c=canny_latent,
        controlnet=use_controlnet,
        mask_step=1,
        mask_token=mask_token,
        diffusion_step=diffusion_step,
        step_style=step_style,
    )
    return denoised_latent


# ─────────────────────────────────────────────────────────────────────────────
# Step matching (unchanged from pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_step(
    jscc,
    encode_features_hat: torch.Tensor,
    power_scalar: torch.Tensor,
    signal_scale: torch.Tensor,
    pipe,
    step_style: str,
    use_jscc_feat: bool,
    use_gt_csi: bool,
    device: torch.device,
):
    """Compute diffusion starting step and estimated SNR.

    Mirrors the step-matching block in inference_one.py lines 93–120.

    Returns (cur_step, cur_snr).
    """
    if step_style == "continuous":
        if use_jscc_feat:
            if use_gt_csi:
                cur_step = float(1 - signal_scale.mean().item())
                cur_snr  = float(jscc.snr)
            else:
                predicted_signal_scale = (
                    jscc.snr_prediction_net(
                        encode_features_hat / power_scalar
                    ).reshape([-1, 1]) ** 2
                )
                cur_step = 1 - predicted_signal_scale
                cur_snr  = 10 * torch.log10(1 / cur_step - 1)
        else:
            cur_step = 1.0
            cur_snr  = float(jscc.snr)

    elif step_style == "discrete":
        if use_jscc_feat:
            alphas = pipe.scheduler.alphas_cumprod.unsqueeze(0).to(device)
            if use_gt_csi:
                cur_step = (
                    torch.argmin(
                        torch.abs(alphas - signal_scale.reshape([-1, 1])),
                        axis=1,
                    )
                    .float()
                    .mean()
                    .int()
                    .item()
                )
            else:
                pred = (
                    jscc.snr_prediction_net(
                        encode_features_hat / power_scalar
                    ).reshape([-1, 1]) ** 2
                )
                cur_step = (
                    torch.argmin(torch.abs(alphas - pred), axis=1)
                    .float()
                    .mean()
                    .int()
                    .item()
                )
        else:
            cur_step = 981
        cur_snr = float(jscc.snr)

    else:
        raise ValueError(f"Unknown step_style: {step_style!r}")

    return cur_step, cur_snr
