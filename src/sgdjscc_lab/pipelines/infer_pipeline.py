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
from pathlib import Path
from typing import Optional, Tuple

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

    Returns
    -------
    torch.Tensor
        ``[N, 3, 128, 128]`` float in [0, 1] – reconstructed patches.
    """
    device = models.device

    # ── Pull inference flags ──────────────────────────────────────────────────
    use_semantic   = bool(cfg.use_semantic)
    use_text       = bool(cfg.use_text)
    use_controlnet = bool(cfg.use_controlnet)

    x = img_tensor.to(device)

    # ── Step 1: Extract semantic guidance ────────────────────────────────────
    gt_text = None
    canny_data = None
    canny_uncertainty = None

    if use_semantic:
        if use_text and models.text_extractor is not None:
            gt_text = models.text_extractor.extract(
                x,
                device,
                offload_device=models.offload_device,
                offload_after=models.offload_caption,
            )
        if models.edge_extractor is not None:
            canny_data, canny_uncertainty = models.edge_extractor.extract(
                x,
                device,
                offload_device=models.offload_device,
                offload_after=models.offload_canny,
            )
        else:
            canny_data = torch.zeros(x.shape[0], 11, 128, 128, device=device)
            canny_uncertainty = torch.zeros_like(canny_data)

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
# Core forward pass (mirrors JSCC_model.forward in inference_one.py)
# ─────────────────────────────────────────────────────────────────────────────

def _jscc_forward(
    x: torch.Tensor,
    jscc,
    pipe,
    gt_text,
    canny_data: Optional[torch.Tensor],
    canny_uncertainty: Optional[torch.Tensor],
    cfg: DictConfig,
    device: torch.device,
) -> torch.Tensor:
    """Orchestrate the full SGDJSCC forward pass.

    Mirrors JSCC_model.forward() from inference_one.py.  Each logical block
    is delegated to a focused private helper to keep this function readable.
    """
    # ── Unpack flags ──────────────────────────────────────────────────────────
    use_semantic   = bool(cfg.use_semantic)
    use_text       = bool(cfg.use_text)
    use_controlnet = bool(cfg.use_controlnet)
    use_jscc_feat  = bool(cfg.use_jscc_feature)
    use_gt_csi     = bool(cfg.use_gt_csi)
    mask_method    = str(cfg.mask_method)
    canny_cr       = str(cfg.canny_cr)
    step_style     = str(cfg.step_style)
    diffusion_step = int(cfg.diffusion_step)
    guidance_scale = float(cfg.guidance_scale)
    ctrl_scale     = float(cfg.controlnet_scale)
    cfg_method     = str(cfg.cfg_method)
    th             = float(cfg.th)

    # ── Block 1: soft edge pre-processing ─────────────────────────────────────
    soft_edge_image, soft_edge_uncertainty = _preprocess_soft_edge(
        canny_data, canny_uncertainty, x, device
    )

    # ── Block 2: VAE encode + normalise ───────────────────────────────────────
    encode_features, encode_features_std = _encode_latent(jscc, x)

    # ── Block 3: AWGN channel + renormalise ───────────────────────────────────
    encode_features_hat, signal_scale = _apply_channel(jscc, encode_features)

    # ── Block 4: non-semantic fallback ────────────────────────────────────────
    if not use_semantic:
        return (jscc.vae.decode(jscc.normalize(encode_features_hat))[0] + 1) / 2

    # ── Block 5a: text guidance ───────────────────────────────────────────────
    semantic_text = (
        list(gt_text[0])
        if use_text and gt_text is not None
        else ["" for _ in range(x.shape[0])]
    )
    thresholded = soft_edge_image

    # ── Block 5b: mask token + power scalar ───────────────────────────────────
    mask_token = generate_mask(
        encode_features, encode_features_std, thresholded, mask_method
    )
    encode_features_hat = _apply_mask_token(encode_features_hat, mask_token)
    power_scalar = _compute_power_scalar(encode_features_hat, mask_token, x)

    # ── Block 5c: step matching ───────────────────────────────────────────────
    cur_step, cur_snr = _compute_step(
        jscc=jscc,
        encode_features_hat=encode_features_hat,
        power_scalar=power_scalar,
        signal_scale=signal_scale,
        pipe=pipe,
        step_style=step_style,
        use_jscc_feat=use_jscc_feat,
        use_gt_csi=use_gt_csi,
        device=device,
    )

    # ── Block 5d: canny JSCC retransmission ───────────────────────────────────
    if canny_cr != "none":
        thresholded = _retransmit_canny(
            jscc, thresholded, soft_edge_uncertainty,
            cur_snr, canny_cr, th, x.size(0), device
        )

    # ── Block 5e: canny latent for ControlNet ─────────────────────────────────
    canny_latent = _encode_canny_latent(jscc, thresholded, device)

    # ── Block 5f: diffusion denoising ─────────────────────────────────────────
    denoised_latent = _run_diffusion(
        pipe=pipe,
        encode_features_hat=encode_features_hat,
        power_scalar=power_scalar,
        semantic_text=semantic_text,
        canny_latent=canny_latent,
        cur_step=cur_step,
        cfg_method=cfg_method,
        guidance_scale=guidance_scale,
        ctrl_scale=ctrl_scale,
        not_control=_build_not_control(encode_features_hat, ctrl_scale, use_controlnet),
        use_jscc_feat=use_jscc_feat,
        use_controlnet=use_controlnet,
        diffusion_step=diffusion_step,
        step_style=step_style,
        mask_token=mask_token,
    )

    # ── Block 6: final VAE decode (inference_one.py line 146) ─────────────────
    return (jscc.vae.decode(jscc.normalize(denoised_latent))[0] + 1) / 2


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
) -> torch.Tensor:
    """Diffusion denoising via pipe.generate() (DPM-Solver++ 2M).

    Returns denoised_latent.
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
