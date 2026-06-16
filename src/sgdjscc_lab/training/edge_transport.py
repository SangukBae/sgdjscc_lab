"""training/edge_transport.py – Stage-3 edge conditioning transport selector.

Stage 3 (ControlNet) needs a *condition latent* ``c`` derived from the edge map.
There are two transport modes (``train.controlnet.edge_transport``):

  ``shared_vae``  (ablation, backward-compatible)
      The edge map is broadcast to 3 channels and pushed through the **image**
      VAE encoder, landing in the same latent geometry as ``f0``.  Simple and
      requires no extra weights, but the edge does not have its own transmission
      path — it is *not* what the paper does.

  ``edge_jscc``  (paper-like baseline)
      The edge map goes through a **dedicated edge-JSCC link**
      (``models/edge_jscc.py``): edge-only encoder (conv or ViT) → wireless
      channel → latent projector.  This matches the paper's architecture (edge
      transmitted over its own DeepJSCC link, then aligned to the latent). The
      codec weights are **loaded from a trained ``edge_codec`` checkpoint** when
      ``train.controlnet.edge_jscc.checkpoint`` is set (the paper's BCE/Dice
      objective is trained by the ``edge_codec`` stage); only when ``checkpoint``
      is left null does it fall back to a randomly-initialised stand-in
      (ablation-grade, with a warning).

Both modes expose the same interface: ``transport(edge_batch) -> c`` with ``c``
shaped like the diffusion latent ``[B, z_ch, h, w]``.  The Stage-3 runner calls
the transport under ``no_grad`` (the edge transport is treated as fixed side
information, like the paper's separately-trained edge link).
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

EDGE_TRANSPORT_SHARED_VAE = "shared_vae"
EDGE_TRANSPORT_EDGE_JSCC = "edge_jscc"
VALID_EDGE_TRANSPORTS = (EDGE_TRANSPORT_SHARED_VAE, EDGE_TRANSPORT_EDGE_JSCC)

# image VAE latent geometry (DDCONFIG z_channels) — see models/jscc_model.py.
_LATENT_CH = 16
_VAE_DOWNSAMPLE = 8


def resolve_edge_transport(cfg: DictConfig) -> str:
    mode = str(OmegaConf.select(
        cfg, "train.controlnet.edge_transport", default=EDGE_TRANSPORT_SHARED_VAE)).lower()
    if mode not in VALID_EDGE_TRANSPORTS:
        from sgdjscc_lab.training.stages import StageConfigError
        raise StageConfigError(
            f"Unknown train.controlnet.edge_transport={mode!r}. "
            f"Valid: {', '.join(VALID_EDGE_TRANSPORTS)}."
        )
    return mode


def build_edge_transport(cfg: DictConfig, jscc, device) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return an ``edge -> c`` callable for the configured transport mode.

    ``shared_vae`` reuses the image VAE; ``edge_jscc`` instantiates a dedicated
    :class:`~sgdjscc_lab.models.edge_jscc.EdgeJSCC` link (with the JSCC AWGN
    channel) and resizes ``c`` to the VAE latent grid.
    """
    mode = resolve_edge_transport(cfg)

    if mode == EDGE_TRANSPORT_SHARED_VAE:
        from sgdjscc_lab.training.stage_runners import _jscc_latent_encoder
        base = _jscc_latent_encoder(jscc)

        def _shared_vae(edge: torch.Tensor) -> torch.Tensor:
            if edge.shape[1] == 1:
                edge = edge.repeat(1, 3, 1, 1)
            return base(edge)

        logger.info("Stage-3 edge transport: shared_vae (image VAE stand-in).")
        return _shared_vae

    # edge_jscc
    from sgdjscc_lab.models.edge_jscc import EdgeJSCC

    ej_cfg = OmegaConf.select(cfg, "train.controlnet.edge_jscc", default=None)
    base_ch = int(OmegaConf.select(ej_cfg, "base_ch", default=64)) if ej_cfg else 64
    norm = str(OmegaConf.select(ej_cfg, "norm", default="group")) if ej_cfg else "group"
    snr_db = float(OmegaConf.select(ej_cfg, "snr_db", default=10.0)) if ej_cfg else 10.0
    use_channel = bool(OmegaConf.select(ej_cfg, "use_channel", default=True)) if ej_cfg else True
    checkpoint = OmegaConf.select(ej_cfg, "checkpoint", default=None) if ej_cfg else None
    arch = str(OmegaConf.select(ej_cfg, "arch", default="conv")) if ej_cfg else "conv"
    vit_cfg = _vit_cfg(ej_cfg)

    channel = getattr(jscc, "_awgn_channel", None) if use_channel else None
    if channel is None and use_channel:
        # Fall back to a standalone AWGN channel if the JSCC model doesn't expose
        # one (keeps the edge link noisy even without the full JSCC bundle).
        from sgdjscc_lab.channels.awgn import AWGNChannel
        channel = AWGNChannel()
    edge_codec = EdgeJSCC(
        latent_ch=_LATENT_CH, base_ch=base_ch, downsample_factor=_VAE_DOWNSAMPLE,
        norm=norm, channel=channel, snr_db=snr_db,  # transport needs no decoder
        arch=arch, vit_cfg=vit_cfg,
    ).to(device)

    # Load the trained edge codec → this is what makes edge_jscc a paper-like
    # transport. Two explicit, NON-overlapping cases:
    #   • checkpoint set  → MUST exist; a missing file is a HARD failure (fail
    #     fast) rather than a silent downgrade — the default baseline config ships
    #     a non-null path, so a missing file would otherwise crash deep in load.
    #   • checkpoint null → deliberate ABLATION-grade random stand-in (warn).
    if checkpoint:
        from pathlib import Path as _Path
        if not _Path(checkpoint).exists():
            raise FileNotFoundError(
                "edge_transport=edge_jscc requires a trained edge codec, but "
                "train.controlnet.edge_jscc.checkpoint points to a missing file:\n"
                f"  {checkpoint}\n"
                "Fix one of:\n"
                "  1) train it first → scripts/train.py --stage edge_codec ... "
                "(writes outputs/checkpoints/edge_codec/best.pth), then re-run;\n"
                "  2) set train.controlnet.edge_jscc.checkpoint: null to use an "
                "ABLATION-grade RANDOM codec (not a paper-like transport);\n"
                "  3) use configs/composed_train_controlnet_shared_vae.yaml for the "
                "shared-VAE ablation.")
        edge_codec.load_codec_state(checkpoint, strict=False)
        trained = True
    else:
        logger.warning(
            "Stage-3 edge transport: edge_jscc with NO trained codec "
            "(train.controlnet.edge_jscc.checkpoint is unset) → weights are a "
            "RANDOM structural stand-in (ablation-grade, NOT a baseline claim). "
            "Train one via 'scripts/train.py --stage edge_codec' and point "
            "edge_jscc.checkpoint at outputs/checkpoints/edge_codec/best.pth.")
        trained = False
    edge_codec.eval()  # fixed side-information generator (see module docstring)

    logger.info("Stage-3 edge transport: edge_jscc (dedicated edge link, "
                "snr=%.1f dB, channel=%s, trained_codec=%s).",
                snr_db, channel is not None, trained)

    def _edge_jscc(edge: torch.Tensor) -> torch.Tensor:
        # Align to the image-VAE latent grid (H/8, W/8).
        h, w = edge.shape[-2] // _VAE_DOWNSAMPLE, edge.shape[-1] // _VAE_DOWNSAMPLE
        return edge_codec.encode(edge, target_hw=(h, w))

    # Expose the codec so callers (e.g. checkpointing) can reach it if needed.
    _edge_jscc.module = edge_codec  # type: ignore[attr-defined]
    return _edge_jscc


def build_edge_codec(cfg: DictConfig, device):
    """Build a trainable :class:`EdgeJSCC` (with decoder head) for the
    ``edge_codec`` stage.

    Reads architecture/channel knobs from ``train.edge_codec`` and attaches a
    standalone AWGN channel (when ``use_channel``).  The result is fully
    trainable (encoder + projector + decoder) so the runner can optimise it with
    BCE + Dice.  Keep ``base_ch`` / ``norm`` consistent with
    ``train.controlnet.edge_jscc`` so the produced checkpoint loads cleanly into
    the Stage-3 transport.
    """
    from sgdjscc_lab.models.edge_jscc import EdgeJSCC

    ec = OmegaConf.select(cfg, "train.edge_codec", default=None)
    base_ch = int(OmegaConf.select(ec, "base_ch", default=64)) if ec else 64
    norm = str(OmegaConf.select(ec, "norm", default="group")) if ec else "group"
    snr_db = float(OmegaConf.select(ec, "snr_db", default=10.0)) if ec else 10.0
    use_channel = bool(OmegaConf.select(ec, "use_channel", default=True)) if ec else True
    arch = str(OmegaConf.select(ec, "arch", default="conv")) if ec else "conv"
    vit_cfg = _vit_cfg(ec)

    channel = None
    if use_channel:
        from sgdjscc_lab.channels.awgn import AWGNChannel
        channel = AWGNChannel()
    codec = EdgeJSCC(
        latent_ch=_LATENT_CH, base_ch=base_ch, downsample_factor=_VAE_DOWNSAMPLE,
        norm=norm, channel=channel, snr_db=snr_db, with_decoder=True,
        arch=arch, vit_cfg=vit_cfg,
    ).to(device)
    logger.info("edge_codec: EdgeJSCC(arch=%s, base_ch=%d, norm=%s, snr=%.1f dB, "
                "channel=%s) — trainable encoder+projector+decoder.",
                arch, base_ch, norm, snr_db, channel is not None)
    return codec


def _vit_cfg(node):
    """Extract the ViT sub-config (embed_dim/depth/num_heads/mlp_ratio) or None."""
    if node is None:
        return None
    vit = OmegaConf.select(node, "vit", default=None)
    if vit is None:
        return None
    return OmegaConf.to_container(vit, resolve=True)
