"""models/edge_jscc.py – Edge-map JSCC codec for Stage-3 ControlNet guidance.

The SGD-JSCC paper transmits the structural (Canny edge) guidance over its own
DeepJSCC link — *not* through the image VAE.  The edge JSCC is trained with a
BCE + Dice objective (Sec. V, "edge transmission") and its received edge map is
then aligned to the diffusion latent space to condition the ControlNet branch.

This module provides a **dedicated edge transport path** that is structurally
separate from the image VAE, with two heads on a shared encoder:

    edge map ─▶ EdgeJSCCEncoder ─▶ channel ─▶ EdgeLatentProjector ─┬─▶ c   (Stage-3 cond.)
                                                                   └─▶ EdgeJSCCDecoder ─▶ edge logits   (codec training)

* ``EdgeJSCCEncoder`` — a small conv encoder mapping a 1-channel edge map to a
  latent tensor whose spatial size matches the diffusion latent grid.
* the channel is any object exposing ``transmit(latent, snr_db)`` — reuses
  ``channels/awgn.py`` (or a Phase-5 channel), so the edge symbols actually pass
  through a noisy channel like the paper's edge link.
* ``EdgeLatentProjector`` — aligns the received edge latent to the diffusion
  latent channel count expected by the ControlNet ``forward_c``.  Its output is
  the condition latent ``c``.
* ``EdgeJSCCDecoder`` — upsamples the *same* projected latent back to a
  1-channel **edge reconstruction logit** map.  This head exists so the codec
  can be trained end-to-end with a BCE + Dice objective (stage ``edge_codec``);
  it is not used at Stage-3 inference time (only ``c`` is).

Training the codec (stage ``edge_codec``, see ``training/stage_runners.py``)
shapes the exact latent that Stage 3 later consumes as ``c`` — so a checkpoint
produced by that stage can be loaded here (``EdgeJSCC.load_codec_state``) to make
the ``edge_jscc`` transport a **trained** edge codec rather than a random
stand-in.  Without such a checkpoint the weights are randomly initialised
(ablation-grade), which is why ``edge_jscc`` is only the Stage-3 *baseline*
transport once a trained codec is loaded.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _norm_layer(kind: str, ch: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "instance":
        return nn.InstanceNorm2d(ch, affine=True)
    if kind in ("group", "groupnorm"):
        return nn.GroupNorm(min(8, ch), ch)
    return nn.Identity()


class EdgeJSCCEncoder(nn.Module):
    """Conv encoder: 1-channel edge map → latent ``[B, latent_ch, h, w]``.

    ``downsample_factor`` controls how many stride-2 stages are applied so the
    output spatial size matches the diffusion latent grid (image VAE downsamples
    by 8: 128×128 → 16×16, so the default factor is 8).
    """

    def __init__(
        self,
        latent_ch: int = 16,
        base_ch: int = 64,
        downsample_factor: int = 8,
        norm: str = "group",
    ) -> None:
        super().__init__()
        n_down = max(0, int(round(math.log2(max(1, downsample_factor)))))
        layers = [nn.Conv2d(1, base_ch, 3, padding=1), nn.SiLU()]
        ch = base_ch
        for _ in range(n_down):
            out = min(ch * 2, 256)
            layers += [
                nn.Conv2d(ch, out, 3, stride=2, padding=1),
                _norm_layer(norm, out),
                nn.SiLU(),
            ]
            ch = out
        layers += [nn.Conv2d(ch, latent_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, edge: torch.Tensor) -> torch.Tensor:
        if edge.shape[1] != 1:
            edge = edge.mean(dim=1, keepdim=True)
        return self.net(edge)


class EdgeLatentProjector(nn.Module):
    """Align a received edge latent to the diffusion latent channel count."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class EdgeJSCCDecoder(nn.Module):
    """Conv decoder: condition latent ``[B, latent_ch, h, w]`` → edge logits ``[B, 1, H, W]``.

    Mirrors :class:`EdgeJSCCEncoder` (``upsample_factor`` stride-2 transposed
    convs) so the reconstructed edge map returns to the input resolution.  The
    output is a **logit** map (no sigmoid) so it can be fed straight into
    ``BCEWithLogits`` + soft-Dice in :class:`~sgdjscc_lab.training.losses.EdgeCodecLoss`.
    """

    def __init__(
        self,
        latent_ch: int = 16,
        base_ch: int = 64,
        upsample_factor: int = 8,
        norm: str = "group",
    ) -> None:
        super().__init__()
        n_up = max(0, int(round(math.log2(max(1, upsample_factor)))))
        # Start from the encoder's deepest width and halve on the way up.
        ch = min(base_ch * (2 ** n_up), 256) if n_up else base_ch
        layers = [nn.Conv2d(latent_ch, ch, 3, padding=1), nn.SiLU()]
        for _ in range(n_up):
            out = max(base_ch, ch // 2)
            layers += [
                nn.ConvTranspose2d(ch, out, 4, stride=2, padding=1),
                _norm_layer(norm, out),
                nn.SiLU(),
            ]
            ch = out
        layers += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


# ─────────────────────────────────────────────────────────────────────────────
# ViT edge codec (paper-like: patch-embedding + transformer, vs the conv variant)
# ─────────────────────────────────────────────────────────────────────────────
#
# The paper transmits the edge map with a ViT-based DeepJSCC (small patch
# embedding, SNR projected to the decoder transformer blocks; SGDJSCC's public
# code uses a WITT ViT, ``model_canny.Semantic_Communication_Model`` with
# ``vit_..._adaln_...``). These classes provide a self-contained ViT codec with
# the same I/O contract as the conv variant (edge → latent grid; latent → 1-ch
# logits), so it is selectable via ``EdgeJSCC(arch="vit")``.
#
# paper-like / scaffold: the structure (patch-embed → transformer → projector →
# optional decoder) matches the paper's ViT design, but this is NOT a WITT-exact
# (Swin-window) reimplementation and the weights are untrained (see EdgeJSCC docstring).
#
# SNR conditioning (paper / WITT: SNR → adaLN in the transformer blocks) is
# IMPLEMENTED as an opt-in hook (``vit.snr_cond``): an SNREmbedder + adaLN-Zero
# blocks (``SNREmbedder``/``_AdaLNBlock``) mirror SGDJSCC's revised_witt
# ``SNREmbedder``/``modulate``/``Head_layer``, and the conditioning VALUE is the
# **linear** SNR scale ``10**(snr_db/10)`` matching the public WITT path
# (``model_canny.py``), not the dB value. CAVEAT: the ``edge_codec`` stage trains
# at a FIXED SNR, so the conditioning is CONSTANT (a fixed modulation) until the
# codec is trained across SNRs (feed the per-forward SNR in EdgeJSCC._snr_tensor).
# Default off. Conditioning location + value are WITT-aligned, but the block is
# DiT-style adaLN, NOT a Swin-window WITT-exact reimplementation.


def _sincos_pos_embed(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    """Parameter-free 2-D sin/cos positional embedding ``[h*w, dim]`` (any grid)."""
    assert dim % 4 == 0, "ViT embed_dim must be divisible by 4 for sin/cos pos-embed"
    gy, gx = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    d = dim // 4
    omega = 1.0 / (10000.0 ** (torch.arange(d, device=device, dtype=torch.float32) / d))
    parts = []
    for grid in (gy.reshape(-1).float(), gx.reshape(-1).float()):
        ang = grid[:, None] * omega[None, :]
        parts += [torch.sin(ang), torch.cos(ang)]
    return torch.cat(parts, dim=1).to(dtype)              # [h*w, dim]


def _transformer(embed_dim: int, depth: int, heads: int, mlp_ratio: float) -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim, nhead=heads, dim_feedforward=int(embed_dim * mlp_ratio),
        batch_first=True, activation="gelu", norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=depth)


# ── adaLN SNR conditioning (mirrors SGDJSCC revised_witt/witt_modules) ─────────

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x*(1+scale)+shift`` over the token dim — same as WITT ``modulate``."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SNREmbedder(nn.Module):
    """Sinusoidal SNR/noise-level embedding + MLP — mirrors WITT ``SNREmbedder``
    (``SGDJSCC/.../revised_witt/witt_modules.py``)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.freq = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.freq, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size))

    @staticmethod
    def _embed(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period)
                          * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t.reshape(-1).float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, snr: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._embed(snr, self.freq).to(self.mlp[0].weight.dtype))


class _AdaLNBlock(nn.Module):
    """DiT/adaLN-Zero transformer block conditioned on the SNR embedding ``c``.

    The conditioning *location* (adaLN modulation of the transformer blocks)
    mirrors the paper / WITT (SNR → adaLN). This is a standard DiT-style block, NOT
    a Swin-window WITT block, so it is **paper-like**, not WITT-exact.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)        # adaLN-Zero: start as identity
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.adaLN(c).chunk(6, dim=1)
        h = _modulate(self.norm1(x), sh_a, sc_a)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g_a.unsqueeze(1) * a
        h = _modulate(self.norm2(x), sh_m, sc_m)
        return x + g_m.unsqueeze(1) * self.mlp(h)


class _AdaLNTransformer(nn.Module):
    """Stack of :class:`_AdaLNBlock` driven by an :class:`SNREmbedder`."""

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.snr_embed = SNREmbedder(dim)
        self.blocks = nn.ModuleList([_AdaLNBlock(dim, heads, mlp_ratio) for _ in range(depth)])

    def forward(self, tok: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        c = self.snr_embed(snr)                      # [B, dim]
        for blk in self.blocks:
            tok = blk(tok, c)
        return tok


class EdgeJSCCViTEncoder(nn.Module):
    """Patch-embed → transformer → token→latent. 1-ch edge ``[B,1,H,W]`` →
    latent ``[B, latent_ch, H/patch, W/patch]`` (patch = downsample factor)."""

    def __init__(self, latent_ch: int = 16, patch_size: int = 8, embed_dim: int = 128,
                 depth: int = 4, num_heads: int = 4, mlp_ratio: float = 2.0,
                 snr_cond: bool = False) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.snr_cond = bool(snr_cond)
        self.patch = nn.Conv2d(1, embed_dim, patch_size, stride=patch_size)
        self.blocks = (_AdaLNTransformer(embed_dim, depth, num_heads, mlp_ratio)
                       if self.snr_cond else _transformer(embed_dim, depth, num_heads, mlp_ratio))
        self.to_latent = nn.Linear(embed_dim, latent_ch)

    def forward(self, edge: torch.Tensor, snr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge.shape[1] != 1:
            edge = edge.mean(dim=1, keepdim=True)
        x = self.patch(edge)                              # [B, D, gh, gw]
        b, d, gh, gw = x.shape
        tok = x.flatten(2).transpose(1, 2)                # [B, N, D]
        tok = tok + _sincos_pos_embed(gh, gw, d, tok.device, tok.dtype)[None]
        tok = self.blocks(tok, snr) if self.snr_cond else self.blocks(tok)
        z = self.to_latent(tok)                           # [B, N, latent_ch]
        return z.transpose(1, 2).reshape(b, -1, gh, gw)   # [B, latent_ch, gh, gw]


class EdgeJSCCViTDecoder(nn.Module):
    """latent ``[B, latent_ch, h, w]`` → transformer → unpatch → edge logits
    ``[B, 1, h*patch, w*patch]`` (mirror of :class:`EdgeJSCCViTEncoder`)."""

    def __init__(self, latent_ch: int = 16, patch_size: int = 8, embed_dim: int = 128,
                 depth: int = 4, num_heads: int = 4, mlp_ratio: float = 2.0,
                 snr_cond: bool = False) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.snr_cond = bool(snr_cond)
        self.from_latent = nn.Linear(latent_ch, embed_dim)
        self.blocks = (_AdaLNTransformer(embed_dim, depth, num_heads, mlp_ratio)
                       if self.snr_cond else _transformer(embed_dim, depth, num_heads, mlp_ratio))
        self.to_pixels = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, c: torch.Tensor, snr: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, lc, h, w = c.shape
        p = self.patch_size
        tok = c.flatten(2).transpose(1, 2)                # [B, N, latent_ch]
        tok = self.from_latent(tok)                       # [B, N, D]
        tok = tok + _sincos_pos_embed(h, w, tok.shape[-1], tok.device, tok.dtype)[None]
        tok = self.blocks(tok, snr) if self.snr_cond else self.blocks(tok)
        px = self.to_pixels(tok)                          # [B, N, p*p]
        # fold tokens back to a [B,1,h*p,w*p] image
        px = px.reshape(b, h, w, p, p).permute(0, 1, 3, 2, 4).reshape(b, h * p, w * p)
        return px.unsqueeze(1)


class EdgeJSCC(nn.Module):
    """End-to-end edge transport: encode → channel → project → condition latent.

    Parameters
    ----------
    latent_ch:
        Output channel count of the condition latent ``c`` (must match the
        diffusion latent channels, i.e. the image VAE z-channels = 16).
    channel:
        Object with ``transmit(latent, snr_db)`` (reuse ``AWGNChannel``). When
        None, no channel noise is applied (clean edge latent).
    snr_db:
        SNR for the edge channel.
    """

    def __init__(
        self,
        latent_ch: int = 16,
        base_ch: int = 64,
        downsample_factor: int = 8,
        norm: str = "group",
        channel: Optional[object] = None,
        snr_db: float = 10.0,
        with_decoder: bool = False,
        arch: str = "conv",
        vit_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.arch = str(arch).lower()
        if self.arch == "vit":
            # ViT codec: patch = downsample_factor so the latent grid matches the
            # conv variant (H/downsample). vit_cfg: embed_dim/depth/num_heads/mlp_ratio.
            vc = dict(vit_cfg or {})
            embed_dim = int(vc.get("embed_dim", 128))
            depth = int(vc.get("depth", 4))
            heads = int(vc.get("num_heads", 4))
            mlp_ratio = float(vc.get("mlp_ratio", 2.0))
            # SNR conditioning via adaLN (WITT-style): location-aligned with the
            # paper (SNR → adaLN in the transformer blocks). Default off; at
            # fixed-SNR training the conditioning is constant — see encode_latent.
            snr_cond = bool(vc.get("snr_cond", False))
            self._snr_cond = snr_cond
            self.encoder = EdgeJSCCViTEncoder(
                latent_ch, downsample_factor, embed_dim, depth, heads, mlp_ratio, snr_cond)
            self.decoder = (
                EdgeJSCCViTDecoder(latent_ch, downsample_factor, embed_dim, depth,
                                   heads, mlp_ratio, snr_cond)
                if with_decoder else None)
        elif self.arch == "conv":
            self.encoder = EdgeJSCCEncoder(latent_ch, base_ch, downsample_factor, norm)
            # The decoder head is only needed to TRAIN the codec (stage edge_codec).
            # Stage-3 inference uses ``encode`` (the condition latent ``c``) only, so
            # the transport builder can omit it to save weights/memory.
            self.decoder = (
                EdgeJSCCDecoder(latent_ch, base_ch, downsample_factor, norm)
                if with_decoder else None)
        else:
            raise ValueError(f"EdgeJSCC arch must be 'conv' or 'vit', got {arch!r}.")
        self.projector = EdgeLatentProjector(latent_ch, latent_ch)
        self.channel = channel
        self.snr_db = float(snr_db)
        if not hasattr(self, "_snr_cond"):
            self._snr_cond = False

    def _snr_tensor(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Per-sample SNR conditioning value for adaLN, or None when off.

        Matches the public WITT edge codec convention: the network is conditioned
        on the **linear** SNR scale ``10**(snr_db/10)``, NOT the dB value
        (``SGDJSCC/models/model_canny.py`` computes ``snr_scale = 10**(snr/10)``
        and passes it to the WITT encoder/decoder). Uses the codec's fixed
        ``snr_db`` — so at fixed-SNR training the conditioning is a CONSTANT (the
        codec learns a fixed modulation). To make it SNR-adaptive, train across
        SNRs and feed the per-forward SNR here.
        """
        if not getattr(self, "_snr_cond", False):
            return None
        snr_scale = 10.0 ** (float(self.snr_db) / 10.0)   # linear SNR (WITT convention)
        return torch.full((x.shape[0],), snr_scale, device=x.device, dtype=x.dtype)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        """L2-normalise per sample (matches the JSCC latent energy convention)."""
        b = x.shape[0]
        flat = x.reshape(b, -1)
        flat = F.normalize(flat, p=2, dim=1) * math.sqrt(flat.shape[1])
        return flat.reshape_as(x)

    def encode_latent(self, edge: torch.Tensor) -> torch.Tensor:
        """Encoder → L2-normalise → (channel) → projector, at the encoder grid.

        Shared by both heads: this is the exact latent the ControlNet sees as
        ``c`` (before any resize) *and* the latent the decoder reconstructs from,
        so training the decoder shapes the Stage-3 condition latent directly.
        """
        snr = self._snr_tensor(edge)
        z = self.encoder(edge, snr) if snr is not None else self.encoder(edge)
        z = self._normalize(z)
        if self.channel is not None:
            z = self._normalize(self.channel.transmit(z, self.snr_db))
        return self.projector(z)

    def encode(self, edge: torch.Tensor, target_hw: Optional[tuple] = None) -> torch.Tensor:
        """Return the condition latent ``c`` for an edge map batch.

        ``target_hw`` (optional) resizes ``c`` to the diffusion latent grid when
        the encoder's downsampling does not exactly match.
        """
        c = self.encode_latent(edge)
        if target_hw is not None and c.shape[-2:] != tuple(target_hw):
            c = F.interpolate(c, size=tuple(target_hw), mode="bilinear", align_corners=False)
        return c

    def reconstruct(self, edge: torch.Tensor) -> torch.Tensor:
        """Codec-training path: edge map → received edge **logits** at input res.

        Requires the decoder head (``with_decoder=True``).  Returns logits (no
        sigmoid) shaped like the input edge map ``[B, 1, H, W]``.
        """
        if self.decoder is None:
            raise RuntimeError(
                "EdgeJSCC.reconstruct requires the decoder head; build the codec "
                "with with_decoder=True (stage 'edge_codec' does this)."
            )
        c = self.encode_latent(edge)
        snr = self._snr_tensor(edge)
        logits = self.decoder(c, snr) if snr is not None else self.decoder(c)
        if logits.shape[-2:] != edge.shape[-2:]:
            logits = F.interpolate(logits, size=edge.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return logits

    def forward(self, edge: torch.Tensor, target_hw: Optional[tuple] = None) -> torch.Tensor:
        return self.encode(edge, target_hw)

    # ── trained-codec loading (Stage-3 baseline transport) ────────────────────
    def load_codec_state(self, path, strict: bool = False) -> None:
        """Load encoder/projector(/decoder) weights from an ``edge_codec`` checkpoint.

        Accepts a checkpoint written by ``pipelines/train_pipeline.py`` for the
        ``edge_codec`` stage (the codec lives under
        ``runner_state.modules.edge_jscc``) or a legacy ``model_state.edge_jscc``
        layout, or a bare ``state_dict``.  ``strict=False`` lets a decoder-less
        Stage-3 transport load encoder/projector while ignoring decoder keys.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"edge_codec checkpoint not found: {p}")
        ckpt = torch.load(p, map_location="cpu")
        sd = None
        if isinstance(ckpt, dict):
            rs = ckpt.get("runner_state")
            if isinstance(rs, dict):
                sd = (rs.get("modules") or {}).get("edge_jscc")
            if sd is None:
                sd = (ckpt.get("model_state") or {}).get("edge_jscc")
            if sd is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                sd = ckpt  # bare state_dict
        if sd is None:
            raise KeyError(
                f"No edge codec weights found in {p} (looked for "
                "runner_state.modules.edge_jscc / model_state.edge_jscc / a bare state_dict)."
            )
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        logger.info("Loaded edge codec weights from %s (missing=%d, unexpected=%d).",
                    p, len(missing), len(unexpected))
