"""paper_mode.py – Guardrails for the paper-faithful reproduction path.

When ``paper_mode: true`` (a top-level config flag, default ``false``) the
framework **enforces** the SGD-JSCC paper's intended configuration and **blocks**
extensions / non-faithful stand-ins, raising an explicit :class:`PaperModeError`.
This exists so the codebase can never silently make a "paper-faithful" claim for
a path that actually deviates (auto-generated captions, Canny edges instead of
MuGE, shared-VAE edge transport, zero-vector CFG null, a single-fixed-SNR edge
codec, etc.).

Fidelity taxonomy (used here, in module docstrings, and in
``docs/paper_gap_closure.md``):

* **paper-faithful** – matches the paper / public ``SGDJSCC`` code (numerically
  or structurally) *given the same data and checkpoints*.
* **paper-like**     – same intent / formula family, but a stated-or-unstated
  detail differs (a value the paper does not give, a simplified module, …).
* **unsupported**    – cannot be reproduced here (non-public data / weights /
  details); guarded so it cannot masquerade as faithful.

``paper_mode`` does **not** delete any extension: every non-faithful path still
works with ``paper_mode: false`` (the default). It only gates them off the
reproduction path. The enforcement is intentionally *config + filesystem* level
and runs at the top of the training CLI, before any checkpoint is loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

from sgdjscc_lab.training.stages import (
    STAGE_CONTROLNET,
    STAGE_EDGE_CODEC,
    STAGE_END_TO_END_FT,
    STAGE_JSCC,
    STAGE_TEXT_DM,
    StageConfigError,
)

logger = logging.getLogger(__name__)


class PaperModeError(StageConfigError):
    """Raised when a config violates a ``paper_mode`` guardrail.

    Subclasses :class:`StageConfigError` so the training CLI reports it through
    the same "fail before loading checkpoints" path as other stage-config errors.
    """


# Provenance sentinel written by ``scripts/generate_captions.py`` into any
# directory whose ``.txt`` sidecars it produced. paper_mode refuses to train a
# text stage on a directory containing it (auto-captions are NOT paper-faithful).
AUTOCAPTION_SENTINEL = "_AUTOCAPTION_PROVENANCE.json"

# Caption sources that may carry *dataset-provided* captions (allowed in
# paper_mode). ``sidecar`` is allowed ONLY when no auto-caption sentinel is
# present (checked separately). ``filename`` is a pseudo-caption → always blocked.
_PAPER_CAPTION_SOURCES = ("sidecar", "manifest", "coco_json", "multi_manifest")

# Edge sources considered structurally faithful (MuGE soft edge, per paper /
# public SGDJSCC). ``canny`` is paper-like/ablation → blocked in paper_mode.
_PAPER_EDGE_SOURCES = ("muge_sidecar", "muge_runtime")


def is_enabled(cfg: DictConfig) -> bool:
    """True if the top-level ``paper_mode`` flag is set."""
    return bool(OmegaConf.select(cfg, "paper_mode", default=False))


# ─────────────────────────────────────────────────────────────────────────────
# Individual guardrails (each is independently unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def _input_dirs(cfg: DictConfig) -> List[Path]:
    dirs: List[Path] = []
    for key in ("train_input_path", "val_input_path"):
        p = OmegaConf.select(cfg, key, default=None)
        if p:
            dirs.append(Path(str(p)))
    return dirs


def enforce_caption_policy(cfg: DictConfig, stage: str,
                           input_dirs: Optional[List[Path]] = None) -> None:
    """Block non-paper-faithful caption sources for text-guided stages.

    * ``caption_source: filename`` → blocked (pseudo-caption, smoke-test only).
    * Any input dir containing :data:`AUTOCAPTION_SENTINEL` → blocked with the
      explicit "CelebA-HQ auto-generated captions are not paper-faithful" message
      (the sentinel is written by ``scripts/generate_captions.py``).
    """
    if stage not in (STAGE_TEXT_DM, STAGE_CONTROLNET, STAGE_END_TO_END_FT):
        return
    src = str(OmegaConf.select(cfg, "train.dataset.caption_source", default="")).lower()
    if src == "filename":
        raise PaperModeError(
            "paper_mode: caption_source='filename' produces pseudo-captions "
            "(smoke-test only) and is NOT paper-faithful. Use dataset-provided "
            "captions (sidecar/manifest/coco_json/multi_manifest)."
        )
    if src and src not in _PAPER_CAPTION_SOURCES:
        raise PaperModeError(
            f"paper_mode: caption_source={src!r} is not an allowed paper source "
            f"{_PAPER_CAPTION_SOURCES}."
        )
    # Honesty caveat: a manually-placed .txt sidecar / manifest CANNOT be verified
    # as the paper's caption set — paper_mode only blocks *known* auto-captions
    # (the provenance sentinel below) and the 'filename' pseudo-source. Warn so the
    # user does not mistake "passed paper_mode" for "verified paper captions".
    if src in ("sidecar", "manifest"):
        logger.warning(
            "paper_mode: caption_source=%r is TRUSTED as dataset-provided but NOT "
            "verified (paper_mode only detects auto-captions via a provenance "
            "sentinel). Ensure these captions ship with the dataset.", src)
    # Filesystem provenance check: refuse auto-generated caption sidecars.
    for d in (input_dirs if input_dirs is not None else _input_dirs(cfg)):
        if not d:
            continue
        hits = []
        if (d / AUTOCAPTION_SENTINEL).is_file():
            hits.append(d / AUTOCAPTION_SENTINEL)
        else:
            try:
                hits = list(d.rglob(AUTOCAPTION_SENTINEL))
            except OSError:
                hits = []
        if hits:
            raise PaperModeError(
                "paper_mode: CelebA-HQ auto-generated captions are not "
                "paper-faithful. Found an auto-caption provenance marker under "
                f"{d} ({hits[0]}). These .txt sidecars were produced by "
                "scripts/generate_captions.py (BLIP2/fixed/filename) and the "
                "paper does not use them. Use a dataset whose captions ship with "
                "it (e.g. COCO captions_*.json via caption_source=coco_json), or "
                "run with paper_mode=false for the paper-like extension path."
            )


def enforce_edge_policy(cfg: DictConfig, stage: str) -> None:
    """Require a MuGE soft-edge source for the structural (edge) stages."""
    if stage not in (STAGE_CONTROLNET, STAGE_EDGE_CODEC):
        return
    src = str(OmegaConf.select(cfg, "train.dataset.edge_source", default="")).lower()
    if src not in _PAPER_EDGE_SOURCES:
        raise PaperModeError(
            f"paper_mode: edge_source={src!r} is not paper-faithful. The paper "
            "and public SGDJSCC use MuGE soft edges. Set train.dataset.edge_source "
            f"to one of {_PAPER_EDGE_SOURCES} (precompute with "
            "scripts/prepare_muge_edges.py for 'muge_sidecar'). 'canny' is a "
            "paper-like/ablation source — use paper_mode=false for it."
        )


def enforce_edge_transport_policy(cfg: DictConfig, stage: str) -> None:
    """Block the ``shared_vae`` ablation transport in paper_mode (Stage 3)."""
    if stage != STAGE_CONTROLNET:
        return
    from sgdjscc_lab.training.edge_transport import (
        EDGE_TRANSPORT_EDGE_JSCC, resolve_edge_transport)
    mode = resolve_edge_transport(cfg)
    if mode != EDGE_TRANSPORT_EDGE_JSCC:
        raise PaperModeError(
            f"paper_mode: edge_transport={mode!r} is an ablation. The paper "
            "transmits the edge over its OWN DeepJSCC link → use "
            "train.controlnet.edge_transport=edge_jscc (with a trained "
            "edge_codec checkpoint). 'shared_vae' is comparison-only."
        )


def enforce_trained_edge_checkpoint_policy(cfg: DictConfig, stage: str) -> None:
    """Require a concrete, existing edge-codec checkpoint for Stage-3 paper mode."""
    if stage != STAGE_CONTROLNET:
        return
    checkpoint = OmegaConf.select(cfg, "train.controlnet.edge_jscc.checkpoint", default=None)
    if not checkpoint:
        raise PaperModeError(
            "paper_mode: Stage-3 ControlNet requires a TRAINED edge_codec "
            "checkpoint. Set train.controlnet.edge_jscc.checkpoint to the "
            "best.pth produced by the edge_codec stage. A null checkpoint would "
            "make edge_jscc a random stand-in, not a paper-oriented edge link."
        )
    path = Path(str(checkpoint))
    if not path.is_file():
        raise PaperModeError(
            "paper_mode: Stage-3 ControlNet points to a missing edge_codec "
            "checkpoint:\n"
            f"  {path}\n"
            "Train edge_codec first, or fix train.controlnet.edge_jscc.checkpoint."
        )


def enforce_jscc_gan_policy(cfg: DictConfig, stage: str) -> None:
    """Require Stage-1 paper mode to use the paper-described MSE + GAN objective."""
    if stage != STAGE_JSCC:
        return
    enabled = bool(OmegaConf.select(cfg, "train.jscc.gan.enabled", default=False))
    if not enabled:
        raise PaperModeError(
            "paper_mode: Stage-1 JSCC must use the paper-described MSE + "
            "patch-GAN objective. Set train.jscc.gan.enabled=true. If you want "
            "the conservative MSE-only ablation, run with paper_mode=false."
        )


def enforce_cfg_null_policy(cfg: DictConfig, stage: str) -> None:
    """Require the learned CFG null token (not the zero-vector simplification)."""
    if stage not in (STAGE_TEXT_DM, STAGE_CONTROLNET, STAGE_END_TO_END_FT):
        return
    mode = str(OmegaConf.select(cfg, "train.dm.cfg_null_mode", default="zero")).lower()
    if mode != "learned":
        raise PaperModeError(
            "paper_mode: train.dm.cfg_null_mode='zero' is a simplification. "
            "Set train.dm.cfg_null_mode=learned so CFG dropout uses a trainable "
            "null token (closer to the conditional/unconditional branch the paper "
            "relies on for CFG scale 4.5)."
        )


# Extension flags that must be OFF for the paper baseline evaluation.
_EVAL_EXTENSION_FLAGS = (
    "use_phase4", "use_phase5", "use_packet_eval",
    "use_regeneration_loop", "use_regeneration_search",
)


def enforce_eval(cfg: DictConfig) -> None:
    """paper_mode guardrail for the EVALUATION path (called from evaluate.py).

    Blocks every extension feature (Phase 4/5, packet eval, regeneration,
    SRS-v2/VQA via the phase flags) and the ``shared_vae`` edge-transport
    ablation, so a config named "paper eval" actually runs the paper baseline.
    No-op when ``paper_mode`` is off.
    """
    if not is_enabled(cfg):
        return
    on = [f for f in _EVAL_EXTENSION_FLAGS
          if bool(OmegaConf.select(cfg, f, default=False))]
    if on:
        raise PaperModeError(
            "paper_mode (eval): the paper baseline disables all extensions, but "
            f"these are enabled: {on}. Set them to false (configs/paper_eval_awgn.yaml "
            "does), or run with paper_mode=false for the extended evaluation."
        )
    mode = str(OmegaConf.select(
        cfg, "train.controlnet.edge_transport", default="")).lower()
    if mode == "shared_vae":
        raise PaperModeError(
            "paper_mode (eval): edge_transport='shared_vae' is an ablation. The "
            "paper transmits the edge over its own DeepJSCC link (edge_jscc)."
        )
    # NOTE: the metric set is checked separately by enforce_eval_metrics() AFTER
    # evaluate.py resolves --profile / --no-clip overrides (those aren't known
    # here). So this only confirms the EXTENSION features are disabled.
    logger.info("paper_mode=ON (eval) → extension features disabled.")


def enforce_eval_metrics(cfg: DictConfig, enabled_metrics, no_clip: bool) -> None:
    """paper_mode guardrail on the FINAL evaluation metric set.

    Called by ``evaluate.py`` AFTER ``--profile`` / config ``metrics_profile`` /
    ``--no-clip`` are resolved, so the actual reported metrics are checked (not
    the pre-override config). In ``paper_mode`` the enabled metrics must be
    **exactly** the paper's reported set (``metric_profiles.PAPER_METRICS`` =
    PSNR/LPIPS/CLIP(img-img,txt-img)/FID):

    * ``--no-clip`` → rejected (CLIP is a paper-reported metric);
    * any ETRI/extended (non-paper) metric → rejected (``--profile extended`` /
      a custom list with SSIM/SRS/object/hallucination);
    * a **reduced** set (e.g. ``metrics: [psnr, lpips]`` missing CLIP/FID) →
      rejected — the log says "paper metric set", so the FULL set is required.

    No-op when ``paper_mode`` is off.
    """
    if not is_enabled(cfg):
        return
    from sgdjscc_lab.utils.metric_profiles import NON_PAPER_METRICS, resolve_profile
    paper_set = resolve_profile("paper")
    enabled = set(enabled_metrics)
    if no_clip:
        raise PaperModeError(
            "paper_mode (eval): --no-clip disables CLIP, which the paper reports. "
            "Drop --no-clip for the paper metric set."
        )
    extra = sorted(enabled - paper_set)
    if extra:
        non_paper = sorted(set(extra) & set(NON_PAPER_METRICS))
        detail = (f"non-paper (ETRI/extended) metrics {non_paper}" if non_paper
                  else f"unexpected metrics {extra}")
        raise PaperModeError(
            f"paper_mode (eval): {detail} are enabled, but the paper reports only "
            f"{sorted(paper_set)}. Use metrics_profile: paper "
            "(configs/paper_eval_awgn.yaml does); do not pass --profile extended/full."
        )
    missing = sorted(paper_set - enabled)
    if missing:
        raise PaperModeError(
            f"paper_mode (eval): the paper reports the FULL set {sorted(paper_set)}, "
            f"but these are missing from the enabled metrics: {missing}. Use "
            "metrics_profile: paper (configs/paper_eval_awgn.yaml does) — a reduced "
            "metric set is not 'the paper metric set'."
        )
    logger.info("paper_mode=ON (eval) → paper metric set confirmed (%s).",
                sorted(paper_set))


def enforce_edge_codec_snr_policy(cfg: DictConfig, stage: str) -> None:
    """Require SNR-conditioned, multi-SNR edge-codec training in paper_mode."""
    if stage != STAGE_EDGE_CODEC:
        return
    arch = str(OmegaConf.select(cfg, "train.edge_codec.arch", default="conv")).lower()
    multi = bool(OmegaConf.select(cfg, "train.edge_codec.multi_snr.enabled", default=False))
    snr_cond = bool(OmegaConf.select(cfg, "train.edge_codec.vit.snr_cond", default=False))
    if not multi:
        raise PaperModeError(
            "paper_mode: edge_codec must train across SNRs. Set "
            "train.edge_codec.multi_snr.enabled=true (samples the edge-link SNR "
            "per step) so the SNR conditioning is actually exercised."
        )
    if arch == "vit" and not snr_cond:
        raise PaperModeError(
            "paper_mode: edge_codec arch='vit' must enable SNR conditioning. "
            "Set train.edge_codec.vit.snr_cond=true."
        )


def _assumed(cfg: DictConfig, dotted: str, default=None):
    return OmegaConf.select(cfg, f"paper_assumed_hparams.{dotted}", default=default)


def _require_float_match(cfg: DictConfig, actual_key: str, assumed_key: str) -> None:
    actual = OmegaConf.select(cfg, actual_key, default=None)
    assumed = _assumed(cfg, assumed_key, default=None)
    if actual is None or assumed is None:
        raise PaperModeError(
            "paper_mode: missing paper-assumed hyperparameter wiring. "
            f"Need both {actual_key!r} and paper_assumed_hparams.{assumed_key!r}."
        )
    if abs(float(actual) - float(assumed)) > 1e-12:
        raise PaperModeError(
            "paper_mode: active hyperparameter differs from the explicit "
            f"paper-assumed value: {actual_key}={actual} but "
            f"paper_assumed_hparams.{assumed_key}={assumed}. Update both "
            "intentionally, or run with paper_mode=false for an ablation."
        )


def _require_str_match(cfg: DictConfig, actual_key: str, assumed_key: str) -> None:
    actual = OmegaConf.select(cfg, actual_key, default=None)
    assumed = _assumed(cfg, assumed_key, default=None)
    if actual is None or assumed is None:
        raise PaperModeError(
            "paper_mode: missing paper-assumed hyperparameter wiring. "
            f"Need both {actual_key!r} and paper_assumed_hparams.{assumed_key!r}."
        )
    if str(actual).lower() != str(assumed).lower():
        raise PaperModeError(
            "paper_mode: active hyperparameter differs from the explicit "
            f"paper-assumed value: {actual_key}={actual!r} but "
            f"paper_assumed_hparams.{assumed_key}={assumed!r}. Update both "
            "intentionally, or run with paper_mode=false for an ablation."
        )


def enforce_assumed_hparams_policy(cfg: DictConfig, stage: str) -> None:
    """Keep unpublished paper-like knobs explicit and in sync with train.* values."""
    if OmegaConf.select(cfg, "paper_assumed_hparams", default=None) is None:
        raise PaperModeError(
            "paper_mode: missing top-level paper_assumed_hparams block. "
            "Unpublished values must be explicit rather than hidden in train.*."
        )

    # Shared optimizer assumptions.
    _require_float_match(cfg, "train.lr", "optimizer.lr")
    _require_float_match(cfg, "train.weight_decay", "optimizer.weight_decay")

    if stage == STAGE_JSCC:
        _require_float_match(cfg, "train.jscc.gan.weight", "jscc_gan.weight")
        _require_str_match(cfg, "train.jscc.gan.mode", "jscc_gan.mode")
        _require_float_match(cfg, "train.jscc.gan.lr", "jscc_gan.lr")
        _require_float_match(cfg, "train.jscc.gan.ndf", "jscc_gan.ndf")
        _require_float_match(cfg, "train.jscc.gan.n_layers", "jscc_gan.n_layers")
        _require_str_match(cfg, "train.jscc.gan.norm", "jscc_gan.norm")

    if stage in (STAGE_TEXT_DM, STAGE_CONTROLNET, STAGE_END_TO_END_FT):
        _require_float_match(cfg, "train.dm.cfg_dropout_prob", "dm.cfg_dropout_prob")
        _require_str_match(cfg, "train.dm.cfg_null_mode", "dm.cfg_null_mode")

    if stage == STAGE_EDGE_CODEC:
        _require_float_match(
            cfg, "train.edge_codec.multi_snr.min_db", "edge_codec.multi_snr_min_db")
        _require_float_match(
            cfg, "train.edge_codec.multi_snr.max_db", "edge_codec.multi_snr_max_db")
        arch = str(OmegaConf.select(cfg, "train.edge_codec.arch", default="")).lower()
        if arch == "vit":
            _require_float_match(cfg, "train.edge_codec.vit.embed_dim", "edge_codec.vit_embed_dim")
            _require_float_match(cfg, "train.edge_codec.vit.depth", "edge_codec.vit_depth")
            _require_float_match(cfg, "train.edge_codec.vit.num_heads", "edge_codec.vit_num_heads")
            _require_float_match(cfg, "train.edge_codec.vit.mlp_ratio", "edge_codec.vit_mlp_ratio")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def enforce(cfg: DictConfig, stage: str,
            input_dirs: Optional[List[Path]] = None) -> None:
    """Run every paper_mode guardrail relevant to *stage* (no-op if disabled).

    Called from the training CLI right after ``validate_stage_config``. Raises
    :class:`PaperModeError` (→ explicit CLI failure) on the first violation.
    """
    if not is_enabled(cfg):
        return
    logger.info("paper_mode=ON → enforcing paper-faithful guardrails for stage=%s", stage)
    enforce_assumed_hparams_policy(cfg, stage)
    enforce_jscc_gan_policy(cfg, stage)
    enforce_caption_policy(cfg, stage, input_dirs)
    enforce_edge_policy(cfg, stage)
    enforce_edge_transport_policy(cfg, stage)
    enforce_trained_edge_checkpoint_policy(cfg, stage)
    enforce_cfg_null_policy(cfg, stage)
    enforce_edge_codec_snr_policy(cfg, stage)


def summary() -> str:
    """Human-readable list of what paper_mode enforces (for logs/docs).

    Worded to match what is ACTUALLY enforced — in particular it blocks *known*
    auto-captions (provenance sentinel) + the 'filename' pseudo-source, but it
    cannot verify that a hand-placed sidecar/manifest is the paper's caption set.
    """
    return (
        "paper_mode enforces: blocks auto-generated captions (provenance sentinel) "
        "and the 'filename' pseudo-source (sidecar/manifest are TRUSTED, not "
        "verified); Stage-1 MSE+GAN (no MSE-only paper claim); MuGE soft edges "
        "(no Canny); trained edge_jscc checkpoint (no random/shared_vae edge link); "
        "learned CFG null token (no zero-vector); explicit paper_assumed_hparams; "
        "multi-SNR edge codec; and (eval) all extensions disabled."
    )
