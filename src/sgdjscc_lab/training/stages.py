"""training/stages.py – Stage definitions and config validation for the
paper-faithful 3-stage SGD-JSCC training procedure.

The SGD-JSCC paper ("Semantics-Guided Diffusion for Deep Joint Source-Channel
Coding in Wireless Image Transmission", Sec. VI "Training Details") trains the
system in **three sequential stages**:

  stage 1 ``jscc``        JSCC encoder/decoder jointly trained on images under a
                          *fixed* AWGN channel (SNR = 10 dB).  MSE (+ patch-GAN)
                          loss.  The JSCC model is *frozen* after this stage.
  stage 2 ``text_dm``     A text-guided latent diffusion model (MDTv2) trained on
                          text-image pairs to predict the clean latent ``f0``
                          from a noised latent ``ft`` (Algorithm 1).
  stage 3 ``controlnet``  Edge-map structural guidance added via a ControlNet
                          branch on top of the *frozen* stage-2 DM; only the
                          structural branches are updated.

This module is intentionally model-free: it only resolves which stage is active
and validates that the config provides the inputs that stage requires.  The
heavy lifting (datasets, forward passes, losses, freeze policy) lives in
``data/datasets.py``, ``training/stage_runners.py``, ``training/losses.py`` and
``training/freeze.py``.
"""

from __future__ import annotations

from typing import Optional

from omegaconf import DictConfig, OmegaConf

# ── Stage identifiers ─────────────────────────────────────────────────────────
# Core baseline = the three sequential SGD-JSCC stages below.
STAGE_JSCC = "jscc"
STAGE_TEXT_DM = "text_dm"
STAGE_CONTROLNET = "controlnet"
# Supporting codec-training step for Stage 3's `edge_jscc` transport: trains the
# dedicated edge JSCC (BCE+Dice) whose checkpoint Stage 3 loads as side info.
# It is NOT a stage of the main image pipeline — it produces a component the
# controlnet baseline consumes. See docs/training_scaffold.md "Edge codec".
STAGE_EDGE_CODEC = "edge_codec"
# Supporting step: train the blind SNR estimator (paper Sec. IV-C, eq. 15) on
# image latents with synthetic noise. See docs/training_scaffold.md "CSI".
STAGE_CSI_ESTIMATION = "csi_estimation"
# Paper-appendix EXTENSION (not part of the core baseline): joint JSCC↔DM
# end-to-end fine-tuning. See docs/training_scaffold.md "End-to-end fine-tuning".
STAGE_END_TO_END_FT = "end_to_end_ft"
# The three core baseline stages, in training order.
CORE_STAGES = (STAGE_JSCC, STAGE_TEXT_DM, STAGE_CONTROLNET)
VALID_STAGES = (
    STAGE_JSCC, STAGE_TEXT_DM, STAGE_CONTROLNET,
    STAGE_EDGE_CODEC, STAGE_CSI_ESTIMATION, STAGE_END_TO_END_FT,
)

# ── Default dataset type per stage ────────────────────────────────────────────
# Used when ``train.dataset.type`` is "auto" (the default).
STAGE_DATASET_TYPE = {
    STAGE_JSCC: "image",
    STAGE_TEXT_DM: "text_image",
    STAGE_CONTROLNET: "text_image_edge",
    # edge codec trains on edge maps only — no captions needed.
    STAGE_EDGE_CODEC: "edge",
    # CSI estimation trains on image latents (image-only data).
    STAGE_CSI_ESTIMATION: "image",
    # e2e needs captions; edge only when the ControlNet branch is fine-tuned.
    STAGE_END_TO_END_FT: "text_image",
}

# Caption sources understood by TextImage / TextImageEdge datasets.
#   single-caption : sidecar | manifest | filename
#   multi-caption  : coco_json (COCO captions_*.json) | multi_manifest (JSON
#                    {filename: [captions]}); one caption is picked per access via
#                    train.dataset.caption_select (first | longest | random).
VALID_CAPTION_SOURCES = ("sidecar", "manifest", "filename", "coco_json", "multi_manifest")
# Edge sources understood by TextImageEdge dataset.
VALID_EDGE_SOURCES = ("sidecar", "canny")


class StageConfigError(ValueError):
    """Raised when a stage's required config inputs are missing or inconsistent.

    The CLI converts this into an early, explicit failure so a misconfigured run
    fails *before* loading checkpoints — never silently doing the wrong thing.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_stage(cfg: DictConfig) -> str:
    """Return the active training stage from ``train.stage`` (default ``jscc``).

    Raises
    ------
    StageConfigError
        If ``train.stage`` is set to an unknown value.
    """
    stage = OmegaConf.select(cfg, "train.stage", default=STAGE_JSCC)
    stage = str(stage).lower().strip()
    if stage not in VALID_STAGES:
        raise StageConfigError(
            f"Unknown train.stage={stage!r}. "
            f"Valid stages: {', '.join(VALID_STAGES)}."
        )
    return stage


def resolve_dataset_type(cfg: DictConfig, stage: Optional[str] = None) -> str:
    """Return the dataset type for *stage*.

    ``train.dataset.type`` may be ``auto`` (→ derived from the stage) or one of
    ``image`` / ``text_image`` / ``text_image_edge`` to override.
    """
    if stage is None:
        stage = resolve_stage(cfg)
    ds_type = str(OmegaConf.select(cfg, "train.dataset.type", default="auto")).lower()
    if ds_type == "auto":
        # end_to_end_ft needs edges only when the ControlNet branch is fine-tuned;
        # promoting the dataset keeps config intent and the forward path aligned.
        if stage == STAGE_END_TO_END_FT and bool(OmegaConf.select(
                cfg, "train.end_to_end_ft.train_controlnet", default=False)):
            return "text_image_edge"
        return STAGE_DATASET_TYPE[stage]
    valid = set(STAGE_DATASET_TYPE.values())
    if ds_type not in valid:
        raise StageConfigError(
            f"Unknown train.dataset.type={ds_type!r}. "
            f"Valid: auto, {', '.join(sorted(valid))}."
        )
    return ds_type


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_caption_source(cfg: DictConfig, stage: str) -> None:
    """Validate the caption source for the text-guided stages (text_dm/controlnet).

    Checks presence, valid value, and that ``caption_path`` is set when the
    source is ``manifest`` — so a bad caption config fails up-front rather than
    at dataset-construction time.
    """
    src = OmegaConf.select(cfg, "train.dataset.caption_source", default=None)
    if src is None:
        hint = ("the base DM is text-guided, so captions are still required"
                if stage == STAGE_CONTROLNET else "a caption/text source is required")
        raise StageConfigError(
            f"stage={stage!r}: {hint}. Set train.dataset.caption_source to one of "
            f"{VALID_CAPTION_SOURCES} (e.g. 'sidecar' for per-image .txt files)."
        )
    if str(src).lower() not in VALID_CAPTION_SOURCES:
        raise StageConfigError(
            f"Unknown train.dataset.caption_source={src!r}. "
            f"Valid: {', '.join(VALID_CAPTION_SOURCES)}."
        )
    if str(src).lower() in ("manifest", "coco_json", "multi_manifest") and not OmegaConf.select(
        cfg, "train.dataset.caption_path", default=None
    ):
        raise StageConfigError(
            f"caption_source={str(src).lower()!r} requires train.dataset.caption_path "
            "(manifest: JSON/CSV filename→caption; coco_json: a COCO captions_*.json; "
            "multi_manifest: JSON {filename: [captions]})."
        )


def validate_stage_config(cfg: DictConfig, stage: Optional[str] = None) -> str:
    """Validate that *cfg* supplies the inputs required by *stage*.

    Returns the resolved stage name on success; raises ``StageConfigError`` with
    an actionable message otherwise.  Performs **only** cheap, file-system-free
    checks so it can run at the very top of the CLI before any model loading.
    """
    if stage is None:
        stage = resolve_stage(cfg)

    train_input = OmegaConf.select(cfg, "train_input_path", default=None)
    file_list_mode = str(OmegaConf.select(
        cfg, "train.dataset.input_mode", default="folder")).lower() == "file_list"
    file_list_path = OmegaConf.select(cfg, "train.dataset.file_list_path", default=None)
    if not train_input and not (file_list_mode and file_list_path):
        raise StageConfigError(
            f"stage={stage!r} requires training images. Either set 'train_input_path' "
            "(or pass --train-list /path/to/images/), OR set "
            "train.dataset.input_mode=file_list + train.dataset.file_list_path."
        )

    if stage == STAGE_JSCC:
        # image-only data is sufficient; nothing else mandatory.
        snr_db = OmegaConf.select(cfg, "train.jscc.snr_db", default=None)
        if snr_db is None:
            raise StageConfigError(
                "stage='jscc' requires train.jscc.snr_db (paper uses a fixed AWGN "
                "SNR = 10 dB for JSCC training)."
            )
        return stage

    if stage == STAGE_EDGE_CODEC:
        # Edge-codec training needs ONLY an edge source (no captions: it is a
        # self-supervised edge reconstruction codec, not text-guided).
        edge_src = OmegaConf.select(cfg, "train.dataset.edge_source", default=None)
        if edge_src is None:
            raise StageConfigError(
                "stage='edge_codec' requires an edge/structure source. "
                "Set train.dataset.edge_source to one of "
                f"{VALID_EDGE_SOURCES} ('canny' computes edges on the fly from "
                "the training images; 'sidecar' reads precomputed edge maps from "
                "train.dataset.edge_dir). No captions are needed."
            )
        if str(edge_src).lower() not in VALID_EDGE_SOURCES:
            raise StageConfigError(
                f"Unknown train.dataset.edge_source={edge_src!r}. "
                f"Valid: {', '.join(VALID_EDGE_SOURCES)}."
            )
        if str(edge_src).lower() == "sidecar" and not OmegaConf.select(
            cfg, "train.dataset.edge_dir", default=None
        ):
            raise StageConfigError(
                "edge_source='sidecar' requires train.dataset.edge_dir "
                "(a directory of edge maps matching the image filenames)."
            )
        return stage

    if stage == STAGE_CSI_ESTIMATION:
        # Image-only: the SNR estimator is trained self-supervised on synthetic
        # noisy latents (√α·f0 + √(1-α)·n). Needs the JSCC VAE for image latents
        # (checked at runtime in build_stage_runner), nothing else mandatory here.
        return stage

    if stage == STAGE_TEXT_DM:
        _validate_caption_source(cfg, stage)
        return stage

    if stage == STAGE_CONTROLNET:
        # ControlNet still needs captions (the base DM is text-guided) AND edges.
        _validate_caption_source(cfg, stage)
        edge_src = OmegaConf.select(cfg, "train.dataset.edge_source", default=None)
        if edge_src is None:
            raise StageConfigError(
                "stage='controlnet' requires an edge/structure source. "
                "Set train.dataset.edge_source to one of "
                f"{VALID_EDGE_SOURCES} ('canny' computes edges on the fly; "
                "'sidecar' reads edge maps from train.dataset.edge_dir)."
            )
        if str(edge_src).lower() not in VALID_EDGE_SOURCES:
            raise StageConfigError(
                f"Unknown train.dataset.edge_source={edge_src!r}. "
                f"Valid: {', '.join(VALID_EDGE_SOURCES)}."
            )
        if str(edge_src).lower() == "sidecar" and not OmegaConf.select(
            cfg, "train.dataset.edge_dir", default=None
        ):
            raise StageConfigError(
                "edge_source='sidecar' requires train.dataset.edge_dir "
                "(a directory of edge maps matching the image filenames)."
            )
        # Validate the edge transport mode early (shared_vae | edge_jscc).
        from sgdjscc_lab.training.edge_transport import resolve_edge_transport
        resolve_edge_transport(cfg)
        return stage

    if stage == STAGE_END_TO_END_FT:
        # End-to-end FT is text-guided, so captions are required.
        _validate_caption_source(cfg, stage)
        ds_type = resolve_dataset_type(cfg, stage)
        train_ctrl = bool(OmegaConf.select(
            cfg, "train.end_to_end_ft.train_controlnet", default=False))

        # Consistency: fine-tuning the ControlNet branch only makes sense if the
        # forward actually receives an edge condition. That requires the
        # text_image_edge dataset. resolve_dataset_type promotes type=auto, but an
        # EXPLICIT type=text_image with train_controlnet=true is contradictory —
        # the branch would train while the forward runs with c=None. Reject it.
        if train_ctrl and ds_type != "text_image_edge":
            raise StageConfigError(
                "stage='end_to_end_ft' with train_controlnet=true needs edge "
                "conditioning, but train.dataset.type resolved to "
                f"{ds_type!r}. Set train.dataset.type=text_image_edge (or 'auto') "
                "and provide train.dataset.edge_source, or disable train_controlnet."
            )

        if ds_type == "text_image_edge":
            edge_src = OmegaConf.select(cfg, "train.dataset.edge_source", default=None)
            if edge_src is None:
                raise StageConfigError(
                    "stage='end_to_end_ft' with edge conditioning requires "
                    "train.dataset.edge_source (one of "
                    f"{VALID_EDGE_SOURCES}). Use dataset.type=text_image to skip "
                    "edges, or provide an edge source."
                )
            if str(edge_src).lower() not in VALID_EDGE_SOURCES:
                raise StageConfigError(
                    f"Unknown train.dataset.edge_source={edge_src!r}. "
                    f"Valid: {', '.join(VALID_EDGE_SOURCES)}."
                )
        # at least one module must be trainable
        train_jscc = bool(OmegaConf.select(cfg, "train.end_to_end_ft.train_jscc", default=True))
        train_dm = bool(OmegaConf.select(cfg, "train.end_to_end_ft.train_diffusion", default=True))
        if not (train_jscc or train_dm or train_ctrl):
            raise StageConfigError(
                "stage='end_to_end_ft' has nothing to train: set at least one of "
                "train.end_to_end_ft.{train_jscc,train_diffusion,train_controlnet} to true."
            )
        return stage

    # Unreachable: resolve_stage already guards the value.
    raise StageConfigError(f"Unhandled stage {stage!r}.")
