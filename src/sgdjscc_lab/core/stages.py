"""Training-stage identifiers and dataset-selection rules.

This module intentionally has no dependency on the training or data packages.
It is the canonical home for definitions needed by both layers.
"""

from __future__ import annotations

from typing import Optional

from omegaconf import DictConfig, OmegaConf

STAGE_JSCC = "jscc"
STAGE_TEXT_DM = "text_dm"
STAGE_CONTROLNET = "controlnet"
STAGE_EDGE_CODEC = "edge_codec"
STAGE_CSI_ESTIMATION = "csi_estimation"
STAGE_END_TO_END_FT = "end_to_end_ft"

CORE_STAGES = (STAGE_JSCC, STAGE_TEXT_DM, STAGE_CONTROLNET)
VALID_STAGES = (
    STAGE_JSCC,
    STAGE_TEXT_DM,
    STAGE_CONTROLNET,
    STAGE_EDGE_CODEC,
    STAGE_CSI_ESTIMATION,
    STAGE_END_TO_END_FT,
)

STAGE_DATASET_TYPE = {
    STAGE_JSCC: "image",
    STAGE_TEXT_DM: "text_image",
    STAGE_CONTROLNET: "text_image_edge",
    STAGE_EDGE_CODEC: "edge",
    STAGE_CSI_ESTIMATION: "image",
    STAGE_END_TO_END_FT: "text_image",
}

VALID_CAPTION_SOURCES = (
    "sidecar",
    "manifest",
    "filename",
    "coco_json",
    "multi_manifest",
)
VALID_EDGE_SOURCES = ("sidecar", "canny", "muge_sidecar", "muge_runtime")


class StageConfigError(ValueError):
    """Raised when a stage configuration is invalid."""


def resolve_stage(cfg: DictConfig) -> str:
    """Return the active training stage from ``train.stage``."""
    stage = str(OmegaConf.select(cfg, "train.stage", default=STAGE_JSCC)).lower().strip()
    if stage not in VALID_STAGES:
        raise StageConfigError(
            f"Unknown train.stage={stage!r}. Valid stages: {', '.join(VALID_STAGES)}."
        )
    return stage


def resolve_dataset_type(cfg: DictConfig, stage: Optional[str] = None) -> str:
    """Resolve the dataset type required by a training stage."""
    if stage is None:
        stage = resolve_stage(cfg)
    ds_type = str(OmegaConf.select(cfg, "train.dataset.type", default="auto")).lower()
    if ds_type == "auto":
        if stage == STAGE_END_TO_END_FT and bool(
            OmegaConf.select(cfg, "train.end_to_end_ft.train_controlnet", default=False)
        ):
            return "text_image_edge"
        return STAGE_DATASET_TYPE[stage]
    valid = set(STAGE_DATASET_TYPE.values())
    if ds_type not in valid:
        raise StageConfigError(
            f"Unknown train.dataset.type={ds_type!r}. "
            f"Valid: auto, {', '.join(sorted(valid))}."
        )
    return ds_type
