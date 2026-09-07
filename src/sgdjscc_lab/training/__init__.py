"""sgdjscc_lab.training – Loss functions, noise schedule, freeze policy and
stage runners for the SGD-JSCC training procedure (3 core stages + end-to-end
fine-tuning extension)."""
from sgdjscc_lab.training.losses import (
    ReconstructionLoss,
    SemanticLoss,
    TotalLoss,
    build_loss,
    PatchDiscriminator,
    build_discriminator,
    GANLoss,
    JSCCStageLoss,
    DiffusionF0Loss,
    EndToEndFTLoss,
    build_stage_loss,
)
from sgdjscc_lab.training.noise_schedule import SigmoidNoiseScheduler
from sgdjscc_lab.training.stages import (
    STAGE_JSCC,
    STAGE_TEXT_DM,
    STAGE_CONTROLNET,
    STAGE_END_TO_END_FT,
    VALID_STAGES,
    StageConfigError,
    resolve_stage,
    resolve_dataset_type,
    validate_stage_config,
)
from sgdjscc_lab.training.edge_transport import (
    build_edge_transport,
    resolve_edge_transport,
    VALID_EDGE_TRANSPORTS,
)
from sgdjscc_lab.training.freeze import apply_stage_freeze_policy
from sgdjscc_lab.training.stage_runners import (
    StageRunner,
    JSCCStageRunner,
    TextDMStageRunner,
    ControlNetStageRunner,
    EndToEndFTStageRunner,
    build_stage_runner,
)

__all__ = [
    # losses
    "ReconstructionLoss", "SemanticLoss", "TotalLoss", "build_loss",
    "PatchDiscriminator", "build_discriminator", "GANLoss", "JSCCStageLoss",
    "DiffusionF0Loss", "EndToEndFTLoss", "build_stage_loss",
    # schedule
    "SigmoidNoiseScheduler",
    # stages
    "STAGE_JSCC", "STAGE_TEXT_DM", "STAGE_CONTROLNET", "STAGE_END_TO_END_FT",
    "VALID_STAGES", "StageConfigError", "resolve_stage", "resolve_dataset_type",
    "validate_stage_config",
    # edge transport
    "build_edge_transport", "resolve_edge_transport", "VALID_EDGE_TRANSPORTS",
    # freeze + runners
    "apply_stage_freeze_policy",
    "StageRunner", "JSCCStageRunner", "TextDMStageRunner",
    "ControlNetStageRunner", "EndToEndFTStageRunner", "build_stage_runner",
]
