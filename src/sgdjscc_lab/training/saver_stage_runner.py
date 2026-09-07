"""Staged optimizer/checkpoint runner for the SAVER tensor pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import torch

from sgdjscc_lab.models.saver.contracts import SaverCheckpointContract, SemanticState
from sgdjscc_lab.pipelines.saver_video_pipeline import SaverForwardOutput, SaverVideoPipeline

from .saver_losses import SaverLoss


SAVER_STAGES = ("sv1", "sv2", "sv3", "end_to_end")


class SaverStageRunner:
    """Train one declared SAVER stage and fail closed on checkpoint mismatch."""

    _STAGE_MODULES = {
        "sv1": ("tokenizer", "memory"),
        "sv2": ("dit_adapters",),
        "sv3": ("router", "codec"),
        "end_to_end": ("tokenizer", "memory", "dit_adapters", "router", "codec"),
    }

    def __init__(
        self,
        pipeline: SaverVideoPipeline,
        loss: SaverLoss,
        contract: SaverCheckpointContract,
        *,
        stage: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ) -> None:
        if stage not in SAVER_STAGES:
            raise ValueError(f"unknown SAVER stage={stage!r}; expected one of {SAVER_STAGES}")
        self.pipeline = pipeline
        self.loss = loss
        self.contract = contract
        self.stage = stage
        self._apply_freeze()
        parameters = [p for p in pipeline.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError(f"stage={stage} has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        self.global_step = 0

    def _apply_freeze(self) -> None:
        for parameter in self.pipeline.parameters():
            parameter.requires_grad_(False)
        for name in self._STAGE_MODULES[self.stage]:
            for parameter in getattr(self.pipeline, name).parameters():
                parameter.requires_grad_(True)

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, p in self.pipeline.named_parameters() if p.requires_grad)

    def forward_and_loss(self, batch: Mapping) -> tuple[SaverForwardOutput, Dict[str, torch.Tensor]]:
        output = self.pipeline(**batch["pipeline_inputs"])
        if not output.enabled or output.assertion is None or output.routing is None:
            raise RuntimeError("SAVER training requires an enabled pipeline")
        targets = batch.get("targets", {})
        delivered = output.memory_update.accepted_mask if output.memory_update is not None else None
        semantic_states = batch["pipeline_inputs"]["semantic_states"]
        losses = self.loss(
            diffusion_prediction=output.diffusion_hidden,
            diffusion_target=targets.get("diffusion_target"),
            state_logits=output.assertion.state_logits,
            state_targets=semantic_states,
            action_logits=output.assertion.action_logits,
            action_targets=batch["pipeline_inputs"].get("teacher_actions"),
            valid_mask=output.assertion.valid_mask,
            memory_anchor=output.memory_state.identity,
            memory_positive=targets.get("memory_positive"),
            memory_negative=targets.get("memory_negative"),
            entity_presence_logits=targets.get("entity_presence_logits"),
            absent_mask=semantic_states.eq(int(SemanticState.CONFIRMED_ABSENT)),
            present_mask=semantic_states.eq(int(SemanticState.PRESENT)),
            rate_penalty=output.routing.budget_penalty,
            decoded_event_features=output.decoded_event_features,
            event_target_features=targets.get("event_target_features"),
            delivered_mask=delivered,
        )
        return output, losses

    def training_step(self, batch: Mapping) -> Dict[str, float]:
        self.pipeline.train()
        self.optimizer.zero_grad(set_to_none=True)
        _, losses = self.forward_and_loss(batch)
        if not losses["loss"].requires_grad:
            raise RuntimeError(f"stage={self.stage} has no gradient path from SAVER loss")
        losses["loss"].backward()
        self.optimizer.step()
        self.global_step += 1
        return {name: float(value.detach().cpu()) for name, value in losses.items()}

    @torch.no_grad()
    def validation_step(self, batch: Mapping) -> Dict[str, float]:
        self.pipeline.eval()
        _, losses = self.forward_and_loss(batch)
        return {name: float(value.detach().cpu()) for name, value in losses.items()}

    def checkpoint_payload(self, extra: Optional[Mapping] = None) -> dict:
        return {
            "kind": "saver_jscc_checkpoint",
            "contract": self.contract.to_dict(),
            "stage": self.stage,
            "global_step": self.global_step,
            "model": self.pipeline.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "extra": dict(extra or {}),
        }

    def save_checkpoint(self, path: str | Path, extra: Optional[Mapping] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.checkpoint_payload(extra), temporary)
        temporary.replace(path)
        return path

    def load_checkpoint_payload(self, payload: Mapping, *, load_optimizer: bool = True) -> None:
        if payload.get("kind") != "saver_jscc_checkpoint":
            raise RuntimeError("not a SAVER-JSCC checkpoint")
        self.contract.assert_compatible(payload["contract"])
        if payload.get("stage") != self.stage:
            raise RuntimeError(
                f"checkpoint stage={payload.get('stage')!r} does not match runner stage={self.stage!r}"
            )
        self.pipeline.load_state_dict(payload["model"], strict=True)
        if load_optimizer:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload.get("global_step", 0))

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool = True) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        self.load_checkpoint_payload(payload, load_optimizer=load_optimizer)

