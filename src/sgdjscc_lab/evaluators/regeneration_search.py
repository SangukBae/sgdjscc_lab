"""evaluators/regeneration_search.py – Multi-strategy regeneration search (Phase 5-C).

Replaces the Phase 4-A one-shot retry with a *search*: several reconstruction
strategies are tried and the output with the highest **verified** score
(SRS or SRS-v2) is kept — "choose the best verified reconstruction, not the first
successful one."

Strategies (each a declarative cfg adjustment):

    strong_text_weak_edge      raise text CFG, lower ControlNet (recover content)
    weak_text_strong_edge      lower text CFG, raise ControlNet (reduce hallucination)
    unconditional              drop text guidance (fallback)
    channel_conditioned_retry  re-run via the Phase 5-A channel-conditioned path

Ordering is delegated to ``controllers/adaptive_search_policy.AdaptiveSearchPolicy``.
The reconstruct / verify callables are injected so the search is unit-testable
without models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class SearchStrategy:
    name: str
    guidance_mult: float = 1.0
    controlnet_mult: float = 1.0
    use_text: Optional[bool] = None
    extra: Optional[Dict] = None     # extra cfg flags (e.g. channel conditioning)


SEARCH_STRATEGIES: Dict[str, SearchStrategy] = {
    "strong_text_weak_edge": SearchStrategy(
        "strong_text_weak_edge", guidance_mult=1.5, controlnet_mult=0.6, use_text=True),
    "weak_text_strong_edge": SearchStrategy(
        "weak_text_strong_edge", guidance_mult=0.6, controlnet_mult=1.5, use_text=True),
    "unconditional": SearchStrategy(
        "unconditional", guidance_mult=0.3, controlnet_mult=1.0, use_text=False),
    "channel_conditioned_retry": SearchStrategy(
        "channel_conditioned_retry", guidance_mult=1.0, controlnet_mult=1.1,
        extra={"use_channel_conditioning": True}),
}


def apply_search_strategy(cfg: DictConfig, strategy: SearchStrategy) -> DictConfig:
    """Return a deep copy of *cfg* with *strategy* applied (never mutates input)."""
    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    out.guidance_scale = round(float(cfg.get("guidance_scale", 4.0)) * strategy.guidance_mult, 6)
    out.controlnet_scale = round(float(cfg.get("controlnet_scale", 0.3)) * strategy.controlnet_mult, 6)
    if strategy.use_text is not None:
        out.use_text = bool(strategy.use_text)
    for k, v in (strategy.extra or {}).items():
        out[k] = v
    return out


class RegenerationSearch:
    """Search over regeneration strategies, keeping the best verified output.

    Parameters
    ----------
    reconstruct_fn:
        ``(cfg) -> reconstruction``.
    verify_fn:
        ``(reconstruction) -> score`` (higher is better; SRS or SRS-v2).
    policy:
        ``AdaptiveSearchPolicy`` for ordering (created lazily if None).
    strategies:
        Optional override of the strategy table.
    """

    def __init__(self, reconstruct_fn: Callable, verify_fn: Callable,
                 policy=None, strategies: Optional[Dict[str, SearchStrategy]] = None) -> None:
        self.reconstruct_fn = reconstruct_fn
        self.verify_fn = verify_fn
        self._policy = policy
        self.strategies = strategies or dict(SEARCH_STRATEGIES)

    def _get_policy(self):
        if self._policy is None:
            from sgdjscc_lab.controllers.adaptive_search_policy import AdaptiveSearchPolicy
            self._policy = AdaptiveSearchPolicy()
        return self._policy

    def search(
        self,
        base_cfg: DictConfig,
        order: Optional[List[str]] = None,
        error_report: Optional[Dict] = None,
        hallucination_score: Optional[float] = None,
        channel_state: Optional[Dict] = None,
        initial_recon=None,
        initial_score: Optional[float] = None,
        max_strategies: Optional[int] = None,
    ) -> Dict:
        """Run the search and return the best verified reconstruction.

        Returns
        -------
        dict with ``best_recon``, ``best_cfg``, ``best_score``, ``best_strategy``
        and ``history`` (list of ``{strategy, score}``).
        """
        if order is None:
            order = self._get_policy().order(
                error_report=error_report, hallucination_score=hallucination_score,
                channel_state=channel_state, max_strategies=max_strategies,
            )

        history: List[Dict] = []
        best_recon, best_cfg, best_strategy = initial_recon, base_cfg, "none"
        best_score = float(initial_score) if initial_score is not None else float("-inf")
        if initial_score is not None:
            history.append({"strategy": "none", "score": float(initial_score)})

        for name in order:
            strat = self.strategies.get(name)
            if strat is None:
                continue
            cfg_i = apply_search_strategy(base_cfg, strat)
            try:
                recon = self.reconstruct_fn(cfg_i)
                score = float(self.verify_fn(recon))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Search strategy %s failed: %s", name, exc)
                continue
            history.append({"strategy": name, "score": score})
            if score > best_score:
                best_recon, best_cfg, best_score, best_strategy = recon, cfg_i, score, name

        return {
            "best_recon": best_recon,
            "best_cfg": best_cfg,
            "best_score": (None if best_score == float("-inf") else best_score),
            "best_strategy": best_strategy,
            "history": history,
        }
