"""video/psss.py – Probability-based Semantic Similarity Score (PSSS).

Implements the PSSS metric from the LGVSC paper
(``reference/paper/LGVSC.../main.tex``, Sec. "Proposed PSSS for Keyframe
Selection", Eq. 1-2), which is the scoring primitive behind SKEM
(semantic-guided keyframe extraction module) — see ``video/skem_selector.py``.

PSSS compares two short text descriptions (``Info. A``/``Info. B`` — e.g. the
current frame's caption vs. the latest keyframe's caption) by asking a model
"Determine whether they are similar from the perspective of
<Semantic Focus>, use yes or no to answer." and reading off the model's
**next-token probability distribution** rather than its sampled/decoded
answer:

    S_abs = P("Yes" | Info A, Info B, Semantic Focus)              ∈ [0, 1)
    S_rel = P("No"  | ...) - P("Yes" | ...)                         ∈ (-1, 1)

``S_rel`` is what SKEM thresholds against ``eta_th`` (default 0.35 in the
paper): higher = more semantically divergent. This module deliberately keeps
three backends with very different evidentiary weight, and every
:class:`PsssScoreResult` is tagged with exactly which one produced it — never
silently interchangeable:

- :class:`MockPsssBackend` — deterministic, dependency-free, TEST-ONLY. Not a
  language-model probability of anything; a lexical-overlap heuristic dressed
  in the same output schema so the SKEM selector and its consumers can be
  unit-tested without any model weights.
- :class:`ClipTextProxyPsssBackend` — uses this repo's existing CLIP text
  encoder (``evaluators/clip_score.py``) to approximate semantic
  (dis)similarity via cosine distance between the two captions' text
  embeddings. A genuinely "already usable model" in this repo, but **not**
  the paper's mechanism (no yes/no token probability at all) — tagged
  ``backend_kind="proxy"``, never reported as real PSSS.
- :class:`MllmTokenProbPsssBackend` — the real mechanism: forwards the
  yes/no prompt through an actual causal-LM (or VLM text tower) and reads the
  softmax probability mass assigned to "Yes"/"No" continuations, handling the
  fact that a tokenizer may split "Yes"/"No" (case/space variants) into more
  than one token. Raises :class:`PsssBackendUnavailableError` — never falls
  back to a guess — when the dependency, weights, or the model/tokenizer
  object it was given cannot produce next-token logits.

Every :class:`PsssScoreResult` records raw per-variant logits, the normalised
probabilities, the final ``s_abs``/``s_rel`` score and enough evidence to
reconstruct the decision later (this is what ``skem_selector.py`` persists
into ``segments.json`` alongside the selector's threshold and reason string).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SEMANTIC_FOCUS = "the main subject and action"

DEFAULT_YES_VARIANTS: Tuple[str, ...] = ("Yes", " Yes", "yes", " yes", "YES")
DEFAULT_NO_VARIANTS: Tuple[str, ...] = ("No", " No", "no", " no", "NO")

_PROMPT_TEMPLATE = (
    "{info_a}, {info_b}. Determine whether they are similar from the "
    "perspective of {semantic_focus}, use yes or no to answer."
)


class PsssBackendUnavailableError(RuntimeError):
    """Raised when a PSSS backend cannot produce a score — missing
    dependency/weights, a model/tokenizer that does not expose next-token
    logits, or (for the real backend) any failure of the underlying model
    call. Never silently swallowed into a fabricated score: callers
    (:class:`~sgdjscc_lab.video.skem_selector.PsssKeyframeSelector`) let this
    propagate — a SKEM run configured for a real MLLM must fail loudly, not
    quietly degrade to mock/proxy behaviour.
    """


@dataclass
class PsssScoreResult:
    """One PSSS evaluation of ``(info_a, info_b)`` under ``semantic_focus``.

    ``s_abs``/``s_rel`` are exactly Eq. 1 / Eq. 2 of the LGVSC paper for the
    ``backend_kind="real"`` case; for ``"mock"``/``"proxy"`` they are
    schema-compatible stand-ins computed by a different, explicitly
    documented mechanism (see the module docstring) — ``backend_kind`` and
    ``notes`` make this unambiguous downstream.
    """

    info_a: str
    info_b: str
    semantic_focus: str
    p_yes: float
    p_no: float
    p_yes_norm: float
    p_no_norm: float
    s_abs: float
    s_rel: float
    backend: str
    backend_kind: str
    model_id: Optional[str] = None
    proxy_of: Optional[str] = None
    clipped: bool = False
    raw_logits: Dict = field(default_factory=dict)
    evidence: Dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "info_a": self.info_a,
            "info_b": self.info_b,
            "semantic_focus": self.semantic_focus,
            "p_yes": self.p_yes,
            "p_no": self.p_no,
            "p_yes_norm": self.p_yes_norm,
            "p_no_norm": self.p_no_norm,
            "s_abs": self.s_abs,
            "s_rel": self.s_rel,
            "backend": self.backend,
            "backend_kind": self.backend_kind,
            "model_id": self.model_id,
            "proxy_of": self.proxy_of,
            "clipped": self.clipped,
            "raw_logits": dict(self.raw_logits),
            "evidence": dict(self.evidence),
            "notes": self.notes,
        }


def _clip01(x: float) -> Tuple[float, bool]:
    if x < 0.0:
        return 0.0, True
    if x >= 1.0:
        return 0.999999, True
    return float(x), False


def _clip_rel(x: float) -> Tuple[float, bool]:
    if x <= -1.0:
        return -0.999999, True
    if x >= 1.0:
        return 0.999999, True
    return float(x), False


class PsssBackend:
    """Common interface every PSSS scoring backend implements."""

    backend_kind = "base"
    model_id: Optional[str] = None

    def score(
        self,
        info_a: str,
        info_b: str,
        semantic_focus: str = DEFAULT_SEMANTIC_FOCUS,
    ) -> PsssScoreResult:
        raise NotImplementedError


class MockPsssBackend(PsssBackend):
    """Deterministic, dependency-free backend for tests/dry runs.

    NOT a language-model probability of anything — ``s_rel`` is derived from
    plain lexical (whitespace-token) Jaccard overlap between ``info_a`` and
    ``info_b``: high overlap ⇒ treated as "similar" (low ``s_rel``), low
    overlap ⇒ "divergent" (high ``s_rel``). This exists purely so the SKEM
    selector's control flow (threshold comparison, min/max segment length,
    variable-length GOP construction) is unit-testable without any model
    weights and with fully reproducible scores for fixed captions.
    """

    backend_kind = "mock"
    backend_name = "mock"

    def score(self, info_a, info_b, semantic_focus=DEFAULT_SEMANTIC_FOCUS) -> PsssScoreResult:
        toks_a = {t.lower() for t in str(info_a or "").split()}
        toks_b = {t.lower() for t in str(info_b or "").split()}
        union = toks_a | toks_b
        inter = toks_a & toks_b
        overlap = (len(inter) / len(union)) if union else 1.0

        p_yes = float(overlap)
        p_no = float(1.0 - overlap)
        s_abs = p_yes
        s_rel = p_no - p_yes

        return PsssScoreResult(
            info_a=str(info_a or ""), info_b=str(info_b or ""), semantic_focus=semantic_focus,
            p_yes=p_yes, p_no=p_no, p_yes_norm=p_yes, p_no_norm=p_no,
            s_abs=s_abs, s_rel=s_rel,
            backend=self.backend_name, backend_kind=self.backend_kind, model_id=None,
            raw_logits={}, evidence={
                "method": "jaccard_token_overlap", "overlap": overlap,
                "tokens_a": sorted(toks_a), "tokens_b": sorted(toks_b),
            },
            notes="MOCK backend: deterministic lexical-overlap heuristic, NOT a real "
                  "PSSS/MLLM token-probability score. Test/dry-run use only.",
        )


class ClipTextProxyPsssBackend(PsssBackend):
    """Proxy backend: approximates PSSS using this repo's existing CLIP text
    encoder (``evaluators/clip_score.py::CLIPScoreEvaluator``) instead of a
    real yes/no token-probability MLLM.

    This is an "already usable model in this repo" stand-in, not the paper's
    mechanism — CLIP never sees a yes/no prompt or produces a token
    probability; it only computes a text-text cosine similarity, which is
    then monotonically mapped onto the same ``[0,1)``/``(-1,1)`` schema so it
    plugs into :class:`~sgdjscc_lab.video.skem_selector.PsssKeyframeSelector`
    unmodified. ``backend_kind="proxy"`` and ``proxy_of="clip_text_similarity"``
    make this unambiguous in every serialised result — never report a proxy
    score as real PSSS.

    Raises :class:`PsssBackendUnavailableError` if CLIP itself is
    unavailable (missing weights/dependency) — never silently falls back to
    :class:`MockPsssBackend`.
    """

    backend_kind = "proxy"
    backend_name = "clip_text_proxy"
    proxy_of = "clip_text_similarity"

    def __init__(self, clip_evaluator=None, device=None, model_name: str = "ViT-B/32") -> None:
        self._clip = clip_evaluator
        self._device = device
        self.model_name = model_name
        self.model_id = f"clip:{model_name}"

    def _get_clip(self):
        if self._clip is not None:
            return self._clip
        try:
            import torch

            from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
            device = self._device or torch.device("cpu")
            self._clip = CLIPScoreEvaluator(model_name=self.model_name, device=device)
        except Exception as exc:  # noqa: BLE001
            raise PsssBackendUnavailableError(
                f"ClipTextProxyPsssBackend could not load a CLIP model "
                f"({self.model_name!r}): {exc}. Install/verify the CLIP dependency, "
                "or use keyframe.psss.backend: mock instead."
            ) from exc
        return self._clip

    def score(self, info_a, info_b, semantic_focus=DEFAULT_SEMANTIC_FOCUS) -> PsssScoreResult:
        clip_eval = self._get_clip()
        try:
            feats = clip_eval._encode_texts([str(info_a or ""), str(info_b or "")])
            sim = float((feats[0] @ feats[1]).item())
        except PsssBackendUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PsssBackendUnavailableError(
                f"ClipTextProxyPsssBackend text encoding failed: {exc}"
            ) from exc

        sim_clamped = max(-1.0, min(1.0, sim))
        p_yes = (sim_clamped + 1.0) / 2.0
        p_no = 1.0 - p_yes
        s_abs = p_yes
        s_rel = p_no - p_yes  # == -sim_clamped

        return PsssScoreResult(
            info_a=str(info_a or ""), info_b=str(info_b or ""), semantic_focus=semantic_focus,
            p_yes=p_yes, p_no=p_no, p_yes_norm=p_yes, p_no_norm=p_no,
            s_abs=s_abs, s_rel=s_rel,
            backend=self.backend_name, backend_kind=self.backend_kind,
            model_id=self.model_id, proxy_of=self.proxy_of,
            raw_logits={}, evidence={
                "method": "clip_text_text_cosine_similarity", "cosine_similarity": sim,
            },
            notes="PROXY backend: CLIP text-embedding cosine similarity mapped onto the "
                  "PSSS score schema. This is NOT the paper's yes/no token-probability "
                  "PSSS — treat as an approximation only, never cite as faithful PSSS.",
        )


class MllmTokenProbPsssBackend(PsssBackend):
    """Real PSSS backend: reads P("Yes")/P("No") off an actual causal-LM's
    next-token distribution for the paper's yes/no prompt (Eq. 1-2).

    Supports dependency injection (``model=``/``tokenizer=``) so tests can
    exercise the exact scoring algorithm — including multi-token "Yes"/"No"
    handling — against small fake objects, with no real weights downloaded.
    When ``model``/``tokenizer`` are not supplied, ``model_id`` is loaded
    lazily via ``transformers.AutoModelForCausalLM``/``AutoTokenizer`` on the
    first :meth:`score` call.

    Multi-token handling
    ---------------------
    A tokenizer may split a surface form like ``" yes"`` into more than one
    token. For each configured variant string, this backend computes the
    TRUE joint sequence probability via teacher-forced autoregressive
    decoding (one forward pass per token in the variant, feeding the
    previous variant token back in) rather than assuming a single-token
    answer — so a 2-3 token "Yes"/"No" spelling is scored correctly, not
    silently truncated to its first token. ``P(Yes)``/``P(No)`` are the sum
    over each side's configured surface variants (deduplicated by token-id
    sequence, so two variants that happen to tokenize identically are not
    double-counted).

    Unavailability
    ---------------
    Raises :class:`PsssBackendUnavailableError` (never falls back to a mock
    score) when: ``transformers`` is not importable, ``model_id`` weights
    cannot be loaded, or the model/tokenizer object (real or injected) does
    not expose next-token logits (e.g. raises, or its output has neither a
    ``.logits`` attribute nor is itself a tensor).
    """

    backend_kind = "real"
    backend_name = "mllm_token_prob"

    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_id: Optional[str] = None,
        device: str = "cpu",
        dtype: str = "fp32",
        yes_variants: Sequence[str] = DEFAULT_YES_VARIANTS,
        no_variants: Sequence[str] = DEFAULT_NO_VARIANTS,
        prompt_template: str = _PROMPT_TEMPLATE,
    ) -> None:
        if model is None and model_id is None:
            raise ValueError(
                "MllmTokenProbPsssBackend requires either an injected `model` "
                "(+ `tokenizer`) or a `model_id` to lazily load from transformers."
            )
        self._model = model
        self._tokenizer = tokenizer
        self.model_id = model_id
        self.device = str(device)
        self.dtype = str(dtype)
        self.yes_variants = tuple(yes_variants)
        self.no_variants = tuple(no_variants)
        self.prompt_template = prompt_template
        self._failed: Optional[str] = None

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if self._failed is not None:
            raise PsssBackendUnavailableError(self._failed)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            self._failed = (
                "MllmTokenProbPsssBackend requires `transformers` (AutoModelForCausalLM/"
                f"AutoTokenizer), which is unavailable here ({exc}). Install it, or use "
                "keyframe.psss.backend: mock / proxy instead."
            )
            raise PsssBackendUnavailableError(self._failed) from exc
        try:
            logger.info("Loading PSSS MLLM backend: %s", self.model_id)
            dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
            torch_dtype = dtype_map.get(self.dtype, torch.float32)
            self._tokenizer = self._tokenizer or AutoTokenizer.from_pretrained(self.model_id)
            self._model = self._model or AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch_dtype,
            ).to(self.device).eval()
        except Exception as exc:  # noqa: BLE001
            self._failed = f"PSSS MLLM weights {self.model_id!r} could not be loaded ({exc})."
            raise PsssBackendUnavailableError(self._failed) from exc

    def _encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Tokenize *text*.

        ``add_special_tokens=False`` is required for the "Yes"/"No"
        CONTINUATION variants (:meth:`_variant_mass`) — many real tokenizers
        otherwise silently wrap a standalone string with BOS/EOS, so a naive
        ``tokenizer.encode("Yes")`` would compute ``P(BOS, "Yes", EOS |
        prompt)`` instead of ``P("Yes" | prompt)``. The PROMPT itself is
        still encoded with the tokenizer's default (``True``) so a real
        causal-LM sees a properly BOS-prefixed prompt.

        Falls back to a plain ``encode(text)`` call (no kwarg) for minimal
        tokenizer stand-ins that don't accept ``add_special_tokens`` at
        all — such tokenizers never add special tokens in the first place,
        so the distinction is moot for them.
        """
        try:
            try:
                ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
            except TypeError:
                ids = self._tokenizer.encode(text)
        except Exception as exc:  # noqa: BLE001
            raise PsssBackendUnavailableError(
                f"PSSS MLLM tokenizer failed to encode {text!r}: {exc}"
            ) from exc
        return list(ids)

    def _model_device(self):
        """Best-effort actual device of the model's input embedding layer —
        NOT ``self.device`` (the constructor arg), which may be stale for a
        model loaded with ``device_map``/``accelerate`` offloading, or simply
        wrong for a caller-injected model that was moved after construction.
        Falls back to a generic parameter, then to ``self.device``, so a
        minimal test fake (no ``.parameters()`` at all) still works."""
        import torch

        model = self._model
        try:
            emb = model.get_input_embeddings()
            if emb is not None:
                return next(emb.parameters()).device
        except Exception:  # noqa: BLE001
            pass
        try:
            return next(model.parameters()).device
        except Exception:  # noqa: BLE001
            pass
        return torch.device(self.device)

    def _forward_logits(self, ids: List[int]):
        import torch

        try:
            device = self._model_device()
            input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
            out = self._model(input_tensor)
        except PsssBackendUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PsssBackendUnavailableError(
                f"PSSS MLLM model call failed (does it accept a [1, T] LongTensor of "
                f"token ids and return next-token logits?): {exc}"
            ) from exc
        logits = getattr(out, "logits", out)
        try:
            last = logits[0, -1, :]
        except Exception as exc:  # noqa: BLE001
            raise PsssBackendUnavailableError(
                "PSSS MLLM model output does not expose per-token logits of shape "
                f"[batch, seq, vocab] (got {type(out).__name__}): {exc}"
            ) from exc
        return last

    def _variant_logprob(self, prompt_ids: List[int], variant_ids: List[int]) -> float:
        """Teacher-forced joint log-probability of *variant_ids* immediately
        following *prompt_ids* (handles multi-token surface forms)."""
        import torch

        ids = list(prompt_ids)
        total_logprob = 0.0
        for tok in variant_ids:
            last_logits = self._forward_logits(ids)
            logprobs = torch.log_softmax(last_logits.float(), dim=-1)
            total_logprob += float(logprobs[int(tok)].item())
            ids.append(int(tok))
        return total_logprob

    def _variant_mass(self, prompt_ids: List[int], variants: Sequence[str]) -> Tuple[float, Dict]:
        """Sum P(variant) over *variants*, de-duplicated by token-id sequence
        (two surface forms that tokenize identically are not double-counted)."""
        seen: Dict[Tuple[int, ...], str] = {}
        per_variant: Dict[str, Dict] = {}
        total = 0.0
        first_token_logits: Dict[str, float] = {}
        for v in variants:
            # add_special_tokens=False: this is a CONTINUATION of prompt_ids,
            # not a standalone sequence — must not be silently wrapped in its
            # own BOS/EOS (see _encode's docstring).
            variant_ids = tuple(self._encode(v, add_special_tokens=False))
            if not variant_ids:
                continue
            if variant_ids in seen:
                per_variant[v] = {"token_ids": list(variant_ids), "duplicate_of": seen[variant_ids]}
                continue
            seen[variant_ids] = v
            logprob = self._variant_logprob(prompt_ids, list(variant_ids))
            prob = math.exp(logprob)
            total += prob
            per_variant[v] = {"token_ids": list(variant_ids), "logprob": logprob, "prob": prob}
            first_logits = self._forward_logits(prompt_ids)
            first_token_logits[v] = float(first_logits[int(variant_ids[0])].item())
        return total, {"per_variant": per_variant, "first_token_logits": first_token_logits}

    def score(self, info_a, info_b, semantic_focus=DEFAULT_SEMANTIC_FOCUS) -> PsssScoreResult:
        self._load()
        prompt = self.prompt_template.format(
            info_a=info_a, info_b=info_b, semantic_focus=semantic_focus,
        )
        prompt_ids = self._encode(prompt)

        p_yes_raw, yes_evidence = self._variant_mass(prompt_ids, self.yes_variants)
        p_no_raw, no_evidence = self._variant_mass(prompt_ids, self.no_variants)

        denom = p_yes_raw + p_no_raw
        p_yes_norm = (p_yes_raw / denom) if denom > 0 else 0.5
        p_no_norm = (p_no_raw / denom) if denom > 0 else 0.5

        s_abs, abs_clipped = _clip01(p_yes_raw)
        s_rel_raw = p_no_raw - p_yes_raw
        s_rel, rel_clipped = _clip_rel(s_rel_raw)

        return PsssScoreResult(
            info_a=str(info_a), info_b=str(info_b), semantic_focus=semantic_focus,
            p_yes=p_yes_raw, p_no=p_no_raw, p_yes_norm=p_yes_norm, p_no_norm=p_no_norm,
            s_abs=s_abs, s_rel=s_rel,
            backend=self.backend_name, backend_kind=self.backend_kind, model_id=self.model_id,
            clipped=bool(abs_clipped or rel_clipped),
            raw_logits={
                "yes": yes_evidence["first_token_logits"],
                "no": no_evidence["first_token_logits"],
            },
            evidence={
                "prompt": prompt, "num_prompt_tokens": len(prompt_ids),
                "yes_variants": yes_evidence["per_variant"],
                "no_variants": no_evidence["per_variant"],
                "p_yes_raw": p_yes_raw, "p_no_raw": p_no_raw,
                "s_rel_raw": s_rel_raw,
            },
            notes="REAL PSSS: P(Yes)/P(No) from the model's next-token distribution "
                  "over the paper's yes/no prompt (Eq. 1-2), summed over configured "
                  "multi-token-safe surface variants.",
        )


_BACKENDS = {
    "mock": MockPsssBackend,
    "proxy": ClipTextProxyPsssBackend,
    "real": MllmTokenProbPsssBackend,
}


def build_psss_backend(name: str, cfg=None, **kwargs) -> PsssBackend:
    """Build one named PSSS backend.

    Parameters
    ----------
    name:
        ``"mock" | "proxy" | "real"``.
    cfg:
        Optional OmegaConf-like mapping providing backend-specific kwargs
        under ``psss.<name>.<key>`` (mirrors
        ``evaluators/presence_backends.py::build_presence_backend``).
        ``kwargs`` override ``cfg`` values.
    """
    name = str(name).lower()
    if name not in _BACKENDS:
        raise NotImplementedError(
            f"Unknown PSSS backend {name!r}; expected one of {sorted(_BACKENDS)}."
        )

    def _get(key, default=None):
        if key in kwargs:
            return kwargs[key]
        if cfg is not None:
            from omegaconf import OmegaConf
            val = OmegaConf.select(cfg, f"{name}.{key}", default=None)
            if val is not None:
                return val
        return default

    if name == "mock":
        return MockPsssBackend()
    if name == "proxy":
        return ClipTextProxyPsssBackend(
            clip_evaluator=_get("clip_evaluator"),
            model_name=str(_get("model_name", "ViT-B/32")),
        )
    if name == "real":
        model_id = _get("model_id")
        if not model_id and _get("model") is None:
            raise ValueError(
                "psss backend 'real' requires keyframe.psss.real.model_id (an HF "
                "causal-LM/VLM id) — see docs for a real-MLLM PSSS config example."
            )
        return MllmTokenProbPsssBackend(
            model=_get("model"), tokenizer=_get("tokenizer"), model_id=model_id,
            device=str(_get("device", "cpu")), dtype=str(_get("dtype", "fp32")),
            yes_variants=tuple(_get("yes_variants", DEFAULT_YES_VARIANTS)),
            no_variants=tuple(_get("no_variants", DEFAULT_NO_VARIANTS)),
        )

    raise NotImplementedError(f"PSSS backend {name!r} is registered but not built.")  # pragma: no cover
