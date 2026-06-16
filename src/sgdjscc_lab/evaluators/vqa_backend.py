"""evaluators/vqa_backend.py – Local VQA backend adapters (Phase 5-C).

Turns a config block into a callable ``vqa_fn(image, question) -> answer`` that
``VQAHallucinationEvaluator`` consumes.  All heavy models are **lazy-imported**
inside the adapter so importing this module is cheap and offline-safe; if a
backend's dependency or weights are unavailable, ``build_vqa_backend`` logs a
warning and returns ``None`` so the caller degrades to the CLIP heuristic.

Supported ``vqa_backend.type``:

- ``"mock"``  : a deterministic, dependency-free backend (for tests / dry runs).
                Answers from an optional ``rules`` map ``{keyword: "yes"/"no"}``;
                default answer is ``mock_answer`` (default ``"no"``).
- ``"blip2"`` : BLIP-2 VQA via ``transformers`` (same family as the Phase-1 caption
                model). Prompt ``"Question: {q} Answer:"``. Default model
                ``Salesforce/blip2-opt-2.7b-coco`` (the checkpoint the caption
                extractor already loads successfully here; the plain
                ``blip2-opt-2.7b`` failed to parse in this environment).
- ``"llava"`` : LLaVA via ``transformers`` ``LlavaForConditionalGeneration``
                (best-effort; returns None if the class / weights are absent).
- ``"mplug"`` : mPLUG-style VQA via ``transformers`` AutoModel (best-effort).

The contract is intentionally tiny — ``answer(image[1,3,H,W] in [0,1], str)->str`` —
so new backends are a single adapter class.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_pil(image: torch.Tensor):
    """Convert a ``[1,3,H,W]`` / ``[3,H,W]`` float tensor in [0,1] to a PIL image."""
    from PIL import Image
    t = image
    if t.dim() == 4:
        t = t[0]
    arr = (t.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


# ── mock backend (deterministic, no deps) ─────────────────────────────────────

class MockVQABackend:
    """Rule-based deterministic VQA backend for tests / dry runs."""

    def __init__(self, rules: Optional[Dict[str, str]] = None, mock_answer: str = "no") -> None:
        self.rules = {k.lower(): v for k, v in (rules or {}).items()}
        self.mock_answer = mock_answer

    def answer(self, image: torch.Tensor, question: str) -> str:
        q = str(question).lower()
        for kw, ans in self.rules.items():
            if kw in q:
                return ans
        return self.mock_answer


# ── BLIP-2 backend (transformers, lazy) ───────────────────────────────────────

class Blip2VQABackend:
    """BLIP-2 visual-question-answering backend (lazy ``transformers`` load).

    Defaults to ``Salesforce/blip2-opt-2.7b-coco`` — the same checkpoint the
    Phase-1 caption extractor already loads successfully in this environment.  The
    plain ``Salesforce/blip2-opt-2.7b`` checkpoint failed to load/parse here (a
    transformers-version-dependent ``ModelWrapper`` mismatch), so reusing the
    known-good coco variant avoids that.  BLIP-2-OPT answers VQA via the
    ``"Question: … Answer:"`` prompt regardless of the caption fine-tune, so the
    coco variant is functionally adequate for yes/no object questions.

    A load failure is cached (``_failed``) so ``answer`` does not re-attempt the
    heavy load on every question.
    """

    def __init__(self, model_id: str = "Salesforce/blip2-opt-2.7b-coco",
                 device: Optional[str] = None, max_new_tokens: int = 8) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._failed = False

    def _load(self):
        if self._model is not None:
            return
        if self._failed:
            raise RuntimeError(f"BLIP-2 VQA backend {self.model_id} previously failed to load")
        try:
            from transformers import AutoProcessor, Blip2ForConditionalGeneration
            logger.info("Loading BLIP-2 VQA backend: %s", self.model_id)
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            dtype = torch.float16 if "cuda" in str(self.device) else torch.float32
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=dtype
            ).to(self.device).eval()
        except Exception:
            self._failed = True   # cache failure → no repeated heavy reloads
            raise

    def answer(self, image: torch.Tensor, question: str) -> str:
        self._load()
        pil = _to_pil(image)
        prompt = f"Question: {question} Answer:"
        inputs = self._processor(images=pil, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        return text.strip()


# ── LLaVA / mPLUG best-effort backends ────────────────────────────────────────

class LlavaVQABackend:
    """LLaVA VQA backend (best-effort; lazy ``transformers``)."""

    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf",
                 device: Optional[str] = None, max_new_tokens: int = 16) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        logger.info("Loading LLaVA VQA backend: %s", self.model_id)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        dtype = torch.float16 if "cuda" in str(self.device) else torch.float32
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self.device).eval()

    def answer(self, image: torch.Tensor, question: str) -> str:
        self._load()
        pil = _to_pil(image)
        prompt = f"USER: <image>\n{question} ASSISTANT:"
        inputs = self._processor(images=pil, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        return text.split("ASSISTANT:")[-1].strip()


class MplugVQABackend:
    """mPLUG-style VQA backend (best-effort; lazy ``transformers`` AutoModel)."""

    def __init__(self, model_id: str = "MAGAer13/mplug-owl2-llama2-7b",
                 device: Optional[str] = None, max_new_tokens: int = 16) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForVision2Seq, AutoProcessor
        logger.info("Loading mPLUG VQA backend: %s", self.model_id)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id, trust_remote_code=True
        ).to(self.device).eval()

    def answer(self, image: torch.Tensor, question: str) -> str:
        self._load()
        pil = _to_pil(image)
        inputs = self._processor(images=pil, text=question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self._processor.batch_decode(out, skip_special_tokens=True)[0].strip()


_BACKENDS = {
    "mock": MockVQABackend,
    "blip2": Blip2VQABackend,
    "llava": LlavaVQABackend,
    "mplug": MplugVQABackend,
}


def build_vqa_backend(cfg=None, **kwargs) -> Optional[Callable]:
    """Build a ``vqa_fn(image, question) -> str`` from a ``vqa_backend`` config.

    Parameters (read from *cfg* mapping or *kwargs*; kwargs win):
        ``type``     : "mock" | "blip2" | "llava" | "mplug" | "none"
        ``model_id`` : HF model id (backend-specific default otherwise)
        ``device``   : compute device string
        ``rules`` / ``mock_answer`` : mock-backend behaviour

    Returns the callable, or ``None`` when the type is none/unknown or the backend
    cannot be loaded (so the caller falls back to the CLIP heuristic).
    """
    def _get(key, default=None):
        if key in kwargs:
            return kwargs[key]
        if cfg is not None:
            try:
                return cfg.get(key, default)
            except AttributeError:
                return getattr(cfg, key, default)
        return default

    btype = str(_get("type", "none")).lower()
    if btype in ("none", "", "clip", "clip_fallback"):
        return None
    if btype not in _BACKENDS:
        logger.warning("Unknown vqa_backend.type=%r; using CLIP fallback.", btype)
        return None

    # Dependency availability check (weights still load lazily on first answer()).
    if btype != "mock":
        try:
            import transformers  # noqa: F401
        except Exception:  # noqa: BLE001
            logger.warning(
                "vqa_backend.type=%r needs `transformers`, which is unavailable; "
                "using CLIP fallback.", btype)
            return None

    try:
        if btype == "mock":
            backend = MockVQABackend(rules=_get("rules"), mock_answer=str(_get("mock_answer", "no")))
        else:
            kw = {"device": _get("device")}
            model_id = _get("model_id")
            if model_id:
                kw["model_id"] = model_id
            backend = _BACKENDS[btype](**kw)
    except Exception as exc:  # noqa: BLE001 – construction failed
        logger.warning("VQA backend %r unavailable (%s); using CLIP fallback.", btype, exc)
        return None

    return backend.answer
