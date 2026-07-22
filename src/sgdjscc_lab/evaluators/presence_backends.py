"""evaluators/presence_backends.py – Object-presence backend interface (ETRI 5차, step 8).

Scope note (read before touching this file)
--------------------------------------------
1~4차's presence judgments (``evaluators/object_preservation.py``,
``evaluators/hallucination.py``, and by extension the packet-derived
``objects`` lists that ``PacketVerifier`` compares) all ultimately rest on one
CLIP global text-image probe. This module's job is **structural**: define one
common interface — "is `object_name` present in this image?" → ``(present,
confidence, evidence)`` — behind which the existing CLIP judge, and future
grounded-detector (OWLv2) / VQA / ground-truth backends, can sit
interchangeably.

This is **not** "OWLv2/VQA model performance verified" work. The OWLv2 backend
here is a real (lazy-loaded) integration point, but it is never required to be
installed: if ``transformers``'s OWLv2 classes or weights are unavailable, it
raises :class:`PresenceBackendUnavailableError` with a clear message rather
than silently degrading to a wrong answer. Tests in this repo exercise
:class:`MockPresenceBackend` only — no real detector/VQA weights are a test
dependency. The VQA backend reuses the existing
``evaluators/vqa_backend.py``/``hallucination_vqa.py`` infrastructure (already
built in an earlier phase) rather than re-implementing VQA calling code.

Common output shape
--------------------
Every backend's ``check()`` returns a :class:`PresenceResult` with:
    ``object_name``  – the object queried.
    ``present``       – bool.
    ``confidence``    – float in ``[0, 1]``.
    ``backend``       – ``"clip" | "owlv2" | "vqa" | "gt" | "mock"``.
    ``evidence``      – backend-specific dict (raw scores, questions/answers, …).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class PresenceBackendUnavailableError(RuntimeError):
    """Raised when a backend cannot answer — missing dependency/weights, no
    image supplied where one is required, or no GT annotation for the object.

    Never silently swallowed into a wrong ``present`` value: callers
    (:class:`~sgdjscc_lab.evaluators.presence_calibration.PresenceCalibrator`)
    catch this to skip the backend (ensemble modes) or let it propagate
    (single-backend modes, where an unavailable backend should fail loudly).
    """


@dataclass
class PresenceResult:
    """One backend's presence judgment for one object (JSON-serialisable)."""

    object_name: str
    present: bool
    confidence: float
    backend: str
    evidence: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "object_name": self.object_name,
            "present": bool(self.present),
            "confidence": float(self.confidence),
            "backend": self.backend,
            "evidence": dict(self.evidence),
        }


class PresenceBackend:
    """Common interface every presence backend implements."""

    backend_name = "base"

    def check(
        self,
        object_name: str,
        image: Optional[torch.Tensor] = None,
        packet: Optional[Dict] = None,
        gt_metadata: Optional[Dict] = None,
    ) -> PresenceResult:
        """Decide whether *object_name* is present.

        Every backend receives the full call signature (even ones that ignore
        most of it) so :class:`~sgdjscc_lab.evaluators.presence_calibration.PresenceCalibrator`
        can call any configured backend uniformly without knowing which
        fields it actually needs — this is what makes ``image=None``
        (held-out remeasurement from saved packets, no pixels available) a
        normal, supported call: image-based backends (clip/owlv2/vqa) raise
        :class:`PresenceBackendUnavailableError`, while image-free backends
        (mock/gt) answer normally.

        Parameters
        ----------
        object_name:
            The object category to check.
        image:
            Optional ``[1, 3, H, W]`` (or ``[3, H, W]``) tensor in ``[0, 1]``
            — the image to inspect. Required by image-based backends
            (clip/owlv2/vqa); ``mock``/``gt`` do not need it and must not
            raise merely because it is ``None``.
        packet:
            Optional semantic packet dict (used by ``mock`` as a
            dependency-free stand-in for a real detector).
        gt_metadata:
            Optional per-call GT annotation override (used by ``gt``; see
            :class:`GtPresenceBackend`). Lets a caller (e.g.
            ``pipelines/heldout_remeasurement.py``) supply per-item ground
            truth without rebuilding the backend per item.
        """
        raise NotImplementedError


class MockPresenceBackend(PresenceBackend):
    """Deterministic, dependency-free backend for tests/dry runs.

    Decides presence from the PACKET's ``objects`` list (no image needed) —
    a stand-in for "some detector said yes/no" that never requires a model.
    """

    backend_name = "mock"

    def __init__(self, present_confidence: float = 0.9, absent_confidence: float = 0.1) -> None:
        self.present_confidence = float(present_confidence)
        self.absent_confidence = float(absent_confidence)

    def check(self, object_name, image=None, packet=None, gt_metadata=None) -> PresenceResult:
        objs = {str(o).lower() for o in (packet or {}).get("objects", []) or []}
        present = str(object_name).lower() in objs
        conf = self.present_confidence if present else self.absent_confidence
        return PresenceResult(
            object_name=object_name, present=present, confidence=conf,
            backend=self.backend_name, evidence={"source": "packet_objects", "packet_objects": sorted(objs)},
        )


class ClipPresenceBackend(PresenceBackend):
    """Wraps the existing CLIP text-image probe (the 1~4차 presence judge —
    see ``evaluators/object_preservation.py``) behind the common interface.

    Not a re-implementation: uses the same ``CLIPScoreEvaluator`` encode calls
    the rest of the codebase already uses.
    """

    backend_name = "clip"

    def __init__(self, clip_evaluator=None, threshold: float = 0.25, device: Optional[torch.device] = None) -> None:
        self._clip = clip_evaluator
        self.threshold = float(threshold)
        self._device = device or torch.device("cpu")

    def _get_clip(self):
        if self._clip is None:
            from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
            self._clip = CLIPScoreEvaluator(device=self._device)
        return self._clip

    def check(self, object_name, image=None, packet=None, gt_metadata=None) -> PresenceResult:
        if image is None:
            raise PresenceBackendUnavailableError(
                "ClipPresenceBackend.check() needs an image tensor; none was provided."
            )
        clip_eval = self._get_clip()
        img_feat = clip_eval._encode_images(image)
        txt_feat = clip_eval._encode_texts([f"a photo of a {object_name}"])
        sim = float((img_feat @ txt_feat.T).reshape(-1)[0].item())
        confidence = float(max(0.0, min(1.0, sim)))
        present = sim >= self.threshold
        return PresenceResult(
            object_name=object_name, present=present, confidence=confidence,
            backend=self.backend_name, evidence={"clip_similarity": sim, "threshold": self.threshold},
        )


class Owlv2PresenceBackend(PresenceBackend):
    """Grounded-detector presence backend (zero-shot OWLv2 object detection).

    OPTIONAL — lazily imports ``transformers``'s OWLv2 classes on first
    ``check()`` call. If the dependency or weights are unavailable, raises
    :class:`PresenceBackendUnavailableError` with a clear message; it never
    silently falls back on its own (that policy decision belongs to
    :class:`~sgdjscc_lab.evaluators.presence_calibration.PresenceCalibrator`).
    Not exercised by this repo's tests — only :class:`MockPresenceBackend` is.
    """

    backend_name = "owlv2"

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        score_threshold: float = 0.1,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.score_threshold = float(score_threshold)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._failed = False

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._failed:
            raise PresenceBackendUnavailableError(
                f"OWLv2 backend {self.model_id!r} previously failed to load."
            )
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except Exception as exc:  # noqa: BLE001
            self._failed = True
            raise PresenceBackendUnavailableError(
                "OWLv2 backend requires a `transformers` version with OWLv2 support, "
                f"which is unavailable here ({exc}). Install/upgrade `transformers`, "
                "or use verifier.presence_mode: clip_only / mock instead."
            ) from exc
        try:
            logger.info("Loading OWLv2 presence backend: %s", self.model_id)
            self._processor = Owlv2Processor.from_pretrained(self.model_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_id).to(self.device).eval()
        except Exception as exc:  # noqa: BLE001 – weights unavailable / download failed
            self._failed = True
            raise PresenceBackendUnavailableError(
                f"OWLv2 weights {self.model_id!r} could not be loaded ({exc})."
            ) from exc

    def check(self, object_name, image=None, packet=None, gt_metadata=None) -> PresenceResult:
        if image is None:
            raise PresenceBackendUnavailableError("Owlv2PresenceBackend.check() needs an image tensor.")
        self._load()
        from sgdjscc_lab.evaluators.vqa_backend import _to_pil
        pil = _to_pil(image)
        inputs = self._processor(text=[[str(object_name)]], images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([pil.size[::-1]])
        result = self._processor.post_process_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=target_sizes
        )[0]
        scores = result.get("scores")
        best = float(scores.max().item()) if scores is not None and len(scores) else 0.0
        present = best >= self.score_threshold
        return PresenceResult(
            object_name=object_name, present=present, confidence=best,
            backend=self.backend_name,
            evidence={"n_boxes": int(len(scores)) if scores is not None else 0, "best_score": best},
        )


class VqaPresenceBackend(PresenceBackend):
    """Wraps ``evaluators/vqa_backend.py``'s ``vqa_fn`` behind the common
    interface — asks a yes/no presence question per object.

    OPTIONAL — raises :class:`PresenceBackendUnavailableError` if no usable
    ``vqa_fn`` was built (dependency missing, backend type "none"/"mock" not
    configured, etc.); it never silently answers "no" for an absent model.
    """

    backend_name = "vqa"

    def __init__(
        self,
        vqa_fn: Optional[Callable] = None,
        vqa_backend_cfg=None,
        question_template: str = "Is there a {obj} in the image?",
    ) -> None:
        if vqa_fn is None and vqa_backend_cfg is not None:
            from sgdjscc_lab.evaluators.vqa_backend import build_vqa_backend
            vqa_fn = build_vqa_backend(vqa_backend_cfg)
        self.vqa_fn = vqa_fn
        self.question_template = question_template

    def check(self, object_name, image=None, packet=None, gt_metadata=None) -> PresenceResult:
        if self.vqa_fn is None:
            raise PresenceBackendUnavailableError(
                "VqaPresenceBackend has no usable vqa_fn (backend unavailable or not configured)."
            )
        if image is None:
            raise PresenceBackendUnavailableError("VqaPresenceBackend.check() needs an image tensor.")
        question = self.question_template.format(obj=object_name)
        answer = self.vqa_fn(image, question)
        present = str(answer).strip().lower().startswith(("y", "true", "1"))
        return PresenceResult(
            object_name=object_name, present=present, confidence=(1.0 if present else 0.0),
            backend=self.backend_name, evidence={"question": question, "answer": str(answer)},
        )


class GtPresenceBackend(PresenceBackend):
    """Ground-truth backend: looks up presence from injected GT metadata.

    ``gt_metadata`` maps ``object_name -> bool`` or ``object_name ->
    {"present": bool, ...}``. Raises :class:`PresenceBackendUnavailableError`
    when no annotation exists for the requested object (never guesses).

    Parameters
    ----------
    gt_metadata:
        The DEFAULT annotation map, used when a call doesn't supply its own
        (see ``check(..., gt_metadata=...)``). Lets a caller either configure
        one fixed GT map up front, or share one backend instance across many
        items and pass each item's own GT map per call (e.g.
        ``pipelines/heldout_remeasurement.py`` forwards
        ``RemeasurementItem.gt_metadata`` this way).
    """

    backend_name = "gt"

    def __init__(self, gt_metadata: Optional[Dict] = None) -> None:
        self.gt_metadata = dict(gt_metadata or {})

    def check(self, object_name, image=None, packet=None, gt_metadata=None) -> PresenceResult:
        source = gt_metadata if gt_metadata is not None else self.gt_metadata
        entry = source.get(object_name, source.get(str(object_name).lower()))
        if entry is None:
            raise PresenceBackendUnavailableError(f"No GT annotation for object_name={object_name!r}.")
        present = bool(entry.get("present")) if isinstance(entry, dict) else bool(entry)
        return PresenceResult(
            object_name=object_name, present=present, confidence=1.0,
            backend=self.backend_name, evidence={"gt_entry": entry if isinstance(entry, dict) else {"present": entry}},
        )


_BACKENDS = {
    "mock": MockPresenceBackend,
    "clip": ClipPresenceBackend,
    "owlv2": Owlv2PresenceBackend,
    "vqa": VqaPresenceBackend,
    "gt": GtPresenceBackend,
}


def build_presence_backend(name: str, cfg=None, **kwargs) -> PresenceBackend:
    """Build one named presence backend.

    Parameters
    ----------
    name:
        ``"mock" | "clip" | "owlv2" | "vqa" | "gt"``.
    cfg:
        Optional OmegaConf-like mapping providing backend-specific kwargs
        (e.g. ``clip.threshold``, ``owlv2.model_id``, ``vqa.vqa_backend``,
        ``gt.metadata``). ``kwargs`` override ``cfg`` values.
    """
    name = str(name).lower()
    if name not in _BACKENDS:
        raise NotImplementedError(
            f"Unknown presence backend {name!r}; expected one of {sorted(_BACKENDS)}."
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
        return MockPresenceBackend(
            present_confidence=float(_get("present_confidence", 0.9)),
            absent_confidence=float(_get("absent_confidence", 0.1)),
        )
    if name == "clip":
        return ClipPresenceBackend(
            clip_evaluator=_get("clip_evaluator"),
            threshold=float(_get("threshold", 0.25)),
        )
    if name == "owlv2":
        return Owlv2PresenceBackend(
            model_id=str(_get("model_id", "google/owlv2-base-patch16-ensemble")),
            score_threshold=float(_get("score_threshold", 0.1)),
        )
    if name == "vqa":
        return VqaPresenceBackend(
            vqa_fn=_get("vqa_fn"),
            vqa_backend_cfg=_get("vqa_backend"),
        )
    if name == "gt":
        return GtPresenceBackend(gt_metadata=_get("metadata", {}))

    raise NotImplementedError(f"presence backend {name!r} is registered but not built.")  # pragma: no cover
