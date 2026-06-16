"""evaluators/fid.py – Fréchet Inception Distance (paper §VI metric).

The SGD-JSCC paper reports FID to assess *visual quality* via the statistical
similarity between the **set** of original images and the **set** of reconstructed
images (Sec. VI "Performance Metrics"). FID is therefore a dataset-level metric,
not a per-image one: it fits a Gaussian to Inception features of each set and
measures the Fréchet distance between them:

    FID = ||μ_r − μ_g||²  +  Tr( Σ_r + Σ_g − 2 (Σ_r Σ_g)^{1/2} )

This evaluator accumulates features incrementally over a dataset (one SNR
condition) and produces a single scalar at the end.

Design
------
* ``feature_fn`` is injectable: a callable mapping an image batch ``[N,3,H,W]`` in
  ``[0,1]`` to features ``[N,D]``. Tests inject a cheap deterministic extractor so
  the Fréchet math is exercised on CPU with no network/weights.
* When ``feature_fn`` is None, a torchvision Inception-v3 (pool3, 2048-D) backend
  is built lazily. If torchvision / its weights are unavailable (e.g. offline),
  the evaluator degrades **gracefully**: it stays "unavailable" and ``compute()``
  returns ``None`` (logged once) instead of raising — so a missing dependency
  never breaks an evaluation run.

paper-faithful / scaffold
-------------------------
With the real Inception backend this is the standard FID used by the paper. The
injected-``feature_fn`` path computes a *Fréchet distance in that feature space*
(a proxy), useful for testing but NOT the paper's Inception-FID — callers must
not report the proxy as FID.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fréchet distance between two Gaussians (numpy)
# ─────────────────────────────────────────────────────────────────────────────

def _frechet_distance(feat_r, feat_g) -> float:
    """Fréchet distance between Gaussians fit to feature matrices ``[N, D]``."""
    import numpy as np

    r = feat_r.astype(np.float64)
    g = feat_g.astype(np.float64)
    mu_r, mu_g = r.mean(axis=0), g.mean(axis=0)
    # rowvar=False → covariance over the D feature dims; need >=2 samples.
    sigma_r = np.cov(r, rowvar=False)
    sigma_g = np.cov(g, rowvar=False)
    sigma_r = np.atleast_2d(sigma_r)
    sigma_g = np.atleast_2d(sigma_g)

    diff = mu_r - mu_g
    mean_term = float(diff @ diff)

    covmean = _matrix_sqrt(sigma_r @ sigma_g)
    tr_term = float(np.trace(sigma_r) + np.trace(sigma_g) - 2.0 * np.trace(covmean))
    return mean_term + tr_term


def _matrix_sqrt(mat):
    """Symmetric PSD-ish matrix square root: scipy.sqrtm, eigen fallback."""
    import numpy as np

    try:  # preferred: scipy (handles non-symmetric product correctly)
        from scipy.linalg import sqrtm
        s = sqrtm(mat)
        if np.iscomplexobj(s):
            s = s.real
        return s
    except Exception:  # pragma: no cover - exercised only without scipy
        # Eigendecomposition of the symmetrized matrix (good enough for a fallback).
        sym = (mat + mat.T) / 2.0
        w, v = np.linalg.eigh(sym)
        w = np.clip(w, 0.0, None)
        return (v * np.sqrt(w)) @ v.T


# ─────────────────────────────────────────────────────────────────────────────
# Inception feature backend (lazy, optional)
# ─────────────────────────────────────────────────────────────────────────────

def _build_inception_feature_fn(device) -> Optional[Callable]:
    """Return a callable ``[N,3,H,W]∈[0,1] → [N,2048]`` Inception pool3 features.

    Returns None if torchvision (or its pretrained weights) is unavailable, so
    the caller can degrade gracefully.
    """
    try:
        import torch.nn.functional as F
        from torchvision.models import inception_v3, Inception_V3_Weights

        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                             aux_logits=True)
        model.fc = torch.nn.Identity()   # → 2048-D pool features
        model.eval().to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        @torch.no_grad()
        def _feat(x: torch.Tensor) -> torch.Tensor:
            x = x.to(device).clamp(0, 1)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            x = (x - mean) / std
            return model(x).detach().cpu()

        logger.info("FID: using torchvision Inception-v3 (pool3, 2048-D) backend.")
        return _feat
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("FID: Inception backend unavailable (%s) — FID will be "
                       "reported as None. Install torchvision or inject a "
                       "feature_fn.", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class FIDEvaluator:
    """Dataset-level FID accumulator.

    Usage::

        fid = FIDEvaluator(device="cpu")
        for real, fake in pairs:        # each [N,3,H,W] in [0,1]
            fid.add(real, fake)
        value = fid.compute()           # float or None (graceful)
    """

    def __init__(
        self,
        feature_fn: Optional[Callable] = None,
        device=None,
        min_samples: int = 2,
    ) -> None:
        self.device = device if device is not None else torch.device("cpu")
        self._injected = feature_fn is not None
        self._feature_fn = feature_fn
        self._built = feature_fn is not None
        self.min_samples = int(min_samples)
        self._real: List[torch.Tensor] = []
        self._fake: List[torch.Tensor] = []
        self._warned = False

    # ── feature extraction ────────────────────────────────────────────────────
    def _features(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._built:
            self._feature_fn = _build_inception_feature_fn(self.device)
            self._built = True
        if self._feature_fn is None:
            return None
        feats = self._feature_fn(x)
        return feats.detach().to("cpu").reshape(feats.shape[0], -1)

    @property
    def available(self) -> bool:
        """True if a feature backend is (or can still be) available."""
        return self._feature_fn is not None or not self._built

    @property
    def is_proxy(self) -> bool:
        """True when using an injected (non-Inception) feature_fn → proxy, not FID."""
        return self._injected

    def ensure_backend(self) -> bool:
        """Force the (lazy) feature backend to build now and return True iff a
        **real Inception** backend is available.

        Returns False for an injected proxy feature_fn or when torchvision/its
        weights are unavailable. Used by ``--require-real-fid`` to fail fast
        BEFORE an expensive evaluation rather than discovering a proxy/None FID
        at the end.
        """
        if not self._built:
            self._feature_fn = _build_inception_feature_fn(self.device)
            self._built = True
        return self.backend_name == "inception"

    @property
    def backend_name(self) -> str:
        """Identify the feature backend so results files can distinguish a true
        Inception-FID from an injected proxy or an unavailable backend.

        ``"proxy"`` (injected feature_fn — NOT Inception-FID) | ``"inception"``
        (real torchvision backend) | ``"unavailable"`` (no backend → FID is None)
        | ``"pending"`` (backend not built yet — no features extracted).
        """
        if self._injected:
            return "proxy"
        if not self._built:
            return "pending"
        return "inception" if self._feature_fn is not None else "unavailable"

    # ── accumulation ──────────────────────────────────────────────────────────
    def add(self, real: torch.Tensor, fake: torch.Tensor) -> None:
        try:
            rf = self._features(real)
            ff = self._features(fake)
        except Exception as exc:  # pragma: no cover - backend runtime error
            if not self._warned:
                logger.warning("FID: feature extraction failed (%s) — skipping.", exc)
                self._warned = True
            return
        if rf is None or ff is None:
            return
        self._real.append(rf)
        self._fake.append(ff)

    def n_samples(self) -> int:
        return sum(t.shape[0] for t in self._real)

    # ── final score ───────────────────────────────────────────────────────────
    def compute(self) -> Optional[float]:
        """Return the scalar FID, or None when unavailable / too few samples."""
        if not self._real or not self._fake:
            return None
        n = self.n_samples()
        if n < self.min_samples:
            logger.warning("FID: only %d samples (<%d) — returning None.",
                           n, self.min_samples)
            return None
        try:
            real = torch.cat(self._real, dim=0).numpy()
            fake = torch.cat(self._fake, dim=0).numpy()
            return float(_frechet_distance(real, fake))
        except Exception as exc:  # pragma: no cover
            logger.warning("FID: computation failed (%s) — returning None.", exc)
            return None

    def reset(self) -> None:
        self._real.clear()
        self._fake.clear()
