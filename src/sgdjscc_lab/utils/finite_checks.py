"""utils/finite_checks.py – Stage-labeled finite-value guards.

A non-finite (NaN/Inf) value produced deep inside a forward pass (VAE encode,
channel transmit, mask/power-scalar, step matching, diffusion decode) used to
surface only much later — as a silently-NaN final reconstruction, or not at
all until a metric computed a "nan" mean. That made the *stage* that actually
went non-finite unrecoverable after the fact (see the digital-packet blind
SNR estimation bug fixed in ``pipelines/infer_pipeline.py::_compute_step``,
which this module's checks would have pinpointed immediately at the
"step_match" stage instead of only at the final decoded image).

:func:`assert_finite` raises :class:`NonFiniteError` — tagged with a
``stage`` name and free-form ``context`` (video/config/frame/etc, filled in
by the caller closest to that information) — the instant a non-finite value
appears, rather than letting it propagate through the rest of a (possibly
expensive, e.g. 50-step diffusion) forward pass first.

Callers that want to keep sweeping other configs/frames after one fails
(e.g. ``scripts/run_transmission_reduction_eval.py``, which must still let a
deliberately-lossy int4 config produce a documented, excluded failure rather
than aborting the whole sweep) should catch :class:`NonFiniteError`
specifically — never a bare ``except Exception`` — so a genuine bug still
propagates and stops the run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class NonFiniteError(RuntimeError):
    """A tensor contained NaN/Inf at a labeled pipeline stage.

    Attributes mirror the constructor arguments so a catching caller can
    build a structured log/CSV row without re-parsing the message string.
    """

    def __init__(
        self,
        stage: str,
        *,
        n_nan: int,
        n_inf: int,
        numel: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.stage = stage
        self.n_nan = int(n_nan)
        self.n_inf = int(n_inf)
        self.numel = int(numel)
        self.context = dict(context or {})
        ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        super().__init__(
            f"non-finite values at stage={stage!r}: {self.n_nan} NaN, {self.n_inf} Inf "
            f"out of {self.numel} elements" + (f" ({ctx_str})" if ctx_str else "")
        )


def assert_finite(
    tensor: torch.Tensor,
    stage: str,
    context: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Raise :class:`NonFiniteError` if *tensor* has any NaN/Inf, else return it unchanged.

    Returns the input tensor so this can be used inline:
    ``x = assert_finite(f(x), "some_stage")``.
    """
    if not torch.is_tensor(tensor):
        return tensor
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return tensor
    bad = ~finite
    n_nan = int(torch.isnan(tensor).sum().item())
    n_inf = int(torch.isinf(tensor).sum().item())
    raise NonFiniteError(
        stage, n_nan=n_nan, n_inf=n_inf, numel=int(bad.numel()), context=context,
    )
