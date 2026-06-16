"""models/reliability_head.py – Receiver reliability estimator (Phase 5-A).

Predicts how reliable a channel observation is — i.e. how much the generative
decoder should trust the received latent vs. lean on the semantic prior.  This
plays the role of DiffCom's posterior confidence and feeds the channel-condition
policy / dynamic sampling-budget routing in Phase 5-B.

Outputs:
    reliability_map  ``[B, 1, H, W]`` per-token reliability in [0, 1]
    confidence       ``[B]`` scalar decode-confidence proxy in [0, 1]

The default is a **training-free** heuristic combining the estimated SNR, the
channel gain and the mask/reliability signal.  A learnable head over the encoder
scalars is provided behind the same interface for later training.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from sgdjscc_lab.channels.measurement import MeasurementBundle


class ReliabilityHead(nn.Module):
    """Estimate per-token reliability and a scalar decode confidence.

    Parameters
    ----------
    snr_ref:
        SNR (dB) mapped to ~0.5 confidence; higher SNR → higher confidence.
    snr_scale:
        Logistic steepness for the SNR→confidence mapping.
    learnable:
        If True, add an ``nn.Linear`` head over the 5 encoder scalars (the
        learned interface). Default heuristic ignores it.
    """

    def __init__(self, snr_ref: float = 5.0, snr_scale: float = 0.15,
                 learnable: bool = False) -> None:
        super().__init__()
        self.snr_ref = float(snr_ref)
        self.snr_scale = float(snr_scale)
        self.learnable = learnable
        if learnable:
            from sgdjscc_lab.models.channel_condition_encoder import NUM_SCALARS
            self.head = nn.Linear(NUM_SCALARS, 1)
        else:
            self.head = None

    def predict(
        self,
        bundle: MeasurementBundle,
        condition: Optional[Dict] = None,
    ) -> Dict:
        """Return ``{reliability_map, confidence}`` for a measurement bundle.

        When *condition* (encoder output) and a learnable head are present, the
        scalar confidence comes from the head; otherwise the SNR/gain/mask
        heuristic is used.
        """
        feat = bundle.best_estimate
        bsz = feat.shape[0]
        device = feat.device

        rel = bundle.reliability if bundle.reliability is not None else bundle.mask
        if rel is None:
            rel = torch.ones(bsz, 1, *feat.shape[-2:], device=device)
        rel = rel.float().clamp(0, 1)

        if self.learnable and condition is not None and self.head is not None:
            conf = torch.sigmoid(self.head(condition["scalars"]).squeeze(-1))  # [B]
        else:
            snr = bundle.snr_db_est if bundle.snr_db_est is not None else bundle.snr_db_true
            snr_val = self.snr_ref if snr is None else float(snr)
            snr_conf = torch.sigmoid(
                torch.tensor(self.snr_scale * (snr_val - self.snr_ref), device=device)
            )
            mask_conf = rel.reshape(bsz, -1).mean(dim=1)            # [B]
            conf = (0.5 * snr_conf + 0.5 * mask_conf).clamp(0, 1)

        return {"reliability_map": rel, "confidence": conf}

    def forward(self, bundle, condition=None):  # noqa: D401
        return self.predict(bundle, condition)
