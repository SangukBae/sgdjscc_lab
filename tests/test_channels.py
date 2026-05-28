"""tests/test_channels.py – Unit tests for AWGNChannel.

No GPU or SGDJSCC checkpoints required — all tests run on CPU tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.channels.awgn import AWGNChannel


class TestAWGNChannelTransmit:
    def test_output_shape_matches_input(self):
        ch = AWGNChannel()
        x = torch.randn(1, 4, 8, 8)
        y = ch.transmit(x, snr_db=10.0)
        assert y.shape == x.shape

    def test_output_dtype_matches_input(self):
        ch = AWGNChannel()
        x = torch.randn(2, 8, 4, 4)
        y = ch.transmit(x, snr_db=10.0)
        assert y.dtype == x.dtype

    def test_noise_is_added(self):
        """Output must differ from input (noise added, not identity)."""
        ch = AWGNChannel()
        x = torch.ones(1, 4, 8, 8)
        y = ch.transmit(x, snr_db=10.0)
        assert not torch.allclose(x, y), "transmit() returned identical tensor — no noise added"

    def test_low_snr_produces_more_noise_than_high_snr(self):
        """Average noise power at SNR=0 dB must exceed that at SNR=30 dB."""
        ch = AWGNChannel()
        torch.manual_seed(0)
        x = torch.ones(1, 4, 16, 16)
        n_trials = 20
        noise_low = sum(
            (ch.transmit(x, snr_db=0.0) - x).pow(2).mean().item()
            for _ in range(n_trials)
        )
        noise_high = sum(
            (ch.transmit(x, snr_db=30.0) - x).pow(2).mean().item()
            for _ in range(n_trials)
        )
        assert noise_low > noise_high

    def test_batch_size_two(self):
        ch = AWGNChannel()
        x = torch.randn(2, 4, 8, 8)
        y = ch.transmit(x, snr_db=10.0)
        assert y.shape == (2, 4, 8, 8)

    def test_zero_snr_does_not_raise(self):
        ch = AWGNChannel()
        x = torch.randn(1, 2, 4, 4)
        ch.transmit(x, snr_db=0.0)

    def test_negative_snr_does_not_raise(self):
        ch = AWGNChannel()
        x = torch.randn(1, 2, 4, 4)
        ch.transmit(x, snr_db=-5.0)
