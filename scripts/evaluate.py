#!/usr/bin/env python
"""evaluate.py – Evaluation script stub (Phase 3+).

Full metric evaluation (PSNR/SSIM/LPIPS/CLIP/object-preservation/
hallucination) is planned for Phase 3.

See evaluators/quality.py for the current PSNR/SSIM interface.
"""

from __future__ import annotations

import argparse
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="sgdjscc_lab evaluation (Phase 3+)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True,
                        help="Path to YAML config")
    parser.add_argument("--results", required=True,
                        help="Directory of reconstructed images")
    parser.add_argument("--reference", required=True,
                        help="Directory of original images")
    return parser.parse_args()


def main() -> None:
    _parse_args()
    print(
        "evaluate.py: Phase 3+ feature not yet implemented.\n"
        "See src/sgdjscc_lab/evaluators/quality.py for PSNR/SSIM wrappers."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
