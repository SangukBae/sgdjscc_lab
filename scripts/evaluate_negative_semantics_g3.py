#!/usr/bin/env python3
"""Summarize precomputed G3 detector/identity rows without running models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sgdjscc_lab.evaluators.negative_semantics_g3 import summarize_g3  # noqa: E402


PROTOCOL_PATH = ROOT / "configs/experiments/negative_semantics/g3_oracle_revoke_protocol.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args(argv)
    rows = [
        json.loads(line) for line in args.detection_rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_g3(
        rows, threshold=args.threshold, bootstrap_iterations=args.bootstrap_iterations
    )
    summary.update({
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "detection_rows_sha256": _sha256(args.detection_rows),
        "heldout_accessed": False,
        "evidence_scope": "PROVISIONAL_G3_PILOT",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
