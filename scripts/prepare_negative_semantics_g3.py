#!/usr/bin/env python3
"""Materialize receiver-only G3 no/append/revocable condition manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sgdjscc_lab.guidance.saver_receiver_state import (  # noqa: E402
    G3_RECEIVER_ARMS,
    build_g3_receiver_condition_manifests,
    validate_g3_receiver_condition_manifest,
)


PROTOCOL_PATH = ROOT / "configs/experiments/negative_semantics/g3_oracle_revoke_protocol.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_state() -> dict:
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
        ).strip()),
    }


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-ids", help="Comma-separated subset; default is all frozen Pilot videos")
    parser.add_argument("--smoke", action="store_true", help="Materialize one deterministic event only")
    args = parser.parse_args(argv)

    from omegaconf import OmegaConf
    protocol = OmegaConf.to_container(OmegaConf.load(PROTOCOL_PATH), resolve=True)
    gt_path = ROOT / protocol["data"]["ground_truth_index"]
    if _sha256(gt_path) != protocol["data"]["ground_truth_index_sha256"]:
        raise SystemExit("G3 ground-truth index hash mismatch")
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    available = sorted(ground_truth["videos"])
    video_ids = (
        [value for value in args.video_ids.split(",") if value]
        if args.video_ids else available
    )
    if args.smoke:
        video_ids = video_ids[:1]
    if not args.smoke and len(video_ids) != int(protocol["data"]["expected_videos"]):
        raise SystemExit(
            f"formal G3 preparation requires {protocol['data']['expected_videos']} videos; "
            "use --smoke for one video"
        )

    manifests = build_g3_receiver_condition_manifests(ground_truth, video_ids)
    expected = {
        video_id: len(ground_truth["videos"][video_id]["frame_names"])
        for video_id in video_ids
    }
    output = args.output_root.resolve()
    paths = {}
    for arm in G3_RECEIVER_ARMS:
        validate_g3_receiver_condition_manifest(manifests[arm], expected)
        path = output / "conditions" / f"{arm}.json"
        if path.is_file() and json.loads(path.read_text()) != manifests[arm]:
            raise SystemExit(f"existing G3 condition differs for arm={arm}; use a new output root")
        _atomic_json(path, manifests[arm])
        paths[arm] = path
    run_spec = {
        "protocol_id": protocol["protocol_id"],
        "protocol_freeze_status": protocol["freeze_status"],
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "ground_truth_index_sha256": _sha256(gt_path),
        "git": _git_state(),
        "smoke": bool(args.smoke),
        "video_ids": video_ids,
        "arms": list(G3_RECEIVER_ARMS),
        "condition_sha256": {arm: _sha256(path) for arm, path in paths.items()},
        "oracle_side_input": "ORACLE_EVAL_ONLY",
        "serialized_in_packet": False,
        "rate_accounted": False,
        "heldout_accessed": False,
        "execution_authorized": False,
        "execution_blocker": "G2 provisional pass required and protocol must be frozen",
    }
    spec_path = output / "run_spec.json"
    if spec_path.is_file() and json.loads(spec_path.read_text()) != run_spec:
        raise SystemExit("existing G3 run_spec differs; use a new output root")
    _atomic_json(spec_path, run_spec)
    print(json.dumps(run_spec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
