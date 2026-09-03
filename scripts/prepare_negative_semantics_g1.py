#!/usr/bin/env python3
"""Materialize the frozen OVIS Pilot view used by G1.

The script creates only lightweight manifests and symlinks.  It never copies
media, samples another split, or reads the sealed held-out payload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parent.parent
G0_MANIFEST = Path("data/negative_semantics/g0/dataset_split_manifest.json")
G0_EVENTS = Path("data/negative_semantics/g0/ovis_official_gt_event_annotations.jsonl")
G0_MAPPING = Path("data/negative_semantics/g0/ovis_category_mapping.json")
DEFAULT_OUTPUT = Path("data/negative_semantics/g1/pilot")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _ensure_symlink(link: Path, target: Path) -> None:
    expected = target.resolve()
    if link.is_symlink():
        if link.resolve() != expected:
            raise RuntimeError(f"refusing to replace mismatched symlink {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing non-symlink {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(expected, target_is_directory=True)


def _presence_by_frame(gt: Dict[str, Any], mapping: Dict[str, str]) -> List[List[str]]:
    video = gt["videos"][0]
    n_frames = int(video["length"])
    category_names = {int(item["id"]): item["name"] for item in gt["categories"]}
    present = [set() for _ in range(n_frames)]
    for annotation in gt["annotations"]:
        concept = mapping.get(category_names[int(annotation["category_id"])])
        if concept is None:
            continue
        for index, segmentation in enumerate(annotation["segmentations"]):
            if index < n_frames and segmentation is not None:
                present[index].add(concept)
    return [sorted(values) for values in present]


def prepare(repo_root: Path = ROOT, output_root: Path = DEFAULT_OUTPUT) -> Dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    output_root = (
        Path(output_root) if Path(output_root).is_absolute() else repo_root / output_root
    ).resolve()
    manifest_path = repo_root / G0_MANIFEST
    events_path = repo_root / G0_EVENTS
    mapping_path = repo_root / G0_MAPPING
    manifest = load_json(manifest_path)
    events = load_jsonl(events_path)
    mapping_doc = load_json(mapping_path)
    mapping = mapping_doc["category_to_concept"]
    pilot = sorted(manifest["splits"]["pilot"], key=lambda row: row["video_id"])

    if len(pilot) != 40 or len(events) != 40:
        raise RuntimeError("G1 requires exactly 40 frozen Pilot videos and 40 events")
    if {row["video_id"] for row in pilot} != {row["video_id"] for row in events}:
        raise RuntimeError("Pilot video IDs and official-GT event video IDs differ")
    if any(row.get("source_id") != "ovis_v1" for row in pilot):
        raise RuntimeError("G1 Pilot must contain only OVIS v1 entries")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "captions").mkdir(exist_ok=True)
    index_videos: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []
    for entry in pilot:
        video_id = entry["video_id"]
        source = (repo_root / entry["input_path"]).resolve()
        gt_path = (repo_root / entry["gt_path"]).resolve()
        if not source.is_dir() or not gt_path.is_file():
            raise FileNotFoundError(f"missing frozen Pilot data for {video_id}")
        gt = load_json(gt_path)
        video = gt["videos"][0]
        frame_names = [Path(value).name for value in video["file_names"]]
        if len(frame_names) != int(entry["n_frames"]):
            raise RuntimeError(f"frame count changed for {video_id}")
        _ensure_symlink(output_root / "frames" / video_id, source)
        csv_rows.append({
            "dataset": "OVIS",
            "split": "pilot",
            "video_id": video_id,
            "processed_file": f"frames/{video_id}",
            "n_frames": len(frame_names),
            "width": int(video["width"]),
            "height": int(video["height"]),
            "fps": 1,
            "time_unit": "annotated_timestep",
            "source_frame_stride": int(entry["annotation_stride_source_frames"]),
            "input_sha256": entry["input_sha256"],
            "gt_sha256": entry["gt_sha256"],
        })
        index_videos[video_id] = {
            "native_video_id": entry["native_video_id"],
            "official_video_id": entry["official_video_id"],
            "source_path": entry["input_path"],
            "source_sha256": entry["input_sha256"],
            "gt_path": entry["gt_path"],
            "gt_sha256": entry["gt_sha256"],
            "frame_names": frame_names,
            "present_concepts": _presence_by_frame(gt, mapping),
        }

    manifest_out = output_root / "manifest.csv"
    fieldnames = list(csv_rows[0])
    with manifest_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    gt_index = {
        "index_id": "negative_semantics_g1_ovis_pilot_gt_v1",
        "source": "OVIS provider validation with-GT",
        "state_unit": "concept_per_annotated_timestep",
        "concepts": sorted(set(mapping.values())),
        "videos": index_videos,
        "events": sorted(events, key=lambda row: row["event_id"]),
    }
    gt_out = output_root / "ground_truth_index.json"
    _atomic_json(gt_out, gt_index)
    preparation = {
        "status": "PREPARED",
        "heldout_accessed": False,
        "video_count": len(csv_rows),
        "event_count": len(events),
        "g0_manifest_sha256": sha256_file(manifest_path),
        "g0_events_sha256": sha256_file(events_path),
        "g0_mapping_sha256": sha256_file(mapping_path),
        "g1_manifest_sha256": sha256_file(manifest_out),
        "g1_ground_truth_index_sha256": sha256_file(gt_out),
    }
    _atomic_json(output_root / "preparation_manifest.json", preparation)
    return preparation


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = prepare(args.repo_root, args.output_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
