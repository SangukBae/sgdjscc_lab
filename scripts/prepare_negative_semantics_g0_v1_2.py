#!/usr/bin/env python3
"""Build the official-GT OVIS Pilot amendment for negative-semantics G0 v1.2.

The script never downloads data.  It consumes the official OVIS validation
frame archive plus the provider-published ``valid_withgt`` JSON file.  It
derives conservative ENTER, EXIT, OCCLUDE, and REAPPEAR events, chooses ten of
each with a deterministic bipartite matching so every event comes from a
different video, extracts only those 40 videos, and rewrites the frozen split.

OVIS frames are annotated at every fifth original frame.  All event boundaries
therefore use ``annotated_timestep`` rather than source-frame indices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BASE = Path("data/negative_semantics/g0")
MANIFEST = BASE / "dataset_split_manifest.json"
MANIFEST_HASH = BASE / "dataset_split_manifest.sha256"
EVENTS = BASE / "ovis_official_gt_event_annotations.jsonl"
OVIS_SOURCE = "ovis_v1"
EVENT_TYPES = ("ENTER", "EXIT", "OCCLUDE", "REAPPEAR")
EVENTS_PER_TYPE = 10
ANNOTATION_STRIDE = 5
CORE_CATEGORY_TO_CONCEPT = {
    "Person": "person",
    "Bird": "bird",
    "Cat": "cat",
    "Dog": "dog",
    "Bicycle": "bicycle",
    "Motorcycle": "motorcycle",
    "Poultry": "bird",
    "Parrot": "bird",
    "Vehical": "vehicle",
}
PARTITIONS = {
    "valid": {
        "frames": Path("data/external/ovis/archives/valid.zip"),
        "gt": Path("data/external/ovis/archives/annotations_valid_withgt.json"),
        "provider_file_ids": {
            "frames": "171Z_pqcVkbJQ5lDMVM_mSfscjOPLBW7Z",
            "gt": "1vNnm4iPPNWhDQxbQmZ4XV8Ueil9VTEaJ",
        },
    },
}
OFFICIAL_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1eE4lLKCbv54E866XBVce_ebh3oXYq99b?usp=drive_link"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def touches_border(bbox: Optional[Sequence[float]], width: int, height: int, margin: int) -> bool:
    if bbox is None:
        return False
    x, y, box_width, box_height = bbox
    return (
        x <= margin
        or y <= margin
        or x + box_width >= width - margin
        or y + box_height >= height - margin
    )


def visible_run_before(segmentations: Sequence[Any], index: int) -> int:
    count = 0
    while index >= 0 and segmentations[index] is not None:
        count += 1
        index -= 1
    return count


def visible_run_after(segmentations: Sequence[Any], index: int) -> int:
    count = 0
    while index < len(segmentations) and segmentations[index] is not None:
        count += 1
        index += 1
    return count


def derive_candidates(partition: str, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    videos = {item["id"]: item for item in dataset["videos"]}
    categories = {item["id"]: item["name"] for item in dataset["categories"]}
    candidates: List[Dict[str, Any]] = []
    for annotation in sorted(dataset["annotations"], key=lambda item: item["id"]):
        native_category = categories[annotation["category_id"]]
        concept = CORE_CATEGORY_TO_CONCEPT.get(native_category)
        if concept is None:
            continue
        video = videos[annotation["video_id"]]
        masks = annotation["segmentations"]
        boxes = annotation["bboxes"]
        occlusion = annotation["occlusion"]
        present = [index for index, mask in enumerate(masks) if mask is not None]
        if not present:
            continue
        first, last = present[0], present[-1]
        common = {
            "partition": partition,
            "official_video_id": annotation["video_id"],
            "official_annotation_id": annotation["id"],
            "native_video_id": video["file_names"][0].split("/", 1)[0],
            "native_category": native_category,
            "concept_id": concept,
        }
        if (
            first >= 2
            and visible_run_after(masks, first) >= 3
            and touches_border(boxes[first], video["width"], video["height"], 2)
        ):
            candidates.append({
                **common,
                "event_type": "ENTER",
                "boundary": first,
                "gap_length": None,
                "rule_id": "ovis_enter_border_v1",
            })
        if (
            last <= len(masks) - 3
            and visible_run_before(masks, last) >= 3
            and touches_border(boxes[last], video["width"], video["height"], 2)
        ):
            candidates.append({
                **common,
                "event_type": "EXIT",
                "boundary": last + 1,
                "gap_length": None,
                "rule_id": "ovis_exit_border_v1",
            })

        index = 1
        while index < len(masks) - 1:
            if masks[index] is not None:
                index += 1
                continue
            gap_start = index
            while index < len(masks) and masks[index] is None:
                index += 1
            if index >= len(masks):
                break
            gap_length = index - gap_start
            qualifies = (
                gap_length >= 1
                and visible_run_before(masks, gap_start - 1) >= 3
                and visible_run_after(masks, index) >= 3
                and occlusion[gap_start - 1] != "no_occlusion"
                and occlusion[index] != "no_occlusion"
                and not touches_border(
                    boxes[gap_start - 1], video["width"], video["height"], 5
                )
                and not touches_border(boxes[index], video["width"], video["height"], 5)
            )
            if qualifies:
                candidates.append({
                    **common,
                    "event_type": "OCCLUDE",
                    "boundary": gap_start,
                    "gap_length": gap_length,
                    "rule_id": "ovis_internal_occlusion_gap_start_v1",
                })
                candidates.append({
                    **common,
                    "event_type": "REAPPEAR",
                    "boundary": index,
                    "gap_length": gap_length,
                    "rule_id": "ovis_internal_occlusion_gap_end_v1",
                })
    return candidates


def select_events(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select 10/type with a deterministic maximum bipartite matching."""
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_type[candidate["event_type"]].append(candidate)
    for event_type in EVENT_TYPES:
        by_type[event_type].sort(
            key=lambda item: (
                item["partition"], item["official_annotation_id"], item["boundary"]
            )
        )

    slots = [(event_type, index) for event_type in EVENT_TYPES for index in range(EVENTS_PER_TYPE)]
    matched_video: Dict[Tuple[str, int], Tuple[Tuple[str, int], Dict[str, Any]]] = {}

    def assign(slot: Tuple[str, int], seen: set[Tuple[str, int]]) -> bool:
        for candidate in by_type[slot[0]]:
            video_key = (candidate["partition"], candidate["official_video_id"])
            if video_key in seen:
                continue
            seen.add(video_key)
            previous = matched_video.get(video_key)
            if previous is None or assign(previous[0], seen):
                matched_video[video_key] = (slot, candidate)
                return True
        return False

    matched = sum(assign(slot, set()) for slot in slots)
    if matched != len(slots):
        raise RuntimeError(f"only {matched}/{len(slots)} distinct OVIS videos can be matched")
    selected = [value for _, value in sorted(matched_video.values(), key=lambda item: item[0])]
    if len({(item["partition"], item["official_video_id"]) for item in selected}) != 40:
        raise AssertionError("selected OVIS events are not video-independent")
    return selected


def extract_selected_frames(
    archive: Path,
    destination: Path,
    videos: Sequence[Dict[str, Any]],
) -> None:
    required = {
        file_name
        for video in videos
        for file_name in video["file_names"]
    }
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        suffix_to_member: Dict[str, str] = {}
        for member in handle.namelist():
            normalized = member.lstrip("./")
            parts = normalized.split("/")
            file_name = "/".join(parts[-2:]) if len(parts) >= 2 else normalized
            if file_name in required:
                if file_name in suffix_to_member:
                    raise RuntimeError(f"duplicate archive member for {file_name}")
                suffix_to_member[file_name] = member
        missing = sorted(required - set(suffix_to_member))
        if missing:
            raise RuntimeError(f"{len(missing)} required OVIS frames missing; first={missing[0]}")
        for file_name in sorted(required):
            target = destination / file_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(suffix_to_member[file_name]) as source, target.open("wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)


def per_video_gt(dataset: Dict[str, Any], official_video_id: int, source_sha256: str) -> Dict[str, Any]:
    video = next(item for item in dataset["videos"] if item["id"] == official_video_id)
    annotations = [
        item for item in dataset["annotations"] if item["video_id"] == official_video_id
    ]
    return {
        "derivation": {
            "dataset": "OVIS",
            "source_annotation_sha256": source_sha256,
            "preserves_official_values": True,
        },
        "info": dataset.get("info"),
        "licenses": dataset.get("licenses"),
        "categories": dataset["categories"],
        "videos": [video],
        "annotations": annotations,
    }


def event_row(candidate: Dict[str, Any], video: Dict[str, Any], gt_path: Path, root: Path, ordinal: int) -> Dict[str, Any]:
    event_type = candidate["event_type"]
    boundary = candidate["boundary"]
    if event_type == "ENTER":
        before, after = "confirmed_absent", "present"
        visibility_before, visibility_after = "out_of_frame", "visible"
    elif event_type == "EXIT":
        before, after = "present", "confirmed_absent"
        visibility_before, visibility_after = "visible", "out_of_frame"
    elif event_type == "OCCLUDE":
        before = after = "present"
        visibility_before, visibility_after = "partial", "occluded"
    else:
        before = after = "present"
        visibility_before, visibility_after = "occluded", "partial"
    video_id = f"ovis_{candidate['partition']}_{candidate['native_video_id']}"
    source_frame = video["file_names"][boundary]
    return {
        "event_id": f"ovis_gt_{ordinal:03d}_{event_type.lower()}",
        "split": "pilot",
        "source_id": OVIS_SOURCE,
        "video_id": video_id,
        "scene_epoch": 0,
        "independence_cluster_id": f"{video_id}_episode_000",
        "entity_id": str(candidate["official_annotation_id"]),
        "concept_id": candidate["concept_id"],
        "event_type": event_type,
        "start_frame": boundary,
        "end_frame": boundary,
        "boundary_unit": "annotated_timestep",
        "source_frame_stride": ANNOTATION_STRIDE,
        "state_scope": "entity",
        "state_before": before,
        "state_after": after,
        "visibility_before": visibility_before,
        "visibility_after": visibility_after,
        "annotation_source": "ovis_provider_withgt_v1",
        "source_frame_id": source_frame.rsplit("/", 1)[-1],
        "playback_transform": "identity",
        "native_category": candidate["native_category"],
        "human_verified": False,
        "human_verification": {
            "status": "not_reviewed",
            "annotator_ids": [],
            "adjudicator_id": None,
        },
        "official_gt_verified": True,
        "official_gt_verification": {
            "status": "official_gt_verified",
            "dataset": "OVIS",
            "partition": candidate["partition"],
            "official_video_id": candidate["official_video_id"],
            "official_annotation_id": candidate["official_annotation_id"],
            "derivation_rule_id": candidate["rule_id"],
            "per_video_gt_path": relative(gt_path, root),
            "per_video_gt_sha256": sha256_file(gt_path),
        },
        "notes": (
            "Deterministically derived from exhaustive provider GT. ENTER/EXIT require "
            "a frame-border transition and three visible annotated timesteps; OCCLUDE/"
            "REAPPEAR require a >=1-timestep internal gap, three visible timesteps on "
            "both sides, non-'no_occlusion' labels on both flanks, and no border contact."
        ),
    }


def artifact_entry(path: Path, root: Path, path_key: str, hash_key: str) -> Dict[str, Any]:
    if path.is_dir():
        digest, count = sha256_tree(path)
        kind = "tree"
    else:
        digest, count, kind = sha256_file(path), 1, "file"
    return {
        path_key: relative(path, root),
        hash_key: digest,
        f"{path_key}_kind": kind,
        f"{path_key}_file_count": count,
    }


def build(repo_root: Path) -> Dict[str, Any]:
    root = repo_root.resolve()
    manifest = load_json(root / MANIFEST)
    frozen_mapping = load_json(root / BASE / "ovis_category_mapping.json")
    if frozen_mapping.get("category_to_concept") != CORE_CATEGORY_TO_CONCEPT:
        raise ValueError("script OVIS mapping differs from frozen ovis_category_mapping.json")
    datasets: Dict[str, Dict[str, Any]] = {}
    gt_source_hashes: Dict[str, str] = {}
    all_candidates: List[Dict[str, Any]] = []
    for partition, paths in PARTITIONS.items():
        frame_archive = root / paths["frames"]
        gt_archive = root / paths["gt"]
        if not frame_archive.is_file() or not gt_archive.is_file():
            raise FileNotFoundError(f"missing OVIS {partition} archive or withgt annotation")
        datasets[partition] = load_json(gt_archive)
        gt_source_hashes[partition] = sha256_file(gt_archive)
        all_candidates.extend(derive_candidates(partition, datasets[partition]))
    selected = select_events(all_candidates)

    by_partition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in selected:
        video = next(
            video for video in datasets[item["partition"]]["videos"]
            if video["id"] == item["official_video_id"]
        )
        by_partition[item["partition"]].append(video)
    for partition, videos in by_partition.items():
        extract_selected_frames(
            root / PARTITIONS[partition]["frames"],
            root / "data/external/ovis/extracted" / partition,
            videos,
        )

    old_pilot = [
        entry for entry in manifest["splits"]["pilot"]
        if entry.get("source_id") == "youtube_vos_2019"
    ]
    for entry in old_pilot:
        entry["playback_transform"] = "identity"
    validation_by_id = {
        entry["video_id"]: entry
        for entry in manifest["splits"]["validation"] + old_pilot
    }
    manifest["splits"]["validation"] = sorted(
        validation_by_id.values(), key=lambda item: item["video_id"]
    )

    pilot_entries: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    for ordinal, item in enumerate(selected, start=1):
        partition = item["partition"]
        dataset = datasets[partition]
        video = next(v for v in dataset["videos"] if v["id"] == item["official_video_id"])
        frame_dir = root / "data/external/ovis/extracted" / partition / item["native_video_id"]
        gt_path = (
            root / "data/external/ovis/extracted" / partition
            / "annotations_by_video" / f"{item['native_video_id']}.json"
        )
        write_json(
            gt_path,
            per_video_gt(
                dataset,
                item["official_video_id"],
                gt_source_hashes[partition],
            ),
        )
        video_id = f"ovis_{partition}_{item['native_video_id']}"
        entry: Dict[str, Any] = {
            "source_id": OVIS_SOURCE,
            "video_id": video_id,
            "native_video_id": item["native_video_id"],
            "official_video_id": item["official_video_id"],
            "source_partition": partition,
            "playback_transform": "identity",
            "annotation_stride_source_frames": ANNOTATION_STRIDE,
        }
        entry.update(artifact_entry(frame_dir, root, "input_path", "input_sha256"))
        entry.update(artifact_entry(gt_path, root, "gt_path", "gt_sha256"))
        entry["n_frames"] = len(video["file_names"])
        pilot_entries.append(entry)
        event_rows.append(event_row(item, video, gt_path, root, ordinal))

    manifest["manifest_id"] = "negative_semantics_g0_split_v1_2"
    manifest["version"] = "1.2.0"
    manifest["frozen_at"] = "2026-09-03"
    manifest["splits"]["pilot"] = sorted(pilot_entries, key=lambda item: item["video_id"])
    retained = [
        item for item in manifest["acquisition_artifacts"]
        if item.get("source_id") != OVIS_SOURCE
    ]
    for partition, paths in PARTITIONS.items():
        for kind, archive_key in (("frames", "frames"), ("annotations", "gt")):
            path = root / paths[archive_key]
            retained.append({
                "source_id": OVIS_SOURCE,
                "partition": partition,
                "artifact_kind": kind,
                "archive_path": relative(path, root),
                "archive_sha256": sha256_file(path),
                "archive_size_bytes": path.stat().st_size,
                "official_url": OFFICIAL_FOLDER_URL,
                "provider_file_id": paths["provider_file_ids"][archive_key],
            })
    manifest["acquisition_artifacts"] = retained
    for source in manifest["source_audit"]:
        if source["source_id"] == "youtube_vos_2019":
            source["assigned_splits"] = ["train", "validation"]
    manifest["source_audit"] = [
        source for source in manifest["source_audit"] if source["source_id"] != OVIS_SOURCE
    ] + [{
        "source_id": OVIS_SOURCE,
        "assigned_splits": ["pilot"],
        "license_status": "approved",
        "redistribution": "raw_media_not_authorized",
        "official_url": "https://songbai.site/ovis/",
        "approval_record": "data/negative_semantics/g0/data_use_approval.json",
        "decision": "approved_internal_noncommercial_research_official_gt_pilot",
    }]
    manifest["selection_decision"]["pilot"] = (
        "Deterministic maximum matching selects 10 provider-GT-derived events of each "
        "type ENTER, EXIT, OCCLUDE, and REAPPEAR from distinct OVIS validation videos."
    )
    manifest["selection_decision"]["validation"] = (
        "All 507 official YouTube-VOS 2019 valid video IDs; the former v1.1 Pilot is "
        "returned to Validation without time reversal."
    )
    manifest["selection_decision"]["remaining_blocker"] = None
    write_json(root / MANIFEST, manifest)
    write_jsonl(root / EVENTS, event_rows)
    (root / MANIFEST_HASH).write_text(
        f"{sha256_file(root / MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8"
    )
    return {
        "selected_events": len(event_rows),
        "event_type_counts": {
            event_type: sum(row["event_type"] == event_type for row in event_rows)
            for event_type in EVENT_TYPES
        },
        "independent_videos": len({row["video_id"] for row in event_rows}),
        "split_counts": {key: len(value) for key, value in manifest["splits"].items()},
        "manifest_sha256": sha256_file(root / MANIFEST),
        "events_sha256": sha256_file(root / EVENTS),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    print(json.dumps(build(Path(args.repo_root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
