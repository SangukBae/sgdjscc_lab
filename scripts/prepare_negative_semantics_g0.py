#!/usr/bin/env python3
"""Build the deterministic negative-semantics G0 split and review package.

The script never downloads data. It consumes the official YouTube-VOS 2019
train/valid and DAVIS 2017 trainval distributions after they have been placed
under data/external. Raw media and rendered review clips remain git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


YT_SOURCE = "youtube_vos_2019"
DAVIS_SOURCE = "davis_2017"
PILOT_TARGET = 33
BASE = Path("data/negative_semantics/g0")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> Tuple[str, int]:
    """Hash a tree as sorted ``relative_path NUL file_sha256 LF`` records."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
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
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(rendered, encoding="utf-8")


def relative_to_repo(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def artifact_entry(
    path: Path,
    repo_root: Path,
    path_key: str,
    hash_key: str,
) -> Dict[str, Any]:
    if path.is_file():
        digest = sha256_file(path)
        count = 1
        kind = "file"
    elif path.is_dir():
        digest, count = sha256_tree(path)
        kind = "tree"
    else:
        raise FileNotFoundError(path)
    return {
        path_key: relative_to_repo(path, repo_root),
        hash_key: digest,
        f"{path_key}_kind": kind,
        f"{path_key}_file_count": count,
    }


def video_entry(
    *,
    repo_root: Path,
    source_id: str,
    video_id: str,
    native_video_id: str,
    frame_dir: Path,
    annotation_dir: Path,
    partition: str,
    playback_transform: str = "identity",
) -> Dict[str, Any]:
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise ValueError(f"no JPEG frames in {frame_dir}")
    entry: Dict[str, Any] = {
        "source_id": source_id,
        "video_id": video_id,
        "native_video_id": native_video_id,
        "source_partition": partition,
        "playback_transform": playback_transform,
    }
    entry.update(artifact_entry(frame_dir, repo_root, "input_path", "input_sha256"))
    entry.update(artifact_entry(annotation_dir, repo_root, "gt_path", "gt_sha256"))
    entry["n_frames"] = len(frames)
    return entry


def youtube_candidates(
    valid_root: Path, category_map: Dict[str, str]
) -> List[Dict[str, Any]]:
    videos = load_json(valid_root / "meta.json")["videos"]
    candidates: List[Dict[str, Any]] = []
    for native_video_id in sorted(videos):
        frame_ids = sorted(
            item.stem for item in (valid_root / "JPEGImages" / native_video_id).glob("*.jpg")
        )
        position = {frame_id: index for index, frame_id in enumerate(frame_ids)}
        per_video: List[Tuple[int, str, str, str]] = []
        for entity_id, obj in sorted(videos[native_video_id]["objects"].items()):
            concept_id = category_map.get(obj["category"])
            if concept_id is None:
                continue
            observed = sorted(position[frame] for frame in obj["frames"] if frame in position)
            if observed and observed[0] > 0:
                per_video.append((observed[0], str(entity_id), obj["category"], concept_id))
        if not per_video:
            continue
        boundary, entity_id, native_category, concept_id = sorted(per_video)[0]
        candidates.append({
            "native_video_id": native_video_id,
            "entity_id": entity_id,
            "native_category": native_category,
            "concept_id": concept_id,
            "natural_enter_index": boundary,
            "n_frames": len(frame_ids),
            "source_frame_ids": frame_ids,
        })
    if len(candidates) < PILOT_TARGET:
        raise RuntimeError(
            f"only {len(candidates)} eligible videos; {PILOT_TARGET} are required"
        )
    return candidates[:PILOT_TARGET]


def build_pilot_events(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        reverse = index % 2 == 0
        transform = "time_reverse" if reverse else "identity"
        event_type = "EXIT" if reverse else "ENTER"
        if reverse:
            # Original first-present frame becomes the final present frame in
            # reversed playback; the following frame is the EXIT boundary.
            boundary = candidate["n_frames"] - candidate["natural_enter_index"]
            before, after = "present", "unknown"
            visibility_before, visibility_after = "visible", "unknown"
        else:
            boundary = candidate["natural_enter_index"]
            before, after = "unknown", "present"
            visibility_before, visibility_after = "unknown", "visible"
        event_id = f"pilot_{index:03d}_{event_type.lower()}"
        video_id = f"ytvos2019_valid_{candidate['native_video_id']}"
        rows.append({
            "event_id": event_id,
            "split": "pilot",
            "source_id": YT_SOURCE,
            "video_id": video_id,
            "scene_epoch": 0,
            "independence_cluster_id": f"{video_id}_episode_000",
            "entity_id": candidate["entity_id"],
            "concept_id": candidate["concept_id"],
            "event_type": event_type,
            "start_frame": boundary,
            "end_frame": boundary,
            "state_before": before,
            "state_after": after,
            "visibility_before": visibility_before,
            "visibility_after": visibility_after,
            "annotation_source": "youtube_vos_2019_mask_candidate",
            "human_verified": False,
            "human_verification": {
                "status": "not_reviewed",
                "annotator_ids": [],
                "adjudicator_id": None,
            },
            "source_frame_id": candidate["source_frame_ids"][
                candidate["natural_enter_index"]
            ],
            "playback_transform": transform,
            "native_category": candidate["native_category"],
            "review_clip_path": (
                f"data/external/youtube_vos_2019/pilot_review/{event_id}.mp4"
            ),
            "notes": (
                "Machine-derived candidate only. Reviewers must confirm the concept, "
                "entity, boundary, and absence interval before state unknown may become "
                "confirmed_absent and human_verified may become true."
            ),
        })
    return rows


def render_review_clip(
    *,
    frames: Sequence[Path],
    output: Path,
    reverse: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(reversed(frames)) if reverse else list(frames)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as handle:
        for frame in ordered:
            escaped = str(frame.resolve()).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
            handle.write("duration 0.2\n")
        escaped = str(ordered[-1].resolve()).replace("'", "'\\''")
        handle.write(f"file '{escaped}'\n")
        handle.flush()
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                "-i", handle.name, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
            ],
            check=True,
        )


def write_review_templates(base: Path, events: Sequence[Dict[str, Any]]) -> None:
    review_dir = base / "pilot_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "event_id", "video_id", "concept_id", "proposed_event_type",
        "proposed_boundary_frame", "playback_transform", "review_clip_path",
        "decision", "reviewed_event_type", "reviewed_boundary_frame",
        "absence_confirmed", "identity_confirmed", "notes",
    ]
    for name in ("annotator_a_template.csv", "annotator_b_template.csv"):
        with (review_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for event in events:
                writer.writerow({
                    "event_id": event["event_id"],
                    "video_id": event["video_id"],
                    "concept_id": event["concept_id"],
                    "proposed_event_type": event["event_type"],
                    "proposed_boundary_frame": event["start_frame"],
                    "playback_transform": event["playback_transform"],
                    "review_clip_path": event["review_clip_path"],
                    "decision": "",
                    "reviewed_event_type": "",
                    "reviewed_boundary_frame": "",
                    "absence_confirmed": "",
                    "identity_confirmed": "",
                    "notes": "",
                })
    adjudication_fields = [
        "event_id", "required", "final_decision", "final_event_type",
        "final_boundary_frame", "absence_confirmed", "identity_confirmed", "notes",
    ]
    with (review_dir / "adjudicator_template.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=adjudication_fields, lineterminator="\n"
        )
        writer.writeheader()
        for event in events:
            writer.writerow({"event_id": event["event_id"], "required": "auto"})


def archive_record(
    *,
    repo_root: Path,
    source_id: str,
    partition: str,
    archive: Path,
    official_url: str,
    provider_file_id: Optional[str] = None,
) -> Dict[str, Any]:
    record = {
        "source_id": source_id,
        "partition": partition,
        "archive_path": relative_to_repo(archive, repo_root),
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "official_url": official_url,
    }
    if provider_file_id:
        record["provider_file_id"] = provider_file_id
    return record


def prepare(repo_root: Path, render_clips: bool) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    base = repo_root / BASE
    yt_external = repo_root / "data/external/youtube_vos_2019"
    yt_train = yt_external / "extracted/train"
    yt_valid = yt_external / "extracted/valid"
    davis = repo_root / "data/external/davis_2017/extracted/DAVIS"
    old_manifest = load_json(base / "dataset_split_manifest.json")

    category_map = load_json(base / "youtube_vos_category_mapping.json")[
        "category_to_concept"
    ]
    pilot_candidates = youtube_candidates(yt_valid, category_map)
    pilot_native_ids = {item["native_video_id"] for item in pilot_candidates}
    pilot_events = build_pilot_events(pilot_candidates)
    transform_by_native = {
        item["video_id"].removeprefix("ytvos2019_valid_"): item["playback_transform"]
        for item in pilot_events
    }

    train_meta = load_json(yt_train / "meta.json")["videos"]
    valid_meta = load_json(yt_valid / "meta.json")["videos"]
    davis_val = [
        line.strip()
        for line in (davis / "ImageSets/2017/val.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    splits: Dict[str, List[Dict[str, Any]]] = {
        "pilot": [],
        "train": [],
        "development": old_manifest["splits"]["development"],
        "validation": [],
        "heldout_test": [],
    }
    for native_id in sorted(pilot_native_ids):
        splits["pilot"].append(video_entry(
            repo_root=repo_root,
            source_id=YT_SOURCE,
            video_id=f"ytvos2019_valid_{native_id}",
            native_video_id=native_id,
            frame_dir=yt_valid / "JPEGImages" / native_id,
            annotation_dir=yt_valid / "Annotations" / native_id,
            partition="valid",
            playback_transform=transform_by_native[native_id],
        ))
    for native_id in sorted(train_meta):
        splits["train"].append(video_entry(
            repo_root=repo_root,
            source_id=YT_SOURCE,
            video_id=f"ytvos2019_train_{native_id}",
            native_video_id=native_id,
            frame_dir=yt_train / "JPEGImages" / native_id,
            annotation_dir=yt_train / "Annotations" / native_id,
            partition="train",
        ))
    for native_id in sorted(set(valid_meta) - pilot_native_ids):
        splits["validation"].append(video_entry(
            repo_root=repo_root,
            source_id=YT_SOURCE,
            video_id=f"ytvos2019_valid_{native_id}",
            native_video_id=native_id,
            frame_dir=yt_valid / "JPEGImages" / native_id,
            annotation_dir=yt_valid / "Annotations" / native_id,
            partition="valid",
        ))
    for native_id in davis_val:
        splits["heldout_test"].append(video_entry(
            repo_root=repo_root,
            source_id=DAVIS_SOURCE,
            video_id=f"davis2017_val_{native_id}",
            native_video_id=native_id,
            frame_dir=davis / "JPEGImages/480p" / native_id,
            annotation_dir=davis / "Annotations/480p" / native_id,
            partition="val",
        ))

    acquisition = [
        archive_record(
            repo_root=repo_root,
            source_id=YT_SOURCE,
            partition="train",
            archive=yt_external / "archives/train.tar",
            official_url="https://drive.google.com/drive/folders/1XwjQ-eysmOb7JdmJAwfVOBZX-aMbHccC?usp=sharing",
            provider_file_id="1lU9jCX-H0ntwh87tt2cA0xEPeWOJzD6S",
        ),
        archive_record(
            repo_root=repo_root,
            source_id=YT_SOURCE,
            partition="valid",
            archive=yt_external / "archives/valid.tar",
            official_url="https://drive.google.com/drive/folders/1XwjQ-eysmOb7JdmJAwfVOBZX-aMbHccC?usp=sharing",
            provider_file_id="1bw8KcpzfrT08HYbuROZmY0bp4TkYl4_g",
        ),
        archive_record(
            repo_root=repo_root,
            source_id=DAVIS_SOURCE,
            partition="trainval_480p",
            archive=repo_root / "data/external/davis_2017/archives/DAVIS-2017-trainval-480p.zip",
            official_url="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip",
        ),
    ]

    manifest = {
        "manifest_id": "negative_semantics_g0_split_v1_1",
        "version": "1.1.0",
        "frozen_at": "2026-09-02",
        "hash_algorithm": "sha256",
        "tree_hash_algorithm": "sha256 over sorted relative_path NUL file_sha256 LF records",
        "split_lists_frozen": True,
        "empty_split_policy": "No split may be populated by a runtime sampler.",
        "leakage_policy": old_manifest["leakage_policy"],
        "acquisition_artifacts": acquisition,
        "splits": splits,
        "source_audit": [
            {
                "source_id": "etri_internal_legacy_v1",
                "assigned_splits": ["development"],
                "license_status": "approved",
                "redistribution": "not_authorized",
                "approval_record": "data/negative_semantics/g0/data_use_approval.json",
                "decision": "development_only",
            },
            {
                "source_id": YT_SOURCE,
                "assigned_splits": ["pilot", "train", "validation"],
                "license_status": "approved",
                "redistribution": "raw_media_not_authorized",
                "official_url": "https://youtube-vos.org/dataset/term/",
                "approval_record": "data/negative_semantics/g0/data_use_approval.json",
                "decision": "approved_internal_noncommercial_research",
            },
            {
                "source_id": DAVIS_SOURCE,
                "assigned_splits": ["heldout_test"],
                "license_status": "approved",
                "redistribution": "raw_media_not_authorized",
                "official_url": "https://davischallenge.org/challenge2017/rulesdates.html",
                "approval_record": "data/negative_semantics/g0/data_use_approval.json",
                "decision": "approved_internal_research_heldout_sealed",
            },
        ],
        "selection_decision": {
            "legacy_development": "All ten ETRI videos remain development-only.",
            "pilot": (
                "The first 33 lexicographically sorted YouTube-VOS valid videos with a "
                "core-ontology mid-video first appearance are used; one event cluster per "
                "native video, alternating identity and time-reverse playback."
            ),
            "train": "All official YouTube-VOS 2019 train video IDs.",
            "validation": "All official YouTube-VOS 2019 valid IDs not assigned to Pilot.",
            "heldout_test": (
                "All 30 official DAVIS 2017 val IDs; source-disjoint and sealed from method tuning."
            ),
            "remaining_blocker": (
                "Pilot candidates require two independent human reviews and adjudication of disagreements."
            ),
        },
    }
    write_json(base / "dataset_split_manifest.json", manifest)
    digest = sha256_file(base / "dataset_split_manifest.json")
    (base / "dataset_split_manifest.sha256").write_text(
        f"{digest}  dataset_split_manifest.json\n", encoding="utf-8"
    )
    write_json(base / "heldout_seal.json", {
        "seal_id": "negative_semantics_davis2017_heldout_v1_1",
        "sealed_at": "2026-09-02",
        "source_id": DAVIS_SOURCE,
        "source_partition": "val",
        "video_count": len(splits["heldout_test"]),
        "split_manifest_sha256": digest,
        "opened_for_method_development": False,
        "authorized_open_condition": (
            "Open once after G1-G3 implementation, all method choices, thresholds, "
            "budgets, and the validation operating point are frozen."
        ),
    })
    write_jsonl(base / "youtube_vos_pilot_event_candidates.jsonl", pilot_events)
    write_review_templates(base, pilot_events)

    if render_clips:
        for event in pilot_events:
            native_id = event["video_id"].removeprefix("ytvos2019_valid_")
            frames = sorted((yt_valid / "JPEGImages" / native_id).glob("*.jpg"))
            render_review_clip(
                frames=frames,
                output=repo_root / event["review_clip_path"],
                reverse=event["playback_transform"] == "time_reverse",
            )

    return {
        "manifest_sha256": digest,
        "split_counts": {split: len(entries) for split, entries in splits.items()},
        "pilot_candidate_events": len(pilot_events),
        "pilot_event_types": {
            event_type: sum(row["event_type"] == event_type for row in pilot_events)
            for event_type in ("ENTER", "EXIT", "OCCLUDE", "REAPPEAR", "SCENE_CUT")
        },
        "review_clips_rendered": render_clips,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--render-review-clips", action="store_true")
    args = parser.parse_args()
    result = prepare(Path(args.repo_root), args.render_review_clips)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
