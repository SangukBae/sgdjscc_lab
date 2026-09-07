"""Receiver-side negative-text conditioning for controlled decoder studies.

The production SGD-JSCC decoder has always used a fixed quality-oriented
negative prompt.  This module keeps that default byte-for-byte unchanged and
adds an explicit, opt-in semantic suffix.  G2 uses the suffix as an
``ORACLE_EVAL_ONLY`` receiver side input; it is deliberately not serialized in
the transmission bundle and therefore must never be reported as a rate-bearing
method.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List


QUALITY_NEGATIVE_PROMPT = (
    "distorted, discontinuous, ugly, blurry, low resolution, "
    "deformed, bad quality, deformed"
)

G2_CONDITION_SCHEMA = "negative_semantics_g2_condition_v1"
G2_ARMS = (
    "no_negative",
    "random_negative",
    "frequency_negative",
    "oracle_negative",
)


def _as_plain(value: Any) -> Any:
    """Convert OmegaConf containers without importing OmegaConf eagerly."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    return value


def resolve_negative_prompts(semantic_text: Sequence[str], cfg: Any = None) -> List[str]:
    """Return one negative prompt per positive prompt.

    With no opt-in block this returns the historical SGD-JSCC quality prompt.
    The supported config contract is::

        negative_conditioning:
          enabled: true
          mode: append       # append | replace
          prompt: "a dog, a cat"  # string broadcasts, or batch-sized list

    ``append`` preserves the historical quality prompt and adds the semantic
    terms. ``replace`` exists for explicit ablations only. Empty append text is
    equivalent to the unchanged baseline.
    """
    batch_size = len(semantic_text)
    baseline = [QUALITY_NEGATIVE_PROMPT for _ in range(batch_size)]
    if cfg is None:
        return baseline

    block = cfg.get("negative_conditioning", None)
    block = _as_plain(block)
    if block is None:
        return baseline
    if not isinstance(block, Mapping):
        raise ValueError("negative_conditioning must be a mapping")
    if not bool(block.get("enabled", False)):
        return baseline

    mode = str(block.get("mode", "append"))
    if mode not in {"append", "replace"}:
        raise ValueError("negative_conditioning.mode must be 'append' or 'replace'")
    if "prompt" not in block:
        raise ValueError("enabled negative_conditioning requires prompt")

    value = _as_plain(block["prompt"])
    if isinstance(value, str):
        suffixes = [value.strip() for _ in range(batch_size)]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        suffixes = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("negative_conditioning.prompt list must contain only strings")
            suffixes.append(item.strip())
        if len(suffixes) != batch_size:
            raise ValueError(
                "negative_conditioning.prompt list length must match semantic_text "
                f"({len(suffixes)} != {batch_size})"
            )
    else:
        raise ValueError("negative_conditioning.prompt must be a string or string list")

    if mode == "replace":
        return suffixes
    return [
        f"{quality}, {suffix}" if suffix else quality
        for quality, suffix in zip(baseline, suffixes)
    ]


def _stable_frame_seed(seed: int, video_id: str, frame_index: int) -> int:
    payload = f"{int(seed)}\0{video_id}\0{int(frame_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _prompt_for(concepts: Sequence[str], queries: Mapping[str, str]) -> str:
    return ", ".join(str(queries[concept]).strip() for concept in concepts)


def build_g2_condition_manifests(
    gt_index: Mapping[str, Any],
    video_ids: Sequence[str],
    *,
    concepts: Sequence[str],
    queries: Mapping[str, str],
    frequency_ranking: Sequence[str],
    random_seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Build the four deterministic, cardinality-matched G2 condition arms.

    Oracle concepts are the closed-vocabulary complement of official GT.
    Random/frequency controls use the *same number* of concepts, but their
    identities do not consult current-frame presence.  This makes text length
    comparable while allowing false suppression from an invalid control to be
    measured rather than hidden.  The no-negative arm retains only the legacy
    quality prompt (its semantic suffix is empty).
    """
    vocabulary = [str(value) for value in concepts]
    if not vocabulary or len(vocabulary) != len(set(vocabulary)):
        raise ValueError("G2 concepts must be a non-empty unique sequence")
    if set(queries) != set(vocabulary):
        raise ValueError("G2 queries must cover exactly the concept vocabulary")
    ranking = [str(value) for value in frequency_ranking]
    if len(ranking) != len(vocabulary) or set(ranking) != set(vocabulary):
        raise ValueError("frequency_ranking must be a permutation of the concept vocabulary")

    common = {
        "schema_version": G2_CONDITION_SCHEMA,
        "prompt_mode": "append",
        "concept_vocabulary": vocabulary,
        "queries": dict(queries),
        "random_seed": int(random_seed),
        "frequency_ranking": ranking,
        "receiver_side_input": True,
        "serialized_in_packet": False,
        "rate_accounted": False,
        "heldout_accessed": False,
        "cardinality_control": "random/frequency match oracle absent-count per frame",
    }
    manifests: Dict[str, Dict[str, Any]] = {
        arm: {
            **common,
            "arm": arm,
            "oracle_eval_only": arm == "oracle_negative",
            "videos": {},
        }
        for arm in G2_ARMS
    }

    videos = gt_index.get("videos") or {}
    for video_id in video_ids:
        if video_id not in videos:
            raise ValueError(f"G2 video missing from ground-truth index: {video_id}")
        metadata = videos[video_id]
        frame_names = list(metadata.get("frame_names") or [])
        present_rows = list(metadata.get("present_concepts") or [])
        if len(frame_names) != len(present_rows):
            raise ValueError(f"frame/GT length mismatch for {video_id}")

        arm_frames = {arm: [] for arm in G2_ARMS}
        for frame_index, present_values in enumerate(present_rows):
            present = {str(value) for value in present_values}
            unknown = sorted(present - set(vocabulary))
            if unknown:
                raise ValueError(f"unknown GT concepts for {video_id}/{frame_index}: {unknown}")
            oracle = [concept for concept in vocabulary if concept not in present]
            count = len(oracle)
            rng = random.Random(_stable_frame_seed(random_seed, video_id, frame_index))
            random_values = rng.sample(vocabulary, count)
            frequency_values = ranking[:count]
            selected = {
                "no_negative": [],
                "random_negative": random_values,
                "frequency_negative": frequency_values,
                "oracle_negative": oracle,
            }
            for arm, negative_concepts in selected.items():
                arm_frames[arm].append({
                    "frame_index": frame_index,
                    "frame_name": frame_names[frame_index],
                    "negative_concepts": negative_concepts,
                    "negative_prompt": _prompt_for(negative_concepts, queries),
                    "oracle_absent_count": count,
                })
        for arm in G2_ARMS:
            manifests[arm]["videos"][video_id] = {
                "frame_count": len(frame_names),
                "frames": arm_frames[arm],
            }

    return manifests


def validate_g2_condition_manifest(
    manifest: Mapping[str, Any],
    expected_video_frames: Mapping[str, int],
) -> Dict[str, List[Dict[str, Any]]]:
    """Validate a receiver condition manifest and return selected video rows."""
    value = json.loads(json.dumps(manifest))  # detach OmegaConf/custom mappings
    if value.get("schema_version") != G2_CONDITION_SCHEMA:
        raise ValueError("unsupported negative-condition manifest schema")
    arm = value.get("arm")
    if arm not in G2_ARMS:
        raise ValueError(f"unknown negative-condition arm: {arm!r}")
    if value.get("prompt_mode") != "append":
        raise ValueError("G2 condition manifests must use append prompt mode")
    if value.get("receiver_side_input") is not True:
        raise ValueError("G2 condition manifest is not marked receiver-side")
    if value.get("serialized_in_packet") is not False or value.get("rate_accounted") is not False:
        raise ValueError("G2 oracle/control text must remain outside packet/rate accounting")
    if value.get("heldout_accessed") is not False:
        raise ValueError("G2 condition manifest may not access held-out data")
    if bool(value.get("oracle_eval_only")) != (arm == "oracle_negative"):
        raise ValueError("oracle_eval_only flag does not match condition arm")

    vocabulary = list(value.get("concept_vocabulary") or [])
    if not vocabulary or len(vocabulary) != len(set(vocabulary)):
        raise ValueError("condition manifest has invalid concept vocabulary")
    allowed = set(vocabulary)
    videos = value.get("videos") or {}
    selected: Dict[str, List[Dict[str, Any]]] = {}
    for video_id, expected_count in expected_video_frames.items():
        if video_id not in videos:
            raise ValueError(f"condition manifest missing selected video {video_id}")
        record = videos[video_id]
        rows = list(record.get("frames") or [])
        if int(record.get("frame_count", -1)) != int(expected_count) or len(rows) != int(expected_count):
            raise ValueError(f"condition manifest frame count mismatch for {video_id}")
        for index, row in enumerate(rows):
            if int(row.get("frame_index", -1)) != index:
                raise ValueError(f"non-contiguous condition frame indices for {video_id}")
            terms = list(row.get("negative_concepts") or [])
            if len(terms) != len(set(terms)) or not set(terms).issubset(allowed):
                raise ValueError(f"invalid negative concepts for {video_id}/{index}")
            prompt = row.get("negative_prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"negative_prompt must be text for {video_id}/{index}")
            if arm == "no_negative" and (terms or prompt.strip()):
                raise ValueError("no_negative arm must have an empty semantic suffix")
        selected[video_id] = rows
    return selected
