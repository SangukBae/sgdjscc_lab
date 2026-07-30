"""pipelines/heldout_remeasurement.py – Held-out re-measurement pipeline (ETRI 5차, step 9).

Re-reads 1~4차 results (``TemporalPipeline.run()`` records, or packets saved to
disk) and recomputes packet-verifier reports + PTC/SFR/SDI **twice**: once
with the plain CLIP-derived packet comparison ("clip_only" — what 1~4차
already reported) and once with the 5차 presence-calibration path
("calibrated" — only actually different from clip_only when a real
``PresenceCalibrator`` is configured with at least one backend that can
actually answer for the available evidence; with the default config it
degenerates to clip_only). Image tensors are **not** required for calibration
to run — image-free backends (``mock``/``gt``) work fine on items with no
``reconstructed_image`` (e.g. loaded via ``items_from_saved_packets``); only
image-based backends (``clip``/``owlv2``/``vqa``) then report themselves
unavailable per object.

Scope note
----------
This module answers "what would 1~4차's numbers look like if verified through
the 5차 calibration structure" — it does **not** independently prove the
calibration is more *accurate*, only that it is *computable* and *diffable*
against the original loop-internal numbers. Every report produced here is
tagged ``metric_role: "held_out"`` (see ``evaluators/packet_verifier.py``) —
this is meant to be a final remeasurement pass, not something a live
regeneration loop consumes. See docs/etri_strategy.md 5차 구현 결과 for what
is and is not claimed.

Three ways to build items (fidelity tradeoffs — read before picking one)
--------------------------------------------------------------------------
- :func:`items_from_temporal_records` — items straight out of a
  ``TemporalPipeline.run()`` call in the SAME process. Richest (real image +
  real packets), but requires re-running the pipeline right now.
- :func:`items_from_saved_packets` — packet JSON pairs already on disk
  (``utils/packet_io.py`` convention). No reconstruction needed, but also
  **no image tensor** — only image-free backends (``mock``/``gt``) can
  calibrate; ``owlv2``/``vqa``/``clip`` report themselves unavailable.
- :func:`items_from_recon_frame_dirs` — a completed run's saved
  ``extracted_frames/`` + ``recon_frames/`` PNGs (e.g.
  ``outputs/etri_video_eval_real_full_step50/baseline/<video>/``). Gives
  ``owlv2``/``vqa``/``clip`` a REAL reconstructed image without re-running any
  model inference (no diffusion/checkpoint load) — but no packet JSON was
  saved alongside that run, so both packets are freshly RE-EXTRACTED from the
  frames via ``SemanticPacketExtractor`` rather than loaded verbatim from
  whatever the original run's in-memory packets actually were. See that
  function's docstring for exactly what is and is not byte-identical to the
  original run.

Object vocabulary filter (optional, off by default)
------------------------------------------------------
:func:`remeasure` accepts an optional
``object_vocabulary_filter`` (``evaluators.object_vocabulary_filter.ObjectVocabularyFilter``)
that strips non-object tokens (count words like ``"one"``, action/event words
like ``"walking"``, scene words like ``"sidewalk"``) out of each item's
packets before they reach either the "clip_only" or "calibrated"
``PacketVerifier`` *and* before they feed PTC/SFR/SDI — both columns see the
identical filtered vocabulary, so the comparison stays fair. See
:func:`_vocabulary_filtered_packet` and ``evaluators/object_vocabulary_filter.py``'s
module docstring for the full rationale (including why this lives here and
not inside ``PacketVerifier``). Disabled (``None``) by default — existing
callers/reports are unaffected.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RemeasurementItem:
    """One frame/segment to re-measure.

    ``reconstructed_image`` is optional — present when built from live
    ``TemporalPipeline`` records (enables real presence-backend recalibration);
    absent when built purely from saved packet JSON (calibration then only
    runs for image-free backends, i.e. ``gt``/``mock``).
    """

    item_id: object
    reference_packet: Dict
    reconstructed_packet: Dict
    reconstructed_image: Optional[object] = None   # torch.Tensor, if available
    role: Optional[str] = None                     # "keyframe" | "inter", for PTC/SFR/SDI
    gt_metadata: Optional[Dict] = None


def items_from_temporal_records(records) -> List[RemeasurementItem]:
    """Build items from a ``TemporalPipeline.run()["records"]`` list.

    This is the rich path: ``.recon`` (the reconstructed frame tensor) is
    carried through, so real presence-backend recalibration is possible.
    """
    items: List[RemeasurementItem] = []
    for r in records:
        items.append(RemeasurementItem(
            item_id=getattr(r, "index", None),
            reference_packet=getattr(r, "orig_packet", None) or {},
            reconstructed_packet=getattr(r, "recon_packet", None) or {},
            reconstructed_image=getattr(r, "recon", None),
            role=getattr(r, "role", None),
        ))
    return items


def convert_gt_to_presence(gt: Dict) -> Dict:
    """Convert a ``data/etri_video_eval/gt/<video>.json`` segment-level GT file
    (``{"n_frames": int, "segments": [{"start_frame", "end_frame", "objects":
    [{"label", "presence"}]}]}``) into the per-item ``{"frame_00000":
    {object_name: bool}, ...}`` mapping the ``gt`` presence backend expects.

    Item ids follow the still-image packet stem convention ``frame_{i:05d}``
    — this matches :func:`items_from_saved_packets`'s ``item_id`` when packets
    were saved with that naming, and is one of the key spellings
    :func:`_lookup_gt_metadata` tries when items use a different id
    convention (e.g. :func:`items_from_recon_frame_dirs`'s plain integer
    index). This is an input-format bridge only — it does not itself judge
    anything.
    """
    n_frames = int(gt.get("n_frames", 0))
    labels = set()
    for seg in gt.get("segments", []):
        for obj in seg.get("objects", []):
            labels.add(str(obj.get("label")))
    presence = {}
    for i in range(n_frames):
        frame_presence = {label: False for label in sorted(labels)}
        for seg in gt.get("segments", []):
            if int(seg.get("start_frame", 0)) <= i <= int(seg.get("end_frame", -1)):
                for obj in seg.get("objects", []):
                    if str(obj.get("presence", "visible")) == "visible":
                        frame_presence[str(obj.get("label"))] = True
        presence[f"frame_{i:05d}"] = frame_presence
    return presence


def looks_like_segment_level_gt(data: Dict) -> bool:
    """True if *data* looks like the raw ``data/etri_video_eval/gt/<video>.json``
    segment-level format (needs :func:`convert_gt_to_presence` first) rather
    than an already-converted ``{item_id: {object_name: bool}}`` mapping."""
    return isinstance(data, dict) and "segments" in data and "n_frames" in data


def _lookup_gt_metadata(gt_metadata_by_id: Dict, item_id) -> Optional[Dict]:
    """Look up *item_id*'s GT metadata, tolerating the handful of key
    spellings different callers/files use for "the same" frame:

    - the exact ``item_id`` (whatever type/value it is);
    - its plain string form (JSON object keys are always strings, so a
      hand-written ``{"0": {...}}`` file parses to key ``"0"``, not int ``0``);
    - the zero-padded ``frame_{idx:05d}`` convention
      :func:`convert_gt_to_presence` emits and ``items_from_saved_packets``
      packet stems typically use, when ``item_id`` is (or looks like) a bare
      integer index — e.g. :func:`items_from_recon_frame_dirs`'s ``item_id``.
    - the bare integer index, when ``item_id`` is a ``"frame_00000"``-style
      string (the reverse direction).

    Without this, a GT file produced for one item-id convention (e.g.
    ``scripts/run_etri_video_eval.py``'s ``gt_presence.json``, keyed
    ``"frame_00000"``) would silently fail to match items built with a
    different convention (e.g. plain integer indices), and the ``gt`` backend
    would report itself unavailable for every object — not a crash, but a
    silent no-op that looks like "GT calibration ran" without actually doing
    anything. Returns ``None`` (no match under any spelling) if nothing
    matches — callers pass this straight through as ``gt_metadata``, and
    ``GtPresenceBackend`` already raises cleanly for ``None``/missing entries.
    """
    if item_id in gt_metadata_by_id:
        return gt_metadata_by_id[item_id]

    candidates = [str(item_id)]
    if isinstance(item_id, str) and item_id.startswith("frame_"):
        try:
            idx = int(item_id[len("frame_"):])
            candidates.append(str(idx))
        except ValueError:
            pass
    else:
        try:
            idx = int(item_id)
            candidates.append(f"frame_{idx:05d}")
        except (TypeError, ValueError):
            pass

    for key in candidates:
        if key in gt_metadata_by_id:
            return gt_metadata_by_id[key]
    return None


def items_from_saved_packets(
    pairs: Sequence[Tuple],
    gt_metadata_by_id: Optional[Dict] = None,
) -> List[RemeasurementItem]:
    """Build items from previously-saved packet JSON pairs (no image tensors).

    Parameters
    ----------
    pairs:
        Iterable of ``(item_id, orig_packet_path, recon_packet_path)`` or
        ``(item_id, orig_packet_path, recon_packet_path, role)`` tuples — the
        on-disk format ``utils/packet_io.py`` already writes for the
        still-image pipeline's ``packet_dir``.
    gt_metadata_by_id:
        Optional ``{item_id: gt_metadata_dict}`` for the ``gt`` presence
        backend — looked up per item via :func:`_lookup_gt_metadata`, so a
        handful of common key-spelling mismatches (str vs int, bare index vs
        ``frame_00000``) are tolerated rather than silently missing.
    """
    from sgdjscc_lab.utils.packet_io import load_packet

    gt_metadata_by_id = gt_metadata_by_id or {}
    items: List[RemeasurementItem] = []
    for entry in pairs:
        item_id, orig_path, recon_path = entry[0], entry[1], entry[2]
        role = entry[3] if len(entry) > 3 else None
        items.append(RemeasurementItem(
            item_id=item_id,
            reference_packet=load_packet(orig_path),
            reconstructed_packet=load_packet(recon_path),
            reconstructed_image=None,
            role=role,
            gt_metadata=_lookup_gt_metadata(gt_metadata_by_id, item_id),
        ))
    return items


def items_from_recon_frame_dirs(
    extracted_frames_dir,
    recon_frames_dir,
    packet_extractor,
    captions: Optional[Sequence[Optional[str]]] = None,
    roles: Optional[Sequence[Optional[str]]] = None,
    gt_metadata_by_id: Optional[Dict] = None,
) -> List[RemeasurementItem]:
    """Build items directly from a completed run's saved frame folders —
    ``<run_dir>/extracted_frames/...`` (the original input frames) and
    ``<run_dir>/recon_frames/...`` (the ACTUAL reconstruction that run
    produced) — with **no reconstruction re-run and no model inference**
    beyond cheap packet extraction. This is the path for reusing an
    already-completed real-model batch (e.g.
    ``outputs/etri_video_eval_real_full_step50/baseline/<video>/``) with
    image-based presence backends (``owlv2``/``vqa``/``clip``), which
    :func:`items_from_saved_packets` cannot do (no image tensor there) and
    :func:`items_from_temporal_records`/``--from-run`` cannot do without
    repeating the original (possibly GPU-heavy, possibly stochastic)
    reconstruction.

    Fidelity — what IS and is NOT byte-identical to the original run
    --------------------------------------------------------------------
    - ``reconstructed_image`` — byte-identical: these are the literal PNG
      bytes the original run wrote to ``recon_frames/``, just decoded back to
      a tensor. No diffusion/JSCC/channel simulation happens here.
    - ``reference_packet`` / ``reconstructed_packet`` — **NOT** loaded from
      the original run (no packet JSON was ever saved alongside
      ``evaluate_video.py``'s ``baseline`` stage — only
      ``temporal_frames.csv``/``temporal_metrics.csv``/``segments.json``,
      none of which carry the packet's ``objects``/``relations``/etc.).
      Both packets are freshly RE-EXTRACTED here from the saved frames via
      *packet_extractor* (``guidance/semantic_packet_extractor.py``,
      typically CLIP-based object/scene detection + the supplied caption
      string — cheap, no diffusion). Given the same CLIP weights and the
      same caption per frame this is deterministic and a faithful
      re-derivation, but it is not literally "the packet the original run
      held in memory". If *packet_extractor* has no ``text_extractor``
      configured (the common case here, to avoid a BLIP-2 load), the
      reconstructed-frame packet's caption is empty — its object list then
      comes from the object extractor alone, without caption-noun folding
      (see ``SemanticPacketExtractor``'s ``caption_objects`` option). If your
      use case needs recon-frame captions too, pass a *packet_extractor*
      built with a real ``text_extractor``.
    - If you need byte-exact PACKET reuse as well (not just images), the
      structural fix is to make a future run also dump packets via
      ``utils/packet_io.py`` (e.g. wiring ``save_packet()`` into
      ``pipelines/eval_pipeline.py``/``video/temporal_pipeline.py``) and then
      use :func:`items_from_saved_packets` instead — that is a larger change
      than this preparation task's scope and is not implemented here.

    Parameters
    ----------
    extracted_frames_dir, recon_frames_dir:
        Directories of PNG (or other ``io.list_image_files``-supported)
        images, sorted lexicographically — this matches both
        ``utils/video_io.py``'s ``frame_%05d.png`` naming and
        ``video/temporal_pipeline.py``'s ``recon_%05d.png`` naming, so index
        ``i`` in one directory corresponds to index ``i`` in the other.
        Raises ``ValueError`` if the two directories don't have the same
        frame count (a real mismatch, not something to silently truncate).
    packet_extractor:
        A ``guidance.semantic_packet_extractor.SemanticPacketExtractor``
        instance (or duck-typed equivalent exposing
        ``.extract(image, frame_id=..., caption=...) -> dict``). Required —
        there is no free-standing default because the caller controls
        whether a CLIP evaluator (and optionally a text extractor) is
        attached.
    captions:
        Optional per-frame caption strings for the REFERENCE (original) frame
        only, same length/order as the frame directories (``None`` entries
        fall back to ``packet_extractor``'s own caption behaviour, i.e. empty
        unless it has a text extractor). Reconstructed-frame captions are
        always passed as ``None`` (see fidelity note above).
    roles:
        Optional per-frame ``"keyframe"``/``"inter"`` role strings (e.g. read
        from that run's ``temporal_frames.csv``), enabling SDI's
        distance-from-keyframe computation. Without this, ``role`` stays
        ``None`` for every item and ``evaluate_sequence()`` degrades SDI to
        ``None`` (no "keyframe" role present) rather than raising.
    gt_metadata_by_id:
        Optional ``{item_id: gt_metadata_dict}`` for the ``gt`` presence
        backend. ``item_id`` here is a plain integer frame index, but the
        lookup goes through :func:`_lookup_gt_metadata`, so a
        ``"frame_00000"``-keyed file (e.g. ``convert_gt_to_presence()``'s
        output, or ``scripts/run_etri_video_eval.py``'s saved
        ``gt_presence.json``) is also accepted directly — not just a bare
        ``{0: {...}, 1: {...}}`` mapping.
    """
    from sgdjscc_lab.io import list_image_files, load_image_as_tensor

    orig_files = list_image_files(extracted_frames_dir)
    recon_files = list_image_files(recon_frames_dir)
    if len(orig_files) != len(recon_files):
        raise ValueError(
            f"Frame count mismatch: {len(orig_files)} extracted frame(s) under "
            f"{extracted_frames_dir} vs {len(recon_files)} reconstructed frame(s) under "
            f"{recon_frames_dir} — these must be the matching extracted/recon pair from "
            f"the same run."
        )

    gt_metadata_by_id = gt_metadata_by_id or {}
    items: List[RemeasurementItem] = []
    for idx, (orig_path, recon_path) in enumerate(zip(orig_files, recon_files)):
        orig_image = load_image_as_tensor(orig_path)
        recon_image = load_image_as_tensor(recon_path)
        caption = captions[idx] if captions is not None and idx < len(captions) else None
        role = roles[idx] if roles is not None and idx < len(roles) else None

        reference_packet = packet_extractor.extract(orig_image, frame_id=f"frame_{idx:05d}", caption=caption)
        reconstructed_packet = packet_extractor.extract(recon_image, frame_id=f"recon_{idx:05d}", caption=None)

        items.append(RemeasurementItem(
            item_id=idx,
            reference_packet=reference_packet,
            reconstructed_packet=reconstructed_packet,
            reconstructed_image=recon_image,
            role=role,
            gt_metadata=_lookup_gt_metadata(gt_metadata_by_id, idx),
        ))
    return items


def _calibrated_recon_packet(reconstructed_packet: Dict, calibrated_report: Dict) -> Dict:
    """Derive a "calibrated" reconstructed packet whose ``objects`` list
    reflects the presence-calibration corrections, so PTC/SFR/SDI (which
    re-derive their own object-set comparison from the packets) can be
    recomputed against the calibrated view instead of the raw one."""
    objs = set(reconstructed_packet.get("objects") or [])
    raw = calibrated_report.get("raw_clip_result") or {}

    still_additional = set(calibrated_report.get("additional_objects") or [])
    removed_additional = set(raw.get("additional_objects") or []) - still_additional
    objs -= removed_additional   # calibration says these aren't really hallucinated → drop

    still_missing = set(calibrated_report.get("missing_objects") or [])
    recovered_missing = set(raw.get("missing_objects") or []) - still_missing
    objs |= recovered_missing    # calibration says these ARE actually present → add back

    out = dict(reconstructed_packet)
    out["objects"] = sorted(objs)
    return out


def _aggregate(rows: List[Dict], temporal_metrics: Dict) -> Dict:
    n = len(rows)
    out = {
        "n_items": n,
        "mean_severity": (sum(r["severity"] for r in rows) / n) if n else None,
        "total_missing_objects": sum(r["missing_object_count"] for r in rows),
        "total_additional_objects": sum(r["additional_object_count"] for r in rows),
    }
    out.update(temporal_metrics)
    return out


def _delta(clip_metrics: Dict, calibrated_metrics: Dict) -> Dict:
    keys = sorted(set(clip_metrics) | set(calibrated_metrics))
    out: Dict = {}
    for k in keys:
        a, b = clip_metrics.get(k), calibrated_metrics.get(k)
        out[f"{k}_clip_only"] = a
        out[f"{k}_calibrated"] = b
        out[f"{k}_diff"] = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
    out["note"] = (
        "5차 보강(calibrated) 결과가 1~4차 CLIP-only(clip_only) 결과와 얼마나 다른지 보여주는 "
        "구조적 diff다 — presence backend 자체의 정확도가 검증된 결과가 아니다. 실제 OWLv2/VQA "
        "가중치 통합 후 재해석이 필요하다(docs/etri_strategy.md 5차 구현 결과 참조)."
    )
    return out


def _vocabulary_filtered_packet(packet: Dict, gt_metadata: Optional[Dict], vocab_filter) -> Tuple[Dict, List[str]]:
    """Return ``(packet_for_metrics, removed_objects)``.

    When *vocab_filter* is ``None`` (the default — filtering disabled),
    returns *packet* unchanged (same object, no copy, no removed objects) —
    this is what keeps the default remeasurement path byte-identical.
    Otherwise returns a shallow copy of *packet* whose ``objects`` field has
    been reduced to :meth:`ObjectVocabularyFilter.filter_objects`'s ``kept``
    list.
    """
    if vocab_filter is None:
        return packet, []
    objs = list(packet.get("objects") or [])
    kept, removed = vocab_filter.filter_objects(objs, gt_metadata=gt_metadata)
    if not removed:
        return packet, []
    out = dict(packet)
    out["objects"] = kept
    return out, removed


def remeasure(items: List[RemeasurementItem], presence_calibrator=None, object_vocabulary_filter=None) -> Dict:
    """Recompute clip-only vs calibrated packet-verifier reports + PTC/SFR/SDI.

    Parameters
    ----------
    items:
        From :func:`items_from_temporal_records` or :func:`items_from_saved_packets`.
    presence_calibrator:
        Optional ``presence_calibration.PresenceCalibrator``. When ``None``
        (default), the "calibrated" column is identical to "clip_only" — there
        is nothing to calibrate against, so the diff is exactly zero
        everywhere (a useful sanity check that the pipeline is wired
        correctly before enabling real calibration).
    object_vocabulary_filter:
        Optional ``evaluators.object_vocabulary_filter.ObjectVocabularyFilter``
        (ETRI 5차 remeasurement follow-up). When given, each item's
        reference/reconstructed packet ``objects`` list is filtered — via
        each item's own ``gt_metadata`` when available, else the excluded
        count/action/scene-term heuristic — BEFORE it reaches either the
        "clip_only" or "calibrated" ``PacketVerifier``, and before it feeds
        ``temporal_consistency.evaluate_sequence()``'s PTC/SFR/SDI. Both
        columns therefore see the identical filtered vocabulary — a fair
        comparison, not just a clip_only-side adjustment. ``None`` (default)
        is a strict no-op: byte-identical rows/metrics to before this
        parameter existed, and no ``filtered_objects``/
        ``filtered_missing_objects``/``filtered_additional_objects`` keys are
        added to any row.

    Returns
    -------
    dict with ``clip_only`` / ``calibrated`` (each ``{"rows": [...], "metrics": {...}}``)
    and ``metric_delta``.
    """
    from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier, METRIC_ROLE_HELD_OUT
    from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence

    clip_verifier = PacketVerifier(metric_role=METRIC_ROLE_HELD_OUT)
    calibrated_verifier = (
        PacketVerifier(presence_calibrator=presence_calibrator, metric_role=METRIC_ROLE_HELD_OUT)
        if presence_calibrator is not None else None
    )

    clip_rows: List[Dict] = []
    calibrated_rows: List[Dict] = []
    clip_sequence: List[Dict] = []
    calibrated_sequence: List[Dict] = []

    for item in items:
        ref_packet, ref_removed = _vocabulary_filtered_packet(
            item.reference_packet, item.gt_metadata, object_vocabulary_filter)
        recon_packet, recon_removed = _vocabulary_filtered_packet(
            item.reconstructed_packet, item.gt_metadata, object_vocabulary_filter)

        filtered_objects = filtered_missing_objects = filtered_additional_objects = None
        if object_vocabulary_filter is not None:
            raw_missing = set(item.reference_packet.get("objects") or []) - set(item.reconstructed_packet.get("objects") or [])
            raw_additional = set(item.reconstructed_packet.get("objects") or []) - set(item.reference_packet.get("objects") or [])
            filtered_objects = sorted(set(ref_removed) | set(recon_removed))
            filtered_missing_objects = sorted(set(ref_removed) & raw_missing)
            filtered_additional_objects = sorted(set(recon_removed) & raw_additional)

        clip_report = clip_verifier.verify(
            ref_packet, recon_packet, item_id=item.item_id,
        )
        if object_vocabulary_filter is not None:
            clip_report["filtered_objects"] = filtered_objects
            clip_report["filtered_missing_objects"] = filtered_missing_objects
            clip_report["filtered_additional_objects"] = filtered_additional_objects
        clip_rows.append(clip_report)
        clip_sequence.append({
            "role": item.role, "orig_packet": ref_packet,
            "recon_packet": recon_packet,
        })

        if calibrated_verifier is not None:
            # reconstructed_image/gt_metadata may both be None here (e.g.
            # --from-packets items have no pixels) — that's fine: image-free
            # backends (mock/gt) still run, only image-based backends
            # (clip/owlv2/vqa) report themselves unavailable per object. See
            # PacketVerifier.verify()'s docstring.
            calibrated_report = calibrated_verifier.verify(
                ref_packet, recon_packet, item_id=item.item_id,
                reconstructed_image=item.reconstructed_image, gt_metadata=item.gt_metadata,
            )
            calibrated_recon = _calibrated_recon_packet(recon_packet, calibrated_report)
        else:
            calibrated_report = dict(clip_report)
            calibrated_report["metric_role"] = METRIC_ROLE_HELD_OUT
            calibrated_recon = recon_packet
        if object_vocabulary_filter is not None:
            calibrated_report["filtered_objects"] = filtered_objects
            calibrated_report["filtered_missing_objects"] = filtered_missing_objects
            calibrated_report["filtered_additional_objects"] = filtered_additional_objects
        calibrated_rows.append(calibrated_report)
        calibrated_sequence.append({
            "role": item.role, "orig_packet": ref_packet,
            "recon_packet": calibrated_recon,
        })

    clip_temporal = evaluate_sequence(clip_sequence)
    calibrated_temporal = evaluate_sequence(calibrated_sequence)

    clip_metrics = _aggregate(clip_rows, clip_temporal)
    calibrated_metrics = _aggregate(calibrated_rows, calibrated_temporal)

    return {
        "clip_only": {"rows": clip_rows, "metrics": clip_metrics},
        "calibrated": {"rows": calibrated_rows, "metrics": calibrated_metrics},
        "metric_delta": _delta(clip_metrics, calibrated_metrics),
    }


_ROW_CSV_FIELDS = (
    "item_id", "object_match_rate", "missing_objects", "additional_objects",
    "missing_object_count", "additional_object_count", "relation_error_count",
    "attribute_error_count", "scene_match", "severity", "metric_role",
)

# Debug-only columns populated when ``remeasure(object_vocabulary_filter=...)``
# was actually used — added to the CSV header ONLY if at least one row carries
# them, so a filter-disabled (default) run's CSV schema is byte-identical to
# before this feature existed (see module docstring's "Scope" note).
_FILTERED_ROW_CSV_FIELDS = ("filtered_objects", "filtered_missing_objects", "filtered_additional_objects")


def _csv_safe(row: Dict, fields) -> Dict:
    out = {}
    for k in fields:
        v = row.get(k)
        out[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
    return out


def _row_csv_fields(rows: List[Dict]) -> List[str]:
    fields = list(_ROW_CSV_FIELDS)
    if any(k in row for row in rows for k in _FILTERED_ROW_CSV_FIELDS):
        fields += list(_FILTERED_ROW_CSV_FIELDS)
    return fields


def _write_rows_csv(rows: List[Dict], path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = _row_csv_fields(rows)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(_csv_safe(row, fields))


def _write_flat_csv(flat: Dict, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat.keys()))
        w.writeheader()
        w.writerow(flat)


def write_remeasurement(
    result: Dict,
    clip_only_json: Optional[str] = None,
    clip_only_csv: Optional[str] = None,
    calibrated_json: Optional[str] = None,
    calibrated_csv: Optional[str] = None,
    metric_delta_json: Optional[str] = None,
    metric_delta_csv: Optional[str] = None,
) -> None:
    """Persist :func:`remeasure`'s output to the requested JSON/CSV paths.

    Each path is optional and independently skipped when ``None`` — callers
    pick whichever artefacts they need.
    """
    if clip_only_json:
        p = Path(clip_only_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(result["clip_only"], fh, indent=2, ensure_ascii=False)
        logger.info("clip_only metrics → %s", p)
    if clip_only_csv:
        _write_rows_csv(result["clip_only"]["rows"], clip_only_csv)
        logger.info("clip_only rows CSV → %s", clip_only_csv)

    if calibrated_json:
        p = Path(calibrated_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(result["calibrated"], fh, indent=2, ensure_ascii=False)
        logger.info("calibrated metrics → %s", p)
    if calibrated_csv:
        _write_rows_csv(result["calibrated"]["rows"], calibrated_csv)
        logger.info("calibrated rows CSV → %s", calibrated_csv)

    if metric_delta_json:
        p = Path(metric_delta_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(result["metric_delta"], fh, indent=2, ensure_ascii=False)
        logger.info("metric_delta → %s", p)
    if metric_delta_csv:
        _write_flat_csv(result["metric_delta"], metric_delta_csv)
        logger.info("metric_delta CSV → %s", metric_delta_csv)
