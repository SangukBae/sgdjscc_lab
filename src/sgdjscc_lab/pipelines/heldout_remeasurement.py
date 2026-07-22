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
        Optional ``{item_id: gt_metadata_dict}`` for the ``gt`` presence backend.
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
            gt_metadata=gt_metadata_by_id.get(item_id),
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


def remeasure(items: List[RemeasurementItem], presence_calibrator=None) -> Dict:
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
        clip_report = clip_verifier.verify(
            item.reference_packet, item.reconstructed_packet, item_id=item.item_id,
        )
        clip_rows.append(clip_report)
        clip_sequence.append({
            "role": item.role, "orig_packet": item.reference_packet,
            "recon_packet": item.reconstructed_packet,
        })

        if calibrated_verifier is not None:
            # reconstructed_image/gt_metadata may both be None here (e.g.
            # --from-packets items have no pixels) — that's fine: image-free
            # backends (mock/gt) still run, only image-based backends
            # (clip/owlv2/vqa) report themselves unavailable per object. See
            # PacketVerifier.verify()'s docstring.
            calibrated_report = calibrated_verifier.verify(
                item.reference_packet, item.reconstructed_packet, item_id=item.item_id,
                reconstructed_image=item.reconstructed_image, gt_metadata=item.gt_metadata,
            )
            calibrated_recon = _calibrated_recon_packet(item.reconstructed_packet, calibrated_report)
        else:
            calibrated_report = dict(clip_report)
            calibrated_report["metric_role"] = METRIC_ROLE_HELD_OUT
            calibrated_recon = item.reconstructed_packet
        calibrated_rows.append(calibrated_report)
        calibrated_sequence.append({
            "role": item.role, "orig_packet": item.reference_packet,
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


def _csv_safe(row: Dict, fields) -> Dict:
    out = {}
    for k in fields:
        v = row.get(k)
        out[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
    return out


def _write_rows_csv(rows: List[Dict], path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(_ROW_CSV_FIELDS))
        w.writeheader()
        for row in rows:
            w.writerow(_csv_safe(row, _ROW_CSV_FIELDS))


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
