"""pipelines/rate_reliability_report.py – rate vs semantic-reliability trade-off
report (ETRI 6차, step 12).

Combines ``pipelines/transmission_accounting.py``'s bit/channel-symbol
accounting summary with the existing temporal-consistency metrics
(``PTC``/``SFR``/``SDI`` — ``evaluators/temporal_consistency.py``) and, when
available, the packet-verifier ``mean_severity`` (``pipelines/packet_verification.py``
/ ``pipelines/heldout_remeasurement.py``) into one row: "at this bit/symbol
rate, how reliable was the reconstructed semantics?"

Scope note
----------
This is a PoC trade-off report, not a final rate-distortion/rate-reliability
curve — see ``accounting/bit_accounting.py``'s module docstring for what is
exact vs proxy in the rate numbers, and docs/etri_strategy.md 6차 구현 결과 for
the overall scope. A single run/config produces exactly one point; the
``curve_csv`` append/merge helpers below let multiple runs (different SNR,
policy, or config) accumulate into one comparable curve over time.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_CURVE_FIELDS = (
    "label", "bits_per_frame", "symbols_per_frame", "bit_reduction", "symbol_reduction",
    "semantic_unit_reduction", "ptc", "sfr", "sdi", "mean_severity",
    "n_generate", "n_reused", "n_recompute", "baseline", "proxy_fraction",
)


def build_rate_reliability_row(
    accounting_summary: Dict,
    temporal_metrics: Optional[Dict] = None,
    mean_severity: Optional[float] = None,
    label: Optional[str] = None,
) -> Dict:
    """Build one rate/reliability trade-off row.

    Parameters
    ----------
    accounting_summary:
        The ``"summary"`` dict from ``transmission_accounting.account_transmission()``.
    temporal_metrics:
        A dict containing ``ptc``/``sfr``/``sdi`` (e.g.
        ``evaluators/temporal_consistency.py::evaluate_sequence()``'s output,
        or a loaded ``temporal_metrics.csv`` row). Optional — missing keys
        become ``None``.
    mean_severity:
        Optional packet-verifier severity (``pipelines/packet_verification.py``
        row aggregate, or ``pipelines/heldout_remeasurement.py``'s
        ``clip_only``/``calibrated`` metrics' ``mean_severity``).
    label:
        Optional identifier for this row (e.g. config/policy name, SNR) —
        lets :func:`append_rate_reliability_row` accumulate a curve across
        multiple runs.
    """
    temporal_metrics = temporal_metrics or {}
    n_frames = accounting_summary.get("n_frames") or 0
    total_bits = accounting_summary.get("total_bits")
    total_symbols = accounting_summary.get("total_channel_symbols")

    return {
        "label": label,
        "bits_per_frame": (float(total_bits) / n_frames) if (total_bits is not None and n_frames) else None,
        "symbols_per_frame": (float(total_symbols) / n_frames) if (total_symbols is not None and n_frames) else None,
        "bit_reduction": accounting_summary.get("bit_reduction"),
        "symbol_reduction": accounting_summary.get("symbol_reduction"),
        "semantic_unit_reduction": accounting_summary.get("semantic_unit_reduction"),
        "ptc": temporal_metrics.get("ptc"),
        "sfr": temporal_metrics.get("sfr"),
        "sdi": temporal_metrics.get("sdi"),
        "mean_severity": mean_severity,
        "n_generate": accounting_summary.get("n_generate"),
        "n_reused": accounting_summary.get("n_reused"),
        "n_recompute": accounting_summary.get("n_recompute"),
        "baseline": accounting_summary.get("baseline"),
        "proxy_fraction": accounting_summary.get("proxy_fraction"),
    }


def write_rate_reliability_summary(row: Dict, output_json: str) -> None:
    """Persist one rate/reliability row as JSON (with the PoC scope note)."""
    p = Path(output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(row)
    payload["note"] = (
        "PoC rate/reliability trade-off point — bits/symbols are accounting-PoC "
        "numbers (see accounting/bit_accounting.py), not a verified bitstream/CBR "
        "measurement. See docs/etri_strategy.md 6차 구현 결과."
    )
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Rate/reliability summary → %s", p)


def append_rate_reliability_row(row: Dict, curve_csv: str) -> None:
    """Append one row to a shared rate/reliability curve CSV.

    Creates the file (with header) if it doesn't exist yet; subsequent calls
    (e.g. from different configs/policies/SNRs) each add one more point,
    building up the curve across independent runs.
    """
    p = Path(curve_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    is_new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(_CURVE_FIELDS))
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in _CURVE_FIELDS})
    logger.info("Rate/reliability curve row appended → %s", p)


def merge_rate_reliability_curves(paths: Sequence[str], output_path: str) -> int:
    """Concatenate multiple existing curve CSVs into one (e.g. results
    gathered from different machines/runs), de-duplicating by ``label`` when
    present (last occurrence wins). Returns the number of rows written.
    """
    rows_by_label: Dict[object, Dict] = {}
    ordered_keys: List[object] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            logger.warning("merge_rate_reliability_curves: %s not found, skipping.", p)
            continue
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = row.get("label") or object()
                if key not in rows_by_label:
                    ordered_keys.append(key)
                rows_by_label[key] = row

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(_CURVE_FIELDS))
        w.writeheader()
        for key in ordered_keys:
            w.writerow({k: rows_by_label[key].get(k) for k in _CURVE_FIELDS})
    logger.info("Merged %d rate/reliability curve row(s) → %s", len(ordered_keys), out_p)
    return len(ordered_keys)
