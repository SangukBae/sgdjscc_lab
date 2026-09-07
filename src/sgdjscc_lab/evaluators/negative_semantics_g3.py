"""Pure event-cluster metrics for G3 append-only versus revocable RSM."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence


G3_EVAL_ARMS = ("no_rsm", "append_only_rsm", "revocable_rsm")


def _quantile(values, probability):
    values = sorted(float(x) for x in values)
    if not values:
        raise ValueError("cannot take quantile of empty values")
    return values[min(len(values) - 1, int(math.floor(probability * (len(values) - 1))))]


def _validate(rows, arms):
    if {str(row["arm"]) for row in rows} != set(arms):
        raise ValueError("G3 rows do not contain exactly the declared arms")
    grids = defaultdict(set)
    for row in rows:
        for field in ("score", "identity_similarity"):
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"non-finite G3 {field}")
        if not isinstance(row.get("gt_present"), bool):
            raise ValueError("G3 gt_present must be bool")
        key = (str(row["event_id"]), int(row["frame_index"]))
        arm = str(row["arm"])
        if key in grids[arm]:
            raise ValueError(f"duplicate G3 row: {arm}/{key}")
        grids[arm].add(key)
    reference = grids[arms[0]]
    if not reference or any(grids[arm] != reference for arm in arms[1:]):
        raise ValueError("G3 arm grids differ")


def _event_metrics(rows, threshold):
    by_event = defaultdict(list)
    for row in rows:
        by_event[str(row["event_id"])].append(row)
    ghost_curves = {}
    identity = {}
    for event_id, values in by_event.items():
        ordered = sorted(values, key=lambda row: int(row["frame_index"]))
        event_type = str(ordered[0]["event_type"])
        boundary = int(ordered[0]["event_frame"])
        if any(str(row["event_type"]) != event_type or int(row["event_frame"]) != boundary
               for row in ordered):
            raise ValueError(f"inconsistent G3 event metadata: {event_id}")
        if event_type == "EXIT":
            after = [row for row in ordered if int(row["frame_index"]) >= boundary]
            ghost_curves[event_id] = [float(row["score"]) >= threshold for row in after]
        present = [float(row["identity_similarity"]) for row in ordered if row["gt_present"]]
        if present:
            identity[event_id] = sum(present) / len(present)
    max_horizon = max((len(curve) for curve in ghost_curves.values()), default=0)
    survival = []
    for lag in range(max_horizon):
        available = [curve[lag] for curve in ghost_curves.values() if lag < len(curve)]
        survival.append(sum(available) / len(available))
    return {
        "ghost_survival_curve": survival,
        "ghost_survival_auc": sum(survival) / len(survival) if survival else None,
        "exit_event_count": len(ghost_curves),
        "identity_by_event": identity,
        "mean_identity_similarity": (
            sum(identity.values()) / len(identity) if identity else None
        ),
    }


def _cluster_bootstrap(metric_by_arm, arms, iterations, seed):
    event_ids = sorted(set.intersection(*[
        set(metric_by_arm[arm]["identity_by_event"]) for arm in arms
    ]))
    if not event_ids:
        return {"append_minus_no_lower_one_sided_95": None,
                "revocable_retention_lower_one_sided_95": None}
    rng = random.Random(seed)
    append_deltas = []
    retention = []
    for _ in range(iterations):
        chosen = [rng.choice(event_ids) for _ in event_ids]
        no = sum(metric_by_arm["no_rsm"]["identity_by_event"][e] for e in chosen) / len(chosen)
        append = sum(metric_by_arm["append_only_rsm"]["identity_by_event"][e] for e in chosen) / len(chosen)
        rev = sum(metric_by_arm["revocable_rsm"]["identity_by_event"][e] for e in chosen) / len(chosen)
        append_deltas.append(append - no)
        if append > 0:
            retention.append(rev / append)
    return {
        "append_minus_no_lower_one_sided_95": _quantile(append_deltas, 0.05),
        "revocable_retention_lower_one_sided_95": (
            _quantile(retention, 0.05) if retention else None
        ),
        "cluster_unit": "event",
        "iterations": iterations,
    }


def summarize_g3(
    detection_rows: Iterable[Mapping[str, Any]], *, threshold: float = 0.2,
    arms: Sequence[str] = G3_EVAL_ARMS, bootstrap_iterations: int = 5000,
    ghost_auc_relative_reduction_min: float = 0.25,
    identity_retention_min: float = 0.90,
) -> Dict[str, Any]:
    rows = [dict(row) for row in detection_rows]
    arms = tuple(str(arm) for arm in arms)
    if arms != G3_EVAL_ARMS:
        raise ValueError(f"G3 arm order must be {G3_EVAL_ARMS}")
    _validate(rows, arms)
    metrics = {
        arm: _event_metrics([row for row in rows if row["arm"] == arm], float(threshold))
        for arm in arms
    }
    append_auc = metrics["append_only_rsm"]["ghost_survival_auc"]
    revocable_auc = metrics["revocable_rsm"]["ghost_survival_auc"]
    ghost_reduction = (
        (append_auc - revocable_auc) / append_auc
        if append_auc not in (None, 0.0) and revocable_auc is not None else None
    )
    bootstrap = _cluster_bootstrap(metrics, arms, int(bootstrap_iterations), 37013)
    checks = {
        "append_identity_improves_over_no_rsm": (
            bootstrap["append_minus_no_lower_one_sided_95"] is not None
            and bootstrap["append_minus_no_lower_one_sided_95"] > 0.0
        ),
        "append_has_measurable_ghost": append_auc is not None and append_auc > 0.0,
        "revocation_reduces_ghost_auc": (
            ghost_reduction is not None
            and ghost_reduction >= float(ghost_auc_relative_reduction_min)
        ),
        "revocable_retains_identity": (
            bootstrap["revocable_retention_lower_one_sided_95"] is not None
            and bootstrap["revocable_retention_lower_one_sided_95"]
            >= float(identity_retention_min)
        ),
    }
    return {
        "threshold": float(threshold),
        "arms": list(arms),
        "metrics": metrics,
        "ghost_auc_relative_reduction": ghost_reduction,
        "identity_bootstrap": bootstrap,
        "provisional_gate_checks": checks,
        "provisional_gate_status": "PASSED" if all(checks.values()) else "NOT_PASSED",
    }
