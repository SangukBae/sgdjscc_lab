#!/usr/bin/env python3
"""Read-only audit of a completed negative-semantics G1 run.

Mirrors the style of ``audit_negative_semantics_g0.py``: every check is a
structural or hash-based fact about files already on disk. This script never
launches reconstruction, never reads held-out media, and never mutates the
run directory it audits.

It exists to separate two questions that are easy to conflate for a frozen
research artifact, and gives each its own CLI switch and report field so a
caller cannot accidentally gate on the wrong one:

1. Did the *runner* complete cleanly? (child-run count, failed-pair count,
   frame counts, resume/signature bookkeeping, held-out non-access.)
   -- ``--require-runner-complete``, report field ``runner_gate_status``.
2. Does the result survive an *honest, effective-seed* reapplication of the
   frozen gate -- not just the declared-seed ``gate_status`` in
   ``g1_summary.json``, which counts a deterministic decoder's triplicated
   seeds as three independent samples?
   -- ``--require-scientific-gate``, report field ``scientific_gate_status``.
   This is read from an already-computed v1.2 amendment derivation
   (``scripts/derive_negative_semantics_g1_v1_2.py``); it is never
   recomputed here, and is reported as ``UNKNOWN_V1_2_NOT_DERIVED`` if that
   derivation has not been produced at ``--v1-2-derived-summary``.

``--require-pass`` is kept only as a **deprecated alias for
--require-runner-complete** (identical check, unchanged exit behavior) --
existing callers keep working, but it prints a deprecation warning to
stderr, since a caller who actually wanted "is the phenomenon scientifically
established" and typed ``--require-pass`` expecting that would have been
silently getting only the weaker, structural check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sgdjscc_lab.evaluators.negative_semantics import effective_seed_groups  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _check(check_id: str, passed: bool, **evidence: Any) -> Dict[str, Any]:
    return {"check_id": check_id, "passed": bool(passed), "evidence": evidence}


DEFAULT_V1_2_DERIVED_SUMMARY = (
    ROOT / "outputs/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2/g1_summary_v1_2.json"
)


def _scientific_gate_status(v1_2_derived_summary_path: Path) -> Dict[str, Any]:
    """Read (never recompute) the v1.2 amendment's effective-seed gate
    reapplication. Recomputing it here would duplicate
    `summarize_g1_v1_2`/`derive_negative_semantics_g1_v1_2.py`'s logic and
    risk drifting from it; this audit only reports what that derivation
    already concluded.
    """
    if not v1_2_derived_summary_path.is_file():
        return {
            "scientific_gate_status": "UNKNOWN_V1_2_NOT_DERIVED",
            "v1_2_derived_summary_path": str(v1_2_derived_summary_path),
            "note": (
                "no v1.2 amendment derivation found at this path; run "
                "scripts/derive_negative_semantics_g1_v1_2.py first. The declared "
                "g1_summary.json gate_status alone (declared-seed, not "
                "effective-seed) is not sufficient evidence of the scientific gate."
            ),
        }
    derived = load_json(v1_2_derived_summary_path)
    reapplication = derived.get("gate_reapplication") or {}
    status = reapplication.get("effective_seed_status", "UNKNOWN_MISSING_GATE_REAPPLICATION")
    return {
        "scientific_gate_status": status,
        "v1_2_derived_summary_path": str(v1_2_derived_summary_path),
        "effective_seed_checks": reapplication.get("effective_seed_checks"),
        "checks_that_flip_to_fail_under_effective_seeds": reapplication.get(
            "checks_that_flip_to_fail_under_effective_seeds"
        ),
        "seed_evidence_method": derived.get("seed_evidence_method"),
    }


def audit(run_root: Path, *, v1_2_derived_summary_path: Path = DEFAULT_V1_2_DERIVED_SUMMARY) -> Dict[str, Any]:
    run_root = run_root.resolve()
    checks: List[Dict[str, Any]] = []

    run_spec = load_json(run_root / "run_spec.json")
    summary = load_json(run_root / "g1_summary.json")

    checks.append(_check(
        "run_spec_not_smoke",
        run_spec.get("smoke") is False,
        smoke=run_spec.get("smoke"),
    ))
    checks.append(_check(
        "run_spec_heldout_not_accessed",
        run_spec.get("heldout_accessed") is False,
        heldout_accessed=run_spec.get("heldout_accessed"),
    ))
    video_ids = list(run_spec.get("video_ids") or [])
    checks.append(_check(
        "run_spec_forty_ovis_pilot_videos",
        len(video_ids) == 40 and all(str(v).startswith("ovis_valid_") for v in video_ids),
        video_count=len(video_ids),
    ))
    declared_policies = dict(run_spec.get("policies") or {})
    checks.append(_check(
        "run_spec_both_policies_present",
        declared_policies == {"few10": 10, "full50": 50},
        policies=declared_policies,
    ))
    declared_seeds = list(run_spec.get("seeds") or [])
    checks.append(_check(
        "run_spec_three_declared_seeds",
        sorted(declared_seeds) == [2025, 2026, 2027],
        seeds=declared_seeds,
    ))

    # Each (policy, seed) child reconstruction run writes its own summary.json
    # via run_transmission_reduction_eval.py with n_videos/n_failed_pairs/
    # run_status. 6 child runs x 40 videos = 240 evaluated pairs.
    child_summaries: Dict[str, Dict[str, Any]] = {}
    missing_children: List[str] = []
    for policy in declared_policies:
        for seed in declared_seeds:
            key = f"{policy}/seed_{seed}"
            path = run_root / "reconstruction" / policy / f"seed_{seed}" / "summary.json"
            if not path.is_file():
                missing_children.append(key)
                continue
            child_summaries[key] = load_json(path)
    total_videos = sum(int(item.get("n_videos", 0)) for item in child_summaries.values())
    total_failed = sum(int(item.get("n_failed_pairs", 0)) for item in child_summaries.values())
    all_completed = all(item.get("run_status") == "completed" for item in child_summaries.values())
    checks.append(_check(
        "child_runs_240_of_240_zero_failures",
        not missing_children and len(child_summaries) == 6
        and total_videos == 240 and total_failed == 0 and all_completed,
        child_run_count=len(child_summaries),
        missing_children=missing_children,
        total_videos=total_videos,
        total_failed_pairs=total_failed,
        all_run_status_completed=all_completed,
    ))

    detection_rows_path = run_root / "evaluator" / "detection_rows.jsonl"
    detection_rows = load_jsonl(detection_rows_path)
    checks.append(_check(
        "detection_rows_nonempty_and_seven_concepts",
        len(detection_rows) > 0
        and {row["concept_id"] for row in detection_rows} == {
            "person", "bird", "cat", "dog", "bicycle", "motorcycle", "vehicle",
        },
        row_count=len(detection_rows),
        distinct_concepts=sorted({row["concept_id"] for row in detection_rows}),
    ))

    checks.append(_check(
        "gate_summary_evidence_scope_is_ovis_pilot_g1",
        summary.get("evidence_scope") == "OVIS_PILOT_G1" and summary.get("heldout_accessed") is False,
        evidence_scope=summary.get("evidence_scope"),
        heldout_accessed=summary.get("heldout_accessed"),
    ))
    gate_checks = dict(summary.get("gate_checks") or {})
    checks.append(_check(
        "gate_checks_all_recorded_true",
        bool(gate_checks) and all(bool(value) for value in gate_checks.values()),
        gate_status=summary.get("gate_status"),
        gate_checks=gate_checks,
    ))

    # Held-out non-access: the only reference to the sealed split anywhere in
    # this run's provenance chain is the G0 audit's manifest-level seal check
    # (dataset_split_manifest.json entries + heldout_seal.json hash), which
    # never opens a held-out media file. Confirm no held-out dataset path is
    # named as a video source in this run.
    heldout_like = [v for v in video_ids if "davis" in str(v).lower()]
    checks.append(_check(
        "no_heldout_source_among_run_videos",
        not heldout_like,
        heldout_like_video_ids=heldout_like,
    ))

    # Effective-seed accounting (v1.2 amendment finding): report, do not gate
    # --require-pass on it, since the runner completed correctly regardless
    # of whether the decoder happens to be seed-invariant.
    seed_groups = effective_seed_groups(detection_rows)
    effective_seed_summary = {
        policy: {
            "declared_seed_count": info["declared_seed_count"],
            "effective_seed_count": info["effective_seed_count"],
            "deterministic_collapse": info["deterministic_collapse"],
        }
        for policy, info in seed_groups.items()
    }
    checks.append(_check(
        "effective_seed_count_reported",
        bool(effective_seed_summary),
        effective_seed_summary=effective_seed_summary,
        interpretation=(
            "reconstruction scores are bit-identical across nominal seeds "
            "for this deterministic decoder path; effective_seed_count=1 "
            "per policy does not invalidate the run, but seed-level "
            "variance/CI claims should use the v1.2 effective-seed view"
        ),
    ))

    required_ids = {
        "run_spec_not_smoke", "run_spec_heldout_not_accessed",
        "run_spec_forty_ovis_pilot_videos", "run_spec_both_policies_present",
        "run_spec_three_declared_seeds", "child_runs_240_of_240_zero_failures",
        "detection_rows_nonempty_and_seven_concepts",
        "gate_summary_evidence_scope_is_ovis_pilot_g1",
        "gate_checks_all_recorded_true", "no_heldout_source_among_run_videos",
    }
    result_by_id = {item["check_id"]: item for item in checks}
    missing_required = sorted(required_ids - set(result_by_id))
    runner_passed = not missing_required and all(
        result_by_id[item]["passed"] for item in required_ids
    )
    scientific = _scientific_gate_status(v1_2_derived_summary_path)
    return {
        "audit_id": "negative_semantics_g1_v1_1_run_audit",
        "run_root": str(run_root),
        "runner_gate_status": "PASSED" if runner_passed else "NOT_PASSED",
        "runner_required_check_count": len(required_ids),
        "runner_required_checks_passed": sum(
            1 for item in required_ids if result_by_id.get(item, {}).get("passed")
        ),
        "runner_missing_required_checks": missing_required,
        "declared_gate_status": summary.get("gate_status"),
        **scientific,
        "checks": checks,
        "paper_claimable_scope_note": (
            "runner_gate_status PASSED verifies the 240/240 OVIS Pilot "
            "inference+evaluation completed without failure and without held-out "
            "access -- a narrower claim than 'method generalizes'. "
            "declared_gate_status PASSED reflects g1_summary.json's own "
            "declared-seed definition (3 nominal seeds counted as independent). "
            "scientific_gate_status is the effective-seed reapplication of the "
            "SAME frozen thresholds (see the v1.2 amendment) and is the one that "
            "should gate a paper claim; for this run it is NOT_PASSED because "
            "effective_seed_count=1 (the decoder never consumes seed) collapses "
            "'not concentrated in one seed' to trivially false."
        ),
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-root", type=Path,
        default=ROOT / "outputs/negative_semantics_g1_pilot_rtx4080_v1_1",
    )
    parser.add_argument(
        "--v1-2-derived-summary", type=Path, default=DEFAULT_V1_2_DERIVED_SUMMARY,
        help="Path to a g1_summary_v1_2.json produced by derive_negative_semantics_g1_v1_2.py.",
    )
    parser.add_argument(
        "--require-runner-complete", action="store_true",
        help="Exit non-zero unless the runner-completion checks all pass (child-run count, held-out non-access, etc).",
    )
    parser.add_argument(
        "--require-scientific-gate", action="store_true",
        help=(
            "Exit non-zero unless scientific_gate_status (the effective-seed gate "
            "reapplication read from --v1-2-derived-summary) is PASSED. Fails "
            "closed (non-zero) if that derivation has not been produced."
        ),
    )
    parser.add_argument(
        "--require-pass", action="store_true",
        help="Deprecated alias for --require-runner-complete; prints a warning.",
    )
    args = parser.parse_args(argv)
    if args.require_pass:
        print(
            "[audit_negative_semantics_g1] --require-pass is deprecated; it checks "
            "only runner completion, not the scientific gate. Use "
            "--require-runner-complete (same behavior) and/or "
            "--require-scientific-gate explicitly.",
            file=sys.stderr,
        )
        args.require_runner_complete = True
    report = audit(args.run_root, v1_2_derived_summary_path=args.v1_2_derived_summary)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    exit_code = 0
    if args.require_runner_complete and report["runner_gate_status"] != "PASSED":
        exit_code = 4
    if args.require_scientific_gate and report["scientific_gate_status"] != "PASSED":
        exit_code = exit_code or 5
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
