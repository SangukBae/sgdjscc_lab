"""diagnostics/report.py – REPORT.md writer for a diagnostic run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# src/sgdjscc_lab/diagnostics/report.py -> repo root is 3 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _doc_link(output_root: Path, doc_relpath: str) -> str:
    """Relative path from *output_root* to ``<repo_root>/<doc_relpath>``.

    ``output_root`` is not always exactly one fixed depth below the repo
    root (e.g. ``outputs/<run>/`` vs. the server driver's
    ``outputs/<run>/stage3_single_frame_paths/``) — computing this instead
    of hardcoding ``"../../docs/..."`` keeps the link correct regardless of
    how deep *output_root* is nested.
    """
    target = _REPO_ROOT / doc_relpath
    try:
        return os.path.relpath(target, start=output_root.resolve())
    except ValueError:
        # Different drives (Windows) or similar — not expected on this
        # project's supported platforms, but never crash report writing over it.
        return doc_relpath


# Ablations whose verdict is auxiliary/supplementary evidence -- kept in
# reports and verdicts.jsonl in full, but never folded into the ONE
# "종합 판정" (verdict_summary) tally, which reflects only ablation ==
# "baseline". Mirrors diagnose_float32_digital_quality.py's
# EDGE_EQUALIZING_ABLATIONS (duplicated here as a plain tuple rather than
# imported, to keep report.py free of a dependency on the CLI script).
AUXILIARY_ABLATIONS = ("serialized_raw_edge", "awgn_edge_retransmit")


def write_report_md(
    output_root: Path,
    *,
    run_kind: str,
    dry_run: bool,
    n_videos: int,
    n_frames: int,
    n_ablations: int,
    verdict_summary: Optional[Dict[str, Any]],
    per_video_verdicts: List[Dict[str, Any]],
    failed_count: int,
    outputs: Dict[str, str],
    n_provisional: int = 0,
    n_auxiliary: int = 0,
) -> Path:
    """Writes ``REPORT.md``. Content is scoped by *dry_run*: a dry-run/mock
    (CPU) invocation never claims a real quality finding — only that the
    harness ran structurally — per the project rule that a root cause must
    not be asserted before real server measurements exist.

    *verdict_summary* must already be pre-filtered by the caller to
    ``ablation == "baseline"`` and ``status == "final"`` rows only — this
    function does not re-derive it, it only renders it plus the counts of
    what was excluded (*n_provisional*, *n_auxiliary*) so the reader can see
    those numbers are accounted for, not silently dropped.
    """
    lines: List[str] = []
    lines.append("# float32 digital 복원 품질 저하 진단 결과\n")
    protocols_link = _doc_link(output_root, "docs/protocols/float32_digital_diagnostics.md")
    open_issues_link = _doc_link(output_root, "docs/current/open_issues.md")
    lines.append(
        f"> 관련 문서: [docs/protocols/float32_digital_diagnostics.md]({protocols_link}), "
        f"[docs/current/open_issues.md]({open_issues_link})\n"
    )

    lines.append("## 실행 정보\n")
    lines.append(f"- run_kind: `{run_kind}`")
    lines.append(f"- dry_run: `{dry_run}`")
    lines.append(f"- videos: {n_videos}, frames/video: {n_frames}, ablations: {n_ablations}")
    lines.append(f"- failed_cases: {failed_count}\n")

    if dry_run:
        lines.append("## 상태\n")
        lines.append(
            "**진단 환경 구현 완료, 서버 실측 대기.** 이 리포트는 CPU/mock 또는 "
            "dry-run 실행 결과이며 실제 GPU 추론 결과가 아니다. 아래 판정/근거는 "
            "구조 검증 목적일 뿐, 실제 품질 저하 원인에 대한 결론이 아니다.\n"
        )
    else:
        lines.append("## 판정 (baseline, 최종 확정분만 집계)\n")
        lines.append(
            "`verdict_summary`는 **`ablation == \"baseline\"` AND `status == \"final\"`** 행만 집계한다 — "
            "`serialized_raw_edge`/`awgn_edge_retransmit` 보조 증거나 아직 `diffusion_bypass_vae_direct` "
            "결과를 기다리는 provisional 판정을 더해 과대 집계하지 않는다.\n"
        )
        if n_provisional:
            lines.append(
                f"- provisional(VAE-direct 증거 대기 중, 집계 제외): {n_provisional}건 — "
                "`--resume`으로 나머지 ablation을 마저 실행하면 확정된다.\n"
            )
        if n_auxiliary:
            lines.append(
                f"- 보조 증거(edge-equalizing ablation, 집계 제외): {n_auxiliary}건 — "
                "아래 표에 `ablation` 열로 구분되어 그대로 보존됨.\n"
            )
        if verdict_summary is None or not verdict_summary.get("counts"):
            lines.append("증거 부족 — `inconclusive`.\n")
        else:
            dominant = verdict_summary.get("dominant_verdict")
            lines.append(f"- 종합 판정(최다, baseline·final 기준): `{dominant or 'inconclusive'}`")
            for label, count in sorted(verdict_summary["counts"].items(), key=lambda kv: -kv[1]):
                lines.append(f"  - `{label}`: {count}건")
            lines.append("")

        lines.append("## (video, frame)별 판정\n")
        lines.append(
            "`ablation` 열이 `baseline`인 행만 위 종합 판정에 집계된다. "
            f"{'/'.join(AUXILIARY_ABLATIONS)}는 별도 보조 증거이며, `status`가 `provisional`인 행은 "
            "아직 확정되지 않았다. `evidence_level`은 baseline 판정이 VAE-direct 증거까지 반영했는지 "
            "명시한다.\n"
        )
        lines.append("| video | frame | ablation | status | evidence_level | verdict | first_divergent_stage | reason |")
        lines.append("|---|---:|---|---|---|---|---|---|")
        for row in per_video_verdicts:
            lines.append(
                f"| {row.get('video')} | {row.get('frame')} | {row.get('ablation')} "
                f"| {row.get('status', 'final')} "
                f"| {row.get('evidence_level', 'legacy_unspecified')} "
                f"| {row.get('verdict')} | {row.get('first_divergent_stage') or ''} "
                f"| {str(row.get('reason', '')).replace(chr(10), ' ')[:200]} |"
            )
        lines.append("")

    lines.append("## 산출물\n")
    for name, rel_path in outputs.items():
        lines.append(f"- `{name}`: `{rel_path}`")
    lines.append("")

    lines.append("## 판정 기준\n")
    lines.append(
        "- in-process와 wire가 다름 → packet/Tx-Rx 문제\n"
        "- 두 digital 경로는 같지만 AWGN보다 낮음 → edge·ControlNet·diffusion 문제\n"
        "- `diffusion_bypass_vae_direct` ablation부터 이미 낮음 → "
        "latent scaling/normalization 문제\n"
        "- 증거가 부족하면 `inconclusive`\n"
    )

    out_path = output_root / "REPORT.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_summary_json(
    output_root: Path,
    *,
    run_kind: str,
    dry_run: bool,
    args: Dict[str, Any],
    verdict_summary: Optional[Dict[str, Any]],
    counts: Dict[str, Any],
) -> Path:
    summary = {
        "run_kind": run_kind,
        "dry_run": dry_run,
        "args": args,
        "verdict_summary": verdict_summary,
        "counts": counts,
    }
    out_path = output_root / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return out_path
