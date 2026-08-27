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
) -> Path:
    """Writes ``REPORT.md``. Content is scoped by *dry_run*: a dry-run/mock
    (CPU) invocation never claims a real quality finding — only that the
    harness ran structurally — per the project rule that a root cause must
    not be asserted before real server measurements exist.
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
        lines.append("## 판정\n")
        if verdict_summary is None or not verdict_summary.get("counts"):
            lines.append("증거 부족 — `inconclusive`.\n")
        else:
            dominant = verdict_summary.get("dominant_verdict")
            lines.append(f"- 종합 판정(최다): `{dominant or 'inconclusive'}`")
            for label, count in sorted(verdict_summary["counts"].items(), key=lambda kv: -kv[1]):
                lines.append(f"  - `{label}`: {count}건")
            lines.append("")

        lines.append("## (video, frame)별 판정\n")
        lines.append("| video | frame | ablation | verdict | first_divergent_stage | reason |")
        lines.append("|---|---:|---|---|---|---|")
        for row in per_video_verdicts:
            lines.append(
                f"| {row.get('video')} | {row.get('frame')} | {row.get('ablation')} "
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
