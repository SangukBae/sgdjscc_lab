#!/usr/bin/env python
"""generate_etri_final_report.py – ETRI 10-video batch final report generator.

Aggregates the summary artefacts written by
``scripts/summarize_etri_video_eval.py`` (plus ``batch_status.json`` from
``scripts/run_etri_video_eval.py``) into a single
``<output-root>/final_report.md`` that:

- lists what ran (stage × video, failures, durations);
- embeds the headline per-stage tables;
- separates "코드/PoC로 완료된 항목" from "외부 모델/GT/판정 데이터가 필요한
  항목" exactly along the boundaries docs/etri_strategy.md draws for 1~6차;
- includes the 1~6차 ↔ 산출물 대응표.

This script only reads files — run the batch driver and the summarizer first.

Usage
-----
python scripts/generate_etri_final_report.py \
    --output-root outputs/etri_video_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

SUMMARIES = (
    ("videos_summary", "1. 영상별 기본 평가 (baseline)"),
    ("motion_sweep_summary", "2. Motion gate sweep (1차 step 3)"),
    ("verifier_summary", "3. Packet Verifier + Controller (2차 step 7)"),
    ("generate_summary", "4. 3-way Generate 분기 (3차 step 5, mock backend)"),
    ("bidirectional_summary", "5. start_only vs bidirectional 비교 (4차 step 6, mock backend)"),
    ("heldout_summary", "6. Held-out 재측정 (5차 step 9, clip_only 기준)"),
    ("accounting_summary_all", "7. 전송량 accounting + rate/reliability (6차 step 11-12, PoC)"),
)

# 단계 ↔ 이 배치의 산출물 대응 (docs/etri_strategy.md "단계별 구현 묶음" 표 기준).
STAGE_ARTIFACT_MAP = (
    ("1차 (순서 0~4)", "mp4 IO · PTC/SFR/SDI · motion gate · segment 구조",
     "baseline/<video>/{recon.mp4, temporal_frames.csv, temporal_metrics.csv, keyframes.json, segments.json, run.log}; motion_sweep/th_*/<video>/"),
    ("2차 (순서 7)", "Packet Verifier + 오류 유형별 controller",
     "verifier/<video>/{packet_match_report.json/csv, controller_decisions.json/csv}"),
    ("3차 (순서 5)", "reuse/recompute/generate 3-way 분기 (mock)",
     "generate/<video>/{generated_frames/, temporal_frames.csv(decision=generate), segments.json(generation)}"),
    ("4차 (순서 6)", "bidirectional generate + 비교 파이프라인 (mock)",
     "bidirectional/<video>/{generation_mode_comparison.json, temporal_metrics_start_only.csv, temporal_metrics_bidirectional.csv}"),
    ("5차 (순서 8~10)", "held-out 재측정 + presence calibration 인터페이스 "
     "(baseline/verifier가 저장한 recon/packet을 재사용하지 않고, 같은 config로 "
     "TemporalPipeline을 처음부터 다시 돌려 독립 재구성함 — --no-models에서는 "
     "결정적이라 baseline과 값이 같지만 확률적 실모델 샘플링에서는 달라질 수 있음)",
     "heldout/<video>/heldout/{clip_only_metrics.*, calibrated_metrics.*, metric_delta.json/csv}, gt_presence.json(입력 포맷 변환)"),
    ("6차 (순서 11~12)", "bit/channel-symbol accounting + rate/reliability 리포트 (PoC)",
     "accounting/<video>/accounting/{frame_accounting.*, segment_accounting.*, accounting_summary.json, rate_reliability_summary.json/curve.csv(영상별 1행)}, "
     "summary/rate_reliability_curve.csv(모든 영상 merge — summarize 단계에서 post-hoc으로 생성, run 중 append 아님)"),
)

DONE_ITEMS = (
    "10개 영상 batch 평가 자동화 (영상·단계별 분리 출력 + run.log) — scripts/run_etri_video_eval.py",
    "결과 summary CSV/JSON/Markdown 자동 생성 — scripts/summarize_etri_video_eval.py",
    "motion gate sweep 자동화 및 threshold별 비교표 (05/06 focus 포함)",
    "Packet Verifier 실행 자동화 + 오류 유형/decision 비율 요약 (rule-based, CLIP/caption packet 기준)",
    "3-way generate 분기 검증 자동화 (mock copy/interpolation backend, generated_frames/ 저장 확인)",
    "start_only vs bidirectional 비교 자동화 (mock bidirectional_interpolation backend)",
    "held-out 재측정 실행 자동화 (clip_only 기준; loop-internal(verifier/)과 held-out(heldout/) 분리 저장, metric_delta 생성) — "
    "단, heldout은 baseline/verifier의 recon/packet을 재사용하지 않고 독립적으로 재구성한 결과임에 유의",
    "transmission accounting 자동화 (frame/segment accounting + bit/symbol 절감률, PoC proxy)",
    "rate-reliability 리포트 자동화 (영상별 accounting/rate_reliability_curve.csv → summarize 단계에서 "
    "summary/rate_reliability_curve.csv로 post-hoc merge; 병렬 accounting 실행 시 공유 파일 append 경쟁 없음)",
    "batch_status.json에 stage/video/threshold별 no_models/snr/device 기록 (동일 출력 디렉터리를 만든 설정 추적 가능)",
    "GT segment JSON → presence backend 입력 포맷({item_id:{object:bool}}) 변환 (heldout/<video>/gt_presence.json)",
)

EXTERNAL_ITEMS = (
    "OWLv2/VQA 실제 weight 기반 presence 판정 성능 검증 — 인터페이스(evaluators/presence_backends.py)만 연결됨; "
    "실 모델로 오탐/미탐 감소를 확인한 결과는 없음 (5차 🟡)",
    "GT/VLM 기반 Temporal SRS Calibration — least-squares fitting 스캐폴드와 GT 입력 포맷 변환까지만; "
    "실제 GT 주석/VLM judge 호출 및 가중치 보정 결과 없음 (5차 🟡)",
    "calibrated ≠ clip_only인 held-out 재측정 — 실 presence backend 연결 전에는 calibrated == clip_only가 "
    "의도된 sanity-check 동작",
    "generate/bidirectional 분기의 실제 생성 품질·drift/flicker 감소 주장 — mock backend 결과는 구조 검증일 뿐, "
    "SVD/Open-Sora 등 실제 생성 모델 통합 후 판단 (3~4차 scope note)",
    "bit/symbol 절감률의 실제 무선 채널 검증 — accounting은 명시적 proxy 기반 PoC이며 실제 변조·부호화·"
    "엔트로피 코딩(CBR/표준 bitstream) 재현이 아님 (6차 🟡)",
    "실모델(SGD-JSCC checkpoint + diffusion) 기반 10개 영상 본평가 — 본 배치의 --no-models 실행은 "
    "identity 복원 스모크; 실모델 배치는 GPU 서버에서 동일 명령(--no-models 제거, --snr/--device 지정)으로 실행",
)


def _read_md_table(summary_dir: Path, name: str):
    """Return the pipe-table body of summary/<name>.md (without its H1), or None."""
    p = summary_dir / f"{name}.md"
    if not p.exists():
        return None
    lines = p.read_text(encoding="utf-8").splitlines()
    # Drop the "# title" heading; keep everything else (table + focus sections).
    body = [ln for i, ln in enumerate(lines) if not (i == 0 and ln.startswith("# "))]
    while body and not body[0].strip():
        body.pop(0)
    return "\n".join(body).rstrip()


def _batch_overview(status: list) -> str:
    if not status:
        return "_batch_status.json 없음 — run_etri_video_eval.py 실행 기록이 없다._"
    by_stage = {}
    for r in status:
        by_stage.setdefault(r.get("stage"), []).append(r)
    lines = ["| stage | runs | ok | failed | skipped | no_models/snr/device | videos |",
              "|---|---|---|---|---|---|---|"]
    for stage, runs in sorted(by_stage.items()):
        ok = sum(1 for r in runs if r.get("status") == "ok")
        failed = sum(1 for r in runs if r.get("status") == "failed")
        skipped = sum(1 for r in runs if r.get("status") == "skipped")
        videos = sorted({r.get("video") for r in runs})
        vids = ", ".join(videos) if len(videos) <= 3 else f"{len(videos)} videos"
        # Distinct (no_models, snr, device) combos that actually PRODUCED the
        # files currently on disk for this stage — traces which config a
        # --no-models smoke vs a real-model run left behind (they land in the
        # same output directory and overwrite each other). For "ok"/"failed"
        # rows this is just this run's own args; for a "skipped" row
        # run_one() recovers it from the prior batch_status.json record when
        # one exists. Two cases are deliberately excluded from this
        # "confident" set (r.get("no_models") is None exactly when the
        # first happened, r.get("prior_status") == "failed" the second):
        #   - unknown: marker file present but no prior record at all;
        #   - uncertain: the prior record itself was "failed" — its config
        #     is only the last ATTEMPTED one, not a confirmed produced-by one
        #     (the marker could be stale/partial — see run_one()'s docstring).
        confident = [r for r in runs if r.get("no_models") is not None
                    and r.get("prior_status") != "failed"]
        configs = sorted(
            {(r.get("no_models"), r.get("snr"), r.get("device")) for r in confident},
            key=lambda t: tuple(str(x) for x in t),
        )
        cfg_str = "; ".join(f"no_models={nm},snr={s},device={d}" for nm, s, d in configs) or "—"
        unknown = sum(1 for r in runs if r.get("status") == "skipped" and r.get("no_models") is None)
        uncertain = sum(1 for r in runs if r.get("prior_status") == "failed")
        if unknown:
            cfg_str += f" (+{unknown} skipped, provenance unknown)"
        if uncertain:
            cfg_str += f" (+{uncertain} skipped, prior run failed — provenance uncertain)"
        lines.append(f"| {stage} | {len(runs)} | {ok} | {failed} | {skipped} | {cfg_str} | {vids} |")
    failures = [r for r in status if r.get("status") == "failed"]
    if failures:
        lines += ["", "**실패한 run:**", ""]
        for r in failures:
            lines.append(f"- {r.get('stage')} / {r.get('video')} → `{r.get('out_dir')}/run.log`")
    return "\n".join(lines)


def build_report(output_root: Path) -> str:
    root = Path(output_root)
    summary_dir = root / "summary"
    status = []
    status_path = root / "batch_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))

    parts = [
        "# ETRI 10-영상 평가 최종 리포트",
        "",
        f"- 생성일: {date.today().isoformat()}",
        f"- 출력 루트: `{root}`",
        "- 생성 도구: `scripts/run_etri_video_eval.py` → "
        "`scripts/summarize_etri_video_eval.py` → `scripts/generate_etri_final_report.py`",
        "- 기준 문서: `docs/etri_strategy.md` (1~6차 구현 실행 순서)",
        "",
        "> **범위 주의**: `--no-models` 실행분은 identity 복원 + caption 기반 packet의",
        "> 구조 검증(스모크)이다. 수치(PTC/SFR/SDI, severity, 절감률)는 파이프라인·산출물",
        "> 구조가 동작함을 보이는 값이지, 실모델 복원 품질에 대한 주장이 아니다.",
        "",
        "## 실행 개요",
        "",
        _batch_overview(status),
        "",
    ]

    for name, title in SUMMARIES:
        table = _read_md_table(summary_dir, name)
        if table is None:
            continue
        parts += [f"## {title}", "", table, ""]
        if name == "accounting_summary_all" and (summary_dir / "rate_reliability_curve.csv").exists():
            parts += ["전체 영상의 rate/reliability 곡선(1점=1영상)은 "
                     f"`{(summary_dir / 'rate_reliability_curve.csv').relative_to(root)}` 참조 "
                     "(accounting 단계별로 개별 저장된 파일을 summarize 단계가 병합; run 중 공유 파일 append 없음).", ""]

    parts += ["## etri_strategy.md 1~6차 ↔ 산출물 대응표", "",
              "| 단계 | 내용 | 이 배치의 산출물 위치 |", "|---|---|---|"]
    for stage, what, artifacts in STAGE_ARTIFACT_MAP:
        parts.append(f"| {stage} | {what} | `{artifacts}` |")

    parts += ["", "## 코드/PoC로 완료된 항목", ""]
    parts += [f"- ✅ {item}" for item in DONE_ITEMS]

    parts += ["", "## 외부 모델/GT/판정 데이터가 필요한 항목 (미완료로 유지)", ""]
    parts += [f"- 🟡 {item}" for item in EXTERNAL_ITEMS]

    parts += [
        "",
        "## 재현 명령",
        "",
        "```bash",
        "cd sgdjscc_lab",
        "# 전체 배치 (스모크: --no-models / 실모델: --no-models 제거 + --snr/--device)",
        "python scripts/run_etri_video_eval.py --stages all --no-models",
        "python scripts/run_etri_video_eval.py --stages motion_sweep \\",
        "    --motion-thresholds off,0.02,0.05,0.08,0.12 --no-models",
        "python scripts/summarize_etri_video_eval.py",
        "python scripts/generate_etri_final_report.py",
        "```",
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the ETRI 10-video final report")
    p.add_argument("--output-root", default=str(_REPO_ROOT / "outputs" / "etri_video_eval"))
    p.add_argument("--output", default=None,
                   help="Report path (default <output-root>/final_report.md)")
    args = p.parse_args()

    root = Path(args.output_root)
    if not (root / "summary").is_dir():
        sys.exit(f"Error: {root}/summary not found — run summarize_etri_video_eval.py first.")
    report = build_report(root)
    out = Path(args.output) if args.output else root / "final_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Final report → {out}")


if __name__ == "__main__":
    main()
