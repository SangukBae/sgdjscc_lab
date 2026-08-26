---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
experiment_commit: unknown
documentation_commit: ec367bb
supersedes:
---

> [← 문서 색인](../README.md)

# OWLv2/VQA Presence Calibration

## 목적

- CLIP object 판정 오탐 보정
- 복원 재실행 없이 held-out metric 재측정
- closed-world와 open-world 주장 분리

## 입력

- 영상: ETRI 10개
- 프레임: 영상당 100개
- 재사용 경로
  - 원본: `extracted_frames/`
  - 복원: `recon_frames/`
- 기준 run
  - `outputs/etri_video_eval_real_full_step50/baseline/`

## 모드

- `owlv2`
  - zero-shot detection
- `vqa`
  - yes/no presence 질문
- `ensemble_nofilter`
  - vocabulary filter 없음
- `ensemble_gt_filter`
  - GT object만 유지
  - object preservation 주장용
- `ensemble_openworld_filter`
  - 잡음 token만 제거
  - hallucination 주장용

## 실행

```bash
python scripts/batch_remeasure_owlv2_vqa_10videos.py --device cuda:0
```

- 결과: 50/50 job 성공
- 출력
  - `outputs/etri_video_eval/remeasure_10videos/summary_metrics.csv`
  - `outputs/etri_video_eval/remeasure_10videos/summary_metrics.md`

## 평균 변화량

| mode | severity | PTC | SFR | missing | additional | hallucination |
|---|---:|---:|---:|---:|---:|---:|
| OWLv2 | -0.1300 | +0.2036 | -0.0337 | -211.6 | -18.6 | -0.0314 |
| VQA | -0.2065 | +0.3337 | -0.0219 | -364.6 | -14.9 | -0.0253 |
| ensemble GT | -0.1872 | +0.3120 | -0.0182 | -72.4 | 0.0 | 0.0 |
| ensemble open-world | -0.1857 | +0.2875 | -0.0190 | -169.4 | -14.9 | -0.0598 |

## 해석

- OWLv2
  - 모든 영상에서 severity 감소
  - 모든 영상에서 PTC 증가
- VQA
  - 보정폭 큼
  - vocabulary 잡음 민감
- GT filter
  - 의미 보존 평가에 사용
  - GT 밖 hallucination을 제거하므로 hallucination 평가에는 사용 금지
- open-world filter
  - additional object·hallucination 평가에 사용
- nofilter
  - contamination 진단용
  - 최종 주장에 사용 금지

## 제한

- `--from-run`: 재구성부터 다시 실행
- `--from-recon-frames`: 저장 frame 재사용
- byte-exact pipeline replay: 미지원
