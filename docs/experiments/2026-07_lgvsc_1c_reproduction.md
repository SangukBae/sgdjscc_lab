---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
experiment_commit: unknown
documentation_commit: ec367bb
supersedes:
---

> [← 문서 색인](../README.md)

# LGVSC 1C 재현 준비

## 판정

- config·batch·summary: 완료
- 실제 10영상×4모드 GPU 실행: 미완료
- faithful LGVSC reproduction: 아님

## 비교 모드

| 모드 | 생성 조건 | 해석 |
|---|---|---|
| `mock_baseline` | 선형 start/end blend | 하한선 |
| `svd_start_only` | start image | 실제 diffusion 참고선 |
| `wan_skim_sfa` | start+caption | SFA 근사 |
| `wan_skem_dsa` | start+end+caption | DSA 근사 |

## 공통 한계

- 네 모드의 keyframe selector: 동일
- SKIM/SKEM 알고리즘: 이 단계에서 미재현
- Wan side-info: 미사용
- 모드 이름: nearest reproducible mapping

## 실행

```bash
# 명령 확인
python scripts/batch_lgvsc_1c_reproduce.py --dry-run

# CPU mock
python scripts/batch_lgvsc_1c_reproduce.py \
    --modes mock_baseline --videos 01_person_walk --no-models

# 전체 GPU
python scripts/batch_lgvsc_1c_reproduce.py \
    --modes mock_baseline,svd_start_only,wan_skim_sfa,wan_skem_dsa \
    --device cuda:0

# 기존 결과 요약
python scripts/batch_lgvsc_1c_reproduce.py --summary-only
```

## 출력

- mode/video별
  - `recon.mp4`
  - `temporal_metrics.csv`
  - `segments.json`
  - `keyframes.json`
  - `run.log`
- 전체
  - `batch_status.json`
  - `summary_metrics.csv|md|json`

## 비교 규칙

- 동일 조건
  - `reuse_threshold`
  - frame 수
  - seed
  - checkpoint
- bidirectional 확인
  - `max_frames > keyframe.max_gop`
  - `conditioning_modes_observed` 확인
- 인용 제한
  - SVD: SFA/DSA 비교에서 분리
  - 1B smoke: 품질 우위 근거로 사용 금지

## 후속

- 실제 SKIM/SKEM selector 비교
- real MLLM PSSS
- CBR matched 평가
- 10영상 전체 실행
