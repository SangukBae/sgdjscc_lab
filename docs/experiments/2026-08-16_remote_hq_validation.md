---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
experiment_commit: unknown
documentation_commit: ec367bb
supersedes:
---

> [← 문서 색인](../README.md)

# 원격 3×RTX 4090 고품질 검증

## 범위

- 이미지
  - Kodak 24장
  - SNR 7개
- 영상
  - ETRI 10개×100프레임
  - diffusion 50 step
- 생성
  - SVD: 512×256, 25 step
  - Wan: 512×256, 30 step

## 환경

- GPU: RTX 4090×3
- SGD-JSCC
  - 영상별 worker 병렬
- Wan 14B
  - 한 pipeline을 3GPU에 분산
  - 영상·모드 순차 실행
- SVD
  - GPU 0 사용

## 실행

```bash
python scripts/run_remote_hq_validation.py \
    --output-root outputs/remote_hq_4090_20260816
```

- 재개
  - `--skip-existing`
- 부분 실행
  - `--phases image,video,remeasure,svd,wan,quality`
- 권장
  - `tmux`
  - console log 저장

## 출력

- 상태
  - `preflight.json`
  - `hq_validation_plan.json`
  - `hq_validation_status.json`
- 평가
  - `image/kodak_snr_sweep.csv`
  - `video_real_step50/`
  - `remeasure/summary_metrics.*`
- 생성
  - `generation/svd_start_only/`
  - `generation/wan_skim_sfa/`
  - `generation/wan_skem_dsa/`
- 품질
  - `quality/*_frames.csv`
  - `quality/*_summary.csv|json`

## 주의

- 생성 frame resize 여부 기록
- Wan 세 프로세스 동시 실행 금지
- 캐시 삭제 시 Hugging Face 재인증 필요 가능
- host RAM 62GiB, swap 없음
