---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: 63b7b23
supersedes: docs/etri_overview.md, docs/phase4.md, docs/phase5.md
---

> [← 문서 색인](../README.md)

# 평가 절차

- 연결 문서
  - 지표 정의: [metrics.md](../architecture/metrics.md)
  - 데이터 준비: [datasets.md](./datasets.md)
  - 코덱 비교: [video_rate_benchmark.md](./video_rate_benchmark.md)

## 실험 설정 규약

- **SNR 범위**
  - `[-5, 0, 5, 10, 15, 20, 25]` dB.
- **비교 그룹**
  - WITT baseline(생성 복원 없음) / DiffJSCC·SGDJSCC baseline(구조 가이드 없음) / 제안(SGDJSCC + 구조 가이드 + 시맨틱·할루시네이션 평가).
- **가이드 손상 규칙**
  - AWGN/Rayleigh는 **JSCC latent·채널 심볼에만** 적용한다.
  - edge: dropout·blur·erasing
  - segmentation: class dropout·영역 제거
  - caption: token dropout
- **입력 크기**
  - 128×128 패치 타일링, H·W를 128 배수로 리사이즈 (예시: `configs/base/dataset/kodak.yaml`).

## Rate–Reliability–Hallucination 평가 계약

- 기록 원칙
  - 단일 합성 점수로 우열 판정 금지
  - 모든 축을 같은 run row에 분리 기록
  - 미출력 열은 향후 harness의 목표 계약으로 관리

| 축 | 필수 항목 |
|---|---|
| Rate | exact bundle bytes/frame, feedback bytes, retransmission bytes, proxy channel symbols |
| Quality | PSNR, SSIM, LPIPS, SRS |
| Hallucination | missing/additional object rate, hallucination score, temporal hallucination |
| Cost | reconstruction·regeneration latency, retry 수, end-to-end latency |
| Provenance | config, checkpoint, seed, dataset split, code commit, `metric_role` |

- exact packet byte와 proxy symbol/FEC 환산은 별도 컬럼으로 둔다.
- analog AWGN visual waveform에 exact byte가 없으면 `N/A`로 기록한다.
- effective rate는 최초 전송뿐 아니라 feedback과 재전송을 포함한다.
- 루프 제어에는 `loop_internal`, 최종 주장은 `held_out` 지표만 사용한다.
- reliable-digital baseline 우선순위
  1. float32
  2. valid ratio 1인 int16
  3. AWGN 임시 품질 기준

## 자동 비교와 통계

- 역할 분리
  - 평가 harness: threshold·budget·retry sweep
  - controller: 단일 관측값에 대한 정책 결정
  - 결과 누적: 평가 harness 담당

- 필수 ablation은 다음 네 정책을 regeneration OFF/ON으로 각각 실행하는 8개 조합이다.

| 정책 | controller 입력 |
|---|---|
| static | 고정 bit·keyframe 예산 |
| SNR-only | 송신 전에 알 수 있는 채널 상태 |
| severity-only | 이전 프레임/GOP의 수신 검증 결과 |
| combined | SNR+risk proxy+이전 severity |

- 비교 조건
  - 동일 영상·frame 범위·seed·checkpoint
  - 영상별 paired difference
  - 전체 평균·표준편차·95% 신뢰구간
- 데이터 분리
  - ETRI 10영상: 개발·비교
  - 별도 영상 split: 최종 held-out 검증

## 이미지 평가

```bash
python scripts/evaluate.py --config configs/recipes/inference/composed.yaml --snr 10
python scripts/evaluate.py --config configs/recipes/inference/composed.yaml --snr-list -5,0,5,10,15,20,25
python scripts/evaluate.py --config configs/recipes/inference/composed.yaml --snr 10 --no-clip
```

- 선택 옵션
  - `--profile {paper,extended,full}`: 지표 집합 선택
  - `--require-real-fid`: FID 실측 강제
- 패킷 평가
  - 게이트: `use_phase4`, `use_packet_eval: true`
  - 파일: `<stem>.orig_packet.json`, `.packet.json`, `.error_report.json`
  - CSV: SRS·객체·관계·속성·segmentation·scene·guidance 열 추가

## 영상 평가

```bash
python scripts/evaluate_video.py --config configs/recipes/video/composed_video.yaml \
    --input /path/ordered_frames_or.mp4 --snr 5 --device cuda:0

# dry run (체크포인트 없음, 캡션 있으면 델타/지표가 의미를 가짐)
python scripts/evaluate_video.py --config configs/recipes/video/composed_video.yaml \
    --input /path/clip.mp4 --no-models --captions /path/captions.txt
```

- 게이트: `use_phase4`, 기본 off
- 출력
  - `keyframes.json`: GOP 구조
  - `temporal_frames.csv`: 프레임별 결과
  - `temporal_metrics.csv`: 시퀀스 지표·`overhead_reduction`
  - 선택: `recon.mp4`, `recon_frames/`

### 성능/속도 옵션 (opt-in)

- diffusion 비용
  - 기준: frame당 128×128 patch 수
  - 예시: 512×256 = frame당 8 patch
- 옵션 규칙
  - 아래 flag는 모두 opt-in
  - 기본 동작 불변

| 플래그 | 효과 | 품질 검증 의미 변화 |
|---|---|---|
| `--diffusion-step N` | `cfg.diffusion_step` 오버라이드 | step<50이면 paper-comparable 아님 |
| `--max-frames N` | 앞 N프레임만 처리 | 부분 클립 결과 |
| `--force-interframe-reuse` | 모든 inter-frame을 키프레임 reuse로 강제 | inter-frame drift/hallucination 미검증 |
| `--no-clip` | CLIP 평가자 자체를 안 만듦 | CLIP 기반 지표 전부 무의미 |
| `--recon-caption-mode {own,skip}` | 복원 프레임 BLIP2 캡션 생략 여부 | `skip`은 hallucination/SRS를 `own`과 비교 불가 |
| `--packet-cache-dir DIR` | 원본 프레임 packet만 디스크 캐시 | 없음(재실행 가속용) |
| `--profile` / `--profile-out PATH` | `progress.json`/`profiling_summary.json` 생성 | 없음 |

- 배치 드라이버: `scripts/run_etri_video_eval.py`
  - 기존 평가 플래그 전달
  - `--parallel`, `--devices`: 멀티 GPU round-robin
  - `--gpu-log-interval`: GPU 사용률 기록
  - 근거: [속도 최적화 실험](../experiments/2026-07-24_video_speed_optimization.md)

### ETRI 10-영상 배치 평가

```bash
python scripts/run_etri_video_eval.py --stages all --no-models   # 배선 검증
python scripts/run_etri_video_eval.py --stages all --snr 5 --device cuda:0   # 실모델
python scripts/summarize_etri_video_eval.py --output-root outputs/etri_video_eval
python scripts/generate_etri_final_report.py --output-root outputs/etri_video_eval
```

- 데이터셋 구성
  - [datasets.md](./datasets.md#etri-10-영상-평가셋)
  - `data/etri_video_eval/README.md`

### Presence(객체 존재) 보정 재측정

- 목적: CLIP 판정을 OWLv2·VQA·GT로 보강
- 입력 재사용
  - `extracted_frames/`
  - `recon_frames/`
- 재실행 범위
  - packet 재추출
  - 이미지 재구성 제외

```bash
python scripts/remeasure_video_metrics.py --config configs/experiments/etri_video_eval/etri_video_eval_owlv2.yaml \
    --from-recon-frames outputs/etri_video_eval_real_full_step50/baseline/01_person_walk \
    --captions data/etri_video_eval/captions/01_person_walk.txt --device cuda:0

# 10개 영상 × 5개 모드(owlv2/vqa/ensemble_nofilter/ensemble_gt_filter/ensemble_openworld_filter) 배치
python scripts/batch_remeasure_owlv2_vqa_10videos.py --dry-run
python scripts/batch_remeasure_owlv2_vqa_10videos.py --device cuda:0
```

- 입력 모드
  - `--from-run`: 처음부터 재구성
  - `--from-recon-frames`: 기존 픽셀 byte-exact 재사용
- 결과: [OWLv2/VQA 보강 실험](../experiments/2026-07-28_owlv2_vqa_calibration.md)

## 관련 문서
- [architecture/metrics.md](../architecture/metrics.md) — 지표 정의
- [datasets.md](./datasets.md) — 데이터셋 역할·준비
- [video_rate_benchmark.md](./video_rate_benchmark.md) — 코덱 대비 전송량·화질 비교
- [reproducibility.md](./reproducibility.md) — checkpoint 선택, `paper_mode`
