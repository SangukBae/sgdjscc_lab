---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: 63b7b23
supersedes: docs/etri_overview.md, docs/phase4.md, docs/phase5.md
---

> [← 문서 색인](../README.md)

# 평가 절차

지표 정의는 [architecture/metrics.md](../architecture/metrics.md), 데이터셋 준비는
[datasets.md](./datasets.md), 코덱 비교 프로토콜은 [video_rate_benchmark.md](./video_rate_benchmark.md)를 따른다.

## 실험 설정 규약

- **SNR 범위** — `[-5, 0, 5, 10, 15, 20, 25]` dB.
- **비교 그룹** — WITT baseline(생성 복원 없음) / DiffJSCC·SGDJSCC baseline(구조
  가이드 없음) / 제안(SGDJSCC + 구조 가이드 + 시맨틱·할루시네이션 평가).
- **가이드 손상 규칙** — AWGN/Rayleigh는 **JSCC latent·채널 심볼에만** 적용한다.
  가이드(edge/seg/caption)는 직접 채널 잡음을 걸지 않고 별도 손상 규칙을 쓴다:
  edge=dropout/blur/erasing, seg=클래스 dropout/영역 제거, 캡션=token dropout.
- **입력 크기** — 128×128 패치 타일링, H·W를 128 배수로 리사이즈
  (예시: `configs/base/dataset/kodak.yaml`).

## Rate–Reliability–Hallucination 평가 계약

단일 합성 점수로 우열을 정하지 않고 다음 축을 같은 run row에 분리해 기록한다.
현재 스크립트가 아직 내보내지 않는 컬럼도 있으며, 이는 향후 harness의 목표 계약이다.

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
- reliable-digital baseline은 float32를 우선하고, 없으면 valid ratio가 1인 int16을
  사용한다. 둘 다 없을 때 AWGN은 별도 임시 품질 기준으로만 표시한다.

## 자동 비교와 통계

평가 harness가 controller의 threshold·budget·retry 설정을 sweep한다. controller가
자체적으로 sweep하거나 결과 파일을 누적하는 구조로 만들지 않는다.

필수 ablation은 다음 네 정책을 regeneration OFF/ON으로 각각 실행하는 8개 조합이다.

| 정책 | controller 입력 |
|---|---|
| static | 고정 bit·keyframe 예산 |
| SNR-only | 송신 전에 알 수 있는 채널 상태 |
| severity-only | 이전 프레임/GOP의 수신 검증 결과 |
| combined | SNR+risk proxy+이전 severity |

모든 조합은 같은 영상, frame 범위, seed, checkpoint를 사용한다. 영상별 paired
difference와 전체 평균·표준편차·95% 신뢰구간을 보고한다. ETRI 10영상은 개발·비교용,
별도 영상 split은 최종 held-out 검증용으로 구분한다.

## 이미지 평가

```bash
python scripts/evaluate.py --config configs/recipes/inference/composed.yaml --snr 10
python scripts/evaluate.py --config configs/recipes/inference/composed.yaml --snr-list -5,0,5,10,15,20,25
python scripts/evaluate.py --config configs/recipes/inference/composed.yaml --snr 10 --no-clip
```

`--profile {paper,extended,full}`로 지표 집합을, `--require-real-fid`로 FID 실측을
강제할 수 있다. Phase 4 패킷 평가(`use_phase4`/`use_packet_eval: true`)를 켜면
이미지별로 `<stem>.orig_packet.json`, `.packet.json`, `.error_report.json`이
추가로 저장되고 CSV에 `srs_base, srs_packet, object_match_rate,
relation_consistency, attribute_consistency, segmentation_consistency,
scene_match, missing/additional_object_count, relation/attribute_error_count,
guidance_regime`이 붙는다.

## 영상 평가

```bash
python scripts/evaluate_video.py --config configs/recipes/video/composed_video.yaml \
    --input /path/ordered_frames_or.mp4 --snr 5 --device cuda:0

# dry run (체크포인트 없음, 캡션 있으면 델타/지표가 의미를 가짐)
python scripts/evaluate_video.py --config configs/recipes/video/composed_video.yaml \
    --input /path/clip.mp4 --no-models --captions /path/captions.txt
```

Phase 4-B 키프레임/시간축 파이프라인은 기본 off(`use_phase4`)다. 출력:
`keyframes.json`(GOP 구조), `temporal_frames.csv`(프레임별), `temporal_metrics.csv`
(시퀀스 지표 + `overhead_reduction`), 옵션 `recon.mp4`/`recon_frames/`.

### 성능/속도 옵션 (opt-in)

`evaluate_video.py`의 diffusion 비용은 프레임당 128×128 패치 수에 선형 비례한다
(512×256 영상은 프레임당 8패치). 아래 플래그는 전부 opt-in — 기본 동작은 바뀌지 않는다.

| 플래그 | 효과 | 품질 검증 의미 변화 |
|---|---|---|
| `--diffusion-step N` | `cfg.diffusion_step` 오버라이드 | step<50이면 paper-comparable 아님 |
| `--max-frames N` | 앞 N프레임만 처리 | 부분 클립 결과 |
| `--force-interframe-reuse` | 모든 inter-frame을 키프레임 reuse로 강제 | inter-frame drift/hallucination 미검증 |
| `--no-clip` | CLIP 평가자 자체를 안 만듦 | CLIP 기반 지표 전부 무의미 |
| `--recon-caption-mode {own,skip}` | 복원 프레임 BLIP2 캡션 생략 여부 | `skip`은 hallucination/SRS를 `own`과 비교 불가 |
| `--packet-cache-dir DIR` | 원본 프레임 packet만 디스크 캐시 | 없음(재실행 가속용) |
| `--profile` / `--profile-out PATH` | `progress.json`/`profiling_summary.json` 생성 | 없음 |

배치 드라이버(`scripts/run_etri_video_eval.py`)는 동일 플래그 패스스루 +
`--parallel N --devices cuda:0,cuda:1,cuda:2`(멀티 GPU 라운드로빈) +
`--gpu-log-interval SEC`(GPU 사용률 로깅)를 지원한다. 실측 속도/버그 수정
이력은 [experiments/2026-07-24_video_speed_optimization.md](../experiments/2026-07-24_video_speed_optimization.md).

### ETRI 10-영상 배치 평가

```bash
python scripts/run_etri_video_eval.py --stages all --no-models   # 배선 검증
python scripts/run_etri_video_eval.py --stages all --snr 5 --device cuda:0   # 실모델
python scripts/summarize_etri_video_eval.py --output-root outputs/etri_video_eval
python scripts/generate_etri_final_report.py --output-root outputs/etri_video_eval
```

데이터셋 구성은 [datasets.md](./datasets.md#etri-10-영상-평가셋)과
`data/etri_video_eval/README.md`.

### Presence(객체 존재) 보정 재측정

CLIP 기반 기본 판정을 OWLv2/VQA/GT로 보강해 재측정하는 절차. 기존 실모델 결과의
`extracted_frames/`/`recon_frames/`를 재사용하고 packet만 다시 추출한다(재구성
재실행 없음).

```bash
python scripts/remeasure_video_metrics.py --config configs/experiments/etri_video_eval/etri_video_eval_owlv2.yaml \
    --from-recon-frames outputs/etri_video_eval_real_full_step50/baseline/01_person_walk \
    --captions data/etri_video_eval/captions/01_person_walk.txt --device cuda:0

# 10개 영상 × 5개 모드(owlv2/vqa/ensemble_nofilter/ensemble_gt_filter/ensemble_openworld_filter) 배치
python scripts/batch_remeasure_owlv2_vqa_10videos.py --dry-run
python scripts/batch_remeasure_owlv2_vqa_10videos.py --device cuda:0
```

`--from-run`(기본, `--input` 지정)은 **처음부터 재구성**하므로 과거 run의 실제
결정/픽셀을 재생하지 않는다 — byte-exact 재사용이 필요하면 `--from-recon-frames`를
쓴다. 완료된 재측정 결과는
[experiments/2026-07-28_owlv2_vqa_calibration.md](../experiments/2026-07-28_owlv2_vqa_calibration.md).

## 관련 문서
- [architecture/metrics.md](../architecture/metrics.md) — 지표 정의
- [datasets.md](./datasets.md) — 데이터셋 역할·준비
- [video_rate_benchmark.md](./video_rate_benchmark.md) — 코덱 대비 전송량·화질 비교
- [reproducibility.md](./reproducibility.md) — checkpoint 선택, `paper_mode`
