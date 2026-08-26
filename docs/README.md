# sgdjscc_lab 개발 문서

## 목적

`sgdjscc_lab`은 원본 `SGDJSCC/` 패키지를 **수정하지 않고** 확장하는 연구용 fork다.
원본은 논문 베이스라인이자 읽기 전용 참조로 두고, 이 패키지는 모듈화·평가·연구
확장을 위한 깨끗한 계층을 얹는다. 추론 forward-pass는 원본과 수치적으로 동일하다.

이 파일은 **문서 색인**이다. 아래는 역할별로 묶었다: 개요/사용법 → 현재 상태 →
향후 계획 → 시스템 구조 → 데이터셋/재현 → 개별 실험 결과 → 과거 기록(`archive/`) →
발표·보고 자료(`reports/`) → 내부 메모(`notes/`).

## 1. 프로젝트 개요·사용법

| 문서 | 내용 |
|---|---|
| [../README.md](../README.md) | 사용자 대상 패키지 사용법 (설치/추론/평가/학습 명령) |
| [etri_overview.md](./etri_overview.md) | ETRI 과제 목표, 전체 파이프라인, SRS, 실험 설정 |
| [training_scaffold.md](./training_scaffold.md) | 학습 CLI: 논문 3-stage(`jscc`/`text_dm`/`controlnet`) + 보조 stage + 데이터 준비 |
| [checkpoint_usage.md](./checkpoint_usage.md) | baseline/custom checkpoint 경로, export 방법, 로컬·원격 가중치 사용 기준 |

## 2. 현재 개발 상태

| 문서 | 내용 |
|---|---|
| [etri_strategy.md](./etri_strategy.md) | 핵심 한계 3가지와 현재 대응 상태 — "무엇이 완료/PoC/스캐폴드인가"의 기준 문서 |
| [phase4.md](./phase4.md) | Phase 4: 패킷 인식 검증기 + 적응형 가이드(4-A), 키프레임/시간적 파이프라인(4-B) |
| [phase5.md](./phase5.md) | Phase 5: 채널 조건화(5-A), 저지연/consistency(5-B), SRS-v2/regeneration search(5-C) |
| [paper_alignment.md](./paper_alignment.md) | 논문 정합성, `paper_mode`, 하이퍼파라미터 출처 |

## 3. 향후 연구개발 계획

| 문서 | 내용 |
|---|---|
| [roadmap.md](./roadmap.md) | 남은 과제를 연구 목표(시간축/할루시네이션/전송량-신뢰도/평가) 기준으로 정리, 일정, ETRI 협의 필요사항 |

## 4. 시스템 구조

| 문서 | 내용 |
|---|---|
| [framework_file_roles.md](./framework_file_roles.md) | 파일별 실행 흐름과 역할 지도 |
| [video_extension_lgvsc.md](./video_extension_lgvsc.md) | LGVSC 논문을 참고한 비디오 전송·복원 확장 설계, 시스템 블록 다이어그램, 모듈 매핑표 |

## 5. 평가 방법과 지표

SRS(Semantic Reliability Score)와 화질/CLIP 지표 정의는 [etri_overview.md](./etri_overview.md),
`PTC`/`SFR`/`SDI` 시간축 지표와 loop-internal/held-out 지표 분리 원칙은
[etri_strategy.md](./etri_strategy.md)에 있다. Presence 판정(CLIP/OWLv2/VQA) 보강은
[etri_owlv2_vqa_readiness.md](./etri_owlv2_vqa_readiness.md) 참고.

## 6. 데이터셋 및 재현 방법

| 문서 | 내용 |
|---|---|
| [dataset_status.md](./dataset_status.md) | 데이터셋 역할·stage 매핑·변환 워크플로 |
| [dev/smoke_training.md](./dev/smoke_training.md) | real-model smoke 학습(1~2 step 배선 검증) |
| [../data/README.md](../data/README.md) | 데이터 디렉터리 구조 |
| [../data/etri_video_eval/README.md](../data/etri_video_eval/README.md) | ETRI 10-영상 평가셋 준비·실행 명령 |

## 7. 개별 실험 결과

| 문서 | 내용 |
|---|---|
| [etri_stage1_validation.md](./etri_stage1_validation.md) | 1차 구현(순서 0~4) 검증 리포트 — 기준 커밋, 테스트/실행 결과, 산출물 |
| [etri_video_speed_optimization.md](./etri_video_speed_optimization.md) | 10-영상 실모델 검증 속도 병목 분석 + 가속화 옵션 + 원격 GPU 검증 결과 |
| [remote_hq_validation.md](./remote_hq_validation.md) | 3×RTX 4090 원격 서버 최종 고품질 검증 |
| [etri_video_rate_benchmark.md](./etri_video_rate_benchmark.md) | 10개 영상 원본·의미 payload·H.264/H.265/AV1 크기와 PSNR/SSIM/LPIPS 비교 |
| [etri_owlv2_vqa_readiness.md](./etri_owlv2_vqa_readiness.md) | 실제 OWLv2/VQA presence calibration 준비 및 완료 결과 |
| [lgvsc_1b_worker_readiness.md](./lgvsc_1b_worker_readiness.md) | 1B 외부 segment 생성 worker 준비/검증 — 실제 GPU 최종 검증 결과 |
| [lgvsc_1c_reproduction_readiness.md](./lgvsc_1c_reproduction_readiness.md) | 1C LGVSC 재현선 검증 준비 — 4개 baseline 모드, batch driver 사용법 |
| [lgvsc_psss_skem_readiness.md](./lgvsc_psss_skem_readiness.md) | PSSS/SKEM variable-length keyframe selector 준비/검증 |

## 8. 과거 계획 및 완료 기록 (`archive/`)

| 문서 | 내용 |
|---|---|
| [archive/etri_implementation_log.md](./archive/etri_implementation_log.md) | 1차~6차, LGVSC 1A/1B/1C 상세 구현 로그 (etri_strategy.md에서 분리) |
| [archive/phases_1to3.md](./archive/phases_1to3.md) | 초기 Phase 1~3 스냅샷 |
| [archive/etri_development_plan_v2.md](./archive/etri_development_plan_v2.md) | 통합 전 개발계획서 보관본 |
| [archive/etri_development_roadmap.md](./archive/etri_development_roadmap.md) | 통합 전 로드맵 보관본 |
| [archive/limitation_reference_map.md](./archive/limitation_reference_map.md) | 통합 전 한계점 지도 보관본 |
| [archive/framework_comparison.md](./archive/framework_comparison.md) | 통합 전 프레임워크 비교 보관본 |
| [archive/paper_gap_closure.md](./archive/paper_gap_closure.md) | 통합 전 paper-mode 정책 문서 보관본 |
| [archive/paper_training_alignment.md](./archive/paper_training_alignment.md) | 통합 전 학습 정합 문서 보관본 |

## 9. 발표·보고 자료 (`reports/`)

ETRI 정기보고용 슬라이드 상세설명 자료(KO/EN, 내부용/외부공유용). 연구 결과 수치는
보고 시점 스냅샷이므로 최신 상태는 위 1~7절 문서를 기준으로 본다.

| 문서 | 내용 |
|---|---|
| [reports/etri_qna_reply.md](./reports/etri_qna_reply.md) | 비전문가 설명용 Q&A 문서 |
| [reports/ETRI_연구진행상황공유_20260816_슬라이드상세설명_KO.md](./reports/ETRI_연구진행상황공유_20260816_슬라이드상세설명_KO.md) | 2026-08-16 진행상황 슬라이드 상세설명 (내부, 국문) |
| [reports/ETRI_연구진행상황공유_20260816_슬라이드상세설명_외부공유용_KO.md](./reports/ETRI_연구진행상황공유_20260816_슬라이드상세설명_외부공유용_KO.md) | 위 자료의 외부공유용 국문판 |
| [reports/ETRI_Research_Progress_Update_20260816_Slide_Detailed_Notes_EN.md](./reports/ETRI_Research_Progress_Update_20260816_Slide_Detailed_Notes_EN.md) | 2026-08-16 진행상황 슬라이드 상세설명 (내부, 영문) |
| [reports/ETRI_Research_Progress_Update_20260816_Slide_Detailed_Notes_External_EN.md](./reports/ETRI_Research_Progress_Update_20260816_Slide_Detailed_Notes_External_EN.md) | 위 자료의 외부공유용 영문판 |
| [reports/ETRI_20260816_부록슬라이드_상세설명_KO.md](./reports/ETRI_20260816_부록슬라이드_상세설명_KO.md) | 부록 슬라이드 상세설명 (국문) |
| [reports/ETRI_20260816_Appendix_Slide_Explanations_EN.md](./reports/ETRI_20260816_Appendix_Slide_Explanations_EN.md) | 부록 슬라이드 상세설명 (영문) |

## 10. 내부 메모 (`notes/`)

| 문서 | 내용 |
|---|---|
| [notes/_paper_writing_notes.md](./notes/_paper_writing_notes.md) | 내부 논문 작성 메모 |

## Phase 현황 요약

| Phase | 상태 | 완료 기준 |
|-------|------|-----------|
| 1 | ✅ | AWGN 단일 이미지/폴더 추론 CLI |
| 2 | ✅ | channels/guidance/models/pipelines 분리 + `_defaults_` config 합성 |
| 3 | ✅ | 평가기 모음, SNR-sweep CSV, depth/seg 가이드, regeneration loop |
| 4 | ✅ | 패킷 인식 검증기 + 적응형 가이드(4-A), 키프레임/시간적 파이프라인(4-B) |
| 5 | ✅ 스캐폴드 | 채널 조건화(5-A), 저지연 샘플링/early-exit(5-B), SRS-v2 + regeneration search(5-C) |

모든 Phase 4/5 기능은 **기본값 off**다. 상위 게이트 `use_phase4` / `use_phase5`가
`false`이면 개별 플래그를 명시적으로 켜도 무시되며, 원본 SGD-JSCC 추론과 수치적으로
동일하게 동작한다. 상세 현재 상태는 [etri_strategy.md](./etri_strategy.md) 참고.

## 개발 원칙

1. **알고리즘 경로 보존** — `SGDJSCC/inference_one.py`의 forward 수치를 그대로 유지한다:
   VAE scaling factor `15.45`, AWGN 잡음 주입, blind SNR 예측, step matching,
   canny 재전송, 최종 decode.
2. **관심사 분리** — 채널은 `channels/`, 가이드는 `guidance/`, 모델은 `models/`,
   오케스트레이션은 `pipelines/`, 지표는 `evaluators/`로 독립 교체 가능하게 둔다.
3. **원본 읽기 전용** — 새 아이디어는 `SGDJSCC/`가 아니라 `sgdjscc_lab/`에 구현한다.

## 디렉터리 구성

```text
src/sgdjscc_lab/
├── config.py, paths.py, runtime.py, io.py, phase_gates.py, paper_mode.py
├── core/           stage definitions · noise schedule · semantic vocabulary
├── channels/       awgn · rayleigh · fast_fading · packet_drop · digital_packet · measurement · complex_ops
├── guidance/       text · edge · depth · segmentation · semantic_packet · object · relation
├── models/         jscc_model · diffusion_wrapper(_channel) · edge_jscc · csi_estimation
├── pipelines/      infer · eval · train · regeneration_loop · channel_conditioned_infer
├── evaluators/     quality · clip_score · object_preservation · hallucination(_vqa)
│                   · semantic_reliability(_v2) · packet_matcher · temporal_consistency · fid
├── controllers/    adaptive_guidance · snr_guidance · regeneration · channel_condition · search
├── acceleration/   ddim_sampler · consistency_decoder · early_exit · latency_profiler · water_filling
├── transmission/   real digital packet path (2026-08-17/18) — quantization(4/6/8/16-bit)
│                   · wire_packet(checksum) · packet_bundle · byte_accounting · receiver_runtime
├── video/          keyframe · temporal_pipeline · generation contracts/backends/worker/factory
├── training/       stage validation · stage_runners · losses · freeze · interrupt · perf
├── data/           datasets · image_dataset · transforms
└── utils/          preprocessing · csv_logger · metrics_io · metric_profiles · packet_io · seed
```

Phase 1~3 코어는 `config/runtime/io` + `channels/guidance/models/pipelines/evaluators`,
Phase 4/5는 `controllers/acceleration/video` + 확장 채널·평가기다. 자세한 매핑은
[framework_file_roles.md](./framework_file_roles.md) 참조.
