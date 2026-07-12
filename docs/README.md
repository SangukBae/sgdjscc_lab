# sgdjscc_lab 개발 문서

## 목적

`sgdjscc_lab`은 원본 `SGDJSCC/` 패키지를 **수정하지 않고** 확장하는 연구용 fork다.
원본은 논문 베이스라인이자 읽기 전용 참조로 두고, 이 패키지는 모듈화·평가·연구
확장을 위한 깨끗한 계층을 얹는다. 추론 forward-pass는 원본과 수치적으로 동일하다.

이 파일은 **문서 색인**이다. 문서 수를 줄이기 위해 전략 문서와 논문 정합 문서를
각각 하나로 합쳤고, 오래된 상세 문서는 `archive/`, 개발자용 점검 문서는 `dev/`,
설명/메모 문서는 `notes/`로 내렸다.

## 핵심 문서

| 문서 | 내용 |
|---|---|
| [etri_overview.md](./etri_overview.md) | ETRI 과제 목표, 전체 파이프라인, SRS, 실험 설정 |
| [etri_strategy.md](./etri_strategy.md) | 핵심 한계 3가지, 개발 순서, 현재 구현 상태를 합친 전략 문서 |
| [paper_alignment.md](./paper_alignment.md) | 논문 정합성, `paper_mode`, 하이퍼파라미터 출처를 합친 정리 문서 |
| [framework_file_roles.md](./framework_file_roles.md) | 파일별 실행 흐름과 역할 지도 |
| [training_scaffold.md](./training_scaffold.md) | 학습 CLI: 논문 3-stage(`jscc`/`text_dm`/`controlnet`) + 보조 stage + 데이터 준비 |
| [phase4.md](./phase4.md) | Phase 4: 패킷 인식 검증기 + 적응형 가이드(4-A), 키프레임/시간적 파이프라인(4-B) |
| [phase5.md](./phase5.md) | Phase 5: 채널 조건화(5-A), 저지연/consistency(5-B), SRS-v2/regeneration search(5-C) |
| [dataset_status.md](./dataset_status.md) | 데이터셋 역할·stage 매핑·변환 워크플로 |

## 보조 문서

| 문서 | 내용 |
|---|---|
| [dev/smoke_training.md](./dev/smoke_training.md) | real-model smoke 학습(1~2 step 배선 검증) |
| [notes/etri_qna_reply.md](./notes/etri_qna_reply.md) | 비전문가 설명용 Q&A 문서 |
| [archive/phases_1to3.md](./archive/phases_1to3.md) | 초기 Phase 1~3 스냅샷 |
| [archive/etri_development_plan_v2.md](./archive/etri_development_plan_v2.md) | 통합 전 개발계획서 보관본 |
| [archive/etri_development_roadmap.md](./archive/etri_development_roadmap.md) | 통합 전 로드맵 보관본 |
| [archive/limitation_reference_map.md](./archive/limitation_reference_map.md) | 통합 전 한계점 지도 보관본 |
| [archive/framework_comparison.md](./archive/framework_comparison.md) | 통합 전 프레임워크 비교 보관본 |
| [archive/paper_gap_closure.md](./archive/paper_gap_closure.md) | 통합 전 paper-mode 정책 문서 보관본 |
| [archive/paper_training_alignment.md](./archive/paper_training_alignment.md) | 통합 전 학습 정합 문서 보관본 |
| [notes/_paper_writing_notes.md](./notes/_paper_writing_notes.md) | 내부 논문 작성 메모 |

## Phase 현황

| Phase | 상태 | 완료 기준 |
|-------|------|-----------|
| 1 | ✅ | AWGN 단일 이미지/폴더 추론 CLI |
| 2 | ✅ | channels/guidance/models/pipelines 분리 + `_defaults_` config 합성 |
| 3 | ✅ | 평가기 모음, SNR-sweep CSV, depth/seg 가이드, regeneration loop |
| 4 | ✅ | 패킷 인식 검증기 + 적응형 가이드(4-A), 키프레임/시간적 파이프라인(4-B) |
| 5 | ✅ 스캐폴드 | 채널 조건화(5-A), 저지연 샘플링/early-exit(5-B), SRS-v2 + regeneration search(5-C) |

모든 Phase 4/5 기능은 **기본값 off**다. 상위 게이트 `use_phase4` / `use_phase5`가
`false`이면 개별 플래그를 명시적으로 켜도 무시되며, 원본 SGD-JSCC 추론과 수치적으로
동일하게 동작한다.

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
├── config.py, runtime.py, io.py, phase_gates.py, paper_mode.py, distributed.py
├── channels/       awgn · rayleigh · fast_fading · packet_drop · measurement · complex_ops
├── guidance/       text · edge · depth · segmentation · semantic_packet · object · relation
├── models/         jscc_model · diffusion_wrapper(_channel) · edge_jscc · csi_estimation
├── pipelines/      infer · eval · train · regeneration_loop · channel_conditioned_infer
├── evaluators/     quality · clip_score · object_preservation · hallucination(_vqa)
│                   · semantic_reliability(_v2) · packet_matcher · temporal_consistency · fid
├── controllers/    adaptive_guidance · snr_guidance · regeneration · channel_condition · search
├── acceleration/   ddim_sampler · consistency_decoder · early_exit · latency_profiler · water_filling
├── video/          keyframe · scene_change · semantic_delta · motion_residual · temporal_pipeline
├── training/       stages · stage_runners · losses · freeze · noise_schedule · interrupt · perf
├── data/           datasets · image_dataset · transforms
└── utils/          preprocessing · csv_logger · metrics_io · metric_profiles · packet_io · seed
```

Phase 1~3 코어는 `config/runtime/io` + `channels/guidance/models/pipelines/evaluators`,
Phase 4/5는 `controllers/acceleration/video` + 확장 채널·평가기다. 자세한 매핑은
[framework_file_roles.md](./framework_file_roles.md) 참조.

## 관련 문서

- [../README.md](../README.md) — 사용자 대상 패키지 사용법
</content>
</invoke>
