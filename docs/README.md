# sgdjscc_lab 개발 계획

## 목적

`sgdjscc_lab`은 원본 `SGDJSCC/` 패키지를 수정하지 않고 `SGDJSCC`를 확장하기 위한
연구·개발 fork이다. 원본 저장소는 읽기 전용 참조이자 논문 베이스라인으로 유지하고,
`sgdjscc_lab`은 모듈화·평가·향후 연구를 위한 깨끗한 패키지 역할을 한다.

이 파일은 **색인(index)**이다. 상세 내용은 주제별 문서로 분리되어 있다(아래 "문서 지도" 참조).

---

## 문서 지도

| 문서 | 내용 |
|---|---|
| [phases_1to3.md](./phases_1to3.md) | Phase 1 / 2 / 3 요약 (추론 CLI, 모듈형 패키지, 평가 프레임워크 + SRS) |
| [limitation_reference_map.md](./limitation_reference_map.md) | Phase 4/5용 외부 참고자료: SGD-JSCC 한계점 우선순위, 참고 표, 참고문헌 매핑 (FAST-GSC / DiffCom / LDM-SemCom) |
| [etri_development_roadmap.md](./etri_development_roadmap.md) | ETRI 지향 개발 순서: 과제 목표 8가지와 우선순위 한계점 `1/2/5/6` 개선 |
| [phase4.md](./phase4.md) | Phase 4 계획 + 구현 현황: 4-A 패킷 인식 검증기 + 적응형 가이드, 4-B 키프레임/시간적; 제공 모듈, config/CLI 사용법, 한계 |
| [phase5.md](./phase5.md) | Phase 5 계획 + 구현 현황: 5-A 채널 조건화, 5-B 저지연/consistency, 5-C 검증기/search; 모듈별 `구현됨 / 연결됨 / 근사됨 / fallback / 미구현` 태그, 통합 현황, 해결된 한계 + 남은 한계 |
| [training_scaffold.md](./training_scaffold.md) | 학습 CLI 설계: `scripts/train.py`, `data/`, `training/`, 파이프라인 모듈, 손실 스캐폴드, 체크포인트 전략, 확장 가이드 |
| [framework_comparison.md](./framework_comparison.md) | 원본 `SGDJSCC/` vs `sgdjscc_lab/` 구조 비교 |
| [framework_file_roles.md](./framework_file_roles.md) | 실행 순서에 따른 파일별 프레임워크 역할 지도 |

---

## Phase 현황

| Phase | 상태 | 완료 기준 |
|-------|--------|---------------------|
| 1 | ✅ 완료 | `python scripts/infer_images.py --config configs/default.yaml`로 AWGN 추론 실행 |
| 2 | ✅ 완료 | channels / guidance / models / pipelines 분리, `_defaults_` composition |
| 3 | ✅ 완료 | 전체 평가기 모음, SNR-sweep CSV, depth/seg 가이드, regeneration loop |
| 4 | ✅ 완료 | Phase 4-A 패킷 인식 검증기 + 적응형 가이드; Phase 4-B 키프레임/시간적 파이프라인 ([phase4_status.md](./phase4.md) 참조) |
| 5 | ✅ 스캐폴드 | Phase 5-A 채널 조건화 확산(Rayleigh/fast-fading/packet-drop + 측정 번들), 5-B 저지연 샘플링/consistency/early-exit, 5-C SRS-v2 + regeneration search ([phase5_status.md](./phase5.md) 참조) |

---

## 저장소 전략

### `SGDJSCC/`
- 원본 코드 보존
- 재현 참조
- 논문 베이스라인
- `sgdjscc_lab`의 연구 반복(iteration)으로 절대 수정하지 않음

### `sgdjscc_lab/`
- 깨끗한 연구 fork
- config 기반 CLI
- 구조적 재구성
- 평가기 및 실험 프레임워크
- 향후 가이드 / 채널 / 비디오 확장

---

## 현재 디렉터리 구성

```text
sgdjscc_lab/
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── composed.yaml
│   ├── channel/awgn.yaml
│   ├── model/sgdjscc.yaml
│   ├── infer/awgn.yaml
│   ├── eval/default.yaml
│   └── dataset/
│       ├── kodak.yaml
│       ├── coco.yaml
│       └── ade20k.yaml
├── scripts/
│   ├── infer_images.py
│   └── evaluate.py
├── src/sgdjscc_lab/
│   ├── config.py
│   ├── io.py
│   ├── runtime.py
│   ├── channels/
│   │   └── awgn.py
│   ├── guidance/
│   │   ├── text_extractor.py
│   │   ├── edge_extractor.py
│   │   ├── depth_extractor.py
│   │   └── segmentation_extractor.py
│   ├── models/
│   │   ├── jscc_model.py
│   │   ├── diffusion_wrapper.py
│   │   └── model_bundle.py
│   ├── pipelines/
│   │   ├── infer_pipeline.py
│   │   ├── eval_pipeline.py
│   │   └── regeneration_loop.py
│   ├── evaluators/
│   │   ├── quality.py
│   │   ├── clip_score.py
│   │   ├── object_preservation.py
│   │   ├── hallucination.py
│   │   └── semantic_reliability.py
│   └── utils/
│       ├── preprocessing.py
│       ├── memory.py
│       ├── seed.py
│       ├── csv_logger.py
│       └── metrics_io.py
└── tests/
    ├── test_config.py
    ├── test_io.py
    ├── test_channels.py
    ├── test_evaluators.py
    └── test_eval_pipeline.py
```

> 위 구성은 Phase 1–3 핵심을 보여준다. Phase 4/5는 `controllers/`, `acceleration/`,
> `video/`, 추가 `channels/` `guidance/` `evaluators/` 모듈, 그리고 추가 config
> preset을 더한다 — 전체 모듈 목록은 [phase4_status.md](./phase4.md)와
> [phase5_status.md](./phase5.md)를 참조한다.

---

## 개발 원칙

### 원칙 1: 원본 알고리즘 경로 보존

모든 핵심 forward-pass 연산은 원본 `SGDJSCC/inference_one.py`와 정합을 유지한다:

- scaling factor `15.45`를 사용하는 VAE encode/decode
- AWGN 잡음 주입
- blind SNR 예측
- step matching
- canny 재전송
- canny latent VAE 인코딩
- diffusion generate 경로
- 최종 정규화 decode

### 원칙 2: 연구 아이디어를 추가하기 전에 인터페이스 분리

각 관심사를 독립적으로 교체할 수 있도록 패키지를 설계한다:

- 채널 모델은 `channels/`
- 시맨틱·구조 추출기는 `guidance/`
- JSCC와 diffusion wrapper는 `models/`
- 추론·평가 오케스트레이션은 `pipelines/`
- 연구 지표는 `evaluators/`

### 원칙 3: 원본 저장소는 읽기 전용 유지

새 아이디어는 `SGDJSCC/`가 아니라 `sgdjscc_lab/`에 구현해야 한다.

---

## 권장 연구 워크플로

1. `SGDJSCC/`는 논문 참조 베이스라인으로만 사용한다.
2. 추론과 평가는 `sgdjscc_lab/`에서 실행한다.
3. 새 가이드·채널·평가기 모듈은 모듈형 패키지 내부에 추가한다.
4. 비디오나 새 채널로 확장하기 전에 Phase 3 지표로 아이디어를 비교한다.

---

## 관련 문서

- [../README.md](../README.md) — 사용자 대상 패키지 사용법
- [framework_comparison.md](./framework_comparison.md) — 원본 `SGDJSCC` vs `sgdjscc_lab` 구조 비교
- [framework_file_roles.md](./framework_file_roles.md) — 파일별 프레임워크 역할 지도
