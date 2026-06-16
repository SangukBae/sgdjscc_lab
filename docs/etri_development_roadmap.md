> [← 문서 색인](./README.md)

# ETRI 개발 로드맵

이 문서는 ETRI 과제를 실제 개발 순서에 맞게 재구성한 것이다.
다음 두 가지를 결합한다.

- [etri_overview.md](./etri_overview.md)의 ETRI 과제 목표 8가지
- [limitation_reference_map.md](./limitation_reference_map.md)의 우선순위가 매겨진
  SGD-JSCC 한계점 `A`, `B`, `C`, `D`

정렬 원칙은 단순하다. **측정 체계와 소프트웨어 베이스라인을 먼저 구축한 뒤,
그 위에 연구적 개선을 얹는다.**

## Phase 마스터 스위치

Phase 4와 Phase 5의 모든 확장 기능은 **기본값 `false`**로 꺼져 있다.
두 개의 상위 게이트 플래그로 한 번에 제어한다.

```yaml
# configs/eval/default.yaml (또는 composed config)
use_phase4: false   # Phase 4-A/B 전체 비활성화 (기본값)
use_phase5: false   # Phase 5-A/B/C 전체 비활성화 (기본값)
```

| 조합 | 동작 |
|------|------|
| 둘 다 `false` | Phase 1~3 기본 경로만 실행 |
| `use_phase4: true` | Phase 4-A/B 개별 플래그가 효력 발생 |
| `use_phase5: true` | Phase 5-A/B/C 개별 플래그가 효력 발생 |
| 둘 다 `true` | 전체 확장 기능 활성 가능 |

**중요**: `use_phase5: true`여도 `use_phase4`는 자동으로 켜지지 않는다.

개별 플래그(`use_packet_eval`, `use_channel_conditioning` 등)는 상위 마스터 스위치가
`true`일 때만 실제로 반영된다. 마스터 스위치가 `false`이면 개별 플래그 값은 무시된다.

자세한 사용법: [phase4.md#master-switch](./phase4.md#master-switch) / [phase5.md#master-switch](./phase5.md#master-switch)

---

## 권장 개발 순서

| 단계 | 개발 항목 | 이 순서인 이유 | 주요 출처 |
|---|---|---|---|
| 1 | 원본 SGD-JSCC 경로와 확장 규칙 보존 | 베이스라인 재현성이 먼저 고정되어야 한다. 그렇지 않으면 이후 비교를 신뢰할 수 없다. | [etri_overview.md](./etri_overview.md) |
| 2 | 모듈형 소프트웨어 구조 정리 | 연구적 변경이 쌓이기 전에 `channels/`, `guidance/`, `models/`, `pipelines/`, `evaluators/`가 안정적이어야 한다. | [etri_overview.md](./etri_overview.md) |
| 3 | End-to-End 평가 프레임워크 골격 구축 | 최적화 작업을 시작하기 전에 입력 → 채널 → 복원 → 평가 → `results.csv` 로깅 경로가 존재해야 한다. | [etri_overview.md](./etri_overview.md) |
| 4 | 시맨틱 의도 중심의 평가 철학 확립 | 단순 픽셀 충실도가 아니라 시맨틱 보존을 중심으로 성공을 정의해야 한다. | [etri_overview.md](./etri_overview.md) |
| 5 | 할루시네이션 지표를 포함한 평가기 모음 구현 | CLIP, 객체 보존, 누락/추가 객체, 할루시네이션, 품질 지표는 의미 있는 실험의 전제 조건이다. | [etri_overview.md](./etri_overview.md) |
| 6 | SRS를 대표 지표로 통합 | SRS는 구성 지표들이 안정된 뒤에야 최종 확정해야 한다. | [etri_overview.md](./etri_overview.md) |
| 7 | 한계점 `A` 개선: 가이드 하에서의 할루시네이션·시맨틱 불일치 감소 | ETRI의 핵심 문제는 픽셀 충실도가 아니라, 그럴듯하지만 틀린 복원을 막고 검증된 시맨틱 신뢰도를 높이는 것이다. | [limitation_reference_map.md](./limitation_reference_map.md) |
| 8 | 한계점 `B` 개선: 시맨틱 부가정보를 견고하고 경량으로 | 캡션·엣지 가이드는 손상 상황에서도 유용해야 하고, 불필요한 전송 오버헤드를 피해야 한다. | [limitation_reference_map.md](./limitation_reference_map.md) |
| 9 | 한계점 `C` 개선: 확산 복원 지연 시간 단축 | Few-step 디코딩과 consistency 계열 가속은 시맨틱 신뢰도 지표와 검증기 루프가 갖춰진 뒤에 평가해야 한다. | [limitation_reference_map.md](./limitation_reference_map.md) |
| 10 | 한계점 `D` 개선: 강한 CSI 가정을 넘어선 블라인드/페이딩 채널 견고성 추가 | 채널 조건화·블라인드 복원은 핵심 시맨틱 신뢰도와 지연 경로가 안정된 뒤에 확장해야 한다. | [limitation_reference_map.md](./limitation_reference_map.md) |
| 11 | 가이드 손상 모델을 채널 잡음과 분리 | 더 풍부한 가이드가 생기면, 현실적인 실험을 위해 가이드 전용 손상 규칙이 필요해진다. | [etri_overview.md](./etri_overview.md) |
| 12 | 공정 비교 프로토콜 확정 | 파이프라인·지표·개선 기법이 모두 정착된 뒤, 최종 프로토콜 고정은 맨 마지막에 와야 한다. | [etri_overview.md](./etri_overview.md) |

## 현재 구현 현황

| 단계 | 상태 | 현재 구현된 내용 | 남은 과제 |
|---|---|---|---|
| 1 | 완료 | 원본 SGD-JSCC forward 경로가 보존되며, 모든 신규 기능은 그 주위에 얹히는 opt-in 확장이다. | 프레임워크 규칙 수준에서는 없음. |
| 2 | 완료 | 패키지가 이미 모듈형으로 분리됨: `channels/`, `guidance/`, `models/`, `pipelines/`, `evaluators/`, `controllers/`, `acceleration/`, `video/`. | 주요 소프트웨어 구조 측면에서는 없음. |
| 3 | 완료 | `scripts/evaluate.py`와 `pipelines/eval_pipeline.py`가 입력 → 복원 → 지표 → CSV 로깅에 이르는 end-to-end 평가 경로를 제공한다. | 베이스라인 이미지 평가 루프 측면에서는 없음. |
| 4 | 완료 | 평가 스택이 시맨틱 우선: PSNR/SSIM/LPIPS를 넘어 CLIP, 객체 보존, 할루시네이션, SRS가 일급 지표다. | 평가 정책 수준에서는 없음. |
| 5 | 완료 | 품질 지표, CLIP 지표, 패킷 인식 지표, 시간적 지표, VQA 기반 할루시네이션 평가가 구현됨. | 일부 고급 시맨틱 평가기는 완전 학습형이 아니라 여전히 휴리스틱이다. |
| 6 | 완료 | `srs_base`, `srs_packet`, `srs_v2`가 평가 경로에 통합되어 config로 활성화 가능하다. | 논문 수준의 최종 가중치 튜닝은 변경될 수 있다. |
| 7 | 부분 | 패킷 인식 지표, regeneration search, 적응형 가이드, VQA 방식 할루시네이션 점검, `SRS-v2`가 시맨틱 신뢰도 경로를 뒷받침한다. | 가이드 오류가 여전히 그럴듯하지만 틀린 출력을 낼 수 있고, 검증은 일부 휴리스틱으로 남아 있다. |
| 8 | 부분 | 시맨틱 델타 전송 시뮬레이션, 패킷 재사용, 캡션/가이드 손상 훅, `overhead_reduction` 리포팅이 구현됨. | 진정한 시맨틱 부가정보 코딩, 더 강한 손상 견고성, drop 인식 제어는 아직 미완이다. |
| 9 | 부분 / 스캐폴드 | DDIM step-budget 제어, 동적 라우팅, early exit, 지연 시간 프로파일링, 벤치마크 CLI가 구현됨. | 학습된 distilled consistency 디코더는 완성 모델이 아니라 여전히 placeholder다. |
| 10 | 부분 | Rayleigh/fast-fading/packet-drop 채널, 채널 측정 번들, 채널 조건화 추론 모드가 구현됨. | 블라인드 견고성은 아직 제한적이며, 더 강한 non-AWGN 검증이 필요하다. |
| 11 | 제한적 | `packet_drop` 채널 지원, 손상 메커니즘으로 의도된 세그멘테이션 영역 dropout 등 보조 요소가 존재한다. | ETRI 개요에 부합하는 완전한 가이드 전용 손상 프레임워크는 아직 미완이다. |
| 12 | 부분 | 메인 평가 루프가 이미 공유 config 하에서 패킷 평가, 채널 조건화, SRS-v2, regeneration search, 비디오 평가를 지원한다. | 모든 베이스라인·ablation을 아우르는 최종 고정 비교 프로토콜은 아직 완전히 확정되지 않았다. |

## 요약 버전

```text
베이스라인 보존
-> 모듈형 구조
-> end-to-end 파이프라인
-> 시맨틱 우선 평가 철학
-> 평가기 모음
-> SRS 통합
-> 한계점 A 개선
-> 한계점 B 개선
-> 한계점 C 개선
-> 한계점 D 개선
-> 가이드 손상 모델
-> 공정 비교 프로토콜
```

## 한계점 A, B, C, D가 이 순서인 이유

- `A`가 먼저인 이유는, 할루시네이션과 시맨틱 불일치가 ETRI 시맨틱 신뢰도 목표에
  가장 직접적으로 반하는 실패 양상이기 때문이다.
- `B`가 뒤따르는 이유는, 부가정보는 손상 상황에서도 견고하게 유지되고 과도한
  전송 예산을 소비하지 않을 때에만 유용하기 때문이다.
- `C`가 핵심 시맨틱 신뢰도 작업 뒤에 오는 이유는, 지연 시간 단축은 런타임만이
  아니라 검증된 시맨틱 품질을 기준으로 판단해야 하기 때문이다.
- `D`가 한계점 개선 단계 중 마지막인 이유는, 블라인드/페이딩 견고성이 중요하지만
  핵심 신뢰도·지연 경로 다음에 와야 하기 때문이다.

## 실무적 해석

- 단계 `1`~`6`은 ETRI 평가 체계와 소프트웨어 베이스라인을 확립한다.
- 단계 `7`~`10`은 우선순위 SGD-JSCC 한계점 개선을 구현한다.
- 단계 `11`~`12`는 현실적인 실험 설계와 논문 수준의 비교 프로토콜을 마무리한다.

## 학습 재현성 (논문 3-stage + 확장)

이제 stage 인식 학습 프레임워크가 논문 재현을 뒷받침한다
([training_scaffold.md](./training_scaffold.md) 참조).

- **Core baseline = Stage 1 / 2 / 3** (`jscc`, `text_dm`, `controlnet`). 이
  세 stage만이 baseline을 구성하며, 각각 별도 runner·데이터셋·손실·강제 freeze
  정책을 가진다.
- **Stage 3 엣지 transport**: baseline은 전용 `edge_jscc` 링크
  (`models/edge_jscc.py`)이고, `shared_vae`(이미지 VAE stand-in)는 비교용
  **ablation**이다.
- **supporting stage `edge_codec`**: stage 3의 `edge_jscc` codec을 BCE+Dice로
  실제 학습한다(encoder+projector+decoder). 그 체크포인트를 stage 3가 로드해
  baseline을 구성한다 — codec은 더 이상 무작위 stand-in이 아니다.
- **운영 규모**: step 기반 학습(`max_steps`, `save/val/log_every_steps`), gradient
  accumulation, AMP를 통해 논문의 약 250k-step DM 스케줄을 실제 데이터에서
  돌릴 수 있다. `global_step`은 체크포인트에 저장·복원된다.
- **extension (baseline 아님)**: `end_to_end_ft`는 3-stage 이후의 *추가* 실험으로
  JSCC와 DM을 함께 미세조정한다. baseline 비교표에는 포함하지 않는다.

남은 차이: patch-GAN 가중치는 구조적 stand-in이고(논문 LDM-GAN 정확 수치 미보장),
`edge_codec`의 학습 데이터·스케줄은 논문 수치와 동일함을 보장하지 않으며(구조·
BCE+Dice 목적은 일치), `end_to_end_ft` 복원 경로는 전체 reverse 과정 대신
single-step denoise를 사용하고, 약 1,400만 pair 규모의 오픈 데이터셋은 번들되어
있지 않다(loader 인터페이스만 제공).

## 관련 문서

- [etri_overview.md](./etri_overview.md) — ETRI 과제 목표 및 프레임워크 범위
- [limitation_reference_map.md](./limitation_reference_map.md) — 우선순위가 매겨진 SGD-JSCC 한계점 및 참고문헌
- [phase4.md](./phase4.md) — Phase 4 현황 및 설계
- [phase5.md](./phase5.md) — Phase 5 현황 및 설계
