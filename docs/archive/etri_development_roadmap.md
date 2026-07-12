> [← 문서 색인](../README.md)

# ETRI 개발 로드맵

> 보관본. 최신 요약은 [etri_strategy.md](../etri_strategy.md)를 우선 참조.

ETRI 과제 목표([etri_overview.md](../etri_overview.md))와 핵심 한계 3가지
([limitation_reference_map.md](./limitation_reference_map.md))를 실제 개발 순서로
결합한다. 정렬 원칙: **측정 체계와 소프트웨어 베이스라인을 먼저 세운 뒤, 그 위에
연구적 개선을 얹는다.**

## Phase 마스터 스위치

Phase 4/5 확장은 **기본값 off**다. 상위 게이트 `use_phase4` / `use_phase5`가 `true`일
때만 개별 플래그가 효력을 갖고, `false`이면 무시된다. `use_phase5: true`여도
`use_phase4`는 자동으로 켜지지 않는다. 상세: [phase4.md](../phase4.md#마스터-스위치) ·
[phase5.md](../phase5.md#마스터-스위치).

## 권장 개발 순서 (1→12)

| 단계 | 항목 | 근거 |
|---|---|---|
| 1~2 | 원본 경로 보존 + 모듈형 구조 정리 | 베이스라인 재현성을 먼저 고정해야 이후 비교를 신뢰할 수 있다 |
| 3~4 | End-to-End 평가 골격 + 시맨틱 우선 철학 확립 | 최적화 전에 입력→채널→복원→평가→CSV 경로와, 픽셀이 아닌 의미 중심 성공 기준이 필요 |
| 5~6 | 의미 평가기 모음 + SRS 통합 | 의미 충실도·신뢰성 지표가 먼저 안정돼야 이후 개선 효과를 검증할 수 있다 |
| 7 | 핵심 한계 1 — 할루시네이션 완화와 객체 추가·누락·왜곡 검출 | ETRI 핵심 문제: 그럴듯하지만 틀린 복원을 막고 정량 검출해야 한다 |
| 8 | 핵심 한계 2 — 화질 중심 평가에서 의미 충실도·신뢰성 평가 체계로 전환 | `PSNR`/`SSIM`만으로는 부족하므로 의미 유사도·객체 보존·누락·할루시네이션 점수를 확립한다 |
| 9 | 핵심 한계 3 — 정지 이미지에서 영상·장면 전환·시간 일관성으로 확장 | keyframe 기반 비디오와 temporal consistency를 실험 축으로 편입한다 |
| 10 | 보조 축 — 가이드 손상·오버헤드 견고성 | 캡션·엣지 손상과 부가정보 비용은 핵심 3축 위에서 다룬다 |
| 11 | 보조 축 — 저지연 복원 | few-step/consistency 가속은 신뢰도 평가 체계가 선행된 뒤 평가한다 |
| 12 | 보조 축 — 블라인드/페이딩 채널 견고성 + 공정 비교 프로토콜 | 채널 조건화와 최종 비교 규칙은 핵심 경로 안정화 뒤 고정한다 |

1번이 먼저인 것은 할루시네이션이 신뢰도 목표를 가장 직접적으로 위반하기 때문이고,
2번은 그 문제를 입증하는 평가 기반이며, 3번은 정지 이미지 실험을 실제 영상 시나리오로
넓히는 확장 축이기 때문이다.

## 현재 구현 현황

| 단계 | 상태 | 요약 |
|---|---|---|
| 1~4 | 완료 | 원본 경로 보존, 모듈화(`channels/guidance/models/pipelines/evaluators/controllers/acceleration/video`), end-to-end 평가, 시맨틱 우선 지표 |
| 5~6 | 완료 | 품질·CLIP·패킷·시간적·VQA 지표 구현, `srs_base/srs_packet/srs_v2`가 config로 활성화 |
| 7 | 부분 | 패킷 검증기·regeneration search·VQA·SRS-v2 구현. 가이드 오류로 인한 그럴듯한 오출력은 여전히 가능하고, 검증 일부는 휴리스틱 |
| 8 | 완료에 근접 | SRS 계열, 객체 보존/누락/추가, CLIP 기반 의미 지표, CSV 기록 경로가 구축됨 |
| 9 | 부분 | keyframe/scene-change/temporal evaluator/`overhead_reduction` 구현. 진정한 motion residual 통합과 더 강한 시간 의미 평가는 추가 작업 필요 |
| 10 | 제한/부분 | `packet_drop`, 세그 영역 dropout, edge codec 등 손상 요소가 있으나 완전한 가이드 전용 손상 프레임워크는 미확정 |
| 11 | 부분/스캐폴드 | DDIM step-budget·동적 라우팅·early-exit·지연 프로파일링 존재. 학습된 distilled consistency 디코더는 placeholder |
| 12 | 부분 | Rayleigh/fast-fading/packet-drop 채널·측정 번들·조건화 추론 존재. 블라인드 견고성과 최종 비교 프로토콜은 제한적 |

**요약**: 단계 1~6은 평가 체계·베이스라인 확립, 7~9는 핵심 한계 3축 대응, 10~12는
보조 연구 축과 실험 설계 마무리다.

## 학습 재현성 (논문 3-stage + 확장)

stage 인식 학습 프레임워크가 논문 재현을 뒷받침한다([training_scaffold.md](../training_scaffold.md)).

- **Core baseline = Stage 1/2/3** (`jscc`, `text_dm`, `controlnet`).
- **Stage 3 엣지 transport**: baseline은 전용 `edge_jscc` 링크, `shared_vae`는 ablation.
- **supporting `edge_codec`**: stage 3 codec을 BCE+Dice로 실제 학습(무작위 stand-in 아님).
- **운영 규모**: step 기반 학습(`max_steps`)·grad accumulation·AMP로 논문의 ~250k-step
  DM 스케줄을 실제 데이터에서 돌릴 수 있다.
- **extension**: `end_to_end_ft`는 baseline 아닌 추가 실험.

남은 차이: patch-GAN 수치·`edge_codec` 학습 데이터/스케줄은 논문 수치 미보장,
`end_to_end_ft`는 single-step denoise, ~14M pair 오픈 데이터셋은 미번들(loader만 제공).

## 관련 문서
- [../etri_strategy.md](../etri_strategy.md) · [../etri_overview.md](../etri_overview.md) · [../phase4.md](../phase4.md) · [../phase5.md](../phase5.md)
</content>
