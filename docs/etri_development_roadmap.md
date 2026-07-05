> [← 문서 색인](./README.md)

# ETRI 개발 로드맵

ETRI 과제 목표([etri_overview.md](./etri_overview.md))와 우선순위 SGD-JSCC 한계점
`A~D`([limitation_reference_map.md](./limitation_reference_map.md))를 실제 개발 순서로
결합한다. 정렬 원칙: **측정 체계와 소프트웨어 베이스라인을 먼저 세운 뒤, 그 위에
연구적 개선을 얹는다.**

## Phase 마스터 스위치

Phase 4/5 확장은 **기본값 off**다. 상위 게이트 `use_phase4` / `use_phase5`가 `true`일
때만 개별 플래그가 효력을 갖고, `false`이면 무시된다. `use_phase5: true`여도
`use_phase4`는 자동으로 켜지지 않는다. 상세: [phase4.md](./phase4.md#마스터-스위치) ·
[phase5.md](./phase5.md#마스터-스위치).

## 권장 개발 순서 (1→12)

| 단계 | 항목 | 근거 |
|---|---|---|
| 1~2 | 원본 경로 보존 + 모듈형 구조 정리 | 베이스라인 재현성을 먼저 고정해야 이후 비교를 신뢰할 수 있다 |
| 3~4 | End-to-End 평가 골격 + 시맨틱 우선 철학 확립 | 최적화 전에 입력→채널→복원→평가→CSV 경로와, 픽셀이 아닌 의미 중심 성공 기준이 필요 |
| 5~6 | 할루시네이션 포함 평가기 모음 + SRS 통합 | 구성 지표가 안정된 뒤에야 SRS를 최종 확정 |
| 7 | 한계 `A` — 가이드 하 할루시네이션·시맨틱 불일치 감소 | ETRI 핵심 문제: 그럴듯하지만 틀린 복원을 막고 검증된 신뢰도를 높인다 |
| 8 | 한계 `B` — 시맨틱 부가정보를 견고·경량으로 | 캡션·엣지가 손상 상황에서도 유용하고 과한 오버헤드를 피해야 |
| 9 | 한계 `C` — 확산 복원 지연 단축 | few-step/consistency 가속은 신뢰도 지표·검증기가 갖춰진 뒤 평가 |
| 10 | 한계 `D` — 블라인드/페이딩 채널 견고성 | 채널 조건화는 핵심 신뢰도·지연 경로가 안정된 뒤 확장 |
| 11~12 | 가이드 손상 모델 분리 + 공정 비교 프로토콜 확정 | 파이프라인·지표·개선이 정착된 뒤 맨 마지막에 고정 |

`A`가 먼저인 것은 할루시네이션·시맨틱 불일치가 신뢰도 목표에 가장 직접 반하기
때문이고, `D`가 마지막인 것은 페이딩 견고성이 핵심 신뢰도·지연 경로 다음에 와야
하기 때문이다.

## 현재 구현 현황

| 단계 | 상태 | 요약 |
|---|---|---|
| 1~4 | 완료 | 원본 경로 보존, 모듈화(`channels/guidance/models/pipelines/evaluators/controllers/acceleration/video`), end-to-end 평가, 시맨틱 우선 지표 |
| 5~6 | 완료 | 품질·CLIP·패킷·시간적·VQA 지표 구현, `srs_base/srs_packet/srs_v2`가 config로 활성화 |
| 7 | 부분 | 패킷 검증기·regeneration search·적응형 가이드·VQA·SRS-v2 구현. 가이드 오류는 여전히 그럴듯한 오출력 가능, 검증 일부는 휴리스틱 |
| 8 | 부분 | 시맨틱 델타 전송·패킷 재사용·손상 훅·`overhead_reduction`. 진정한 부가정보 코딩·drop 인식 제어는 미완 |
| 9 | 부분/스캐폴드 | DDIM step-budget·동적 라우팅·early-exit·지연 프로파일링·벤치마크 CLI. 학습된 distilled consistency 디코더는 placeholder |
| 10 | 부분 | Rayleigh/fast-fading/packet-drop 채널·측정 번들·조건화 추론. 블라인드 견고성 제한적 |
| 11~12 | 제한/부분 | `packet_drop`·세그 영역 dropout 등 손상 요소 존재. 완전한 가이드 전용 손상 프레임워크·최종 고정 비교 프로토콜은 미확정 |

**요약**: 단계 1~6은 평가 체계·베이스라인 확립, 7~10은 한계 `A~D` 개선, 11~12는
실험 설계·비교 프로토콜 마무리.

## 학습 재현성 (논문 3-stage + 확장)

stage 인식 학습 프레임워크가 논문 재현을 뒷받침한다([training_scaffold.md](./training_scaffold.md)).

- **Core baseline = Stage 1/2/3** (`jscc`, `text_dm`, `controlnet`).
- **Stage 3 엣지 transport**: baseline은 전용 `edge_jscc` 링크, `shared_vae`는 ablation.
- **supporting `edge_codec`**: stage 3 codec을 BCE+Dice로 실제 학습(무작위 stand-in 아님).
- **운영 규모**: step 기반 학습(`max_steps`)·grad accumulation·AMP로 논문의 ~250k-step
  DM 스케줄을 실제 데이터에서 돌릴 수 있다.
- **extension**: `end_to_end_ft`는 baseline 아닌 추가 실험.

남은 차이: patch-GAN 수치·`edge_codec` 학습 데이터/스케줄은 논문 수치 미보장,
`end_to_end_ft`는 single-step denoise, ~14M pair 오픈 데이터셋은 미번들(loader만 제공).

## 관련 문서
- [etri_overview.md](./etri_overview.md) · [limitation_reference_map.md](./limitation_reference_map.md) · [phase4.md](./phase4.md) · [phase5.md](./phase5.md)
</content>
