> [← 문서 색인](./README.md)

# Phase 4 — 패킷 인식 평가 + 영상 확장

> **한 문장:** SGD-JSCC forward pass는 한 줄도 바꾸지 않고, 그 위에 *의미 단위
> 평가·제어(4-A)* 와 *키프레임 영상 확장(4-B)* 을 얹은 레이어. 안 켜면 Phase 3와 동일.

## 마스터 스위치

모든 Phase 4 기능은 **기본값 off**다. 상위 게이트 `use_phase4` 하나로 제어한다.

```yaml
# configs/eval/default.yaml
use_phase4: false          # 기본 — 4-A/B 전체 비활성화

# Phase 4만 활성화
use_phase4: true
use_packet_eval: true          # 패킷 빌드 + srs_base/srs_packet + 오류 카운트
use_adaptive_guidance: true    # SNR 구간별 가이드 스케일링
use_packet_regeneration: false # 오류 유형 인식 재시도 (use_packet_eval 필요)
```

`use_phase4: false`이면 개별 플래그를 `true`로 둬도 무시되고, Phase 4-B
(`evaluate_video.py`)는 에러로 종료한다. 런타임 체크는 모두
`phase_gates.effective_flag(cfg, flag, phase=4)`를 거친다. Phase 4와 5는 독립
스위치다(`use_phase5: true`가 4를 켜지 않음).

---

## 4-A: 신뢰도 우선 이미지 확장

기존 SRS(전체 유사도)를 **무엇이 어떻게 틀렸는지**로 분해한다.

| 영역 | 모듈 |
|---|---|
| 시맨틱 패킷 | `guidance/semantic_packet_extractor.py`, `object_extractor.py`, `relation_extractor.py`, `importance_estimator.py`, `utils/packet_io.py` |
| 적응형 가이드 | `controllers/snr_guidance_policy.py`, `adaptive_guidance_controller.py` |
| 패킷 인식 검증기 | `evaluators/semantic_packet_matcher.py`, `relation_consistency.py`, `attribute_consistency.py` |
| SRS 확장 | `evaluators/semantic_reliability.py` (`srs_base`, `srs_packet`) |
| 구조화된 regeneration | `controllers/regeneration_policy.py` (실패 양상별 재시도) |
| 평가 통합 | `pipelines/eval_pipeline.py` |

- **시맨틱 패킷** — 캡션·객체·장면·관계·속성·엣지/세그/depth 요약을 하나의 "의미
  명세서"로 구성해 이미지 옆에 `packet.json`으로 직렬화(채널 전송은 아직 안 함).
- **패킷 검증기** — 원본 vs 복원 패킷을 비교해 누락/추가 객체, 관계·속성 오류를
  **개수로** 집계.
- **적응형 가이드** — 추정 SNR에 따라 가이드 강도·확산 스텝을 조절
  (저 SNR: 강한 가이드 + 최대 스텝 / 고 SNR: 약한 가이드 또는 skip 경로).
- **구조화된 regeneration** — 객체 누락→텍스트 가이드 강화, 할루시네이션→텍스트 약화,
  구조 왜곡→제어 신호·스텝 증가.

**실행**

```bash
# eval/default.yaml에서 use_phase4/use_packet_eval: true 설정 후
python scripts/evaluate.py --config configs/composed.yaml --snr 0 -i ../inputs/
```

이미지별 출력: `<stem>.orig_packet.json`, `.packet.json`, `.error_report.json`.
CSV에 `srs_base, srs_packet, object_match_rate, relation_consistency,
attribute_consistency, segmentation_consistency, scene_match,
missing/additional_object_count, relation/attribute_error_count, guidance_regime`이
추가된다.

---

## 4-B: 키프레임 / 시간적 확장

영상을 매 프레임 통째로 보내는 대신, **키프레임만 전부 전송**하고 나머지는 바뀐
부분(시맨틱 델타)만 전송/재사용한다.

| 영역 | 모듈 |
|---|---|
| 장면 전환 | `video/scene_change_detector.py` (히스토그램 + 선택적 CLIP/LPIPS) |
| 키프레임/GOP | `video/keyframe_extractor.py` |
| 시맨틱 델타 | `video/semantic_delta.py` |
| 모션 residual | `video/motion_residual.py` |
| 시간적 파이프라인 | `video/temporal_pipeline.py` (키프레임 full / 인터프레임 재사용+델타) |
| 시간적 지표 | `evaluators/temporal_consistency.py` |
| CLI/config | `scripts/evaluate_video.py`, `configs/{video/default,composed_video}.yaml` |

**실행**

```bash
# 전체 실행 (SGD-JSCC + CLIP/BLIP2)
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /path/ordered_frames/ --snr 5 --device cuda:0

# dry run (체크포인트 없음, 캡션 주면 델타/지표가 의미를 가짐)
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /path/ordered_frames/ --no-models --captions /path/captions.txt
```

출력: `keyframes.json`(GOP 구조), `temporal_frames.csv`(프레임별), `temporal_metrics.csv`
(시퀀스 지표 + `overhead_reduction` = 프레임별 전체 전송 대비 시맨틱 유닛 절감).
이미지 평가기의 패킷/오류 JSON은 SNR별 namespace(`packet_dir/snr_<snr>/…`)로 분리된다.

---

## 참고 매핑 · 한계

- **FAST-GSC** → 시맨틱 유닛 패킷 설계, 중요도 기반 전송 순서, 시맨틱 차이 계산,
  단계적 조건부 디노이징 근사(`temporal_pipeline.build_staged_schedule`).
- **SGD-JSCC** → 프레임별 복원 경로를 그대로 재사용. 적응형 가이드는 그 변경되지
  않은 경로가 *어떤* config로 실행될지만 선택한다.

**알려진 한계**
- 패킷은 메타데이터일 뿐 — 실제 시맨틱 패킷 채널 코딩/drop 시뮬레이션은 Phase 5 보류.
- 객체/관계 추출은 CLIP/캡션 휴리스틱(scene-graph·POPE-VQA 아님).
- 단계적 디노이징은 **prompt 레벨** 연결(`cfg.prompt_override`)이다. 샘플러 루프
  *내부*의 스텝별 prompt 전환은 SGD-JSCC 샘플러 수정이 필요해(알고리즘 보존 불변식)
  구현하지 않았다.
- 인터프레임 재사용은 키프레임 복원을 복사한다(진정한 델타-워프/모션 보상은 향후 작업).

---

### 관련 문서
- [training_scaffold.md](./training_scaffold.md) · [paper_alignment.md](./paper_alignment.md) · [phase5.md](./phase5.md)
</content>
