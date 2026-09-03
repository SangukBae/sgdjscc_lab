---
status: completed
updated: 2026-09-03
owner: ETRI SGD-JSCC 연구팀
protocol_version: 1.2.0
gate: G0_PASSED
supersedes: 2026-09-02_negative_semantics_g0_acquisition_amendment_v1_1.md
---

> [← 문서 색인](../README.md) ·
> [v1.1 역사 기록](./2026-09-02_negative_semantics_g0_acquisition_amendment_v1_1.md) ·
> [논문 연구 계획](../current/negative_semantic_paper_plan.md)

# Negative semantics G0 official-GT amendment v1.2

## 판정

G0 Pilot의 사람 A/B/C 검수 조건을 **OVIS provider GT 기반의 결정론적 자동 검증**으로
교체했다. OVIS validation의 서로 다른 실제 영상 40개에서 `ENTER`, `EXIT`, `OCCLUDE`,
`REAPPEAR`를 각 10개 고정했다. 각 이벤트는 서로 다른 영상·independence cluster를
사용하며, 공식 프레임·GT·파생 스크립트·목록의 SHA-256을 기록했다.

`scripts/audit_negative_semantics_g0.py --require-pass`가 13/13, exit 0을 반환하므로
**G0는 `PASSED`이고 G1 시작 조건을 충족한다.** 이는 GPU smoke test나 실제 학습 결과가
아니라 데이터·프로토콜·재현성 gate의 완료다. G1부터 구현 및 GPU 현상 실험을 수행한다.

## 왜 OVIS인가

공식 OVIS 배포는 전체 901개 영상, 25개 category, 5,223개 instance와 296k mask를
제공하며, 정해진 category의 객체를 빠짐없이 instance identity·mask로 주석한다.
가림이 심한 장면을 중심으로 구성됐고 원본 5 frame마다 한 번 주석한다.

- 공식 프로젝트·라이선스: https://songbai.site/ovis/
- 공식 배포 폴더: https://drive.google.com/drive/folders/1eE4lLKCbv54E866XBVce_ebh3oXYq99b?usp=drive_link
- 보수적 이용 조건: CC BY-NC-SA 4.0, 내부 비상업 연구, raw media 비재배포

공식 `train.zip`은 당시 Google Drive 다운로드 한도에 걸렸으나 validation 배포본은
정상 취득됐다. 숫자를 맞추기 위해 중복 영상을 쓰지 않고, ontology에 정확히 매핑되는
provider category를 versioned mapping으로 확장해 validation만 사용했다. `Poultry`와
`Parrot`은 `bird`, provider 철자 `Vehical`은 `car`가 아닌 새 generic `vehicle` concept에
매핑했다. 나머지 category는 `unknown`이다.

## 취득·무결성

| 공식 artifact | bytes | SHA-256 |
|---|---:|---|
| OVIS `valid.zip` | 1,975,600,278 | `94d90115d5935bf71e733562752c5949bbb68148c361b05170c0863cf24003a9` |
| OVIS `annotations_valid_withgt.json` | 59,478,999 | `2fc5592499c7e774c272d5ad24f8835bc05df0ded002a97a05e6ed1440fae91f` |

전체 validation archive의 ZIP CRC를 검사했다. 선택된 40개 영상만
`data/external/ovis/extracted/valid/`에 풀었으며 605 MiB다. 원본 외부 파일은 Git에
넣지 않고, manifest가 영상별 frame tree hash와 per-video official-GT JSON hash를
검사한다.

## 동결된 자동 파생 규칙

모든 단위는 source frame이 아니라 **annotated timestep**이다. 한 timestep은 최대 약
5 source frame 간격이므로 기존 v1.1의 `±1 source frame` 주장은 폐기하고
`±1 annotated timestep`으로 바꿨다.

| 이벤트 | provider GT 기반 규칙 |
|---|---|
| ENTER | 처음 2 timestep은 mask 없음, 이후 3 timestep 이상 연속 mask 있음, 첫 bbox가 2 px frame border 접촉 |
| EXIT | 3 timestep 이상 연속 mask 뒤 끝까지 mask 없음, 마지막 bbox가 2 px frame border 접촉 |
| OCCLUDE | 영상 내부 mask gap 시작; gap 1 timestep 이상, 양쪽 각 3 timestep 이상 visible, 양쪽 occlusion label이 `no_occlusion` 아님, bbox가 5 px border에 닿지 않음 |
| REAPPEAR | 위와 같은 internal gap이 끝나 동일 official annotation/instance ID가 다시 나타나는 timestep |

terminal mask 소실을 무조건 EXIT로 간주하지 않는다. border 조건이 없는 terminal
disappearance와 미주석 category는 `unknown`이다. 이벤트 row의 state는 concept 전체가
아니라 official `entity_id` 범위다. 반면 `H_add`의 concept-level confirmed-absent
opportunity는 해당 timestep에 같은 concept으로 매핑된 official instance가 하나도 없을
때만 자동 생성한다. 둘을 혼동해 객체의 물리적 부재라는 더 강한 의미로 확대하지 않는다.

candidate pool은 ENTER 75, EXIT 85, OCCLUDE 33, REAPPEAR 33개였고, 서로 다른 영상 수는
각 26, 26, 24, 24개였다. `(event_type, partition, annotation_id, boundary)` 정렬과
결정론적 bipartite matching으로 각 type 10개를 선택했다. 결과는
[OVIS event JSONL](../../data/negative_semantics/g0/ovis_official_gt_event_annotations.jsonl),
mapping은 [ovis_category_mapping.json](../../data/negative_semantics/g0/ovis_category_mapping.json)에
고정했다.

## v1.2 split

| split | source | 영상 수 | 역할 |
|---|---|---:|---|
| Pilot | OVIS validation official GT | 40 | event/state 계약 자동 검증 |
| Train | YouTube-VOS 2019 train | 3,471 | G1 이후 학습 |
| Development | ETRI legacy | 10 | 과거 개발 결과 재현 전용 |
| Validation | YouTube-VOS 2019 valid | 507 | threshold·operating point 선택 |
| Held-out Test | DAVIS 2017 val | 30 | source-disjoint, 계속 봉인 |

v1.1의 YouTube-VOS Pilot 33개는 playback transform을 제거하고 Validation으로 되돌렸다.
DAVIS 30개는 목록이나 바이트를 바꾸지 않았고 새 manifest hash에 맞춰 다시 봉인했다.

## 사람 없는 평가의 claim 경계

G0 사건 정답은 `human_verified`가 아니라 `official_gt_verified`다. 이후 생성 결과도
사람 없이 평가하려면 논문 표현은 “사람이 지각한 hallucination”이 아니라
**official-GT-anchored automatic additional-object / ghost-track**으로 제한한다.
G1에서 selector와 weight를 공유하지 않는 evaluator, threshold, mask/track association을
동결하기 전에는 모델 출력의 primary endpoint를 생성할 수 없다.

## 재현 명령

```bash
conda run -n ptest python scripts/prepare_negative_semantics_g0_v1_2.py --repo-root .
conda run -n ptest python scripts/audit_negative_semantics_g0.py --repo-root . \
  --output data/negative_semantics/g0/audit_report_v1_2.json
conda run -n ptest python scripts/audit_negative_semantics_g0.py \
  --repo-root . --require-pass
```

준비 스크립트는 다운로드를 수행하지 않으며 공식 두 artifact가 위 manifest 경로에
있어야 한다. 같은 입력에서 이벤트 JSONL과 manifest hash가 동일해야 한다.

## 다음 gate

G1은 Oracle `ABSENT` 신호가 실제 복원 경로의 additional-object/ghost-track 지표를 줄일
수 있는지 확인하는 현상 gate다. 먼저 evaluator 독립성·threshold를 Pilot에서 고정하고,
그 뒤 GPU smoke test와 정식 paired run을 구분해 실행한다. Held-out Test는 G1~G3의
방법·threshold·budget·Validation operating point가 모두 고정될 때까지 열지 않는다.
