---
status: acquisition_complete_pending_human_review
updated: 2026-09-02
owner: ETRI SGD-JSCC 연구팀
base_commit: b2bd447
protocol_version: 1.1.0
gate: G0_NOT_PASSED_12_OF_13
---

> [← 문서 색인](../README.md) ·
> [v1.0 동결 기록](./2026-09-02_negative_semantics_g0_protocol.md) ·
> [논문 연구 계획](../current/negative_semantic_paper_plan.md)

# Negative semantics G0 데이터 취득 amendment v1.1

## 판정

공식 데이터 취득, 사용 승인 기록, 5-way split, 영상별 content hash, category mapping,
Pilot 후보, source-disjoint Held-out 봉인을 완료했다. 강화된 자동 감사 13개 조건 중
**12개가 통과**했다.

현재 G0는 여전히 **`NOT_PASSED`**다. 유일한 미통과 조건은 실제 사람 두 명이 독립적으로
Pilot 이벤트를 검수하고, 불일치를 제3자가 조정하는 절차다. 기계 후보나 사용 승인 문구를
사람의 영상 판정으로 간주하지 않는다.

## 사용 승인과 취득 증거

프로젝트 소유자가 ETRI 내부 자산, YouTube-VOS 2019, DAVIS 2017을 이 연구의 내부
비상업 용도로 사용하도록 승인했다. 기록은
[data_use_approval.json](../../data/negative_semantics/g0/data_use_approval.json)에 있다.
이 승인은 원본 미디어 재배포 허가나 제3자 권리 부여로 확대 해석하지 않는다. 외부 원본과
검수 MP4는 계속 Git에서 제외한다.

| 공식 배포본 | bytes | SHA-256 |
|---|---:|---|
| YouTube-VOS 2019 `train.tar` | 9,258,444,800 | `5a81246005cfc2245bcd1f7e8fff41868963dfc3873e92bb810ea5f35a7cf8c0` |
| YouTube-VOS 2019 `valid.tar` | 1,296,824,320 | `3c847cd8f64ec6219db88671681303e84537e5478d53a88534592761ba74eb03` |
| DAVIS 2017 `trainval-480p.zip` | 832,766,765 | `e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492` |

YouTube-VOS는 공식 dataset terms와 공식 Google Drive 배포 폴더, DAVIS는 공식 2017
download와 challenge rules를 사용했다. 출처 URL·Google Drive file ID·archive hash는
[split manifest](../../data/negative_semantics/g0/dataset_split_manifest.json)에 고정했다.

## 동결 split

| split | source | 영상 수 | 역할 |
|---|---|---:|---|
| Pilot | YouTube-VOS 2019 valid | 33 | event 계약 sanity와 사람 검수 |
| Train | YouTube-VOS 2019 train | 3,471 | 이후 RSM/allocator 학습 |
| Development | ETRI legacy | 10 | 기존 개발 결과 재현 전용 |
| Validation | YouTube-VOS 2019 valid | 474 | threshold·operating point 선택 |
| Held-out Test | DAVIS 2017 val | 30 | source-disjoint 최종 평가, 현재 봉인 |

총 4,018개 assigned video의 JPEG/annotation directory를 각각 tree hash로 기록했다. tree
hash는 파일을 상대경로 순으로 정렬한 뒤 `relative_path NUL file_sha256 LF` record를
SHA-256으로 집계한다. Pilot·Validation 사이 native video 중복은 없으며 DAVIS source는
모든 이전 split과 분리됐다. [heldout seal](../../data/negative_semantics/g0/heldout_seal.json)은
method·threshold·budget·validation operating point 동결 전 개봉을 금지한다.

## Pilot 후보와 사람 검수 패키지

core ontology에 매핑되고 mid-video first appearance가 있는 YouTube-VOS valid 영상 중
video ID 사전순 첫 33개를 선택했다. 영상 하나당 한 independence cluster만 허용한다.
홀수 후보는 원래 시간축의 ENTER, 짝수 후보는 같은 규칙을 시간역전한 EXIT다.

- 후보: 33개 — ENTER 17, EXIT 16
- 상태: 전부 `human_verified=false`
- 검수 영상: `data/external/youtube_vos_2019/pilot_review/*.mp4` 33개, 36MB
- 후보 파일: [youtube_vos_pilot_event_candidates.jsonl](../../data/negative_semantics/g0/youtube_vos_pilot_event_candidates.jsonl)
- 입력 양식: [pilot_review/](../../data/negative_semantics/g0/pilot_review/)

시간역전은 자연 발생 EXIT를 주장하기 위한 표본이 아니라 G0에서 EXIT state transition과
ghost-survival 측정 계약을 검사하는 통제 변환이다. 최종 논문 결과에서는 자연 EXIT와
통제 EXIT를 분리 보고해야 한다. OCCLUDE·REAPPEAR·SCENE_CUT의 최종 event density는
Validation을 개봉하기 전에 별도 annotation round로 확보해야 한다.

두 annotator는 각자 template 복사본을 따로 작성한다. `decision`은
`accept|reject|uncertain`, event type과 boundary는 명시적으로 입력하고,
`absence_confirmed`와 `identity_confirmed`는 `yes|no`로 기록한다. 독립 작성이 끝난 뒤에만
두 파일을 합친다. ±1 frame 이내 동일 판정은 합의로 처리하고, 나머지는 독립 adjudicator가
결정한다.

```bash
python scripts/compile_negative_semantics_pilot_reviews.py \
  --annotator-a reviews/annotator_a.csv --annotator-a-id reviewer_a \
  --annotator-b reviews/annotator_b.csv --annotator-b-id reviewer_b \
  --adjudicator reviews/adjudicator.csv --adjudicator-id reviewer_c
```

compiler는 annotator ID 중복, 빈 판정, 불완전한 absence/identity 확인, 미조정 불일치를
거부한다. accepted independence cluster가 30~50개가 아니면 exit 4다. 성공 결과를 받은
뒤 v1.2 amendment에서 final annotation hash를 동결해야 G0를 통과시킬 수 있다.

## 재현 명령과 감사 결과

공식 아카이브를 manifest 경로에 배치하고 해제한 뒤 다음을 실행한다.

```bash
conda run -n ptest python scripts/prepare_negative_semantics_g0.py \
  --repo-root . --render-review-clips
conda run -n ptest python scripts/audit_negative_semantics_g0.py --repo-root .
conda run -n ptest python scripts/audit_negative_semantics_g0.py \
  --repo-root . --require-pass
```

현재 일반 감사 결과는 `12/13`, `gate_status=NOT_PASSED`,
`eligible_to_start_g1=false`다. `--require-pass`는 의도대로 exit 4여야 한다. 이번 단계는
GPU smoke test나 학습이 아니라 CPU 기반 취득·해시·검수자료 준비다.

## 남은 단일 G0 행동

실제 사람 두 명이 33개 검수 MP4를 독립 판정하고 필요 시 제3자가 조정한다. 최소 30개가
통과하면 final JSONL과 hash를 v1.2로 동결하고 `--require-pass` exit 0을 확인한다. 그
전에는 G1 논문 evidence 생성이나 Held-out 개봉을 시작하지 않는다.
