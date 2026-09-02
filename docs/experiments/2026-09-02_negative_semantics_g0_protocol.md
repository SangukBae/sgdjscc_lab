---
status: historical_artifact_superseded_by_v1_1
updated: 2026-09-02
owner: ETRI SGD-JSCC 연구팀
source_commit: 74d72a2
protocol_version: 1.0.0
gate: G0_NOT_PASSED
---

> [← 문서 색인](../README.md) ·
> [논문 연구 계획](../current/negative_semantic_paper_plan.md) ·
> [동결 config](../../configs/experiments/negative_semantics/g0_protocol.yaml)

> Historical artifact: 이 문서는 v1.0 감사 시점을 보존한다. 현재 상태는
> [v1.1 acquisition amendment](./2026-09-02_negative_semantics_g0_acquisition_amendment_v1_1.md)를 따른다.

# Negative semantics G0 protocol freeze와 데이터 감사

## 판정

G0 방법론 계약은 `negative_semantics_g0_v1`로 동결했다. 그러나 G0 gate는
**`NOT_PASSED`**이며 G1 실험을 시작할 수 없다.

감사 도구가 확인한 10개 조건 중 7개는 통과했다. 미통과 조건은 다음 세 가지다.

1. Pilot의 사람 검수 독립 이벤트가 0개다. 동결 기준은 30~50개다.
2. source-disjoint Held-out Test 목록과 실제 바이트가 없다.
3. ETRI 내부 영상의 사용 권한 기록 및 외부 데이터 약관 승인 기록이 없다.

빈 split을 임의 샘플링으로 채우지 않았다. 현재 부족한 증거를 그대로 드러내기 위한
fail-closed 판정이다.

## 동결한 연구·전송 계약

| 항목 | v1 결정 |
|---|---|
| 연구 track | Full Track |
| primary venue | IEEE Transactions on Multimedia |
| fallback venue | IEEE TCSVT |
| primary rate unit | total wire bytes / source frame |
| exact-byte budget cap | 2,048 / 4,096 / 8,192 / 16,384 bytes per source frame |
| generation seed | 2025 / 2026 / 2027 |
| paired unit | 같은 video·seed·budget |
| bootstrap | video cluster, deterministic 5,000회, 95% CI |
| gate 규칙 | 모든 열화 지표의 one-sided CI 상한이 margin 이하여야 함 |

wire byte에는 positive·negative·protection·feedback·retransmission·header·padding을
모두 포함한다. 각 방법은 같은 cap 아래 독립적으로 재최적화하며, padding만으로 만든
동률은 Pareto 우위 근거로 사용하지 않는다.

## split 동결 상태

| split | 현재 영상 수 | 용도·판정 |
|---|---:|---|
| Pilot | 0 | 차단 — 30~50개 독립 이벤트 필요 |
| Train | 0 | 차단 — RSM adapter/risk predictor 학습 데이터 미확보 |
| Development | 10 | 기존 ETRI 전량; 반복 사용 이력 때문에 개발 전용 |
| Validation | 0 | 차단 — threshold/operating point 선택 불가 |
| Held-out Test | 0 | 차단 — 최종 주장 불가 |

ETRI 10개 영상의 processed MP4와 GT JSON 각각에 SHA-256을 기록했다. 구체적인 목록과
해시는 [dataset split manifest](../../data/negative_semantics/g0/dataset_split_manifest.json),
manifest 자체의 detached hash는
[dataset_split_manifest.sha256](../../data/negative_semantics/g0/dataset_split_manifest.sha256)에
있다. ETRI 영상은 최종 test에 다시 배치할 수 없다.

### 기존 이벤트 증거

현재 GT 구간에서 기계적으로 유도 가능한 row는 5개, causal independence cluster는
3개다. `02`와 `07`의 ENTER, `09`의 SCENE_CUT·chair EXIT·car ENTER다. 모두 별도 사람
검수를 받지 않았으므로 Pilot gate에는 **0개**로 센다. `08_car_exit`는 이름과 달리 현재
GT가 전 프레임에서 car present이므로 EXIT로 등록하지 않았다.

## 데이터 라이선스 감사

| 후보 | 공식 근거에서 확인한 내용 | v1 판정 |
|---|---|---|
| ETRI 내부 10영상 | 로컬 자산은 있으나 provenance·사용 권한 문서가 저장소에 없음 | Development only, 검토 필요 |
| YouTube-VOS 2019 | annotation CC BY 4.0; data는 non-commercial research 조건과 기관 의무가 있음 | Pilot/Train/Validation 우선 후보, 기관 승인 전 배정 금지 |
| DAVIS 2017 | 공식 challenge의 새 annotation은 CC BY 4.0 | source-disjoint held-out 우선 후보, 원영상 권리 확인 전 배정 금지 |
| TAO/BURST | 여러 원천 dataset을 결합한 federated source | 원천별 약관 감사 전 backup only |
| BDD100K | 코드 저장소 라이선스만으로 영상 데이터 권리를 대체할 수 없음 | 미선택 |

공식 확인 URL은 split manifest의 `source_audit`에 고정했다. 이는 법률 자문이 아니며,
실제 다운로드 전에 ETRI 내부 책임자의 약관·배포 범위 승인이 필요하다.

## ontology·상태·이벤트 계약

- ontology: 12개 core concept과 alias를
  [ontology manifest](../../data/negative_semantics/g0/ontology_manifest.json)에 고정했다.
- state: `present`, `confirmed_absent`, `unknown`만 허용한다.
- `confirmed_absent`: 사람이 ontology scope와 시간 구간을 확인한 absent opportunity에서만
  부여한다. detector miss, 미주석 category, 애매한 crop은 전부 `unknown`이다.
- event: `ENTER`, `EXIT`, `OCCLUDE`, `REAPPEAR`, `SCENE_CUT`만 허용한다.
- 독립 표본: `independence_cluster_id`로 센다. 같은 scene cut에서 생긴 여러 entity
  transition은 한 cluster다.
- boundary tolerance: ±1 frame이다.

기계 검증 형식은 [event JSON Schema](../../data/negative_semantics/g0/event_schema.schema.json)에
있다. ontology/category mapping 변경은 version bump와 사전 amendment가 필요하다.

## endpoint 정의와 margin

### Primary endpoint

- `H_add`: `generated absent-concept event 수 / confirmed-absent opportunity 수`다.
  단순 frame 수나 detector가 못 찾은 frame을 분모로 쓰지 않는다. positive-only 대비
  paired difference의 목표 margin은 `-0.01 absolute`다.
- ghost survival AUC: GT EXIT 다음 16 frame의 binary presence curve 평균이다.
  16 frame을 확보하지 못한 right-censored event는 primary AUC에서 제외하고 별도
  개수·민감도 분석을 보고한다. paired difference 목표 margin은 `-0.05`다.

### Safety·quality endpoint

| 지표 | candidate-minus-reference 허용 상한 |
|---|---:|
| false suppression rise | 0.015 absolute |
| object recall drop | 0.02 absolute |
| reappearance identity drop | 0.05 absolute |
| PSNR drop | 0.5 dB |
| SSIM drop | 0.01 |
| LPIPS rise | 0.02 |

reappearance identity consistency는 REAPPEAR 뒤 객체가 occlusion 전 같은 entity인지
blind majority가 맞춘 이벤트 비율이다. 같은 class의 다른 instance로 바뀌면 실패다.

## 독립 evaluator와 사람 주석 규칙

### GT/event annotation

1. 두 annotator가 원본 영상을 전체 속도와 frame-step으로 독립 검수한다.
2. concept state, entity ID, event type·boundary, independence cluster를 각각 기록한다.
3. 둘 중 한 명이라도 객체 존재·경계·identity를 확정하지 못하면 `unknown`으로 둔다.
4. boundary 차이가 ±1 frame을 넘거나 type/identity가 다르면 제3 adjudicator가 원본을
   보고 확정한다.
5. 두 검수자의 합의 또는 adjudication이 끝난 row만 `human_verified=true`로 바꾼다.

### 생성 결과 평가

1. 방법·budget·seed를 숨긴 random sample ID로 세 명에게 제시한다.
2. `H_add`는 해당 opportunity에서 absent concept이 실제 객체처럼 생성됐는지 yes/no로
   판정한다. 그림자·텍스트·반사처럼 애매한 경우는 uncertain으로 보내 adjudication한다.
3. EXIT 뒤 16 frame은 frame별 presence를 판정해 ghost curve를 만든다.
4. REAPPEAR는 occlusion 전 reference crop과 복원 후 crop을 함께 보되 방법 정보는 숨기고
   same/different/uncertain으로 판정한다.
5. majority vote 후 unresolved case는 별도 adjudicator가 확정한다. raw agreement와
   Fleiss' kappa를 모두 보고한다.

자동 OWLv2 screen은 annotation queue를 만드는 secondary 도구일 뿐 primary endpoint의
최종 판정자가 아니다. selector가 OWLv2 weight를 사용하면 같은 weight의 screen 결과도
primary 근거로 사용할 수 없다.

## 재현과 감사

실행 환경은 [Conda environment](../../environments/sgdjscc-py39-cu118.yml)에 고정했다.
저장소 루트에서 다음을 실행한다.

```bash
conda env create -f environments/sgdjscc-py39-cu118.yml
conda activate sgdjscc-py39-cu118
python scripts/audit_negative_semantics_g0.py --repo-root .
python scripts/audit_negative_semantics_g0.py --repo-root . --require-pass
```

첫 감사 명령은 artifact 무결성과 현재 판정을 보고하며 성공 종료한다. 두 번째 명령은
현재 의도대로 exit code 4를 반환한다. `PASSED` 전환은 기존 JSON을 사후 덮어쓰지 않고,
외부 데이터 목록·해시·승인 기록과 사람 검수 event를 추가한 새 version에서 수행한다.

## 다음 행동과 중단 규칙

1. YouTube-VOS 약관의 기관 사용 승인을 기록하고 필요한 subset metadata를 취득한다.
2. video ID를 무작위 런타임 선택하지 말고 Pilot/Train/Validation 목록으로 명시한다.
3. Pilot에서 30~50개 독립 event를 두 명+adjudicator 절차로 검수한다.
4. DAVIS 원영상 이용 조건을 확인한 뒤 source-disjoint held-out video ID와 SHA-256을
   별도 봉인한다.
5. `--require-pass`가 0이 되기 전에는 G1 결과를 논문 evidence로 생산하지 않는다.

이번 작업에서는 외부 dataset 다운로드, 사람 검수, GPU 생성 실험, G1 구현을 수행하지
않았다. 따라서 protocol freeze는 완료됐지만 데이터 확보를 포함한 G0 gate는 미완료다.
