---
status: ready_for_formal_run
updated: 2026-09-03
gate: G1
evidence_split: OVIS Pilot
heldout_opened: false
---

# Negative-semantics G1 실행 준비

## 현재 판정

G1의 코드·동결 설정·재개형 단일 GPU 실행 경로를 준비했다. 이 단계는 학습이 아니라
고정된 SGD-JSCC checkpoint로 복원 영상을 생성하고 additional-object/ghost 현상이 실제로
존재하는지 측정하는 추론·평가 실험이다.

아직 정식 40영상 결과가 없으므로 현재 상태는 `G1_READY_TO_RUN`이며 `G1_PASSED`가 아니다.
정식 실행이 끝난 뒤 `g1_summary.json`의 gate를 확인해야 한다.

## 동결된 실험 행렬

| 항목 | 값 |
|---|---|
| 데이터 | OVIS official-with-GT Pilot 40영상, 40 독립 event |
| 변환 | longest side 512, aspect ratio 유지, bilinear antialias, upscale 없음 |
| 전송 | `fixed_int4`, 10 dB fixed-reference digital step policy |
| guide | `candidate_both_omit` |
| 시간축 | 기존 operating point와 동일한 fixed max-GOP 16, reuse threshold 0.2 |
| sampler | `few10`, `full50` |
| seed | 2025, 2026, 2027 |
| evaluator | `google/owlv2-base-patch16-ensemble` |
| caption | source frame에서 미리 생성한 `Salesforce/blip2-opt-2.7b-coco` caption |
| selector | fixed, evaluator와 공유 weight 없음 |
| held-out | 접근 금지, 실행기에도 포함되지 않음 |

원본 OVIS 이미지는 MP4로 재인코딩하지 않는다. 기존 전송 실험기가 annotated image
sequence를 직접 읽고, 512-pixel 변환은 resume signature에 기록한다. BLIP2 caption은 먼저
생성해 파일로 고정한 다음 BLIP2를 GPU에서 내린다. 복원은 `use_text=false`로 모델을
적재하되 그 caption을 실제 bundle과 receiver diffusion condition으로 전달한다. 이 방식은
text condition을 제거하는 ablation이 아니라 16 GiB GPU에서 동시 모델 적재를 피하는
실행 분리다.

## evaluator 동결 순서

1. G0 13/13 감사를 다시 통과한다.
2. OVIS source frame과 official concept presence만 사용해 OWLv2 score를 계산한다.
3. 사전 정의한 grid에서 recall 0.70 이상, false-positive rate 0.10 이하를 모두 만족하는
   가장 높은 global threshold를 선택한다.
4. 조건을 만족하는 threshold가 없으면 복원 실행 전에 fail-closed한다.
5. threshold 동결 후에만 reconstruction output을 평가한다. detector 호출에는 method,
   sampler step, seed label을 주지 않는다.

`H_add`는 official GT에서 concept가 없는 `(video, annotated timestep, concept)` 중 detector가
양성인 비율이다. 같은 concept의 연속 양성 run은 하나의 additional event로 병합한다.
EXIT 이후 16 annotated timestep이 모두 남아 있고 concept가 계속 GT-absent인 경우만 ghost
survival AUC에 포함하며 나머지는 right-censored로 별도 보고한다.

## 정식 실행 명령

`ptest` 환경과 로컬 RTX 4080 한 장을 사용한다. 동일 명령을 다시 실행하면 caption,
reconstruction pair, evaluator cache를 재사용하거나 기존 전송 runner의 signature 기반으로
완료 pair를 건너뛴다.

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
mkdir -p outputs/negative_semantics_g1_pilot_rtx4080
set -o pipefail
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/sangukbae/anaconda3/bin/conda run --no-capture-output -n ptest \
  python scripts/run_negative_semantics_g1.py \
    --run-root outputs/negative_semantics_g1_pilot_rtx4080 \
    --device cuda:0 \
  2>&1 | tee -a outputs/negative_semantics_g1_pilot_rtx4080/operator.log
```

터미널 연결이 끊길 수 있으면 위 명령을 `tmux` 안에서 실행한다. 다른 GPU 작업과 동시에
실행하지 않는다. 진행 중에는 다음 두 명령으로 확인한다.

```bash
tail -f outputs/negative_semantics_g1_pilot_rtx4080/operator.log
nvidia-smi
```

## 완료 판독

정상 종료의 필수 파일은 다음과 같다.

- `g1_summary.json`: 최종 `PASSED` 또는 `NOT_PASSED`와 개별 gate
- `g1_policy_summary.csv`: few10/full50별 H_add와 분포
- `evaluator/evaluator_freeze.json`: source-only calibration threshold와 model revision
- `evaluator/detection_rows.jsonl`: 재현 가능한 frame/concept score
- `reconstruction/<policy>/seed_<seed>/`: 기존 runner의 frame, packet, metric, manifest

`NOT_PASSED`도 유효한 G1 결과다. 이는 실행 실패와 다르며, 현상이 사전 선언한 prevalence
또는 분산 기준에 미달해 계획의 Stop Track 검토가 필요하다는 뜻이다. `PASSED`가 나와도
Pilot 현상 근거일 뿐, method 개선이나 held-out 일반화를 입증하지 않는다.

## 구현·검증 범위

- G1 metric 단위 테스트와 기존 G0/전송 runner 회귀 테스트: 70 passed
- G0 재감사: 13/13 `PASSED`
- RTX 4080/CUDA 11.8, SGD-JSCC 4 checkpoint, BLIP2/OWLv2 local snapshot preflight: `PASSED`
- 실제 GPU smoke: 초기 2-frame end-to-end 연결 통과; 기존 operating point와 동일한
  max-GOP/reuse 설정으로 수정 후 최종 재실행 예정
- 정식 40영상 × 2 policy × 3 seed: 사용자 장시간 실행 대기
