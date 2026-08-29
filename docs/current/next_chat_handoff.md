---
status: active
updated: 2026-08-29
owner: ETRI SGD-JSCC 연구팀
source_commit: 7089b24
---

> [← 문서 색인](../README.md)

# 다음 채팅용 연구개발 인계 요약

## 프로젝트 기준

- 저장소: `/home/sangukbae/ETRI/Semantic/sgdjscc_lab`
- 브랜치: `main`, 원격: `origin/main`
- 원본 `SGDJSCC/`는 읽기 전용 baseline이며 확장은 `sgdjscc_lab/`에서 수행한다.
- GPU 장시간 실험은 원격 Docker에서 3 GPU 병렬로 실행하되, 컨테이너 ID는 매번
  `docker ps`로 다시 확인한다.
- 로컬의 PPTX·ZIP·`scripts/make_english_pptx.py` 등 무관한 untracked 사용자 파일은
  수정·삭제·커밋하지 않는다.
- tracked 파일 변경 시 관련 검증 후 커밋하고 `origin/main`에 푸시한다.

## 문서와 결과 파일의 역할

| 파일·디렉터리 | 역할 |
|---|---|
| [`docs/README.md`](../README.md) | 전체 문서 색인 |
| [`status.md`](./status.md) | 지금 실제로 완료·PoC·미구현인 기능과 실험 상태 |
| [`roadmap.md`](./roadmap.md) | 다음 구현 순서와 held-out 연기 결정의 기준 문서 |
| [`open_issues.md`](./open_issues.md) | 알려진 한계, 일반화 금지 조건, 기술 부채 |
| `docs/experiments/` | 날짜별 실행 조건·결과·해석을 고정한 실험 기록 |
| `docs/protocols/` | 평가·재현·전송 정상화·결과 보존 절차 |
| `docs/architecture/` | Tx/Rx 계약, 지표 정의 등 장기 설계 |
| [`results/README.md`](../../results/README.md) | Git 추적 결과 보존 규칙 |
| [`results/registry.csv`](../../results/registry.csv) | 보존된 모든 run의 경로·commit·핵심 결론 색인 |
| `results/<run>/README.md` | 해당 run의 핵심 수치와 과학적 해석 |
| `results/<run>/manifest.json` | commit·조건·dataset·provenance |
| `results/<run>/checksums.sha256` | 보존 artifact의 SHA-256 무결성 |
| `outputs/` | Git 비추적 원본 프레임·packet·로그·대용량 산출물 |

다음 채팅에서는 먼저 이 문서, `status.md`, `roadmap.md`, `open_issues.md`,
`results/registry.csv`, 최신 Git 상태를 직접 확인한다. 과거 실험의 정확한 수치는
해당 `results/<run>/`의 CSV·JSON을 기준으로 한다.

## 완료된 핵심 검증과 결론

### 1. float32 reliable-digital 전송 진단

- 10dB, 3 core condition × 100 frame, 실패·non-finite·stage conflict 0.
- float32 wire round-trip 300/300 bit-exact.
- digital wire는 AWGN보다 평균 PSNR +0.721dB였고 in-process/wire 최대 PSNR 차이는
  0.000752dB였다.
- 결론: **float32 wire 직렬화·전송은 품질 저하 원인이 아니다.**
- 근거: [실험 문서](../experiments/2026-08-28_float32_digital_step_normalization_full.md),
  [보존 결과](../../results/float32_digital_normalization_full_20260827/README.md)

### 2. 10dB 양자화 재평가

- 10영상 × fixed AWGN/float32/int16/int8/int6/int4 = 60/60 pair, 실패·non-finite 0.
- int16/int8/int6/int4가 모두 float32 대비 품질 기준을 통과했다.
- 최소 bit-depth `fixed_int4`: byte -28.45%, PSNR -0.0526dB,
  SSIM -0.00201, LPIPS 변화 -0.000487.
- AWGN은 참고 기준이며 digital Pareto baseline으로 사용하지 않는다.
- 결론: **전송 양자화 operating point는 `fixed_int4`.**
- 근거: [실험 문서](../experiments/2026-08-28_quantization_reevaluation_10db.md),
  [보존 결과](../../results/quantization_reevaluation_10db_20260828/README.md)

### 3. fixed–SKEM exact matched-rate

- 100/100 pair와 50/50 rate row 완료, actual transmitting count exact.
- raw byte 차이는 최대 0.004953%, padding 후 effective byte는 exact match.
- proxy SKEM이 10/10 영상에서 fixed와 같은 keyframe/transmission schedule로 수렴해
  품질 차이도 정확히 0이었다.
- 결론: **현재 proxy SKEM은 fixed 대비 이점이 없으며 `fixed_int4`를 유지한다.**
- 근거: [실험 문서](../experiments/2026-08-28_fixed_skem_matched_rate_10db.md),
  [보존 결과](../../results/fixed_skem_matched_rate_10db_20260828/README.md)

### 4. edge·uncertainty 전송 절감

- 10영상 × 16 guide profile = 160/160 pair, 실패·non-finite 0.
- baseline `fixed_int4` packet의 edge와 uncertainty가 각각 약 45.41%, 합계 90.83%.
- reliable-digital 경로에서 uncertainty는 실제 decoder에 소비되지 않았다.
- 1차 최소 후보 `combined_ds4`는 356,824.7 bytes/video(-85.11%)였고 pixel 변화는
  수치 오차 수준이었다.
- 이후 통합 검증에서 빈 조합을 닫았으며 `candidate_both_omit`은
  219,459.7 bytes/video로 baseline 대비 **90.843% 절감**했다.
- full50에서는 both-omit의 PSNR/SSIM/LPIPS 변화가 사실상 0이고 closed/open semantic
  지표가 baseline과 정확히 같았다.
- 결론: **현재 reliable-digital 개발 조건의 guide 후보는 both-omit.** 기본 경로를
  즉시 교체하지 말고 opt-in Tx/Rx 정책으로 정리해야 한다.
- 근거: [1차 실험](../experiments/2026-08-28_edge_uncertainty_ablation_10db.md),
  [보존 결과](../../results/edge_uncertainty_ablation_10db_20260828/README.md)

### 5. 통합 semantic·hallucination·temporal 평가

- 조건: fixed selector, `fixed_int4`, 10dB, seed 2025.
- 10영상 × 3 decoder(full50/few10/VAE-direct) × 4 guide = 120/120 pair,
  총 12,000 frame, 실패·non-finite 0.
- CLIP·OWLv2·VQA가 각각 47,744건 기여했고 3-GPU provenance가 정상이다.
- 공식 base: `fixed_int4 + baseline guides + full50`.
- 개발셋 잠정 후보: `fixed_int4 + candidate_both_omit + few10`.

| 지표 | base | 잠정 후보 | 변화 |
|---|---:|---:|---:|
| bundle bytes/video | 2,396,632.7 | 219,459.7 | -90.843% |
| reconstruction elapsed/video | 108.0504 s | 39.4540 s | -63.486%, 2.74× |
| PSNR | 23.47628 | 23.23277 | -0.24351dB |
| SSIM | 0.733301 | 0.731419 | -0.001882 |
| LPIPS | 0.253619 | 0.269202 | +0.015583 |
| closed PTC | 0.7716 | 0.7703 | -0.0013 |
| open hallucination rate | 0.0160 | 0.0355 | +0.0195 |
| additional objects/100 frames | 2.7 | 4.8 | +2.1 |

- PSNR·SSIM·LPIPS와 closed semantic/temporal 평균 gate는 통과했다.
- open hallucination 증가 CI 상한 0.0525와 additional-object 증가 CI 상한 0.0570이
  margin 0.05를 조금 넘었다. 증가는 `01_person_walk`, `02_car_pass`에 집중됐다.
- VAE-direct는 23.4885s/video로 가장 빠르고 PSNR·LPIPS도 개선됐지만 SSIM 하락
  0.01129가 margin 0.01을 넘어 탈락했다.
- 결론: **`fixed_int4 + both-omit + few10`은 최종 모델이 아니라 개발셋 잠정
  후보**다. 보수적 비교점은 `fixed_int4 + both-omit + full50`이다.
- 근거: [실험 문서](../experiments/2026-08-29_integrated_semantic_validation_10db.md),
  [보존 결과](../../results/integrated_semantic_validation_10db_20260829/README.md)

## 현재 임시 결정

- bit-depth: `fixed_int4`
- selector: fixed. proxy SKEM은 현재 이점 없음.
- guide: `candidate_both_omit`을 opt-in 개발 후보로 사용.
- decoder: `few10`을 잠정 후보, `full50`을 보수적 기준으로 유지.
- 합친 잠정 후보: **`fixed_int4 + candidate_both_omit + few10`**.
- 이 후보를 final/best generalized operating point라고 부르면 안 된다.

## held-out 검증 상태

- 별도 독립 데이터셋을 당장 만들 수 없어 2026-08-29에 **데이터 준비 시점까지
  명시적으로 연기**했다. 취소하거나 통과한 것으로 간주하지 않는다.
- 기존 개발 10영상으로 threshold를 더 튜닝하지 않는다.
- 새 데이터가 준비되면 권장 20영상 이상, 영상당 100 frame으로 다음 3조건을 비교한다.
  1. `fixed_int4 + baseline + full50`
  2. `fixed_int4 + both-omit + full50`
  3. `fixed_int4 + both-omit + few10`
- 최종 판정에서는 평균뿐 아니라 hallucination/additional-object paired CI 상한까지
  margin 안에 들어와야 한다.

## 바로 이어서 할 작업

1. **guide Tx/Rx 계약 정리**
   - both-omit을 명시적 opt-in 정책으로 고정한다.
   - baseline 동작을 유지하고 manifest, packet accounting, resume signature에 정책을
     기록한다.
2. **verifier를 실제 sampler에 연결**
   - 현재 결정·로그만 하는 action을 retry, stop, negative prompt, prompt emphasis,
     fallback에 실제 반영한다.
   - 최대 retry, 실패 fallback, 추가 지연·시도 횟수·추가 전송 byte를 기록한다.
3. **verifier 폐루프 ablation**
   - OFF / 로그 전용 / retry·stop / prompt 제어 포함 조건을 비교한다.
   - 품질, hallucination, temporal, 추가 연산량과 지연을 함께 본다.
4. **동적 전송 예산 controller**
   - 채널 상태, uncertainty, verifier 위험도로 전송량과 복원 연산량을 결정한다.
   - feedback, retransmission byte, RTT를 accounting에 포함한다.
5. **데이터 준비 후 held-out과 최종 문서 마감**
   - 최종 operating point, paired CI, Pareto 표·그래프·재현 명령·checksum을 확정한다.

## 작업 시 주의할 과학적 경계

- both-omit의 무손실 결론은 현재 reliable-digital checkpoint/config 개발 조건에 한정한다.
- few10은 학습된 distilled/consistency model이 아니라 production sampler의 10-step
  근사다.
- 물리 channel symbol·FEC 환산은 여전히 proxy이며 bundle byte만 exact하다.
- verifier action은 아직 sampler에 실제 개입하지 않는다.
- held-out 전에는 최종 일반화·최종 운영점 주장을 하지 않는다.
