---
status: completed
date: 2026-09-07
owner: ETRI SGD-JSCC 연구팀
source_commit: 8fbe6d98b1bc541387f120e4b7b98944aea8efbb
protocol_id: negative_semantics_int6_bridge_v1_0
evidence_scope: OVIS_PILOT_INT6_BRIDGE
---

> [← 문서 색인](../README.md) ·
> [실행 준비 기록](./2026-09-06_negative_semantics_int6_bridge_preparation.md) ·
> [후속 운용점 결정](./2026-09-07_fixed_int4_primary_operating_point_decision.md)

# fixed_int4–fixed_int6 negative-semantics bridge 정식 결과

## 목적과 판정 범위

G1 v1.1의 `fixed_int4 + candidate_both_omit` 결과를 신규 bit-depth에 임의로
일반화하지 않고, 동일 OVIS Pilot 40영상·정책·평가기로 `fixed_int6`를 paired
비교했다. 이 bridge는 bit-depth 변화에 따른 exact byte, 복원 품질, 실행시간,
additional-object와 ghost 지표의 변화를 측정한다. G1의 scientific gate를 다시
판정하거나 held-out 일반화를 주장하는 실험은 아니다.

## 실행 무결성

| 항목 | 결과 |
|---|---|
| 실행 상태 | `formal`, `PASSED` |
| Git | `8fbe6d98b1bc541387f120e4b7b98944aea8efbb`, 시작 시 tracked-clean |
| 데이터 | OVIS Pilot 40영상, held-out 미접근 |
| 조건 | `{fixed_int4,fixed_int6}` × `{few10,full50}` |
| seed | 2025, deterministic decoder 선언, effective seed 1개 |
| 신규 int6 reconstruction | 정책당 40/40, 실패 pair 0 |
| evaluator | OWLv2 threshold 0.2, G1 source calibration 재사용 |
| LPIPS | VGG, `lpips==0.1.4`, state-dict SHA-256 기록 |
| 산출물 무결성 | 10,589파일, 1,217,972,224 bytes, SHA-256·크기 불일치 0 |
| 실측 wall time | 약 8시간 49분 |

원본 실행은 `outputs/negative_semantics_int6_bridge_rtx4080/`에 있다. 핵심 파일은
`status.json`, `run_spec.json`, `int4_reuse_verification.json`,
`int6_bridge_summary.json`, `artifact_checksums.json`이다. `operator.log`에는
오류·OOM·실패가 없고 의존성 deprecation warning만 있다.

## 품질·전송량·시간

아래 값은 영상별 값의 평균이다. delta는 `fixed_int6 - fixed_int4`이며, 낮을수록
좋은 LPIPS를 제외한 품질 지표는 클수록 좋다.

| 정책 | 조건 | bytes/video | bytes/frame | PSNR | SSIM | LPIPS | elapsed/video |
|---|---|---:|---:|---:|---:|---:|---:|
| few10 | fixed_int4 | 1,279,726.3 | 19,123.1 | 26.8199 | 0.762891 | 0.194847 | 269.521 s |
| few10 | fixed_int6 | 1,845,588.7 | 27,567.1 | 26.9200 | 0.764569 | 0.193616 | 269.225 s |
| full50 | fixed_int4 | 1,279,726.3 | 19,123.1 | 27.5772 | 0.778365 | 0.180688 | 482.186 s |
| full50 | fixed_int6 | 1,845,588.7 | 27,567.1 | 27.7227 | 0.780641 | 0.178974 | 478.018 s |

| 정책 | byte 변화 | PSNR delta (95% CI) | SSIM delta (95% CI) | LPIPS delta (95% CI) | 시간 delta (95% CI) |
|---|---:|---:|---:|---:|---:|
| few10 | **+44.217%** | +0.1002 [+0.0825,+0.1185] dB | +0.001678 [+0.001403,+0.001943] | -0.001231 [-0.001456,-0.001004] | -0.296 [-0.859,+0.292] s |
| full50 | **+44.217%** | +0.1455 [+0.1243,+0.1677] dB | +0.002276 [+0.001990,+0.002568] | -0.001714 [-0.001965,-0.001455] | -4.168 [-5.137,-3.264] s |

두 정책 모두 사전 동결한 PSNR/SSIM/LPIPS non-inferiority budget을 통과했다.
그러나 int6의 품질 이득은 작고 전송량은 44.217% 증가한다. few10 시간 차이는
CI가 0을 포함하며, full50의 약 4.17초 감소도 bit-depth 선택을 좌우할 정도의
크기는 아니다.

## additional-object 결과

### Pooled absolute estimand

| 정책 | 조건 | raw H_add | source-paired H_add | source-paired count/opportunities |
|---|---|---:|---:|---:|
| few10 | fixed_int4 | 1.9793% | 1.0493% | 156/14,867 |
| few10 | fixed_int6 | 2.0124% | 1.0628% | 158/14,867 |
| full50 | fixed_int4 | 1.8536% | 0.9081% | 135/14,867 |
| full50 | fixed_int6 | 1.8138% | 0.8610% | 128/14,867 |

### Video-paired int6−int4 delta

| 정책 | estimand | 평균 delta | video-clustered bootstrap 95% CI |
|---|---|---:|---:|
| few10 | raw H_add | +0.0259%p | [-0.0880,+0.1296]%p |
| few10 | source-paired H_add | +0.0163%p | [-0.0963,+0.1228]%p |
| full50 | raw H_add | -0.1940%p | [-0.7038,+0.1070]%p |
| full50 | source-paired H_add | -0.2100%p | [-0.7158,+0.0841]%p |

모든 CI가 0을 포함한다. 따라서 fixed_int6가 additional-object를 유의하게
악화하거나 개선했다는 근거는 없다. int6에서도 source-paired full50 H_add는
0.8610%로 G1의 1% prevalence 기준보다 낮다.

## ghost 결과

정책별 10개 EXIT event 중 4개만 int4/int6 모두 uncensored였고 6개는 jointly
censored였다. 4개 유효 event의 paired delta는 모두 0이었지만 표본이 너무 작아
bit-depth 간 ghost 동등성을 주장하지 않는다.

## 결론과 운용점 결정 입력

1. bridge 실행과 품질 non-inferiority 검증은 성공했다.
2. int6는 int4보다 화질이 소폭 좋지만 전송량이 44.217% 많다.
3. int6가 additional-object 또는 ghost 문제를 해결한다는 증거는 없다.
4. 통신 효율을 primary 목표로 하는 이후 논문 실험은 `fixed_int4`가 더 적절하다.
5. `fixed_int6`는 품질 민감도·bit-depth robustness ablation과 대역폭 여유가 있는
   시연용 opt-in 조건으로만 유지한다.
6. 이 결과는 G1의 `effective_seed_count=1`, full50 source-paired prevalence 미달,
   ghost 유효 표본 부족을 해소하지 않는다. G1 scientific gate는 계속
   `NOT_PASSED`다.

후속 기준은 [fixed_int4 primary 운용점 결정](./2026-09-07_fixed_int4_primary_operating_point_decision.md)에
고정한다.
