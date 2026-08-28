---
status: prepared
updated: 2026-08-28
owner: ETRI SGD-JSCC 연구팀
---

> [← 문서 색인](../README.md)

# fixed_int4 edge·uncertainty 전송 절감 ablation 준비

## 목적

10dB `fixed_int4` packet의 90.90%를 차지하는 edge·uncertainty 표현을 실제
직렬화 경계에서 압축하고, 기존 품질 gate를 만족하는 최소 전송량 후보를 찾는다.

## 고정 조건

- selector: `fixed`
- visual latent: `int4`
- decoder step: `fixed_reference`, 10dB
- 영상: ETRI 10개 전체
- GPU: `cuda:0,cuda:1,cuda:2`
- seed: 2025
- 기준선: `fixed_int4__baseline`
- 전체 profile: 16개, 예상 video-profile pair: 160개

## profile

각 성분에 대해 다섯 개의 독립 ablation을 실행한다.

| 성분 | profile | 조작 |
|---|---|---|
| edge | `edge_q4` | 8→4 bit |
| edge | `edge_ds2` | 공간 해상도 1/2, Rx bilinear 복원 |
| edge | `edge_ds4` | 공간 해상도 1/4, Rx bilinear 복원 |
| edge | `edge_reuse2` | visual 전송 2회 중 1회만 guide 전송, 이전 Rx cache 재사용 |
| edge | `edge_omit` | 전송 생략, Rx 0-map |
| uncertainty | `uncertainty_q4` | 8→4 bit |
| uncertainty | `uncertainty_ds2` | 공간 해상도 1/2, Rx bilinear 복원 |
| uncertainty | `uncertainty_ds4` | 공간 해상도 1/4, Rx bilinear 복원 |
| uncertainty | `uncertainty_reuse2` | visual 전송 2회 중 1회만 guide 전송, 이전 Rx cache 재사용 |
| uncertainty | `uncertainty_omit` | 전송 생략, Rx 0-map |

결합 후보는 `combined_q4`, `combined_ds2`, `combined_ds4`,
`combined_reuse2`, `combined_q4_ds2_reuse2`다.

## 수신 경계와 공정 byte 회계

- reuse frame은 송신단 tensor를 참조하지 않고, 이전 bundle에서 decode한 receiver cache만 사용한다.
- omit은 다른 guide를 유지한 채 대상 guide만 0-map으로 바꾼다.
- downsample tensor의 실제 shape와 bit-packed payload가 `.sgbundle`에 저장된다.
- 사람이 읽는 profile/config 이름은 wire manifest에 넣지 않는다. profile 이름 길이가
  전송량 차이로 오인되는 것을 막기 위해 action은 항상 한 자리 코드 두 개
  (`0=transmit`, `1=reuse`, `2=zero`)로 기록한다.
- `packet_components.csv`에서 edge·uncertainty byte와 action을 독립 계측하고,
  모든 component 합이 실제 bundle byte와 일치해야 한다.

## 출력과 fail-closed 검증

- `guide_ablation_effect.csv`: baseline 대비 품질·byte 변화
- `guide_component_bytes.csv`: component 합계와 transmit/reuse/zero 횟수
- `guide_pareto_frontier.csv`: 품질 gate 내 Pareto 후보
- `guide_ablation_validation.json`: 160 pair, 실패/non-finite, 10dB, profile,
  component reconciliation 검증
- `GUIDE_ABLATION_REPORT.md`: 요약 리포트

품질 gate는 PSNR 하락 ≤ 0.5dB, SSIM 하락 ≤ 0.01, LPIPS 증가 ≤ 0.02다.

## 실행

```bash
RUN_DIR="edge_uncertainty_ablation_10db_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_edge_uncertainty_ablation_10db.sh \
  --output-root "outputs/$RUN_DIR"
```

실제 장시간 GPU 실행과 결과 확정은 아직 수행하지 않았다.
