# float32 digital decoder-step 정상화 full 결과

> **보존 사본.** 원본 `outputs/f32dig_20260827_164017/`에서 재현·검증에 필요한
> CSV·JSON·JSONL·Markdown·config 50개를 byte-identical로 복사하고, 수정된 리포트
> 계약으로 다시 생성한 `INTEGRATED_REPORT.md` 1개를 함께 보존한다.

## 상태와 결론

- 상태: **완료(frozen)**
- 실행 commit: `c5721cb6f3501af53555137fb0aaf131b5c2d71b`
- Git dirty: `unknown` — 실행 manifest의 실측값을 그대로 사용하며 clean으로 추정하지 않음
- 실행: RTX 4090 3장, seed `2025`, float32, `fixed_reference=10dB`, full profile
- stage 실패·NaN/Inf·conflict: 0
- float32 wire round-trip: 300/300 bit-exact
- 300프레임 digital wire − AWGN:
  - PSNR 평균 `+0.721163dB`(최솟값 `-0.316044dB`)
  - SSIM 평균 `+0.00552`
  - LPIPS 평균 `-0.00223`
- in-process–wire 최대 PSNR 차이: `0.000751495dB`
- 결론: 기존 품질 저하의 원인은 float32 transport가 아니라 60dB decoder-step
  계약이었으며, 10dB 정책에서 정상화됐다.

## 리포트 계약 보정

- `INTEGRATED_REPORT.original.md`: 실행 당시 원문, SHA-256
  `0b482697507cb9652ca98bd4e629d86aeb0b1ffe1b232e2f1904c652dfe253f8`.
- `INTEGRATED_REPORT.md`: 같은 raw CSV/JSONL에 현재 리포트 로직을 적용한 파생본.
  - 실제 증거가 있고 품질 기준을 넘지 않은 기존 `inconclusive`를
    `no_issue_detected`로 분리: instrumented baseline 20/20
  - tensor 계측을 의도적으로 끈 stage 6을 metric-only 계약으로 집계:
    `no_issue_detected` 300/300
  - 기존 auxiliary `packet_tx_rx_issue` 2건은 overall에서 계속 제외하고
    중립적 `transport_delta_detected`로 표시
- raw `verdicts.jsonl`은 역사적 원본을 변경하지 않았다. 새 실행부터는 classifier가
  위 새 label과 auxiliary 전용 tolerance를 직접 사용한다.

## 보존 범위

- 최상위: execution plan, 원본/보정 통합 리포트
- stage 3/4/5: path 비교, tensor 통계·쌍 비교, verdict, summary,
  initial/final manifest, resume signature, composed config, 원본 REPORT
- stage 6 worker 3개: 300프레임 path 비교, summary, initial/final manifest,
  resume signature, 원본 REPORT
- 제외: `frames_cache/` 600개 PNG, stage log, 임시 캐시. 원본 660개·76MB는
  로컬 `outputs/`에 유지한다.
- 보존본: `checksums.sha256` 제외 53개 파일·약 2.2MB. 그중 실행 원본 50개는
  원본 경로와 직접 SHA-256 대조했고 모두 일치했다. 나머지는 보정 통합 리포트,
  이 README, 보존 manifest다.

## 재현성

| 항목 | 값·위치 |
|---|---|
| 원격–로컬 raw tree hash | `1acebe14e825a96a8d11f446fd34625ea736ecf8a32fbaa0283a59047c0519e7` |
| execution plan hash | `075fcbf589317fe199276bbdaca82fe191f3f7d707e3de89455e4fcae13fe86a` |
| dataset content hash | `c1995a7b6e03a2a11299f25ce46d317a1cc37fc10cb1c19b3cc73e71eb3c34c4` |
| checkpoint hash | 각 stage의 `run_manifest.json.checkpoints.items` |
| resolved config hash | 각 stage의 `run_signature.json.resolved_config_sha256` |
| 실행 명령 | `bash scripts/run_float32_digital_diagnostics.sh --profile full --parallel-devices 0,1,2 --fixed-reference-snr-db 10` |
| 파일 검증 | 이 디렉터리에서 `sha256sum -c checksums.sha256` |

## 관련 문서

- [실험 해석](../../docs/experiments/2026-08-28_float32_digital_step_normalization_full.md)
- [진단 프로토콜](../../docs/protocols/float32_digital_diagnostics.md)
- [향후 작업](../../docs/current/roadmap.md)

