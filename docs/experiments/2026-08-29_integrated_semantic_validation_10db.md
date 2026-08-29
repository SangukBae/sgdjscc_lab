# 통합 semantic·hallucination·temporal 10dB 검증 결과

## 범위와 완전성

- 원본: `outputs/integrated_semantic_validation_10db_20260828_093017/`
- 조건: fixed selector, `fixed_int4`, fixed-reference 10dB, seed 2025
- 격자: 10영상 × 3 decoder policy × 4 guide profile = 120 pair
- 결과: 120/120 pair, 12,000/12,000 frame, 실패·non-finite 0
- GPU provenance: `cuda:0/1/2`에 48/36/36 pair
- CLIP·OWLv2·VQA evidence: backend별 47,744건
- 실험 commit: `5a8f2aa357925776b3c472aebbc0a3967ae17001`, clean main

원격과 로컬의 27,548개 파일, 2,563,294,610 bytes 및 파일별 SHA-256이 모두
일치했다. 핵심 결과는
[`results/integrated_semantic_validation_10db_20260829/`](../../results/integrated_semantic_validation_10db_20260829/README.md)에
고정했다.

## 개발셋 잠정 운영점

평균 screening gate를 통과한 8개 중 byte 우선, 동률 시 reconstruction 시간을
최소화한 결과 `few10 + candidate_both_omit`이 선택됐다.

| 항목 | full50 + baseline | few10 + both omit | 변화 |
|---|---:|---:|---:|
| bytes/video | 2,396,632.7 | 219,459.7 | -90.843% |
| reconstruction elapsed/video | 108.0504 s | 39.4540 s | -63.486% |
| PSNR | 23.47628 | 23.23277 | -0.24351 dB |
| SSIM | 0.733301 | 0.731419 | -0.001882 |
| LPIPS | 0.253619 | 0.269202 | +0.015583 |
| closed PTC | 0.7716 | 0.7703 | -0.0013 |
| open hallucination rate | 0.0160 | 0.0355 | +0.0195 |
| open additional objects/100 frames | 2.7 | 4.8 | +2.1 |

PSNR·SSIM·LPIPS 및 closed semantic/temporal 평균 gate는 모두 통과했다. LPIPS
paired CI 상한도 0.01944로 margin 0.02 안이다.

## 남은 통계 위험

open hallucination 증가는 평균 0.0195로 margin 0.05 이내지만 paired 95% CI는
`[0, 0.0525]`다. additional object/frame 증가도 평균 0.0210이나 CI가
`[0, 0.0570]`이다. 두 CI 상한이 margin을 약간 넘으므로 평균 gate 통과를 최종
확정으로 해석하지 않는다.

영상별로는 `01_person_walk`의 hallucination/additional-object rate가 각각 +0.060,
`02_car_pass`가 +0.135/+0.150이었고 나머지 8영상은 0이었다. 별도 held-out은 이 두
유형처럼 사람·차량 객체가 많은 장면을 포함해야 한다.

## guide와 decoder의 분리 판정

`full50 + candidate_both_omit`은 219,459.7 bytes/video로 baseline 대비 90.843%
작다. PSNR 하락 0.00000735dB, SSIM 하락 0.000000081, LPIPS 증가
0.000000089였고 closed/open semantic 지표 차이는 0이었다. 따라서 현재
reliable-digital 개발 조건에서는 edge·uncertainty를 모두 생략하는 guide profile을
held-out에 올리는 것이 타당하다.

decoder는 아직 확정하지 않는다. few10은 2.739배 빠르지만 hallucination CI 경고가
있다. VAE-direct는 4.600배 빠르고 PSNR·LPIPS는 개선됐으나 SSIM 하락 0.01129가
사전 margin 0.01을 넘어 평균 gate에서 탈락했다.

## 다음 검증

별도 held-out 영상에서 `full50 + candidate_both_omit`을 보수적 기준으로 두고
`few10 + candidate_both_omit`을 primary candidate로 paired 비교한다. 최종 운영점은
pixel·closed semantic gate와 함께 open hallucination/additional-object CI 상한이
margin 안에 들어올 때만 확정한다. VAE-direct는 primary 후보가 아니라 탐색적
비교로 유지한다.

상세한 실행 계약은
[준비 문서](./2026-08-28_integrated_semantic_validation_preparation.md), 원시 수치와
checksum은 [보존 결과](../../results/integrated_semantic_validation_10db_20260829/README.md)를
기준으로 한다.
