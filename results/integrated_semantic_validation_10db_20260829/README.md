# 통합 semantic·hallucination·temporal 평가 (2026-08-29)

> **보존 사본.** 원본은 Git 비추적 경로
> `outputs/integrated_semantic_validation_10db_20260828_093017/`에 있다. 원격
> GPU 서버의 전체 원본을 로컬로 회수한 뒤 파일 수, 총 byte, 파일별 SHA-256을
> 비교했으며 모두 일치했다. 이 디렉터리에는 재현·판정에 필요한 핵심
> CSV·JSON·manifest만 보존한다.

## 결론

- 10영상 × 3 decoder policy × 4 guide profile의 **120/120 pair**와 총
  12,000 frame이 완료됐다. 실패와 non-finite는 0건이며 세 GPU worker provenance도
  계획과 일치한다.
- 평균 screening gate를 통과한 8개 후보 중 개발용 잠정 운영점은
  `few10 + candidate_both_omit`이다. 평균 219,459.7 bytes/video로
  `full50 + baseline`의 2,396,632.7 bytes/video보다 **90.843% 작고**,
  reconstruction 시간은 108.0504초에서 39.4540초로 **63.486% 감소**했다.
- 기준점 대비 PSNR 하락 0.2435dB, SSIM 하락 0.00188, LPIPS 증가 0.01558로
  사전 선언 평균 gate를 통과했다. closed PTC·severity·SFR·SDI도 통과했다.
- 하지만 open-world hallucination 증가는 평균 0.0195로 margin 0.05 이내인 반면
  paired 95% CI 상한은 **0.0525**다. additional object/frame 증가도 평균 0.0210,
  CI 상한 **0.0570**으로 margin을 넘는다. 증가는 `01_person_walk`과
  `02_car_pass`에 집중됐다. 따라서 이 후보는 **개발셋 잠정 후보**이며 최종
  운영점이 아니다.
- `full50 + candidate_both_omit`은 baseline보다 byte가 90.843% 작으면서 pixel,
  closed/open semantic 지표가 사실상 동일했다. 현재 reliable-digital 개발 조건에서는
  edge와 uncertainty를 모두 보내지 않는 guide 정책을 held-out 후보로 지지한다.
- VAE-direct는 23.4885초/video로 가장 빠르고 PSNR·LPIPS도 개선됐지만 SSIM 하락
  0.01129가 0.01 gate를 넘어 탈락했다. few10은 학습된 distilled model이 아니라
  현 production sampler의 10-step 근사다.

## 잠정 후보의 paired 효과

기준은 `full50 + baseline`, 후보는 `few10 + candidate_both_omit`이다.

| 지표 | 평균 효과 | paired 95% CI | margin | 판정 |
|---|---:|---:|---:|---|
| PSNR 하락 | 0.243512 dB | [0.147492, 0.381360] | 0.5 | 통과 |
| SSIM 하락 | 0.001882 | [-0.000367, 0.004046] | 0.01 | 통과 |
| LPIPS 증가 | 0.015583 | [0.012178, 0.019442] | 0.02 | 통과 |
| closed PTC 하락 | 0.001300 | [-0.011000, 0.014800] | 0.05 | 통과 |
| closed severity 증가 | 0.001500 | [-0.010800, 0.014700] | 0.05 | 통과 |
| open hallucination 증가 | 0.019500 | [0, **0.052500**] | 0.05 | 평균 통과, CI 경고 |
| additional object/frame 증가 | 0.021000 | [0, **0.057000**] | 0.05 | 평균 통과, CI 경고 |
| closed SFR 증가 | 0 | [0, 0] | 0.05 | 통과 |
| closed SDI 절대 변화 | 0.000261 | [0.000089, 0.000432] | 0.01 | 통과 |

## decoder·guide 판정

| policy + guide | bytes/video | elapsed/video | 핵심 판정 |
|---|---:|---:|---|
| full50 + baseline | 2,396,632.7 | 108.0504 s | 기준점 |
| full50 + both omit | 219,459.7 | 101.5461 s | semantic 포함 사실상 무손실 guide 제거 |
| few10 + both omit | **219,459.7** | **39.4540 s** | 평균 gate 통과, 개발셋 잠정 선택 |
| VAE-direct + both omit | 219,459.7 | 23.4885 s | SSIM gate 실패 |

`combined_ds4`는 356,824.7 bytes/video, edge-ds4 + uncertainty-omit은
288,092.4 bytes/video, both-omit은 219,459.7 bytes/video다. guide profile 자체의
영향은 full50에서 극미세하므로 both-omit이 현재 개발 조건의 최소-byte 후보다.

## 완전성·provenance

- `integrated_per_video.csv`: 120행, unique video-policy-profile 120개
- 모든 pair: 100 frame, 총 12,000 frame
- failed pair / non-finite metric: 0 / 0
- fixed-reference SNR: 10dB, config `fixed_int4`, seed 2025
- semantic backend evidence: CLIP/OWLv2/VQA 각각 47,744건
- worker provenance:
  - worker_00 → `cuda:0` → 4영상 → 48 pair
  - worker_01 → `cuda:1` → 3영상 → 36 pair
  - worker_02 → `cuda:2` → 3영상 → 36 pair
- 실험 commit: `5a8f2aa357925776b3c472aebbc0a3967ae17001`, tracked dirty false,
  branch main
- dataset manifest SHA-256:
  `8e5e192304cfd7582f42cf0087a1ac28ad063003fb20cadbfb751b3dc56f930c`

## 원본 회수·체크섬

- 원격/로컬 파일: 각각 27,548개
- 원격/로컬 총 크기: 각각 2,563,294,610 bytes
- 원격·로컬 전체 파일별 SHA-256 목록: 전부 일치
- 전체 목록 파일의 SHA-256:
  `68609800f826119810b5eac36b224943621caa852ba71b3b1801d103ca286a2c`
- 최초 rsync에서 root 소유 읽기 제한 파일 294개가 누락됐다. 원격 run 디렉터리에
  읽기 권한만 추가하고 증분 rsync한 뒤 전 파일 checksum을 다시 대조했다.
- 최상위 자동 artifact 6개는 `artifact_sha256.json`과 모두 일치한다.
- `checksums.sha256`은 자체 파일을 제외한 보존 파일 전부의 SHA-256을 기록한다.

## 다음 결정

별도 held-out 영상에서는 최소한 `full50 + candidate_both_omit`과
`few10 + candidate_both_omit`을 paired 비교한다. 최종 선택은 평균 gate뿐 아니라
hallucination/additional-object CI 상한까지 margin 안에 들어오는지로 결정한다.
VAE-direct는 primary confirmatory 후보에서 제외하고 SSIM 기준을 바꾸지 않는 한
탐색적 비교로만 유지한다.

## 관련 문서

- [실험 해석](../../docs/experiments/2026-08-29_integrated_semantic_validation_10db.md)
- [실행 준비](../../docs/experiments/2026-08-28_integrated_semantic_validation_preparation.md)
- [현재 상태](../../docs/current/status.md)
- [로드맵](../../docs/current/roadmap.md)
- [열린 이슈](../../docs/current/open_issues.md)
