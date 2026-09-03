---
status: frozen_amendment
updated: 2026-09-03
gate: G1
supersedes: negative_semantics_g1_v1_0
heldout_opened: false
---

# G1 v1.1 memory-safety amendment

## 발생 사실

정식 G1 v1.0 단일 RTX 4080 실행은 caption 40/40과 source-only OWLv2 calibration
40/40을 완료한 뒤 `few10`, seed 2025의 네 번째 영상 `ovis_valid_1b664206`에서
중단됐다. 앞의 세 영상은 복원됐지만 전체 v1.0 run은 논문 증거에서 제외한다.

오류는 MuGE EfficientNet edge extractor에서 3.94 GiB를 추가 할당하려다 발생한 CUDA
OOM이다. 실패 당시 PyTorch allocated memory는 11.13 GiB였고 GPU 여유는 2.69 GiB였다.
실험 종료 후 GPU memory가 정상 회수됐으므로 다른 프로세스와의 충돌은 아니다.

## 원인

v1.0은 aspect ratio를 유지해 longest side를 512로 줄였다. 실패 영상은
`1280×630 → 512×252`가 됐다. 원본 SGD-JSCC `split_image_v2`의 stride는 dimension의
128 나머지에 의존한다. 252는 `252 % 128 = 124`라서 height stride가 4가 되고,
정상적인 128-grid보다 훨씬 많은 overlapping patch를 만든다. MuGE가 이 patch batch를
한 번에 처리하면서 OOM이 발생했다.

## v1.1 변경

- aspect-ratio resize와 longest-side 512는 유지한다.
- 모델 입력 직전에 height와 width를 각각 다음 128 배수까지 중앙 zero-padding한다.
- 홀수 padding의 추가 1 pixel은 bottom/right에 둔다.
- 평가 시 padding 영역은 다시 잘라내고 원래 resized content만 OWLv2에 전달한다.
- resize/padding 값은 child reconstruction resume signature에 포함한다.
- v1.0 formal output root는 전체를 `INVALIDATED_PROTOCOL_V1_0`으로 표시하고 재사용하지 않는다.
- v1.0 caption은 padding 이전 source-only resize에서 이미 생성됐으므로 SHA-256, frame 수,
  model revision, v1.0 run spec을 검사한 경우에만 v1.1에서 adoption할 수 있다.
- v1.0 OWLv2 cache와 세 개 reconstruction은 재사용하지 않는다.
- smoke video를 실제 실패 영상 `ovis_valid_1b664206`으로 고정한다.

이 변경은 dataset, concept GT, evaluator model, threshold grid, fixed_int4, guide profile,
sampler step, seed, max-GOP 또는 reuse threshold를 바꾸지 않는다. Held-out은 열지 않았다.

## 새 실행 경계

새 결과 디렉터리는 `outputs/negative_semantics_g1_pilot_rtx4080_v1_1`이다. 기존
`outputs/negative_semantics_g1_pilot_rtx4080`에 이어 쓰면 안 된다. v1.1 실패 영상
2-frame smoke가 통과한 뒤에만 정식 재실행한다.
