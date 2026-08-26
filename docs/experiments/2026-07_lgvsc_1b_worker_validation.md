---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
experiment_commit: unknown
documentation_commit: ec367bb
supersedes:
---

> [← 문서 색인](../README.md)

# LGVSC 1B 외부 Worker 검증

## 판정

- 1B: 완료
- 실제 GPU 검증
  - Wan start-only: 성공
  - Wan bidirectional: 성공
  - SVD start-only: 성공
- 1C 전체 재현: 별도 미완료

## 구조

```text
TemporalPipeline
  → ExternalSegmentWorkerGenerator
    → manifest.json + keyframe PNG
    → 별도 Python subprocess
      → mock | SVD | Wan | callable
    → result.json | error.json
```

## Backend

| backend | 입력 | 상태 | 용도 |
|---|---|---|---|
| mock | start/end frame | 통과 | IPC 테스트 |
| SVD | start frame | GPU 통과 | 최소 실제 생성 |
| Wan I2V | start+caption | GPU 통과 | start-only |
| Wan FLF2V | start+end+caption | GPU 통과 | bidirectional |

## 핵심 수정

- bidirectional 실패 원인
  - 잘못된 I2V checkpoint 사용
  - FLF2V용 `pos_embed_seq_len` 부재
- 해결
  - end keyframe 있음: `Wan2.1-FLF2V-14B-720P`
  - end keyframe 없음: `Wan2.1-I2V-14B-480P`
  - segment별 checkpoint 자동 선택
- frame mapping
  - `target_index - start_frame_index`
  - 비연속 target 회귀 테스트 추가

## 실측

- Wan start-only
  - 64×64
  - 4 step
  - 약 63초
  - caption 사용 확인
- Wan bidirectional
  - `conditioning_mode=bidirectional`
  - `end_keyframe_index=12`
  - `n_generate=11`
- SVD
  - 생성 frame 1개
  - validation 통과

## 환경

- Wan I2V checkpoint: 약 84~90GB
- I2V+FLF2V 동시 보유: 약 180GB 가능
- 16GB GPU
  - sequential offload 사용
- `semantic-diffusers`
  - user-site package 필요
  - torch·torchaudio ABI 정합 필요

## 제한

- side-info 조건화: 미사용
- GPU smoke 성공 ≠ 영상 품질 우위
- 실제 10영상 비교: 1C에서 수행

## 관련

- [1C 재현 준비](./2026-07_lgvsc_1c_reproduction.md)
- [Tx/Rx 계약](../architecture/tx_rx_contract.md)
