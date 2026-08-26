---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes:
---

> [← 문서 색인](../README.md)

# ETRI 10개 영상 전송량·코덱 비교 방법

## 목적

- 입력
  - 영상 수: 10개
  - 해상도: 512×256
  - frame rate: 10 fps
  - 길이: 100 frame
- 측정 항목

1. 원본(raw) 및 실험 입력(processed) MP4 크기
2. 의미통신 수신단 입력 payload 크기
3. H.264, H.265, AV1 압축 파일 크기·bitrate·bits/pixel
4. 동일 원본 대비 PSNR, SSIM, LPIPS
5. 의미통신 결과와 PSNR이 같거나 높은 코덱 지점의 크기 비교

- 실행 도구는 `scripts/benchmark_etri_video_rate.py`이다.

## 반드시 구분할 전송량

- 현재 제약
  - Wan 검증은 JSCC bitstream 미저장
  - AWGN 경로는 실수형 channel symbol 사용
  - 변조·채널부호·양자화 wire format 미확정
- 기록 원칙
  - 실제 byte와 proxy 값을 분리

- `reference_payload_bytes`
  - 내용: keyframe PNG·UTF-8 caption·선택적 side-info·manifest
  - 형식: length-prefix binary container
  - 의미: 비교용 실제 파일 크기
  - 주의: JSCC wire byte 아님
- `jscc_visual_channel_symbols_proxy`
  - 근거: 실제 keyframe 수
  - 가정: 128×128 patch당 4096 symbol
- `jscc_float32_storage_bytes_proxy`
  - 가정: symbol당 float32 저장
  - 주의: network byte 아님
- `estimated_wire_bytes`
  - 조건: `--bits-per-channel-symbol` 지정
  - 성격: 가정 기반 값
- H.264/H.265/AV1의 `size_bytes`: FFmpeg가 만든 실제 압축 파일 크기.

- 금지 표현
  - `reference_payload_bytes` = 실제 JSCC 전송 byte
- 실제 wire byte 비교의 선행 조건
  - 송신 channel tensor 저장
  - 양자화 규칙 확정
  - 변조·채널부호 규칙 확정

## 전체 10개 영상 실행

- 컨테이너 내부 연구 폴더에서 실행한다.

```bash
cd ~/SangukBae/Semantic/sgdjscc_lab

python scripts/benchmark_etri_video_rate.py \
  --semantic-run-root outputs/remote_hq_4090_20260816/generation/wan_skem_dsa \
  --output-root outputs/remote_hq_4090_20260816/rate_benchmark \
  --lpips-device cuda:0
```

- FFmpeg 선택 순서
  - 요청 codec 지원 여부 검사
  - `/opt/ptest/bin/ffmpeg` 미지원 시 `/usr/bin/ffmpeg` 확인
  - system FFmpeg 미설치 시 아래 명령으로 설치

```bash
apt-get update
apt-get install -y ffmpeg
```

- 실행 파일 고정
  - `--ffmpeg /usr/bin/ffmpeg`
  - `--ffprobe /usr/bin/ffprobe`

- H.265 설정: `numa-pools=0`
  - 목적: Docker `set_mempolicy(2)` 경고 억제
  - 비영향: CRF·preset·영상 품질

- 기본 인코딩 지점은 다음과 같다.

- H.264/libx264: CRF 18, 23, 28, 33
- H.265/libx265: CRF 18, 23, 28, 33
- AV1/libaom-av1: CRF 20, 30, 40, 50

- 생성량: 10영상 × 12설정 = 120개
- 재시작
  - 기본: 기존 압축 파일 재사용
  - 전체 재인코딩: `--overwrite`

## 빠른 동작 확인

- LPIPS와 전체 조합을 생략하여 한 영상·코덱별 한 지점만 확인한다.

```bash
python scripts/benchmark_etri_video_rate.py \
  --semantic-run-root outputs/remote_hq_4090_20260816/generation/wan_skem_dsa \
  --output-root outputs/rate_benchmark_smoke \
  --video-ids 01_person_walk \
  --crf h264=28 --crf h265=28 --crf av1=40 \
  --no-lpips
```

- 실행 계획만 검사하려면 `--dry-run`을 사용한다.

## 실제 side information이 저장된 경우

- 영상별 side-info 파일이 아래처럼 존재할 때만 실제 파일 byte를 포함할 수 있다.

```text
side_info_root/
  01_person_walk/...
  02_car_pass/...
```

- 실행 옵션: `--side-info-root side_info_root`
- 상태 규칙
  - 사용 표시 + 파일 없음: `required_but_missing`
  - 현재 Wan 검증: `not_used`

## 결과 파일

- `source_sizes.csv`: raw/processed 원본 크기
- `payloads.csv`: 키프레임·캡션·side-info 및 symbol accounting
- `payloads/*.sgdref`: 실제 생성된 결정적 reference payload
- `codec_results.csv`: 각 코덱/CRF의 크기와 PSNR·SSIM·LPIPS
- `comparison.csv`: 원본·의미통신·일반 코덱 통합 표
- `aggregate.csv`: 10개 영상 평균 및 총 크기
- `quality_matched.csv`: 의미통신 PSNR 이상을 만족하는 가장 작은 코덱 결과
- `summary.json`, `plan.json`: 실행 상태와 실험 계획

- quality match 규칙
  - 목표 PSNR 충족 지점 있음: 최소 크기 선택
  - 목표 PSNR 충족 지점 없음: `closest_available`
  - 정밀 비교: 인접 CRF 추가 후 재실행

## CRF 지점 변경 예시

```bash
python scripts/benchmark_etri_video_rate.py \
  --semantic-run-root outputs/remote_hq_4090_20260816/generation/wan_skem_dsa \
  --output-root outputs/remote_hq_4090_20260816/rate_benchmark \
  --crf h264=20,22,24,26,28 \
  --crf h265=20,22,24,26,28 \
  --crf av1=24,28,32,36,40 \
  --lpips-device cuda:0
```

- LPIPS
  - 최종 평가: `--lpips-stride 1`
  - 빠른 근사: `--lpips-stride 5`
