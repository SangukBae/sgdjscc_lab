# ETRI 10개 영상 전송량·코덱 비교 방법

## 목적

동일한 10개 영상(512×256, 10 fps, 100프레임)에 대해 다음을 한 번에
측정한다.

1. 원본(raw) 및 실험 입력(processed) MP4 크기
2. 의미통신 수신단 입력 payload 크기
3. H.264, H.265, AV1 압축 파일 크기·bitrate·bits/pixel
4. 동일 원본 대비 PSNR, SSIM, LPIPS
5. 의미통신 결과와 PSNR이 같거나 높은 코덱 지점의 크기 비교

실행 도구는 `scripts/benchmark_etri_video_rate.py`이다.

## 반드시 구분할 전송량

현재 완료된 Wan 검증은 실제 JSCC bitstream을 저장하지 않았다. AWGN 채널에는
실수형 channel symbol이 전달되며, 변조·채널부호·양자화 규칙도 아직 하나의 실제
wire format으로 정의되지 않았다. 따라서 결과에는 다음 값을 분리해서 기록한다.

- `reference_payload_bytes`: 선택된 원본 키프레임 PNG, 세그먼트별 UTF-8 캡션,
  선택적 side-info 파일, manifest를 길이-prefix binary container로 직렬화한 실제
  파일 크기. 재현 가능한 비교용 payload이며 JSCC wire byte는 아니다.
- `jscc_visual_channel_symbols_proxy`: 실제 선택된 키프레임 수와 고정 모델 구조
  (128×128 patch당 4096 symbol)로 계산한 시각 symbol 수.
- `jscc_float32_storage_bytes_proxy`: 위 symbol을 float32 파일로 저장한다고 가정한
  저장 크기. 네트워크 크기가 아니다.
- `estimated_wire_bytes`: `--bits-per-channel-symbol`을 사용자가 명시한 경우에만
  계산되는 가정 기반 값.
- H.264/H.265/AV1의 `size_bytes`: FFmpeg가 만든 실제 압축 파일 크기.

따라서 논문이나 발표에서 `reference_payload_bytes`를 “실제 JSCC 전송 byte”라고
표현하면 안 된다. 실제 wire byte 비교에는 송신 시점의 channel tensor를 저장하고
양자화·변조·채널부호 규칙을 확정하는 후속 구현이 필요하다.

## 전체 10개 영상 실행

컨테이너 내부 연구 폴더에서 실행한다.

```bash
cd ~/SangukBae/Semantic/sgdjscc_lab

python scripts/benchmark_etri_video_rate.py \
  --semantic-run-root outputs/remote_hq_4090_20260816/generation/wan_skem_dsa \
  --output-root outputs/remote_hq_4090_20260816/rate_benchmark \
  --lpips-device cuda:0
```

기본 인코딩 지점은 다음과 같다.

- H.264/libx264: CRF 18, 23, 28, 33
- H.265/libx265: CRF 18, 23, 28, 33
- AV1/libaom-av1: CRF 20, 30, 40, 50

총 10×12=120개 압축 영상을 생성한다. 중간에 중단돼도 이미 만들어진 압축 파일은
재사용한다. 모두 다시 인코딩하려면 `--overwrite`를 추가한다.

## 빠른 동작 확인

LPIPS와 전체 조합을 생략하여 한 영상·코덱별 한 지점만 확인한다.

```bash
python scripts/benchmark_etri_video_rate.py \
  --semantic-run-root outputs/remote_hq_4090_20260816/generation/wan_skem_dsa \
  --output-root outputs/rate_benchmark_smoke \
  --video-ids 01_person_walk \
  --crf h264=28 --crf h265=28 --crf av1=40 \
  --no-lpips
```

실행 계획만 검사하려면 `--dry-run`을 사용한다.

## 실제 side information이 저장된 경우

영상별 side-info 파일이 아래처럼 존재할 때만 실제 파일 byte를 포함할 수 있다.

```text
side_info_root/
  01_person_walk/...
  02_car_pass/...
```

이때 `--side-info-root side_info_root`를 전달한다. 생성 로그에서 side-info 사용이
표시됐는데 파일이 없으면 `side_info_status=required_but_missing`으로 기록하여 완전한
payload라는 주장을 막는다. 현재 Wan 검증은 side-info를 생성 조건으로 사용하지 않아
`not_used`가 정상이다.

## 결과 파일

- `source_sizes.csv`: raw/processed 원본 크기
- `payloads.csv`: 키프레임·캡션·side-info 및 symbol accounting
- `payloads/*.sgdref`: 실제 생성된 결정적 reference payload
- `codec_results.csv`: 각 코덱/CRF의 크기와 PSNR·SSIM·LPIPS
- `comparison.csv`: 원본·의미통신·일반 코덱 통합 표
- `aggregate.csv`: 10개 영상 평균 및 총 크기
- `quality_matched.csv`: 의미통신 PSNR 이상을 만족하는 가장 작은 코덱 결과
- `summary.json`, `plan.json`: 실행 상태와 실험 계획

`quality_matched.csv`에서 샘플링한 CRF 중 의미통신 PSNR 이상인 지점이 없으면
`match_status=closest_available`로 표시한다. 정밀한 동급 화질 비교가 필요하면 해당
구간의 CRF를 추가해 다시 실행한다.

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

LPIPS는 기본적으로 100프레임 전부 평가한다. 빠른 근사 측정은
`--lpips-stride 5`처럼 설정할 수 있지만 최종 발표 수치는 기본값 1을 사용한다.
