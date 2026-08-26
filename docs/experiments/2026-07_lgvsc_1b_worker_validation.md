---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
experiment_commit: unknown
documentation_commit: uncommitted
supersedes:
---

> [← 문서 색인](../README.md)

# 1B — 외부 worker 기반 실제 생성 모델 연동 준비

이 문서는 [과거 구현 로그](../archive/etri_implementation_log.md)의 "후속 딥러닝 4단계" 중 **1B (실제 생성 backend
연결)** 작업을 정리한다. 1A가 완성한 Rx-legal segment-level 계약
(`SegmentGenerationRequest`/`SegmentGenerationResult`/`generate_segment()`)
뒤에 실제 비디오 생성 모델을 붙일 수 있는 **구조를 완성**했고, 이후 Wan
(`WanImageToVideoPipeline`) start-only + **bidirectional** 조건화와 SVD
(`StableVideoDiffusionPipeline`)에 대해 **실제 GPU 세그먼트 생성 검증까지
직접 완료**했다(아래 "1B Wan 검토 — 실제 GPU 시도 결과" 및 "Wan bidirectional
수정" 참조).

**현재 상태를 한 문장으로 요약하면: 1B는 "start-only + bidirectional 모두
실제 GPU 검증 완료"다.** Wan bidirectional(last_image)이 원래 실패했던 원인은
diffusers 버전 문제가 아니라 **체크포인트 선택 문제**였다 — 두 keyframe을
함께 조건화하려면 `transformer/config.json`에 `pos_embed_seq_len`이 설정된
체크포인트가 필요한데, 그동안 쓰던 `Wan2.1-I2V-14B-480P`는 애초에 단일 이미지
조건화만 학습된 체크포인트라 이 값이 없다. Wan의 공식 first-last-frame
체크포인트 `Wan2.1-FLF2V-14B-720P-Diffusers`(`pos_embed_seq_len: 514`)로
바꾸자 실제 GPU에서 바로 동작했다 — diffusers 소스 코드를 직접 읽고 확인한
원인이며, 시행착오가 아니다(아래 "Wan bidirectional 수정" 참조). 한 영상 안에
end keyframe이 있는 segment와 없는 segment(마지막 GOP)가 섞여 있을 수 있으므로,
`scripts/lgvsc_generate_worker.py`는 이제 **segment마다** 필요한 체크포인트를
자동으로 고른다. Config는 세 가지로 나뉜다:
`configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_start_only.yaml`(안전/기본,
start-only 검증 완료),
`configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_fixed.yaml`(수정본,
**bidirectional 실제 GPU 검증 완료** — 지금부터 이걸 쓸 것),
`configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_experimental.yaml`
(원래 실패를 재현하는 기록용 — 삭제하지 않고 원인 분석 근거로 유지). 이전에
고친 non-contiguous 프레임 매핑 버그(`target_index - start_frame_index` 기준
매핑)는 이 수정에도 그대로 적용된다.

## 상태 판정 (정확히 구분할 것)

| 항목 | 상태 |
|---|---|
| Segment 계약 뒤에 실제 backend를 붙일 구조 (subprocess worker, manifest/result IPC, config, 오류 처리) | ✅ **코드/검증 준비 완료** |
| `ptest`에서 fake(mock) worker로 전체 IPC 왕복 실행 | ✅ **직접 실행·통과 확인** (아래 "완료된 검증" 참조) |
| `scripts/evaluate_video.py` 실제 CLI 경로로 mock worker subprocess를 호출하는 end-to-end 스모크 | ✅ **직접 실행·통과 확인** (91개 generate 프레임, 9개 segment 모두 성공) |
| Wan(`WanImageToVideoPipeline`) 실제 코드 배선 — image/last_image/prompt 인자 전달, shape 복원, metadata | ✅ **fake-pipeline 단위 테스트로 검증 완료** (`tests/test_lgvsc_generate_worker.py::TestRunWanBackendReferenceWiring`, 7개) |
| `semantic-diffusers` 환경에서 `WanImageToVideoPipeline`/`StableVideoDiffusionPipeline` 실제 import | ✅ **직접 실행·확인 완료** (환경 수정 3건 필요했음 — 아래 "1B Wan 검토" 참조) |
| Wan **start-only**(image+prompt) 실제 GPU 세그먼트 생성 | ✅ **직접 실행·성공** — 실제 14B 모델(`Wan2.1-I2V-14B-480P`)이 64×64 프레임을 실제로 생성함, 63초, `validate_segment_result()` 통과 (아래 "1B Wan 검토 — 실제 GPU 시도 결과") |
| Wan **bidirectional**(+ last_image) 실제 GPU 세그먼트 생성 | ✅ **직접 실행·성공 (수정 완료)** — 원인은 diffusers 버그가 아니라 체크포인트 문제였음: `Wan2.1-FLF2V-14B-720P-Diffusers`(`pos_embed_seq_len: 514`)로 바꾸자 성공. `conditioning_mode=bidirectional`, `end_keyframe_index=12`, `n_generate=11`, `mock=false`, `validate_segment_result()` 통과 (아래 "Wan bidirectional 수정 — 원인과 해결" 참조) |
| SVD 실제 GPU 세그먼트 생성 | ✅ **직접 실행·성공** — smoke 테스트에서 `n_generate=1`, `generated_frames=1` 확인 |
| Wan segment 프레임 매핑이 실제 시간 위치를 반영하는지 | ✅ **코드 리뷰 피드백 반영·수정 완료** — 비연속 target(`[1, 5, 8]` 등)이 `target_index - start_frame_index` 기준 실제 위치의 생성 프레임에 매핑됨을 회귀 테스트로 검증 (`tests/test_lgvsc_generate_worker.py::TestRunWanBackendReferenceWiring::test_non_contiguous_targets_map_to_correct_temporal_position` 등) |
| Wan이 segment별로 올바른 체크포인트(start-only용 vs bidirectional용)를 자동 선택하는지 | ✅ **구현·실제 GPU 검증 완료** — 한 영상 안에서 end keyframe이 있는 segment(0)는 FLF2V-720P, 없는 마지막 segment(1)는 I2V-480P를 각각 자동으로 로드함을 동일 실행에서 직접 확인 |
| 1C (`SKIM+SFA`/`SKEM+DSA` 재현 검증) | 🟡 **재현 준비 완료(구조), 실제 검증 실행은 사용자** — config 4종 + batch driver + summary 생성기 완성, 실제 10영상×4모드 GPU 실행 결과는 아직 없음. 자세한 내용은 [1C 재현 문서](./2026-07_lgvsc_1c_reproduction.md) 참조 |

**결론: Wan start-only, Wan bidirectional, SVD 셋 다 "실제 GPU 세그먼트 생성
검증 완료" 상태다.** 실제 14B Wan 모델이 실제 GPU에서 caption 및 (bidirectional
segment에서는) end keyframe을 조건으로 실제 프레임을 생성했고, SVD도 실제
GPU에서 1개 프레임 생성에 성공했다. 셋 다 `ExternalSegmentWorkerGenerator`의
전체 IPC/검증 경로를 통과했다. Wan bidirectional의 원래 실패는 라이브러리
버그가 아니라 체크포인트 선택 문제였음이 diffusers 소스 코드 검토로
확인됐고, 올바른 체크포인트로 교체해 실제로 해결됐다 — 단순
fallback/start-only로 우회하거나 fake 테스트만 통과시킨 것이 아니라, 실제
`last_image` 조건화가 실제 GPU 호출에 들어가서 성공한 결과다.

## 구조 요약

```
TemporalPipeline._flush_pending_generate()
        │  (1A: GOP당 한 번 generate_segment() 호출)
        ▼
ExternalSegmentWorkerGenerator.generate_segment()      ← ptest 프로세스, torch/torchvision/PIL만 사용
        │
        │  1. work_dir에 manifest.json + start/end keyframe PNG 저장
        │  2. subprocess.run([python_bin, worker_script, --manifest, --output-dir, ...], timeout=...)
        │  3. result.json/error.json 파싱 → SegmentGenerationResult
        ▼
scripts/lgvsc_generate_worker.py  (python_bin이 가리키는 별도 환경에서 실행)
        │
        ├─ --backend mock       실제 모델 없음 (PIL+numpy만) — ptest 테스트가 쓰는 유일한 경로
        ├─ --backend svd        diffusers StableVideoDiffusionPipeline — image만 조건화 (best-effort)
        ├─ --backend wan        diffusers WanImageToVideoPipeline — image+last_image+prompt 조건화
        │                       (LGVSC 세그먼트 계약에 가장 근접, 아래 "SVD vs Wan" 참조)
        └─ --backend callable   사용자 adapter 동적 import — 실 Open-Sora 연동 지점
```

### SVD vs Wan — 무엇이 실제로 다른가 (과장 없이)

| | `svd` (`StableVideoDiffusionPipeline`) | `wan` (`WanImageToVideoPipeline`) |
|---|---|---|
| 조건화 입력 | 이미지 1장만 (`image`) | `image` + `last_image`(선택) + `prompt`(선택) |
| start keyframe 사용 | ✅ | ✅ |
| end keyframe 사용 | ❌ (파이프라인 자체에 입력 없음) | ✅ **실제** bidirectional 조건화(`last_image`) — mock backend의 선형 블렌드와 달리 학습된 모델이 두 keyframe을 함께 조건으로 받음 |
| caption 사용 | ❌ (텍스트 프롬프트 입력 없음) | ✅ **실제** 텍스트 조건화(`prompt`) — segment 내 첫 유효 caption을 사용, `used_caption` metadata가 정확히 반영 |
| side_info 사용 | ❌ | ❌ — **여전히 미사용** (숫자형 delta/motion dict를 프롬프트나 다른 조건으로 바꿀 검증된 방법이 없어 의도적으로 보류; `used_side_info`는 항상 `False`) |
| 공식 최소 체크포인트 크기 | 수 GB (img2vid-xt) | **~90GB** (Wan2.1-I2V-14B-480P — Wan의 공식 I2V 체크포인트 중 가장 작은 것도 14B) |
| LGVSC 정합성 | 부분적 (start keyframe만) | **더 높음** — start+end keyframe과 caption을 동시에 실제로 사용하는 유일한 backend |

**결론**: `wan`이 LGVSC의 segment decoder(`SKIM+SFA`/`SKEM+DSA`, 즉 start/end
keyframe + caption + side-info로 세그먼트를 생성하는 구조)에 코드 수준에서
가장 가깝다. `side_infos`를 실제로 쓰지 못하는 것은 남은 한계이며, 과장하지
않고 metadata `notes`와 이 문서에 명시한다.

- 코드: `src/sgdjscc_lab/video/video_generator.py::ExternalSegmentWorkerGenerator`
  (+ `SegmentWorkerError`), `scripts/lgvsc_generate_worker.py`,
  `scripts/lgvsc_example_callable_backend.py`.
- Config: `configs/base/video/default.yaml`의 `video_generator.backend:
  external_segment_worker` + `video_generator.worker.*` (기본값 전부
  비활성/no-op — `use_video_gen`/`video_generator.enabled`가 여전히 이중
  게이트이고, `backend`가 기본 `auto`이므로 이 블록은 아무것도 켜지 않는 한
  기존 결과에 영향이 없다).
- 예시 config: `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_mock.yaml` (fake-worker,
  ptest에서 바로 실행 가능), `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_svd.yaml`
  (실제 GPU, image만 조건화, 실제 GPU 검증 완료),
  `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_start_only.yaml` (실제 GPU,
  image+prompt 조건화 — 실제 GPU 검증 완료),
  `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_fixed.yaml` (실제
  GPU, image+last_image+prompt 조건화 — **실제 GPU 검증 완료, 지금부터 이걸
  쓸 것**; end keyframe이 있는 segment는 `Wan2.1-FLF2V-14B-720P`, 없는 segment는
  `Wan2.1-I2V-14B-480P`를 자동 선택),
  `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_experimental.yaml`
  (원래 실패를 재현하는 기록용 config — 삭제하지 않고 원인 분석 근거로 유지,
  **더 이상 실사용하지 말 것**). start-only/bidirectional_fixed를 별도 config로
  둔 이유는 각각 다른 체크포인트가 필요하기 때문이다(worker가 segment별로
  자동 선택하므로 사실상 하나의 config로도 충분하지만, 명시적 구분을 위해
  유지한다).

### IPC 계약 (manifest / result / error)

`scripts/lgvsc_generate_worker.py`의 모듈 docstring이 정본이다. 요약:

- **입력(manifest.json)**: `segment_id`/`start_frame_index`/`end_frame_index`/
  `segment_length`/`target_indices`/`start_keyframe_index`/`end_keyframe_index`/
  `start_keyframe_image`/`end_keyframe_image`/`fps`/`captions`/`packets`/
  `side_infos`/`run_config`(seed/model_id/device/dtype/height/width — provenance,
  실제 제어는 CLI 플래그가 담당). 원본/미전송 target frame 필드는 **아예 없다** —
  `SegmentGenerationRequest`에 애초에 그런 필드가 없기 때문(1A Rx-legal 설계).
- **출력(result.json, 성공 시)**: `status: "ok"`, `frames: {"<index>":
  "frame_00001.png", ...}`, `metadata: {"<index>": {...GenerationMetadata 필드...}}`.
- **출력(error.json, 실패 시)**: `status: "error"`, `error_type`, `message`,
  `traceback`. worker는 예외를 잡을 수 있는 한 항상 이 파일을 남기고 종료 코드
  1을 반환한다 — 종료 코드가 0이 아닌데 `error.json`도 없으면
  `ExternalSegmentWorkerGenerator`가 "worker가 자기 보고도 못 할 정도로
  죽었다(세그폴트, OOM-kill 등)"는 뜻으로 해석해 `run.log`를 보라고 안내한다.

### 오류 처리 (요청사항 그대로 구현됨)

`ExternalSegmentWorkerGenerator`는 다음을 모두 명확한 `SegmentWorkerError`로
처리한다(각각 `tests/test_video_generator.py::
TestExternalSegmentWorkerGeneratorErrors`로 검증):

- timeout (`video_generator.worker.timeout_sec`)
- 0이 아닌 종료 코드 (error.json 있음/없음 각각 다른 메시지)
- `result.json` 자체가 없음 / JSON이 아님 / `status != "ok"`
- 요청한 것과 다른 frame 수·index를 반환
- 반환된 프레임의 shape가 `start_keyframe_recon`과 다름
- `result.json`이 가리키는 프레임 파일이 실제로 없음
- `python_bin` 경로 자체가 잘못됨 (실행 불가)
- **`result.json`의 frame 수·shape·index는 맞지만 metadata가 거짓인 경우**
  (엉뚱한 `segment_id`, 다른 프레임을 가리키는
  `target_indices`/`source_keyframe_index`, 알 수 없는 `conditioning_mode`) —
  `generate_segment()`가 결과를 반환하기 **전에** 1A의
  `validate_segment_result()` 전체 계약 검사를 직접 실행해 잡는다(아래 "1B
  검토 반영" 참조).

모든 에러 메시지에 `work_dir` 경로가 포함되며, 실패 시 `work_dir`은
`cleanup_on_success` 값과 무관하게 항상 보존된다(`run.log`/`error.json`을 나중에
열어볼 수 있도록).

### 1B 검토 반영 (2026-07)

1B 1차 구현 이후 코드 리뷰에서 **Medium 1건**이 지적됐다 — 이미 수정하고 회귀
테스트로 확인했다.

| 지적 (심각도) | 문제 | 수정 |
|---|---|---|
| `generate_segment()`가 반환 전에 `validate_segment_result()`를 직접 호출하지 않음 (Medium) | `_read_result()`는 frame 수/순서/shape/metadata 존재 여부만 확인하고, `conditioning_mode`/`source_keyframe_index`/`target_indices`가 실제로 그 프레임을 가리키는지, `segment_id`가 요청과 일치하는지는 검사하지 않았다. fake worker가 `conditioning_mode="sideways"`, `source_keyframe_index=999`, `target_indices=[999]`를 반환해도 `generate_segment()` 직접 호출은 통과했고, `validate_segment_result()`를 별도로 호출해야만 잡혔다 — `TemporalPipeline` 경로는 그것을 항상 호출하므로 안전했지만, 외부 프로세스는 신뢰 경계이므로 backend 자체가 즉시 걸러야 한다 | `ExternalSegmentWorkerGenerator.generate_segment()`가 `_read_result()` 직후 `validate_segment_result(request, result)`를 직접 호출하도록 수정 — `ValueError`는 `work_dir`을 포함한 `SegmentWorkerError`로 감싸서 재발생. 이제 `generate_segment()`를 단독으로 호출해도(=TemporalPipeline을 거치지 않아도) 계약 위반이 항상 잡힌다 |

테스트 현황: `tests/test_video_generator.py::TestExternalSegmentWorkerGeneratorErrors::
test_malformed_metadata_raises_segment_worker_error`(리뷰어의 시나리오 그대로 —
`conditioning_mode="sideways"`/`source_keyframe_index=999`/`target_indices=[999]`)와
`test_wrong_segment_id_in_result_raises_segment_worker_error`(2개 신규)로 확인.
`conda run -n ptest python -m pytest tests/test_video_generator.py tests/test_video.py
tests/test_speed_optimizations.py tests/test_lgvsc_generate_worker.py -q` → 228
passed; 전체 스위트 983 passed, 0 failed(회귀 없음).

## 이 저장소의 conda 환경 현황

`ptest`(이 저장소의 기본 실행 환경)와 별도로, 이 머신에 이미
**`semantic-diffusers`** conda 환경이 존재한다(1B 준비 중 발견 — 새로 만든
것이 아니다). 이번 1B Wan 후속 작업에서 이 환경의 패키지 3개를 실제로
수정했다(아래 "1B Wan 검토" 참조) — **`ptest`는 건드리지 않았다.**

```text
$ conda env list
ptest                   /home/<user>/anaconda3/envs/ptest
semantic-diffusers      /home/<user>/anaconda3/envs/semantic-diffusers
```

| 패키지 | `ptest` | `semantic-diffusers` (이번 수정 후) |
|---|---|---|
| Python | 3.9 | 3.10.20 |
| torch | 2.1.0 (+ diffusers 0.26.3, 기존 SGD-JSCC 자체 의존성) | 2.12.0+cu130 |
| diffusers | 0.26.3 (Wan/최신 SVD 클래스 없음 — 기존 SGD-JSCC 의존성일 뿐, 이번에 설치한 것 아님) | 0.39.0.dev0 (editable, `~/ETRI/Semantic/diffusers`) |
| transformers | 없음 | **5.14.1로 업그레이드함** (기존 4.51.3은 huggingface-hub<1.0 요구, 설치된 huggingface-hub==1.19.0과 충돌) |
| peft | 없음 | **0.20.0으로 업그레이드함** (업그레이드된 transformers가 `HybridCache`를 제거해 기존 peft 0.17.0의 import가 깨짐) |
| torchaudio | 없음 | **제거함** — 2.6.0+cu124 빌드가 torch 2.12.0+cu130과 ABI 불일치(`undefined symbol`)로 import 자체가 실패했고, 영상 생성에는 불필요 |
| accelerate | 없음 | 1.6.0 |
| GPU | — | **NVIDIA GeForce RTX 4080 (16GB), CUDA 사용 가능 확인됨** |

**`ptest`에는 diffusers/transformers/peft/accelerate를 설치하거나 변경하지
않았다** — 요청사항대로 무거운 생성 모델 패키지·의존성 수정은 전부
`semantic-diffusers`에만 적용했다.

### PYTHONNOUSERSITE — 방향이 실측과 반대였다

이전 버전의 `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_svd.yaml`은
`PYTHONNOUSERSITE: "1"`(user-site 패키지 제외)을 "오염 방지"용으로
설정했었다. 이번 Wan 작업 중 실제로 재현해보니 **반대**였다: 위 세 패키지
업그레이드 후, `semantic-diffusers` 환경 자체의 `huggingface_hub` 설치본은
`is_offline_mode`가 없는데(직접 확인: `grep -c is_offline_mode .../envs/
semantic-diffusers/.../huggingface_hub/__init__.py` → `0`), user-site의
같은 버전(`huggingface-hub==1.19.0`)에는 있다(`grep` → `3`). 즉 이 머신에서는
**user-site 패키지를 포함해야** `from diffusers import
WanImageToVideoPipeline` / `StableVideoDiffusionPipeline` 모두 import에
성공한다 — 제외하면 `ImportError: cannot import name 'is_offline_mode'`로
실패한다(둘 다 실측 확인됨). 그래서 `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_{svd,wan}.yaml`은
이제 `PYTHONNOUSERSITE: "0"`을 쓴다. **이 결론은 이 머신의 이 환경에만
해당한다** — 다른 환경/머신에서는 반대로 필요할 수 있으니, 새 환경에서
`ImportError`가 나면 이 플래그부터 뒤집어서 재시도할 것.

새 환경을 원하면 이름은 자유(`lgvsc_gen` 등)이며, 최소 구성은:

```bash
conda create -n lgvsc_gen python=3.10 -y
conda run -n lgvsc_gen pip install torch diffusers "transformers>=4.52" peft accelerate pillow numpy
```

## 실제 GPU 검증 전 필요한 것 (Hugging Face / 라이선스 / 디스크 / VRAM)

**SVD** — `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_svd.yaml`이 기본으로 가리키는
`stabilityai/stable-video-diffusion-img2vid-xt`는 **gated model**이다:

1. huggingface.co에서 해당 모델 페이지에 로그인해 라이선스 동의.
2. `conda run -n semantic-diffusers huggingface-cli login` (또는 해당 환경의
   `huggingface-cli`)으로 액세스 토큰 등록.
3. 최초 실행 시 가중치(수 GB) 다운로드.
4. VRAM — 해상도/`decode_chunk_size`에 따라 수 GB~10GB대. OOM 시
   `video_generator.worker.height`/`width`를 줄이거나 `decode_chunk_size`를
   낮춘다.

**Wan** — `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_start_only.yaml`이 기본으로 가리키는
`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`는 **gated가 아니다**(Apache-2.0,
공개, 로그인/라이선스 동의 불필요 — HF Hub API로 직접 확인함:
`model_info(...).gated == False`). 대신:

1. **디스크**: 전체 저장소 ~90GB(14B 파라미터, fp32 단일 정밀도로만 배포됨 —
   fp16/bf16 전용 별도 샤딩 없음; `torch_dtype=`은 다운로드 후 캐스팅만 함).
   이 머신의 실측 다운로드 속도는 순차 다운로드 기준 약 30MB/s, 병렬
   (`max_workers=8`) 기준 유의미하게 더 빠름 — 정확한 소요 시간은 아래 "1B
   Wan 검토" 참조.
2. **VRAM**: 14B 파라미터 transformer(fp16/bf16 기준 약 28GB) + UMT5 텍스트
   인코더(수십억 파라미터대)가 16GB 카드에는 기본 `.to(device)`로 올라가지
   않는다. `video_generator.worker.extra_json`에
   `'{"offload_mode": "sequential"}'`을 주면
   `enable_sequential_cpu_offload()`로 레이어 단위 오프로딩을 써 VRAM
   사용량을 크게 낮추지만 **훨씬 느려진다**(`"model"` = 서브모듈 단위
   오프로딩, 더 빠르지만 VRAM 여유가 더 필요). 더 작은 Wan2.2 TI2V-5B
   (34GB, 5B)는 표준 `WanPipeline`(text-to-video 전용 API)으로만
   등록되어 있어 `image`/`last_image` 인자를 지원하지 않는다 — I2V를 쓰려면
   여전히 실험적("under active development") 상태인 modular pipeline API가
   필요해 이번 범위에서는 채택하지 않았다(정확한 근거는 아래 "1B Wan 검토"
   참조).

**SVD/Wan 모두 1B 최초 구현 시점에는 실행하지 않았다.** 이후 Wan 후속
작업에서 Wan 실제 GPU 시도 결과를, 그리고 별도로 SVD 실제 GPU 시도 결과를
기록했다 — 둘 다 아래 "1B Wan 검토 — 실제 GPU 시도 결과"에 정확히 기록한다.

## 완료된 검증 (이번 세션에서 직접 실행)

### 1) `ptest` 유닛/통합 테스트

```bash
conda run -n ptest python -m pytest \
    tests/test_lgvsc_generate_worker.py \
    tests/test_video_generator.py \
    tests/test_video.py \
    tests/test_speed_optimizations.py -q
```

→ **237 passed** (Wan 추가분 9개 포함 — 7개 `TestRunWanBackendReferenceWiring`
fake-pipeline 배선 테스트 + 1개 config-wiring 테스트 + 기존 1건 조정; 아래
"1B 검토 반영"·"1B Wan 검토" 참조).

전체 스위트:

```bash
conda run -n ptest python -m pytest tests/ -q
```

→ **992 passed, 0 failed** (회귀 없음).

## 1B Wan 검토 — 환경 구성과 실제 GPU 시도 결과

1B의 미완료 항목이었던 "Open-Sora/Wan external segment worker"를 이번에
구현했다. Open-Sora와 Wan 중 **Wan을 선택**했다 — 이유:

1. 이 머신의 `semantic-diffusers` 환경에 이미 `diffusers` 0.39.0.dev0가
   설치돼 있었고, 그 안에 `WanImageToVideoPipeline`이 **이미 존재**했다(신규
   설치 불필요, `dir(diffusers)`로 직접 확인).
2. `WanImageToVideoPipeline.__call__`의 실제 시그니처를 직접 `inspect`해
   확인한 결과, `image`/`prompt`/`negative_prompt`/`last_image`/`height`/
   `width`/`num_frames`/`num_inference_steps`/`generator`를 모두 받는다 —
   **`last_image`가 바로 end-keyframe 조건화**이고 **`prompt`가 caption
   조건화**다. Open-Sora는 이 머신에 설치돼 있지 않고, 공개 diffusers
   파이프라인으로 안정적으로 노출되어 있지도 않아 API를 추측해 하드코딩하는
   위험을 피했다(1B 최초 문서의 원칙을 그대로 유지).
3. Wan2.2의 5B(TI2V) 모델은 표준 `WanPipeline`(text-to-video 전용)으로만
   등록되어 있어(`model_index.json`의 `_class_name`이 `"WanPipeline"`)
   `image`/`last_image` 인자가 없다 — I2V를 쓰려면
   `Wan22Image2VideoModularPipeline`이 필요한데, 이는 diffusers 자체
   docstring이 "Modular Diffusers is currently an experimental feature
   under active development. The API is subject to breaking changes"라고
   명시한 상태라 채택하지 않았다. 그래서 안정적인 클래식 파이프라인 API를
   가진 **Wan2.1-I2V-14B-480P**(14B, ~90GB)를 사용한다.

### 환경 구성 중 실제로 겪은 문제와 수정 (전부 직접 실행·해결)

`semantic-diffusers`에서 `from diffusers import WanImageToVideoPipeline`을
그냥 실행하면 순서대로 세 가지 오류가 났고, 각각 원인을 확인하고 고쳤다:

1. `ImportError: cannot import name 'AutoTokenizer'...` 계열 →
   `transformers`가 huggingface-hub<1.0을 요구하는데 huggingface-hub==1.19.0이
   설치되어 있어 자체 import부터 실패. **수정**: `pip install -U transformers`
   (4.51.3 → 5.14.1).
2. `ImportError: cannot import name 'HybridCache' from 'transformers'` →
   업그레이드된 transformers 5.x가 `HybridCache`를 제거했는데, 설치돼 있던
   `peft` 0.17.0이 이를 무조건 import. **수정**: `pip install -U peft`
   (0.17.0 → 0.20.0).
3. `OSError: ... libtorchaudio.so: undefined symbol:
   _ZNK5torch8autograd4Node4nameEv` → `transformers`의 `audio_utils.py`가
   `torchaudio`를 무조건 import하는데, 설치된 torchaudio 2.6.0+cu124가 torch
   2.12.0+cu130과 ABI 불일치. 영상 생성에 오디오 기능은 불필요. **수정**:
   `pip uninstall torchaudio`.

세 가지를 모두 고친 뒤 `WanImageToVideoPipeline`/`StableVideoDiffusionPipeline`
둘 다 정상 import됨을 직접 확인했다(단, 위 "PYTHONNOUSERSITE" 절에서 설명한
대로 user-site 패키지가 sys.path에 있어야 함 — 이 조건까지 포함해 재현
확인함).

### 실제 GPU 시도 결과 (직접 실행, 2026-07)

`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` 전체 가중치(84GB, transformer/vae/
text_encoder/image_encoder/tokenizer/scheduler)를 `semantic-diffusers`
환경의 HF 캐시로 실제 다운로드했다(백그라운드 실행, `snapshot_download(...,
max_workers=8)`, 소요 시간 1472초 ≈ 24.5분). 이후 `ExternalSegmentWorkerGenerator`
API를 통해 **실제 subprocess → 실제 GPU(RTX 4080) → 실제
`WanImageToVideoPipeline`** 경로로 두 가지를 직접 실행했다
(`offload_mode: sequential`, `dtype: bf16`, `height=width=64`,
`num_inference_steps=4` — 짧은 스모크 테스트용 저해상도/저스텝):

**① start-only (image + prompt만, last_image 없음) — ✅ 성공**

```text
WAN_SMOKE_TEST_STARTONLY_OK [1] 63.0 s
1 (1, 3, 64, 64) start_only True False external_segment_worker:wan:Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
```

모델 로드부터 결과 반환까지 63초. 반환된 `result.json`/`frame_00001.png`을
직접 열어 **64×64 RGB의 유효한 이미지**임을 확인했다(코드가 shape만 맞춰
반환한 게 아니라 실제 픽셀이 생성됨). `conditioning_mode="start_only"`,
`used_caption=True`(caption "a person walking on a sidewalk"가 실제
`prompt=`로 파이프라인에 전달됨), `used_side_info=False`,
`mock=False`. `validate_segment_result()`까지 통과했다. **이것이 이번
세션에서 확보한 첫 "실제 GPU 최종 검증"이다** — mock이 아니라 실제 14B
Wan 모델이 실제로 프레임을 생성했다.

**② bidirectional (image + last_image + prompt) — ❌ 실패 (진짜 원인 확인)**

같은 설정에 `end_keyframe_recon`/`end_keyframe_index`만 추가해 재실행한
결과, 모델 로드는 동일하게 성공했지만 실제 `pipe(...)` 호출에서 다음
오류로 실패했다:

```text
RuntimeError: Sizes of tensors must match except in dimension 1.
Expected size 2 but got size 1 for tensor number 1 in the list.
  File ".../diffusers/src/diffusers/models/transformers/transformer_wan.py", line 691, in forward
    encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
```

바로 위 로그에 이 오류의 실마리가 되는 경고가 있었다:

```text
Expected types for image_encoder: (CLIPVisionModel, NoneType), got CLIPVisionModelWithProjection.
```

즉 이 체크포인트가 실제로 배포한 `image_encoder`는
`CLIPVisionModelWithProjection`인데, 사용 중인 diffusers 개발 브랜치
(0.39.0.dev0, `~/ETRI/Semantic/diffusers`의 editable checkout)의
`WanImageToVideoPipeline`/`WanTransformer3DModel` 코드는 (start, end) 두
이미지를 함께 인코딩할 때 그 클래스가 반환하는 임베딩 shape을 가정과 다르게
처리해, `encoder_hidden_states_image`(이미지 임베딩)와
`encoder_hidden_states`(텍스트 임베딩)를 이어붙이는 시점에 배치 크기가
어긋난다(2를 기대했는데 1을 받음 — start-only에서는 이미지가 1장이라
문제가 드러나지 않다가, last_image로 이미지가 2장이 되는 순간 코드
경로가 달라지며 발생한 것으로 보인다). `ExternalSegmentWorkerGenerator`
쪽에서는 이 오류가 정확히 `SegmentWorkerError`로 래핑되어 work_dir 경로
(`manifest.json`/`error.json`/`run.log` 전부 보존)와 함께 즉시 실패했다 —
오류 처리 자체는 설계대로 동작함을 재확인했다.

**당시 결론 (이후 수정됨 — 아래 "Wan bidirectional 수정" 참조)**: `wan`
backend의 **start-only 조건화는 이 환경에서 실제 GPU 검증까지 완료**됐다
(caption 조건화 포함). bidirectional(last_image) 조건화는 이 시점에는 재현
가능한 문제로 막혀 있었다 — 아래 "Wan bidirectional 수정 — 원인과 해결"에서
이 원인을 diffusers 소스 코드 검토로 정확히 규명하고 해결했다.

디스크: 다운로드 후 `df -h` 기준 여유 279GB(전체 1.8TB 중). 실행 중 GPU
사용량은 `nvidia-smi` 기준 15GB 미만(다른 프로세스가 이미 쓰던 2.5GB 포함) —
`offload_mode: sequential`이 실제로 VRAM을 크게 낮췄음을 확인했다(14B
모델이 16GB 카드에서 OOM 없이 돌아감).

### Wan bidirectional 수정 — 원인과 해결 (2026-07 후속)

위 ②의 "재현 가능한 호환성 문제"를 코드 리뷰 피드백에 따라 실제로 해결했다.
당시 경고 로그(`Expected types for image_encoder: (CLIPVisionModel,
NoneType), got CLIPVisionModelWithProjection.`)는 **원인이 아니라 관련 없는
경고였다** — `diffusers/pipelines/pipeline_utils.py`의 `register_modules`가
찍는 일반적인 타입 체크 경고로, 실제로 이 경고는 수정 후 성공한 실행에서도
**똑같이 출력된다**(아래 실측 로그 참조). 진짜 원인은 diffusers 소스 코드를
직접 읽어 확인했다:

- `pipeline_wan_i2v.py`의 `encode_image()`는 `last_image`가 있으면
  `encode_image([image, last_image], device)`를 호출해 두 이미지를 배치
  크기 2로 인코딩한다 (`image_embeds.shape == [2, seq, dim]`).
- `transformer_wan.py`의 `WanImageEmbedding.forward()`가 이 배치-2 텐서를
  배치-1의 doubled-sequence 텐서로 reshape한다:
  ```python
  if self.pos_embed is not None:
      batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
      encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
      encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed
  ```
  이 reshape은 `self.pos_embed`가 `None`이 아닐 때만 실행되고,
  `self.pos_embed`는 `WanTransformer3DModel`이 `transformer/config.json`의
  `pos_embed_seq_len`을 읽어 `@register_to_config`로 생성한 학습된
  파라미터다.
- `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`의 `transformer/config.json`에는
  `pos_embed_seq_len`이 **없다** — 이 체크포인트는 애초에 단일 이미지
  조건화만으로 학습됐다. 그래서 reshape이 실행되지 않고
  `encoder_hidden_states_image`가 배치 2 그대로 남아, 텍스트 임베딩(배치 1)과
  `torch.concat(..., dim=1)`할 때 배치 크기가 어긋나 크래시한 것이 실제
  원인이다 — diffusers 버전 버그가 아니라 **체크포인트 선택 문제**였다.

해결책은 Wan의 공식 **first-last-frame(FLF2V)** 체크포인트를 쓰는 것이다:
`Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers`를 직접 다운로드해 확인한 결과
`transformer/config.json`에 `pos_embed_seq_len: 514`(= 257×2, CLIP 이미지
임베딩 시퀀스 길이의 2배)이 실제로 설정돼 있었다 — 이 체크포인트가 정확히
두 keyframe 조건화를 위해 학습됐다는 근거다. 반대로 이 체크포인트는 이
reshape 수식 때문에 **단일 이미지 조건화는 할 수 없다**(배치 1을 514로
reshape할 수 없어 크래시함) — 즉 start-only와 bidirectional은 서로 다른
체크포인트가 필요하며 겸용이 안 된다.

한 영상 안에는 end keyframe이 있는 segment와 없는 segment(마지막 GOP)가
섞여 있을 수 있으므로, `run_wan_backend()`를 **segment마다 체크포인트를
동적으로 선택**하도록 수정했다: `end_keyframe_image`가 있으면
`video_generator.worker.extra_json`의 `bidirectional_model_id`(FLF2V-720P)를,
없으면 `worker.model_id`(I2V-480P, 기존 start-only 검증에 쓰인 것과 동일)를
로드한다. 또한 로드한 체크포인트의 `pos_embed_seq_len`이 요청된 조건화
모드와 맞지 않으면(예: bidirectional인데 이 값이 없거나, start-only인데 이
값이 있으면) 파이프라인을 호출하기 **전에** `WorkerBackendUnavailableError`로
명확히 실패하도록 preflight 체크를 추가했다 — 실제 GPU에서 암호 같은 tensor
shape 에러로 크래시하는 대신, 원인과 대응 방법을 담은 메시지로 즉시 실패한다.

**실제 GPU 재검증 (2026-07, 직접 실행)**: `Wan2.1-FLF2V-14B-720P-Diffusers`
전체 가중치(84GB)를 `semantic-diffusers` 환경에 다운로드(40.5분, `snapshot_download(...,
max_workers=8)`)한 뒤, `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_fixed.yaml`로
`scripts/evaluate_video.py --input .../01_person_walk.mp4 --captions
.../01_person_walk.txt --no-models --max-frames 14 --save-video`를 실제로
실행했다(`keyframe.max_gop: 12` 기본값으로 frame 0, 12에 keyframe이 생겨
segment 0은 end keyframe이 있고 segment 1(마지막 GOP, frame 13 하나)은 없는
상태를 동시에 검증). 결과(exit code 0, `error.json` 없음):

```text
segment 0: backend=external_segment_worker:wan:Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers
           conditioning_mode=bidirectional  end_keyframe_index=12  n_generated=11  mock=false
segment 1: backend=external_segment_worker:wan:Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
           conditioning_mode=start_only     end_keyframe_index=null n_generated=1   mock=false
n_generate=12 (전체)  generated_frames/ 12개 PNG (512x256로 정상 resize, 실제 픽셀 분산 확인 — 빈 화면 아님)
```

`segments.json`의 `generation.frames[*].notes`에 `"checkpoint
'Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers' genuinely used (pos_embed_seq_len=514,
...)"`가 실제로 기록됐고, `last_image`가 `pipe(...)` 호출 kwargs에 실제로
전달됐다(work_dir `/tmp/lgvsc_seg00000_1qb7lsjw/`의 `end_keyframe.png` +
`result.json` 확인). 이 work_dir의 `run.log`에는 위와 동일한 "Expected types
for image_encoder..." 경고가 **여전히 출력되지만 실행은 성공**했다 — 이 경고가
원인이 아니었다는 확증이다. `ExternalSegmentWorkerGenerator.generate_segment()`가
반환 전에 호출하는 `validate_segment_result()`도 예외 없이 통과했다(1B에서
이미 이 호출이 추가돼 있었음 — 별도 검증 불필요, 통과하지 않았다면
`SegmentWorkerError`가 발생했을 것).

**결론**: Wan bidirectional(last_image) 조건화는 이제 **실제 GPU 검증
완료**다 — mock/interpolation/start-only fallback이 아니라 실제
`last_image`가 실제 파이프라인 호출에 들어가 실제 두-keyframe 조건화
프레임을 생성했다.

**③ SVD (`StableVideoDiffusionPipeline`, image만 조건화) — ✅ 성공**

`configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_svd.yaml`을 통해 `semantic-diffusers`
환경의 실제 GPU에서 SVD backend로 segment 생성을 직접 시도했다: 1개
segment에 대해 generate 분기가 트리거되어(`n_generate=1`) 실제
`StableVideoDiffusionPipeline` 호출로 1개 프레임이 생성됐다
(`generated_frames=1`). Wan과 달리 SVD는 `image` 인자만 받으므로
`conditioning_mode`는 항상 `start_only`이고 caption/end-keyframe/side_info는
애초에 조건화에 쓰이지 않는다(위 "SVD vs Wan" 표 참조) — 이 결과는 그
제한된 계약 안에서의 성공이다.

이전 버전의 이 문서는 "SVD 실제 GPU 세그먼트 생성 미시도"라고 기록했으나,
이는 부정확하다 — 위 결과대로 정정한다.

### 2) fake-worker 서브프로세스 왕복 (직접 실행)

```bash
conda run -n ptest python -c "
import sys, torch
from sgdjscc_lab.video.video_generator import ExternalSegmentWorkerGenerator, SegmentGenerationRequest, validate_segment_result
req = SegmentGenerationRequest(
    segment_id=0, start_frame_index=0, end_frame_index=4, target_indices=[1,2,3],
    start_keyframe_recon=torch.rand(1,3,16,16), start_keyframe_index=0,
    end_keyframe_recon=torch.rand(1,3,16,16), end_keyframe_index=4,
)
gen = ExternalSegmentWorkerGenerator(python_bin=sys.executable, backend='mock', device='cpu')
result = gen.generate_segment(req)
validate_segment_result(req, result)
print('OK', result.target_indices)
"
```

→ 실제 subprocess가 실행되어 `bidirectional` 조건화(상대 위치 0.25/0.5/0.75)로
정확히 블렌딩된 프레임을 반환함을 확인.

### 3) `scripts/evaluate_video.py` 실제 CLI 경로 end-to-end 스모크

`configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_mock.yaml`로 실제 ETRI 테스트 영상
(`01_person_walk.mp4`, 100프레임)을 `--no-models`로 돌렸다:

```bash
conda run -n ptest python scripts/evaluate_video.py \
    --config configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_mock.yaml \
    --input data/etri_video_eval/processed/01_person_walk.mp4 \
    --captions data/etri_video_eval/captions/01_person_walk.txt \
    --no-models --save-video
```

기본 `temporal.reuse_threshold`(0.2)에서는 이 영상의 프레임 간 캡션/객체
변화가 작아 전부 `reuse`로 처리되어 generate 분기가 트리거되지 않았다(이는
정상 동작 — 1A/1B는 generate가 항상 발생해야 한다고 요구하지 않는다). generate
분기 자체가 실제로 subprocess를 통해 프레임을 생성하는지 확인하기 위해
`temporal.reuse_threshold: 0.0` + `video_generator.generate_delta_min/max:
0.0/1.0`로 강제한 임시 설정으로 같은 영상을 재실행했다: **91개 inter-frame
전부 `generate` 분기로 라우팅되어 `external_segment_worker`(mock) subprocess가
segment(GOP)당 한 번씩(9개 segment) 호출됐고, `generated_frames/`에 91개
PNG가 실제로 저장됐다.** `segments.json`의 `generation` 필드도
`backend: "external_segment_worker:mock"`, `conditioning_mode:
"bidirectional"`, 프레임별 `relative_position`을 정확히 기록했다. (이 임시
설정과 산출물은 검증 후 삭제했다 — 저장소에 남아있지 않음.)

## 사용자가 실행할 실제 GPU 최종 검증 명령어

```bash
cd sgdjscc_lab

# (A) Wan I2V start-only (실제 GPU 검증 완료). 첫 실행은 ~90GB 다운로드
#     (로그인/라이선스 불필요, Apache-2.0) — 시간이 걸린다. VRAM이 부족하면
#     configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_start_only.yaml의
#     video_generator.worker.extra_json을
#     '{"offload_mode": "sequential"}' 그대로 두거나(느리지만 VRAM 절약)
#     height/width/num_inference_steps를 더 줄인다.
conda run -n ptest python scripts/evaluate_video.py \
    --config configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_start_only.yaml \
    --input data/etri_video_eval/processed/01_person_walk.mp4 \
    --captions data/etri_video_eval/captions/01_person_walk.txt \
    --snr 5 --device cuda:0 --max-frames 4

# (B) SVD (더 가벼움, image만 조건화, 실제 GPU 검증 완료 — Hugging Face 로그인 + 라이선스 동의 필요)
conda run -n semantic-diffusers huggingface-cli login
conda run -n ptest python scripts/evaluate_video.py \
    --config configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_svd.yaml \
    --input data/etri_video_eval/processed/01_person_walk.mp4 \
    --captions data/etri_video_eval/captions/01_person_walk.txt \
    --snr 5 --device cuda:0 --max-frames 4

# (C) Wan I2V bidirectional — 권장, 실제 GPU 검증 완료 (수정본).
#     end keyframe이 있는 segment는 Wan2.1-FLF2V-14B-720P를, 없는 마지막
#     segment는 Wan2.1-I2V-14B-480P를 worker가 자동으로 선택한다 (위 "Wan
#     bidirectional 수정 — 원인과 해결" 참조). 첫 실행은 두 체크포인트
#     합쳐 ~180GB 다운로드가 필요할 수 있다. --max-frames는 최소한 하나의
#     non-final GOP가 생기도록 keyframe.max_gop(기본 12)보다 커야
#     bidirectional 경로가 실제로 실행된다 — 14 이상을 권장.
conda run -n ptest python scripts/evaluate_video.py \
    --config configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_fixed.yaml \
    --input data/etri_video_eval/processed/01_person_walk.mp4 \
    --captions data/etri_video_eval/captions/01_person_walk.txt \
    --snr 5 --device cuda:0 --max-frames 14

# (D) Wan I2V bidirectional — *** EXPERIMENTAL, 원래 실패를 재현하는 기록용 ***
#     Wan2.1-I2V-14B-480P에 last_image를 억지로 넣으면 어떻게 실패하는지
#     보여주는 config다(pos_embed_seq_len 없는 체크포인트 — 위 "Wan
#     bidirectional 수정" 참조). 정상적인 실행에는 (C)를 쓴다.
conda run -n ptest python scripts/evaluate_video.py \
    --config configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_experimental.yaml \
    --input data/etri_video_eval/processed/01_person_walk.mp4 \
    --captions data/etri_video_eval/captions/01_person_walk.txt \
    --snr 5 --device cuda:0 --max-frames 4
```

`evaluate_video.py` 자체는 계속 `ptest`에서 실행한다 — `python_bin`을 통해
worker subprocess만 `semantic-diffusers`(또는 새로 만든 `lgvsc_gen`) 환경에서
실행되므로, `ptest`는 diffusers를 설치할 필요가 전혀 없다. `--max-frames 4`는
`scripts/evaluate_video.py`가 이미 지원하는 플래그로, 입력 영상을 앞 4프레임만
잘라 짧은 스모크 테스트로 만든다.

모든 config가 `temporal.reuse_threshold: 0.0` +
`video_generator.generate_delta_min/max: 0.0/1.0`을 이미 설정해 두었다 —
`--no-models`나 `--max-frames`로 캡션이 거의 안 바뀌는 짧은 구간을 잘라도
generate 분기가 실제로 트리거되도록(안 그러면 전부 reuse로 빠져 worker가
호출되지 않는다).

**Open-Sora, 또는 side_info까지 실제로 쓰는 통합**을 원하면 `wan`/`svd`
대신 `callable` backend를 쓴다:

```yaml
video_generator:
  worker:
    python_bin: "/home/<user>/anaconda3/envs/lgvsc_gen/bin/python"
    backend: callable
    backend_entrypoint: "your_module:generate_segment"
```

`scripts/lgvsc_example_callable_backend.py`가 정확한 함수 시그니처 템플릿이다
(그 자체는 mock으로 fallback하는 스모크 테스트용 예제일 뿐, 실제 Open-Sora
호출 코드가 아니다 — 직접 작성해야 한다).

## 예상 산출물 경로 / 로그 확인 방법

| 산출물 | 경로 |
|---|---|
| 복원 프레임/영상 | `outputs/etri_video_eval/manual_lgvsc_worker_{mock,svd,wan_start_only,wan_bidirectional_fixed,wan_bidirectional_experimental}/recon_frames/`, `recon.mp4` |
| 생성된(generate 분기) 프레임 | `.../generated_frames/generated_{index:05d}.png` |
| GOP/segment 구조 + generate 요약 | `.../segments.json`의 각 segment `generation` 필드 |
| 프레임별 로그(결정/모션/생성 조건화 모드) | `.../temporal_frames.csv` |
| worker 서브프로세스 작업 디렉터리(실패 시 보존) | `ExternalSegmentWorkerGenerator(work_dir=...)`로 지정한 경로 아래 `lgvsc_seg<NNNNN>_<random>/manifest.json`, `result.json` 또는 `error.json`, `run.log` — 기본값(`work_dir: null`)은 시스템 임시 디렉터리; `wan`/`svd` config는 `cleanup_on_success: false`라 성공해도 보존됨 |
| Wan 가중치 캐시 | `~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers/` (`semantic-diffusers` 환경 기준) |

**실패 시 확인 순서**: (1) `SegmentWorkerError` 메시지 자체에 work_dir 경로가
포함됨 → (2) 그 디렉터리의 `error.json`(구조화된 에러: `error_type`/`message`/
`traceback`) → (3) `run.log`(worker의 raw stdout/stderr, `error.json`도 못 남길
정도로 죽었을 때 특히 중요) → (4) `manifest.json`(실제로 무엇을 요청했는지
재현 확인용).

## 1C — 재현 준비는 완료, 실제 검증 결과는 아직 없다

**`SKIM+SFA`/`SKEM+DSA` 재현 검증(1C)의 "검증 준비"는 완료했지만, 실제 검증
실행/결과 보고는 이 저장소의 작업 범위가 아니라 사용자가 직접 한다** — 자세한
내용은 [1C 재현 문서](./2026-07_lgvsc_1c_reproduction.md)
참조. 1B가 완성한 것은 "Rx-legal segment 계약 뒤에 실제 모델을 붙일 수 있는
구조와 검증 경로"이고, 1C가 이번에 완성한 것은 그 backend들을 재사용한
"재현 baseline 4모드(`mock_baseline`/`svd_start_only`/`wan_skim_sfa`/
`wan_skem_dsa`) config + batch driver + summary 생성기"다 — 어느 쪽도
LGVSC 논문의 재현 **결과**는 아니다. 1C가 완료로 인정되려면 최소:

- 실제 segment-level world model(1B에서 검증된 backend)로 실제 GPU 실행
  결과가 나와야 하고(이 저장소는 dry-run/mock smoke만 직접 실행함),
- `SKIM`(고정 간격 keyframe)과 `SKEM`(의미 기반 keyframe, PSSS)의 실제
  구분은 **아직 구현돼 있지 않다** — 이번 재현 baseline 4모드는 keyframe
  선택을 전부 공유하고 decoder 조건화(단일 vs 두 keyframe)만 다르다는
  점을 재확인,
- 가변 길이 segment(DSA에 해당하는 능력) 지원 확인,
- seam/temporal 지표와 재현 수준(`faithful`/`paper-aligned`/`nearest
  reproducible`)을 산출물로 함께 보고

해야 완료로 본다([과거 구현 로그](../archive/etri_implementation_log.md)의 "1단계는 다시 세 개의 독립 완료
게이트로 관리한다" 참조).

## 관련 문서

- [etri_strategy.md](../current/status.md) — "후속 딥러닝 4단계", 1A/1B 구현 결과
- [video_extension_lgvsc.md](../architecture/tx_rx_contract.md) — 6.0-a/6.2/6.3, LGVSC
  재현선과 ETRI 개선선 구분
- [etri_owlv2_vqa_readiness.md](./2026-07-28_owlv2_vqa_calibration.md) — 이 문서와 같은
  형식의 이전 readiness 문서(5차 OWLv2/VQA 준비) — 실제 검증 결과까지 기록된
  완료 사례 참고용
