> [← 문서 색인](./README.md)

# ETRI 1차 구현 검증 리포트 (Stage-1 Validation)

**결론: 1차 구현(개발 순서 0~4)은 완료됐다. 단, PTC/SFR/SDI와 객체 존재 판정은
CLIP/packet 기반 잠정 구현이므로, 최종 평가 주장은 5차 OWLv2/VQA 보강 후
재측정을 거쳐야 한다.**

이 문서는 [etri_strategy.md](./etri_strategy.md)의 1차 구현 묶음(순서 0~4)에 대한
검증 기록이다. 검증 일자: 2026-07-17 ~ 2026-07-18.

## 기준 커밋

| 커밋 | 내용 |
|---|---|
| `e2b1ffe` | feat(eval): presence threshold + uncertain band 배선 (순서 0) |
| `7f02fab` | feat(eval): PTC/SFR/SDI 잠정 시간축 지표 (순서 2) |
| `50ee42b` | feat(video): motion 이중 게이트 + GOP segment record (순서 3·4) |
| `54bb376` | feat(video): mp4 IO + 복원 출력 + stale frame/recon 정리 (순서 1) |
| `f000b3c` | docs(etri): 1차 상태 기록 + PPT 슬라이드 재번호 |
| `e9d648e` | fix(test): DDP dry-run 로그 기대 문구를 rank-0 전용 로깅에 맞춤 |

1차 구현 코드의 기준은 `f000b3c`(원격 검증 시점의 HEAD)이며, `e9d648e` 이후
전체 테스트 스위트가 green이다. 이 리포트와 `configs/composed_video_smoke.yaml`은
후속 docs 커밋으로 추가된다.

## 구현 범위 (순서 0~4)

| 순서 | 내용 | 상태 |
|---|---|---|
| 0 | Presence threshold / uncertain band(히스테리시스) 배선 — config → EvalContext → SRS → ObjectPreservation/Hallucination | 완료 (기본값에서 기존 결과와 bit-identical) |
| 1 | mp4 ↔ frame IO, 복원 frame folder + 복원 mp4 저장 | 완료 (cv2 → ffmpeg CLI 백엔드 자동 선택) |
| 2 | `PTC`/`SFR`/`SDI` 시간축 의미 지표 | 완료 — **CLIP/packet 기반 잠정 지표** (5차 재측정 필수) |
| 3 | semantic delta + motion 이중 게이트 | 완료 — 기본 OFF(`temporal.motion_threshold: null` = 기존 동작), 실데이터 threshold 튜닝은 후속 |
| 4 | GOP/segment 추상화 (`SegmentRecord`, `segments.json`) | 완료 — `generation` 필드는 3차 generate 분기용 예약(항상 null) |

제외(계획된 후속): OWLv2/VQA, Generate 분기, Adapter/Critic, Temporal SRS
Calibration, bit accounting.

## 주요 구현 파일

- `src/sgdjscc_lab/utils/video_io.py` — mp4↔frames (신규)
- `src/sgdjscc_lab/video/segment.py` — SegmentRecord / build_segments (신규)
- `src/sgdjscc_lab/video/temporal_pipeline.py` — 이중 게이트, decision 로그, segment 연동
- `src/sgdjscc_lab/evaluators/temporal_consistency.py` — ptc / sfr / sdi
- `src/sgdjscc_lab/evaluators/object_preservation.py`, `hallucination.py`,
  `semantic_reliability.py`, `pipelines/eval_pipeline.py`, `scripts/evaluate.py`
  — presence threshold/band 배선
- `scripts/evaluate_video.py` — mp4 입력, 복원 frame/mp4 저장, segments.json, stale 정리
- `configs/video/default.yaml`, `configs/eval/default.yaml`, `src/sgdjscc_lab/config.py`
  — 신규 config 키 + 경로 해석
- 테스트: `tests/test_video.py`, `tests/test_video_io.py`, `tests/test_evaluators.py`,
  `tests/test_ddp.py`(기대 문구 수정)

## 로컬 테스트 (ptest env, RTX 4080)

```bash
python -m pytest -q tests/test_video.py tests/test_video_io.py tests/test_evaluators.py tests/test_ddp.py
# → 86 passed
python -m pytest -q tests/
# → 518 passed  (e9d648e 이후 전체 green)
```

이전에 실패하던 `tests/test_ddp.py::test_entrypoint_torchrun_dryrun`은 1차 구현과
무관한 기존 실패였다: `95d3c40`(non-rank0 콘솔 로그 억제)에서 train.py의 DDP
로그가 rank-0 전용 `"DDP: world_size=…"` 한 줄로 바뀌었는데, 테스트는 옛 per-rank
`"DDP: rank="` 문구를 기대하고 있었다. train.py의 로깅이 의도된 동작이므로 테스트
기대 문구를 수정했다(`e9d648e`).

## 원격 컨테이너 테스트 (155.230.15.67, `sgdjscc` 컨테이너, `f000b3c`)

```bash
# host: git pull --ff-only → f000b3c, working tree clean
docker exec -w .../sgdjscc_lab sgdjscc bash -lc \
  "python -m pytest -q tests/test_video.py tests/test_video_io.py tests/test_evaluators.py"
# → 82 passed
```

dry-run(모델 미로딩, `--no-models`) 검증 — 8프레임 mp4 실행 후 같은 stem의
3프레임 mp4로 재실행:

```bash
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /tmp/vdemo/a/clip.mp4 --no-models --save-video   # 8프레임
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /tmp/vdemo/b/clip.mp4 --no-models --save-video   # 3프레임, 같은 stem
```

결과: 재실행 로그에 `Removed 8 stale frame_*.png` / `Removed 8 stale
recon_*.png`가 남고, 추출 폴더와 `recon_frames/`에 **정확히 3개씩만** 남음
(**stale cleanup 재실행 검증 통과**). `temporal_metrics.csv`에 ptc/sfr/sdi,
`temporal_frames.csv`에 decision/motion_score, `segments.json`·`recon.mp4` 생성
확인. 컨테이너에는 cv2가 없어 ffmpeg CLI 백엔드로 동작.

## 실제 모델 경로 샘플 실행 (로컬 RTX 4080, 2026-07-18)

체크포인트 4종(JSCC/MDTv2/ControlNet/MuGE) + BLIP2(blip2-opt-2.7b-coco) +
CLIP(ViT-B/32·ViT-L/14) 실로딩 경로. 샘플: `inputs/test_1.png`에서 만든 6프레임
256×256 mp4(프레임 0~3 = 카메라 팬 크롭, 4~5 = 장면 전환), 4 fps.

```bash
python scripts/evaluate_video.py --config configs/composed_video_smoke.yaml \
    --input outputs/smoke_video/sample.mp4 --snr 5 --save-video
```

`composed_video_smoke.yaml`은 `composed_video.yaml`과 동일하되 검증 속도를 위해
`diffusion_step: 10`(평가용은 50)이고 motion gate가 켜져 있다(`motion_threshold:
0.08` — smoke 값, 튜닝 값 아님). **정상 완료** (모델 로딩 포함 수 분).

산출물(전부 생성 확인):

| 산출물 | 확인 결과 |
|---|---|
| `temporal_metrics.csv` | 생성, `ptc`/`sfr`/`sdi` 컬럼 존재 (ptc≈0.543, sfr≈0.596, sdi≈−0.007 — 6프레임 smoke 값, 아래 주의 참조) |
| `temporal_frames.csv` | `decision`/`motion_score` 컬럼 존재, 프레임별 값 기록 (keyframe 2, recompute_semantic 4) |
| `segments.json` | 세그먼트 2개, frame_decisions/delta/motion/temporal 요약, `generation: null` |
| `recon_frames/` | 정확히 6개 `recon_*.png` (현재 실행 프레임 수와 일치) |
| `recon.mp4` | 생성 (6프레임 @ 4 fps, 원본 fps 유지) |

주의(정직한 해석): 이 smoke 클립에서는 BLIP2 캡션이 크롭마다 달라져 semantic
delta가 항상 reuse 임계(0.2)를 넘었고, 따라서 4개 inter-frame 모두
`recompute_semantic`으로 갔다(모션 게이트 발동 조건인 "의미 동일 + 모션 큼"
구간이 없었음). motion-트리거 recompute 경로 자체는 오프라인 단위 테스트
(`tests/test_video.py::TestMotionGate`)와 dry-run에서 검증됐다. 또한 위 지표
수치는 6프레임 합성 클립의 잠정(CLIP 기반) 값일 뿐 평가 결과가 아니다.

## 남은 후속 항목

| 항목 | 단계 |
|---|---|
| OWLv2/VQA presence 보강 + `PTC`/`SFR`/`SDI` 재측정 (held-out 분리 포함) | 5차 |
| `motion_threshold`/`motion_weight` 실데이터(실모션 비디오) 튜닝 및 baseline 분별력 검증 | 4~5차 |
| Generate 분기 (start-only → bidirectional; `SegmentRecord.generation`에 부착) | 3~4차 |
| Packet Verifier + 오류 유형별 regeneration controller | 2차 |
| channel-symbol 절감 PoC / bit accounting | 6차 |
