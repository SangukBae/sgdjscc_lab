# results/ — 추적 재현성 결과

> [← 문서 색인](../docs/README.md) · 절차 문서: [docs/protocols/results_registry.md](../docs/protocols/results_registry.md)

- 목적
  - `outputs/`는 git 비추적 대용량 실행 결과(로그·프레임·packet·mp4)
  - `results/`는 git 추적, 핵심 CSV·JSON·README만 보존
  - 원본 `outputs/` 경로는 각 run의 `manifest.json`에 기록, 파일 자체는 이동하지 않음

## 구조

```text
results/
  README.md          이 문서
  registry.csv        run 색인 — run_id/category/date/status/경로/commit/headline
  <run_id>/
    README.md         outputs/<run_id>/README.md 사본 — provenance/해석 문구만 보정 가능
                       (수치는 절대 변경 안 함; manifest.json의 artifacts."README.md"
                       .matches가 원본과 다르면 그 이유가 여기 명시됨)
    manifest.json      재현성 manifest (아래 참고)
    summary.json       run 요약 (있는 경우)
    *.csv              핵심 지표·accounting CSV (프레임 이미지·바이너리 packet 제외)
    config_source/     실행 당시 root/pre-merge config 원본 사본 — **resolved config 아님**
                        (실제 merge·CLI override 결과를 확실히 복원할 수 있을 때만
                        manifest.json의 resolved_config.status가 "resolved"가 됨)
```

## registry.csv 컬럼

- `run_id`: `results/<run_id>/`와 1:1
- `category`: `transmission` / `video` / `image` 등
- `run_date`, `status`: `frozen`(고정) / `active`(진행 중)
- `results_path`, `original_output_root`: 추적 사본 vs 원본 대용량 경로
- `git_commit`, `git_dirty`: 실행 시점 값 — 미확인 시 `unknown`
- `doc_link`: `docs/experiments/`의 해설 문서
- `headline_result`: 한 줄 핵심 수치

## manifest.json 스키마

- 생성: `sgdjscc_lab.utils.run_manifest.build_run_manifest()` +
  `write_run_manifest()` ([src/sgdjscc_lab/utils/run_manifest.py](../src/sgdjscc_lab/utils/run_manifest.py))
- 필드
  - `git`: commit / dirty / branch — `include_git=False`로 명시 생략 가능(과거 run 등)
  - `command`: `{text, argv, source}` — `source`는 `captured`(실행 시점 실측) /
    `reconstructed`(문서·로그에서 사후 재구성, 정확한 명령이라 단정 금지) /
    `unknown` 중 하나. 실측 run은 argv 배열을 보존하고 `text`는 shell-safe하게 생성
  - `seed`: 실제 값, 또는 `"unknown"`(미확인) / `"not_set"`(seed 없음이 확인됨) —
    **`null` 사용 금지**(두 의미가 구분 안 됨). `None` 전달 시 `ValueError`
  - `resolved_config`: `{status, resolved, config_source}`
    - `status: "resolved"` — 실행 당시 최종(merge+override 반영) config를
      확실히 복원 가능할 때만. 경로 입력은 파일·sha256 검증 실패 시 즉시 중단
    - `status: "config_source_only"` — merge 전 root config만 있고 최종본은
      복원 불가할 때. `resolved`는 `"unknown"`, `config_source`에 경로+sha256+사유
    - `status: "unknown"` — 아무 config도 없을 때
  - `dataset`: ref·sha256
  - `checkpoints`: `{status, items}`
    - `recorded`: 파일별 경로·sha256 확인
    - `unknown`: 사용 checkpoint 미확인
    - `not_set`: checkpoint를 사용하지 않았음이 확인됨
  - `environment`: python_version / platform / torch_version / cuda_version / gpu_name
  - `evaluator_versions`
  - `original_artifact_paths`: `outputs/` 내 원본 대용량 경로
  - `artifacts`: `results/`로 복사한 파일별 `{copied_path, copied_sha256,
    original_path, original_sha256, matches}` — `matches`는 두 해시를 모두
    구할 수 있을 때만 `true`/`false`, 아니면 `"unknown"`(추측 금지).
    `matches: false`는 오류가 아니라 "의도적으로 사후 보정된 사본"일 수 있음
    (README.md처럼) — 이유는 해당 run README나 `extra`에 명시
  - `accounting.exact_fields` / `accounting.proxy_fields`: 실측 vs 추정 구분
  - `nan_or_failure_counts`
- 원칙: 확인되지 않은 값은 절대 추측하지 않고 문자열 `"unknown"`으로 기록
- 과거 run(예: `transmission_20260818`)은 실행 당시 git/환경 정보가 기록되지 않아
  `unknown`이며, `manifest.json`의 `extra.*_note` 필드에 그 이유를 남김
- 앞으로 실행하는 스크립트에서 `build_run_manifest()`를 직접 호출하면
  commit/dirty/python/torch/cuda/gpu는 자동 수집됨(`include_git`/
  `include_environment`를 `False`로 주지 않는 한)

## 현재 등록된 run

| run_id | 분류 | 상태 | 설명 문서 |
|---|---|---|---|
| [transmission_20260818](./transmission_20260818/) | transmission | frozen | [docs/experiments/2026-08-18_transmission_reduction.md](../docs/experiments/2026-08-18_transmission_reduction.md) |
| [transmission_normalization_20260826](./transmission_normalization_20260826/) | transmission | frozen | [docs/experiments/2026-08-26_transmission_normalization.md](../docs/experiments/2026-08-26_transmission_normalization.md) |
| [float32_digital_normalization_full_20260827](./float32_digital_normalization_full_20260827/) | diagnostics | frozen | [docs/experiments/2026-08-28_float32_digital_step_normalization_full.md](../docs/experiments/2026-08-28_float32_digital_step_normalization_full.md) |
| [quantization_reevaluation_10db_20260828](./quantization_reevaluation_10db_20260828/) | transmission | frozen | [docs/experiments/2026-08-28_quantization_reevaluation_10db.md](../docs/experiments/2026-08-28_quantization_reevaluation_10db.md) |
| [fixed_skem_matched_rate_10db_20260828](./fixed_skem_matched_rate_10db_20260828/) | transmission | frozen | [docs/experiments/2026-08-28_fixed_skem_matched_rate_10db.md](../docs/experiments/2026-08-28_fixed_skem_matched_rate_10db.md) |
| [edge_uncertainty_ablation_10db_20260828](./edge_uncertainty_ablation_10db_20260828/) | transmission | frozen | [docs/experiments/2026-08-28_edge_uncertainty_ablation_10db.md](../docs/experiments/2026-08-28_edge_uncertainty_ablation_10db.md) |
