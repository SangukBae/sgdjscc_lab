# Parallel transmission normalization — transmission_normalization_parallel_20260827_225722

- 상태
  - `completed`
  - GPU worker: 3개 (`cuda:0, cuda:1, cuda:2`)
  - 완료 pair: 60개
  - 실패 pair: 0개
- 안전성
  - worker별 독립 디렉터리: `workers/worker_NN/`
  - 공용 CSV 동시 쓰기 없음
  - 모든 worker 종료 후 상위 CSV·Pareto·effect·manifest 생성
  - 재개 시 `parallel_plan.json`과 다른 commit/device/영상 배분/설정은 거부
- 대용량 산출물
  - packet과 복원 영상은 복사하지 않고 각 worker 디렉터리에 보존
  - 상위 디렉터리는 병합된 표와 재현성 metadata만 제공
