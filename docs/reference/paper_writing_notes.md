---
status: draft
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes:
---

> [← 문서 색인](../README.md)

# 논문 작성 보조 노트 (내부용)

- 문서 성격
  - 논문 작성용 내부 draft
  - 최신 구현 상태의 근거 아님
  - 현재 상태 기준: [status.md](../current/status.md)

## Phase 4 포지셔닝

- 주장 범위
  - 신뢰성 평가·제어 framework
  - 영상 확장
  - 새 JSCC 전송 알고리즘으로 주장하지 않음

1. **신뢰성 평가 세분화**
   - 분해: 객체 누락·추가, 관계·속성 오류
   - 비교: `srs_base` vs `srs_packet`
   - 핵심 그림: 저 SNR의 객체 누락·할루시네이션 증가
2. **채널 적응 제어 효과**
   - 비교: 고정 guidance vs SNR 적응 guidance
   - 추가: 실패 유형별 regeneration
3. **영상 시맨틱 전송 효율**
   - 방식: keyframe + delta
   - 목표: overhead 절감 + temporal SRS 유지

- 선제 공개 한계
  - packet: 평가·제어 metadata
  - 객체·관계: CLIP·caption heuristic
  - 단계적 denoising: prompt 수준
  - inter-frame: keyframe 복사

- **실험 ablation**
  - ① baseline SGD-JSCC ② +적응형 가이드 ③ +패킷 검증기 ④ 키프레임 전용 full packet ⑤ 키프레임+델타 재사용.

### 그대로 쓸 수 있는 문장
> "Rather than proposing a new transmission scheme, we build a *reliability-oriented
> evaluation and control layer* on top of an **unmodified** SGD-JSCC inference path,
> and extend it to keyframe-level video semantic transmission."
