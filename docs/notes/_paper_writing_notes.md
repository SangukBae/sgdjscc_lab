# 논문 작성 보조 노트 (내부용, untracked)

> repo-facing 문서가 아니라 논문 작성용 내부 메모다. GitHub 문서 세트에서 분리하려고
> `phase4.md`에서 이관했다. tracked 문서로 올리려면 `git add` 필요.

## Phase 4 포지셔닝

Phase 4는 **새 전송 알고리즘이 아니라 신뢰성 평가·제어 프레임워크 + 영상 확장**이다.
"더 나은 JSCC 전송 기법"으로 주장하면 안 되고, 다음 세 가지를 기여로 둔다.

1. **신뢰성 평가의 세분화 (가장 강한 기여)** — SRS를 객체 누락/추가·관계·속성 오류로
   분해(`srs_base` vs `srs_packet`)해 '그럴듯하지만 틀린' 복원을 정량 진단. 저 SNR에서
   SRS는 비슷한데 객체 누락/할루시네이션이 급증하는 그림이 핵심 메시지.
2. **채널 적응 제어의 효과 (ablation)** — 고정 가이드 대비 SNR 적응 가이드 +
   실패유형별 regeneration이 같은 채널에서 SRS 개선.
3. **영상 시맨틱 전송 효율 (4-B)** — 키프레임 + 델타 전송이 프레임별 전송 대비
   오버헤드를 절감하면서 시간적 SRS 유지(`overhead_reduction`).

**Limitations에 선제 공개**: 패킷은 평가/제어 메타데이터(채널 코딩 아님), 객체/관계는
CLIP/캡션 휴리스틱, 단계적 디노이징은 prompt 레벨, 인터프레임은 키프레임 복사.

**실험 ablation** — ① baseline SGD-JSCC ② +적응형 가이드 ③ +패킷 검증기
④ 키프레임 전용 full packet ⑤ 키프레임+델타 재사용.

### 그대로 쓸 수 있는 문장
> "Rather than proposing a new transmission scheme, we build a *reliability-oriented
> evaluation and control layer* on top of an **unmodified** SGD-JSCC inference path,
> and extend it to keyframe-level video semantic transmission."
