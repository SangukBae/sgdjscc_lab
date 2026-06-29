# 개발계획서 (업데이트판)
## 생성 AI 기반 시맨틱 미디어 전송 신뢰성 평가·고도화 프레임워크

> 본 문서는 초기 개발계획서를 기반으로, 현재 개발 프레임워크의 문서 및 구현 상태를
> 반영해 최신화한 개발계획서다. 단순 요약이 아니라, "초기 계획 대비 무엇이 확장되었고
> 어떤 연구개발 방향으로 정교화되었는지"가 드러나도록 작성하였다.
>
> 작성 원칙: ① **논문 정합(paper-faithful)** 영역과 **ETRI 확장** 영역을 구분한다.
> ② **구현 완료 / 부분 구현·스캐폴드 / 미구현·검증대기**를 구분한다.
> ③ 계획이 아닌 "현재 구현 상태"를 명시한다.

---

## 용어 정리 (먼저 읽어주세요)

본문에는 통신·생성형 AI 용어가 함께 등장한다. 비전공 검토자를 위해 핵심 용어를 한 줄로
정리한다. (이미 익숙하면 건너뛰어도 된다.)

### 충실도(구현 수준) 4단 분류

본 문서는 각 기능이 "논문/원본에 얼마나 충실한지"를 아래 4단계로 구분해 표기한다.

| 분류 | 의미 |
|---|---|
| **논문 정합 (paper-faithful)** | 원본/논문 알고리즘을 **수치까지 동일하게** 재사용·재현한 부분 |
| **paper-like** | 구조·목적은 논문과 같으나 정확한 수치(데이터·하이퍼파라미터·손실 가중)는 **보장하지 않는 근사** |
| **scaffold (골격)** | 실행·테스트 가능한 **뼈대만** 갖추고, 학습된 모델·정밀 수치는 아직 미완인 단계 |
| **ETRI 확장** | 논문에는 없으나 ETRI 과제 목표를 위해 **새로 추가한** 요소 |

> 상태 아이콘(8.3 표): ✅ 구현·연결됨 / 🟡 부분·스캐폴드 / 🔲 미구현·검증대기.
> 본문의 "세션 외"는 "현재 개발 환경 밖(네트워크·사전학습 가중치가 있어야 수치 산출
> 가능)"을 뜻한다.

### 핵심 용어

| 용어 | 한 줄 설명 |
|---|---|
| SGD-JSCC | 본 과제 기반 모델. 텍스트·엣지 같은 의미 가이드로 확산 모델을 "채널 잡음 복원기"로 쓰는 전송 기법 |
| JSCC | 소스 압축과 채널 부호화를 신경망으로 한꺼번에 수행하는 전송 방식 |
| AWGN / SNR | AWGN=가장 기본적인 무선 잡음 채널, SNR=신호 대 잡음비(채널 품질, dB) |
| SNR sweep | 여러 SNR 값에서 반복 평가해 성능 곡선을 얻는 실험 |
| Rayleigh / fast-fading / packet-drop | 신호 세기가 변동하는 페이딩 채널, 빠르게 변하는 페이딩, 전송 단위(패킷) 손실 채널 |
| CSI | 채널 상태 정보. 수신기가 채널 이득·위상을 아는 정도(모르면 "블라인드") |
| 확산 모델(Diffusion) / DiT·MDTv2 | 잡음을 단계적으로 제거해 이미지를 복원하는 생성 모델(DiT·MDTv2는 그 구조 변형) |
| ControlNet | 엣지 등 구조 정보를 확산 모델에 조건으로 주입하는 보조 분기 |
| VAE / latent | 이미지를 압축한 잠재표현(latent)으로 바꾸고 다시 복원하는 오토인코더 |
| step matching | 채널 잡음 수준을 확산 복원의 시작 시점에 대응시키는 연산 |
| water-filling | 원소별 잡음 정도에 맞춰 디노이징을 배분하는 페이딩 대응 기법 |
| frozen denoiser | 재학습하지 않고 가중치를 고정해 둔 확산 복원망 |
| CLIP / BLIP2 | CLIP=이미지·텍스트 의미 유사도 측정 모델, BLIP2=이미지를 캡션으로 설명하는 모델 |
| PSNR / SSIM / LPIPS | 각각 픽셀 오차·구조 유사도·지각(신경망 기반) 유사도 품질 지표 |
| FID | 생성 이미지의 분포 품질 지표(낮을수록 좋음) |
| 할루시네이션(hallucination) | 원본에 없던 내용을 그럴듯하게 지어내는 생성 오류 |
| semantic packet | 이미지를 객체·관계·속성·장면으로 분해한 "의미 명세서" |
| packet-aware verifier | 원본·복원의 의미 명세서를 비교해 누락/추가 객체 등을 개수로 집계하는 검증기 |
| VQA | 이미지에 질문해 답을 얻는 방식의 평가(할루시네이션 점검에 사용) |
| SRS / srs_packet / srs_v2 | 의미 보존도 종합 지표(SRS) → 의미 단위 분해(srs_packet) → 시간축·VQA까지 통합(srs_v2) |
| regeneration | 채널 재전송 없이 수신단에서 복원을 다시 시도해 신뢰도를 회복하는 것 |
| adaptive guidance | 채널 상태(SNR)에 맞춰 가이드 강도·복원 스텝을 조절하는 제어 |
| DDIM step-budget / early-exit | 확산 복원 스텝 수를 줄이거나 충분히 좋아지면 도중에 멈춰 지연을 단축하는 기법 |
| scaffold / placeholder | 골격만 갖춘 단계 / 아직 학습·완성 전의 임시 자리표시 |

---

## 0. 초기 계획 대비 주요 변경점 요약

초기 계획서는 "SGD-JSCC를 채택하여 AWGN 환경에서 semantic fidelity와 hallucination을
평가하는 프로토타입을 만든다"는 단일 이미지 중심의 구상이었다. 현재는 이 구상을
유지하면서 다음 방향으로 확장·정교화되었다.

- **평가 체계의 세분화**: 단일 SRS(전체 유사도 복합 지표)에서 출발해, 의미 단위 분해
  지표(srs_packet)와 VQA·시간축까지 포괄하는 srs_v2로 3계층화되었다.
- **단일 이미지 → 영상/시간축 확장**: 키프레임·시맨틱 델타·시간적 일관성 지표를 다루는
  비디오 파이프라인이 추가되었다.
- **AWGN → 다중 채널 + 채널 조건화**: Rayleigh / fast-fading / packet-drop 채널과,
  수신 채널 evidence를 복원에 활용하는 채널 조건화 추론 경로가 추가되었다.
- **고정 복원 → 적응·재생성·저지연 복원**: SNR 적응형 가이드, 실패유형 인식
  regeneration, 다중 전략 regeneration search, DDIM step-budget·early-exit 기반
  저지연 복원이 추가되었다.
- **추론 전용 → 학습 재현 scaffold 추가**: 논문 3-stage 학습 절차(JSCC → text-DM →
  ControlNet)와 보조 stage(edge codec, CSI 추정)를 구조적으로 재현하는 학습 경로가
  추가되었다. 단, 이는 추론 경로와 달리 **부분 충실(구조적 근사)** 수준이다.
- **데이터셋 현실화**: 초기의 "COCO·Kodak·ADE20K·CelebA-HQ + mIoU 중심" 구상은,
  현재 실제 구비된 데이터(ImageNet / COCO / JourneyDB / CelebA)와 학습·평가 구조에
  맞게 학습축·평가축으로 재정리되었다. mIoU 중심 표현은 현재 구현된
  segmentation-consistency 기반 평가 구조로 보정한다(아래 6장 참조).
- **마스터 스위치 도입**: 모든 확장 기능은 기본값 off이며 상위 게이트로 일괄 제어된다.
  끄면 원본 SGD-JSCC 추론 경로와 수치적으로 동일하게 동작하여, 베이스라인 재현성을
  보장한다.

이 확장의 핵심은 **원본 `SGDJSCC/`의 논문 정합 추론 코어는 그대로 보존**하면서,
원본이 비워둔 **평가·제어·채널·영상·학습 골격을 메웠다**는 데 있다. 원본 대비 무엇이
공백이었고 그것을 어떻게·왜 메웠는지는 [8장](#8-현재-구현-진행-현황-및-원본-sgd-jscc-대비-보완-신규-구현-상태-기준)에서
기능군별로 상세히 다룬다. 상세 변경표는 본문 마지막
[12장](#12-초기-계획-대비-변경확장-포인트-요약표)에 정리한다.

---

## 1. 개발 목표

SGD-JSCC의 단순 재현이 아니라, ETRI 과제용 **시맨틱 미디어 전송 신뢰성 향상·평가
플랫폼**으로의 고도화를 목표로 한다.

- SGD-JSCC 기반 End-to-End 무선 이미지 전송 프로토타입 구축 및 **베이스라인 재현성 고정**
- 노이즈 채널 통과 후 **송신 의도(semantic intent)의 보존도**와 **할루시네이션 발생**을
  동시에 정량화하는 평가 체계 설계
- text + edge guidance를 넘어 depth / segmentation / 시맨틱 패킷 등으로 확장 가능한
  guidance 구조 확보
- AWGN 단일 이미지에서 출발해, 다중 채널·영상(시간축)·채널 적응·저지연 복원까지
  단계적으로 확장 가능한 모듈형 연구 프레임워크 구축

> 핵심 관점: 본 과제의 성공 기준은 픽셀 완벽 복원(PSNR 최대화)이 아니라 **"노이즈
> 채널을 통과한 뒤에도 원본의 의미가 얼마나 신뢰성 있게 전달되는가"** 의 정량화다.

---

## 2. 기본 모델 선정 방침

- **백본: SGD-JSCC로 고정.**
- 선정 근거:
  - 송신단에서 **semantic guidance(텍스트·엣지)를 명시적으로 추출·전송**하는 구조 →
    "의도 일치도"와 "할루시네이션 완화"라는 ETRI 핵심 목표에 직접 부합한다.
  - 확산 모델을 후처리 생성기가 아니라 **채널 오염 latent의 복원기(denoiser)** 로
    활용한다 → 채널 적응을 확산 모델에 위임하고 JSCC 코어는 고정할 수 있어, 채널·
    가이드·평가를 모듈로 분리·확장하기에 유리하다.
- 비교 모델과의 차이:
  - DiffJSCC는 Kodak·ADE20K·CelebA-HQ 중심의 realism / downstream segmentation을
    강조하는 후처리형 접근이다. 과제 우선순위가 "의미 왜곡 억제"에 있으므로,
    semantic-guided denoising 구조의 SGD-JSCC가 더 적합하다.
- **알고리즘 보존 불변식**: 추론 순전파의 수치(VAE 스케일링 팩터, AWGN 잡음 공식,
  blind SNR 추정, step matching, canny 재전송, 최종 decode)는 변경하지 않는다.
  모든 신규 기능은 이 경로를 감싸는 opt-in 확장으로 구현한다.

---

## 3. 논문 한계점 및 해결 전략 *(신규)*

본 장은 SGD-JSCC 논문 및 관련 비교 논문(DiffJSCC, CDDM 계열 등) 관점에서 현재 문제로
간주되는 항목을 정리하고, 각 문제를 **이 과제에서 구현·개발할 알고리즘/딥러닝 모듈
블록 단위**로 어떻게 해결하는지를 1:1 또는 1:N으로 매핑한다.

### 3.1 한계점 ↔ 해결 모듈 매핑

| # | 논문/비교군 관점의 한계 | 연구적 문제 | 해결 모듈·알고리즘 블록 (이 과제) | 상태 |
|---|---|---|---|---|
| L1 | 생성형 복원의 randomness로 인해 **그럴듯하지만 틀린(hallucinated)** 복원 발생, 이를 측정할 지표 부재 | 의미 붕괴·할루시네이션의 정량화 | **semantic packet** + **packet-aware verifier**(객체/관계/속성/장면 매칭), **VQA hallucination evaluator**, **SRS / srs_packet / srs_v2** | 구현 |
| L2 | 고-SNR에서도 가이드가 충실 복원을 방해하는 등 **가이드 강도가 채널 상태에 비적응** | 채널 적응형 복원 제어 | **adaptive guidance controller**(SNR 구간별 guidance/step 스케일), **regeneration policy / regeneration search**(실패유형별 재시도) | 구현 |
| L3 | 텍스트 캡션이 **완벽 전송 가정**, 손상 시 디코더 오도. 논문도 -15dB 텍스트 전송 곤란을 미해결로 명시 | 부가정보 손상 견고성 | **packet-drop 채널**, **edge codec**(BCE+Dice 학습형 엣지 전송 링크) (구현). **가이드 전용 손상 모델**(엣지 dropout/blur/erasing, seg 영역제거, 캡션 token dropout)은 **설계 방향**으로, 현재는 segmentation의 region-dropout, 학습 단계의 CFG label-dropout 등 일부 요소만 존재 | 부분/설계 방향 |
| L4 | 엣지맵 가이드가 **추가 전송 오버헤드**를 유발, 영상에서는 프레임마다 전 정보 전송 시 낭비 | 부가정보 경량화·중복 제거 | **semantic delta** 전송 + **temporal keyframe pipeline**(키프레임 full + 인터프레임 델타 재사용), **importance estimator**(중요도 기반 전송 순서) | 구현(메타데이터 수준) |
| L5 | 확산 복원이 **50-step·고지연**(논문 기준 약 1.5s/이미지), distillation은 미구현 | 저지연 복원 | **DDIM step-budget**(50/20/10/5), **dynamic routing**(SNR·신뢰도→step 수), **early-exit**(샘플러 내부 인터럽트), **consistency decoder 인터페이스**, **latency profiler** | 구현(증류 student는 placeholder) |
| L6 | fading 대응 water-filling이 **강한 CSI 가정**, blind/imperfect CSI 견고성 부족 | 채널 적응·블라인드 복원 | **Rayleigh / fast-fading 채널** + **MMSE 등화**, **fast-fading water-filling denoising(Alg.4)**, **channel-conditioned diffusion**(measurement bundle → channel condition encoder → latent/joint/blind 모드), **CSI estimation**(SNR 추정망 학습·연결) | 부분/스캐폴드 |
| L7 | 평가가 **PSNR/perceptual 중심**, 시맨틱 신뢰성 지표 부재 | 시맨틱 우선 평가 철학 | **evaluator suite**(quality / CLIP / object preservation / hallucination) + **SRS 통합** + **FID**(논문 §VI 정합, fail-fast 옵션) | 구현 |
| L8 | 평가 범위가 **128×128 정지 이미지**에 국한, 시간축·패킷손상 시나리오 부재 | 평가 범위 확장 | **temporal consistency evaluator**(temporal_srs, srs_flicker, object_identity_consistency, temporal_segmentation_iou, temporal_hallucination_rate, overhead_reduction) | 구현 |
| L9 | MIMO / OFDM / 다중사용자 미포함(논문 결론 향후과제) | 시스템 확장 | 핵심 신뢰도·지연·채널 트랙 안정화 후 **후속 확장 트랙**으로 분리 | 계획 |

### 3.2 해결 전략의 우선순위 원칙

해결 순서는 "측정 체계와 베이스라인을 먼저 고정한 뒤, 그 위에 연구적 개선을 얹는다"는
원칙을 따른다.

- **A(할루시네이션·의미 불일치)** → **B(부가정보 견고·경량화)** → **C(저지연)** →
  **D(블라인드/페이딩 채널)** 순서.
- A가 먼저인 이유: 할루시네이션·의미 불일치가 ETRI 신뢰도 목표에 가장 직접적으로
  반하는 실패 양상이기 때문.
- C(저지연)가 신뢰도 작업 뒤인 이유: 지연 단축은 런타임이 아니라 **검증된 시맨틱
  품질**을 기준으로 판단해야 하기 때문.

---

## 4. 송수신단 아키텍처 고도화 방향 *(신규)*

각 연구개발 요소가 송신단(Tx) 또는 수신단(Rx)에서 **어떤 구조적 변화(블록 추가/변경)**
로 연결되는지, 그리고 전송 기술 개발 측면에서 어떤 의미를 갖는지를 정리한다.

### 4.1 송신단(Tx) 고도화

- **semantic guidance 추출 블록 확장**
  - 기존: BLIP2 텍스트 캡션 + MuGE soft edge.
  - 확장: depth(DPT), segmentation(SegFormer) 추출기를 동일한 추출 인터페이스로
    추가 → guidance의 종류를 채널/콘텐츠 상황에 맞게 선택 가능.
  - 역할 차이: **text**=거친(coarse) 의미 조건, **edge**=구조적 윤곽(미세 조건),
    **segmentation**=영역/클래스 단위 조건, **depth**=기하 구조 조건. 서로 보완적이며
    채널 예산·콘텐츠 특성에 따라 조합한다.
- **semantic packet / side-information 계층 추가**
  - 캡션·객체·장면·관계·속성·엣지/세그/depth 요약을 통합한 "의미 명세서"(packet)를
    Tx에서 구성한다. 현재는 **평가·제어용 메타데이터**로 직렬화되며, 실제 패킷 채널
    코딩은 향후 과제다.
  - 전송 기술 의미: 어떤 의미 요소를 우선 전송할지(중요도 추정), 무엇을 재전송/재사용할지를
    결정하는 **부가정보 전송 정책**의 기반이 된다.
- **semantic delta(시간축 중복 제거)**
  - 영상에서 키프레임은 전체 패킷을 전송하고, 인터프레임은 직전 키프레임 대비
    변경분(신규/제거 객체, 변경된 관계·속성·장면)만 전송/재사용한다.
  - 전송 기술 의미: 프레임마다 전 정보를 보내는 대비 **시맨틱 오버헤드 절감**
    (절감률 지표로 정량 리포팅).
- **edge codec(전용 엣지 전송 링크)**
  - 엣지 가이드를 이미지 VAE에 섞지 않고, 전용 인코더→채널→정렬(projector) 링크로
    전송하고 latent를 정렬한다. codec은 BCE+Dice로 학습한다(논문의 "엣지 자체 링크
    전송 후 latent 정렬" 설계 의도와 정렬).
  - 전송 기술 의미: 부가정보를 **독립 채널 자원**으로 다루는 구조 → 가이드 손상·예산을
    JSCC latent와 분리해 분석 가능.

### 4.2 수신단(Rx) 고도화

- **채널 측정값·CSI·reliability를 조건으로 활용하는 구조**
  - 수신단에서 received/equalized latent, channel gain, noise variance, mask,
    SNR 추정, reliability map을 **measurement bundle**로 추상화한다(채널 관측을
    operator 형태로 노출하는 패턴).
  - 이를 **channel condition encoder**가 조건 특징(토큰)으로 압축하고, reliability
    head가 신뢰도를 산출한다.
  - 전송 기술 의미: 수신 신호 자체를 복원 조건으로 사용하는 **채널 적응형 복원**의
    기반. 단, 현재 조건 토큰은 frozen denoiser가 직접 소비하지 않으며(재학습 필요),
    조건화는 received-latent 초기화 + reliability 기반 guidance/step 스케일로 작동한다.
- **channel-conditioned reconstruction (latent / joint / blind 모드)**
  - 채널 상태 정보 가용성에 따라 표준(latent), 고충실도 joint, 블라인드 복원 모드를
    전환한다. one-pass로 패치별 측정→이미지 레벨 집계→동일 received latent 재사용
    디코딩을 수행한다.
  - 전송 기술 의미: AWGN 외 Rayleigh/fast-fading/packet-drop, 미지/불완전 CSI 환경에서
    **채널 적응형 복원** 경로를 제공.
- **regeneration loop / regeneration search (재전송 대체로서의 재생성)**
  - 복원 결과의 신뢰도(SRS/SRS-v2)가 낮으면, 채널 재전송 없이 **수신단에서 가이드
    조합을 바꿔 재복원**한다. 단발 재시도(loop)에서 다중 전략 search(강/약 텍스트,
    무조건, 채널 조건화 재시도)로 확장되며, 검증된 점수가 가장 높은 출력을 선택한다.
  - 전송 기술 의미: 채널 재전송 비용 없이 신뢰도를 높이는 **재생성 기반 신뢰도 회복
    정책**.
- **temporal keyframe reuse (시간축 재사용)**
  - 인터프레임은 최신 키프레임의 패킷·복원을 재사용하고 델타만 반영한다.
  - 전송 기술 의미: 영상 전송에서 **중복 제거 + 시간적 일관성 유지**.
- **저지연 복원 경로(low-latency sampling)**
  - DDIM step-budget, dynamic routing, early-exit로 디노이징 스텝 수를 채널·신뢰도에
    맞춰 줄인다. 향후 consistency distillation student를 끼울 인터페이스를 둔다.
  - 전송 기술 의미: 품질-지연 트레이드오프를 명시적으로 제어하는 **실시간성 확보 경로**.

---

## 5. 사용 데이터셋 *(현실화)*

초기 계획의 "COCO2017 + Kodak + ADE20K + CelebA-HQ" 구상을, 현재 실제 구비된
데이터와 학습/평가 구조에 맞게 **학습 축 / 평가 축**으로 재정리한다.

### 5.1 학습 축 (코드로 구동 가능한 데이터)

| 데이터셋 | 입력 형태 | 용도 | 비고 |
|---|---|---|---|
| ImageNet | image-only | Stage 1 JSCC / CSI 추정 / edge codec | 캡션 불필요 |
| COCO2017 | text-image (pair) | Stage 2 text-DM / Stage 3 ControlNet | 단일 캡션 또는 이미지당 5캡션(멀티캡션) |
| JourneyDB subset | text-image (pair) | Stage 2/3 보조 학습 | 추출된 text-image pair |
| CelebA (in-the-wild) | image-only | 도메인 특화 보조 | text stage 사용 시 캡션 자동 생성 필요(자동 캡션은 paper-like) |

- 엣지맵은 별도 파일 없이 **on-the-fly Canny**로 계산한다.
- 대규모 운영을 위해 폴더 스캔 외에 **경로 리스트 입력**을 지원한다.
- **규모 한계(정직 공개)**: 논문의 약 1,400만 text-image pair(SA-1B/JourneyDB/CC3M/
  Datacomp/CelebA-HQ) 및 250k-step 스케줄은 포함하지 않는다. 로컬 데이터는 학습
  파이프라인을 구동하기 위한 **소규모 subset**이며 논문 수치 재현용이 아니다.

### 5.2 평가 축

| 데이터셋 | 용도 |
|---|---|
| COCO val2017 | CLIP image-image/FID, 객체 보존 (text alignment는 캡션 제공 시 확장) |
| Kodak | 고해상도 perceptual fidelity(PSNR/SSIM/LPIPS/FID) |
| ADE20K validation | 영역/의미 보존성 분석(segmentation consistency) |

- 입력은 128×128 패치 타일링 후 merge하며, H·W를 128의 배수로 맞춘다.
- **text alignment(CLIP text-image)에 대한 주의**: 표준 평가 경로는 기본적으로 캡션을
  로드하지 않으므로 text-image 정합은 자동 산출되지 않는다. 이는 캡션(또는 캡션
  추출)을 함께 공급할 때 동작하는 **확장 지표**이며, 기본 동작이 아니다.
- **mIoU 표현 보정**: 초기 계획의 "ADE20K mIoU" 표현은, 현재 구현 구조상 별도 학습된
  segmentation 정량 mIoU 파이프라인이 아니라 **segmentation-consistency**(원본 대비
  영역/클래스 보존 일관성)와 temporal_segmentation_iou로 다룬다. 정식 mIoU 벤치마크는
  외부 segmentation 모델 정량화가 추가될 때 보강 항목으로 둔다.

### 5.3 운용 결론

- **학습 핵심 축**: ImageNet(Stage 1/보조) + COCO2017(Stage 2/3).
- **학습 보조/확장 축**: JourneyDB subset, CelebA(도메인 특화·캡션 생성 경유).
- **평가 핵심 축**: COCO val2017 + Kodak + ADE20K.
- **평가 보조 축**: CelebA 계열(얼굴 도메인 artifact/hallucination 분석).

---

## 6. 평가 지표 *(강화)*

지표를 단순 나열하지 않고, "시맨틱 미디어 전송 플랫폼" 안에서 각 지표가 **무엇을
검증하는지**와, 본 과제의 핵심 연구개발 요소가 **어떤 지표와 정량적으로 매핑되는지**를
함께 기술한다.

### 6.1 지표별 기술적 의미

- **PSNR / SSIM / LPIPS** — 픽셀·구조·지각 충실도. "신호가 얼마나 보존됐는가"를
  검증한다. 단, 의미 붕괴를 잡지 못하는 한계가 있어 시맨틱 지표와 함께 봐야 한다.
  (SSIM은 논문 외 확장 지표로 유지)
- **CLIP image-image** — 원본↔복원 임베딩 유사도. "전체적 의미 정합"을 검증한다.
- **CLIP text-image** — (BLIP2) 캡션↔복원 유사도. "텍스트로 표현된 송신 의도가
  수신 이미지에 반영됐는가"를 검증한다. 단, 표준 평가 경로는 기본적으로 캡션을 로드하지
  않으므로 이 지표는 캡션을 함께 공급할 때만 산출되는 **확장 지표**다.
- **object_preservation_rate / missing_object_rate** — 원본 객체의 보존/누락.
  "있어야 할 것이 살아남았는가"를 검증한다.
- **additional_object_rate / hallucination_score** — 없던 객체의 생성 정도.
  "지어내지 않았는가(할루시네이션)"를 검증한다 — ETRI 핵심 실패 양상.
- **SRS (Semantic Reliability Score)** — 위 시맨틱 지표를 가중 결합한 대표 지표
  (clip image-image 0.30 + clip text-image 0.25 + object preservation 0.25
  − missing 0.10 − additional 0.10). "기본 수준의 시맨틱 신뢰성"을 대표한다.
  (clip text-image 항은 캡션이 공급될 때만 기여하므로, 캡션 없이 평가하면 SRS는
  나머지 항으로 구성된다.)
- **srs_packet** — 의미 단위(객체/관계/속성/장면) 분해 검증 기반 SRS. "유사도로는
  안 보이는 의미 단위 오류"까지 반영한 **세분화된 신뢰성**을 대표한다.
- **srs_v2** — base + packet + temporal + VQA 항을 통합한 최상위 신뢰성 지표.
  "영상·시간축·강한 할루시네이션 검증까지 포함한 종합 신뢰성"을 대표한다.
- **temporal_srs / srs_flicker / object_identity_consistency /
  temporal_segmentation_iou / temporal_hallucination_rate** — 영상/시간축 관점.
  각각 "시퀀스 평균 신뢰성 / 프레임 간 신뢰성 흔들림(낮을수록 안정) / 객체 동일성
  유지 / 영역 일관성 / 시간적 할루시네이션 발생률"을 검증한다.
- **overhead_reduction** — 프레임별 전 정보 전송 대비 시맨틱 유닛 전송 절감률.
  "효율(중복 제거)"을 검증한다.
- **FID** — 분포 수준 perceptual 품질(논문 §VI 정합). 데이터셋/SNR 단위 지표로,
  같은 SNR 그룹 전체에 동일 값이 기록된다. 실제 Inception 백본이 있을 때만 논문과
  비교 가능하며, proxy/미가용 시 fail-fast 처리한다.

### 6.2 기록되는 지표 구성

- 표준 평가 경로의 기본 산출 지표: psnr, ssim, lpips, clip image-image,
  object preservation rate, missing object rate, additional object rate,
  hallucination score, semantic reliability score(SRS), fid(+백본 출처 표기).
  - clip text-image는 기본 산출에 포함되지 않으며, 캡션(또는 캡션 추출)을 함께
    공급할 때만 산출되는 확장 지표다(5.2·6.1 참조).
- 확장 지표(패킷 인식 또는 SRS-v2/regeneration-search 활성 시): srs_base, srs_packet,
  srs_v2, object match rate, relation consistency, attribute consistency,
  segmentation consistency, scene match, missing/additional object count,
  relation/attribute error count, 적용된 가이드 강도(guidance regime),
  선택된 재생성 전략(regeneration strategy).
  - 단, **srs_v2는 패킷 인식만으로는 생성되지 않으며**, 별도의 SRS-v2 활성
    플래그(또는 regeneration-search에서 검증 지표로 지정)가 켜져야 계산된다(조건부 생성).

### 6.3 핵심 연구개발 요소 ↔ 검증 지표 대응표

| 연구개발 요소 | 1차 검증 지표 | 보조 검증 지표 |
|---|---|---|
| hallucination 완화 모듈 (VQA evaluator, regeneration) | hallucination_score, additional_object_rate | srs_v2, srs_packet |
| packet-aware verifier (semantic packet) | srs_packet, object_match_rate | relation_consistency, attribute_consistency, segmentation_consistency, missing/additional_object_count |
| adaptive guidance controller | SNR sweep 전반의 SRS/srs_packet 개선 | 적용된 가이드 강도 로깅 |
| regeneration search | SRS/srs_v2(검증 metric 선택) 최댓값 | 선택된 재생성 전략 로깅 |
| temporal pipeline (keyframe/delta) | temporal_srs, overhead_reduction | srs_flicker, object_identity_consistency, temporal_segmentation_iou, temporal_hallucination_rate |
| channel-conditioned reconstruction | SNR sweep 전반 SRS/FID/CLIP 개선 | reliability 기반 guidance/step 로깅 |
| low-latency sampling (DDIM/early-exit) | 지연(sec/image, step 수) 대비 SRS/LPIPS 유지 | 50-step 대비 speedup |
| edge codec (전용 엣지 링크) | edge-link BCE/Dice/IoU@0.5/F1@0.5 | 다운스트림 srs_packet(구조 보존) |

---

## 7. 핵심 연구개발 요소와 평가지표 대응 구조 *(신규, 통합 관점)*

본 과제의 연구개발 요소는 "어떤 한계(3장) → 어떤 송수신 구조 변화(4장) → 어떤 지표로
검증(6장)"의 삼중 대응으로 설계된다. 통합 관점은 다음과 같다.

- **신뢰성 분해 트랙(L1, L7)**: semantic packet · packet-aware verifier · VQA
  evaluator → Rx 검증 계층 추가 → srs_packet / srs_v2 / hallucination_score.
- **채널 적응 트랙(L2, L6)**: adaptive guidance · channel-conditioned diffusion ·
  CSI estimation → Rx 조건화 복원 + Tx 가이드 강도 제어 → SNR sweep 전반의 SRS/FID/CLIP.
- **부가정보 효율 트랙(L3, L4)**: semantic delta · temporal reuse · edge codec(구현) +
  가이드 전용 손상 모델(설계 방향) → Tx 부가정보 계층/전송정책 → overhead_reduction /
  손상 견고성 하의 SRS.
- **저지연 트랙(L5)**: DDIM step-budget · early-exit · consistency decoder →
  Rx 복원 경로 단축 → 지연 대비 SRS/LPIPS 유지 곡선.
- **시간축 트랙(L8)**: temporal pipeline · temporal evaluator → 영상 전송 구조 →
  temporal 계열 지표군.

---

## 8. 현재 구현 진행 현황 및 원본 SGD-JSCC 대비 보완 *(신규, 구현 상태 기준)*

본 장은 계획이 아니라 **현재 구현 상태**를, "원본 `SGDJSCC/` 패키지가 직접 제공하던
것 ↔ 제공하지 않던 것 ↔ 본 과제에서 어떻게 메웠는지"의 관점으로 정리한다. 각 항목은
**무엇을, 왜 그렇게 구현했는지**와 **충실도 분류(논문 정합 / paper-like / scaffold /
ETRI 확장)**, 그리고 **ETRI 과제(시맨틱 신뢰도 평가·향상) 관점의 필요성**을 함께
기술한다.

### 8.1 원본 SGD-JSCC가 직접 제공하던 것 / 제공하지 않던 것

원본 `SGDJSCC/`는 **단일 추론 스크립트 중심**의 공개 코드다. 즉 논문 알고리즘의
*추론 순전파*는 충실히 담고 있으나, 과제 수행에 필요한 *반복 평가·비교실험·기능 확장·
학습 재현*을 위한 바깥쪽 구조는 거의 제공하지 않는다. (원본 README의 TODO에도
"전처리 스크립트"와 "DM/ControlNet 학습 가이드라인"이 **미공개**로 명시되어 있다.)

**원본이 직접 제공하며 강한 부분 (그대로 보존·재사용)**

- 논문 핵심 forward 경로: VAE 인코딩·정규화, AWGN 채널 손상, blind SNR 예측,
  연속 timestep DiT 확산 디노이징, ControlNet 구조 가이드 조건화, 최종 VAE 디코드.
- step matching, canny(구조 가이드) 재전송, mask token·power scalar 등 복원 핵심 연산.
- 확산/ControlNet/VAE/blind SNR 예측망/캡션·엣지 추출 등 **모델 빌딩블록**.
- (부분) fading 유틸: Rayleigh 가중치 형태의 채널 함수가 코드에는 존재하나, 공개
  추론 경로 자체는 AWGN 중심이며 연구용 채널 추상화·CSI 모드로 정리되어 있지는 않다.

→ 본 과제는 이 forward 경로의 **수치적 특성을 변경하지 않고 그대로 재사용**한다
(알고리즘 보존 불변식). 따라서 이 부분의 충실도는 **논문 정합(paper-faithful)** 이다.

**원본이 제공하지 않거나 제한적이던 부분 (본 과제가 메운 공백)**

- 모듈화: 채널·가이드·복원·평가가 단일 스크립트에 결합되어 교체·확장이 어려움.
- End-to-End 평가 경로와 결과 로깅(CSV) 부재 → 반복 실험·SNR sweep·비교가 어려움.
- 시맨틱 의도 보존을 측정하는 **종합 신뢰도 지표(SRS) 부재**.
- 할루시네이션·객체 보존을 정량화하는 평가 체계 부재.
- 의미 단위 검증(packet-aware verifier) 부재.
- 영상/시간축(temporal) 평가 및 의미 델타 재사용 부재.
- 다중 채널(Rayleigh/fast-fading/packet-drop) 연구 실험 구조 및 채널 조건화·블라인드
  복원 실험 구조 부족.
- 저지연(few-step·early-exit·지연 프로파일) 실험 구조 부재.
- stage-aware 학습 scaffold 부재(학습 가이드라인 자체가 원본 README상 미공개 TODO).

### 8.2 기능군별 구현 배경과 설계 이유 (원본 한계 → 보완 → 기대 효과)

각 기능군을 "① 원본의 한계 → ② 구현한 보완(무엇을·왜 그렇게) → ③ 충실도 분류 ·
ETRI 관점 기대 효과" 3단으로 설명한다.

**(1) 모듈형 구조 (channels / guidance / models / pipelines / evaluators / controllers
/ acceleration / video / training)**
- ① 원본은 단일 추론 스크립트 구조라, 채널을 바꾸거나 평가기를 끼우거나 가이드를
  추가하려면 핵심 경로를 직접 수정해야 했고, 회귀·재현성 관리가 어려웠다.
- ② 책임을 분리해 **교체 가능한 모듈**로 재구성했다: 채널 손상 로직, 시맨틱/구조
  가이드 추출, 모델 조립, 실행 오케스트레이션, 연구 지표, 적응 제어, 가속, 영상,
  학습을 각각 독립 모듈로 두고, 원본 모델 코드는 런타임에 *읽기 전용*으로 재사용한다.
  설정(config) 조합과 마스터 스위치로 기능을 게이팅해, **끄면 원본 경로와 동일**하게
  동작하도록 했다.
- ③ **ETRI 확장(구조)** — 알고리즘은 보존하고 구조만 개선. 비교실험·단위 테스트·
  기능 추가를 회귀 없이 가능하게 하는 토대로, 이후 모든 보완의 전제가 된다.

**(2) End-to-End 평가 경로 + 결과 로깅**
- ① 원본은 전처리·평가 스크립트가 미공개여서, 데이터셋 단위 반복 평가와 SNR sweep
  결과의 체계적 수집이 불가능했다.
- ② 입력→복원→지표→결과 행 스트리밍(CSV)으로 이어지는 평가 파이프라인을 두고,
  추론 코어를 *대체하지 않고 감싸도록* 했다(복원 수치는 동일 유지). 단일 SNR과
  SNR sweep을 같은 경로로 처리한다.
- ③ **ETRI 확장** — "노이즈 채널 통과 후 의미 보존도"를 정량 비교하려면 반복·기록
  가능한 평가 골격이 선행되어야 하기 때문.

**(3) 시맨틱 신뢰도 지표 (SRS → srs_packet → srs_v2)**
- ① 원본/논문은 PSNR·perceptual·CLIP 중심으로, 송신 의도 보존을 직접 대표하는
  종합 지표가 없었다.
- ② CLIP(이미지·텍스트)·객체 보존·누락/추가를 가중 결합한 **SRS**를 대표 지표로
  설계하고, 이후 의미 단위 분해(srs_packet)와 시간축·VQA 통합(srs_v2)으로 단계적
  확장했다. 구성 지표가 안정된 뒤 종합 지표를 확정한다는 순서를 따랐다.
- ③ **ETRI 확장** — 과제의 성공 기준 자체가 픽셀이 아니라 의미 신뢰도이므로,
  이를 대표할 단일·세분 지표가 반드시 필요했다.

**(4) 할루시네이션·객체 보존 + packet-aware verifier**
- ① 원본은 "그럴듯하지만 틀린(hallucinated)" 복원, 객체 누락/추가를 측정할 체계가
  없었다.
- ② 객체 보존·누락·추가율과 할루시네이션 점수를 평가기로 추가하고, 이미지를
  객체·관계·속성·장면의 **의미 명세서(semantic packet)** 로 분해해 원본↔복원을 비교하는
  검증기를 두어 의미 단위 오류를 **개수로** 집계한다.
- ③ **ETRI 확장 / 일부 scaffold** — 객체·관계 추출은 현재 캡션+CLIP 휴리스틱(scene-graph
  아님)이라 근사다. 그러나 할루시네이션 정량화는 ETRI 핵심 실패 양상이므로, 휴리스틱
  단계에서라도 측정 체계를 확보하는 것이 우선이었다.

**(5) adaptive guidance + regeneration loop / search**
- ① 원본은 채널 상태와 무관하게 고정된 복원이라, 저SNR 의미 붕괴나 고SNR 가이드
  부작용에 능동 대응이 없었다.
- ② SNR 구간별로 가이드 강도·확산 스텝을 조절하는 적응 제어와, 신뢰도가 낮을 때
  채널 재전송 없이 **가이드 조합을 바꿔 재복원**하는 재시도(단발 loop → 다중 전략
  search)를 추가했다. search는 검증 점수가 가장 높은 출력을 선택한다.
- ③ **ETRI 확장** — "채널 재전송 비용 없이 수신단에서 신뢰도를 회복"하는 정책은
  시맨틱 전송 신뢰성 향상의 직접 수단이기 때문.

**(6) temporal pipeline / semantic delta / keyframe reuse**
- ① 원본은 단일 이미지 전송만 다루며, 영상의 프레임 간 중복이나 의미 델타 재사용을
  고려하지 않았다.
- ② 키프레임은 전체 패킷을, 인터프레임은 변경분(델타)만 전송/재사용하는 시간축
  파이프라인과 시간적 일관성 지표(temporal_srs, srs_flicker 등)를 추가했다.
- ③ **ETRI 확장 / scaffold** — 인터프레임은 현재 키프레임 복원 재사용 수준이고
  단계적 조건화는 prompt 레벨 근사다. 그럼에도 키프레임 미디어 전송의 효율·시간적
  신뢰도를 측정하려면 시간축 골격이 필요했다.

**(7) 다중 채널 (Rayleigh / fast-fading / packet-drop) + channel-conditioned inference**
- ① 원본 공개 경로는 AWGN 중심이고(fading 유틸은 코드에 일부 존재하나 연구용으로
  정리되어 있지 않음), 미지/불완전 CSI·블라인드 복원 실험 구조가 없었다.
- ② AWGN과 호환되는 인터페이스로 Rayleigh/fast-fading/packet-drop 채널을 추가하고,
  수신 측정값(received/equalized latent, gain, noise, mask, SNR, reliability)을
  **측정 번들**로 추상화해 latent/joint/blind 모드의 채널 조건화 추론을 둔다.
  MMSE 등화와 fast-fading water-filling 디노이징(Alg.4)도 포함한다.
- ③ **혼합** — 등화·water-filling 알고리즘은 **논문 정합**(단, 실제 수치는 체크포인트
  의존, 위상은 실수 gain 근사)이고, 채널 조건화·CSI 모드·블라인드 실험 골격은
  **ETRI 확장**이다. 현실 무선 환경의 신뢰도 평가에는 AWGN을 넘는 채널과 CSI 가정
  완화가 필요하기 때문.

**(8) 저지연 복원 (DDIM step-budget / dynamic routing / early-exit / latency profiler)**
- ① 원본 복원은 다단계 디노이징(약 50 step)으로 지연이 크고, 품질-지연 트레이드오프를
  실험할 구조가 없었다(논문도 distillation을 향후 과제로 언급).
- ② 스텝 예산 조절, 채널·신뢰도 기반 동적 라우팅, 샘플러 내부 early-exit, 지연
  프로파일러를 추가하고, 향후 distilled student를 끼울 인터페이스를 마련했다.
- ③ **ETRI 확장** — distilled 디코더는 아직 placeholder(few-step 근사)다. 다만 저지연은
  "검증된 시맨틱 품질을 유지한 채" 판단해야 하므로, 신뢰도 지표 위에서 지연을
  실험할 구조가 필요했다.

**(9) stage-aware training scaffold (jscc / text_dm / controlnet / edge_codec /
end_to_end_ft / csi_estimation)**
- ① 원본은 학습 가이드라인 자체가 미공개(README TODO)였고, 손실 함수는 있으나 학습
  루프·옵티마이저·stage 스케줄이 없어 베이스라인 재학습·ablation이 불가능했다.
- ② 논문 3-stage(JSCC → text-DM → ControlNet)를 stage별 데이터/순전파/손실/freeze
  정책으로 구조 재현하고, 보조 stage(전용 엣지 codec, CSI 추정)와 확장(end-to-end
  미세조정)을 추가했다. step 기반 학습(AMP·grad-accum)과 체크포인트 변환도 둔다.
- ③ **paper-like / scaffold** — 추론 경로는 논문 정합이지만, 학습 scaffold는 구조적
  근사다(대규모 데이터·정확 손실 가중·논문 수치는 미보장). 베이스라인을 직접 학습·
  비교하려면 구조라도 갖춘 학습 골격이 필요했다.

### 8.3 기능군별 현황 (요약 표)

상태 표기: ✅ 구현·연결됨 / 🟡 부분·스캐폴드 / 🔲 미구현·검증대기.

| 기능군 | 상태 | 비고 |
|---|---|---|
| 원본 추론 경로 보존 | ✅ | VAE 스케일링·step matching·canny 재전송 등 paper-faithful |
| 모듈화 구조 | ✅ | 채널/가이드/모델/파이프라인/평가/제어/가속/영상/학습 분리 완료 |
| 평가 파이프라인 + CSV 로깅 | ✅ | 단일 SNR + SNR sweep |
| 품질/CLIP 지표 | ✅ | PSNR/SSIM/LPIPS, CLIP image-image (text-image는 캡션 공급 시) |
| 객체 보존/할루시네이션(휴리스틱) | ✅ | CLIP/캡션 휴리스틱 기반 |
| SRS | ✅ | 대표 지표 |
| FID | ✅(연결) | 실제 Inception 백본은 네트워크/가중치 의존(세션 외) |
| guidance 확장(depth/segmentation) | ✅ | 최초 사용 시 외부 모델 다운로드 |
| semantic packet | 🟡 | 캡션+CLIP 휴리스틱(scene-graph 아님), 메타데이터 직렬화 |
| packet-aware verifier | ✅ | 의미 단위 오류 카운트(객체/관계/속성/장면) |
| adaptive guidance | ✅ | SNR 구간별 강/중/약 |
| regeneration policy / loop | ✅ | 실패유형 인식 재시도 |
| 비디오/시간축 파이프라인 | 🟡 | 인터프레임=키프레임 복사, 단계적 prompt는 prompt 레벨 |
| 채널(Rayleigh/fast-fading/packet-drop) | ✅ | AWGN 호환 전송 + 풍부한 관측 |
| 가이드 전용 손상 모델 | 🟡 | 설계 방향. 현재는 segmentation region-dropout, 학습 단계 CFG label-dropout 등 일부 요소만 존재 |
| 채널 조건화 추론 | 🟡 | 조건 토큰을 frozen denoiser가 직접 소비하진 않음(재학습 필요) |
| water-filling 디노이징(Alg.4) | 🟡 | 알고리즘·배선 완료, 실제 DM 수치는 GPU/체크포인트 의존 |
| CSI 추정 | 🟡 | SNR 추정망 학습·추론 연결됨; phase/joint(Alg.3)은 scaffold |
| 저지연(step-budget/early-exit) | 🟡 | 연결됨, distilled student는 placeholder |
| VQA / SRS-v2 / regeneration search | ✅(연결) | VQA 백본 가중치는 미번들(없으면 CLIP fallback) |
| 학습 scaffold (3-stage) | 🟡 | 구조 재현(부분 충실), 대규모 데이터·정확 수치 미보장 |
| edge codec 학습 | 🟡 | conv 기본/ViT 옵션, WITT-exact 아님 |
| 마스터 스위치 게이트 | ✅ | 확장 기능 일괄 on/off, 베이스라인 동등성 보장 |

### 8.4 미구현 / 추가 검증 필요 (정직 공개)

- 채널 조건 **토큰을 소비하는 조건 인식(재학습) denoiser** — FiLM/cross-attention/
  posterior-gradient guidance는 미구현(frozen denoiser 제약).
- 학습된 **consistency/distilled student 디코더** — 인터페이스만, 현재 few-step 근사.
- 실제 **확산 체크포인트 기반 water-filling 수치** — GPU/체크포인트 의존.
- **phase/joint CSI 추정망(Alg.3)** — 복소 위상 채널 확장 전엔 scaffold.
- **실제 Inception-FID 수치**, **patch-GAN/LPIPS 수치**, **약 1,400만 pair/250k-step
  재현** — 데이터·컴퓨트·네트워크 의존(세션 외).
- semantic packet의 **실제 채널 코딩/드롭** — 현재 평가·제어 메타데이터 수준.
- **가이드 전용 손상 프레임워크**(엣지 dropout/blur/erasing, 캡션 token dropout 등의
  통합 손상 규칙) — 설계 방향이며 아직 미완. 현재는 일부 요소만 부분 존재.
- 표준 평가 CLI의 **text alignment(CLIP text-image) 기본 미산출** — 캡션 공급 시
  동작하는 확장 지표.
- 정식 **mIoU 벤치마크** — 현재는 segmentation-consistency로 대체.

> 검증 상태: 단위·통합·synthetic 테스트는 통과한다(라우팅·어댑터·패치별 evidence·
> CSI 정책·FID fail-fast 등). 단, 실제 체크포인트 기반 수치 산출, DM water-filling
> 수치, Inception-FID 수치는 구현 경로 기준으로만 검증되었고 수치 재현은 별도다.

### 8.5 현재 구현의 기술적 의의

위 보완들이 모인 결과, 본 과제의 구현은 원본 대비 다음 세 가지 구조적 변화를 만든다.

- **송수신단 구조**: 원본의 고정된 단일 추론 경로가, 송신단에서는 가이드·의미 패킷·
  델타 계층을, 수신단에서는 채널 측정·조건화·재생성·저지연 경로를 *선택적으로*
  얹을 수 있는 구조로 확장되었다. 원본 forward 수치는 보존되므로, 확장은 베이스라인을
  훼손하지 않는다.
- **평가 체계**: PSNR/perceptual 중심에서, 의미 보존·할루시네이션·시간적 일관성을
  대표·세분 지표로 정량화하는 체계로 전환되었다. 이는 ETRI 과제의 성공 기준(시맨틱
  신뢰도)을 직접 측정 가능하게 한다.
- **실험 가능성**: 단일 이미지 1회 추론에서, 데이터셋·SNR sweep·다중 채널·영상·
  ablation·(부분) 재학습을 마스터 스위치로 켜고 끄며 반복·비교할 수 있는 연구
  프레임워크로 바뀌었다.

요약하면, 본 과제는 **원본의 강점(논문 정합 추론 코어)은 그대로 두고**, 원본이 비워둔
**측정·제어·확장·학습 골격을 메워** "시맨틱 미디어 전송 신뢰성"을 정량적으로 다룰 수
있게 만든 것이다. 다만 그 메움의 충실도는 항목마다 논문 정합 / paper-like / scaffold /
ETRI 확장으로 다르며(8.2~8.4 참조), 일부는 여전히 휴리스틱·근사·placeholder 단계임을
명확히 구분한다.

---

## 9. 개발 범위 및 단계별 일정 *(재작성)*

원칙: 원본 SGD-JSCC를 기준 모델로 두고, 별도 연구 패키지에서 단계적으로 고도화한다.
각 Phase는 "무엇을 만드는 단계인지"와 "왜 이 순서인지"가 드러나도록 정리한다.

### Phase 1 — 베이스라인 재현 *(완료)*
- **목표**: 원본 AWGN 추론을 재현 가능하게 고정(이후 비교의 신뢰성 확보).
- **핵심 산출물**: AWGN 단일 이미지/폴더 추론, 설정 기반 실행.
- **검증**: 원본 forward 경로와 수치 일치(알고리즘 보존 불변식).

### Phase 2 — 모듈형 구조 *(완료)*
- **목표**: 연구적 변경이 쌓이기 전에 채널/가이드/모델/파이프라인을 독립 모듈로 분리.
- **핵심 산출물**: 설정 조합(composition), 단위 테스트.
- **검증**: 설정/입출력/채널 단위 테스트.
- **왜 이 순서인가**: 확장 지점을 먼저 안정화해야 이후 기능 추가가 회귀 없이 가능.

### Phase 3 — 시맨틱 우선 평가 기반 *(완료)*
- **목표**: 입력→복원→지표→결과 로깅의 end-to-end 평가 경로 구축, 성공 기준을 픽셀이
  아닌 의미 보존으로 정의.
- **핵심 산출물**: 평가기 모음(품질/CLIP/객체보존/할루시네이션/SRS), SNR sweep 결과,
  depth/segmentation 가이드, regeneration loop 프로토타입.
- **검증**: 데이터셋별 평가 실행 + 결과 컬럼 일관성.

### Phase 4 — 신뢰도 세분화 + 영상 확장 *(대부분 완료)*
- **목표(4-A)**: "비슷한가"를 넘어 "무엇이 누락/추가/왜곡됐는가"를 의미 단위로 측정·
  제어. **목표(4-B)**: 단일 이미지 → 영상 키프레임 단위 효율 전송.
- **핵심 산출물**: semantic packet(의미 명세서), packet verifier 오류 리포트,
  adaptive guidance, 실패유형 regeneration, 키프레임/델타 + temporal 지표,
  오버헤드 절감 리포팅.
- **검증**: 패킷 인식 SNR sweep(저SNR에서 SRS는 비슷한데 누락/할루시네이션 급증
  여부), 키프레임 재사용 vs 프레임별 전송 오버헤드 비교.
- **왜 이 순서인가**: 채널/속도 최적화 이전에, **무엇이 의미적으로 깨지는지를 측정**할
  수 있어야 한다.

### Phase 5 — 채널 연구 + 저지연 + 강한 검증 *(부분/스캐폴드)*
- **목표(5-A)**: AWGN을 넘는 채널과 채널 조건화·블라인드 복원. **목표(5-B)**: 저지연
  복원. **목표(5-C)**: 더 강한 검증기와 다중 전략 재생성으로 5-A/5-B를 방어 가능하게.
- **핵심 산출물**: Rayleigh/fast-fading/packet-drop 채널, measurement bundle,
  채널 조건화 추론(latent/joint/blind), water-filling 디노이징, DDIM step-budget·
  early-exit·지연 프로파일, VQA evaluator·SRS-v2·regeneration search.
- **검증**: 비자명한 미지 채널 설정에서 신뢰도 개선, 제한된 SRS 열화로 지연 감소,
  search가 실험 루프에 통합.
- **왜 이 순서인가**: 지연·채널 견고성은 **검증된 시맨틱 품질을 기준**으로 판단해야
  하므로, 신뢰도 세분화(Phase 4) 이후에 둔다.

### (병행 트랙) 학습 재현 scaffold *(부분 충실)*
- **목표**: 논문 3-stage(JSCC → text-DM → ControlNet) 및 보조 stage(edge codec,
  CSI 추정)를 구조적으로 재현해, 베이스라인 학습·ablation을 가능하게 한다.
- **핵심 산출물**: stage별 학습 절차/손실/freeze 정책, step-based 학습(AMP·grad-accum),
  체크포인트 변환, 데이터 입력 확장(멀티캡션·경로 리스트·캡션 생성).
- **검증**: 배선 dry-run + 실모델 gradient 검증(소규모). 단, 대규모 데이터·정확
  수치는 미보장(추론 경로는 paper-faithful, 학습 scaffold는 부분 충실).

### 비교 실험 트랙 *(진행)*
- DeepJSCC / DiffJSCC / SGD-JSCC(baseline) / Proposed(SGD-JSCC + 구조 가이드 +
  신뢰도·할루시네이션 평가)의 head-to-head 및 ablation 체계화.

---

## 10. 실험 시나리오

- **채널**: AWGN 기본, 확장으로 Rayleigh / fast-fading / packet-drop.
- **SNR sweep**: 기본 비교 [-5, 0, 5, 10] dB, 확장 [-5, 0, 5, 10, 15, 20, 25] dB.
- **전송률**: SGD-JSCC·DiffJSCC 논문의 저율 구간 중심.
- **입력 단위**: 128×128 패치 기반 + 고해상도 patch merge.
- **가이드 손상 규칙(설계 방향)**: JSCC latent/채널 심볼에만 AWGN/Rayleigh를 적용하고
  가이드에는 직접 채널잡음을 주지 않는다는 원칙 하에, 엣지=dropout/blur/erasing,
  세그=영역 제거, 캡션=token dropout으로 손상하는 것을 목표로 한다. 이는 **계획된
  실험 규칙**이며, 현재는 segmentation region-dropout 등 일부 요소만 구현되어 있고
  가이드 전용 통합 손상 프레임워크는 아직 미완이다.
- **대표 ablation**:
  1) baseline SGD-JSCC → 2) +adaptive guidance → 3) +packet verifier →
  4) 키프레임 전용 full packet → 5) 키프레임 + semantic delta →
  6) 채널 조건화(블라인드 유무) → 7) few-step(저지연) 변형.
- **평가 순서**: COCO val2017(semantic 정합; text alignment는 캡션 공급 시 확장) →
  Kodak(perceptual fidelity) → ADE20K(segmentation consistency) → CelebA 계열(도메인
  artifact·hallucination).

---

## 11. 기대 산출물 및 최종 방침

### 11.1 기대 산출물
- SGD-JSCC 기반 모듈형 연구 패키지, 마스터 스위치로 베이스라인 동등성 보장.
- 데이터셋별 재현 가능한 추론/평가/학습 설정.
- 3계층 시맨틱 신뢰성 평가 체계(SRS / srs_packet / srs_v2) + 시간축 지표군 + FID.
- 할루시네이션 완화(검증기·재생성), 채널 적응 복원, 저지연 복원, 영상 효율 전송 모듈.
- 학습 재현 scaffold(부분 충실)와 비교 실험·ablation용 표/그래프/정성 결과.

### 11.2 최종 방침
- **기본 모델**: SGD-JSCC 채택(추론 경로 paper-faithful 보존).
- **데이터 운용**: 학습=ImageNet+COCO(핵심)+JourneyDB/CelebA(보조), 평가=COCO+Kodak+
  ADE20K(핵심)+CelebA(보조).
- **포지셔닝**: 본 과제는 "새 전송 알고리즘"이 아니라, 변경되지 않은 SGD-JSCC 복원
  경로 위에 얹은 **신뢰성 측정·제어 + 영상/채널 확장 레이어**다. 따라서 기여는
  "유사도 지표가 못 잡는 의미 붕괴/할루시네이션의 분해·정량화 및 채널·시간축 확장"에
  둔다.
- **기대 효과**: 송신 의도 보존, 수신 의미 일치도 검증, 할루시네이션 완화에 직접 부합.

---

## 12. 초기 계획 대비 변경/확장 포인트 요약표

| 항목 | 초기 계획 | 현재 업데이트 방향 |
|---|---|---|
| 평가 지표 체계 | PSNR/SSIM/LPIPS/FID/CLIP/mIoU + 추가 지표(object preservation 등) | SRS → srs_packet → srs_v2 3계층 + 시간축 지표군 + FID(fail-fast) |
| 할루시네이션 평가 | hallucinated object count(휴리스틱) | CLIP 휴리스틱 + VQA evaluator + packet 기반 카운트(누락/추가/관계/속성) |
| 복원 제어 | 고정 복원 | SNR 적응형 가이드 + 실패유형 regeneration + 다중전략 search |
| 채널 | AWGN 단일 | AWGN + Rayleigh/fast-fading/packet-drop + 채널 조건화(latent/joint/blind) + water-filling |
| 복원 속도 | (미고려) | DDIM step-budget + dynamic routing + early-exit + consistency decoder 인터페이스 |
| 매체 범위 | 단일 이미지(128×128) | + 영상 키프레임/델타/temporal 지표(오버헤드 절감 등) |
| 부가정보 | text + edge | + depth/segmentation + semantic packet/delta + edge codec(전용 링크) |
| 학습 | (추론 전용 구상) | 논문 3-stage + edge codec/CSI 추정 학습 scaffold(부분 충실) |
| 데이터셋 | COCO/Kodak/ADE20K/CelebA-HQ, mIoU 중심 | 학습축(ImageNet/COCO/JourneyDB/CelebA) + 평가축(COCO/Kodak/ADE20K); mIoU→segmentation-consistency 보정 |
| 확장 제어 | (없음) | 마스터 스위치로 확장 기능 일괄 on/off, 베이스라인 동등성 보장 |
| 충실도 구분 | (미구분) | 추론=paper-faithful / 학습·채널·평가=paper-like·scaffold·확장으로 명시 구분 |

---

## 참고 링크
- SGD-JSCC: https://arxiv.org/abs/2501.01138
- DiffJSCC: https://arxiv.org/abs/2404.17736
- DeepJSCC: https://arxiv.org/abs/1809.01733
