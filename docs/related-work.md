# 관련 연구와 데이터셋 조사

00단계 산출물. README 「관련 연구와 차별점」 절의 근거 문서이고,
09.10 제안서의 관련연구 절은 이 문서에서 추린다.

- 조사일 — 2026.09.08
- 담당 — 개발 B (이수현)
- 범위 — 지연 버퍼 계열, 갑툭튀 오디오 특징, 트리거 워닝 상용 제품, CLIP zero-shot 안전성, 평가 데이터셋

## 요약 — 이 조사로 바뀐 것

1. **차별점 문장을 다시 좁혔다.** 라이브 스트리밍의 송출 지연 버퍼를 탐지에 활용하는
   연구가 이미 있다([Time Buffers](https://arxiv.org/abs/2010.03016), **정독 완료**).
   README가 PSE 비교 뒤에 결론으로 쓴 "의미론적 사건을 lookahead로 잡는다가 새롭다"는
   이 논문이 이미 하는 일이다. → 남는 주장은
   **"있는 버퍼를 쓴 것이 아니라, 개입을 위해 없던 지연을 만들었다"** 이다.
   겹치는 것은 여기까지이고, **lookahead 요구량 분류 · 차단 · 클라이언트 실시간은 겹치지 않음을
   정독으로 확인**했다. 덤으로 3초의 근거와 미구현 갭을 얻었다.
   [§1.2](#12-버퍼-시간을-탐지에-활용--online-action-detection에-선례-있음)
2. **평가셋으로 쓸 수 있는 공개 데이터는 사실상 XD-Violence 하나다.** 나머지는 전부 영화 기반이라
   [범위 선언](../README.md#범위-선언)의 DRM 제외에 걸린다. 공개 데이터셋의 역할을
   "평가셋"이 아니라 "라벨 기준 정의 · 특징 검증 · 사전학습"으로 못박는다. [§3.5](#35-결론--공개-데이터셋의-역할)
3. **데이터셋 착수 순서는 XD-Violence 먼저, LIRIS-ACCEDE는 선택이다.** XD-Violence는 즉시 받을 수 있고
   용도가 더 급하다 — 라벨링 30~50시간을 시작하기 전에 오디오 규칙이 되는지부터 검증할 수 있다.
   LIRIS-ACCEDE는 **지도교수 서명이 필요하고**(EULA가 "학술기관 정규직" 서명을 요구해 학생 단독 신청 불가)
   무료 이메일로 보내면 반려되며 처리에 최대 1주가 걸린다. 논문 실험 절을 강화하는 옵션이지
   없으면 막히는 항목이 아니므로 지도교수를 만날 일이 있을 때 묶어서 처리한다.
   [§3.2](#32-liris-accede--mediaeval-emotional-impact-of-movies)

## 1. 선행 연구 계열

### 1.1 지연 버퍼 기반 사전 차단 — PSE 연구에 선례 있음

광과민성 발작(PSE) 도메인에는 프레임 버퍼로 위험 자극을 사용자 도달 전에 완화하는 연구가 이미 있다.
README 「관련 연구와 차별점」에 비교표로 정리되어 있다.

- [Flikcer: A Chrome Extension to Resolve Online Epileptogenic Visual Content](https://arxiv.org/pdf/2108.09491) — 크롬 확장 + 실시간 영상 분석. 구조적으로 가장 가까움
- [Parallel scheme for real-time detection of photosensitive seizures](https://www.sciencedirect.com/science/article/abs/pii/S0010482516000123) — 8프레임 버퍼
- [A new adaptive temporal filter](https://pubmed.ncbi.nlm.nih.gov/11145469/) — 프레임 메모리 버퍼로 사전 감쇠
- [Evaluating Conformance of Video Safety Tools for Photosensitive Epilepsy](https://pmc.ncbi.nlm.nih.gov/articles/PMC12249941/) — 평가 프레임워크

차별점은 **버퍼의 역할**이다. PSE는 시간축 필터링(플리커 평활화)이고 본 연구는 사건 탐지 후 차단이다.
버퍼 크기(0.3초 vs 3초)는 역할이 다른 결과일 뿐 차별점이 아니다.

### 1.2 버퍼 시간을 탐지에 활용 — online action detection에 선례 있음

**이번 조사에서 새로 발견한 계열이다. 2026.09.08 정독 완료.**

- [Online Action Detection in Streaming Videos with Time Buffers](https://arxiv.org/abs/2010.03016)
  (Bowen Zhang, Hao Chen, Meng Wang, Yuanjun Xiong, 2020)
  — 라이브 스트리밍에는 송출 지연이 원래 존재한다는 점에 착안해, 그 **버퍼 시간을 탐지에 활용하는
  문제 설정 자체**를 제안한다. 지연의 출처로 네트워크 전송과 함께 **욕설·노출·방송사고를 막기 위한
  의도적 지연**을 든다. flattened I3D + window-based suppression.
  실험 버퍼는 78프레임 = **약 2~3초**. THUMOS'14 · ActivityNet v1.3 · HACS Segment에서 검증,
  THUMOS'14 기준 **mAP +13.8%**
- [Real-time Online Video Detection with Temporal Smoothing Transformers (TeSTra)](https://arxiv.org/abs/2209.09236) — 실시간 online detection의 latency-정확도 트레이드오프
- [Temporally smooth online action detection using cycle-consistent future anticipation (FATSnet)](https://www.sciencedirect.com/science/article/abs/pii/S0031320321001412)
- [Online human action detection and anticipation in videos: A survey](https://www.sciencedirect.com/science/article/abs/pii/S0925231222003617) — 계열 전체 지도

**왜 문제인가.** README는 PSE 비교표 다음에 "버퍼를 쓴다는 사실은 새롭지 않다.
의미론적 사건을 lookahead로 잡는다가 새롭다"로 결론을 냈다. Time Buffers 논문이 정확히 그것을 한다 —
버퍼 안에서 의미론적 사건(action)을 탐지한다. PSE 비교표에서 "저수준 신호 vs 의미론적 사건"으로
갈라놓은 그 쪽에 이 논문이 이미 서 있다. **겹치는 것은 여기까지다.**

**남는 차이 — 정독으로 확인함.**

| | Time Buffers (2020) | 본 연구 | 확인 근거 |
|---|---|---|---|
| 버퍼의 출처 | 라이브 방송에 **이미 존재하는** 송출 지연 | VOD에 **의도적으로 도입하는** 지연 | 문제 설정 자체가 "있는 지연을 쓰자" |
| 카테고리별 lookahead 요구량 | **구분하지 않음.** 모든 액션에 균일한 버퍼 적용 | 예측 불가형 / 지속 상태형으로 분리 | 분류 개념 부재 확인 |
| 목적 | 탐지 **정확도(mAP)** | 탐지 후 **사용자 도달 전 차단** | 개입·차단·필터링 미구현 |
| 실행 위치 | **언급 없음** (알고리즘 성능만 보고) | **브라우저 클라이언트 실시간** | 배포 위치·FPS 논의 없음 |
| 산출 | 벤치마크 점수 | 동작하는 확장 + 사용자 실험 | |

→ **주장을 "버퍼 활용"이 아니라 "개입을 위해 없던 지연을 만들어낸 설계 결정"에 둔다.**
공짜로 있는 버퍼를 이용한 것과, 사용자에게 3초를 손해 보게 하면서까지 lookahead를 확보한 것은
설계 문제로서 다르다. 이쪽이 오히려 방어하기 쉽다.

**정독으로 얻은 것 둘.**

1. **3초라는 숫자에 근거가 생겼다.** 이 논문의 실험 버퍼가 약 2~3초이고 그 구간에서 mAP 13.8%
   향상을 보고했다. 우리 3초를 **"선행 연구에서 유효성이 보고된 구간을 채택했다"** 로 쓸 수 있다.
   "왜 하필 3초냐"에 대한 답이 임의 선택에서 인용 있는 설계 결정으로 바뀐다
2. **이 논문이 스스로 갭을 열어놨다.** 응용 예시로 *"라이브 웹캐스트에서 NSFW(노출·폭력)가
   탐지되면 실시간 red flag를 보낸다"* 를 들지만 **동기로만 언급하고 구현하지 않는다.**
   게다가 red flag는 알림이지 차단이 아니다. 관련연구에 이를 명시하면
   **"기존 연구가 가능성만 언급한 자리를 구현하고 사용자 실험까지 했다"** 는 포지션이 선다

### 1.3 갑툭튀 오디오 특징 — affective computing에 선례 있음

"jump scare"라는 이름의 논문은 없으나 02단계에서 쓸 특징 설계는 이미 표준이다.

- [Affective Content Analysis in Comedy and Horror Videos by Audio Emotional Event Detection](https://ieeexplore.ieee.org/document/1521500/)
- [Affective Video Retrieval: Violence Detection in Hollywood Movies](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0078506) — energy entropy 기반 급작스러운 큰 소리 측정, 혈흔 탐지가 표준 특징
- [Affective Video Content Analysis: A Multidisciplinary Insight](https://dl.acm.org/doi/10.1109/TAFFC.2017.2661284) — 서베이
- [Multi-modal learning for affective content analysis in movies](https://link.springer.com/article/10.1007/s11042-018-5662-9)

→ 검증된 특징의 재사용이라 안전하나 "우리가 처음 제안한다"고 쓰지 않는다.
이 계열은 전부 오프라인 분석·검색이 목적이고 재생 중 개입이 아니다.

### 1.4 트리거 워닝 브라우저 확장 — 상용 제품 있음

- [PhobiaBlocker](https://chromewebstore.google.com/detail/phobiablocker/gjgghmahciffkcelboddbmkjkjchjnfk) — 주변 텍스트 NLP 기반
- [Trigger Warning Extension](https://chromewebstore.google.com/detail/trigger-warning-extension/emlcofoghclkckaajlejfoegblhahoff) · [Youtube Trigger Warning](https://chromewebstore.google.com/detail/youtube-trigger-warning/dhaighcmggigocehbgigdbmgpkhcekgl?hl=en) · SKIP IT — 크라우드소싱 타임스탬프·메타데이터 기반

→ 픽셀을 보지 않는다. 라벨 없는 영상에 무력하고 실시간이 아니다.

### 1.5 CLIP zero-shot 안전성 — 한계의 근거로 인용

04단계 "파인튜닝 없이 5종"의 낙관성을 문헌 수치로 뒷받침하거나 반박할 수 있는 자리다.
README 위험 표의 「CLIP zero-shot 정밀도」 행을 추정이 아니라 인용으로 바꿀 수 있다.

- [UnsafeBench: Benchmarking Image Safety Classifiers on Real-World and AI-Generated Images](https://arxiv.org/pdf/2405.03486)
  — **정독 완료.** 11개 유해 카테고리에서 전용 분류기 5종과 범용 VLM 3종을 비교한다.
  전체 최고가 **GPT-4V F1 0.709**, 경량 zero-shot은 **LLaVA 0.549 · InstructBLIP 0.559**,
  CLIP 임베딩 기반 전용 분류기 Q16은 0.533. 카테고리별로는 Sexual·Shocking이 약 0.8로 가장 높고
  Hate·Harassment·Self-Harm이 0.6 미만. **Violence는 GPT-4V 기준 0.738**
- [Safe-CLIP: Removing NSFW Concepts from Vision-and-Language Models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07009.pdf) (ECCV 2024)
- [SafeR-CLIP](https://arxiv.org/abs/2511.16743) — 안전성 파인튜닝이 일반화 성능을 떨어뜨리는 문제. 우리가 파인튜닝 대신 zero-shot을 쓰는 근거로도 인용 가능

## 2. 차별점 정리 (갱신)

| 주장 | 상태 | 근거 |
|---|---|---|
| 지연 버퍼로 사전 차단 | ❌ 선례 있음 | PSE 연구 [§1.1](#11-지연-버퍼-기반-사전-차단--pse-연구에-선례-있음) |
| **버퍼 시간을 탐지에 활용** | ❌ **선례 있음 (신규 발견)** | Time Buffers [§1.2](#12-버퍼-시간을-탐지에-활용--online-action-detection에-선례-있음) |
| **버퍼 안에서 의미론적 사건 탐지** | ❌ **선례 있음 (신규 발견)** | 위와 동일 — 기존 README 결론 문장은 폐기 |
| 갑툭튀 오디오 특징 | ❌ 선례 있음 | affective computing [§1.3](#13-갑툭튀-오디오-특징--affective-computing에-선례-있음) |
| 공포증 콘텐츠 블러 확장 | ❌ 상용 제품 존재 | [§1.4](#14-트리거-워닝-브라우저-확장--상용-제품-있음) |
| **개입을 위해 없던 지연을 도입** | ✅ | 기존 계열은 이미 존재하는 버퍼를 이용하거나 오프라인 분석 |
| **재생 중 개입 (online intervention)** | ✅ | 기존 계열의 목적은 정확도·검색. Time Buffers는 NSFW 알림을 동기로만 언급하고 미구현 |
| **lookahead 요구량 기준 아키텍처 분류** | ✅ | Time Buffers는 모든 액션에 균일 버퍼 — 정독 확인 |
| **브라우저 내 실행 × 다중 카테고리 × zero-shot 확장의 조합** | ✅ | 조합 자체가 공백 |

## 3. 평가 데이터 후보

### 3.1 XD-Violence — 가장 잘 맞음

[프로젝트 페이지](https://roc-ng.github.io/XD-Violence/) · [ECCV 2020 "Not only Look, but also Listen"](https://roc-ng.github.io/XD-Violence/)

- 4,754개 untrimmed 영상, 총 217시간, **오디오 포함**, video-level weak label
- 학습 3,954개(폭력 1,905 / 비폭력 2,049), 테스트 800개(500 / 300)
- 6개 클래스 — Abuse, Car Accident, Explosion, Fighting, Riot, Shooting
- 출처가 영화 **및 유튜브**라 우리 범위와 부분적으로 겹친다
- RGB · 오디오 · flow 3개 모달리티 제공
- **접근** — Baidu Netdisk · AliyunDrive · OneDrive로 배포. 국내에서는 **OneDrive 경로가 현실적**이다.
  원본 영상과 **사전 추출 특징(VGGish 오디오 · I3D RGB&Flow)** 이 모두 제공된다.
  02단계 원시 오디오 규칙에는 원본 영상이 필요하고, 03단계 fusion 비교에는 사전 추출 특징이 바로 쓰인다
- **라이선스 미명시** — 프로젝트 페이지에 이용 약관·재배포 조건이 없다. 06단계 재현 절차에서
  원본을 재배포하지 않고 참조만 하는 우리 방침과는 충돌하지 않으나, 논문에 쓰기 전 저자에게 확인한다

**쓰임** — 갑툭튀 라벨은 아니지만 Explosion·Shooting이 "정적 → 급작스러운 큰 소리" 패턴과 겹쳐
02단계 오디오 규칙의 사전 검증에 쓸 수 있다. 더 중요한 것은 **untrimmed라 사건이 없는 시간이
그대로 들어있다**는 점이다 — [클립 추출 규칙](../README.md#클립-추출-규칙)의 대조 클립과 성격이 같아
오탐률 측정에 쓸 수 있다.

후속 연구도 early/late fusion 비교(03단계)의 참고가 된다 —
[Modality-Aware Contrastive Instance Learning](https://arxiv.org/pdf/2207.05500).

### 3.2 LIRIS-ACCEDE / MediaEval Emotional Impact of Movies

[데이터셋](https://www.interdigital.com/data_sets/liris-accede) · [MediaEval 2017 태스크](http://www.multimediaeval.org/mediaeval2017/emotionalimpact/)

- 160편(전문·아마추어) 기반. 호러·코미디·드라마·액션 등
- MediaEval 2017~2018에서 **fear 구간의 시작·종료 시각**이 라벨됨. 개발셋 44편 15시간 20분
- 발렌스·어라우절은 초 단위 라벨

**쓰임** — 시간축 라벨이 붙은 공개 벤치마크로는 갑툭튀에 가장 가깝다.
fear는 지속적 공포도 포함해 갑툭튀와 정확히 같지 않지만, **발표된 수치와 비교할 수 있다**는 점이
03단계 논문 표를 단단하게 만든다.

**신청 절차** — [EULA PDF](https://s3.amazonaws.com/files.interdigital.com/55bd0288af0b0930ba599bd0c4b7ca38/resources/uploads/data_sets/EULA.pdf)를
출력·서명·스캔해 `accede@liris.cnrs.fr`로 보낸다. 처리에 최대 1주.

주의할 조건이 둘 있다.

- **학생은 서명할 수 없다.** EULA 2조는 *"학술기관에 정규직(permanent position)으로 재직 중인 사람"* 이
  서명자여야 한다고 못박는다. **지도교수 서명이 필요하다.** 같은 기관 소속 연구자를 최대 5명까지
  문서 말미에 기재할 수 있고, 기재된 사람만 데이터를 쓸 수 있다 — 여기에 팀원 2명을 올린다
- **무료 이메일은 거부된다.** gmail·hotmail·yahoo 등으로 신청하면 반려되므로 학교 계정으로 보낸다

기재 항목 — 서명자 이름 · 소속 · 주소 · 이메일 · 추가 연구자 1~5 · 사용 계획(선택) · 날짜/장소 · 서명.

**사용 제한** — 학술 연구 전용이다. 상용 시스템의 성능 입증·학습·테스트, 데이터 판매, 군사 응용,
공공장소용 정부 시스템 개발이 금지 목록에 명시돼 있다. 캡스톤 연구와 논문은 문제없으나,
**이 데이터로 만든 모델을 확장으로 일반 배포하는 것은 회색지대**이므로 04단계 전에 판단해둔다.

**재배포 금지** — 학술 발표를 명확히 하기 위한 소량 인용을 빼면 데이터 배포가 금지된다.
06단계 "URL + 타임스탬프 + 라벨만 공개" 방침과 충돌하지 않는다(오히려 같은 방향).

**인용 의무** — 사용한 서브셋마다 지정된 논문을 인용해야 한다. fear 라벨을 쓰면
MediaEval 2017/2018 Emotional Impact of Movies Task 논문(Dellandréa et al.)이 대상이다.

### 3.3 Where's the Jump / Where's the Scares — 라벨 기준 참고용

- [wheresthejump.com](https://wheresthejump.com/full-movie-list/) — 수백 편 규모, 영화별 갑툭튀 개수·평점. 활성
- [wheresthescares.com](https://wheresthescares.com/) — "수동 검증" 표방. 788개 타임스탬프 / 영화 152편 · TV 4시즌 · 게임 3종

**제약** — 둘 다 API·일괄 다운로드가 없어 스크래핑이 필요하고 ToS 확인이 선행되어야 한다.
그리고 결정적으로 **둘 다 영화 기반**이라 [범위 선언](../README.md#범위-선언)의 DRM 제외에 걸린다.
라벨 기준(무엇을 갑툭튀 1건으로 셀 것인가) 정의용으로만 쓴다.

### 3.4 AudioSet / YAMNet — 사이렌은 거의 공짜

AudioSet 온톨로지에 `Siren`, `Screaming`, `Explosion`, `Gunshot` 클래스가 이미 있다.
04단계의 "사이렌은 YAMNet 기존 클래스에 임계값만(약 1일)" 계획은 이 근거 위에 서 있다.

### 3.5 결론 — 공개 데이터셋의 역할

**평가셋은 우리가 직접 라벨링한 유튜브 VOD여야 한다.** 범위 선언이 DRM 콘텐츠를 제외했고,
위 데이터셋 중 XD-Violence를 빼면 전부 영화 기반이기 때문이다.

공개 데이터셋의 역할은 셋으로 한정한다.

1. **라벨 기준 정의** — 무엇을 1건으로 셀지, 구간을 어디까지 잡을지 (Where's the Jump, LIRIS-ACCEDE)
2. **특징 사전 검증** — 우리 평가셋 라벨링 전에 오디오 규칙이 되는지 확인 (XD-Violence)
3. **사전학습·비교 기준** — YAMNet 임베딩, 발표된 수치와의 대조 (AudioSet, MediaEval)

이 구분을 01단계 라벨 기준 문서 첫 줄에 명시한다. 나중에 "왜 공개 데이터셋으로 평가 안 했나"라는
질문에 대한 답이 곧 범위 선언이다.

## 4. 읽는 순서

| 순서 | 문헌 | 왜 | 상태 |
|---|---|---|---|
| 1 | [Time Buffers](https://arxiv.org/abs/2010.03016) | 차별점 문장이 여기 걸림 | ☑ 2026.09.08 정독 — [§1.2](#12-버퍼-시간을-탐지에-활용--online-action-detection에-선례-있음) |
| 2 | [UnsafeBench](https://arxiv.org/pdf/2405.03486) | 위험 표의 CLIP 정밀도 행을 인용으로 교체 | ☑ 2026.09.08 정독 — [§1.5](#15-clip-zero-shot-안전성--한계의-근거로-인용) |
| 3 | [XD-Violence (ECCV 2020)](https://roc-ng.github.io/XD-Violence/) | 데이터셋 구조·평가 프로토콜 파악 | ☑ 2026.09.08 확인 — [§3.1](#31-xd-violence--가장-잘-맞음) |
| 4 | [Flikcer](https://arxiv.org/pdf/2108.09491) | PSE 비교표의 근거. 이미 표는 작성됨 | ☐ |
| 5 | [TeSTra](https://arxiv.org/abs/2209.09236) | latency-정확도 트레이드오프 표현 방식 참고 | ☐ |
| 6 | [OAD Survey](https://www.sciencedirect.com/science/article/abs/pii/S0925231222003617) | 계열 안에서 우리 위치 잡기 | ☐ |
| 7 | [Video Anomaly Detection in 10 Years](https://arxiv.org/pdf/2405.19387) | 배경 서베이 | ☐ |

## 5. 미해결

- **Where's the Jump 스크래핑의 ToS 적법성** 미확인. 라벨 기준 참고용이라 소량 수동 열람으로
  대체 가능한지 먼저 판단한다
- **XD-Violence는 라이선스가 아예 명시돼 있지 않다.** 프로젝트 페이지에 이용 약관이 없어
  학술 사용 조건이 불분명하다. 논문에 실험 결과를 싣기 전 저자에게 메일로 확인한다
- **LIRIS-ACCEDE로 만든 모델을 확장으로 일반 배포할 수 있는지** 불명확. EULA가 학술 연구 전용이고
  상용 시스템 학습·테스트를 금지한다. 무료 배포가 "상용"인지 해석이 필요하므로 04단계 전에 판단한다

## 6. 논문에 쓸 수 있는 것 — 근거 카드

조사 결과를 "논문의 어느 자리에 어떻게 쓰는가"로 재배열한 것이다. 위 절들이 자료라면 이 절은 용도다.

### 6.1 설계 결정의 근거 — "왜 그 숫자냐"에 대한 답

| 결정 | 근거 | 출처 |
|---|---|---|
| **지연 3초** | online action detection에서 약 2~3초 버퍼로 THUMOS'14 mAP +13.8% 보고 | [Time Buffers](https://arxiv.org/abs/2010.03016) |
| **파인튜닝 대신 zero-shot** | 안전성 파인튜닝이 일반화 성능을 크게 떨어뜨린다는 보고 | [SafeR-CLIP](https://arxiv.org/abs/2511.16743) |
| **갑툭튀에 오디오 우선** | energy entropy 기반 급작스러운 큰 소리는 affective computing의 표준 특징 | [Violence Detection in Hollywood Movies](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0078506) |
| **사이렌은 임계값만** | AudioSet 온톨로지에 Siren·Screaming·Explosion·Gunshot 클래스가 이미 있음 | AudioSet / YAMNet |
| **04단계 목표를 F1 0.6대로** | 경량 zero-shot VLM의 문헌 성능이 0.55 안팎, 최고치도 GPT-4V 0.709 | [UnsafeBench](https://arxiv.org/pdf/2405.03486) |
| **DRM 콘텐츠 제외** | EME 규격상 픽셀·오디오 접근이 원천 차단 | 브라우저 규격 |
| **평가셋 직접 라벨링** | 공개 정답지가 전부 영화 기반이라 범위 선언과 충돌 | [§3.5](#35-결론--공개-데이터셋의-역할) |
| **대조 클립 30% 이상** | 사건 주변만 자르면 밀도가 과장돼 precision이 낙관적으로 나옴 | **자체 논리 — 외부 근거 없음** |
| **720p 무드롭 60초** | M1 조기 판정 기준 | **자체 설정** |

마지막 두 줄은 외부 근거가 없다. 심사에서 물어볼 수 있는 자리이므로,
대조 클립 비율은 클래스 불균형 문헌에서 근거를 더 찾아 붙이는 것이 좋다.

### 6.2 우리가 실제로 실험할 수 있는 것

| # | 실험 | 무엇을 보이나 | 필요한 것 | 단계 |
|---|---|---|---|---|
| **E1** | **지연 시간별 성능 곡선** (0 · 1 · 2 · 3 · 5초) | **"지연이 필요하다"의 직접 증거.** 0초에서 성능이 무너지면 주장이 증명된다 | 평가셋 + 규칙 탐지기 | 05 |
| **E3** | **카테고리별 성능** (갑툭튀 vs 지속 상태형) | **"lookahead로 분류해야 한다"의 직접 증거.** 지속 상태형은 지연 0에서도 잡히고 갑툭튀는 못 잡히면 분류가 증명된다 | 멀티카테고리 라벨 | 05 |
| E2 | 규칙기반 / 학습모델 / CLIP zero-shot / GPT-4o 4자 비교 | latency-정확도 트레이드오프. 논문 핵심 표 | 4개 경로 구현 | 03·05 |
| E4 | 720p·1080p 링버퍼 자원 측정 | 브라우저 클라이언트 실행 가능성의 경계 | M1 빌드 | 01 |
| E5 | 사용자 실험 — 원본 vs 필터 놀람 자가보고 | **개입이 실제로 효과가 있는가.** 기존 계열에 없는 유일한 산출 | 15~20명 | 05 |
| E6 | XD-Violence로 오디오 규칙 사전 검증 | 라벨링 착수 전 규칙 타당성 확인 | XD-Violence | 00~01 |

**E1과 E3가 논문의 두 주장에 각각 대응한다.** 나머지는 보강이다.
E1이 없으면 "3초가 필요하다"가 주장으로만 남고, E3가 없으면 lookahead 분류가 설계 취향이 된다.
일정이 밀려도 이 둘은 지킨다.

### 6.3 기존 연구가 가능성만 말하고 하지 않은 것

| 계열 | 언급한 것 | 하지 않은 이유·상태 | 우리가 하는 것 |
|---|---|---|---|
| [Time Buffers](https://arxiv.org/abs/2010.03016) | "라이브 웹캐스트에서 NSFW 탐지 시 실시간 red flag" | **동기로만 언급, 미구현.** red flag는 알림이지 차단이 아님 | 차단까지 구현하고 사용자 실험으로 검증 |
| affective computing | 공포·폭력 구간 탐지 | 목적이 **오프라인 분석·검색** | 재생 중 개입 |
| PSE 연구 | 사용자 도달 전 자극 완화 | **저수준 신호**(휘도 주파수)만 | 의미론적 사건 |
| 트리거 워닝 확장 | 트리거 콘텐츠 블러 | **픽셀을 보지 않음** (메타데이터·NLP) | 픽셀·오디오 직접 분석 |

**이 표가 논문 「관련 연구」 절의 뼈대다.** 각 계열이 우리 방향을 인정했으나 실행하지 않았다는 구조라,
참신성 주장이 "우리가 새롭다"가 아니라 **"문헌이 남긴 빈칸을 채운다"** 가 된다. 후자가 방어하기 쉽다.

### 6.4 결과가 나빠도 살아남는 문장

UnsafeBench가 현실적 상한을 알려준다 — 전체 최고가 GPT-4V **F1 0.709**, 경량 zero-shot은 **0.55 안팎**.
브라우저 안에서 도는 우리 CLIP은 후자에 가깝다.

따라서 04단계 목표를 **"확장 5종 F1 0.6대"** 로 잡는 것이 문헌 대비 합리적이고,
그보다 낮게 나와도 **"경량 zero-shot의 적용 한계를 측정했다"** 로 성립한다.
06단계 논문 프레이밍이 이미 결과 독립적으로 쓰여 있으므로, 여기에 수치 근거가 붙는 셈이다.
