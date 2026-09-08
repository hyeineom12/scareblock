# 관련 연구와 데이터셋 조사

00단계 산출물. README 「관련 연구와 차별점」 절의 근거 문서이고,
09.10 제안서의 관련연구 절은 이 문서에서 추린다.

- 조사일 — 2026.09.08
- 담당 — 개발 B (이수현)
- 범위 — 지연 버퍼 계열, 갑툭튀 오디오 특징, 트리거 워닝 상용 제품, CLIP zero-shot 안전성, 평가 데이터셋

## 요약 — 이 조사로 바뀐 것

1. **차별점 문장을 다시 좁혀야 한다.** 라이브 스트리밍의 송출 지연 버퍼를 탐지에 활용하는
   연구가 이미 있다([Time Buffers](https://arxiv.org/abs/2010.03016)). README가 PSE 비교 뒤에
   결론으로 쓴 "의미론적 사건을 lookahead로 잡는다가 새롭다"는 이 논문이 이미 하는 일이다.
   → 남는 주장은 **"있는 버퍼를 쓴 것이 아니라, 개입을 위해 없던 지연을 만들었다"** 이다. [§1.2](#12-버퍼-시간을-탐지에-활용--online-action-detection에-선례-있음)
2. **평가셋으로 쓸 수 있는 공개 데이터는 사실상 XD-Violence 하나다.** 나머지는 전부 영화 기반이라
   [범위 선언](../README.md#범위-선언)의 DRM 제외에 걸린다. 공개 데이터셋의 역할을
   "평가셋"이 아니라 "라벨 기준 정의 · 특징 검증 · 사전학습"으로 못박는다. [§3.5](#35-결론--공개-데이터셋의-역할)
3. **LIRIS-ACCEDE는 EULA 신청이 필요해 리드타임이 있다.** 00단계에 바로 신청을 건다. [§3.2](#32-liris-accede--mediaeval-emotional-impact-of-movies)

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

**이번 조사에서 새로 발견한 계열이고, 지금까지 확인된 것 중 주장이 가장 크게 겹친다.**

- [Online Action Detection in Streaming Videos with Time Buffers](https://arxiv.org/abs/2010.03016) (Zhang et al., 2020)
  — 라이브 스트리밍에는 최신 캡처 프레임과 시청 프레임 사이에 송출 지연이 존재한다는 점에 착안해,
  그 **버퍼 시간을 탐지에 활용하는 문제 설정**을 제안한다. flattened I3D + window-based suppression으로
  표준 temporal action detection 벤치마크 3종에서 기존 online 모델 대비 정확도 향상을 보인다
- [Real-time Online Video Detection with Temporal Smoothing Transformers (TeSTra)](https://arxiv.org/abs/2209.09236) — 실시간 online detection의 latency-정확도 트레이드오프
- [Temporally smooth online action detection using cycle-consistent future anticipation (FATSnet)](https://www.sciencedirect.com/science/article/abs/pii/S0031320321001412)
- [Online human action detection and anticipation in videos: A survey](https://www.sciencedirect.com/science/article/abs/pii/S0925231222003617) — 계열 전체 지도

**왜 문제인가.** README는 PSE 비교표 다음에 "버퍼를 쓴다는 사실은 새롭지 않다.
의미론적 사건을 lookahead로 잡는다가 새롭다"로 결론을 냈다. Time Buffers 논문이 정확히 그것을 한다 —
버퍼 안에서 의미론적 사건(action)을 탐지한다. PSE 비교표에서 "저수준 신호 vs 의미론적 사건"으로
갈라놓은 그 쪽에 이 논문이 이미 서 있다.

**그래도 남는 차이.**

| | Time Buffers (2020) | 본 연구 |
|---|---|---|
| 버퍼의 출처 | 라이브 방송에 **이미 존재하는** 송출 지연 | VOD에 **의도적으로 도입하는** 지연 (시청 지연을 비용으로 지불) |
| 목적 | 탐지 **정확도 향상** | 탐지 후 **사용자 도달 전 개입** |
| 실행 위치 | 서버·오프라인 규모 모델(I3D) | **브라우저 클라이언트 실시간** |
| 산출 | 벤치마크 점수 | 동작하는 확장 + 사용자 실험 |

→ **주장을 "버퍼 활용"이 아니라 "개입을 위해 없던 지연을 만들어낸 설계 결정"에 둔다.**
공짜로 있는 버퍼를 이용한 것과, 사용자에게 3초를 손해 보게 하면서까지 lookahead를 확보한 것은
설계 문제로서 다르다. 이쪽이 오히려 방어하기 쉽다.

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

- [UnsafeBench: Benchmarking Image Safety Classifiers on Real-World and AI-Generated Images](https://arxiv.org/pdf/2405.03486) — 이미지 안전성 분류기 벤치마크
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
| **재생 중 개입 (online intervention)** | ✅ | 기존 계열의 목적은 정확도·검색이지 차단이 아님 |
| **lookahead 요구량 기준 아키텍처 분류** | ✅ | 선행 사례 확인되지 않음 |
| **브라우저 내 실행 × 다중 카테고리 × zero-shot 확장의 조합** | ✅ | 조합 자체가 공백 |

## 3. 평가 데이터 후보

### 3.1 XD-Violence — 가장 잘 맞음

[프로젝트 페이지](https://roc-ng.github.io/XD-Violence/) · [ECCV 2020 "Not only Look, but also Listen"](https://roc-ng.github.io/XD-Violence/)

- 4,754개 untrimmed 영상, 총 217시간, **오디오 포함**, video-level weak label
- 학습 3,954개(폭력 1,905 / 비폭력 2,049), 테스트 800개(500 / 300)
- 6개 클래스 — Abuse, Car Accident, Explosion, Fighting, Riot, Shooting
- 출처가 영화 **및 유튜브**라 우리 범위와 부분적으로 겹친다
- RGB · 오디오 · flow 3개 모달리티 제공

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

**제약** — InterDigital EULA 서명이 필요하다. **리드타임이 있으므로 00단계에 바로 신청한다.**

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
| 1 | [Time Buffers](https://arxiv.org/abs/2010.03016) | 차별점 문장이 여기 걸림. 09.10 제안서 전 필수 | ☐ |
| 2 | [UnsafeBench](https://arxiv.org/pdf/2405.03486) | 위험 표의 CLIP 정밀도 행을 인용으로 교체 | ☐ |
| 3 | [XD-Violence (ECCV 2020)](https://roc-ng.github.io/XD-Violence/) | 데이터셋 구조·평가 프로토콜 파악 | ☐ |
| 4 | [Flikcer](https://arxiv.org/pdf/2108.09491) | PSE 비교표의 근거. 이미 표는 작성됨 | ☐ |
| 5 | [TeSTra](https://arxiv.org/abs/2209.09236) | latency-정확도 트레이드오프 표현 방식 참고 | ☐ |
| 6 | [OAD Survey](https://www.sciencedirect.com/science/article/abs/pii/S0925231222003617) | 계열 안에서 우리 위치 잡기 | ☐ |
| 7 | [Video Anomaly Detection in 10 Years](https://arxiv.org/pdf/2405.19387) | 배경 서베이 | ☐ |

## 5. 미해결

- **Time Buffers 논문이 lookahead 요구량 기준 분류에 해당하는 개념을 쓰는지** 정독 전까지 확정 불가.
  쓰지 않는 것으로 보이나, 쓴다면 [§2](#2-차별점-정리-갱신)의 마지막 ✅ 두 줄 중 하나가 무너진다
- **Where's the Jump 스크래핑의 ToS 적법성** 미확인. 라벨 기준 참고용이라 소량 수동 열람으로
  대체 가능한지 먼저 판단한다
- **XD-Violence 라이선스와 재배포 조건** 미확인. 06단계 재현 절차의
  "URL + 타임스탬프 + 라벨만 공개" 방침과 충돌하지 않는지 확인 필요
