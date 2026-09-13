"""매칭과 지표 — P/R/F1 + 적시성.

적시성이 이 하니스의 존재 이유다. 탐지가 맞았는지가 아니라
**사용자에게 도달하기 전에 준비됐는지**를 센다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import Detection
from .labels import Label

MATCH_TOL = 0.5   # labeling-guide §6 — onset이 ±0.5초 안이면 같은 사건
RENDER_S = 0.0    # 블러 렌더링 시간. M1 실측 뒤 이 상수만 올리면 표가 다시 나온다


@dataclass
class Metrics:
    lookahead: float
    n_true: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    in_time: int          # 매칭됐고 **라벨 onset 기준** 지연이 lookahead 안인 것
    render_s: float = 0.0   # 적시성에 더한 블러 렌더링 시간 (M1 전에는 0)
    onset_capped: int = 0   # 되짚기 상한에 걸린 탐지 수 — onset 추정 실패 빈도
    # 매칭된 사건마다 `확신 시각 − 라벨 onset`(렌더링 제외). lookahead와 무관하게 같다.
    # **E1의 헤드라인은 이 분포다** — 중앙값·p90·최댓값
    latencies: tuple[float, ...] = ()

    @property
    def precision(self) -> float:
        return self.tp / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_true if self.n_true else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def timeliness(self) -> float:
        """도달 전 개입 성공률.

        분자는 **`라벨 onset → 확신` + 렌더링 시간 ≤ lookahead**인 참 사건 수다.
        탐지기가 되짚어 만든 onset을 쓰면 자기 자신과 비교하게 되고, 지연이
        되짚기가 멈춘 거리로 바뀐다 (#19 리뷰 🔴1).

        **적시성 = recall × 잡은 것 중 제때.** 분자는 매칭된 사건 중에서만 세므로
        탐지를 못 한 사건도 「늦음」으로 들어가고, 그래서 적시성은 recall을 넘을 수
        없다(항등식). 「잡았는가」와 「제때인가」를 가르려면 `timely_of_detected`를 본다.
        """
        return self.in_time / self.n_true if self.n_true else 0.0

    @property
    def timely_of_detected(self) -> float:
        """잡은 사건 중 제때 준비된 비율 = in_time / TP.

        lookahead별로 읽으면 곧 **탐지 지연(+렌더링)의 누적 분포(CDF)** 다 — 곡선과
        헤드라인(지연 분포)이 같은 것을 가리킨다.
        """
        return self.in_time / self.tp if self.tp else 0.0


def _same_clip(t: Label, d: Detection) -> bool:
    """클립을 섞지 않는다. 모든 클립의 시간축이 0에서 시작하므로 합쳐서 매칭하면
    클립 A의 탐지가 클립 B의 라벨과 붙는다 (#19 리뷰 🔴2).
    `clip_id`가 비어 있으면(단일 클립 호출) 검사하지 않는다."""
    return not d.clip_id or d.clip_id == t.clip_id


def match(truth: list[Label], dets: list[Detection], tol: float = MATCH_TOL):
    """onset 근접도로 1:1 매칭한다. 같은 클립 안에서, 가까운 쌍부터 탐욕적으로."""
    pairs: list[tuple[Label, Detection]] = []
    used_t, used_d = set(), set()
    cand = sorted(
        (
            (abs(t.onset - d.onset), ti, di)
            for ti, t in enumerate(truth)
            for di, d in enumerate(dets)
            if _same_clip(t, d) and abs(t.onset - d.onset) <= tol
        )
    )
    for _, ti, di in cand:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        pairs.append((truth[ti], dets[di]))
    return pairs, used_t, used_d


def latency_summary(latencies) -> dict | None:
    """탐지 지연 분포의 요약 — E1의 헤드라인. 매칭이 없으면 None.

    `L = 0` 열은 확신 시각을 분석 창의 끝으로 정의했기 때문에 **정의상** 0이다.
    그래서 결과로 부를 수 있는 것은 이 분포다 — 「필요한 지연량」은 p90으로 읽는다.
    사건 수가 적으면 p90은 보간값이라 n을 함께 낸다.
    """
    xs = np.asarray(sorted(latencies), dtype=float)
    if xs.size == 0:
        return None
    return {"n": int(xs.size), "median": float(np.median(xs)),
            "p90": float(np.percentile(xs, 90)), "max": float(xs.max())}


def score(truth: list[Label], dets: list[Detection], lookahead: float,
          tol: float = MATCH_TOL, render_s: float = RENDER_S) -> Metrics:
    pairs, used_t, used_d = match(truth, dets, tol)
    # 사용자가 노출되는 시점은 **라벨 onset**이다. 탐지기의 추정치가 아니다.
    in_time = sum(1 for t, d in pairs if (d.fire - t.onset) + render_s <= lookahead)
    return Metrics(
        lookahead=lookahead,
        n_true=len(truth),
        n_pred=len(dets),
        tp=len(pairs),
        fp=len(dets) - len(used_d),
        fn=len(truth) - len(used_t),
        in_time=in_time,
        render_s=render_s,
        onset_capped=sum(1 for d in dets if d.onset_capped),
        latencies=tuple(sorted(d.fire - t.onset for t, d in pairs)),
    )
