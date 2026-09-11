"""매칭과 지표 — P/R/F1 + 적시성.

적시성이 이 하니스의 존재 이유다. 탐지가 맞았는지가 아니라
**사용자에게 도달하기 전에 준비됐는지**를 센다.
"""
from __future__ import annotations

from dataclasses import dataclass

from .detect import Detection
from .labels import Label

MATCH_TOL = 0.5  # labeling-guide §6 — onset이 ±0.5초 안이면 같은 사건


@dataclass
class Metrics:
    lookahead: float
    n_true: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    in_time: int          # 매칭됐고 지연이 lookahead 안에 들어온 것

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
        """도달 전 개입 성공률. lookahead 0에서 무너져야 주장이 증명된다."""
        return self.in_time / self.n_true if self.n_true else 0.0


def match(truth: list[Label], dets: list[Detection], tol: float = MATCH_TOL):
    """onset 근접도로 1:1 매칭한다. 가까운 쌍부터 탐욕적으로 묶는다."""
    pairs: list[tuple[Label, Detection]] = []
    used_t, used_d = set(), set()
    cand = sorted(
        (
            (abs(t.onset - d.onset), ti, di)
            for ti, t in enumerate(truth)
            for di, d in enumerate(dets)
            if abs(t.onset - d.onset) <= tol
        )
    )
    for _, ti, di in cand:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        pairs.append((truth[ti], dets[di]))
    return pairs, used_t, used_d


def score(truth: list[Label], dets: list[Detection], lookahead: float,
          tol: float = MATCH_TOL) -> Metrics:
    pairs, used_t, used_d = match(truth, dets, tol)
    in_time = sum(1 for _, d in pairs if d.latency <= lookahead)
    return Metrics(
        lookahead=lookahead,
        n_true=len(truth),
        n_pred=len(dets),
        tp=len(pairs),
        fp=len(dets) - len(used_d),
        fn=len(truth) - len(used_t),
        in_time=in_time,
    )
