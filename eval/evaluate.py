"""매칭과 지표 — P/R/F1 + 적시성.

적시성이 이 하니스의 존재 이유다. 탐지가 맞았는지가 아니라
**사용자에게 도달하기 전에 준비됐는지**를 센다.
"""
from __future__ import annotations

from dataclasses import dataclass

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
        """
        return self.in_time / self.n_true if self.n_true else 0.0


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
    )
