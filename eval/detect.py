"""규칙 기반 점프 스케어 탐지기.

핵심은 **탐지 지연을 사건마다 기록**하는 것이다. E1이 그 값으로
lookahead L을 쓸어가며 "도달 전 개입이 되는가"를 판정한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import Features


@dataclass
class Params:
    quiet_of_peak: float = 0.15   # 직전 1초가 최근 최댓값 대비 이만큼 아래면 정적
    surge_of_quiet: float = 5.0   # 그 정적 대비 이만큼 뛰면 급등
    surge_of_peak: float = 0.30   # 최근 최댓값 대비 최소치 (잔물결 배제)
    drms_min: float = 0.0         # dRMS/dt 하한 — 0이면 기울기 조건을 끈다
    cooldown_s: float = 2.0
    silence_floor: float = 1e-4   # 무음 구간 판정 건너뛰기


@dataclass
class Detection:
    onset: float      # 자극이 올라가기 시작한 추정 시각
    fire: float       # 규칙이 확신에 이른 시각
    score: float

    @property
    def latency(self) -> float:
        """onset부터 확신까지 걸린 시간. E1의 L과 비교되는 값이다."""
        return self.fire - self.onset


def _backtrack_onset(f: Features, i: int) -> float:
    """RMS가 정적 수준을 벗어나기 시작한 프레임까지 되짚는다.

    발화 시점이 아니라 시작점을 onset으로 삼아야 적시성이 의미를 갖는다
    (labeling-guide §3 — onset은 정점이 아니라 시작점).
    """
    floor = max(f.quiet[i] * 1.5, f.peak[i] * 0.02)
    j = i
    while j > 0 and f.rms[j] > floor:
        j -= 1
    return f.t(j)


def run(f: Features, p: Params | None = None) -> list[Detection]:
    p = p or Params()
    out: list[Detection] = []
    cooldown_frames = int(round(p.cooldown_s / f.hop))
    blocked_until = -1

    for i in range(len(f)):
        if i < blocked_until or f.peak[i] < p.silence_floor:
            continue
        quiet, cur, peak = f.quiet[i], f.rms[i], f.peak[i]
        if not (quiet < peak * p.quiet_of_peak):
            continue
        if not (cur > max(quiet * p.surge_of_quiet, peak * p.surge_of_peak)):
            continue
        if p.drms_min > 0 and f.drms[i] < p.drms_min:
            continue

        onset = _backtrack_onset(f, i)
        out.append(Detection(onset=onset, fire=f.t(i), score=float(cur / (peak + 1e-9))))
        blocked_until = i + cooldown_frames

    return out


def detections_per_minute(dets: list[Detection], duration_s: float) -> float:
    """PoC 실측(일반 영상 5.2건/분, 사건 빽빽 7.9건/분)과 같은 축의 값."""
    return len(dets) / (duration_s / 60.0) if duration_s > 0 else 0.0
