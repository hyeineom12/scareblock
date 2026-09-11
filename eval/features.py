"""오디오 특징 — RMS · dRMS/dt · 스펙트럼 플럭스 · 직전 정적.

홉을 10 ms로 둔다. PoC의 50 ms 샘플링으로는 놀람 반사 문헌이 말하는
2~100 ms 상승 시간을 분해할 수 없었다 (poc/README.md 탐지 절).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

HOP_S = 0.010   # 10 ms — 상승 시간을 분해하려면 이 이하여야 한다
WIN_S = 0.025   # 25 ms


@dataclass
class Features:
    hop: float
    rms: np.ndarray          # 프레임별 실효값
    drms: np.ndarray         # dRMS/dt (초당 변화량) — 크기가 아니라 기울기
    flux: np.ndarray         # 스펙트럼 플럭스
    quiet: np.ndarray        # 직전 1초 RMS 중앙값
    peak: np.ndarray         # 감쇠하는 최근 최댓값 (볼륨 비종속 정규화용)

    def __len__(self) -> int:
        return len(self.rms)

    def t(self, i: int) -> float:
        return i * self.hop


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """모노 float32로 읽는다. mp4는 먼저 wav로 뽑는다 (eval/README.md 참고)."""
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = float(np.abs(x).max())
    if peak > 1.0:        # 정수 PCM으로 읽힌 경우 [-1, 1]로 맞춘다
        x /= peak
    return x, sr


def extract(x: np.ndarray, sr: int, hop_s: float = HOP_S, win_s: float = WIN_S) -> Features:
    hop = max(1, int(round(sr * hop_s)))
    win = max(hop, int(round(sr * win_s)))

    n = 1 + max(0, (len(x) - win)) // hop
    idx = np.arange(n) * hop
    frames = np.stack([x[i:i + win] for i in idx]) if n else np.zeros((0, win), np.float32)

    rms = np.sqrt((frames ** 2).mean(axis=1)) if n else np.zeros(0, np.float32)

    # 기울기 — 말소리는 천천히, 점프 스케어는 2~100 ms 만에 커진다
    drms = np.gradient(rms) / hop_s if n > 1 else np.zeros_like(rms)

    # 스펙트럼 플럭스 (양의 변화만)
    if n:
        _, _, Z = stft(x, fs=sr, nperseg=win, noverlap=win - hop, boundary=None, padded=False)
        mag = np.abs(Z)
        d = np.diff(mag, axis=1, prepend=mag[:, :1])
        flux = np.maximum(d, 0).sum(axis=0)
        flux = np.resize(flux, n)
    else:
        flux = np.zeros(0, np.float32)

    # 직전 1초 중앙값 — "조용했는가"
    w = max(1, int(round(1.0 / hop_s)))
    quiet = np.empty_like(rms)
    for i in range(n):
        lo = max(0, i - w)
        quiet[i] = np.median(rms[lo:i]) if i > lo else rms[i]

    # 감쇠 최댓값 — 볼륨 슬라이더에 종속되지 않게 한다 (PoC #5 리뷰 🟡3)
    peak = np.empty_like(rms)
    p = 0.0
    decay = 0.5 ** (hop_s / 35.0)   # 약 35초 반감
    for i in range(n):
        p = max(rms[i], p * decay)
        peak[i] = p

    return Features(hop=hop_s, rms=rms, drms=drms, flux=flux, quiet=quiet, peak=peak)
