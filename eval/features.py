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
    win: float               # 분석 창 길이 — 확신 시각은 창의 **끝**이다
    rms: np.ndarray          # 프레임별 실효값
    drms: np.ndarray         # dRMS/dt (초당 변화량) — 크기가 아니라 기울기
    flux: np.ndarray         # 스펙트럼 플럭스
    quiet: np.ndarray        # 직전 1초 RMS 중앙값
    peak: np.ndarray         # 감쇠하는 최근 최댓값 (볼륨 비종속 정규화용)

    def __len__(self) -> int:
        return len(self.rms)

    def t(self, i: int) -> float:
        """프레임 i의 창 **시작** 시각. onset 추정처럼 '언제부터'를 볼 때 쓴다."""
        return i * self.hop

    def t_ready(self, i: int) -> float:
        """프레임 i의 특징을 **쓸 수 있게 되는** 시각 = 창의 끝.

        창 시작을 확신 시각으로 쓰면 창 안의 자극을 창이 시작할 때 이미 알았다는
        뜻이 되어 지연이 음수로 나온다(#19 리뷰 🔴1을 고치고 나서 드러났다).
        실시간에서 프레임을 계산할 수 있는 가장 이른 시각은 창의 끝이다.
        """
        return i * self.hop + self.win


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """모노 float32로 읽는다. mp4는 먼저 wav로 뽑는다 (eval/README.md 참고).

    정규화는 **dtype 범위**로 한다. 클립의 실제 최댓값으로 나누면 조용한
    클립일수록 크게 증폭돼, 절대값인 `silence_floor`가 클립마다 다른 음압을
    가리키게 된다 (#19 리뷰 🟡5).
    """
    sr, x = wavfile.read(path)
    scale = 1.0
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:               # int16 · int32 — 대칭 PCM
            scale = float(-info.min)
        else:                          # uint8 — 128을 0으로 옮긴다
            mid = (float(info.max) + 1.0) / 2.0
            x = x.astype(np.float32) - mid
            scale = mid
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if scale != 1.0:
        x /= scale
    return x, sr


def extract(x: np.ndarray, sr: int, hop_s: float = HOP_S, win_s: float = WIN_S) -> Features:
    hop = max(1, int(round(sr * hop_s)))
    win = max(hop, int(round(sr * win_s)))

    n = 1 + max(0, (len(x) - win)) // hop
    idx = np.arange(n) * hop
    frames = np.stack([x[i:i + win] for i in idx]) if n else np.zeros((0, win), np.float32)

    rms = np.sqrt((frames ** 2).mean(axis=1)) if n else np.zeros(0, np.float32)

    # 기울기 — 말소리는 천천히, 점프 스케어는 2~100 ms 만에 커진다.
    # **후방차분이어야 한다.** np.gradient는 중심차분이라 drms[i]가 rms[i+1]에
    # 의존하는데, 확신 시각은 t_ready(i)로 적힌다 — 아직 나오지 않은 프레임으로
    # 확신했다고 기록하는 셈이다 (#19 리뷰 🟡3). 창 시작/끝 문제와 같은 종류다.
    drms = np.diff(rms, prepend=rms[:1]) / hop_s if n else np.zeros_like(rms)

    # 스펙트럼 플럭스 (양의 변화만)
    if n:
        _, _, Z = stft(x, fs=sr, nperseg=win, noverlap=win - hop, boundary=None, padded=False)
        mag = np.abs(Z)
        d = np.diff(mag, axis=1, prepend=mag[:, :1])
        flux = np.maximum(d, 0).sum(axis=0)
        # 길이 맞추기 — np.resize는 짧으면 앞 구간을 **반복해 채운다**.
        # 자르고 0으로 패딩해야 뒤쪽에 가짜 값이 들어가지 않는다 (#19 리뷰 🟡4).
        if len(flux) >= n:
            flux = flux[:n]
        else:
            flux = np.pad(flux, (0, n - len(flux)))
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

    return Features(hop=hop_s, win=win / sr, rms=rms, drms=drms, flux=flux,
                    quiet=quiet, peak=peak)
