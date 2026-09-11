"""E1 — lookahead 시간별 성능 곡선. 논문의 표 하나.

사용:
    python3 -m eval.e1 --labels labels.csv --audio _local/wav
    python3 -m eval.e1 --selftest          # 합성 오디오로 파이프라인 점검

주장: lookahead 0에서 적시성이 무너지면 "지연이 필요하다"가 증명된다.
곡선이 1초 근처에서 평평해지면 3초는 과잉이라는 뜻이고, 그때는
"필요한 양은 3초가 아니라 X초"로 쓴다 — 그쪽이 더 강한 결과다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import labels as L
from .detect import Params, detections_per_minute, run
from .evaluate import score
from .features import extract, load_wav

LOOKAHEADS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]


def analyse(clip_audio: dict[str, Path], truth: list[L.Label], params: Params,
            lookaheads=LOOKAHEADS):
    all_true, all_det, total_s = [], [], 0.0

    for clip_id, wav in sorted(clip_audio.items()):
        x, sr = load_wav(wav)
        f = extract(x, sr)
        dets = run(f, params)
        t = [y for y in truth if y.clip_id == clip_id and y.category == "jumpscare"]
        all_true += t
        all_det += dets
        total_s += len(x) / sr

    rows = [score(all_true, all_det, la) for la in lookaheads]
    return rows, detections_per_minute(all_det, total_s), total_s


def render(rows, per_min: float, total_s: float) -> str:
    out = [
        f"평가 구간 {total_s / 60:.1f}분 · 참 사건 {rows[0].n_true}건 · "
        f"탐지 {rows[0].n_pred}건 ({per_min:.1f}건/분)",
        "",
        "| lookahead | 적시성 | precision | recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in rows:
        out.append(
            f"| {m.lookahead:.1f}s | **{m.timeliness:.3f}** | {m.precision:.3f} | "
            f"{m.recall:.3f} | {m.f1:.3f} | {m.tp} | {m.fp} | {m.fn} |"
        )
    out += ["", "적시성 = 사용자 도달 전에 개입이 준비된 참 사건의 비율."]
    return "\n".join(out)


# ── 합성 데이터 자체 점검 ────────────────────────────────────────────────
def _selftest() -> int:
    import numpy as np
    from scipy.io import wavfile

    sr, dur = 16000, 30.0
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.004, int(sr * dur)).astype(np.float32)  # 조용한 바닥

    onsets = [5.0, 12.0, 21.5]
    for t0 in onsets:                       # 2 ms 상승 — 점프 스케어
        a, n = int(t0 * sr), int(0.6 * sr)
        env = np.ones(n, np.float32)
        env[: int(0.002 * sr)] = np.linspace(0, 1, int(0.002 * sr))
        env *= np.linspace(1, 0.2, n)
        x[a:a + n] += (rng.normal(0, 0.35, n) * env).astype(np.float32)

    for t0 in [16.0]:                       # 1.5초에 걸쳐 서서히 — 크레셴도, 잡히면 안 됨
        a, n = int(t0 * sr), int(1.5 * sr)
        x[a:a + n] += (rng.normal(0, 0.12, n) * np.linspace(0, 1, n)).astype(np.float32)

    tmp = Path("_selftest.wav")
    wavfile.write(tmp, sr, np.clip(x, -1, 1))
    try:
        truth = [
            L.Label("synth", "c001", 0.0, t0, t0 + 0.6, "jumpscare", annotator="synthetic")
            for t0 in onsets
        ]
        rows, per_min, total = analyse({"c001": tmp}, truth, Params())
        print(render(rows, per_min, total))

        lat = [d.latency for d in run(extract(*load_wav(tmp)), Params())]
        print(f"\n탐지 지연: {[round(v, 3) for v in lat]}")

        ok = rows[-1].recall >= 2 / 3 and rows[0].timeliness < rows[-1].timeliness
        print("\n자체 점검:", "통과" if ok else "실패 — 파라미터 확인 필요")
        return 0 if ok else 1
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 — lookahead 시간별 성능 곡선")
    ap.add_argument("--labels", help="docs/labeling-guide.md §1 스키마 CSV")
    ap.add_argument("--audio", help="clip_id.wav 들이 있는 디렉터리")
    ap.add_argument("--drms-min", type=float, default=0.0, help="dRMS/dt 하한 (0이면 끔)")
    ap.add_argument("--selftest", action="store_true", help="합성 오디오로 파이프라인 점검")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not (a.labels and a.audio):
        ap.error("--labels 와 --audio 가 필요하다 (또는 --selftest)")

    truth = L.load(a.labels)
    problems = L.validate(truth)
    if problems:
        print("라벨 규약 위반:", *problems, sep="\n  ", file=sys.stderr)

    clips = L.by_clip(truth)
    ratio = L.control_ratio(clips)
    if ratio < 0.30:
        print(f"경고: 대조 클립 비율 {ratio:.0%} — §4 기준 30% 미만이라 "
              f"정밀도 수치를 방어할 수 없다", file=sys.stderr)

    audio = {p.stem: p for p in Path(a.audio).glob("*.wav")}
    missing = set(clips) - set(audio)
    if missing:
        print(f"경고: wav 없는 클립 {sorted(missing)}", file=sys.stderr)

    rows, per_min, total = analyse(
        {k: v for k, v in audio.items() if k in clips}, truth, Params(drms_min=a.drms_min)
    )
    print(render(rows, per_min, total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
