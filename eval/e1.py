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
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from . import labels as L
from .detect import Detection, Params, detections_per_minute, run
from .evaluate import RENDER_S, match, score
from .features import extract, load_wav

LOOKAHEADS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]


def analyse(clip_audio: dict[str, Path], truth: list[L.Label], params: Params,
            lookaheads=LOOKAHEADS, render_s: float = RENDER_S):
    all_true, all_det, total_s = [], [], 0.0

    for clip_id, wav in sorted(clip_audio.items()):
        x, sr = load_wav(wav)
        f = extract(x, sr)
        # clip_id를 탐지에 심는다 — 매칭이 클립을 넘지 않게 하는 근거 (#19 리뷰 🔴2)
        dets = run(f, params, clip_id=clip_id)
        t = [y for y in truth if y.clip_id == clip_id and y.category == "jumpscare"]
        all_true += t
        all_det += dets
        total_s += len(x) / sr

    rows = [score(all_true, all_det, la, render_s=render_s) for la in lookaheads]
    return rows, detections_per_minute(all_det, total_s), total_s


def render(rows, per_min: float, total_s: float) -> str:
    head = (f"평가 구간 {total_s / 60:.1f}분 · 참 사건 {rows[0].n_true}건 · "
            f"탐지 {rows[0].n_pred}건 ({per_min:.1f}건/분)")
    if rows[0].onset_capped:
        head += f" · onset 되짚기 상한 {rows[0].onset_capped}건"
    if rows[0].render_s:
        head += f" · 렌더링 {rows[0].render_s:.2f}s 포함"
    out = [
        head,
        "",
        "| lookahead | 적시성 | precision | recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in rows:
        out.append(
            f"| {m.lookahead:.1f}s | **{m.timeliness:.3f}** | {m.precision:.3f} | "
            f"{m.recall:.3f} | {m.f1:.3f} | {m.tp} | {m.fp} | {m.fn} |"
        )
    out += ["", "적시성 = (라벨 onset → 확신) + 렌더링 시간 ≤ lookahead 인 참 사건의 비율."]
    return "\n".join(out)


# ── 합성 데이터 자체 점검 ────────────────────────────────────────────────
def _synth_clip(sr: int, dur: float, floor: float, onsets: list[float],
                crescendos: list[float], seed: int):
    """조용한/시끄러운 바닥 위에 2 ms 상승 점프 스케어와 크레셴도를 얹는다."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, floor, int(sr * dur)).astype(np.float32)
    for t0 in onsets:                       # 2 ms 상승 — 점프 스케어
        a, n = int(t0 * sr), int(0.6 * sr)
        env = np.ones(n, np.float32)
        env[: int(0.002 * sr)] = np.linspace(0, 1, int(0.002 * sr))
        env *= np.linspace(1, 0.2, n)
        x[a:a + n] += (rng.normal(0, 0.35, n) * env).astype(np.float32)
    for t0 in crescendos:                   # 1.5초에 걸쳐 서서히 — 잡히면 안 됨
        a, n = int(t0 * sr), int(1.5 * sr)
        x[a:a + n] += (rng.normal(0, 0.12, n) * np.linspace(0, 1, n)).astype(np.float32)
    return np.clip(x, -1, 1)


def _selftest() -> int:
    sr, dur = 16000, 30.0
    # 클립이 둘이어야 클립 간 교차 매칭(#19 리뷰 🔴2)이 회귀로 잡힌다.
    # c002는 바닥이 시끄러워 되짚기 상한(🟡3)도 함께 지나간다.
    specs = {
        "c001": dict(floor=0.004, onsets=[5.0, 12.0, 21.5], crescendos=[16.0], seed=0),
        "c002": dict(floor=0.020, onsets=[12.4, 24.0], crescendos=[], seed=1),
    }
    tmpdir = Path(tempfile.mkdtemp(prefix="eval-selftest-"))
    try:
        clip_audio, truth = {}, []
        for clip_id, spec in specs.items():
            wav = tmpdir / f"{clip_id}.wav"
            wavfile.write(wav, sr, _synth_clip(sr, dur, **spec))
            clip_audio[clip_id] = wav
            truth += [
                L.Label("synth", clip_id, 0.0, t0, t0 + 0.6, "jumpscare",
                        annotator="synthetic")
                for t0 in spec["onsets"]
            ]

        rows, per_min, total = analyse(clip_audio, truth, Params())
        print(render(rows, per_min, total))

        # 사건별 지연 — **라벨 onset 기준**이다 (#19 리뷰 🔴1)
        dets = []
        for clip_id, wav in sorted(clip_audio.items()):
            dets += run(extract(*load_wav(wav)), Params(), clip_id=clip_id)
        pairs, _, _ = match(truth, dets)
        lat = sorted(round(d.fire - t.onset, 3) for t, d in pairs)
        print(f"\n탐지 지연 (라벨 onset 기준): {lat}")
        print(f"되짚기 거리 (진단용): {sorted(round(d.backtrack_s, 3) for d in dets)}")

        # 🔴2 회귀 — 클립이 다르고 시각만 가까운 쌍은 절대 매칭되면 안 된다.
        # 전체 파이프라인에서는 5:5로 맞아떨어져 가드가 없어도 교차가 안 생기므로,
        # 가드를 직접 겨눈다.
        far = L.Label("synth", "cX", 0.0, 12.0, 12.6, "jumpscare")
        near = Detection(onset=12.05, fire=12.1, score=1.0, clip_id="cY")
        guarded = len(match([far], [near])[0])
        blind = len(match([far], [replace(near, clip_id="")])[0])
        no_cross = guarded == 0 and blind == 1 and all(
            t.clip_id == d.clip_id for t, d in pairs)
        print(f"클립 경계: 다른 클립 0.05초 차 매칭 {guarded}건 "
              f"(clip_id 무시 시 {blind}건) · 본 평가 교차 {'없음' if no_cross else '있음'}")

        ok = (
            rows[-1].recall >= 2 / 3
            and rows[0].timeliness < rows[-1].timeliness
            and no_cross
        )
        print("\n자체 점검:", "통과" if ok else "실패 — 파라미터 확인 필요")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 — lookahead 시간별 성능 곡선")
    ap.add_argument("--labels", help="docs/labeling-guide.md §1 스키마 CSV")
    ap.add_argument("--audio", help="clip_id.wav 들이 있는 디렉터리")
    ap.add_argument("--drms-min", type=float, default=0.0, help="dRMS/dt 하한 (0이면 끔)")
    ap.add_argument("--render-s", type=float, default=RENDER_S,
                    help="블러 렌더링 시간(초). M1 실측값을 넣으면 적시성에 더해진다")
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
        {k: v for k, v in audio.items() if k in clips}, truth,
        Params(drms_min=a.drms_min), render_s=a.render_s,
    )
    print(render(rows, per_min, total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
