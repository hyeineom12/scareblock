"""E1 — 탐지 지연 분포와 lookahead 시간별 성능 곡선. 발표(11.05)의 결과 표.

사용:
    python3 -m eval.e1 --labels labels.csv --audio _local/wav
    python3 -m eval.e1 --selftest          # 합성 오디오로 파이프라인 점검

헤드라인은 **탐지 지연 분포**(라벨 onset → 확신; 중앙값·p90·최댓값)다.
lookahead 0 열의 적시성은 라벨 onset이 상승 시작점이면 0이다(확신 시각은 분석 창의 끝) —
결과가 아니다. 라벨이 정점 근처에 찍히면 지연이 음수가 되어 0이 아니게 되므로 음수 건수를 함께 낸다.
p90이 1초 근처면 "필요한 양은 3초가 아니라 X초"로 쓴다.
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
from .evaluate import RENDER_S, latency_summary, match, score
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


def evaluation_set(truth: list[L.Label], clip_ids: list[str], audio: dict[str, Path]):
    """실제로 평가할 집합을 정한다. (라벨, 클립별 wav, 명세 밖 라벨 clip_id, wav 없는 clip_id,
    명세 밖 wav) 를 돌려준다.

    분모는 **명세 ∩ wav**다. 대조 비율도 이 집합으로 계산해야 출력된 표의 precision을
    방어한다 — 명세 전체로 재면 유일한 대조 클립의 wav가 없을 때 경고가 침묵한다
    (#26 재리뷰 🟡2).

    **명세 밖 clip_id의 라벨은 알리고 뺀다.** analyse가 wav 기준으로 라벨을 거르므로
    조용히 두면 참 사건이 분모에서 사라져 recall이 부풀고, by_clip이 그 clip_id를
    대조 비율 분모에 넣어 30% 게이트까지 뒤집는다 (#26 재리뷰 🔴1). 분모를 라벨에서
    역산하지 말라는 🔴1의 거울쪽이다.
    """
    spec = set(clip_ids)
    stray = sorted({x.clip_id for x in truth} - spec)
    missing = sorted(spec - set(audio))
    extra = sorted(set(audio) - spec)
    used = {k: v for k, v in audio.items() if k in spec}
    kept = [x for x in truth if x.clip_id in used]
    return kept, used, stray, missing, extra


def render(rows, per_min: float, total_s: float,
           n_used: int | None = None, n_expected: int | None = None,
           annotator: str | None = None, has_spec: bool = True) -> str:
    head = ""
    # 표를 옮길 때 같이 따라가는 자리에 **이 표의 성격**을 둔다. 경고는 stderr로 가고
    # 표만 논문에 옮겨지므로, 누구 라벨인지·클립이 다 들어갔는지·명세가 있었는지가
    # 머리줄에 없으면 사라진다 (#26 리뷰 🟡2 · 재리뷰 🟡3·🟡4).
    if annotator:
        head += f"주석자 {annotator} · "
    if n_expected is not None:
        if not has_spec:
            # 폴백은 wav 목록이 곧 분모라 n_used == n_expected가 항상 성립한다.
            # 비교가 무의미하므로 명세가 없었다는 사실 자체를 적는다.
            head += f"클립 {n_used}개 (명세 없음)"
        else:
            head += f"클립 {n_used}/{n_expected}개"
            if n_used != n_expected:
                head += " ⚠️ 명세보다 적다"
        head += " · "
    head += (f"평가 구간 {total_s / 60:.1f}분 · 참 사건 {rows[0].n_true}건 · "
             f"탐지 {rows[0].n_pred}건 ({per_min:.1f}건/분)")
    if rows[0].onset_capped:
        head += f" · onset 되짚기 상한 {rows[0].onset_capped}건"
    if rows[0].render_s:
        head += f" · 렌더링 {rows[0].render_s:.2f}s 포함"
    # 헤드라인 — 탐지 지연 분포. 머리줄(표의 성격) 바로 다음, 표보다 먼저 둔다.
    # lookahead 0 열은 정의상 0이라 결과로 부를 수 있는 것은 이 분포다(6e 분석, 사용자 결정 09.13)
    s = latency_summary(rows[0].latencies)
    render_s = rows[0].render_s
    if s is None:
        headline = ["**탐지 지연** — 매칭된 사건이 없어 분포를 낼 수 없다"]
    else:
        need = s["p90"] + render_s
        # 렌더링 시간은 이미 need에 들어 있다. 「+ 렌더링」으로 쓰면 두 번 더한 것으로 읽힌다(#33 리뷰 🟡3)
        tail = f" (렌더링 {render_s:.2f} s 포함)" if render_s else " (렌더링 시간 미측정)"
        headline = [
            f"**탐지 지연** (라벨 onset → 확신, 매칭 {s['n']}건) — "
            f"중앙값 {s['median']:.3f} s · p90 {s['p90']:.3f} s · 최댓값 {s['max']:.3f} s",
            # 음수 p90이 「필요한 지연량 −0.25 s」로 헤드라인에 나가지 않게 바닥을 둔다(#33 리뷰 🔴1)
            f"→ 필요한 지연량(p90 기준) ≈ {max(need, 0.0):.3f} s{tail}"
            + (" · 사건 10건 미만이라 p90은 보간값" if s["n"] < 10 else ""),
        ]
        if s["negative"]:
            headline.append(
                f"⚠ 음수 지연 {s['negative']}건 — 라벨 onset이 탐지기 확신보다 늦게 찍혔다. "
                f"정점 근처를 찍었는지 라벨을 확인한다(labeling-guide §3)"
                + (" · 필요한 지연량은 0으로 바닥을 뒀다" if need < 0 else ""))
    out = [
        head,
        *headline,
        "",
        "| lookahead | 적시성 | 잡은 것 중 제때 | precision | recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in rows:
        out.append(
            f"| {m.lookahead:.1f}s | **{m.timeliness:.3f}** | {m.timely_of_detected:.3f} | "
            f"{m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} | {m.tp} | {m.fp} | {m.fn} |"
        )
    out += [
        "",
        "적시성 = (라벨 onset → 확신) + 렌더링 시간 ≤ lookahead 인 참 사건의 비율.",
        "적시성 = recall × 잡은 것 중 제때 — 못 잡은 사건도 「늦음」으로 세지므로 recall을 넘을 수 없다.",
        "잡은 것 중 제때 = in_time / TP — lookahead별로 읽으면 탐지 지연 분포의 누적 분포다.",
        "lookahead 0 열은 라벨 onset이 상승 시작점이면 0이다(확신은 분석 창의 끝) — 결과가 아니다. "
        "0보다 크면 라벨이 늦게 찍힌 사건이 섞였다.",
    ]
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
    # c003은 **대조 클립**이다 — 소리는 나지만 라벨이 0건이다. 라벨을 분모로 쓰면
    # 이 클립이 통째로 빠져 오탐이 한 건도 안 세진다 (#19 리뷰 🔴1).
    specs = {
        "c001": dict(floor=0.004, onsets=[5.0, 12.0, 21.5], crescendos=[16.0], seed=0),
        "c002": dict(floor=0.020, onsets=[12.4, 24.0], crescendos=[], seed=1),
        "c003": dict(floor=0.004, onsets=[6.0, 14.0, 23.0], crescendos=[], seed=2),
    }
    CONTROL = {"c003"}
    tmpdir = Path(tempfile.mkdtemp(prefix="eval-selftest-"))
    try:
        clip_audio, truth = {}, []
        for clip_id, spec in specs.items():
            wav = tmpdir / f"{clip_id}.wav"
            wavfile.write(wav, sr, _synth_clip(sr, dur, **spec))
            clip_audio[clip_id] = wav
            if clip_id in CONTROL:
                continue                     # 대조 클립은 라벨 행을 남기지 않는다
            truth += [
                L.Label("synth", clip_id, 0.0, t0, t0 + 0.6, "jumpscare",
                        annotator="synthetic")
                for t0 in spec["onsets"]
            ]

        rows, per_min, total = analyse(clip_audio, truth, Params())
        print(render(rows, per_min, total,
                     n_used=len(clip_audio), n_expected=len(specs)))

        # 🔴1 회귀 — 분모를 라벨에서 역산하면 대조 클립이 사라진다.
        from_labels = L.by_clip(truth)                       # 옛 동작
        from_spec = L.by_clip(truth, list(clip_audio))       # 고친 동작
        seen_control = L.control_ratio(from_spec)
        control_ok = (
            set(from_labels) == set(specs) - CONTROL         # 옛 동작은 대조를 놓친다
            and set(from_spec) == set(specs)
            and abs(seen_control - len(CONTROL) / len(specs)) < 1e-9
            and rows[0].fp > 0                               # 대조 클립의 오탐이 세진다
        )
        print(f"대조 클립: 분모 {len(from_spec)}개 중 {len(CONTROL)}개 · "
              f"비율 {seen_control:.0%} (라벨만 보면 {L.control_ratio(from_labels):.0%}) · "
              f"오탐 {rows[0].fp}건")

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

        # 🔴2 회귀 — 같은 사건을 두 사람이 찍은 CSV에서 annotator를 안 거르면
        # 참 사건이 두 번 세지고 1:1 매칭이라 하나가 자동으로 미탐이 된다.
        dual = truth + [L.replace_annotator(x, "A") for x in truth[:1]]
        r_one = score(L.pick_annotator(dual, "synthetic"), dets, 3.0)
        r_both = score(dual, dets, 3.0)
        annot_ok = r_one.n_true == len(truth) and r_both.n_true == len(truth) + 1 \
            and r_one.recall > r_both.recall
        print(f"주석자 필터: 단독 {r_one.n_true}건 recall {r_one.recall:.3f} · "
              f"이중 {r_both.n_true}건 recall {r_both.recall:.3f}")

        # 🟡2 회귀 — wav가 하나 없으면 표 머리줄에 드러나야 한다
        short = render(rows, per_min, total, n_used=len(specs) - 1, n_expected=len(specs))
        head_ok = "명세보다 적다" in short.splitlines()[0] and \
            "명세보다 적다" not in render(rows, per_min, total,
                                     n_used=len(specs), n_expected=len(specs))
        print(f"클립 수 표기: {short.splitlines()[0].split(' · ')[0]}")

        # 재리뷰 🔴1·🟡2 회귀 — 명세 밖 라벨은 빠지고 알려지며, 대조 비율은 평가 집합 기준이다
        typo = truth + [L.Label("synth", "c999", 0.0, 3.0, 3.6, "jumpscare",
                                annotator="synthetic")]
        kept, used_x, stray, _, _ = evaluation_set(typo, list(specs), clip_audio)
        no_c003 = {k: v for k, v in clip_audio.items() if k != "c003"}
        _, used_y, _, missing_y, _ = evaluation_set(truth, list(specs), no_c003)
        ratio_y = L.control_ratio(L.by_clip(truth, list(used_y)))
        mirror_ok = (stray == ["c999"] and all(x.clip_id != "c999" for x in kept)
                     and L.control_ratio(L.by_clip(kept, list(used_x))) == len(CONTROL) / len(specs)
                     and missing_y == ["c003"] and ratio_y == 0.0)
        print(f"명세 밖 라벨: {stray} 제외 · 대조 클립 wav 없을 때 평가 집합 비율 {ratio_y:.0%}")

        # 🟡3·🟡4 — 머리줄에 주석자와 명세 유무가 남는다
        nospec = render(rows, per_min, total, n_used=3, n_expected=3, has_spec=False)
        withwho = render(rows, per_min, total, n_used=3, n_expected=3, annotator="B")
        head2_ok = ("명세 없음" in nospec.splitlines()[0]
                    and withwho.splitlines()[0].startswith("주석자 B · 클립 3/3개"))

        # 🟢5 — 명세 파서의 세 갈래
        def _raises(body, needle):
            pth = tmpdir / "spec.csv"
            pth.write_text(body, encoding="utf-8")
            try:
                L.load_clips(pth)
            except ValueError as e:
                return needle in str(e)
            return False
        good = tmpdir / "good.csv"
        good.write_text("clip_id,video_id\nc001,v\nc002,v\n", encoding="utf-8")
        parser_ok = (L.load_clips(good) == ["c001", "c002"]
                     and _raises("video_id\nv\n", "열이 없다")
                     and _raises("clip_id\n", "0개")
                     and _raises("clip_id\nc001\nc001\n", "두 번"))
        print(f"명세 파서: 정상·열 없음·빈 명세·중복 {'통과' if parser_ok else '실패'}")

        # 사용자 결정(09.13) — in_time/tp 열과 지연 분포 헤드라인.
        # 합성 오디오는 지연이 전부 0.005 s이고 recall이 1이라 새 열·헤드라인을 못 가른다(#33 리뷰 🔴2).
        # 지연이 lookahead 경계(0.5·1.0·2.0) 양쪽에 흩어지고 미탐·음수 지연이 섞인 픽스처를 직접 만들어
        # **손으로 센 기대값**과 대조한다 — 틀리게 구현하면 깨지는 값이다.
        #   (라벨 onset, 탐지기 onset, 확신)       지연 = 확신 − 라벨 onset
        U = [(10.0, 10.0, 10.3), (20.0, 20.0, 20.7), (30.0, 30.0, 31.4),
             (40.0, 40.0, 42.5),
             (50.0, 50.45, 50.6),   # 라벨 기준 0.6 · 탐지기 onset 기준이면 0.15 — 둘을 섞으면 L=0.5 열이 바뀐다
             (60.0, None, None)]    # 미탐 → recall 5/6
        ut = [L.Label("u", "u1", 0.0, on, on + 0.6, "jumpscare") for on, _, _ in U]
        ud = [Detection(onset=do, fire=fi, score=1.0, clip_id="u1") for _, do, fi in U if do is not None]
        urows = [score(ut, ud, la) for la in LOOKAHEADS]
        want_tod = [0 / 5, 1 / 5, 3 / 5, 4 / 5, 5 / 5, 5 / 5]   # 지연 [0.3 0.6 0.7 1.4 2.5] 중 제때
        want_tl = [w * 5 / 6 for w in want_tod]                  # 참 사건 6건
        us = latency_summary(urows[0].latencies)
        lat_ok = (all(abs(m.timely_of_detected - w) < 1e-9 for m, w in zip(urows, want_tod))
                  and all(abs(m.timeliness - w) < 1e-9 for m, w in zip(urows, want_tl))
                  and us is not None and us["n"] == 5 and abs(us["median"] - 0.7) < 1e-9
                  and abs(us["p90"] - 2.06) < 1e-9 and abs(us["max"] - 2.5) < 1e-9
                  and us["negative"] == 0)
        # 🔴1 — 라벨 onset이 정점 근처에 찍혀 지연이 음수인 사건
        nt = [L.Label("u", "u2", 0.0, 10.3, 10.9, "jumpscare")]
        nd = [Detection(onset=10.0, fire=10.05, score=1.0, clip_id="u2")]
        nrows = [score(nt, nd, la) for la in LOOKAHEADS]
        ntxt = render(nrows, 1.0, 60.0)
        neg_ok = ("음수 지연 1건" in ntxt and "≈ 0.000 s" in ntxt and "정의상 0" not in ntxt
                  and abs(nrows[0].timeliness - 1.0) < 1e-9)
        # 🟡3 — 이미 더한 렌더링 시간을 「+ 렌더링」으로 또 붙이지 않는다
        rtxt = render([score(ut, ud, la, render_s=0.2) for la in LOOKAHEADS], 1.0, 60.0)
        # 표 아래 주석의 「확신) + 렌더링 시간」은 정의라서 제외하고, 헤드라인 줄만 본다
        rhead = next((l for l in rtxt.splitlines() if l.startswith("→ 필요한 지연량")), "")
        render_ok = rhead.startswith("→ 필요한 지연량(p90 기준) ≈ 2.260 s (렌더링 0.20 s 포함)") and "+ 렌더링" not in rhead
        txt = render(rows, per_min, total, n_used=len(clip_audio), n_expected=len(specs))
        lines = txt.splitlines()
        head_order_ok = (lines[0].startswith("클립") and lines[1].startswith("**탐지 지연**")
                         and "잡은 것 중 제때" in txt)
        print(f"헤드라인: {lines[1]}")
        print(f"지연 픽스처(경계 양쪽 · 미탐 1): 잡은 것 중 제때·적시성·중앙값·p90 "
              f"{'일치' if lat_ok else '불일치'} · 음수 지연 {'알림' if neg_ok else '실패'} · "
              f"렌더링 표기 {'통과' if render_ok else '실패'}")

        ok = (
            lat_ok
            and neg_ok
            and render_ok
            and head_order_ok
            and mirror_ok
            and head2_ok
            and parser_ok
            and head_ok
            and rows[-1].recall >= 2 / 3
            and rows[0].timeliness < rows[-1].timeliness
            and no_cross
            and control_ok
            and annot_ok
        )
        print("\n자체 점검:", "통과" if ok else "실패 — 파라미터 확인 필요")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 — lookahead 시간별 성능 곡선")
    ap.add_argument("--labels", help="docs/labeling-guide.md §1 스키마 CSV")
    ap.add_argument("--audio", help="clip_id.wav 들이 있는 디렉터리")
    ap.add_argument("--clips", help="클립 명세 CSV — 평가 분모. 없으면 wav 디렉터리 전체")
    ap.add_argument("--annotator", help="이 주석자의 라벨만 쓴다 (기본: 가장 많이 단 사람)")
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

    # 주석자 — 두 사람 라벨이 한 CSV에 있으면 참 사건이 두 번 세지고,
    # 1:1 매칭이라 둘 중 하나가 자동으로 미탐이 된다 (#19 리뷰 🔴2).
    who = L.annotators(truth)
    pick = a.annotator
    if pick is None and len(who) > 1:
        pick = max(who, key=lambda w: sum(x.annotator == w for x in truth))
        print(f"알림: 주석자 {who}가 섞여 있어 '{pick}'만 쓴다 — "
              f"바꾸려면 --annotator. 두 사람을 함께 읽는 것은 κ 계산뿐이다",
              file=sys.stderr)
    coverage = {w: len({x.clip_id for x in truth if x.annotator == w}) for w in who}
    if pick is not None:
        truth = L.pick_annotator(truth, pick)
        if not truth:
            print(f"주석자 '{pick}'의 라벨이 없다 (있는 것: {who})", file=sys.stderr)
            return 1

    audio = {p.stem: p for p in Path(a.audio).glob("*.wav")}

    # 평가 분모 — **라벨이 아니라 클립 목록이다.** 대조 클립은 라벨 행이 0개라
    # 라벨에서 역산하면 통째로 빠지고 오탐이 안 세진다 (#19 리뷰 🔴1).
    if a.clips:
        clip_ids = L.load_clips(a.clips)
    else:
        clip_ids = sorted(audio)
        print("알림: --clips가 없어 wav 디렉터리 전체를 분모로 쓴다", file=sys.stderr)

    # 고른 주석자가 분모의 일부만 라벨링했으면 **나머지가 전부 대조 클립으로 세진다**.
    # §6의 이중 라벨링은 10~15%만 겹치므로 --annotator A로 돌리면 비율이 80%대로
    # 뛰고 30% 검사를 가뿐히 넘는다 — 평가가 가장 망가진 상태에서 경고가 조용해진다
    # (#26 리뷰 🟡1). 막지는 않는다. A 라벨만 따로 볼 일이 있다.
    if pick is not None and coverage and coverage.get(pick, 0) < max(coverage.values()):
        mine, best = coverage.get(pick, 0), max(coverage.values())
        print(f"경고: 주석자 '{pick}'는 클립 {mine}개만 라벨링했다 "
              f"(가장 많이 단 사람은 {best}개). 나머지가 대조 클립으로 세지므로 "
              f"**평가에 쓸 값이 아니다** — κ 계산용이면 --annotator를 빼라",
              file=sys.stderr)

    truth, used, stray, missing, extra = evaluation_set(truth, clip_ids, audio)
    if stray:
        # --clips 없이 돌리면 분모가 wav 목록이라, 이 라벨이 빠지는 이유는 명세가 아니라
        # wav가 없어서다 — 경고가 스스로와 모순되지 않게 문구를 가른다 (#26 재리뷰 🟢4)
        where = "명세에 없는 클립" if a.clips else "wav 없는 클립"
        print(f"경고: {where}의 라벨 {stray} — 평가에서 뺀다 "
              f"(clip_id 오타면 참 사건이 사라져 recall이 부푼다)", file=sys.stderr)
    if missing:
        print(f"경고: wav 없는 클립 {missing}", file=sys.stderr)
    if extra:
        print(f"경고: 명세에 없는 wav {extra} — 평가에서 뺀다", file=sys.stderr)

    # 평가할 wav가 하나도 없으면 **표를 내지 않는다.** 0으로 채운 표는 「lookahead 0에서
    # 적시성이 무너졌다」와 생김새가 같아서, --audio 경로를 잘못 준 결과가 E1이 주장하려는
    # 그림으로 읽힌다. 이 하니스가 막아온 것은 조용히 틀린 숫자인데, 이건 그럴듯한 숫자다
    # (#26 재리뷰 🟡1).
    if not used:
        print(f"실패: 평가할 wav가 없다 — --audio {a.audio} 에서 명세의 클립을 하나도 찾지 못했다. "
              f"표를 내지 않는다", file=sys.stderr)
        return 1

    # 대조 비율은 **실제로 평가된 집합**으로 잰다 — 30% 기준이 지키려는 것은 출력된 표다
    clips = L.by_clip(truth, list(used))
    ratio = L.control_ratio(clips)
    if ratio < 0.30:
        print(f"경고: 대조 클립 비율 {ratio:.0%} ({sum(c.is_control for c in clips.values())}"
              f"/{len(clips)}, 평가된 클립 기준) — §4 기준 30% 미만이라 "
              f"정밀도 수치를 방어할 수 없다", file=sys.stderr)

    shown = pick if pick is not None else (who[0] if len(who) == 1 and who[0] else None)
    rows, per_min, total = analyse(
        used, truth, Params(drms_min=a.drms_min), render_s=a.render_s,
    )
    print(render(rows, per_min, total, n_used=len(used), n_expected=len(clip_ids),
                 annotator=shown, has_spec=bool(a.clips)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
