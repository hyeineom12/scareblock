"""주석자 간 일치도 — `labeling-guide.md` §6.

세 가지를 함께 낸다. **하나로는 우리 경우를 설명할 수 없다.**

1. **사건 매칭 κ** — §6에 적힌 절차 그대로. onset ±0.5초로 매칭하고 카테고리
   일치로 Cohen's κ를 낸다. 매칭 안 된 사건은 상대가 「없음」인 것으로 센다
2. **시간 구간 κ** — 클립을 100 ms 폭으로 잘라 각 구간이 사건인지 아닌지로 κ를 낸다.
   **항목이 고정돼 한쪽만 단 사건이 자동으로 불일치가 되고, 시간 정밀도가 반영된다.**
   우연 일치가 낮아서 고르는 것이 아니다 — 사건 구간이 드물어 pe는 오히려 더 높다
   (아래 예를 300초 클립 1개로 돌리면 0.939). 그래서 유병률에 흔들린다
3. **양성 특정 일치도(PSA)** — `2|A∩B| / (|A|+|B|)`. 사건이 없는 시간의 양에
   영향을 받지 않는다

**왜 셋인가 — 1번만으로는 우리 경우에 κ가 무너진다.**
09.16 논문은 `jumpscare` 한 종류만 쓴다. 그러면 매칭된 사건은 **전부 카테고리가
같고**(둘 다 jumpscare), 불일치는 「한쪽만 단 것」에서만 나온다. 주변합이
「거의 전부 jumpscare, 드물게 없음」으로 치우치면 우연 일치 `pe`가 0.8을 넘어서
**80% 일치인데 κ가 음수로 나온다.** 손으로 확인한 예:

    사건 8건 일치 + A만 1건 + B만 1건
    po = 0.800 · pe = 0.820 · κ = -0.111

이것은 데이터가 나쁜 것이 아니라 지표가 그 상황에 맞지 않는 것이다
(희소 범주에서 생기는 이른바 κ 역설). 그래서 2·3번을 함께 낸다.
**보고할 때는 구간 κ와 PSA를 나란히 싣는다** — 셋 중 유병률에 흔들리지 않는 것은 PSA뿐이다.

사용:
    python3 -m eval.iaa --labels labels.csv --clips clips.csv
    python3 -m eval.iaa --labels B.csv A.csv --clips clips.csv \\
                        --a B --b A --clips-a clips-labeled-B.csv --clips-b clips-labeled-A.csv
    python3 -m eval.iaa --selftest
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .labels import Label, load

TOL = 0.5        # §6 — onset이 이 안이면 같은 사건
SLICE_S = 0.1    # 보고값은 100 ms 고정 — §3의 목표 정밀도 ±0.1초와 같은 눈금. --slice-s는 민감도 확인용
NONE = "없음"

# §6 기준표
BANDS = [(0.8, "좋음 — 그대로 진행"),
         (0.6, "보통 — 불일치 사례를 함께 보고 기준을 다듬는다"),
         (0.0, "기준이 모호하다 — 문서를 고치고 다시 라벨링")]


def _fmt(v: float) -> str:
    return "정의 안 됨" if v != v else f"{v:.3f}"


def band(k: float) -> str:
    if k != k:                          # nan은 모든 비교가 False라 「모호하다」로 떨어진다(#32 재리뷰 🔴)
        return "판정 불가 — κ가 정의되지 않는다"
    for lo, text in BANDS:
        if k >= lo:
            return text
    return BANDS[-1][1]


@dataclass
class Pair:
    a: Label | None
    b: Label | None

    @property
    def matched(self) -> bool:
        return self.a is not None and self.b is not None

    @property
    def agreed(self) -> bool:
        return self.matched and self.a.category == self.b.category

    @property
    def gap(self) -> float | None:
        return abs(self.a.onset - self.b.onset) if self.matched else None


def match_events(ea: list[Label], eb: list[Label], tol: float = TOL) -> list[Pair]:
    """onset 근접도로 1:1 매칭. **카테고리를 보지 않고 매칭한다.**

    카테고리로 먼저 거르면 「같은 사건을 다른 카테고리로 본 경우」가
    매칭에서 빠져 카테고리 불일치를 관찰할 수 없다 (§6 3단계가 그것을 본다).
    """
    cand = sorted((abs(x.onset - y.onset), i, j)
                  for i, x in enumerate(ea) for j, y in enumerate(eb)
                  if abs(x.onset - y.onset) <= tol)
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[Pair] = []
    for _, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append(Pair(ea[i], eb[j]))
    pairs += [Pair(x, None) for i, x in enumerate(ea) if i not in used_a]
    pairs += [Pair(None, y) for j, y in enumerate(eb) if j not in used_b]
    return pairs


def cohen_kappa(items: list[tuple[str, str]]) -> tuple[float, float, float]:
    """(κ, 관측 일치 po, 우연 일치 pe). 항목은 (A의 판정, B의 판정) 쌍."""
    n = len(items)
    if n == 0:                          # 항목이 없으면 일치도 자체가 없다 — 0으로 두면 「모호하다」로 읽힌다
        return float("nan"), float("nan"), float("nan")
    labels = sorted({v for pair in items for v in pair})
    po = sum(1 for x, y in items if x == y) / n
    pe = 0.0
    for c in labels:
        pa = sum(1 for x, _ in items if x == c) / n
        pb = sum(1 for _, y in items if y == c) / n
        pe += pa * pb
    if pe >= 1.0:                       # 모든 항목이 한 값 — κ가 정의되지 않는다
        return float("nan"), po, pe
    return (po - pe) / (1 - pe), po, pe


def _slices(events: list[Label], dur: float, w: float) -> set[int]:
    """사건이 겹치는 구간 번호들.

    `dur`보다 뒤에서 시작한 사건은 빈 집합이 된다 — **호출하는 쪽이 `dur`를 마지막 offset까지
    늘려서 넘긴다**(`report`). 여기서 조용히 버리면 한쪽만 길이 밖으로 나간 불일치가 사라져
    일치도가 항상 올라간다(#32 재리뷰 🔴).
    """
    out: set[int] = set()
    # 밀리초 정수로 나눈다. `10.0 // 0.1 == 99.0`처럼 부동소수 내림이 0.05초 차이 라벨을
    # 구간 두 개만큼 어긋나게 해 일치도에 편향으로 들어갔다(#27 리뷰 🟡4)
    w_ms = max(1, round(w * 1000))
    n = max(1, round(dur * 1000) // w_ms)
    for e in events:
        on_ms, off_ms = round(e.onset * 1000), round(e.offset * 1000)
        lo = max(0, on_ms // w_ms)
        hi = min(n - 1, max(lo, (off_ms - 1) // w_ms))
        out.update(range(lo, hi + 1))
    return out


def slice_kappa(per_clip: dict[str, tuple[list[Label], list[Label], float]],
                w: float = SLICE_S) -> tuple[float, float, float, int, float]:
    """구간 단위 κ. (κ, po, pe, 전체 구간 수, 사건 구간 비율)"""
    items: list[tuple[str, str]] = []
    for _cid, (ea, eb, dur) in sorted(per_clip.items()):
        n = max(1, round(dur * 1000) // max(1, round(w * 1000)))
        sa, sb = _slices(ea, dur, w), _slices(eb, dur, w)
        for i in range(n):
            items.append(("사건" if i in sa else NONE, "사건" if i in sb else NONE))
    k, po, pe = cohen_kappa(items)
    pos = sum(1 for x, y in items if x == "사건" or y == "사건")
    return k, po, pe, len(items), (pos / len(items) if items else 0.0)


def psa(per_clip: dict[str, tuple[list[Label], list[Label], float]],
        w: float = SLICE_S) -> float:
    """양성 특정 일치도 = 2|A∩B| / (|A|+|B|). 사건 없는 시간의 양과 무관하다."""
    inter = a_tot = b_tot = 0
    for _cid, (ea, eb, dur) in per_clip.items():
        sa, sb = _slices(ea, dur, w), _slices(eb, dur, w)
        inter += len(sa & sb)
        a_tot += len(sa)
        b_tot += len(sb)
    return 2 * inter / (a_tot + b_tot) if (a_tot + b_tot) else float("nan")   # 둘 다 사건 0 — 정의 안 됨


def read_ids(path: Path) -> set[str]:
    """검토 명세(`elan2csv --out-clips`)의 clip_id — 사건 0건 클립도 여기엔 있다."""
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return {(r.get("clip_id") or "").strip() for r in csv.DictReader(f)} - {""}


def read_clip_durations(path: Path) -> dict[str, float]:
    import csv
    out: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = (r.get("clip_id") or "").strip()
            if cid:
                try:
                    out[cid] = float(r.get("duration_s") or r.get("clip_s") or 0)
                except ValueError:
                    out[cid] = 0.0
    return out


def report(labels: list[Label], who_a: str, who_b: str, durations: dict[str, float],
           tol: float = TOL, w: float = SLICE_S, category: str | None = None,
           reviewed_a: set[str] | None = None,
           reviewed_b: set[str] | None = None) -> tuple[str, float]:
    """보고서 문자열과 「대표 κ」를 돌려준다."""
    la = [x for x in labels if x.annotator == who_a]
    lb = [x for x in labels if x.annotator == who_b]
    if category:
        la = [x for x in la if x.category == category]
        lb = [x for x in lb if x.category == category]

    # **두 사람이 함께 라벨링한 클립만** 본다. 한쪽만 단 클립을 넣으면 그 클립 전체가
    # 불일치로 세져 일치도가 무의미해진다 (§6은 10~15%만 이중 라벨링한다).
    # **검토 명세가 있으면 그것으로 정한다.** 라벨이 있는 클립으로 정하면 한 사람이
    # 「사건 0건」으로 본 클립이 통째로 빠져, 한쪽이 사건을 놓친 불일치가 「완전 일치」로
    # 보고된다(#27 리뷰 🔴3). `elan2csv --out-clips`가 낸 명세가 그 기록이다.
    ca, cb = {x.clip_id for x in la}, {x.clip_id for x in lb}
    by_spec = reviewed_a is not None and reviewed_b is not None
    if by_spec:
        ca, cb = set(reviewed_a), set(reviewed_b)
    both = sorted(ca & cb)
    only_a, only_b = sorted(ca - cb), sorted(cb - ca)

    out: list[str] = []
    out.append(f"주석자 '{who_a}' vs '{who_b}'"
               + (f" · 카테고리 {category}" if category else " · 전체 카테고리"))
    basis = "검토 명세 기준" if by_spec else "라벨 기준 — 사건 0건으로 본 클립은 빠진다. --clips-a/--clips-b 권장"
    out.append(f"이중 라벨링 클립 {len(both)}개 ({basis}): {', '.join(both) if both else '없음'}")
    if only_a or only_b:
        # 명세 모드에서 이 수는 라벨이 아니라 명세 차이에서 온다(#32 재리뷰 짧은 것)
        out.append(f"  한쪽만 {'검토한(명세에 있는)' if by_spec else '라벨링한'} 클립은 제외 — {who_a}만 {len(only_a)}개 · "
                   f"{who_b}만 {len(only_b)}개")
    if not both:
        out.append("\n두 사람이 함께 본 클립이 없다 — 일치도를 계산할 수 없다.")
        return "\n".join(out), float("nan")

    # 건수 줄과 지표가 같은 집합을 보게 `both`로 좁힌다. 명세 밖 클립의 라벨은 따로 알린다(#32 재리뷰 🟡)
    # **각자의 명세와 각자의 라벨을 대조한다.** 명세는 주석자마다 따로 받으니 한쪽만 낡는 것이 기본 경로다.
    # 두 명세 모두에서 빼는 식(`- ca - cb`)은 한쪽 명세에만 없는 클립을 못 보고, 그 클립의 이중 라벨링이
    # 조용히 빠진 채 「한쪽만 라벨링했다」로 거꾸로 보고됐다(#32 재리뷰 🟡). both에 넣지는 않는다 —
    # 명세가 낡았는지 clip_id가 틀렸는지는 사람이 본다
    stray_ab = [(x, who_a) for x in la if x.clip_id not in ca] + [(x, who_b) for x in lb if x.clip_id not in cb] \
        if by_spec else []
    stray = sorted({f"{x.clip_id}({w})" for x, w in stray_ab})
    n_stray = len(stray_ab)
    la = [x for x in la if x.clip_id in both]
    lb = [x for x in lb if x.clip_id in both]

    per_clip: dict[str, tuple[list[Label], list[Label], float]] = {}
    pairs: list[Pair] = []
    no_dur: list[str] = []
    over: list[str] = []
    for cid in both:
        ea = [x for x in la if x.clip_id == cid]
        eb = [x for x in lb if x.clip_id == cid]
        ps = match_events(ea, eb, tol)
        pairs += ps
        dur = durations.get(cid, 0.0)
        if dur <= 0:
            if not (ea or eb):
                # 길이도 사건도 없으면 구간 수를 지어낼 수 없다. 1초로 두면 0건 클립이 구간 10개짜리가
                # 되어 구간 κ를 왜곡한다(#32 리뷰 🟡) — 구간 계산에서 빼고 알린다
                no_dur.append(cid)
                continue
            dur = max(e.offset for e in ea + eb)   # 명세에 길이가 없으면 마지막 offset까지로 본다
        elif ea or eb:
            # 명세 길이(실측 파일 길이)보다 뒤까지 찍힌 사건을 지우지 않는다 — 길이를 라벨까지 늘리고 알린다.
            # 지우면 대개 한쪽만 밖으로 나간 불일치가 사라져 일치도가 부풀려진다(#32 재리뷰 🔴)
            last = max(e.offset for e in ea + eb)
            if last > dur:
                over.append(f"{cid}(+{last - dur:.2f}초)")
                dur = last
        per_clip[cid] = (ea, eb, dur)

    n_match = sum(1 for p in pairs if p.matched)
    n_agree = sum(1 for p in pairs if p.agreed)
    n_a_only = sum(1 for p in pairs if p.a is not None and p.b is None)
    n_b_only = sum(1 for p in pairs if p.a is None and p.b is not None)

    # ① §6 절차 — 사건 매칭 κ
    items = [((p.a.category if p.a else NONE), (p.b.category if p.b else NONE))
             for p in pairs]
    k_ev, po_ev, pe_ev = cohen_kappa(items)

    # ② 시간 구간 κ
    k_sl, po_sl, pe_sl, n_sl, prev = slice_kappa(per_clip, w)

    # ③ 양성 특정 일치도
    a_psa = psa(per_clip, w)

    if no_dur:
        out.append(f"  ⚠ 클립 길이를 몰라 구간 계산에서 뺀 사건 0건 클립 {len(no_dur)}개: {', '.join(no_dur)} — --clips를 준다")
    if over:
        out.append(f"  ⚠ 명세 길이보다 뒤까지 찍힌 사건이 있어 구간 계산 길이를 마지막 offset까지 늘렸다: {', '.join(over)}")
    if stray:
        out.append(f"  ⚠ 자기 검토 명세에 없는 클립의 라벨 {n_stray}건({', '.join(stray)}) — 계산에서 빠졌다. 그 주석자의 명세가 낡았는지 확인한다")
    gaps = [p.gap for p in pairs if p.matched]
    out.append("")
    out.append(f"사건 — {who_a} {len(la)}건 · {who_b} {len(lb)}건 · "
               f"매칭 {n_match}건 (카테고리 일치 {n_agree}건) · "
               f"{who_a}만 {n_a_only}건 · {who_b}만 {n_b_only}건")
    if gaps:
        out.append(f"매칭된 쌍의 onset 차이 — 평균 {sum(gaps)/len(gaps):.3f}초 · "
                   f"최대 {max(gaps):.3f}초 (§3 목표 ±0.1초)")

    out.append("")
    out.append("| 지표 | 값 | 무엇을 재는가 |")
    out.append("|---|---|---|")
    out.append(f"| **사건 매칭 κ** (§6 절차) | {_fmt(k_ev)} | 매칭된 사건의 카테고리 일치. "
               f"po={_fmt(po_ev)} pe={_fmt(pe_ev)} |")
    out.append(f"| **시간 구간 κ** ({w*1000:.0f} ms) | {_fmt(k_sl)} | 구간 {n_sl}개 중 "
               f"사건 구간 {prev:.1%}. po={_fmt(po_sl)} pe={_fmt(pe_sl)} |")
    out.append(f"| **양성 특정 일치도** | {_fmt(a_psa)} | 사건 없는 시간의 양과 무관 |")

    # κ 역설 경고 — 관측 일치가 높은데 κ가 낮으면 지표가 상황에 안 맞는 것이다
    out.append("")
    # 진짜 카테고리 혼동은 **매칭된 쌍이 서로 다른 카테고리를 말한 것**뿐이다. 한쪽만 단 사건은 상대가
    # 「없음」이라 혼동이 아니라 누락이다. 카테고리 수를 items 전체에서 세면 한쪽만 단 사건 하나가
    # siren이라는 이유로 역설 감지가 꺼져 「모호하다」로 떨어졌다(#32 재리뷰 🔴)
    n_confused = sum(1 for p in pairs if p.matched and not p.agreed)
    if not items:
        # 두 사람 모두 사건 0건 — 대조 클립에서 판정이 일치한 것이다. 일치도를 낼 사건이 없을 뿐
        # 「기준이 모호하다」가 아니다(#32 리뷰 🔴, 🔴3 반영으로 0건 클립이 both에 들어오며 열린 길)
        out.append("→ **두 사람 모두 사건 0건** — 대조 클립 판정이 일치했다. 사건이 없어 κ·PSA는 "
                   "정의되지 않는다. 일치도는 사건이 있는 이중 라벨링 클립이 있어야 낼 수 있다")
        headline = float("nan")
    elif k_ev != k_ev:                    # nan — 모든 항목이 같은 판정이다
        out.append(f"⚠ **사건 매칭 κ가 정의되지 않는다.** 매칭된 사건 {n_match}건이 "
                   f"전부 같은 카테고리이고 한쪽만 단 사건도 없어, 판정에 분산이 "
                   f"없다(pe={pe_ev:.3f}). 완전 일치인데 κ를 낼 수 없는 경우다.")
        if k_sl != k_sl:
            # 구간 κ까지 nan이면 두 사람의 판정이 구간 단위로도 전부 같다 — 불일치가 아니므로 판정하지 않는다
            out.append("→ **시간 구간 κ도 정의되지 않는다** — 구간 단위로도 판정이 전부 같다. "
                       "불일치가 아니라 분산이 없는 것이므로 기준표로 판정하지 않는다")
        else:
            out.append(f"→ **시간 구간 κ {k_sl:.3f}** 를 대표값으로 본다: {band(k_sl)}")
        headline = k_sl
    # 역설 판정은 **매칭된 쌍 사이에 카테고리 혼동이 없을 때만** 한다(n_confused == 0). 진짜 혼동으로 κ가
    # 낮은 것을 구간 κ로 넘기면 카테고리를 안 보는 지표가 그 불일치를 지운다(#27 리뷰 🟡7). 한쪽만 단
    # 사건의 카테고리는 혼동이 아니라 누락이라 세지 않는다(#32 재리뷰 🔴). 카테고리가 1종이면 늘 0이다
    elif items and n_confused == 0 and pe_ev > 0.5 and po_ev >= 0.7 and k_ev < 0.6:
        out.append(f"⚠ **사건 매칭 κ를 그대로 쓰면 안 된다.** 관측 일치가 "
                   f"{po_ev:.1%}인데 κ가 {k_ev:.3f}다. 우연 일치 pe가 {pe_ev:.3f}로 "
                   f"높기 때문이고, 원인은 매칭된 쌍이 전부 같은 카테고리라(혼동 0건) 불일치가 "
                   f"「한쪽만 단 사건」에서만 나오고, 그래서 주변합이 "
                   f"「거의 전부 사건, 드물게 없음」으로 치우친 것이다(κ 역설). "
                   f"데이터가 나쁜 것이 아니라 지표가 이 상황에 맞지 않는다.")
        out.append(f"→ **시간 구간 κ {k_sl:.3f}** 를 대표값으로 본다: {band(k_sl)}")
        headline = k_sl
    else:
        out.append(f"→ 사건 매칭 κ {k_ev:.3f}: {band(k_ev)}")
        headline = k_ev
    out.append(f"   보고 — 구간 κ {_fmt(k_sl)}와 양성 특정 일치도 {_fmt(a_psa)}를 나란히 싣는다 "
               f"(PSA만 유병률에 흔들리지 않는다)")
    if abs(w - SLICE_S) > 1e-9:
        out.append(f"   ⚠ 구간 폭 {w*1000:.0f} ms — 보고값은 {SLICE_S*1000:.0f} ms 고정이고 이 결과는 민감도 확인이다")
    if abs(tol - TOL) > 1e-9:
        out.append(f"   ⚠ onset 허용오차 ±{tol:g}초 — 보고값은 §6의 ±{TOL:g}초 고정이고 이 결과는 민감도 확인이다")

    bad = [p for p in pairs if not p.agreed]
    if bad:
        out.append("")
        out.append(f"불일치 {len(bad)}건 — §6 2단계대로 함께 보고 기준을 다듬는다")
        for p in sorted(bad, key=lambda q: ((q.a or q.b).clip_id, (q.a or q.b).onset)):
            if p.matched:
                out.append(f"  {p.a.clip_id} @{p.a.onset:.2f}/{p.b.onset:.2f} — "
                           f"카테고리가 다르다: {who_a}={p.a.category} {who_b}={p.b.category}")
            elif p.a is not None:
                out.append(f"  {p.a.clip_id} @{p.a.onset:.2f}~{p.a.offset:.2f} — "
                           f"{who_a}만 달았다 ({p.a.category}"
                           f"{', 애매' if p.a.ambiguous else ''})")
            else:
                out.append(f"  {p.b.clip_id} @{p.b.onset:.2f}~{p.b.offset:.2f} — "
                           f"{who_b}만 달았다 ({p.b.category}"
                           f"{', 애매' if p.b.ambiguous else ''})")
    return "\n".join(out), headline


# ── 자체 점검 ───────────────────────────────────────────────────────────
def _lab(cid: str, on: float, off: float, who: str, cat: str = "jumpscare") -> Label:
    return Label("v", cid, 0.0, on, off, cat, annotator=who)


def _selftest() -> int:
    dur = {"c001": 300.0, "c002": 300.0, "c009": 300.0}
    checks: dict[str, bool] = {}

    # ① 완전 일치 → κ = 1
    same = [_lab("c001", 10, 11, "B"), _lab("c001", 50, 51, "B"),
            _lab("c001", 10.05, 11.02, "A"), _lab("c001", 50.04, 51.1, "A")]
    txt, k = report(same, "B", "A", dur)
    k_same, _, pe_same = cohen_kappa(
        [((p.a.category if p.a else NONE), (p.b.category if p.b else NONE))
         for p in match_events([x for x in same if x.annotator == "B"],
                               [x for x in same if x.annotator == "A"])])
    # 완전 일치면 판정에 분산이 없어 **κ가 정의되지 않는다**(pe=1). 그때는 구간 κ를 본다.
    checks["완전 일치면 사건 κ가 정의 안 되고 구간 κ로 넘어간다"] = (
        k_same != k_same and pe_same >= 1.0 and k >= 0.9 and "정의되지 않는다" in txt)

    # ② 허용오차 밖(0.8초 차)이면 두 건의 불일치가 된다
    off = [_lab("c001", 10, 11, "B"), _lab("c001", 10.8, 11.8, "A")]
    ps = match_events([x for x in off if x.annotator == "B"],
                      [x for x in off if x.annotator == "A"])
    checks["±0.5초 밖은 매칭 안 된다"] = all(not p.matched for p in ps) and len(ps) == 2

    # ③ 한쪽만 라벨링한 클립은 빠진다
    mixed = same + [_lab("c002", 20, 21, "B")]
    txt3, k3 = report(mixed, "B", "A", dur)
    both_line = txt3.split("이중 라벨링 클립")[1].split("\n")[0]
    checks["한쪽만 본 클립은 제외"] = ("c002" not in both_line and "c001" in both_line
                                and "B만 1개" in txt3 and k3 >= 0.9)

    # ④ κ 역설 — 8건 일치 + 양쪽 1건씩 → 사건 κ는 음수, 구간 κ는 쓸 수 있다
    ev = []
    for i in range(8):
        t = 10 + i * 20
        ev += [_lab("c001", t, t + 1, "B"), _lab("c001", t + 0.05, t + 1.02, "A")]
    ev += [_lab("c001", 200, 201, "B"), _lab("c001", 230, 231, "A")]
    txt4, k4 = report(ev, "B", "A", dur)
    ev_items = [((p.a.category if p.a else NONE), (p.b.category if p.b else NONE))
                for p in match_events([x for x in ev if x.annotator == "B"],
                                      [x for x in ev if x.annotator == "A"])]
    k_ev, po_ev, _ = cohen_kappa(ev_items)
    checks["κ 역설을 감지한다"] = (k_ev < 0.6 and po_ev >= 0.7
                              and "κ 역설" in txt4 and k4 > k_ev)

    # ⑤ 카테고리 불일치가 보인다
    cat = [_lab("c001", 10, 11, "B", "jumpscare"), _lab("c001", 10.1, 11, "A", "siren")]
    txt5, _ = report(cat, "B", "A", dur)
    checks["카테고리 불일치를 적발"] = "카테고리가 다르다" in txt5

    # ⑥ 🔴3 — 한 사람만 사건을 찾은 클립(c005)이 계산에서 빠지면 안 된다
    z = [_lab("c001", 10, 11, "B"), _lab("c001", 10.05, 11.0, "A"), _lab("c005", 40, 41, "A")]
    dur5 = {**dur, "c005": 300.0}
    txt6, _ = report(z, "A", "B", dur5, reviewed_a={"c001", "c005"}, reviewed_b={"c001", "c005"})
    txt6_old, _ = report(z, "A", "B", dur5)
    checks["검토 명세 기준 — 한쪽만 찾은 사건이 불일치로 세진다"] = (
        "A만 1건" in txt6 and "c005" in txt6.split("이중 라벨링 클립")[1].split("\n")[0])
    checks["명세 없이 돌리면 그 한계를 알린다"] = "라벨 기준" in txt6_old

    # ⑦ 🟡4 — onset 10.0과 10.05가 같은 첫 구간
    s1 = _slices([_lab("c1", 10.0, 10.5, "A")], 300.0, 0.1)
    s2 = _slices([_lab("c1", 10.05, 10.5, "A")], 300.0, 0.1)
    checks["경계 반올림 — 10.0과 10.05가 같은 첫 구간(100)"] = min(s1) == min(s2) == 100

    # ⑧ 🟡7 — 카테고리 2종의 진짜 혼동은 역설로 넘기지 않는다
    mc = []
    for i in range(10):
        t = 10 + i * 20
        mc += [_lab("c001", t, t + 1, "B"),
               _lab("c001", t + 0.05, t + 1, "A", "siren" if i < 3 else "jumpscare")]
    txt8, _ = report(mc, "B", "A", dur)
    checks["카테고리 2종의 혼동은 κ 역설로 처리하지 않는다"] = "κ 역설" not in txt8 and "→ 사건 매칭 κ" in txt8

    # ⑨ #32 리뷰 🔴 — 두 사람 모두 사건 0건인 파일럿이 「모호하다」로 나오면 안 된다
    txt9, k9 = report([], "A", "B", {"c009": 300.0}, reviewed_a={"c009"}, reviewed_b={"c009"})
    checks["둘 다 0건이면 「모호하다」가 아니라 일치로 보고"] = (
        "두 사람 모두 사건 0건" in txt9 and "기준이 모호하다" not in txt9
        and "정의 안 됨" in txt9 and k9 != k9)
    # ⑩ #32 리뷰 🟡 — 길이를 모르는 0건 클립을 1초짜리로 지어내지 않는다
    txt10, _ = report(z, "A", "B", {"c001": 300.0, "c005": 300.0},
                      reviewed_a={"c001", "c005", "c009"}, reviewed_b={"c001", "c005", "c009"})
    checks["길이 모르는 0건 클립은 구간 계산에서 빼고 알린다"] = "클립 길이를 몰라" in txt10 and "c009" in txt10

    # ⑪ #32 재리뷰 🔴 — 명세 길이 밖 사건을 지우지 않는다. 길이를 1.3초 줄여도 일치도가 오르면 안 된다
    tail = [_lab("c001", 12.0, 13.0, "B"), _lab("c001", 12.05, 13.02, "A"),
            _lab("c001", 298.9, 299.4, "B"), _lab("c001", 298.6, 299.2, "A")]
    txt11a, _ = report(tail, "B", "A", {"c001": 300.0})
    txt11b, _ = report(tail, "B", "A", {"c001": 298.7})
    row = lambda t, name: t.split(f"| **{name}**")[1].split("|")[1].strip()   # 값 칸
    checks["명세 길이 밖 사건을 지우지 않고 알린다"] = (
        row(txt11a, "시간 구간 κ") == row(txt11b, "시간 구간 κ")
        and row(txt11a, "양성 특정 일치도") == row(txt11b, "양성 특정 일치도")
        and "명세 길이보다 뒤까지" in txt11b and "c001(+0.70초)" in txt11b)

    # ⑫ #32 재리뷰 🔴 — 사건 κ·구간 κ가 둘 다 nan이면 「모호하다」가 아니다
    full = [_lab("c001", 0.0, 1.0, "B"), _lab("c001", 0.0, 1.0, "A")]
    txt12, _ = report(full, "B", "A", {"c001": 1.0})
    checks["band(nan)은 판정 불가 · 둘 다 nan이면 「모호하다」가 아니다"] = (
        band(float("nan")).startswith("판정 불가") and "기준이 모호하다" not in txt12
        and "시간 구간 κ도 정의되지 않는다" in txt12)

    # ⑬ #32 재리뷰 🟡 — --tol을 바꾸면 민감도 확인이라고 적는다
    txt13, _ = report(same, "B", "A", dur, tol=5.0)
    checks["--tol을 바꾸면 민감도 경고"] = "허용오차 ±5초" in txt13 and "허용오차" not in txt[txt.find("보고 —"):]

    # ⑭ #32 재리뷰 🟡 — 건수 줄은 both만 세고, 명세 밖 라벨은 알린다
    # 명세 밖 라벨을 **양쪽 모두에** 둔다 — 한쪽에만 두면 다른 쪽 좁히기를 지워도 점검이 통과한다(자가 검사)
    txt14, _ = report(z + [_lab("c777", 5, 6, "B"), _lab("c888", 7, 8, "A")], "A", "B", dur5,
                      reviewed_a={"c001", "c005"}, reviewed_b={"c001", "c005"})
    checks["건수는 이중 라벨링 클립만 · 명세 밖 라벨은 알린다"] = (
        "A 2건" in txt14 and "B 1건" in txt14
        and "명세에 없는 클립의 라벨 2건(c777(B), c888(A))" in txt14)

    # ⑮ #32 재리뷰 🟡 — 한쪽 명세에만 없는 클립. B는 c005에 라벨을 달았는데 B의 명세가 낡았다.
    # 두 명세 모두에서 빼는 식으로는 c005가 A 명세에 있어 안 걸리고, 조용히 빠진다
    half = [_lab("c001", 10, 11, "A"), _lab("c001", 10.05, 11, "B"),
            _lab("c005", 40, 41, "A"), _lab("c005", 40.1, 41, "B")]
    txt15, _ = report(half, "A", "B", {"c001": 300.0, "c005": 300.0},
                      reviewed_a={"c001", "c005"}, reviewed_b={"c001"})
    checks["한쪽 명세에만 없는 클립의 라벨도 알린다"] = "명세에 없는 클립의 라벨 1건(c005(B))" in txt15

    # ⑯ #32 재리뷰 🔴 — 한쪽만 단 사건의 카테고리를 바꿔도 판정이 안 바뀐다. 매칭된 8쌍은 전부 일치하고
    # 혼동은 0건이라, 짝 없는 사건 하나가 siren이든 jumpscare든 두 사람의 일치 양상은 같다
    ev_siren = ev[:-1] + [_lab("c001", 230, 231, "A", "siren")]
    txt16, k16 = report(ev_siren, "B", "A", dur)
    checks["한쪽만 단 사건의 카테고리를 바꿔도 판정이 안 바뀐다"] = (
        k16 == k16 and abs(k16 - k4) < 1e-9 and "κ 역설" in txt16 and "기준이 모호하다" not in txt16)

    print("=" * 68)
    print(txt4)
    print("=" * 68)
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else '실패'}] {name}")
    allok = all(checks.values())
    print("\n자체 점검:", "통과" if allok else "실패")
    return 0 if allok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="주석자 간 일치도 (labeling-guide §6)")
    ap.add_argument("--labels", nargs="*", default=[],
                    help="라벨 CSV. 한 파일에 두 주석자가 있어도 되고 파일을 나눠도 된다")
    ap.add_argument("--clips", help="클립 명세 CSV — 구간 κ에 클립 길이가 필요하다")
    ap.add_argument("--clips-a", help="주석자 A의 검토 명세 (elan2csv --out-clips) — 0건 클립을 넣는다")
    ap.add_argument("--clips-b", help="주석자 B의 검토 명세")
    ap.add_argument("--a", help="주석자 A (기본: 라벨을 많이 단 사람)")
    ap.add_argument("--b", help="주석자 B")
    ap.add_argument("--category", help="이 카테고리만 본다 (예: jumpscare)")
    ap.add_argument("--tol", type=float, default=TOL,
                    help=f"onset 허용오차 (기본 {TOL}초 — 보고값은 고정, 바꾸면 민감도 확인)")
    ap.add_argument("--slice-s", type=float, default=SLICE_S,
                    help=f"시간 구간 폭 (기본 {SLICE_S}초)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not a.labels:
        ap.error("--labels 가 필요하다 (또는 --selftest)")

    labels: list[Label] = []
    for f in a.labels:
        labels += load(Path(f))
    who = sorted({x.annotator for x in labels})
    if len(who) < 2 and not (a.a and a.b):
        print(f"주석자가 {who}뿐이다 — 이중 라벨링 CSV가 필요하다 (§6)", file=sys.stderr)
        return 1
    counts = {w: sum(1 for x in labels if x.annotator == w) for w in who}
    ranked = sorted(who, key=lambda w: -counts[w])
    who_a = a.a or ranked[0]
    who_b = a.b or next((w for w in ranked if w != who_a), ranked[-1])

    durations = read_clip_durations(Path(a.clips)) if a.clips else {}
    if not durations:
        print("알림: --clips가 없어 클립 길이를 라벨의 마지막 offset으로 본다 — "
              "구간 κ가 사건 비율을 과대평가한다", file=sys.stderr)

    ra = read_ids(Path(a.clips_a)) if a.clips_a else None
    rb = read_ids(Path(a.clips_b)) if a.clips_b else None
    if (ra is None) != (rb is None):
        ap.error("--clips-a 와 --clips-b 는 함께 준다")
    if ra is not None and not a.clips:
        ap.error("--clips-a/--clips-b를 쓸 때는 --clips로 클립 길이도 준다 — 없으면 사건 0건 클립의 구간 수를 알 수 없다")
    if ra is not None and not (a.a and a.b):
        ap.error("--clips-a/--clips-b를 쓸 때는 --a/--b로 주석자를 명시한다 — 명세와 주석자의 짝이 바뀌면 안 된다")
    txt, _k = report(labels, who_a, who_b, durations, a.tol, a.slice_s, a.category, ra, rb)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
