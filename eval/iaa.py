"""주석자 간 일치도 — `labeling-guide.md` §6.

세 가지를 함께 낸다. **하나로는 우리 경우를 설명할 수 없다.**

1. **사건 매칭 κ** — §6에 적힌 절차 그대로. onset ±0.5초로 매칭하고 카테고리
   일치로 Cohen's κ를 낸다. 매칭 안 된 사건은 상대가 「없음」인 것으로 센다
2. **시간 구간 κ** — 클립을 고정 폭(기본 100 ms)으로 잘라 각 구간이 사건인지
   아닌지로 κ를 낸다. 항목 집합이 고정돼 주변합이 덜 치우친다
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

사용:
    python3 -m eval.iaa --labels labels.csv --clips clips.csv
    python3 -m eval.iaa --labels B.csv A.csv --clips clips.csv
    python3 -m eval.iaa --selftest
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .labels import Label, load

TOL = 0.5        # §6 — onset이 이 안이면 같은 사건
SLICE_S = 0.1    # 시간 구간 폭. §3의 목표 정밀도 ±0.1초와 같은 눈금
NONE = "없음"

# §6 기준표
BANDS = [(0.8, "좋음 — 그대로 진행"),
         (0.6, "보통 — 불일치 사례를 함께 보고 기준을 다듬는다"),
         (0.0, "기준이 모호하다 — 문서를 고치고 다시 라벨링")]


def band(k: float) -> str:
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
    if n == 0:
        return 0.0, 0.0, 0.0
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
    """사건이 겹치는 구간 번호들."""
    out: set[int] = set()
    n = max(1, int(round(dur / w)))
    for e in events:
        lo = max(0, int(e.onset // w))
        hi = min(n - 1, int((e.offset - 1e-9) // w))
        out.update(range(lo, hi + 1))
    return out


def slice_kappa(per_clip: dict[str, tuple[list[Label], list[Label], float]],
                w: float = SLICE_S) -> tuple[float, float, float, int, float]:
    """구간 단위 κ. (κ, po, pe, 전체 구간 수, 사건 구간 비율)"""
    items: list[tuple[str, str]] = []
    for _cid, (ea, eb, dur) in sorted(per_clip.items()):
        n = max(1, int(round(dur / w)))
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
    return 2 * inter / (a_tot + b_tot) if (a_tot + b_tot) else 0.0


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
           tol: float = TOL, w: float = SLICE_S, category: str | None = None) -> tuple[str, float]:
    """보고서 문자열과 「대표 κ」를 돌려준다."""
    la = [x for x in labels if x.annotator == who_a]
    lb = [x for x in labels if x.annotator == who_b]
    if category:
        la = [x for x in la if x.category == category]
        lb = [x for x in lb if x.category == category]

    # **두 사람이 함께 라벨링한 클립만** 본다. 한쪽만 단 클립을 넣으면 그 클립 전체가
    # 불일치로 세져 일치도가 무의미해진다 (§6은 10~15%만 이중 라벨링한다).
    ca, cb = {x.clip_id for x in la}, {x.clip_id for x in lb}
    both = sorted(ca & cb)
    only_a, only_b = sorted(ca - cb), sorted(cb - ca)

    out: list[str] = []
    out.append(f"주석자 '{who_a}' vs '{who_b}'"
               + (f" · 카테고리 {category}" if category else " · 전체 카테고리"))
    out.append(f"이중 라벨링 클립 {len(both)}개: {', '.join(both) if both else '없음'}")
    if only_a or only_b:
        out.append(f"  한쪽만 라벨링한 클립은 제외 — {who_a}만 {len(only_a)}개 · "
                   f"{who_b}만 {len(only_b)}개")
    if not both:
        out.append("\n두 사람이 함께 본 클립이 없다 — 일치도를 계산할 수 없다.")
        return "\n".join(out), float("nan")

    per_clip: dict[str, tuple[list[Label], list[Label], float]] = {}
    pairs: list[Pair] = []
    for cid in both:
        ea = [x for x in la if x.clip_id == cid]
        eb = [x for x in lb if x.clip_id == cid]
        ps = match_events(ea, eb, tol)
        pairs += ps
        dur = durations.get(cid, 0.0)
        if dur <= 0:                    # 명세에 길이가 없으면 마지막 offset까지로 본다
            dur = max([e.offset for e in ea + eb] + [1.0])
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
    k_ev_txt = "정의 안 됨" if k_ev != k_ev else f"{k_ev:.3f}"
    out.append(f"| **사건 매칭 κ** (§6 절차) | {k_ev_txt} | 매칭된 사건의 카테고리 일치. "
               f"po={po_ev:.3f} pe={pe_ev:.3f} |")
    out.append(f"| **시간 구간 κ** ({w*1000:.0f} ms) | {k_sl:.3f} | 구간 {n_sl}개 중 "
               f"사건 구간 {prev:.1%}. po={po_sl:.3f} pe={pe_sl:.3f} |")
    out.append(f"| **양성 특정 일치도** | {a_psa:.3f} | 사건 없는 시간의 양과 무관 |")

    # κ 역설 경고 — 관측 일치가 높은데 κ가 낮으면 지표가 상황에 안 맞는 것이다
    out.append("")
    if k_ev != k_ev:                      # nan — 모든 항목이 같은 판정이다
        out.append(f"⚠ **사건 매칭 κ가 정의되지 않는다.** 매칭된 사건 {n_match}건이 "
                   f"전부 같은 카테고리이고 한쪽만 단 사건도 없어, 판정에 분산이 "
                   f"없다(pe={pe_ev:.3f}). 완전 일치인데 κ를 낼 수 없는 경우다.")
        out.append(f"→ **시간 구간 κ {k_sl:.3f}** 를 대표값으로 본다: {band(k_sl)}")
        headline = k_sl
    elif items and pe_ev > 0.5 and po_ev >= 0.7 and k_ev < 0.6:
        out.append(f"⚠ **사건 매칭 κ를 그대로 쓰면 안 된다.** 관측 일치가 "
                   f"{po_ev:.1%}인데 κ가 {k_ev:.3f}다. 우연 일치 pe가 {pe_ev:.3f}로 "
                   f"높기 때문이고, 원인은 카테고리가 사실상 한 종류여서 주변합이 "
                   f"「거의 전부 사건, 드물게 없음」으로 치우친 것이다(κ 역설). "
                   f"데이터가 나쁜 것이 아니라 지표가 이 상황에 맞지 않는다.")
        out.append(f"→ **시간 구간 κ {k_sl:.3f}** 를 대표값으로 본다: {band(k_sl)}")
        headline = k_sl
    else:
        out.append(f"→ 사건 매칭 κ {k_ev:.3f}: {band(k_ev)}")
        headline = k_ev

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
    ap.add_argument("--a", help="주석자 A (기본: 라벨을 많이 단 사람)")
    ap.add_argument("--b", help="주석자 B")
    ap.add_argument("--category", help="이 카테고리만 본다 (예: jumpscare)")
    ap.add_argument("--tol", type=float, default=TOL, help=f"onset 허용오차 (기본 {TOL}초)")
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

    txt, _k = report(labels, who_a, who_b, durations, a.tol, a.slice_s, a.category)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
