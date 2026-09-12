"""ELAN 내보내기 → 라벨 CSV 변환.

ELAN의 `File > Export As > Tab-delimited Text` 결과를
`docs/labeling-guide.md` §1 스키마로 바꾼다. 변환 규약은
`docs/elan-setup.md` §4(라벨 텍스트 → `ambiguous`/`note`)와 §5(내보내기 옵션)다.

**대조 클립은 빈 파일로 온다.** ELAN이 내보내는 것은 트랙이 아니라 구간이므로,
사건이 하나도 없는 클립은 행이 한 줄도 없다. 그게 정상이고 오류가 아니다 —
그래서 클립의 존재는 `clips.csv`로만 알 수 있다(#19 재리뷰 🔴1, #22 재리뷰 🔴).

사용:
    python3 -m eval.elan2csv --in _local/dataset/elan --clips _local/dataset/clips.csv \\
                             --annotator B --out _local/dataset/labels.csv
    python3 -m eval.elan2csv --selftest
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .labels import CATEGORIES, validate
from .labels import load as load_labels

# §1 스키마. **`duration`은 넣지 않는다** — `offset − onset`으로 계산한다.
FIELDS = ["video_id", "clip_id", "clip_offset", "onset", "offset",
          "category", "ambiguous", "annotator", "note"]

# 「?」는 맨 앞에 온다(elan-setup §4). 전각 물음표도 받는다 — 한글 입력 상태에서 나온다.
AMBIGUOUS_MARK = re.compile(r"^[?？]\s*")

# `merged×2`의 ×(U+00D7)는 macOS에서 바로 안 나온다. x·*·× 셋 다 받아 하나로 맞춘다.
MERGED = re.compile(r"^merged\s*[x*×]\s*(\d+)$", re.IGNORECASE)

MAX_PLAUSIBLE_S = 3600.0   # 클립은 5분이다. 초 단위 값이 이보다 크면 단위를 잘못 읽은 것


@dataclass
class Row:
    clip_id: str
    category: str
    onset: float
    offset: float
    ambiguous: bool
    note: str


class ConvertError(Exception):
    """변환을 계속하면 조용히 틀린 라벨이 나오는 경우에만 던진다."""


# ── 시간 파싱 ────────────────────────────────────────────────────────────
# 시간 칸의 종류. **ELAN은 켜놓은 형식을 모두 열로 낸다** — 실제 내보내기에서
# 시작 시각 하나가 `00:01:03.340` · `63.34` · `63340` · `00:01:03:08` 네 열로 나왔다.
# 그래서 "시간처럼 생긴 칸"을 순서대로 집으면 시작과 끝이 같은 값이 된다.
# 종류별로 모아서 **한 종류만 골라** 쓴다.
PREFER = ("sec", "hms", "int")   # 소수 초 > hh:mm:ss.mmm > 정수(밀리초일 수 있음)


def _classify(cell: str) -> tuple[str, float] | None:
    """(종류, 초) 또는 None. SMPTE는 값을 못 믿으므로 종류만 표시한다."""
    c = cell.strip()
    if not c:
        return None
    colons = c.count(":")
    if colons >= 3:                            # SMPTE hh:mm:ss:ff — 프레임률이 필요하다
        return ("smpte", 0.0)
    if colons:                                 # hh:mm:ss.mmm 또는 mm:ss.mmm
        try:
            nums = [float(x) for x in c.split(":")]
        except ValueError:
            return None
        total = 0.0
        for n in nums:
            total = total * 60 + n
        return ("hms", total)
    if "." in c:
        try:
            return ("sec", float(c))
        except ValueError:
            return None
    if c.lstrip("-").isdigit():
        return ("int", float(c))
    return None


def _looks_like_path(cell: str) -> bool:
    c = cell.strip().lower()
    return c.endswith((".eaf", ".mp4", ".wav", ".mov", ".mkv")) or "/" in c


# ── 라벨 텍스트 → ambiguous · note ──────────────────────────────────────
def parse_label_text(text: str) -> tuple[bool, str]:
    """elan-setup.md §4 표 그대로.

    | ELAN 라벨 텍스트 | ambiguous | note |
    |---|---|---|
    | (비어 있음) | false | (빈 칸) |
    | `?` | true | (빈 칸) |
    | `merged×2` | false | `merged×2` |
    | `? 케첩인지` | true | `케첩인지` |
    """
    t = (text or "").strip()
    ambiguous = bool(AMBIGUOUS_MARK.match(t))
    if ambiguous:
        t = AMBIGUOUS_MARK.sub("", t).strip()
    m = MERGED.match(t)
    if m:
        t = f"merged×{m.group(1)}"             # 표기를 하나로 맞춘다
    return ambiguous, t


# ── 한 파일 ─────────────────────────────────────────────────────────────
def _tier_name(cell: str) -> str | None:
    """트랙 이름 칸인지 본다. 대소문자가 달라도 받아준다.

    elan-setup.md가 이름을 손으로 치면 `jumpScare`가 된다고 경고하는데,
    거기서 조용히 건너뛰면 사건이 통째로 사라진다. 받아주고 알려주는 쪽이 낫다.
    """
    c = cell.strip()
    if c in CATEGORIES:
        return c
    low = c.lower()
    for known in CATEGORIES:
        if low == known.lower():
            return known
    return None


def parse_export(path: Path, clip_id: str) -> tuple[list[Row], list[str]]:
    """탭 구분 텍스트 한 개를 읽는다. (행 목록, 알림 목록)을 돌려준다.

    **열 위치를 고정하지 않는다.** ELAN 대화상자에서 무엇을 고르느냐에 따라
    파일 이름 열이 붙기도 하고 `duration` 열이 끼기도 한다. 그래서 각 줄에서
    트랙 이름처럼 생긴 칸과 시간처럼 생긴 칸을 찾아 쓴다.

    **빈 라벨 텍스트는 열이 아예 없을 수 있다.** 대부분의 라벨이 빈 칸이므로
    (elan-setup §4) 열 개수로 파싱하면 대부분의 줄에서 깨진다 (#22 재리뷰 🟡).
    """
    notes: list[str] = []
    raw = path.read_text(encoding="utf-8-sig").splitlines()
    parsed: list[tuple[str, float, float, str]] = []
    ints_only = True

    for ln, line in enumerate(raw, start=1):
        if not line.strip():
            continue
        cells = line.split("\t")

        tier, tier_at = None, -1
        for i, c in enumerate(cells):
            tier = _tier_name(c)
            if tier:
                tier_at = i
                if c.strip() != tier:
                    notes.append(f"{path.name}:{ln} 트랙 이름 '{c.strip()}'을 "
                                 f"'{tier}'로 읽었다")
                break
        if not tier:
            known = [c.strip() for c in cells if c.strip()]
            if known:
                notes.append(f"{path.name}:{ln} 트랙 이름이 없어 건너뛴다 — {known[:2]}")
            continue

        by_kind: dict[str, list[tuple[int, float]]] = {}
        last_time_at = tier_at
        for i in range(tier_at + 1, len(cells)):
            got = _classify(cells[i])
            if got is None:
                continue
            kind, val = got
            by_kind.setdefault(kind, []).append((i, val))
            last_time_at = i

        use = next((k for k in PREFER if len(by_kind.get(k, [])) >= 2), None)
        if use is None:
            if "smpte" in by_kind:
                raise ConvertError(
                    f"{path.name}:{ln} 시간이 SMPTE 형식뿐이다 ('{cells[by_kind['smpte'][0][0]]}'). "
                    f"프레임률을 알아야 초로 바꿀 수 있다 — 내보내기에서 초 단위(ss.msec)나 "
                    f"hh:mm:ss.ms를 함께 켜서 다시 내보내라 (elan-setup.md §5)")
            raise ConvertError(f"{path.name}:{ln} 시작·끝 시각을 찾지 못했다 — {cells}")
        if use == "int":
            ints_only = True
        else:
            ints_only = False

        picked = by_kind[use]
        onset, offset = picked[0][1], picked[1][1]
        # `duration` 열이 켜져 있으면 세 번째 값이 끝−시작과 같다. 무시한다(§1).
        if len(picked) > 2 and abs(picked[2][1] - (offset - onset)) < 1e-3:
            notes.append(f"{path.name}:{ln} duration 열이 있다 — 무시한다 (§1)")
        if len(by_kind) > 1:
            notes.append(f"{path.name}:{ln} 시간 형식 {sorted(by_kind)} 중 "
                         f"'{use}'를 쓴다")

        # 라벨 텍스트 = **모든 시간 칸 뒤**에 남은 첫 칸. 파일 이름 열은 건너뛴다.
        text = next((c for i, c in enumerate(cells)
                     if i > last_time_at and c.strip() and not _looks_like_path(c)), "")
        parsed.append((tier, onset, offset, text))

    # 단위 — 정수만 나오고 값이 너무 크면 밀리초다
    scale = 1.0
    if parsed and ints_only:
        biggest = max(max(on, off) for _, on, off, _ in parsed)
        if biggest > MAX_PLAUSIBLE_S:
            scale = 1e-3
            notes.append(f"{path.name}: 시간이 정수이고 최대 {biggest:.0f}이라 "
                         f"밀리초로 읽었다 — 초로 내보내는 게 규약이다 (elan-setup §5)")

    rows: list[Row] = []
    for tier, onset, offset, text in parsed:
        amb, note = parse_label_text(text)
        rows.append(Row(clip_id=clip_id, category=tier,
                        onset=round(onset * scale, 3), offset=round(offset * scale, 3),
                        ambiguous=amb, note=note))
    return rows, notes


# ── 클립 명세 ───────────────────────────────────────────────────────────
def load_manifest(path: Path) -> dict[str, tuple[str, float]]:
    """`clips.csv`에서 clip_id → (video_id, clip_offset)을 읽는다.

    §1의 `video_id`·`clip_offset`은 ELAN 파일에 없다. 원본과 대조할 유일한
    단서라 명세에서 가져와야 한다.
    """
    out: dict[str, tuple[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            cid = (r.get("clip_id") or "").strip()
            if not cid:
                raise ConvertError(f"{path}:{i} clip_id가 비어 있다")
            try:
                off = float(r.get("clip_offset") or 0)
            except ValueError as e:
                raise ConvertError(f"{path}:{i} clip_offset을 읽지 못했다 — {e}") from e
            out[cid] = ((r.get("video_id") or "").strip(), off)
    if not out:
        raise ConvertError(f"{path}: 클립이 없다")
    return out


def clip_id_of(path: Path) -> str:
    """파일 이름에서 clip_id를 읽는다. `c007_export.txt` → `c007`."""
    stem = path.stem
    for tail in ("_export", "-export", "_라벨", "-라벨"):
        if stem.lower().endswith(tail.lower()):
            stem = stem[: -len(tail)]
    return stem.strip()


def resolve_clip_id(stem: str, known: list[str]) -> str | None:
    """파일 이름을 명세의 clip_id에 맞춘다.

    내보낼 때 이름에 접미사가 붙는다 — `c001_s.txt`·`c001 사본.txt`처럼.
    명세에 있는 id 중 **이름의 접두사인 가장 긴 것**을 쓴다.
    """
    if stem in known:
        return stem
    cands = [k for k in known if stem.startswith(k)]
    return max(cands, key=len) if cands else None


def convert(paths: list[Path], manifest: dict[str, tuple[str, float]] | None,
            annotator: str) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    notes: list[str] = []
    for p in sorted(paths):
        cid = clip_id_of(p)
        got, n = parse_export(p, cid)
        notes += n
        vid, off = ("", 0.0)
        if manifest is not None:
            hit = resolve_clip_id(cid, sorted(manifest))
            if hit is None:
                raise ConvertError(
                    f"{p.name}: clip_id '{cid}'를 클립 명세에서 찾지 못했다. 파일 이름이 "
                    f"clip_id로 시작해야 한다 (있는 것: {sorted(manifest)[:5]}…)")
            if hit != cid:
                notes.append(f"{p.name}: 이름에서 clip_id '{hit}'를 읽었다")
                cid = hit
            vid, off = manifest[cid]
        if not got:
            notes.append(f"{p.name}: 사건 0건 — 대조 클립이면 정상이다")
        for r in got:
            # 명세에서 푼 cid를 쓴다 — parse_export는 파일 이름 그대로 받았다
            rows.append({"video_id": vid, "clip_id": cid, "clip_offset": off,
                         "onset": r.onset, "offset": r.offset, "category": r.category,
                         "ambiguous": str(r.ambiguous).lower(), "annotator": annotator,
                         "note": r.note})
    rows.sort(key=lambda r: (r["clip_id"], r["onset"], r["category"]))
    return rows, notes


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


# ── 자체 점검 ───────────────────────────────────────────────────────────
SELFTEST_CASES = {
    # §5 규약대로 내보낸 것 — 초 단위, 열 넷, 빈 라벨은 열이 없다
    "c001.txt": "jumpscare\t5.000\t5.600\n"
                "jumpscare\t12.000\t12.600\t?\n"
                "blood\t20.100\t24.800\t? 케첩인지\n",
    # 파일 이름 열 + duration 열이 붙은 경우, 시각이 hh:mm:ss.mmm
    "c002_export.txt": "c002.eaf\tjumpscare\t00:00:08.250\t00:00:08.900\t00:00:00.650\tmerged x2\n"
                       "siren\t00:01:02.000\t00:01:09.500\t00:00:07.500\t\n",
    # 밀리초로 내보낸 경우 + 트랙 이름 대소문자가 틀린 경우
    "c003.txt": "jumpScare\t15000\t15700\n"
                "spider\t90000\t96000\t？\n",
    # 대조 클립 — 빈 파일이 정상이다
    "c004.txt": "",
}


def _selftest() -> int:
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="elan2csv-selftest-"))
    for name, body in SELFTEST_CASES.items():
        (tmp / name).write_text(body, encoding="utf-8")
    man = tmp / "clips.csv"
    man.write_text("clip_id,video_id,clip_offset\n"
                   "c001,aqz-KE-bpKQ,742.0\nc002,vid2,10.5\nc003,vid3,0\nc004,vid4,300\n")

    rows, notes = convert(sorted(tmp.glob("c0*.txt")), load_manifest(man), "B")
    out = tmp / "labels.csv"
    write_csv(out, rows)

    print(f"입력 {len(SELFTEST_CASES)}개 → 라벨 {len(rows)}건")
    for n in notes:
        print("  알림:", n)
    print()
    print(out.read_text().rstrip())
    print()

    got = {(r["clip_id"], r["category"], r["onset"], r["offset"],
            r["ambiguous"], r["note"]) for r in rows}
    want = {
        ("c001", "jumpscare", 5.0, 5.6, "false", ""),
        ("c001", "jumpscare", 12.0, 12.6, "true", ""),
        ("c001", "blood", 20.1, 24.8, "true", "케첩인지"),
        ("c002", "jumpscare", 8.25, 8.9, "false", "merged×2"),
        ("c002", "siren", 62.0, 69.5, "false", ""),
        ("c003", "jumpscare", 15.0, 15.7, "false", ""),      # 밀리초 → 초
        ("c003", "spider", 90.0, 96.0, "true", ""),          # 전각 물음표
    }
    checks = {
        "값이 기대와 같다": got == want,
        "duration 열이 없다": "duration" not in FIELDS,
        "대조 클립(c004)은 행이 0건": not any(r["clip_id"] == "c004" for r in rows),
        "clip_offset이 명세에서 왔다":
            all(float(r["clip_offset"]) == 742.0 for r in rows if r["clip_id"] == "c001"),
        "파일 이름의 _export가 떨어졌다": any(r["clip_id"] == "c002" for r in rows),
        "하니스가 이 CSV를 읽는다": len(load_labels(out)) == len(rows),
    }
    if got != want:
        print("  기대에 없는 것:", sorted(got - want))
        print("  빠진 것:", sorted(want - got))

    problems = validate(load_labels(out))
    print("규약 검사:", "통과" if not problems else f"{len(problems)}건")
    for p in problems:
        print("  ", p)

    for name, ok in checks.items():
        print(f"  [{'OK' if ok else '실패'}] {name}")
    allok = all(checks.values())
    print("\n자체 점검:", "통과" if allok else "실패")
    return 0 if allok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="ELAN 내보내기 → 라벨 CSV (labeling-guide §1)")
    ap.add_argument("--in", dest="inp", nargs="*", default=[],
                    help="탭 구분 텍스트 파일들, 또는 그것들이 든 디렉터리")
    ap.add_argument("--clips", help="클립 명세 CSV — video_id·clip_offset을 여기서 가져온다")
    ap.add_argument("--annotator", help="라벨을 단 사람 (§1). 이중 라벨링 구분에 쓴다")
    ap.add_argument("--out", help="출력 CSV 경로 (없으면 화면에 낸다)")
    ap.add_argument("--selftest", action="store_true", help="합성 내보내기로 변환 점검")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not a.inp:
        ap.error("--in 이 필요하다 (또는 --selftest)")
    if not a.annotator:
        ap.error("--annotator 가 필요하다 — 이중 라벨링을 구분할 수 없으면 "
                 "평가에서 참 사건이 두 번 세진다 (§6)")

    paths: list[Path] = []
    for item in a.inp:
        p = Path(item)
        paths += sorted(p.glob("*.txt")) + sorted(p.glob("*.tsv")) if p.is_dir() else [p]
    paths = [p for p in paths if p.name != "clips.csv"]
    if not paths:
        print("입력 파일이 없다", file=sys.stderr)
        return 1

    manifest = None
    if a.clips:
        manifest = load_manifest(Path(a.clips))
    else:
        print("경고: --clips가 없어 video_id와 clip_offset이 빈 값이 된다. "
              "§1이 요구하는 열이고 원본과 대조할 유일한 단서다", file=sys.stderr)

    try:
        rows, notes = convert(paths, manifest, a.annotator)
    except ConvertError as e:
        print(f"변환 실패 — {e}", file=sys.stderr)
        return 1

    for n in notes:
        print(f"알림: {n}", file=sys.stderr)

    if a.out:
        out = Path(a.out)
        write_csv(out, rows)
        print(f"라벨 {len(rows)}건 → {out}")
        problems = validate(load_labels(out))
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
        problems = []

    if problems:
        print(f"\n규약 위반 {len(problems)}건 — 고치고 다시 내보내라:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
    n_clips = len({r["clip_id"] for r in rows})
    blanks = sum(1 for n in notes if "사건 0건" in n)
    print(f"클립 {n_clips}개에서 사건 {len(rows)}건 · 사건 0건인 클립 {blanks}개",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
