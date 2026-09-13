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

from .labels import CATEGORIES, Label, validate
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


def parse_export(path: Path, clip_id: str,
                 clip_dur: float | None = None) -> tuple[list[Row], list[str]]:
    """탭 구분 텍스트 한 개를 읽는다. (행 목록, 알림 목록)을 돌려준다.

    **열 위치를 고정하지 않는다.** ELAN 대화상자에서 무엇을 고르느냐에 따라
    파일 이름 열이 붙기도 하고 `duration` 열이 끼기도 한다. 그래서 각 줄에서
    트랙 이름처럼 생긴 칸과 시간처럼 생긴 칸을 찾아 쓴다.

    **빈 라벨 텍스트는 열이 아예 없을 수 있다.** 대부분의 라벨이 빈 칸이므로
    (elan-setup §4) 열 개수로 파싱하면 대부분의 줄에서 깨진다 (#22 재리뷰 🟡).
    """
    notes: list[str] = []
    raw = path.read_text(encoding="utf-8-sig").splitlines()
    parsed: list[tuple[int, str, float, float, str]] = []   # (줄 번호, 트랙, onset, offset, 텍스트)
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
        # 파일 전체가 정수일 때만 밀리초 후보다. 줄마다 덮어쓰면 마지막 줄이 파일 전체의
        # 단위를 정한다(#27 리뷰 🟡9)
        ints_only = ints_only and use == "int"

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
        parsed.append((ln, tier, onset, offset, text))

    # 단위 — **클립 길이를 알면 그것으로 판정한다.** 5분 클립에 onset 2500초는 그 자체로
    # 불가능하다. 3600초 상한만 쓰면 사건이 전부 앞 3.6초 안인 밀리초 파일이 조용히
    # 1000배 틀린다(#27 리뷰 🔴1). 길이를 모를 때만 3600초 휴리스틱으로 떨어진다.
    scale = 1.0
    limit = (clip_dur + 1.0) if clip_dur else MAX_PLAUSIBLE_S
    if parsed:
        big_ln, biggest = max(((ln, max(on, off)) for ln, _, on, off, _ in parsed), key=lambda x: x[1])
        if clip_dur and ints_only:
            # **클립 길이를 알고 파일 전체가 정수면 크기와 무관하게 밀리초다.** 크기로만 가르면
            # 사건이 클립 앞 0.3초 안에 통째로 든 밀리초 파일(200/260)이 여전히 초로 읽힌다
            # (#27 재리뷰 🟡2). ELAN의 초·hh:mm:ss.ms는 소수를 쓰고, 실제 내보내기(형식 전부
            # 켬)에서는 소수 초 열이 먼저 골라져 이 경로에 안 온다. 끝자리 0을 지우는 ELAN이
            # 정수처럼 보이는 초를 낼 수는 있지만(`64.4`), 파일의 모든 값이 정확히 0 밀리초일
            # 일은 사실상 없다.
            scale = 1e-3
            notes.append(f"{path.name}: 시간이 전부 정수라 밀리초로 읽었다(클립 길이 {clip_dur:.0f}초 기준) "
                         f"— 초로 내보내는 게 규약이다 (elan-setup §5)")
            if biggest * scale > limit:
                raise ConvertError(
                    f"{path.name}:{big_ln} 밀리초로 읽어도 {biggest * scale:.2f}초가 클립 길이 "
                    f"{clip_dur:.0f}초를 넘는다 — 오타이거나 다른 클립의 파일이다")
        elif biggest > limit:
            if ints_only and biggest * 1e-3 <= limit:
                scale = 1e-3
                notes.append(f"{path.name}: 시간이 정수이고 최대 {biggest:.0f}이라 상한 "
                             f"{MAX_PLAUSIBLE_S:.0f}초를 넘어 밀리초로 읽었다 — 초로 내보내는 게 규약이다 "
                             f"(elan-setup §5)")
            elif clip_dur:
                # 줄 번호를 달고 원인에 오타를 넣는다 — 실제로는 이쪽이 제일 흔하다(#27 재리뷰 🟡3)
                raise ConvertError(
                    f"{path.name}:{big_ln} 사건 시각 {biggest:.2f}가 클립 길이 {clip_dur:.0f}초를 넘는다 — "
                    f"오타이거나(예: 248.0 → 2480.0) 다른 클립의 파일이거나 시간 형식이 섞였다")

    rows: list[Row] = []
    for _ln, tier, onset, offset, text in parsed:
        amb, note = parse_label_text(text)
        # 규약은 「?」를 맨 앞에 쓰는 것이다. 뒤에 붙인 것은 조용히 넘기지 않고 알린다 —
        # 가장 어기기 쉬운 사람이 처음 쓰는 제3자다(#27 리뷰 🟡10)
        if not amb and (text or "").strip().endswith(("?", "？")):
            notes.append(f"{path.name}: 라벨 텍스트 '{text.strip()}' 끝의 '?'는 ambiguous로 "
                         f"읽지 않았다 — '?'는 맨 앞에 쓴다 (elan-setup §4)")
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


def load_durations(path: Path) -> dict[str, float]:
    """`clips.csv`의 `duration_s`(실측 클립 길이) — 단위 판정에 쓴다."""
    out: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = (r.get("clip_id") or "").strip()
            try:
                d = float(r.get("duration_s") or 0)
            except ValueError:
                d = 0.0
            if cid and d > 0:
                out[cid] = d
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
            annotator: str,
            durations: dict[str, float] | None = None) -> tuple[list[dict], list[str], list[str]]:
    """(라벨 행, 알림, **검토한 clip_id 목록**).

    세 번째가 중요하다. **사건 0건인 대조 클립은 라벨 행을 남기지 않으므로**,
    라벨 CSV만으로는 「검토했고 아무것도 없었다」와 「아직 안 했다」를 구별할 수
    없다. 평가 분모는 이 목록이어야 한다 (#19 재리뷰 🔴1).
    """
    rows: list[dict] = []
    notes: list[str] = []
    seen: list[str] = []
    src: dict[str, str] = {}
    errors: list[str] = []
    for p in sorted(paths):
      # 파일 하나의 오류로 배치 전체를 멈추지 않는다. 34개 중 12번째에서 멈추면 나머지 22개의
      # 상태를 모른 채 고치고 다시 돌리기를 반복하게 된다(#27 재리뷰 🟡3). 전부 보고 한 번에 낸다
      try:
        cid = clip_id_of(p)
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
        # 같은 클립을 가리키는 파일이 둘이면 사건이 두 번 세진다(#27 리뷰 🟡8). 다시 내보내면
        # c001.txt와 c001_s.txt가 함께 남는 일이 실제로 있다
        if cid in src:
            raise ConvertError(f"{p.name}와 {src[cid]}가 둘 다 clip_id '{cid}'를 가리킨다 — "
                               f"같은 사건이 두 번 세진다. 하나를 지우고 다시 돌려라")
        src[cid] = p.name
        got, n = parse_export(p, cid, (durations or {}).get(cid))
        notes += n
        seen.append(cid)
        if not got:
            notes.append(f"{p.name}: 사건 0건 — 대조 클립이면 정상이다")
        for r in got:
            # 명세에서 푼 cid를 쓴다 — parse_export는 파일 이름 그대로 받았다
            rows.append({"video_id": vid, "clip_id": cid, "clip_offset": off,
                         "onset": r.onset, "offset": r.offset, "category": r.category,
                         "ambiguous": str(r.ambiguous).lower(), "annotator": annotator,
                         "note": r.note})
      except ConvertError as e:
        errors.append(str(e))
    if errors:
        raise ConvertError(f"파일 {len(errors)}개에서 멈췄다 — 나머지 {len(seen)}개는 읽혔다\n  "
                           + "\n  ".join(errors))
    rows.sort(key=lambda r: (r["clip_id"], r["onset"], r["category"]))
    return rows, notes, seen


def write_clip_subset(src: Path, dest: Path, seen: list[str]) -> int:
    """클립 명세에서 **검토한 클립만** 남긴 부분집합을 쓴다.

    20개에서 멈추면 평가셋은 34개가 아니라 그 20개다. 대조 비율의 분모도
    이쪽이다. 열은 원본 명세를 그대로 가져간다.
    """
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        keep = [r for r in reader if (r.get("clip_id") or "").strip() in set(seen)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    return len(keep)


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
    man.write_text("clip_id,video_id,clip_offset,duration_s\n"
                   "c001,aqz-KE-bpKQ,742.0,300\nc002,vid2,10.5,300\nc003,vid3,0,300\nc004,vid4,300,300\n")

    rows, notes, seen = convert(sorted(tmp.glob("c0*.txt")), load_manifest(man), "B", load_durations(man))
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
        "대조 클립도 검토 목록에는 들어간다": "c004" in seen and len(seen) == 4,
        "clip_offset이 명세에서 왔다":
            all(float(r["clip_offset"]) == 742.0 for r in rows if r["clip_id"] == "c001"),
        "파일 이름의 _export가 떨어졌다": any(r["clip_id"] == "c002" for r in rows),
        "하니스가 이 CSV를 읽는다": len(load_labels(out)) == len(rows),
    }
    # 🔴1 — 5분 클립을 밀리초로 내보냈고 사건이 앞 3.6초 안에 있다
    (tmp / "ms").mkdir()
    (tmp / "ms" / "c003.txt").write_text("jumpscare\t2500\t3100\n", encoding="utf-8")
    r_dur, _ = parse_export(tmp / "ms" / "c003.txt", "c003", 300.0)
    r_nodur, _ = parse_export(tmp / "ms" / "c003.txt", "c003", None)
    checks["클립 길이로 밀리초를 판정한다 (2500 → 2.5초)"] = r_dur[0].onset == 2.5
    checks["길이를 모를 때만 3600초 상한으로 떨어진다"] = r_nodur[0].onset == 2500.0
    # 🟡8 — 같은 클립을 가리키는 파일 둘
    (tmp / "dup").mkdir()
    for nm in ("c001.txt", "c001_s.txt"):
        (tmp / "dup" / nm).write_text("jumpscare\t5.0\t5.6\n", encoding="utf-8")
    try:
        convert(sorted((tmp / "dup").glob("*.txt")), load_manifest(man), "B", load_durations(man))
        dup_raised = False
    except ConvertError:
        dup_raised = True
    checks["같은 클립 파일이 둘이면 멈춘다"] = dup_raised
    # 🟡9 — 줄 순서를 뒤집어도 판정이 같다(형식이 섞이면 둘 다 멈춘다)
    def _raises(body: str) -> bool:
        f = tmp / "mix.txt"
        f.write_text(body, encoding="utf-8")
        try:
            parse_export(f, "c003", 300.0)
            return False
        except ConvertError:
            return True
    checks["줄 순서와 무관한 단위 판정"] = (
        _raises("siren\t90000\t96000\njumpscare\t00:00:15.000\t00:00:15.700\n")
        and _raises("jumpscare\t00:00:15.000\t00:00:15.700\nsiren\t90000\t96000\n"))
    # 🟡10 — 뒤에 붙은 ?
    (tmp / "tq").mkdir()
    (tmp / "tq" / "c001.txt").write_text("blood\t1.0\t2.0\t케첩인지?\n", encoding="utf-8")
    _, tq_notes = parse_export(tmp / "tq" / "c001.txt", "c001", 300.0)
    checks["뒤에 붙은 ?를 알린다"] = any("끝의 '?'" in n for n in tq_notes)

    # #27 재리뷰 🟡2 — 클립 길이를 알고 파일 전체가 정수면 크기와 무관하게 밀리초다
    import tempfile as _tf
    _t = Path(_tf.mkdtemp(prefix="elan2csv-units-"))
    (_t / "c001.txt").write_text("jumpscare\t200\t260\n", encoding="utf-8")
    r_small, _ = parse_export(_t / "c001.txt", "c001", 300.0)
    checks["정수 파일은 크기와 무관하게 밀리초로 읽는다 (200 → 0.2초)"] = r_small[0].onset == 0.2

    # #27 재리뷰 🟡3 — 오타는 줄 번호를 달고, 파일 하나가 배치를 멈추지 않고 전부 보고한다
    (_t / "c002.txt").write_text("jumpscare\t5.0\t5.6\nblood\t20.1\t2480.0\n", encoding="utf-8")
    (_t / "c003.txt").write_text("siren\t10.5\t9999.0\n", encoding="utf-8")
    try:
        convert([_t / "c002.txt", _t / "c003.txt"], {"c002": ("v", 0.0), "c003": ("v", 0.0)}, "B",
                {"c002": 300.0, "c003": 300.0})
        batch_msg = ""
    except ConvertError as e:
        batch_msg = str(e)
    checks["오타는 줄 번호를 달고 배치 전체를 한 번에 보고한다"] = (
        "c002.txt:2" in batch_msg and "c003.txt:1" in batch_msg and "오타" in batch_msg)

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
    ap.add_argument("--out-clips", help="검토한 클립만 담은 명세를 쓸 경로 — "
                                       "E1의 --clips로 쓴다 (대조 비율의 분모)")
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
    durations = None
    if a.clips:
        manifest = load_manifest(Path(a.clips))
        durations = load_durations(Path(a.clips))
    else:
        print("경고: --clips가 없어 video_id와 clip_offset이 빈 값이 된다. "
              "§1이 요구하는 열이고 원본과 대조할 유일한 단서다. "
              "그리고 **단위 판정이 3600초 상한으로 떨어진다** — 사건이 앞 3.6초 안에 몰린 "
              "밀리초 파일은 1000배 틀린 채 알림 없이 나온다", file=sys.stderr)

    try:
        rows, notes, seen = convert(paths, manifest, a.annotator, durations)
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
        # 화면으로만 볼 때도 규약 검사를 한다 — 처음 한 번 확인하는 그때가 제일 필요하다(#27 리뷰 🟡10)
        problems = validate([Label(video_id=r["video_id"], clip_id=r["clip_id"],
                                   clip_offset=float(r["clip_offset"]), onset=float(r["onset"]),
                                   offset=float(r["offset"]), category=r["category"],
                                   ambiguous=r["ambiguous"] == "true", annotator=r["annotator"],
                                   note=r["note"]) for r in rows])

    if problems:
        print(f"\n규약 위반 {len(problems)}건 — 고치고 다시 내보내라:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
    n_clips = len({r["clip_id"] for r in rows})
    blanks = sum(1 for n in notes if "사건 0건" in n)
    print(f"검토한 클립 {len(seen)}개 · 사건 {len(rows)}건 · "
          f"사건 있는 클립 {n_clips}개 · 사건 0건인 클립 {blanks}개", file=sys.stderr)
    if a.out_clips:
        if not a.clips:
            print("--out-clips는 --clips가 있어야 쓸 수 있다", file=sys.stderr)
            return 1
        n = write_clip_subset(Path(a.clips), Path(a.out_clips), seen)
        print(f"검토한 클립 명세 {n}개 → {a.out_clips}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
