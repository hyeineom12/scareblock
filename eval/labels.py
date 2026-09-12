"""라벨 로드·검증 — docs/labeling-guide.md §1 스키마를 그대로 읽는다."""
from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

CATEGORIES = {"jumpscare", "blood", "syringe", "spider", "siren"}
MERGE_GAP = 1.5  # §2 — 앞 offset에서 이 시간 안에 다음 onset이 오면 1건


@dataclass
class Label:
    video_id: str
    clip_id: str
    clip_offset: float
    onset: float
    offset: float
    category: str
    ambiguous: bool = False
    annotator: str = ""
    note: str = ""

    @property
    def duration(self) -> float:
        """§1 — duration은 기록하지 않고 계산한다."""
        return self.offset - self.onset

    def has_note(self, tag: str) -> bool:
        """note는 자유 메모다. 태그가 있으면 쓰고 없으면 무시한다."""
        return tag.lower() in self.note.lower()


@dataclass
class Clip:
    clip_id: str
    labels: list[Label] = field(default_factory=list)

    @property
    def is_control(self) -> bool:
        """§4 — 점프 스케어가 하나도 없으면 대조 클립."""
        return not any(x.category == "jumpscare" for x in self.labels)


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "y", "yes"}


def load(path: str | Path) -> list[Label]:
    """§1 스키마 CSV를 읽는다. 없는 선택 열은 기본값으로 채운다."""
    rows: list[Label] = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            try:
                rows.append(
                    Label(
                        video_id=r["video_id"].strip(),
                        clip_id=r["clip_id"].strip(),
                        clip_offset=float(r.get("clip_offset") or 0),
                        onset=float(r["onset"]),
                        offset=float(r["offset"]),
                        category=r["category"].strip(),
                        ambiguous=_to_bool(r.get("ambiguous", "")),
                        annotator=(r.get("annotator") or "").strip(),
                        note=(r.get("note") or "").strip(),
                    )
                )
            except (KeyError, ValueError) as e:
                raise ValueError(f"{path}:{i} 행을 읽지 못했다 — {e}") from e
    return rows


def load_clips(path: str | Path) -> list[str]:
    """클립 명세(`clips.csv`)에서 clip_id 목록을 읽는다.

    **라벨에서 클립 목록을 역산하면 안 된다.** 대조 클립은 사건이 없는 것이
    정상이라 라벨 행을 하나도 남기지 않는다 — 그래서 라벨을 분모로 쓰면
    대조 클립이 평가에서 통째로 빠지고 오탐이 한 건도 세지지 않는다
    (#19 리뷰 🔴1). 라벨은 「무엇이 있었나」를, 이 파일은 「무엇을 봤나」를 말한다.

    `clip_id` 열만 요구하고 나머지는 무시한다 — 명세가 열을 늘려도 안 깨진다.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "clip_id" not in (rows[0].keys() if rows else {}):
        raise ValueError(f"{path}: clip_id 열이 없다")
    out, seen = [], set()
    for i, r in enumerate(rows, start=2):
        cid = (r.get("clip_id") or "").strip()
        if not cid:
            raise ValueError(f"{path}:{i} clip_id가 비어 있다")
        if cid in seen:
            raise ValueError(f"{path}:{i} clip_id '{cid}'가 두 번 나온다")
        seen.add(cid)
        out.append(cid)
    return out


def pick_annotator(labels: list[Label], who: str) -> list[Label]:
    """한 주석자의 라벨만 남긴다.

    이중 라벨링(§6) CSV를 그대로 평가에 넣으면 같은 사건이 두 번 세지고,
    매칭이 1:1이라 둘 중 하나는 자동으로 미탐이 된다 (#19 리뷰 🔴2).
    두 주석자를 함께 읽는 것은 κ 계산뿐이다.
    """
    return [x for x in labels if x.annotator == who]


def replace_annotator(label: Label, who: str) -> Label:
    """주석자만 바꾼 사본. 자체 점검이 이중 라벨링 상황을 만들 때 쓴다."""
    return dataclasses.replace(label, annotator=who)


def annotators(labels: list[Label]) -> list[str]:
    """라벨에 등장하는 주석자. 빈 문자열도 한 명으로 센다."""
    return sorted({x.annotator for x in labels})


def validate(labels: list[Label]) -> list[str]:
    """라벨링 규약 위반을 찾는다. 빈 목록이면 통과."""
    problems: list[str] = []
    for x in labels:
        where = f"{x.clip_id} @{x.onset:.2f}"
        if x.category not in CATEGORIES:
            problems.append(f"{where}: 알 수 없는 category '{x.category}'")
        if x.offset <= x.onset:
            problems.append(f"{where}: offset이 onset보다 앞서거나 같다")

    # §2 — 같은 클립·같은 카테고리에서 MERGE_GAP 안에 붙은 두 건은 1건이어야 한다.
    # **주석자별로 본다.** 이중 라벨링(§6)에서 두 사람이 같은 사건을 각자 찍으면
    # 당연히 겹치는데, 그것까지 위반으로 세면 09.13 CSV가 경고로 뒤덮인다.
    key = lambda x: (x.annotator, x.clip_id, x.category, x.onset)  # noqa: E731
    ordered = sorted(labels, key=key)
    for a, b in zip(ordered, ordered[1:]):
        if (a.annotator, a.clip_id, a.category) != (b.annotator, b.clip_id, b.category):
            continue
        gap = b.onset - a.offset
        # 하한이 0이면 **겹치거나 완전히 똑같은 두 행이 검사를 그냥 통과한다**
        # (#19 리뷰 🟡4). 음수 간격은 병합이 아니라 별개의 오류다.
        if gap < 0:
            same = (a.onset, a.offset) == (b.onset, b.offset)
            problems.append(
                f"{a.clip_id}: {a.onset:.2f}와 {b.onset:.2f}가 "
                + ("완전히 같은 구간이다 — 중복 행" if same
                   else f"{-gap:.2f}s 겹친다 — §2는 겹침을 허용하지 않는다")
            )
        elif gap < MERGE_GAP:
            problems.append(
                f"{a.clip_id}: {a.onset:.2f}와 {b.onset:.2f}가 {gap:.2f}s 간격 — "
                f"§2에 따라 1건으로 병합해야 한다"
            )
    return problems


def by_clip(labels: list[Label], clip_ids: list[str] | None = None) -> dict[str, Clip]:
    """클립별 라벨. `clip_ids`를 주면 **라벨이 0건인 클립도 들어간다.**

    주지 않으면 라벨에 등장하는 클립만 보이고, 그러면 대조 클립이 사라진다
    (#19 리뷰 🔴1). 분모가 필요한 자리에서는 반드시 준다.
    """
    clips: dict[str, Clip] = {cid: Clip(cid) for cid in (clip_ids or [])}
    for x in labels:
        clips.setdefault(x.clip_id, Clip(x.clip_id)).labels.append(x)
    return clips


def control_ratio(clips: dict[str, Clip]) -> float:
    """§4 — 대조 클립 비율. 0.3 미만이면 정밀도 수치를 방어할 수 없다."""
    if not clips:
        return 0.0
    return sum(c.is_control for c in clips.values()) / len(clips)
