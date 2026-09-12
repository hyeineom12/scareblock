"""라벨 로드·검증 — docs/labeling-guide.md §1 스키마를 그대로 읽는다."""
from __future__ import annotations

import csv
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


def validate(labels: list[Label]) -> list[str]:
    """라벨링 규약 위반을 찾는다. 빈 목록이면 통과."""
    problems: list[str] = []
    for x in labels:
        where = f"{x.clip_id} @{x.onset:.2f}"
        if x.category not in CATEGORIES:
            problems.append(f"{where}: 알 수 없는 category '{x.category}'")
        if x.offset <= x.onset:
            problems.append(f"{where}: offset이 onset보다 앞서거나 같다")

    # §2 — 같은 클립·같은 카테고리에서 MERGE_GAP 안에 붙은 두 건은 1건이어야 한다
    key = lambda x: (x.clip_id, x.category, x.onset)  # noqa: E731
    ordered = sorted(labels, key=key)
    for a, b in zip(ordered, ordered[1:]):
        if a.clip_id != b.clip_id or a.category != b.category:
            continue
        gap = b.onset - a.offset
        if 0 <= gap < MERGE_GAP:
            problems.append(
                f"{a.clip_id}: {a.onset:.2f}와 {b.onset:.2f}가 {gap:.2f}s 간격 — "
                f"§2에 따라 1건으로 병합해야 한다"
            )
    return problems


def by_clip(labels: list[Label]) -> dict[str, Clip]:
    clips: dict[str, Clip] = {}
    for x in labels:
        clips.setdefault(x.clip_id, Clip(x.clip_id)).labels.append(x)
    return clips


def control_ratio(clips: dict[str, Clip]) -> float:
    """§4 — 대조 클립 비율. 0.3 미만이면 정밀도 수치를 방어할 수 없다."""
    if not clips:
        return 0.0
    return sum(c.is_control for c in clips.values()) / len(clips)
