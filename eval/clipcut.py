"""클립 절단 — 무작위 오프셋으로 5분 클립을 뽑고 클립 명세를 낸다.

`labeling-guide §4`의 절단 규칙을 코드로 옮긴 것이다. 규칙 셋이 핵심이다.

1. **경계를 사건 기준으로 잡지 않는다** — 오프셋은 무작위다. 사람이 "여기 좋은
   장면 있다"고 고르면 클립 내 사건 밀도가 실제 영상보다 높아지고 precision이
   낙관적으로 나온다
2. **씨앗을 고정한다** — 같은 후보 목록 + 같은 씨앗이면 같은 오프셋이 나온다.
   06단계 재현 절차가 URL·타임스탬프·라벨만 공개하므로, 타임스탬프를 다시
   뽑을 수 있어야 한다
3. **`clips.csv`를 낸다** — 라벨이 0건인 대조 클립도 한 줄로 남는다.
   라벨은 "무엇이 있었나"만 말하므로 "무엇을 봤나"는 별도 파일이어야 한다
   (#19 재리뷰 🔴1)

사용:
    python3 -m eval.clipcut --candidates _local/candidates.txt --out _local/clips --dry-run
    python3 -m eval.clipcut --candidates _local/candidates.txt --out _local/clips
"""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CLIP_S = 300.0      # §4 — 클립 길이 5분 이내
MARGIN_S = 60.0     # 앞뒤로 비우는 구간 — 인트로 타이틀·엔딩 크레딧을 피한다
WAV_SR = 16000      # 라벨링·탐지기 공용. features.py가 모노를 기대한다
MANIFEST = "clips.csv"
FIELDS = ["clip_id", "video_id", "clip_offset", "duration_s", "source_class", "url"]


@dataclass
class Candidate:
    url: str
    n_clips: int = 1
    source_class: str = "scare"   # scare | neutral — 대조 클립 비율 관리용
    note: str = ""


def parse_candidates(path: Path) -> list[Candidate]:
    """한 줄에 하나. `<URL 또는 영상ID> [클립수] [분류] # 메모`

        https://youtu.be/aqz-KE-bpKQ  2  scare   # 공포 모음
        dQw4w9WgXcQ                   1  neutral # 조용한 구간 위주
    """
    out: list[Candidate] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line, _, note = raw.partition("#")
        parts = line.split()
        if not parts:
            continue
        url = parts[0]
        n = int(parts[1]) if len(parts) > 1 else 1
        cls = parts[2] if len(parts) > 2 else "scare"
        if cls not in {"scare", "neutral"}:
            raise ValueError(f"분류는 scare 또는 neutral이어야 한다 — '{cls}'")
        out.append(Candidate(url=url, n_clips=n, source_class=cls, note=note.strip()))
    return out


def probe(url: str) -> tuple[str, float]:
    """영상 ID와 길이(초)를 읽는다. 내려받지 않는다."""
    r = subprocess.run(["yt-dlp", "--no-warnings", "--print", "%(id)s\t%(duration)s", url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"길이를 읽지 못했다 — {url}\n{r.stderr.strip()[:300]}")
    vid, _, dur = r.stdout.strip().splitlines()[0].partition("\t")
    if not dur or dur == "NA":
        raise RuntimeError(f"길이가 없다(라이브·DRM 가능) — {url}")
    return vid, float(dur)


def draw_offsets(duration: float, n: int, rng: random.Random,
                 clip_s: float = CLIP_S, margin_s: float = MARGIN_S) -> list[float]:
    """겹치지 않는 무작위 오프셋 n개. 영상이 짧으면 가능한 개수만 돌려준다.

    앞뒤 `margin_s`를 비우는 것은 **사건 기준이 아니라 구조 기준**이다 —
    인트로 타이틀과 엔딩 크레딧은 어느 영상에나 있고 내용이 없다.
    """
    lo, hi = margin_s, duration - margin_s - clip_s
    if hi <= lo:                      # 여유가 없으면 마진을 포기하고 중앙에서 뽑는다
        lo, hi = 0.0, duration - clip_s
    if hi <= lo:
        return []
    picked: list[float] = []
    for _ in range(n * 40):           # 겹침 회피 — 시도 상한을 둔다
        if len(picked) == n:
            break
        t = rng.uniform(lo, hi)
        if all(abs(t - p) >= clip_s for p in picked):
            picked.append(t)
    return sorted(round(t, 2) for t in picked)


def cut(url: str, offset: float, clip_s: float, dest: Path) -> None:
    """해당 구간만 내려받는다. 원본 전체를 받지 않는다 (§7 로컬 분석 전용)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["yt-dlp", "--no-warnings", "-f", "bv*+ba/b",
         "--download-sections", f"*{offset}-{offset + clip_s}",
         "--force-keyframes-at-cuts",     # 구간 경계를 정확히 자른다
         "-o", str(dest), url],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"절단 실패 — {url} @{offset}\n{r.stderr.strip()[:300]}")


def to_wav(src: Path, dest: Path, sr: int = WAV_SR) -> None:
    """모노 wav를 뽑는다. ELAN 파형과 탐지기가 같은 파일을 본다."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ac", "1", "-ar", str(sr), "-vn", str(dest)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"wav 변환 실패 — {src}\n{r.stderr.strip()[:300]}")


def media_duration(path: Path) -> float:
    """실제로 받아진 길이. 요청한 5분과 다를 수 있다(영상 끝에 걸린 경우)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return round(float(r.stdout.strip()), 2)
    except ValueError:
        return 0.0


def write_manifest(path: Path, rows: list[dict]) -> None:
    """§4의 대조 클립 비율을 나중에 계산할 수 있는 유일한 근거다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="무작위 오프셋 클립 절단 (labeling-guide §4)")
    ap.add_argument("--candidates", required=True, help="후보 목록 파일")
    ap.add_argument("--out", required=True, help="산출 디렉터리 (mp4/ · wav/ · clips.csv)")
    ap.add_argument("--seed", type=int, default=20260912, help="오프셋 씨앗 — 재현용")
    ap.add_argument("--clip-s", type=float, default=CLIP_S)
    ap.add_argument("--margin-s", type=float, default=MARGIN_S)
    ap.add_argument("--start-index", type=int, default=1, help="clip_id 시작 번호")
    ap.add_argument("--dry-run", action="store_true", help="오프셋만 뽑고 내려받지 않는다")
    a = ap.parse_args()

    out = Path(a.out)
    cands = parse_candidates(Path(a.candidates))
    if not cands:
        print("후보가 없다", file=sys.stderr)
        return 1

    rows: list[dict] = []
    idx = a.start_index
    failed: list[str] = []

    for c in cands:
        try:
            vid, dur = probe(c.url)
        except RuntimeError as e:
            print(f"건너뜀 — {e}", file=sys.stderr)
            failed.append(c.url)
            continue

        # 씨앗을 영상 ID에 묶는다. 후보 목록의 순서가 바뀌어도 같은 오프셋이 나온다
        rng = random.Random(f"{a.seed}:{vid}")
        offsets = draw_offsets(dur, c.n_clips, rng, a.clip_s, a.margin_s)
        if not offsets:
            print(f"건너뜀 — 길이 {dur:.0f}초로는 {a.clip_s:.0f}초 클립을 못 뽑는다 ({vid})",
                  file=sys.stderr)
            failed.append(c.url)
            continue
        if len(offsets) < c.n_clips:
            print(f"알림 — {vid}에서 {c.n_clips}개 요청, {len(offsets)}개만 가능", file=sys.stderr)

        for off in offsets:
            clip_id = f"c{idx:03d}"
            idx += 1
            mp4 = out / "mp4" / f"{clip_id}.mp4"
            wav = out / "wav" / f"{clip_id}.wav"
            print(f"{clip_id}  {vid} @{off:.2f}s  ({c.source_class})")
            got = a.clip_s
            if not a.dry_run:
                try:
                    cut(c.url, off, a.clip_s, mp4)
                    to_wav(mp4, wav)
                    got = media_duration(mp4)
                except RuntimeError as e:
                    print(f"  실패 — {e}", file=sys.stderr)
                    failed.append(f"{vid}@{off}")
                    continue
            rows.append({"clip_id": clip_id, "video_id": vid, "clip_offset": off,
                         "duration_s": got, "source_class": c.source_class, "url": c.url})

    if not rows:
        print("만들어진 클립이 없다", file=sys.stderr)
        return 1

    manifest = out / MANIFEST
    if a.dry_run:
        print(f"\n[dry-run] 클립 {len(rows)}개 계획 — 명세를 쓰지 않았다")
    else:
        write_manifest(manifest, rows)
        print(f"\n클립 {len(rows)}개 · 명세 {manifest}")
    total = sum(r["duration_s"] for r in rows) / 60
    n_neutral = sum(r["source_class"] == "neutral" for r in rows)
    print(f"총 {total:.1f}분 · neutral 출처 {n_neutral}/{len(rows)}개 ({n_neutral / len(rows):.0%})")
    print("※ 대조 클립 비율은 라벨링 뒤에 확정된다 — neutral 출처라도 사건이 있을 수 있다")
    if failed:
        print(f"실패 {len(failed)}건: {failed}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
