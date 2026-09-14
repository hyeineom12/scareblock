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
MEDIA_EXT = {".mp4", ".mkv", ".webm", ".mov"}   # 병합 산출물로 받아들일 확장자
MAX_H = 720         # M1 기준이 720p다. 라벨링에 그 이상은 쓸모가 없고 내려받기만 느려진다

# ELAN이 macOS에서 여는 조합으로 고정한다 — h264 + aac.
# vp9·av01은 컨테이너가 mp4여도 ELAN 파형·재생이 안 열릴 수 있다.
FMT = (f"bv*[vcodec^=avc1][height<={MAX_H}]+ba[ext=m4a]"
       f"/b[vcodec^=avc1][height<={MAX_H}]"
       f"/bv*[height<={MAX_H}]+ba/b[height<={MAX_H}]/bv*+ba/b")
MANIFEST = "clips.csv"

# 오프셋이 **무작위였다**는 것을 이 파일 하나로 말할 수 있어야 한다.
# 타임스탬프만 있으면 "그 값이었다"까지고, 사람이 골랐는지 무작위였는지는
# 구별되지 않는다 — §4 규칙의 존재 이유가 바로 그 구분이다 (#24 리뷰 🔴2).
# seed·clip_s·margin_s·video_duration_s 넷이 있으면 draw_offsets를 그대로 재실행할 수 있다.
FIELDS = ["clip_id", "video_id", "clip_offset", "duration_s", "source_class", "url",
          "seed", "clip_s", "margin_s", "video_duration_s"]


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
    # --no-playlist — 복사한 링크에 &list=가 붙어도 그 영상 하나만 본다(#24 리뷰 🟡)
    r = subprocess.run(["yt-dlp", "--no-warnings", "--no-playlist", "--print", "%(id)s\t%(duration)s", url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"길이를 읽지 못했다 — {url}\n{r.stderr.strip()[:300]}")
    vid, _, dur = r.stdout.strip().splitlines()[0].partition("\t")
    if not dur or dur == "NA":
        raise RuntimeError(f"길이가 없다(라이브·DRM 가능) — {url}")
    return vid, float(dur)


def is_whole(duration: float, clip_s: float) -> bool:
    """원본을 통째로 쓰는가. **`draw_offsets`와 내려받기가 같은 판정을 쓴다** — 둘이 갈리면
    구간 지정 없이 원본 전체를 받게 되고, 롱플레이면 몇 시간짜리다(#24 재리뷰 🟢6)."""
    return duration <= clip_s


def draw_offsets(duration: float, n: int, rng: random.Random,
                 clip_s: float = CLIP_S, margin_s: float = MARGIN_S) -> list[float]:
    """겹치지 않는 무작위 오프셋 n개. 영상이 짧으면 가능한 개수만 돌려준다.

    앞뒤 `margin_s`를 비우는 것은 **사건 기준이 아니라 구조 기준**이다 —
    인트로 타이틀과 엔딩 크레딧은 어느 영상에나 있고 내용이 없다.
    """
    # **원본이 클립 길이보다 짧거나 비슷하면 통째로 쓴다.** §4는 「5분 이내」이므로
    # 3분짜리 단편을 통째로 쓰는 것이 규칙에 맞고, 자를 지점을 고를 일이 없어
    # 표집 편향이 0이 된다. 단편은 클라이맥스가 끝에 있어 창을 뽑으면 놓치기 쉽다.
    if is_whole(duration, clip_s):
        return [0.0]
    lo, hi = margin_s, duration - margin_s - clip_s
    if hi <= lo:
        # 여유가 모자라면 마진을 **비례해서** 줄인다. 0으로 떨구면 인트로 타이틀을 피하려던
        # 목적이 그 경우에만 통째로 사라진다(#24 리뷰 🟢). 남는 여유의 1/4씩만 앞뒤로 비운다.
        m = (duration - clip_s) / 4
        lo, hi = m, duration - m - clip_s
    if hi <= lo:
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


def _pick_media(stem: Path) -> Path | None:
    """`cut()`과 `--manifest-only`가 **같은 규칙으로** 파일을 고른다(#24 리뷰 🟡1).

    yt-dlp가 병합에 실패하거나 끊기면 `c001.f137.mp4` 같은 포맷 조각이 남는다. 알파벳
    순으로는 조각이 먼저라 조용히 틀린 파일을 집는다. 조각을 거르고 가장 최근 것을 쓴다.
    """
    got = [q for q in stem.parent.glob(f"{stem.name}.*")
           if q.suffix.lower() in MEDIA_EXT and "." not in q.name[len(stem.name) + 1:]]
    return max(got, key=lambda q: q.stat().st_mtime) if got else None


def canonical_url(vid: str) -> str:
    """명세의 `url` 열에는 실제 URL을 쓴다 — §7 공개 방침의 「URL」이 이 열이다.
    후보 목록에 영상 ID만 적었거나 `&list=`가 붙어 있어도 같은 모양으로 맞춘다."""
    return f"https://www.youtube.com/watch?v={vid}"


def cut(url: str, offset: float, clip_s: float, stem: Path, exact: bool = False) -> Path:
    """해당 구간만 내려받는다. 원본 전체를 받지 않는다 (§7 로컬 분석 전용).

    확장자는 yt-dlp가 붙인다. `-o`에 `.mp4`를 박으면 실제 컨테이너가 webm일 때
    `c001.mp4.webm`이 나와 뒤 단계가 파일을 못 찾는다.

    `exact`는 구간 경계에 키프레임을 강제한다 — **정확하지만 전 구간을
    재인코딩하므로 4배 이상 느리다**(실측: 30초 클립 24초 → 105초).
    오프셋은 무작위라 경계 정밀도가 필요하지 않으므로 기본은 끈다.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["yt-dlp", "--no-warnings", "--no-playlist", "-f", FMT, "--merge-output-format", "mp4"]
    if clip_s > 0:
        cmd += ["--download-sections", f"*{offset}-{offset + clip_s}"]
    if exact:
        cmd.append("--force-keyframes-at-cuts")
    cmd += ["-o", f"{stem}.%(ext)s", url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"절단 실패 — {url} @{offset}\n{r.stderr.strip()[:300]}")
    # yt-dlp가 병합에 실패하거나 끊기면 c001.f137.mp4 같은 **포맷별 조각**이 남는다.
    # 알파벳 순으로는 f137이 mp4보다 앞이라 조각이 먼저 잡히고, 그러면 조용히
    # 틀린 파일로 wav를 뽑는다 (#24 리뷰 🟡4). 확장자를 아는 것으로 한정하고
    # 그 중 가장 최근 것을 집는다.
    got = _pick_media(stem)
    if got is None:
        raise RuntimeError(f"절단은 됐다는데 쓸 수 있는 파일이 없다 — {stem}.* "
                           f"(남은 것: {sorted(q.name for q in stem.parent.glob(f'{stem.name}.*'))})")
    return got


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
        # 옛 명세 행을 합쳐 쓸 때 열 구성이 다를 수 있다 — 없는 열은 빈 칸, 모르는 열은 버린다
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def merge_manifest(manifest: Path, rows: list[dict], start_index: int) -> list[dict]:
    """이번 실행의 행과 기존 명세를 합친 결과를 돌려준다. 명세는 쓰지 않는다 — `.bak`은 남길 수 있다.

    **기존 명세에 이번 실행이 안 만든 행이 있으면** 그게 명세가 날아가는 조건이다 —
    `--start-index`는 그 한 가지 방법일 뿐이다(#24 재리뷰 🟡). 그래서 남는 행은 항상 계산한다.

    - `start_index != 1` — 끝에 더하는 실행이다. 남는 행을 합친다
    - `start_index == 1`인데 남는 행이 있다 — `--start-index`를 깜빡했거나 후보 목록이 줄었다.
      덮어쓰기 전에 옛 파일을 `.bak`으로 남기고 크게 알린다. 정상적인 전체 재실행이면 남는 행이
      없어 조용하다
    """
    if not manifest.exists():
        return rows
    new_ids = {r["clip_id"] for r in rows}
    with open(manifest, newline="", encoding="utf-8") as f:
        keep = [r for r in csv.DictReader(f) if r.get("clip_id") not in new_ids]
    # DictReader가 낸 값은 전부 문자열이다. 합계를 내는 열은 숫자로 되돌린다 — 안 그러면
    # 명세를 쓴 **직후** 합계에서 죽어 성공한 실행이 실패로 보인다(#24 재리뷰 🔴)
    for r in keep:
        try:
            r["duration_s"] = float(r.get("duration_s") or 0)
        except ValueError:
            r["duration_s"] = 0.0
    if not keep:
        return rows
    if start_index != 1:
        print(f"알림 — --start-index {start_index}: 기존 명세 {len(keep)}행을 유지하고 "
              f"{len(rows)}행을 더한다", file=sys.stderr)
        return sorted(keep + rows, key=lambda r: r["clip_id"])
    # **이미 있는 .bak은 덮지 않는다.** 이 안전망이 필요한 사람은 경고를 못 본 사람이고, 그 사람은
    # 대개 한 번 더 돌린다 — 그때 좋은 백업(44행)이 방금 망가진 명세(10행)로 덮이면 복구가 불가능하다.
    # 가장 오래된 백업이 가장 귀하다(#24 재리뷰 🟡1)
    bak = manifest.with_name(manifest.name + ".bak")
    k = 1
    while bak.exists():
        k += 1
        bak = manifest.with_name(f"{manifest.name}.bak{k}")
    bak.write_bytes(manifest.read_bytes())
    # 원인은 하나가 아니다 — 영상 하나를 일시적으로 못 읽어도 그 클립 행이 남는다(#24 재리뷰 🟡3).
    # 옛 행을 이어받는 처리는 다음 PR로 두고, 여기서는 원인을 전부 열어 둔다
    print(f"경고 — 기존 명세에 이번 실행이 만들지 않은 행이 {len(keep)}개 있다 "
          f"({', '.join(sorted(r['clip_id'] for r in keep)[:5])}…). 덮어쓴다. 원인은 셋 중 하나다 — "
          f"--start-index를 깜빡했거나(끝에 더하려던 거면 주고 다시 돌린다) · 후보 목록이 줄었거나 · "
          f"이번에 읽기 실패한 영상이 있다(그 영상만 다시 돌린다). 옛 명세는 {bak.name}에 남겼다",
          file=sys.stderr)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="무작위 오프셋 클립 절단 (labeling-guide §4)")
    ap.add_argument("--candidates", required=True, help="후보 목록 파일")
    ap.add_argument("--out", required=True, help="산출 디렉터리 (video/ · wav/ · clips.csv)")
    ap.add_argument("--seed", type=int, default=20260912, help="오프셋 씨앗 — 재현용")
    ap.add_argument("--clip-s", type=float, default=CLIP_S)
    ap.add_argument("--margin-s", type=float, default=MARGIN_S)
    ap.add_argument("--start-index", type=int, default=1, help="clip_id 시작 번호")
    ap.add_argument("--exact-cuts", action="store_true",
                    help="구간 경계에 키프레임을 강제한다 — 4배 이상 느리다")
    ap.add_argument("--dry-run", action="store_true", help="오프셋만 뽑고 내려받지 않는다")
    ap.add_argument("--manifest-only", action="store_true",
                    help="이미 받아둔 클립의 명세만 다시 쓴다 — 내려받지 않는다")
    a = ap.parse_args()

    out = Path(a.out)

    # 이전 명세의 원본 길이 — 재현 도구로 쓸 때 「그때와 같은 원본인가」를 대조한다(#24 리뷰 ②)
    old_dur: dict[str, float] = {}
    if (out / MANIFEST).exists():
        with open(out / MANIFEST, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    # 키는 영상이다 — clip_id는 원본이 바뀌면 밀릴 수 있는 바로 그 대상이라,
                    # clip_id로 대조하면 밀린 뒤 엉뚱한 쌍을 비교한다(#24 재리뷰 🟡3)
                    old_dur[r["video_id"]] = float(r.get("video_duration_s") or 0)
                except (KeyError, ValueError):
                    pass
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
            # 길이를 못 읽어도 **요청한 개수만큼 번호를 비워둔다.** 안 비우면 뒤 후보의 clip_id가
            # 전부 당겨지고, clip_id는 라벨·ELAN 파일과의 유일한 조인 키다(#24 리뷰 🔴②)
            print(f"건너뜀 — {e} · 번호 {c.n_clips}개를 비워둔다", file=sys.stderr)
            failed.append(c.url)
            idx += c.n_clips
            continue

        # 씨앗을 영상 ID에 묶는다. 후보 목록의 순서가 바뀌어도 같은 오프셋이 나온다
        rng = random.Random(f"{a.seed}:{vid}")
        offsets = draw_offsets(dur, c.n_clips, rng, a.clip_s, a.margin_s)
        if not offsets:
            print(f"건너뜀 — 길이 {dur:.0f}초로는 {a.clip_s:.0f}초 클립을 못 뽑는다 ({vid}) · "
                  f"번호 {c.n_clips}개를 비워둔다", file=sys.stderr)
            failed.append(c.url)
            # 지금은 도달 불가지만, 번호가 런타임 결과에 안 묶인다는 불변식을 여기서도 지킨다(#24 재리뷰 🟢5)
            idx += c.n_clips
            continue
        if len(offsets) < c.n_clips:
            print(f"알림 — {vid}에서 {c.n_clips}개 요청, {len(offsets)}개만 가능", file=sys.stderr)

        for off in offsets:
            # 번호는 받기 **전에** 소비한다 — 내려받기가 실패해도 뒤 클립 번호가 밀리지 않는다.
            # 결과적으로 clip_id는 candidates.txt(순서·개수) + 씨앗 + 원본 길이의 함수다
            clip_id = f"c{idx:03d}"
            idx += 1
            stem = out / "video" / clip_id
            wav = out / "wav" / f"{clip_id}.wav"
            print(f"{clip_id}  {vid} @{off:.2f}s  ({c.source_class})")
            got = a.clip_s
            if a.manifest_only:
                # 이미 받아둔 파일에서 실제 길이를 읽는다. 씨앗이 같으면 오프셋도
                # 같으므로, 명세만 다시 써도 같은 클립을 가리킨다.
                media = _pick_media(stem)
                if media is None:
                    print(f"  건너뜀 — 받아둔 파일이 없다 ({stem.name})", file=sys.stderr)
                    failed.append(f"{vid}@{off}")
                    continue
                # wav이 없으면 영상에서 **그 자리에서 뽑는다.** 명세에서 빼면 이미 라벨이 달린 클립이
                # labels.csv에는 있는데 clips.csv에는 없어 조인에서 미아가 된다(#24 재리뷰 🟡4).
                # 뽑기까지 실패할 때만 뺀다 — 넣으면 라벨 0건이라 대조 클립으로 잘못 세진다
                if not wav.exists():
                    try:
                        to_wav(media, wav)
                        print(f"  알림 — {wav.name}이 없어 {media.name}에서 뽑았다", file=sys.stderr)
                    except RuntimeError as e:
                        print(f"  건너뜀 — {wav.name}을 뽑지 못했다. 명세에 넣으면 대조 클립으로 "
                              f"잘못 세진다 — {e}", file=sys.stderr)
                        failed.append(f"{vid}@{off}")
                        continue
                got = media_duration(media)
                # 디스크의 파일이 이 오프셋·길이로 받은 것인지 — 길이로라도 대조한다
                expect = min(a.clip_s, dur)
                if abs(got - expect) > 20:
                    print(f"  경고 — {media.name} 길이 {got:.0f}초가 기대 {expect:.0f}초와 다르다. "
                          f"다른 --clip-s나 씨앗으로 받은 파일일 수 있다", file=sys.stderr)
                prev = old_dur.get(vid)
                if prev and abs(prev - dur) > 1.0:
                    print(f"  경고 — {vid} 원본 길이가 명세({prev:.0f}초)와 지금({dur:.0f}초) 다르다. "
                          f"재업로드·편집이면 같은 씨앗이라도 오프셋이 달라진다", file=sys.stderr)
            elif not a.dry_run:
                try:
                    # 통째로 쓰는 클립(오프셋 0 · 원본이 더 짧음)은 구간 지정 없이 받는다
                    span = 0.0 if is_whole(dur, a.clip_s) else a.clip_s
                    media = cut(c.url, off, span, stem, a.exact_cuts)
                    to_wav(media, wav)
                    got = media_duration(media)
                except RuntimeError as e:
                    print(f"  실패 — {e}", file=sys.stderr)
                    failed.append(f"{vid}@{off}")
                    continue
            rows.append({"clip_id": clip_id, "video_id": vid, "clip_offset": off,
                         "duration_s": got, "source_class": c.source_class, "url": canonical_url(vid),
                         "seed": a.seed, "clip_s": a.clip_s, "margin_s": a.margin_s,
                         "video_duration_s": round(dur, 2)})

    if not rows:
        print("만들어진 클립이 없다", file=sys.stderr)
        return 1

    manifest = out / MANIFEST
    if a.dry_run:
        print(f"\n[dry-run] 클립 {len(rows)}개 계획 — 명세를 쓰지 않았다")
    else:
        # 이번 실행의 행만 쓰면 기존 명세의 나머지가 조용히 사라진다(#24 재리뷰 🟡1·🟡)
        rows = merge_manifest(manifest, rows, a.start_index)
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
