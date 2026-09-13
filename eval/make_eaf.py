"""클립마다 ELAN 작업 파일(`.eaf`)을 미리 만든다.

`File > New`를 클립 수만큼 반복하지 않게 하는 도구다. 26개를 손으로 열면
같은 실수를 26번 할 기회가 생긴다 — 특히 **wav을 연결하지 않는 실수**가 그렇다.
mp4만 열면 파형이 빈 칸으로 남고, 파형 없이 찍으면 영상을 처음부터 끝까지
봐야 한다(`elan-setup.md` §3).

그래서 영상과 wav을 **미리 연결한** `.eaf`를 클립마다 하나씩 낸다.
`docs/scareblock.etf`와 같은 트랙 5개를 담으므로 템플릿을 고를 필요도 없다.
라벨 담당자는 파일을 열어 바로 찍기 시작한다.

사용:
    python3 -m eval.make_eaf --clips _local/dataset/clips.csv --out _local/dataset/elan
    python3 -m eval.make_eaf --clips ... --out ... --order    # 라벨링 순서만 본다
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

# `docs/scareblock.etf`와 같은 순서·같은 이름이어야 한다. 변환기가 트랙 이름으로 읽는다.
TIERS = ["jumpscare", "blood", "syringe", "spider", "siren"]
LTYPE = "event"
KST = timezone(timedelta(hours=9))


def _media(url: Path, mime: str, relative: str, extracted_from: str | None = None) -> ET.Element:
    e = ET.Element("MEDIA_DESCRIPTOR", {
        "MEDIA_URL": url.resolve().as_uri(),
        "MIME_TYPE": mime,
        "RELATIVE_MEDIA_URL": relative,
    })
    if extracted_from:
        # wav이 이 영상에서 뽑힌 것임을 알려주면 ELAN이 그 영상의 **파형 원본**으로 쓴다.
        # 실제로 clipcut.py가 같은 mp4에서 뽑았으므로 사실이기도 하다.
        e.set("EXTRACTED_FROM", extracted_from)
    return e


def build_eaf(video: Path, wav: Path | None, eaf_dir: Path) -> ET.ElementTree:
    """EAF 3.0 문서 하나. 구간은 비어 있고 트랙만 서 있다."""
    root = ET.Element("ANNOTATION_DOCUMENT", {
        "AUTHOR": "scareblock",
        "DATE": datetime.now(KST).replace(microsecond=0).isoformat(),
        "FORMAT": "3.0",
        "VERSION": "3.0",
    })
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://www.mpi.nl/tools/elan/EAFv3.0.xsd")

    # EAF 3.0의 요소 순서 — HEADER → TIME_ORDER → TIER* → LINGUISTIC_TYPE*
    header = ET.SubElement(root, "HEADER", {"MEDIA_FILE": "", "TIME_UNITS": "milliseconds"})

    def rel(p: Path) -> str:
        try:
            return "./" + str(p.resolve().relative_to(eaf_dir.resolve()))
        except ValueError:
            import os
            return "./" + os.path.relpath(p.resolve(), eaf_dir.resolve())

    header.append(_media(video, "video/mp4", rel(video)))
    if wav is not None:
        header.append(_media(wav, "audio/x-wav", rel(wav),
                             extracted_from=video.resolve().as_uri()))

    ET.SubElement(root, "TIME_ORDER")
    for name in TIERS:
        ET.SubElement(root, "TIER", {"LINGUISTIC_TYPE_REF": LTYPE, "TIER_ID": name})
    ET.SubElement(root, "LINGUISTIC_TYPE", {
        "GRAPHIC_REFERENCES": "false", "LINGUISTIC_TYPE_ID": LTYPE, "TIME_ALIGNABLE": "true"})
    return ET.ElementTree(root)


def write_eaf(tree: ET.ElementTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="    ")
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def read_clips(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        ids = [(r.get("clip_id") or "").strip() for r in csv.DictReader(f)]
    out = [c for c in ids if c]
    if not out:
        raise SystemExit(f"{path}: clip_id가 없다")
    return out


def label_order(clip_ids: list[str], seed: int) -> list[str]:
    """라벨링 순서를 씨앗으로 섞는다.

    전부 찍지 못하고 중간에 멈출 수 있으므로 **앞에서부터 찍은 것이 무작위
    부분집합**이 되어야 한다. `clip_id` 순서대로 찍으면 앞쪽 영상만 라벨링되고
    뒤쪽 영상은 하나도 안 들어가, 「무작위 오프셋」의 의미가 절반 사라진다.
    """
    order = list(clip_ids)
    random.Random(f"order:{seed}").shuffle(order)
    return order


def make_all(ids: list[str], vdir: Path, wdir: Path, out: Path,
             force: bool = False) -> tuple[int, list[str], list[str]]:
    """클립마다 .eaf를 쓴다. (만든 수, 이미 있어 건너뛴 것, 영상이 없는 것)."""
    made, skipped, missing = 0, [], []
    for cid in ids:
        video = vdir / f"{cid}.mp4"
        wav = wdir / f"{cid}.wav"
        if not video.exists():
            missing.append(cid)
            continue
        dest = out / f"{cid}.eaf"
        if dest.exists() and not force:
            skipped.append(cid)
            continue
        write_eaf(build_eaf(video, wav if wav.exists() else None, out), dest)
        if not wav.exists():
            print(f"경고: {cid}.wav이 없다 — 파형 없이 열린다", file=sys.stderr)
        made += 1
    return made, skipped, missing


def _selftest() -> int:
    """생성물을 XML로 읽어 구조를 확인한다. ELAN을 띄울 수는 없으므로, 실제로 열린
    `docs/scareblock.etf`(#22)와 같은 구조인지를 본다."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="make-eaf-selftest-"))
    (tmp / "video").mkdir()
    (tmp / "wav").mkdir()
    ids = ["c001", "c002"]
    for c in ids:
        (tmp / "video" / f"{c}.mp4").write_bytes(b"")
    (tmp / "wav" / "c001.wav").write_bytes(b"")        # c002는 wav 없음
    (tmp / "clips.csv").write_text("clip_id,video_id\nc001,v1\nc002,v2\n", encoding="utf-8")
    out = tmp / "elan"

    made, _, missing = make_all(read_clips(tmp / "clips.csv"), tmp / "video", tmp / "wav", out)
    again = make_all(ids, tmp / "video", tmp / "wav", out)
    r1 = ET.parse(out / "c001.eaf").getroot()
    md1 = list(r1.iter("MEDIA_DESCRIPTOR"))
    md2 = list(ET.parse(out / "c002.eaf").getroot().iter("MEDIA_DESCRIPTOR"))
    many = [f"c{i:03d}" for i in range(1, 45)]
    o1, o2, o3 = label_order(many, 20260912), label_order(many, 20260912), label_order(many, 7)
    checks = {
        "클립 2개 생성 · 영상 누락 없음": made == 2 and not missing,
        "이미 있으면 건너뛴다 (찍어둔 라벨 보호)": again[0] == 0 and again[1] == ids,
        "요소 순서 HEADER → TIME_ORDER → TIER×5 → LINGUISTIC_TYPE":
            [ch.tag for ch in r1] == ["HEADER", "TIME_ORDER"] + ["TIER"] * 5 + ["LINGUISTIC_TYPE"],
        "트랙 이름·순서가 scareblock.etf와 같다": [x.get("TIER_ID") for x in r1.iter("TIER")] == TIERS,
        "wav이 있으면 미디어 2개 + EXTRACTED_FROM":
            len(md1) == 2 and md1[1].get("MIME_TYPE") == "audio/x-wav"
            and md1[1].get("EXTRACTED_FROM", "").endswith("c001.mp4"),
        "wav이 없으면 영상 1개만": len(md2) == 1,
        "라벨링 순서는 씨앗으로 결정되는 순열": o1 == o2 and sorted(o1) == many and o1 != o3,
    }
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else '실패'}] {name}")
    allok = all(checks.values())
    print("\n자체 점검:", "통과" if allok else "실패")
    return 0 if allok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="클립마다 ELAN 작업 파일을 미리 만든다")
    ap.add_argument("--clips", help="클립 명세 CSV")
    ap.add_argument("--out", help=".eaf를 낼 디렉터리")
    ap.add_argument("--video-dir", help="기본: <clips.csv가 있는 곳>/video")
    ap.add_argument("--wav-dir", help="기본: <clips.csv가 있는 곳>/wav")
    ap.add_argument("--seed", type=int, default=20260912, help="라벨링 순서 씨앗")
    ap.add_argument("--order", action="store_true", help="순서만 내고 파일은 만들지 않는다")
    ap.add_argument("--force", action="store_true",
                    help="이미 있는 .eaf도 덮어쓴다 (찍어둔 라벨이 사라진다)")
    ap.add_argument("--selftest", action="store_true", help="임시 명세로 .eaf 생성과 순서를 점검")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not (a.clips and a.out):
        ap.error("--clips 와 --out 이 필요하다 (또는 --selftest)")

    clips = Path(a.clips)
    base = clips.parent
    vdir = Path(a.video_dir) if a.video_dir else base / "video"
    wdir = Path(a.wav_dir) if a.wav_dir else base / "wav"
    out = Path(a.out)

    ids = read_clips(clips)
    order = label_order(ids, a.seed)

    if a.order:
        print(f"라벨링 순서 (씨앗 order:{a.seed}) — 앞에서부터 찍으면 무작위 부분집합이 된다\n")
        for i, cid in enumerate(order, start=1):
            mark = "  ← 20개에서 멈출 지점" if i == 20 else ""
            print(f"{i:3d}. {cid}{mark}")
        return 0

    made, skipped, missing = make_all(ids, vdir, wdir, out, a.force)

    print(f".eaf {made}개 → {out}")
    if skipped:
        print(f"건너뜀 {len(skipped)}개 (이미 있음 — 덮어쓰려면 --force): "
              f"{', '.join(skipped[:6])}{'…' if len(skipped) > 6 else ''}")
    if missing:
        print(f"경고: 영상 없는 클립 {missing}", file=sys.stderr)
    print(f"\n라벨링 순서는 --order로 본다. 앞에서부터 찍으면 중간에 멈춰도 무작위 부분집합이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
