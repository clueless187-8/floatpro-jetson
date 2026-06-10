#!/usr/bin/env python3
"""
Extract frames from capture sessions (or raw videos) for labeling.

Usage:
    python ml/extract_frames.py captures/ --every 5 --out ml/dataset/images/raw
    python ml/extract_frames.py clips/serve1.mp4 --every 5 --out ml/dataset/images/raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def extract_from_video(video: Path, every: int, out: Path, prefix: str) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  SKIP (can't open): {video}")
        return 0
    n = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % every == 0:
            fname = out / f"{prefix}_{n:05d}.png"
            cv2.imwrite(str(fname), frame)
            saved += 1
        n += 1
    cap.release()
    return saved


def extract_from_session(session: Path, every: int, out: Path) -> int:
    pngs = sorted(session.glob("frame_*.png"))
    saved = 0
    for i, p in enumerate(pngs):
        if i % every == 0:
            (out / f"{session.name}_{p.name}").write_bytes(p.read_bytes())
            saved += 1
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="captures/ dir, session dir, or video file")
    ap.add_argument("--every", type=int, default=5,
                    help="keep every Nth frame (default 5)")
    ap.add_argument("--out", default="ml/dataset/images/raw")
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    if src.is_file():
        total = extract_from_video(src, args.every, out, src.stem)
    elif (src / "metadata.json").exists():            # single session
        total = extract_from_session(src, args.every, out)
    elif src.is_dir():                                 # captures/ root
        for session in sorted(src.iterdir()):
            if (session / "metadata.json").exists():
                n = extract_from_session(session, args.every, out)
                print(f"  {session.name}: {n} frames")
                total += n
        for vid in sorted(src.glob("*.mp4")) + sorted(src.glob("*.mov")):
            n = extract_from_video(vid, args.every, out, vid.stem)
            print(f"  {vid.name}: {n} frames")
            total += n
    else:
        print(f"ERROR: {src} not found")
        return 1

    print(f"\n{total} frames → {out}")
    print("Next: label in Roboflow/Label Studio (class: ball), "
          "export YOLOv8 format to ml/dataset/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
