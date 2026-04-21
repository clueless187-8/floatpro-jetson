#!/usr/bin/env python3
"""
Ingest a video file into a FloatPro session directory.

Converts any phone/camera video (MP4, MOV, AVI, ...) into the same
`captures/<session_id>/` layout that `capture.py` produces, so the spin
estimator and FastAPI server can analyze it identically to a live capture.

Common flows:

    # Ingest the whole clip
    python3 ingest_video.py serve_001.mp4

    # Trim to the 2.5s–5.0s window (drop toss setup and post-serve)
    python3 ingest_video.py serve_001.mp4 --start 2.5 --end 5.0

    # Preview detection before extracting (fast — samples 5 frames)
    python3 ingest_video.py serve_001.mp4 --preview

    # Ingest and immediately run spin analysis
    python3 ingest_video.py serve_001.mp4 --analyze

    # Force 120fps interpretation (useful when phone metadata lies)
    python3 ingest_video.py serve_001.mp4 --fps 240

    # Downscale 4K → 1080p for faster processing
    python3 ingest_video.py serve_001.mp4 --scale 0.5

The ingested session is indistinguishable from a hardware capture as far
as the rest of the pipeline is concerned. You can open it in the dashboard
at http://localhost:8080, scrub through frames, and hit "Analyze spin"
like any other session.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from floatpro.cameras.base import CameraInfo
from floatpro.spin_estimator import detect_ball_simple, estimate_session_spin


def probe_video(path: Path) -> dict:
    """Read metadata without extracting frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    cap.release()
    info["duration_s"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    info["fourcc_str"] = "".join(
        [chr((info["fourcc"] >> (8 * k)) & 0xFF) for k in range(4)]
    )
    return info


def preview_detection(path: Path, n_samples: int = 5) -> None:
    """Sample a few frames and run ball detection — cheap sanity check
    before committing to extract 500+ frames."""
    info = probe_video(path)
    print(f"\nVideo: {path.name}")
    print(f"  {info['width']}x{info['height']} @ {info['fps']:.2f} fps")
    print(f"  {info['frame_count']} frames, {info['duration_s']:.2f}s, "
          f"codec {info['fourcc_str']}")

    if info["frame_count"] < n_samples:
        n_samples = info["frame_count"]

    cap = cv2.VideoCapture(str(path))
    sample_indices = np.linspace(0, info["frame_count"] - 1,
                                 n_samples, dtype=int)
    print(f"\nSampling {n_samples} frames for detection preview:")
    hits = 0
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {idx:>6}: read failed")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = detect_ball_simple(gray)
        if det is None:
            print(f"  frame {idx:>6}: no ball detected")
        else:
            hits += 1
            print(f"  frame {idx:>6}: ball at ({det.cx:.0f}, {det.cy:.0f}) "
                  f"r={det.r:.1f} conf={det.confidence:.2f}")
    cap.release()

    rate = hits / n_samples
    print(f"\nDetection hit rate: {rate:.0%} ({hits}/{n_samples})")
    if rate < 0.5:
        print("WARNING: low detection rate. Real footage may need the YOLOv8")
        print("  detector (not yet shipped). The simple threshold detector")
        print("  works well on clean mock frames but struggles with:")
        print("   - dark gym floors with bright highlights competing for attention")
        print("   - balls that aren't the brightest object in frame")
        print("   - motion-blurred balls at max stretch")
        print("  Try --scale 0.5 or crop to just the ball's flight path.")


def ingest(path: Path,
           captures_dir: Path,
           session_id: str | None = None,
           start_s: float = 0.0,
           end_s: float | None = None,
           max_frames: int | None = None,
           scale: float = 1.0,
           fps_override: float | None = None,
           grayscale: bool = True) -> Path:
    """
    Extract frames from a video into a FloatPro session directory.

    Returns the path to the created session directory.
    """
    vinfo = probe_video(path)
    src_fps = fps_override if fps_override else vinfo["fps"]
    if src_fps <= 0:
        raise RuntimeError(
            f"Could not determine FPS from {path}. Pass --fps explicitly."
        )

    # Frame window
    start_frame = int(start_s * src_fps)
    end_frame = vinfo["frame_count"] if end_s is None else int(end_s * src_fps)
    end_frame = min(end_frame, vinfo["frame_count"])
    if max_frames:
        end_frame = min(end_frame, start_frame + max_frames)

    n_to_extract = end_frame - start_frame
    if n_to_extract <= 0:
        raise RuntimeError(
            f"No frames in requested window: start={start_frame} end={end_frame}"
        )

    # Target resolution
    out_w = int(vinfo["width"] * scale)
    out_h = int(vinfo["height"] * scale)
    if scale != 1.0:
        print(f"Scaling {vinfo['width']}x{vinfo['height']} → {out_w}x{out_h}")

    # Create session directory
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + "_video"
    session_dir = captures_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session directory: {session_dir}")
    print(f"Extracting frames {start_frame}..{end_frame} "
          f"({n_to_extract} frames @ {src_fps:.1f} fps)")

    # Fake camera info consistent with the CameraInfo dataclass so the
    # server + estimator don't need special cases for "this came from
    # a video file."
    cam_info = CameraInfo(
        backend="video_file",
        model=f"{path.name} ({vinfo['fourcc_str']})",
        serial=str(path.resolve()),
        width=out_w,
        height=out_h,
        fps=float(src_fps),
        pixel_format="mono8" if grayscale else "bgr8",
        is_color=not grayscale,
        is_global_shutter=False,   # almost always rolling shutter on phones
    )

    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_meta = []
    t0 = time.time()
    i = 0
    while i < n_to_extract:
        ok, frame = cap.read()
        if not ok:
            print(f"  read failed at i={i} (wanted {n_to_extract})")
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h),
                               interpolation=cv2.INTER_AREA)
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fname = f"frame_{i:05d}.png"
        cv2.imwrite(str(session_dir / fname), frame)

        frames_meta.append({
            "i": i,
            # Synthesized monotonic timestamp — real wall-clock doesn't
            # matter for spin estimation; only intervals do.
            "t": round(i / src_fps, 6),
            "file": fname,
        })
        i += 1
        if i % 60 == 0 or i == n_to_extract:
            pct = 100 * i / n_to_extract
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{n_to_extract} ({pct:.0f}%)  {rate:.0f} fps extract")
    cap.release()

    # Write metadata.json matching capture.py's output exactly
    meta = {
        "session_id": session_id,
        "camera": asdict(cam_info),
        "frame_count": len(frames_meta),
        "frames": frames_meta,
        "timing": {
            # For video ingestion, interval is deterministic — 1/fps.
            "avg_interval_ms": round(1000.0 / src_fps, 3),
            "min_interval_ms": round(1000.0 / src_fps, 3),
            "max_interval_ms": round(1000.0 / src_fps, 3),
            "effective_fps": round(src_fps, 3),
        },
        "source": {
            "type": "video_file",
            "path": str(path.resolve()),
            "original_size": [vinfo["width"], vinfo["height"]],
            "original_fps": vinfo["fps"],
            "scale": scale,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "codec": vinfo["fourcc_str"],
        },
    }
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nIngest complete: {len(frames_meta)} frames in {session_dir}")
    return session_dir


def run_analysis(session_dir: Path) -> None:
    """Run spin estimation right after ingest. Same code path as
    POST /api/sessions/{id}/analyze."""
    meta = json.loads((session_dir / "metadata.json").read_text())
    fps = meta["timing"]["effective_fps"]

    print(f"\nLoading {meta['frame_count']} frames...")
    frames = []
    for entry in meta["frames"]:
        img = cv2.imread(str(session_dir / entry["file"]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)

    print(f"Running spin estimation at {fps:.1f} fps...")
    t0 = time.time()
    result = estimate_session_spin(frames, fps=fps)
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"  SPIN ANALYSIS")
    print(f"{'='*50}")
    if result.rpm is None:
        print("  RPM: could not estimate")
    else:
        print(f"  RPM:           {result.rpm:.1f}")
        print(f"  Direction:     {result.direction or '—'}")
        print(f"  Variability:   ±{result.rpm_std:.1f} RPM (MAD)")
        print(f"  Method:        {result.method}")
    print(f"  Valid pairs:   {result.n_valid_pairs} / {result.n_frames - 1}")
    print(f"  Analysis time: {elapsed:.1f}s")
    print(f"  Notes:")
    for note in result.notes:
        print(f"    - {note}")

    # Cache the result on disk (same filename the server uses)
    cache_path = session_dir / "spin_result.json"
    cache_path.write_text(json.dumps(asdict(result), indent=2, default=str))
    print(f"\nResult cached to {cache_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Ingest video files into FloatPro sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("video", type=Path, help="Input video file (MP4/MOV/AVI/...)")
    p.add_argument("--captures", default="captures",
                   help="Output captures directory (default: ./captures)")
    p.add_argument("--session-id", default=None,
                   help="Custom session id (default: auto-timestamped)")
    p.add_argument("--start", type=float, default=0.0,
                   help="Start time in seconds (default: 0)")
    p.add_argument("--end", type=float, default=None,
                   help="End time in seconds (default: end of video)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap on total frames extracted")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Resolution scale factor (0.5 = half, default: 1.0)")
    p.add_argument("--fps", type=float, default=None,
                   help="Override FPS (useful if video metadata is wrong)")
    p.add_argument("--color", action="store_true",
                   help="Preserve color instead of converting to grayscale")
    p.add_argument("--preview", action="store_true",
                   help="Only probe + test detection, don't extract")
    p.add_argument("--analyze", action="store_true",
                   help="Run spin estimation after ingest")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.video.exists():
        print(f"ERROR: {args.video} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.preview:
        preview_detection(args.video)
        return

    captures_dir = Path(args.captures)
    captures_dir.mkdir(exist_ok=True)

    session_dir = ingest(
        args.video,
        captures_dir,
        session_id=args.session_id,
        start_s=args.start,
        end_s=args.end,
        max_frames=args.max_frames,
        scale=args.scale,
        fps_override=args.fps,
        grayscale=not args.color,
    )

    if args.analyze:
        run_analysis(session_dir)

    print(f"\nOpen in dashboard:")
    print(f"  python3 -m floatpro.server --port 8080")
    print(f"  → http://localhost:8080  (session: {session_dir.name})")


if __name__ == "__main__":
    main()
