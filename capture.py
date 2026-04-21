#!/usr/bin/env python3
"""
FloatPro capture — backend-agnostic ring buffer + trigger save.

Pick your camera at the command line:

    python3 capture.py --backend ov9281
    python3 capture.py --backend ar0234
    python3 capture.py --backend flir   --exposure 500 --gain 6
    python3 capture.py --backend basler --fps 240 --width 720 --height 540
    python3 capture.py --backend mock   # no hardware needed

Controls in the preview window:
    SPACE   save the last --buffer seconds of frames
    q       quit
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2

from floatpro.cameras import (
    CameraConfig,
    PRESETS,
    available_backends,
    make_camera,
)
from floatpro.ring_buffer import RingBuffer


def save_session(frames, info, output_root: Path):
    """Dump frames + metadata to disk. Runs in a worker thread so the
    capture loop never blocks on I/O."""
    if not frames:
        print("[save] empty buffer, nothing to write")
        return

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    session_dir = output_root / ts_str
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"[save] {len(frames)} frames -> {session_dir}")
    t0 = time.time()

    meta = {
        "session_id": ts_str,
        "camera": asdict(info),
        "frame_count": len(frames),
        "frames": [],
    }

    for i, (frame_ts, frame) in enumerate(frames):
        fname = f"frame_{i:05d}.png"
        cv2.imwrite(str(session_dir / fname), frame)
        meta["frames"].append({
            "i": i,
            "t": round(frame_ts, 6),
            "file": fname,
        })

    if len(frames) > 1:
        intervals = [frames[i + 1][0] - frames[i][0] for i in range(len(frames) - 1)]
        avg_ms = 1000 * sum(intervals) / len(intervals)
        max_ms = 1000 * max(intervals)
        min_ms = 1000 * min(intervals)
        meta["timing"] = {
            "avg_interval_ms": round(avg_ms, 3),
            "min_interval_ms": round(min_ms, 3),
            "max_interval_ms": round(max_ms, 3),
            "effective_fps": round(1000 / avg_ms, 2),
        }
        print(f"[save] effective fps={meta['timing']['effective_fps']}  "
              f"avg={avg_ms:.2f}ms min={min_ms:.2f}ms max={max_ms:.2f}ms")

    with open(session_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[save] done in {time.time() - t0:.1f}s")


def draw_hud(frame, fps_actual, buf_len, buf_cap, drops, backend):
    """Live on-screen stats."""
    if len(frame.shape) == 2:
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        display = frame.copy()

    h, w = display.shape[:2]
    scale = min(1.0, 960 / max(w, h))
    if scale < 1.0:
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

    hud1 = f"[{backend}]  FPS: {fps_actual:5.1f}  |  Buffer: {buf_len}/{buf_cap}  |  Drops: {drops}"
    hud2 = "SPACE = save   Q = quit"
    cv2.putText(display, hud1, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(display, hud2, (10, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return display


def parse_args():
    p = argparse.ArgumentParser(description="FloatPro camera capture")
    p.add_argument("--backend", required=False,
                   choices=["ov9281", "ar0234", "flir", "basler", "mock"],
                   help="Camera backend")
    p.add_argument("--device", default="",
                   help="Device path (/dev/video0) or serial number")
    p.add_argument("--width", type=int, help="Frame width (default: preset)")
    p.add_argument("--height", type=int, help="Frame height (default: preset)")
    p.add_argument("--fps", type=int, help="Target framerate (default: preset)")
    p.add_argument("--pixel-format", default=None,
                   choices=["mono8", "bgr8", "auto"],
                   help="Pixel format (default: backend choice)")
    p.add_argument("--exposure", type=int, default=None,
                   help="Exposure in microseconds (FLIR/Basler only)")
    p.add_argument("--gain", type=float, default=None,
                   help="Analog gain in dB (FLIR/Basler only)")
    p.add_argument("--buffer", type=float, default=5.0,
                   help="Ring buffer duration in seconds")
    p.add_argument("--output", default="captures",
                   help="Output directory")
    p.add_argument("--no-preview", action="store_true",
                   help="Run headless (useful over SSH without X forwarding)")
    p.add_argument("--list-backends", action="store_true",
                   help="Show which backends are installable and exit")
    return p.parse_args()


def build_config(args) -> CameraConfig:
    preset = PRESETS[args.backend]
    return CameraConfig(
        width=args.width or preset.width,
        height=args.height or preset.height,
        fps=args.fps or preset.fps,
        pixel_format=args.pixel_format or preset.pixel_format,
        exposure_us=args.exposure,
        gain_db=args.gain,
        device=args.device,
    )


def main():
    args = parse_args()

    if args.list_backends:
        avail = available_backends()
        print("Backend availability:")
        for name, ok in avail.items():
            mark = "OK " if ok else "NO "
            print(f"  [{mark}] {name}")
        sys.exit(0)

    if not args.backend:
        print("ERROR: --backend required. Use --list-backends to see options.",
              file=sys.stderr)
        sys.exit(2)

    config = build_config(args)
    capacity = int(args.buffer * config.fps)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    bytes_per_frame = config.width * config.height * (1 if config.pixel_format == "mono8" else 3)
    ram_mb = capacity * bytes_per_frame / 1024 / 1024

    print("=" * 60)
    print("FloatPro Capture")
    print("=" * 60)
    print(f"Backend     : {args.backend}")
    print(f"Device      : {config.device or '(default)'}")
    print(f"Resolution  : {config.width}x{config.height} @ {config.fps}fps")
    print(f"Pixel fmt   : {config.pixel_format}")
    print(f"Buffer      : {args.buffer}s = {capacity} frames  (~{ram_mb:.0f} MB)")
    print(f"Output      : {output_dir.absolute()}")
    print("=" * 60)

    cam = make_camera(args.backend, config)
    rb = RingBuffer(cam, capacity)

    try:
        rb.start()
    except Exception as e:
        print(f"\nERROR: camera failed to start: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Streaming started: {rb.info.model} ({rb.info.serial})")

    save_threads = []
    try:
        if args.no_preview:
            print("Headless mode. Send SIGINT (Ctrl-C) to save and exit.")
            try:
                while True:
                    time.sleep(1.0)
                    print(f"  fps={rb.fps_actual:5.1f} "
                          f"buf={rb.buffer_len}/{rb.capacity} "
                          f"drops={rb.drop_count}")
            except KeyboardInterrupt:
                print("[main] saving on exit...")
                save_session(rb.snapshot(), rb.info, output_dir)
        else:
            while True:
                frame = rb.latest()
                if frame is not None:
                    hud = draw_hud(frame, rb.fps_actual, rb.buffer_len,
                                   rb.capacity, rb.drop_count, args.backend)
                    cv2.imshow("FloatPro", hud)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    frames = rb.snapshot()
                    t = threading.Thread(
                        target=save_session,
                        args=(frames, rb.info, output_dir),
                        daemon=False,
                    )
                    t.start()
                    save_threads.append(t)
    except KeyboardInterrupt:
        print("\n[main] interrupt")
    finally:
        print("[main] stopping capture...")
        rb.stop()
        cv2.destroyAllWindows()
        for t in save_threads:
            t.join()
        print(f"[main] total frames: {rb.frame_count}  drops: {rb.drop_count}")


if __name__ == "__main__":
    main()
