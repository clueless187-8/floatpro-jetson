#!/usr/bin/env python3
"""
FloatPro camera smoke test.

Usage:
    python3 check_camera.py --backend ov9281
    python3 check_camera.py --backend ar0234
    python3 check_camera.py --backend flir
    python3 check_camera.py --backend basler
    python3 check_camera.py --backend mock

Verifies the selected backend can stream at its preset framerate with
no drops and produces non-blank frames. Run this first after installing
a driver, before running capture.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from statistics import mean, stdev

import cv2
import numpy as np

from floatpro.cameras import PRESETS, available_backends, make_camera


def header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_v4l2_devices(required):
    if not required:
        print("  (skipped — backend doesn't use v4l2)")
        return True
    try:
        r = subprocess.run(["v4l2-ctl", "--list-devices"],
                           capture_output=True, text=True, timeout=5)
        print(r.stdout or "(no output)")
        if "/dev/video" not in r.stdout:
            print("FAIL: no /dev/video* devices found")
            return False
        return True
    except FileNotFoundError:
        print("FAIL: v4l2-ctl not installed. sudo apt install v4l-utils")
        return False


def check_opencv_gstreamer(required):
    if not required:
        print("  (skipped — backend doesn't use GStreamer)")
        return True
    info = cv2.getBuildInformation()
    gst_line = [l for l in info.split("\n") if "GStreamer" in l]
    for l in gst_line:
        print(f"  {l.strip()}")
    ok = any("YES" in l for l in gst_line)
    if not ok:
        print("FAIL: OpenCV was built without GStreamer.")
        print("  On Jetson: sudo apt install python3-opencv  "
              "(don't use pip opencv-python on ARM64)")
    return ok


def check_sdk_available(backend):
    avail = available_backends()
    ok = avail.get(backend, False)
    if ok:
        print(f"  [{backend}] backend constructor available")
    else:
        print(f"FAIL: [{backend}] backend not importable")
        if backend == "flir":
            print("  Install Spinnaker SDK + PySpin from flir.com")
        elif backend == "basler":
            print("  Install Pylon SDK + pypylon: pip install pypylon")
    return ok


def check_framerate(backend, n_frames=300):
    preset = PRESETS[backend]
    target_fps = preset.fps
    print(f"Target {target_fps}fps over {n_frames} frames")
    print(f"Preset: {preset.width}x{preset.height} pix={preset.pixel_format}\n")

    cam = make_camera(backend, preset)
    try:
        info = cam.start()
    except Exception as e:
        print(f"FAIL: could not start camera: {e}")
        return False, None

    print(f"Started: {info.model} sn={info.serial}")

    # Warmup — first frames are often slower as pipeline primes
    for _ in range(30):
        cam.read()

    intervals = []
    last_t = time.monotonic()
    received = 0
    t0 = time.monotonic()
    for _ in range(n_frames):
        ok, frame, ts = cam.read()
        now = time.monotonic()
        if ok:
            received += 1
            intervals.append(now - last_t)
        last_t = now
    elapsed = time.monotonic() - t0

    last_frame = frame
    cam.stop()

    if received < n_frames * 0.9:
        print(f"FAIL: only got {received}/{n_frames} frames")
        return False, last_frame

    measured_fps = received / elapsed
    avg_ms = 1000 * mean(intervals)
    sd_ms = 1000 * stdev(intervals) if len(intervals) > 1 else 0
    max_ms = 1000 * max(intervals)

    print(f"  Received       : {received}/{n_frames}")
    print(f"  Elapsed        : {elapsed:.2f}s")
    print(f"  Measured FPS   : {measured_fps:.1f}")
    print(f"  Interval avg   : {avg_ms:.2f}ms  (target {1000/target_fps:.2f}ms)")
    print(f"  Interval stdev : {sd_ms:.2f}ms")
    print(f"  Interval max   : {max_ms:.2f}ms")

    ok = measured_fps >= target_fps * 0.9
    if not ok:
        print("WARN: measured fps well below target.")
        print("  - Orin Nano not in max perf mode?  sudo nvpmodel -m 0 && sudo jetson_clocks")
        print("  - USB-2 hub in path for industrial cam?  use direct USB3")
        print("  - Python falling behind?  try --no-preview")
    return ok, last_frame


def check_frame_content(frame):
    if frame is None:
        print("FAIL: no frame available")
        return False
    mean_v = float(np.mean(frame))
    std_v = float(np.std(frame))
    print(f"  Shape        : {frame.shape}")
    print(f"  dtype        : {frame.dtype}")
    print(f"  Mean pixel   : {mean_v:.1f}")
    print(f"  Stdev pixel  : {std_v:.1f}")

    if std_v < 2.0:
        print("FAIL: frame looks blank / solid color. Check lens cap & lighting.")
        return False
    if mean_v < 5 or mean_v > 250:
        print("WARN: exposure extreme — tune --exposure / --gain")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True,
                   choices=["ov9281", "ar0234", "flir", "basler", "mock"])
    args = p.parse_args()

    uses_v4l2 = args.backend in ("ov9281", "ar0234")
    uses_gstreamer = args.backend in ("ov9281", "ar0234")

    checks = []

    header("1. SDK / backend importable")
    checks.append(("backend available", check_sdk_available(args.backend)))

    header("2. v4l2 devices")
    checks.append(("v4l2 devices", check_v4l2_devices(uses_v4l2)))

    header("3. OpenCV GStreamer support")
    checks.append(("OpenCV GStreamer", check_opencv_gstreamer(uses_gstreamer)))

    header(f"4. Framerate test — {args.backend}")
    ok_fps, last_frame = check_framerate(args.backend)
    checks.append(("framerate", ok_fps))

    header("5. Frame content sanity")
    checks.append(("frame content", check_frame_content(last_frame)))

    header("SUMMARY")
    for name, ok in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {name}")

    all_ok = all(ok for _, ok in checks)
    print()
    if all_ok:
        print(f"All checks passed for [{args.backend}]. "
              f"Run: python3 capture.py --backend {args.backend}")
        sys.exit(0)
    else:
        print(f"One or more checks failed for [{args.backend}].")
        sys.exit(1)


if __name__ == "__main__":
    main()
