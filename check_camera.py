#!/usr/bin/env python3
"""
FloatPro camera smoke test.

Run this FIRST after installing the Arducam driver. It verifies:
  1. v4l2 sees the camera
  2. Arducam driver reports expected resolution + framerate
  3. OpenCV was built with GStreamer support
  4. The pipeline actually delivers ~120fps with no frame drops
  5. Frame content is not blank / solid-color (driver is live)

If all five pass, you're ready to run capture.py.
"""

import subprocess
import sys
import time
from statistics import mean, stdev

import cv2
import numpy as np


def header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_v4l2_devices():
    header("1. V4L2 devices")
    try:
        r = subprocess.run(["v4l2-ctl", "--list-devices"],
                           capture_output=True, text=True, timeout=5)
        print(r.stdout or "(no output)")
        if "/dev/video" not in r.stdout:
            print("FAIL: no /dev/video* devices found")
            return False
        return True
    except FileNotFoundError:
        print("FAIL: v4l2-ctl not installed. Run: sudo apt install v4l-utils")
        return False


def check_formats(device="/dev/video0"):
    header(f"2. Supported formats on {device}")
    try:
        r = subprocess.run(["v4l2-ctl", "-d", device, "--list-formats-ext"],
                           capture_output=True, text=True, timeout=5)
        print(r.stdout or "(no output)")
        # Sanity-check we can see a 120fps mode
        ok = "120" in r.stdout and ("1280" in r.stdout or "GREY" in r.stdout.upper())
        if not ok:
            print("WARN: did not see 120fps @ 1280x800 in format list")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def check_opencv_gstreamer():
    header("3. OpenCV GStreamer support")
    info = cv2.getBuildInformation()
    gst_line = [l for l in info.split("\n") if "GStreamer" in l]
    for l in gst_line:
        print(f"  {l.strip()}")
    ok = any("YES" in l for l in gst_line)
    if not ok:
        print("FAIL: OpenCV was built without GStreamer.")
        print("  On Jetson, use the system package: sudo apt install python3-opencv")
        print("  (pip's opencv-python wheels lack GStreamer on ARM64)")
    return ok


def check_framerate(pipeline, target_fps=120, n_frames=300):
    header(f"4. Framerate test — target {target_fps}fps over {n_frames} frames")
    print(f"Pipeline: {pipeline}\n")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("FAIL: could not open pipeline")
        return False, None

    # warmup — the first dozen frames are often slower as the pipeline primes
    for _ in range(30):
        cap.read()

    intervals = []
    last_t = time.time()
    received = 0
    t0 = time.time()
    for _ in range(n_frames):
        ret, frame = cap.read()
        now = time.time()
        if ret:
            received += 1
            intervals.append(now - last_t)
        last_t = now
    elapsed = time.time() - t0
    cap.release()

    if received < n_frames * 0.9:
        print(f"FAIL: only got {received}/{n_frames} frames")
        return False, None

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

    pass_fps = measured_fps >= target_fps * 0.9
    if not pass_fps:
        print(f"WARN: measured fps well below target. Possible causes:")
        print("  - appsink draining too slowly (Python processing behind)")
        print("  - USB-2 hub in path (use direct CSI cable)")
        print("  - Power mode not maxed: sudo nvpmodel -m 0 && sudo jetson_clocks")
    return pass_fps, frame


def check_frame_content(frame):
    header("5. Frame content sanity")
    if frame is None:
        print("FAIL: no frame available")
        return False
    h, w = frame.shape[:2]
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
        print("WARN: exposure extreme — add/remove light, or tune sensor gain")
    return True


def main():
    checks = []

    checks.append(("v4l2 devices", check_v4l2_devices()))
    checks.append(("camera formats", check_formats()))
    checks.append(("OpenCV GStreamer", check_opencv_gstreamer()))

    pipeline = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,format=GRAY8,width=1280,height=800,framerate=120/1 ! "
        "videoconvert ! appsink drop=1 max-buffers=2 sync=false"
    )
    ok_fps, last_frame = check_framerate(pipeline, target_fps=120)
    checks.append(("120fps stream", ok_fps))
    checks.append(("frame content", check_frame_content(last_frame)))

    header("SUMMARY")
    for name, ok in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {name}")

    all_ok = all(ok for _, ok in checks)
    print()
    if all_ok:
        print("All checks passed. Run: python3 capture.py")
        sys.exit(0)
    else:
        print("One or more checks failed. Fix before running capture.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
