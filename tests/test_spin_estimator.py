"""
Validate the spin estimator against the mock camera.

The mock rotates the ball's satellite dots at exactly 7 degrees per frame.
At 120 fps that is 7 * 120 * 60 / 360 = 140 RPM. If the estimator recovers
that within a reasonable tolerance, the log-polar + phase correlation
pipeline is wired correctly and we can move on to real footage.
"""
from __future__ import annotations

import math
import sys

import numpy as np

# Allow running from repo root without install
sys.path.insert(0, ".")

from floatpro.cameras import make_camera, CameraConfig
from floatpro.spin_estimator import (
    detect_ball_simple,
    estimate_rotation,
    crop_ball_patch,
    estimate_session_spin,
)


def grab_frames(n: int, fps: int = 120, width: int = 640, height: int = 480):
    cam = make_camera("mock", CameraConfig(width=width, height=height, fps=fps))
    cam.start()
    frames = []
    try:
        for _ in range(n):
            ok, frame, ts = cam.read()
            if ok:
                frames.append(frame)
    finally:
        cam.stop()
    return frames


def test_ball_detection_finds_ball_in_most_frames():
    frames = grab_frames(30)
    detections = [detect_ball_simple(f) for f in frames]
    hit_rate = sum(d is not None for d in detections) / len(detections)
    print(f"  detection hit rate: {hit_rate:.2f}")
    assert hit_rate > 0.9, f"detection too unreliable: {hit_rate}"


def test_pairwise_rotation_matches_ground_truth():
    # Mock rotates 7 deg/frame
    expected_rad_per_frame = math.radians(7)
    frames = grab_frames(10)
    dets = [detect_ball_simple(f) for f in frames]
    assert all(d is not None for d in dets), "some detections failed"

    pairs = []
    for i in range(len(frames) - 1):
        pa = crop_ball_patch(frames[i], dets[i])
        pb = crop_ball_patch(frames[i + 1], dets[i + 1])
        angle, conf = estimate_rotation(pa, pb)
        pairs.append((angle, conf))

    angles = [abs(a) for a, c in pairs if c > 0.05]
    median_angle = float(np.median(angles))
    err_deg = abs(math.degrees(median_angle) - 7.0)
    print(f"  median angle/frame: {math.degrees(median_angle):.2f} deg "
          f"(expected 7.00, err {err_deg:.2f} deg)")
    print(f"  confidence range: {min(c for _,c in pairs):.3f} "
          f"..{max(c for _,c in pairs):.3f}")
    # Tight tolerance — this is a clean synthetic test
    assert err_deg < 1.0, f"rotation estimate off by {err_deg:.2f} deg"


def test_session_rpm_matches_ground_truth():
    frames = grab_frames(60, fps=120)
    result = estimate_session_spin(frames, fps=120)
    print(f"  estimated RPM: {result.rpm:.1f} (expected 140.0)")
    print(f"  std: {result.rpm_std:.1f}   valid pairs: {result.n_valid_pairs}")
    print(f"  notes: {result.notes}")

    assert result.rpm is not None
    err_rpm = abs(result.rpm - 140.0)
    assert err_rpm < 10.0, f"RPM estimate off by {err_rpm:.1f}"


def test_session_handles_lower_fps():
    # At 60 fps with the same 7 deg/frame (mock uses frame index, not time)
    # we'd still see the same per-frame rotation — but the mock uses
    # t = frame_i/fps for position. Rotation is frame_i * 7 regardless of fps.
    # So at 60 fps, 7 deg/frame * 60 = 420 deg/sec = 70 RPM.
    frames = grab_frames(40, fps=60)
    result = estimate_session_spin(frames, fps=60)
    print(f"  @60fps: estimated RPM: {result.rpm:.1f} (expected 70.0)")
    assert result.rpm is not None
    assert abs(result.rpm - 70.0) < 8.0


def run_all():
    tests = [
        test_ball_detection_finds_ball_in_most_frames,
        test_pairwise_rotation_matches_ground_truth,
        test_session_rpm_matches_ground_truth,
        test_session_handles_lower_fps,
    ]
    failures = 0
    for t in tests:
        name = t.__name__
        print(f"\n[RUN] {name}")
        try:
            t()
            print(f"[PASS] {name}")
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            failures += 1
        except Exception as e:
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
            failures += 1
    print(f"\n{'='*50}")
    print(f"  {len(tests) - failures}/{len(tests)} passed")
    print(f"{'='*50}")
    return failures


if __name__ == "__main__":
    sys.exit(run_all())
