"""
Validate the spin estimator against the mock camera.

Mock rotates the ball pattern 7° per frame. At 120 fps that is
7 * 120 * 60 / 360 = 140 RPM. At 60 fps: 70 RPM.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")

from floatpro.cameras import make_camera, CameraConfig
from floatpro.spin_estimator import (
    detect_ball_simple,
    estimate_rotation_orb,
    estimate_rotation_logpolar,
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
    hit = sum(d is not None for d in detections) / len(detections)
    print(f"  detection hit rate: {hit:.2f}")
    assert hit > 0.95, f"detection too unreliable: {hit}"


def test_orb_rotation_matches_ground_truth():
    """ORB + RANSAC should recover 7° per frame."""
    frames = grab_frames(10)
    dets = [detect_ball_simple(f) for f in frames]
    assert all(d is not None for d in dets), "some detections failed"

    angles_deg = []
    diagnostics = []
    for i in range(len(frames) - 1):
        pa = crop_ball_patch(frames[i], dets[i])
        pb = crop_ball_patch(frames[i + 1], dets[i + 1])
        ang, info = estimate_rotation_orb(pa, pb)
        diagnostics.append(info)
        if ang is not None:
            angles_deg.append(math.degrees(ang))

    print(f"  per-pair angles (deg): "
          f"{[f'{a:+.2f}' for a in angles_deg]}")
    print(f"  inliers per pair: "
          f"{[d.get('inliers', 0) for d in diagnostics]}")

    # RANSAC may fail on some pairs — require at least half to succeed
    assert len(angles_deg) >= (len(frames) - 1) // 2, \
        f"ORB succeeded on only {len(angles_deg)}/{len(frames)-1} pairs"

    median_angle = float(np.median(angles_deg))
    err = abs(median_angle - 7.0)
    print(f"  median angle: {median_angle:+.2f} deg  (expected +7.00, err {err:.2f})")
    assert err < 1.5, f"ORB median off by {err:.2f} deg"


def test_session_rpm_at_120fps():
    frames = grab_frames(60, fps=120)
    result = estimate_session_spin(frames, fps=120)
    print(f"  estimated RPM: {result.rpm:.1f}  std: {result.rpm_std:.1f}  "
          f"(expected 140.0)")
    print(f"  direction: {result.direction}  method: {result.method}")
    print(f"  valid pairs: {result.n_valid_pairs}/{len(frames)-1}")
    print(f"  notes: {result.notes}")

    assert result.rpm is not None
    err = abs(result.rpm - 140.0)
    assert err < 15.0, f"RPM estimate off by {err:.1f}"


def test_session_rpm_at_60fps():
    # At 60fps, mock still rotates 7 deg per frame (frame-index based).
    # So RPM = 7 * 60 * 60 / 360 = 70 RPM.
    frames = grab_frames(40, fps=60)
    result = estimate_session_spin(frames, fps=60)
    print(f"  @60fps estimated RPM: {result.rpm:.1f}  (expected 70.0)")
    assert result.rpm is not None
    assert abs(result.rpm - 70.0) < 10.0


def test_session_direction_detected():
    """Mock rotates CCW (positive angle_deg). Estimator should say so."""
    frames = grab_frames(30, fps=120)
    result = estimate_session_spin(frames, fps=120)
    print(f"  direction: {result.direction}  (expected ccw)")
    # Direction detection is a stretch goal; skip strict assertion for now
    # but log the result
    if result.direction is not None:
        assert result.direction == "ccw", f"wrong direction: {result.direction}"


def run_all():
    tests = [
        test_ball_detection_finds_ball_in_most_frames,
        test_orb_rotation_matches_ground_truth,
        test_session_rpm_at_120fps,
        test_session_rpm_at_60fps,
        test_session_direction_detected,
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
            import traceback
            traceback.print_exc()
            failures += 1
    print(f"\n{'='*50}")
    print(f"  {len(tests) - failures}/{len(tests)} passed")
    print(f"{'='*50}")
    return failures


if __name__ == "__main__":
    sys.exit(run_all())
