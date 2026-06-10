"""
Validate trajectory analysis against the mock's deterministic motion.

Mock motion model (per mock.py):
    t = frame_i / fps
    x(t) = 0.1*W + 0.8*W * (t % 2) / 2        → vx = 0.4*W px/s (constant)
    y(t) = 0.3*H + 300*sin(pi * (t % 2) / 2)  → vy = 150*pi*cos(pi*t/2) px/s
Ball radius: 30 px. Volleyball real radius 0.105 m
    → meters_per_px (ball calibration) = 0.105/30 = 0.0035
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")

from floatpro.cameras import make_camera, CameraConfig
from floatpro.spin_estimator import detect_ball_simple
from floatpro.trajectory import analyze_trajectory
from floatpro.analysis import analyze_session, knuckle_index, classify_serve

W, H, FPS = 640, 480, 120


def grab(n):
    cam = make_camera("mock", CameraConfig(width=W, height=H, fps=FPS))
    cam.start()
    frames = []
    for _ in range(n):
        ok, f, ts = cam.read()
        if ok:
            frames.append(f)
    cam.stop()
    return frames


def expected_median_speed_px_s(n_frames):
    """Numerically compute the mock's ground-truth speed over the window."""
    speeds = []
    for i in range(n_frames):
        t = i / FPS
        vx = 0.4 * W
        vy = 150 * math.pi * math.cos(math.pi * t / 2)
        speeds.append(math.hypot(vx, vy))
    return float(np.median(speeds))


def test_trajectory_speed_matches_ground_truth():
    n = 60
    frames = grab(n)
    dets = [detect_ball_simple(f) for f in frames]
    for i, d in enumerate(dets):
        if d:
            d.frame_index = i

    result = analyze_trajectory(dets, fps=FPS)
    expected = expected_median_speed_px_s(n)

    print(f"  measured speed: {result.speed_px_s:.1f} px/s")
    print(f"  expected speed: {expected:.1f} px/s")
    err_pct = abs(result.speed_px_s - expected) / expected * 100
    print(f"  error: {err_pct:.1f}%")
    assert err_pct < 10, f"speed error {err_pct:.1f}% > 10%"


def test_ball_diameter_calibration():
    frames = grab(30)
    dets = [detect_ball_simple(f) for f in frames]
    result = analyze_trajectory(dets, fps=FPS)
    # Mock ball R=30px, real volleyball R=0.105m → 3.5 mm/px
    expected_mpp = 0.105 / 30
    print(f"  meters_per_px: {result.meters_per_px*1000:.3f} mm/px "
          f"(expected {expected_mpp*1000:.3f})")
    err_pct = abs(result.meters_per_px - expected_mpp) / expected_mpp * 100
    # minEnclosingCircle slightly overestimates radius on anti-aliased
    # edges; allow 10%
    assert err_pct < 10, f"calibration error {err_pct:.1f}%"


def test_break_and_wobble_computed():
    frames = grab(60)
    dets = [detect_ball_simple(f) for f in frames]
    result = analyze_trajectory(dets, fps=FPS)
    print(f"  break: {result.break_px:.1f} px = {result.break_in:.2f} in")
    print(f"  wobble: {result.wobble_px:.3f} px")
    # Mock path is curved (sinusoidal y), so break must be positive
    assert result.break_px is not None and result.break_px > 0
    assert result.wobble_px is not None and result.wobble_px >= 0


def test_knuckle_index_math():
    # Perfect float: fast, no spin, max wobble
    ki = knuckle_index(speed_mph=50, rpm=0, wobble_px=3.0)
    print(f"  perfect float KI: {ki:.1f}")
    assert abs(ki - 100.0) < 0.1

    # Heavy topspin kills the score regardless of speed
    ki = knuckle_index(speed_mph=60, rpm=400, wobble_px=3.0)
    print(f"  heavy topspin KI: {ki:.1f}")
    assert ki == 0.0

    # Missing inputs → None
    assert knuckle_index(None, 50, 1.0) is None

    # Mid-grade float
    ki = knuckle_index(speed_mph=40, rpm=45, wobble_px=1.5)
    print(f"  mid float KI: {ki:.1f}")
    assert 0 < ki < 100


def test_classifier():
    assert classify_serve(rpm=20, speed_mph=45) == "float"
    assert classify_serve(rpm=100, speed_mph=45) == "jump_float"
    assert classify_serve(rpm=100, speed_mph=25) == "hybrid"
    assert classify_serve(rpm=500, speed_mph=55) == "topspin"
    assert classify_serve(rpm=None, speed_mph=50) is None
    print("  all classification boundaries correct")


def test_full_analysis_pipeline():
    frames = grab(60)
    payload = analyze_session(frames, fps=FPS)

    print(f"  rpm:           {payload['rpm']:.1f}")
    print(f"  speed_mph:     {payload['speed_mph']:.1f}")
    print(f"  break_in:      {payload['break_in']:.2f}")
    print(f"  wobble_px:     {payload['wobble_px']:.3f}")
    print(f"  knuckle_index: {payload['knuckle_index']}")
    print(f"  serve_type:    {payload['serve_type']}")

    assert payload["rpm"] is not None
    assert payload["speed_mph"] is not None
    assert payload["break_in"] is not None
    assert payload["serve_type"] is not None
    assert payload["analysis_version"] == 2
    # JSON-serializable check
    import json
    s = json.dumps(payload)
    assert len(s) > 100
    print(f"  payload serializes: {len(s)} chars")


def run_all():
    tests = [
        test_trajectory_speed_matches_ground_truth,
        test_ball_diameter_calibration,
        test_break_and_wobble_computed,
        test_knuckle_index_math,
        test_classifier,
        test_full_analysis_pipeline,
    ]
    failures = 0
    for t in tests:
        print(f"\n[RUN] {t.__name__}")
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            failures += 1
    print(f"\n{'='*50}\n  {len(tests)-failures}/{len(tests)} passed\n{'='*50}")
    return failures


if __name__ == "__main__":
    sys.exit(run_all())
