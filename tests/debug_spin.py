"""Diagnose the scale error in spin estimation."""
import sys
sys.path.insert(0, ".")

import math
import numpy as np
import cv2

from floatpro.cameras import make_camera, CameraConfig
from floatpro.spin_estimator import detect_ball_simple, crop_ball_patch, _log_polar

# Grab two consecutive frames from the mock
cam = make_camera("mock", CameraConfig(width=640, height=480, fps=120))
cam.start()
frames = []
for _ in range(4):
    ok, frame, ts = cam.read()
    if ok: frames.append(frame)
cam.stop()

# Test 1: is detection center stable?
print("TEST 1: Detection center stability")
for i, f in enumerate(frames):
    d = detect_ball_simple(f)
    print(f"  frame {i}: cx={d.cx:.2f} cy={d.cy:.2f} r={d.r:.2f}")

# Test 2: use EXACT known center (bypass detector)
# mock places ball at deterministic (x, y) per frame_index
print("\nTEST 2: Use exact known ball center (bypass detector)")
# Reconstruct the mock's ball position logic
W, H = 640, 480
def mock_ball_pos(frame_i, fps=120):
    t = frame_i / fps
    x = int(0.1 * W + (0.8 * W) * (t % 2.0) / 2.0)
    y = int(H * 0.3 + 300 * np.sin(np.pi * (t % 2.0) / 2.0))
    y = max(20, min(H - 20, y))
    return x, y

for i, f in enumerate(frames):
    x, y = mock_ball_pos(i)
    print(f"  frame {i}: mock says ball at ({x}, {y})")
    d = detect_ball_simple(f)
    print(f"    detector: ({d.cx:.2f}, {d.cy:.2f})  delta=({d.cx-x:.2f}, {d.cy-y:.2f})")

# Test 3: crop with exact center, phase correlate
print("\nTEST 3: Phase correlation with exact center, linear polar")

def crop_exact(frame, cx, cy, half=40, size=128):
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    padded = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_CONSTANT, 0)
    cx_p, cy_p = cx + half, cy + half
    patch = padded[cy_p - half:cy_p + half, cx_p - half:cx_p + half]
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)

def linear_polar(patch):
    h, w = patch.shape
    center = (w / 2.0, h / 2.0)
    max_radius = min(h, w) / 2.0
    return cv2.warpPolar(patch, (w, h), center, max_radius, cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)

def rotation_from_polar(polar_a, polar_b):
    pa = polar_a.astype(np.float32)
    pb = polar_b.astype(np.float32)
    window = cv2.createHanningWindow((pa.shape[1], pa.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(pa * window, pb * window)
    h = pa.shape[0]
    angle = dy * 2 * np.pi / h
    return angle, resp

# With exact centers
for i in range(len(frames) - 1):
    xa, ya = mock_ball_pos(i)
    xb, yb = mock_ball_pos(i + 1)
    pa = crop_exact(frames[i], xa, ya, half=40, size=128)
    pb = crop_exact(frames[i + 1], xb, yb, half=40, size=128)

    # Log polar
    la, lb = _log_polar(pa), _log_polar(pb)
    angle_log, resp_log = rotation_from_polar(la, lb)

    # Linear polar
    linA, linB = linear_polar(pa), linear_polar(pb)
    angle_lin, resp_lin = rotation_from_polar(linA, linB)

    print(f"  pair {i}->{i+1}:")
    print(f"    log-polar:    {math.degrees(angle_log):+.3f} deg  (conf {resp_log:.3f})")
    print(f"    linear polar: {math.degrees(angle_lin):+.3f} deg  (conf {resp_lin:.3f})")

# Test 4: synthetic rotation on a known pattern — sanity check the math itself
print("\nTEST 4: Synthetic rotation of known test pattern (no camera)")
# Draw a simple 3-dot pattern at exactly known angle, rotate by exactly 7 deg,
# and verify we recover it.
def make_test_pattern(angle_deg, size=128, dot_r=3, orbit_r=40):
    img = np.full((size, size), 40, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), orbit_r + 10, 255, -1)  # outer "ball"
    for k in range(3):
        a = math.radians(angle_deg + k * 120)
        x = int(size / 2 + orbit_r * math.cos(a))
        y = int(size / 2 + orbit_r * math.sin(a))
        cv2.circle(img, (x, y), dot_r, 100, -1)
    return img

for true_angle in [3.0, 5.0, 7.0, 10.0, 15.0]:
    a = make_test_pattern(0)
    b = make_test_pattern(true_angle)
    ang_log, resp_log = rotation_from_polar(_log_polar(a), _log_polar(b))
    ang_lin, resp_lin = rotation_from_polar(linear_polar(a), linear_polar(b))
    print(f"  true {true_angle:+5.2f} deg  "
          f"log={math.degrees(ang_log):+6.2f} (c={resp_log:.2f})  "
          f"linear={math.degrees(ang_lin):+6.2f} (c={resp_lin:.2f})")
