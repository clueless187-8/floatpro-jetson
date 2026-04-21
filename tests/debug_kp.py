import sys, os
sys.path.insert(0, ".")
os.makedirs("debug_out", exist_ok=True)

import cv2
import numpy as np

from floatpro.cameras import make_camera, CameraConfig
from floatpro.spin_estimator import detect_ball_simple, crop_ball_patch, _ORB

cam = make_camera("mock", CameraConfig(width=640, height=480, fps=120))
cam.start()
frames = [cam.read()[1] for _ in range(5)]
cam.stop()

for i, f in enumerate(frames[:3]):
    d = detect_ball_simple(f)
    patch = crop_ball_patch(f, d, size=128, padding=1.15)
    kp, desc = _ORB.detectAndCompute(patch, None)
    print(f"frame {i}: {len(kp)} keypoints")
    center = (64, 64)
    distances = [np.hypot(k.pt[0] - center[0], k.pt[1] - center[1]) for k in kp]
    print(f"  radial dist distribution: min={min(distances):.1f} "
          f"max={max(distances):.1f} mean={np.mean(distances):.1f}")
    # Print ball radius in patch
    R_in_patch = d.r / (d.r * 1.15) * 64  # should be ~56
    print(f"  ball radius in patch: ~{R_in_patch:.1f} px (of 64 half-width)")

    # Draw keypoints for visual
    vis = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    vis = cv2.drawKeypoints(vis, kp, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(f"debug_out/kp_frame_{i}.png", vis)
    print(f"  saved: debug_out/kp_frame_{i}.png")
