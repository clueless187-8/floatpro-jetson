"""Dump patches and polar transforms to PNG for visual inspection."""
import sys
sys.path.insert(0, ".")
import os
os.makedirs("debug_out", exist_ok=True)

import math
import numpy as np
import cv2

from floatpro.cameras import make_camera, CameraConfig
from floatpro.spin_estimator import detect_ball_simple, crop_ball_patch

cam = make_camera("mock", CameraConfig(width=640, height=480, fps=120))
cam.start()
frames = [cam.read()[1] for _ in range(5)]
cam.stop()

# Save frames
for i, f in enumerate(frames):
    cv2.imwrite(f"debug_out/frame_{i}.png", f)

# Save crops
for i, f in enumerate(frames):
    d = detect_ball_simple(f)
    if d:
        patch = crop_ball_patch(f, d, size=128, padding=1.3)
        cv2.imwrite(f"debug_out/crop_{i}.png", patch)

# Save log-polar
def log_polar(patch, out_size=256):
    h, w = patch.shape
    return cv2.warpPolar(patch, (out_size, out_size), (w/2, h/2),
                         min(h, w)/2, cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR)

# Save linear-polar  
def lin_polar(patch, out_size=256):
    h, w = patch.shape
    return cv2.warpPolar(patch, (out_size, out_size), (w/2, h/2),
                         min(h, w)/2, cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)

for i, f in enumerate(frames):
    d = detect_ball_simple(f)
    if d:
        patch = crop_ball_patch(f, d, size=128, padding=1.3)
        cv2.imwrite(f"debug_out/logpolar_{i}.png", log_polar(patch))
        cv2.imwrite(f"debug_out/linpolar_{i}.png", lin_polar(patch))

# Now try phase correlation with NO window, and print what we get
print("Log-polar rotation estimates (no hanning):")
for i in range(len(frames) - 1):
    da = detect_ball_simple(frames[i])
    db = detect_ball_simple(frames[i+1])
    if not da or not db:
        continue
    pa = crop_ball_patch(frames[i], da, size=128, padding=1.3)
    pb = crop_ball_patch(frames[i+1], db, size=128, padding=1.3)
    
    la = log_polar(pa).astype(np.float32)
    lb = log_polar(pb).astype(np.float32)
    
    (dx, dy), resp = cv2.phaseCorrelate(la, lb)
    h = la.shape[0]
    angle = dy * 2 * np.pi / h
    print(f"  {i}->{i+1}: no-window log:    {math.degrees(angle):+6.2f} deg (conf {resp:.3f})")
    
    # With window
    win = cv2.createHanningWindow((la.shape[1], la.shape[0]), cv2.CV_32F)
    (dx2, dy2), resp2 = cv2.phaseCorrelate(la*win, lb*win)
    angle2 = dy2 * 2 * np.pi / h
    print(f"           windowed log:    {math.degrees(angle2):+6.2f} deg (conf {resp2:.3f})")
    
    # Linear polar
    Lia = lin_polar(pa).astype(np.float32)
    Lib = lin_polar(pb).astype(np.float32)
    (dx3, dy3), resp3 = cv2.phaseCorrelate(Lia, Lib)
    angle3 = dy3 * 2 * np.pi / h
    print(f"           no-window linear: {math.degrees(angle3):+6.2f} deg (conf {resp3:.3f})")
    
    (dx4, dy4), resp4 = cv2.phaseCorrelate(Lia*win, Lib*win)
    angle4 = dy4 * 2 * np.pi / h
    print(f"           windowed linear:  {math.degrees(angle4):+6.2f} deg (conf {resp4:.3f})")

print("\nExpected: 7 deg per frame")
print("\nFiles in debug_out/: frame_*, crop_*, logpolar_*, linpolar_*")

# List the debug files
for f in sorted(os.listdir("debug_out"))[:15]:
    print(f"  {f}")
