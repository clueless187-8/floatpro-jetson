"""Hypothesis: the bright ball disc dominates phase correlation.
Fix: mask to ball interior only, subtract DC before polar transform."""
import sys, os, math
sys.path.insert(0, ".")
os.makedirs("debug_out", exist_ok=True)

import numpy as np
import cv2

from floatpro.cameras import make_camera, CameraConfig
from floatpro.spin_estimator import detect_ball_simple, crop_ball_patch

cam = make_camera("mock", CameraConfig(width=640, height=480, fps=120))
cam.start()
frames = [cam.read()[1] for _ in range(8)]
cam.stop()


def masked_patch(frame, det, patch_size=128, padding=1.15):
    """Crop + apply circular mask (keep interior, zero outside)."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    half = int(det.r * padding)
    padded = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_CONSTANT, 0)
    cx_p = int(round(det.cx)) + half
    cy_p = int(round(det.cy)) + half
    patch = padded[cy_p - half:cy_p + half, cx_p - half:cx_p + half].astype(np.float32)
    patch = cv2.resize(patch, (patch_size, patch_size))

    # Circular mask — keep pixels inside ~0.95 * ball radius
    yy, xx = np.ogrid[:patch_size, :patch_size]
    cx_c, cy_c = patch_size / 2, patch_size / 2
    r_norm = np.sqrt((xx - cx_c) ** 2 + (yy - cy_c) ** 2) / (patch_size / 2)
    mask = (r_norm < 0.90).astype(np.float32)
    interior_mean = float((patch * mask).sum() / (mask.sum() + 1e-9))
    # Subtract mean INSIDE mask, then reapply mask so outside is zero
    centered = (patch - interior_mean) * mask
    return centered, mask


def polar(patch, out_size=256, log=True):
    h, w = patch.shape
    flag = cv2.WARP_POLAR_LOG if log else cv2.WARP_POLAR_LINEAR
    return cv2.warpPolar(patch, (out_size, out_size), (w/2, h/2),
                         min(h, w)/2, flag + cv2.INTER_LINEAR)


print("Masked + DC-subtracted approach (ground truth = 7 deg/frame):\n")
print(f"{'pair':6s} {'log':>10s} {'linear':>10s} {'log+win':>10s} {'linear+win':>12s}")

for i in range(len(frames) - 1):
    da = detect_ball_simple(frames[i])
    db = detect_ball_simple(frames[i + 1])
    if not da or not db:
        continue
    pa, _ = masked_patch(frames[i], da)
    pb, _ = masked_patch(frames[i + 1], db)

    # Save images of first pair for inspection
    if i == 2:
        cv2.imwrite("debug_out/masked_patch_a.png",
                    ((pa - pa.min()) / (pa.max() - pa.min() + 1e-9) * 255).astype(np.uint8))
        cv2.imwrite("debug_out/masked_patch_b.png",
                    ((pb - pb.min()) / (pb.max() - pb.min() + 1e-9) * 255).astype(np.uint8))

    results = {}
    for name, log_flag in [("log", True), ("linear", False)]:
        la = polar(pa, log=log_flag)
        lb = polar(pb, log=log_flag)
        if i == 2:
            cv2.imwrite(f"debug_out/masked_polar_{name}_a.png",
                        ((la - la.min()) / (la.max() - la.min() + 1e-9) * 255).astype(np.uint8))
        # no window
        (dx, dy), resp = cv2.phaseCorrelate(la, lb)
        angle = dy * 2 * np.pi / la.shape[0]
        results[name] = (math.degrees(angle), resp)
        # with window
        win = cv2.createHanningWindow((la.shape[1], la.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(la * win, lb * win)
        angle = dy * 2 * np.pi / la.shape[0]
        results[name + "+win"] = (math.degrees(angle), resp)

    print(f"{i}->{i+1}  "
          f"{results['log'][0]:+6.2f}({results['log'][1]:.2f})  "
          f"{results['linear'][0]:+6.2f}({results['linear'][1]:.2f})  "
          f"{results['log+win'][0]:+6.2f}({results['log+win'][1]:.2f})  "
          f"{results['linear+win'][0]:+6.2f}({results['linear+win'][1]:.2f})")

# Also try ORB feature matching as alternative
print("\nORB feature matching approach:")
orb = cv2.ORB_create(nfeatures=200)
for i in range(len(frames) - 1):
    da = detect_ball_simple(frames[i])
    db = detect_ball_simple(frames[i + 1])
    if not da or not db:
        continue
    # Use un-masked patches but tighter crop
    pa = crop_ball_patch(frames[i], da, size=128, padding=1.1)
    pb = crop_ball_patch(frames[i + 1], db, size=128, padding=1.1)

    kp_a, desc_a = orb.detectAndCompute(pa, None)
    kp_b, desc_b = orb.detectAndCompute(pb, None)

    if desc_a is None or desc_b is None or len(kp_a) < 3 or len(kp_b) < 3:
        print(f"  {i}->{i+1}: not enough features (a={len(kp_a) if kp_a else 0}, b={len(kp_b) if kp_b else 0})")
        continue

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_a, desc_b)
    matches = sorted(matches, key=lambda m: m.distance)[:20]

    if len(matches) < 3:
        print(f"  {i}->{i+1}: not enough matches ({len(matches)})")
        continue

    # Estimate rotation from matched points using angles around center
    center = 64.0  # half of 128
    angles = []
    for m in matches:
        pa_pt = kp_a[m.queryIdx].pt
        pb_pt = kp_b[m.trainIdx].pt
        a_a = math.atan2(pa_pt[1] - center, pa_pt[0] - center)
        a_b = math.atan2(pb_pt[1] - center, pb_pt[0] - center)
        d = a_b - a_a
        # Wrap to [-pi, pi]
        while d > math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        angles.append(d)

    # Robust estimate: median
    angles_arr = np.array(angles)
    median = math.degrees(np.median(angles_arr))
    # Also: filter outliers and take mean
    filtered = angles_arr[np.abs(angles_arr - np.median(angles_arr)) < math.radians(15)]
    mean_filt = math.degrees(np.mean(filtered)) if len(filtered) > 0 else float('nan')
    print(f"  {i}->{i+1}: matches={len(matches):3d}  median={median:+6.2f}  filtered_mean={mean_filt:+6.2f}")
