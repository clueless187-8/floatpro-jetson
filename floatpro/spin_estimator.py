"""
Spin estimator — ORB feature matching with RANSAC.

Why ORB + RANSAC instead of log-polar + phase correlation
----------------------------------------------------------
Phase correlation on log-polar transforms is the textbook approach for
image rotation estimation, but it has two properties that hurt here:

1. Dominant DC signal. A ball is a bright disc on dark background. The
   disc silhouette is rotationally symmetric, contributes nothing to the
   rotation estimate, but dominates the phase correlation peak — biasing
   estimates toward zero shift.
2. Weak angular resolution. Polar images need to be large (256+ px tall)
   for sub-degree precision, which slows the pipeline.

ORB + RANSAC flips this: we detect distinctive keypoints (panel-seam
corners, printed-logo edges, pattern features) in each ball patch, match
them with a brute-force Hamming matcher, and fit a rotation + translation
+ uniform-scale model with cv2.estimateAffinePartial2D, which rejects
outliers via RANSAC. The benefits:

- Direct measurement: per-keypoint angular shift around the ball center
- Automatic outlier rejection — spurious matches get filtered out
- Sub-pixel keypoint localization gives sub-degree angular precision
- Also recovers the residual translation, telling us how much the
  detected ball center moved sub-pixel between frames

Fallback: when ORB finds too few matches (motion-blurred frames, plain
white ball with no pattern visible), we fall back to log-polar phase
correlation on masked patches (circular mask + DC subtraction to
suppress the disc silhouette).

Assumptions
-----------
- Ball is detected in each frame with sub-ball-radius center accuracy
- Rotation between consecutive frames is < 60° (at 120fps this is the
  equivalent of 1200 RPM, well above any volleyball serve)
- Scale is approximately constant within the spin-measurement window
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    frame_index: int
    cx: float
    cy: float
    r: float
    confidence: float = 1.0


def detect_ball_simple(frame: np.ndarray,
                       min_radius: int = 6,
                       max_radius: int = 80,
                       bright_threshold: int = 180) -> Optional[Detection]:
    """Threshold + largest-blob detector. Placeholder until we ship YOLO."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    _, thresh = cv2.threshold(gray, bright_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < np.pi * min_radius ** 2:
            continue
        if area > best_area:
            best_area = area
            best = c
    if best is None:
        return None

    (cx, cy), r = cv2.minEnclosingCircle(best)
    if r < min_radius or r > max_radius:
        return None

    perimeter = cv2.arcLength(best, True)
    circularity = (4 * np.pi * best_area) / (perimeter ** 2) if perimeter > 0 else 0
    return Detection(
        frame_index=-1,
        cx=float(cx), cy=float(cy), r=float(r),
        confidence=float(min(1.0, circularity)),
    )


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def crop_ball_patch(frame: np.ndarray, det: Detection,
                    padding: float = 1.15, size: int = 128) -> np.ndarray:
    """Crop a square region centered on the ball and resize to a fixed
    pixel size so feature scales are consistent across detections."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape
    half = int(det.r * padding)
    padded = cv2.copyMakeBorder(gray, half, half, half, half,
                                cv2.BORDER_CONSTANT, value=0)
    cx_p = int(round(det.cx)) + half
    cy_p = int(round(det.cy)) + half
    patch = padded[cy_p - half:cy_p + half, cx_p - half:cx_p + half]
    if patch.shape[0] == 0 or patch.shape[1] == 0:
        return np.zeros((size, size), dtype=np.uint8)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Primary estimator: ORB + RANSAC
# ---------------------------------------------------------------------------

# Module-level detector — reused across calls. nfeatures=500 is overkill
# for a ball patch but cheap, and having headroom helps when features
# cluster on one side of the ball.
_ORB = cv2.ORB_create(
    nfeatures=500,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=5,      # allow features near patch edge
    patchSize=15,         # small patches suit our small ball crops
    fastThreshold=10,     # lower = more keypoints on low-contrast balls
)


_INTERIOR_MASK_CACHE: dict[int, np.ndarray] = {}


def _interior_mask(size: int, inner_ratio: float = 0.80) -> np.ndarray:
    """Circular mask (uint8, 0/255) that keeps only the ball interior,
    excluding the high-gradient edge where keypoints are rotation-
    invariant. Cached so we don't rebuild it every call."""
    key = (size, inner_ratio)
    cached = _INTERIOR_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    yy, xx = np.ogrid[:size, :size]
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2)
    mask = np.where(r < inner_ratio, 255, 0).astype(np.uint8)
    _INTERIOR_MASK_CACHE[key] = mask
    return mask


def estimate_rotation_orb(patch_a: np.ndarray, patch_b: np.ndarray
                          ) -> tuple[Optional[float], dict]:
    """
    ORB + BFMatcher + RANSAC affine-partial-2d fit.

    Returns
    -------
    angle_rad : float | None
        Rotation in radians. None if insufficient inliers.
    info : dict
        Diagnostics: kp counts, match count, inlier count, residual.
    """
    info = {"kp_a": 0, "kp_b": 0, "matches": 0, "inliers": 0}

    # Mask ORB detection to the ball interior so the rotation-invariant
    # ball edge does not dominate the keypoint set.
    mask = _interior_mask(patch_a.shape[0])
    kp_a, desc_a = _ORB.detectAndCompute(patch_a, mask)
    kp_b, desc_b = _ORB.detectAndCompute(patch_b, mask)
    info["kp_a"] = len(kp_a) if kp_a else 0
    info["kp_b"] = len(kp_b) if kp_b else 0

    if desc_a is None or desc_b is None or info["kp_a"] < 8 or info["kp_b"] < 8:
        return None, info

    # Cross-checked Hamming BF match — high precision, moderate recall
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = list(bf.match(desc_a, desc_b))
    info["matches"] = len(matches)
    if len(matches) < 5:
        return None, info

    # Sort by descriptor distance and keep the best 40 (RANSAC handles outliers)
    matches.sort(key=lambda m: m.distance)
    matches = matches[:40]

    src = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # estimateAffinePartial2D fits [rotation, uniform scale, translation]
    # with RANSAC. Rejects spurious matches. Returns the affine matrix and
    # a per-match inlier mask.
    M, inlier_mask = cv2.estimateAffinePartial2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.995,
    )
    if M is None:
        return None, info

    info["inliers"] = int(inlier_mask.sum()) if inlier_mask is not None else 0
    # Require a meaningful inlier consensus. Below ~6 the RANSAC fit is
    # likely overfitted to noise and produces wild per-pair estimates
    # that wreck the session statistics.
    if info["inliers"] < 6:
        return None, info

    # Recover rotation from the 2x3 affine matrix.
    # [ s*cos(θ)  -s*sin(θ)  tx ]
    # [ s*sin(θ)   s*cos(θ)  ty ]
    angle_rad = math.atan2(M[1, 0], M[0, 0])
    info["scale"] = math.hypot(M[0, 0], M[1, 0])
    info["tx"] = float(M[0, 2])
    info["ty"] = float(M[1, 2])
    return float(angle_rad), info


# ---------------------------------------------------------------------------
# Fallback estimator: masked log-polar phase correlation
# ---------------------------------------------------------------------------

def _masked_patch(patch: np.ndarray, mask_ratio: float = 0.90) -> np.ndarray:
    """Apply circular mask (keep interior) and subtract interior mean to
    kill the disc's DC signal. Output is float32, zero outside mask."""
    size = patch.shape[0]
    yy, xx = np.ogrid[:size, :size]
    r_norm = np.sqrt((xx - size/2) ** 2 + (yy - size/2) ** 2) / (size/2)
    mask = (r_norm < mask_ratio).astype(np.float32)
    p = patch.astype(np.float32)
    mean = (p * mask).sum() / (mask.sum() + 1e-9)
    return (p - mean) * mask


def estimate_rotation_logpolar(patch_a: np.ndarray, patch_b: np.ndarray
                               ) -> tuple[float, float]:
    """Fallback: phase correlation on log-polar of masked patches."""
    ma = _masked_patch(patch_a)
    mb = _masked_patch(patch_b)
    size = ma.shape[0]
    polar_size = 256
    center = (size / 2.0, size / 2.0)
    max_radius = size / 2.0
    la = cv2.warpPolar(ma, (polar_size, polar_size), center, max_radius,
                       cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR)
    lb = cv2.warpPolar(mb, (polar_size, polar_size), center, max_radius,
                       cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR)
    (dx, dy), response = cv2.phaseCorrelate(la, lb)
    h = la.shape[0]
    angle_rad = dy * 2 * np.pi / h
    return float(angle_rad), float(response)


# ---------------------------------------------------------------------------
# Session-level estimator
# ---------------------------------------------------------------------------

@dataclass
class SpinResult:
    rpm: Optional[float]
    rpm_std: Optional[float]
    direction: Optional[str]             # "cw" or "ccw" (image plane)
    axis_deg: Optional[float]
    n_valid_pairs: int
    n_frames: int
    method: str                          # "orb" or "logpolar" or "mixed"
    detections: list[Optional[Detection]] = field(default_factory=list)
    per_pair_rpm: list[float] = field(default_factory=list)
    per_pair_angle_deg: list[float] = field(default_factory=list)
    per_pair_method: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def estimate_session_spin(frames: list[np.ndarray],
                          fps: float,
                          patch_size: int = 128,
                          max_reasonable_rpm: float = 3000.0,
                          use_fallback: bool = True
                          ) -> SpinResult:
    """
    Run ORB-primary, log-polar-fallback pipeline over a frame sequence.

    Volleyball context:
      - float serve target:  < 60 RPM  (< 1 rev from contact to reception)
      - topspin serve:       ~300-900 RPM
      - jump float:          60-180 RPM
    """
    notes: list[str] = []
    detections: list[Optional[Detection]] = []

    for i, frame in enumerate(frames):
        d = detect_ball_simple(frame)
        if d is not None:
            d.frame_index = i
        detections.append(d)

    n_detected = sum(1 for d in detections if d is not None)
    if n_detected < 2:
        notes.append(f"only {n_detected} detections — cannot estimate spin")
        return SpinResult(
            rpm=None, rpm_std=None, direction=None, axis_deg=None,
            n_valid_pairs=0, n_frames=len(frames), method="none",
            detections=detections, notes=notes,
        )

    per_pair_rpm: list[float] = []
    per_pair_angle_deg: list[float] = []
    per_pair_method: list[str] = []
    orb_failures = 0

    for i in range(len(frames) - 1):
        da, db = detections[i], detections[i + 1]
        if da is None or db is None:
            continue

        pa = crop_ball_patch(frames[i], da, size=patch_size)
        pb = crop_ball_patch(frames[i + 1], db, size=patch_size)

        # Primary: ORB + RANSAC
        angle_rad, info = estimate_rotation_orb(pa, pb)
        method = "orb"

        if angle_rad is None and use_fallback:
            angle_rad, response = estimate_rotation_logpolar(pa, pb)
            if response < 0.10:
                continue
            method = "logpolar"
            orb_failures += 1

        if angle_rad is None:
            orb_failures += 1
            continue

        rpm = abs(angle_rad) * fps * 60.0 / (2 * np.pi)
        if rpm > max_reasonable_rpm:
            continue

        per_pair_rpm.append(rpm)
        per_pair_angle_deg.append(math.degrees(angle_rad))
        per_pair_method.append(method)

    if not per_pair_rpm:
        notes.append(f"no pair yielded a usable estimate "
                     f"(orb_failures={orb_failures})")
        return SpinResult(
            rpm=None, rpm_std=None, direction=None, axis_deg=None,
            n_valid_pairs=0, n_frames=len(frames), method="none",
            detections=detections, notes=notes,
        )

    # Robust central tendency: median is far more resistant to the
    # occasional RANSAC catastrophic outlier (we saw 1700+ RPM blips on
    # otherwise-good sessions) than any trimmed-mean scheme.
    rpm_arr = np.array(per_pair_rpm)
    rpm_est = float(np.median(rpm_arr))

    # MAD-based spread estimate — also robust
    mad = float(np.median(np.abs(rpm_arr - rpm_est)))
    rpm_std = 1.4826 * mad  # MAD→stdev conversion for a normal distribution

    # Direction: sign of the median of the SIGNED angle array. Filter to
    # pairs whose RPM magnitude is within 3*MAD of the estimate, so rare
    # sign-flipped outliers don't fight the consensus.
    angle_arr = np.array(per_pair_angle_deg)
    consistent = np.abs(rpm_arr - rpm_est) < max(3 * mad, 5.0)
    direction = None
    if consistent.sum() >= 3:
        sign = np.sign(np.median(angle_arr[consistent]))
        direction = "ccw" if sign > 0 else "cw"

    methods_used = set(per_pair_method)
    method_tag = "orb" if methods_used == {"orb"} else (
        "logpolar" if methods_used == {"logpolar"} else "mixed"
    )

    notes.append(
        f"{len(per_pair_rpm)} valid pairs / {len(frames)-1} total  "
        f"detections={n_detected}/{len(frames)}  "
        f"orb_failures={orb_failures}  method={method_tag}"
    )

    return SpinResult(
        rpm=rpm_est,
        rpm_std=rpm_std,
        direction=direction,
        axis_deg=None,
        n_valid_pairs=len(per_pair_rpm),
        n_frames=len(frames),
        method=method_tag,
        detections=detections,
        per_pair_rpm=per_pair_rpm,
        per_pair_angle_deg=per_pair_angle_deg,
        per_pair_method=per_pair_method,
        notes=notes,
    )
