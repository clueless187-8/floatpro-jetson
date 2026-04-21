"""
Spin estimator.

Pipeline
--------
1. Detect ball in each frame (simple threshold/blob for now; YOLO later)
2. Crop a square patch centered on the ball, padded slightly beyond its radius
3. Log-polar transform each patch — rotation about the patch center becomes
   vertical translation in the polar image
4. Phase-correlate consecutive polar patches — the vertical translation
   directly encodes the rotation angle (radians per frame)
5. Convert to RPM: angle_per_frame * fps * 60 / (2π)

Why log-polar + phase correlation
----------------------------------
Phase correlation is translation-only; it can't directly measure rotation.
But a rotation in Cartesian about the image center maps to a pure
translation in log-polar space. So we reduce the hard problem (rotation) to
the easy one (translation) by the coordinate transform. This is the
classical "Fourier-Mellin" registration approach, pared down here because
we handle scale separately (we know the ball radius from detection).

Assumptions
-----------
- Ball is approximately centered in each patch (detection is good)
- Rotation between consecutive frames is < π (no aliasing at 120fps unless
  the ball spins faster than 60 rev/sec = 3600 RPM; volleyball caps out at
  ~5 RPM for a good float, ~200 RPM for a heavy topspin serve)
- Scale is approximately constant within a short serve window — true for
  a camera 15+ ft away; a ball that fills the frame and changes size
  frame-to-frame breaks the assumption
"""
from __future__ import annotations

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
    r: float                   # radius in pixels
    confidence: float = 1.0


def detect_ball_simple(frame: np.ndarray,
                       min_radius: int = 6,
                       max_radius: int = 80,
                       bright_threshold: int = 180) -> Optional[Detection]:
    """
    Threshold + largest-blob detector.

    Intended for the mock and for smoke-testing. Real deployment replaces
    this with YOLOv8-nano trained on volleyball footage — same interface,
    returns the same Detection dataclass.
    """
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

    # Circularity as a cheap confidence signal
    perimeter = cv2.arcLength(best, True)
    circularity = (4 * np.pi * best_area) / (perimeter ** 2) if perimeter > 0 else 0

    return Detection(
        frame_index=-1,  # caller fills in
        cx=float(cx),
        cy=float(cy),
        r=float(r),
        confidence=float(min(1.0, circularity)),
    )


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def crop_ball_patch(frame: np.ndarray, det: Detection,
                    padding: float = 1.4, size: int = 96) -> np.ndarray:
    """
    Crop a square region centered on the detected ball, then resize to a
    fixed size so log-polar has a consistent scale across detections.

    padding > 1.0 means include some background around the ball — useful
    because the ball edge itself carries a lot of the rotating texture.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape

    half = int(det.r * padding)
    # Pad the full frame so crops near the edge don't fail
    padded = cv2.copyMakeBorder(gray, half, half, half, half,
                                cv2.BORDER_CONSTANT, value=0)
    cx_p = int(round(det.cx)) + half
    cy_p = int(round(det.cy)) + half

    patch = padded[cy_p - half:cy_p + half, cx_p - half:cx_p + half]
    if patch.shape[0] == 0 or patch.shape[1] == 0:
        return np.zeros((size, size), dtype=np.uint8)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Rotation estimation
# ---------------------------------------------------------------------------

def _log_polar(patch: np.ndarray, polar_size: int = 256) -> np.ndarray:
    """Log-polar transform.

    polar_size controls the angular resolution of the output: a height of
    256 gives 360°/256 = 1.4° per pixel, which phase correlation can
    subdivide to roughly 0.1° of rotation. Going larger gains precision at
    the cost of runtime; 256 is a good compromise for 120fps volleyball.
    """
    h, w = patch.shape
    center = (w / 2.0, h / 2.0)
    max_radius = min(h, w) / 2.0
    return cv2.warpPolar(
        patch,
        (polar_size, polar_size),
        center,
        max_radius,
        cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR,
    )


def estimate_rotation(patch_a: np.ndarray, patch_b: np.ndarray
                      ) -> tuple[float, float]:
    """
    Estimate the rotation (radians) that takes patch_a to patch_b.

    Returns (angle_rad, response). `response` is the phase correlation peak
    strength in [0, 1]; low values mean the frames don't actually share a
    rotating feature (e.g. detection jitter, ball off-frame, motion blur).
    """
    pa = _log_polar(patch_a).astype(np.float32)
    pb = _log_polar(patch_b).astype(np.float32)

    window = cv2.createHanningWindow((pa.shape[1], pa.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(pa * window, pb * window)

    # The polar image wraps 2π across its full height
    h = pa.shape[0]
    angle_rad = dy * 2 * np.pi / h
    return float(angle_rad), float(response)


# ---------------------------------------------------------------------------
# Session-level estimator
# ---------------------------------------------------------------------------

@dataclass
class SpinResult:
    rpm: Optional[float]
    rpm_std: Optional[float]       # variability across frame pairs
    axis_deg: Optional[float]      # rotation axis in the image plane (future)
    n_valid_pairs: int
    n_frames: int
    detections: list[Optional[Detection]] = field(default_factory=list)
    per_pair_rpm: list[float] = field(default_factory=list)
    per_pair_confidence: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def estimate_session_spin(frames: list[np.ndarray],
                          fps: float,
                          patch_size: int = 96,
                          min_confidence: float = 0.08,
                          max_reasonable_rpm: float = 3000.0
                          ) -> SpinResult:
    """
    Run the full pipeline over a list of frames and return a summary.

    Volleyball context:
      - float serve target: < 60 RPM (< 1 rev from contact to reception)
      - topspin serve:      ~300-900 RPM
      - jump float:         60-180 RPM
      - max_reasonable_rpm cuts off clearly-bogus estimates (blur, detection fail)
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
            rpm=None, rpm_std=None, axis_deg=None,
            n_valid_pairs=0, n_frames=len(frames),
            detections=detections, notes=notes,
        )

    # Pair-wise rotation estimates
    per_pair_rpm: list[float] = []
    per_pair_conf: list[float] = []
    for i in range(len(frames) - 1):
        da, db = detections[i], detections[i + 1]
        if da is None or db is None:
            continue

        pa = crop_ball_patch(frames[i], da, size=patch_size)
        pb = crop_ball_patch(frames[i + 1], db, size=patch_size)

        angle_rad, response = estimate_rotation(pa, pb)
        if response < min_confidence:
            continue

        # Radians per frame → RPM. Use absolute value for now; axis
        # direction estimation comes later (needs 2+ rotation axes or a
        # color-pattern prior).
        rpm = abs(angle_rad) * fps * 60.0 / (2 * np.pi)
        if rpm > max_reasonable_rpm:
            continue

        per_pair_rpm.append(rpm)
        per_pair_conf.append(response)

    if not per_pair_rpm:
        notes.append("no frame pair produced a confident rotation estimate")
        return SpinResult(
            rpm=None, rpm_std=None, axis_deg=None,
            n_valid_pairs=0, n_frames=len(frames),
            detections=detections,
            per_pair_rpm=[], per_pair_confidence=[],
            notes=notes,
        )

    rpm_arr = np.array(per_pair_rpm)
    # Robust central tendency: trimmed mean of middle 60%
    lo, hi = np.percentile(rpm_arr, [20, 80])
    trimmed = rpm_arr[(rpm_arr >= lo) & (rpm_arr <= hi)]
    rpm_est = float(np.mean(trimmed)) if len(trimmed) else float(np.median(rpm_arr))
    rpm_std = float(np.std(rpm_arr))

    notes.append(
        f"{len(per_pair_rpm)} valid pairs / {len(frames)-1} total  "
        f"detections={n_detected}/{len(frames)}"
    )

    return SpinResult(
        rpm=rpm_est,
        rpm_std=rpm_std,
        axis_deg=None,
        n_valid_pairs=len(per_pair_rpm),
        n_frames=len(frames),
        detections=detections,
        per_pair_rpm=per_pair_rpm,
        per_pair_confidence=per_pair_conf,
        notes=notes,
    )
