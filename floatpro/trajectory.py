"""
Trajectory analysis: velocity, path, and break (lateral deviation).

Calibration strategy
--------------------
Two ways to convert pixels to meters:

1. **Ball-diameter calibration (default, zero-setup).** A regulation
   volleyball is 21 cm in diameter. The detector gives us the ball's
   radius in pixels every frame, so meters_per_px = 0.105 / median(r_px).
   Works in any gym with no court-point tapping. Accuracy degrades if
   the ball moves significantly toward/away from the camera during the
   measurement window (scale changes), so we use the median radius and
   flag high radius variance in the notes.

2. **Court homography (future).** Tap 4 known court points → full
   plane-to-plane mapping. More accurate, supports landing-zone
   metrics. Not implemented yet; the API accepts an explicit
   meters_per_px so a homography layer can slot in later.

Break measurement (v1, single camera, side view)
------------------------------------------------
We fit a straight line to the first ~30% of the flight path (the
"launch direction") and measure the maximum perpendicular deviation of
the remaining path from that line. In a side view this conflates
gravity drop with aerodynamic break — separating them requires either
a rear-view camera or fitting a ballistic model and measuring residuals.
v1 reports raw deviation and is honest about the limitation in `notes`.
For float-serve coaching the *frame-to-frame wobble* (non-smooth
deviation) is the more diagnostic signal anyway, which we report as
`wobble_px` — RMS of the second derivative of the lateral offset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .spin_estimator import Detection

# Regulation volleyball: 65-67 cm circumference → 20.7-21.3 cm diameter
VOLLEYBALL_RADIUS_M = 0.105

MPH_PER_MS = 2.23694


@dataclass
class TrajectoryResult:
    n_points: int
    speed_px_s: Optional[float] = None      # median flight speed
    peak_speed_px_s: Optional[float] = None
    meters_per_px: Optional[float] = None
    speed_m_s: Optional[float] = None
    speed_mph: Optional[float] = None
    peak_speed_mph: Optional[float] = None
    break_px: Optional[float] = None        # max deviation from launch line
    break_m: Optional[float] = None
    break_in: Optional[float] = None        # inches, coach-friendly
    wobble_px: Optional[float] = None       # RMS jerk of lateral offset
    path: list = field(default_factory=list)   # [(t, x, y), ...] smoothed
    notes: list = field(default_factory=list)


def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average; ends padded by edge replication."""
    if len(arr) < window:
        return arr.astype(float)
    kernel = np.ones(window) / window
    padded = np.concatenate([
        np.full(window // 2, arr[0]),
        arr,
        np.full(window // 2, arr[-1]),
    ])
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def analyze_trajectory(detections: list[Optional[Detection]],
                       fps: float,
                       meters_per_px: Optional[float] = None,
                       ball_radius_m: float = VOLLEYBALL_RADIUS_M
                       ) -> TrajectoryResult:
    """
    Compute velocity + break metrics from a detection sequence.

    If meters_per_px is None, derives it from the ball's detected pixel
    radius (ball-diameter calibration).
    """
    notes: list[str] = []

    # Collect valid (t, x, y, r) samples
    samples = []
    for i, d in enumerate(detections):
        if d is None:
            continue
        samples.append((i / fps, d.cx, d.cy, d.r))

    n = len(samples)
    if n < 8:
        notes.append(f"only {n} detections — too few for trajectory analysis")
        return TrajectoryResult(n_points=n, notes=notes)

    t = np.array([s[0] for s in samples])
    x = _smooth(np.array([s[1] for s in samples]))
    y = _smooth(np.array([s[2] for s in samples]))
    r = np.array([s[3] for s in samples])

    # --- Calibration ------------------------------------------------------
    if meters_per_px is None:
        r_med = float(np.median(r))
        r_cv = float(np.std(r) / (r_med + 1e-9))
        meters_per_px = ball_radius_m / r_med
        notes.append(f"ball-diameter calibration: r_med={r_med:.1f}px → "
                     f"{meters_per_px*1000:.2f} mm/px")
        if r_cv > 0.15:
            notes.append(f"WARN: ball radius varies {r_cv:.0%} across flight — "
                         f"depth change degrades scale accuracy")

    # --- Velocity ---------------------------------------------------------
    # Central-difference gradients on smoothed positions w.r.t. real time
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    speed = np.hypot(vx, vy)            # px/s per sample

    # Trim ends — gradient endpoints are one-sided and noisier
    core = speed[2:-2] if len(speed) > 8 else speed
    speed_med = float(np.median(core))
    speed_peak = float(np.percentile(core, 95))

    speed_m_s = speed_med * meters_per_px
    speed_mph = speed_m_s * MPH_PER_MS
    peak_mph = speed_peak * meters_per_px * MPH_PER_MS

    # --- Break (deviation from launch line) -------------------------------
    # Launch direction: total-least-squares line through first 30% of path
    n_launch = max(4, int(0.3 * n))
    lx, ly = x[:n_launch], y[:n_launch]
    cx0, cy0 = float(np.mean(lx)), float(np.mean(ly))
    # Principal direction via covariance eigenvector
    cov = np.cov(np.stack([lx - cx0, ly - cy0]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]   # unit vector
    normal = np.array([-direction[1], direction[0]])  # perpendicular

    # Signed perpendicular offset of every point from the launch line
    rel = np.stack([x - cx0, y - cy0], axis=1)
    lateral = rel @ normal                            # px, signed

    break_px = float(np.max(np.abs(lateral[n_launch:]))) \
        if n > n_launch else float(np.max(np.abs(lateral)))
    break_m = break_px * meters_per_px
    break_in = break_m * 39.3701

    notes.append("break v1 conflates gravity drop with aerodynamic break "
                 "in side view — see module docstring")

    # --- Wobble (float-serve signature) -----------------------------------
    # Second derivative of lateral offset: a clean ballistic arc has a
    # smooth lateral profile; a knuckling float shows high-frequency
    # direction changes. RMS of the discrete second difference.
    if len(lateral) >= 5:
        jerk = np.diff(lateral, n=2)
        wobble_px = float(np.sqrt(np.mean(jerk ** 2)))
    else:
        wobble_px = None

    path = [(float(ti), float(xi), float(yi)) for ti, xi, yi in zip(t, x, y)]

    return TrajectoryResult(
        n_points=n,
        speed_px_s=speed_med,
        peak_speed_px_s=speed_peak,
        meters_per_px=meters_per_px,
        speed_m_s=speed_m_s,
        speed_mph=speed_mph,
        peak_speed_mph=peak_mph,
        break_px=break_px,
        break_m=break_m,
        break_in=break_in,
        wobble_px=wobble_px,
        path=path,
        notes=notes,
    )
