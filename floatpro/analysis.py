"""
Session analysis orchestrator.

Combines spin estimation + trajectory analysis into one JSON-able
report, and computes the Knuckle Index — FloatPro's composite
float-serve quality score.

Knuckle Index (v1 heuristic)
----------------------------
A great float serve is FAST, LOW-SPIN, and MOVES. KI multiplies three
normalized factors and scales to 0-100:

    speed_norm = min(speed_mph / 50, 1.0)      # 50+ mph saturates
    spin_norm  = max(0, 1 - rpm / 120)         # 0 RPM = 1.0, 120+ RPM = 0
    move_norm  = min(wobble_px / 3.0, 1.0)     # knuckle wobble signature

    KI = 100 * (speed_norm * spin_norm * move_norm) ** (1/3)

Geometric mean keeps any single weak factor from being masked by the
other two. This is a v1 heuristic — expect to re-tune the constants
once we have a corpus of real serves graded by a coach's eye. All three
inputs and the formula version are included in the output so historical
scores can be recomputed when the formula evolves.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np

from .spin_estimator import estimate_session_spin, SpinResult
from .trajectory import analyze_trajectory, TrajectoryResult

KI_FORMULA_VERSION = 1


def knuckle_index(speed_mph: Optional[float],
                  rpm: Optional[float],
                  wobble_px: Optional[float]) -> Optional[float]:
    """Composite float quality score, 0-100. None if inputs missing."""
    if speed_mph is None or rpm is None or wobble_px is None:
        return None
    speed_norm = min(max(speed_mph, 0.0) / 50.0, 1.0)
    spin_norm = max(0.0, 1.0 - rpm / 120.0)
    move_norm = min(max(wobble_px, 0.0) / 3.0, 1.0)
    product = speed_norm * spin_norm * move_norm
    if product <= 0:
        return 0.0
    return float(100.0 * product ** (1.0 / 3.0))


def classify_serve(rpm: Optional[float],
                   speed_mph: Optional[float]) -> Optional[str]:
    """Rough serve-type classification from spin + speed. v1 heuristic."""
    if rpm is None:
        return None
    if rpm < 60:
        return "float"
    if rpm < 180:
        return "jump_float" if (speed_mph or 0) > 35 else "hybrid"
    return "topspin"


def analyze_session(frames: list[np.ndarray],
                    fps: float,
                    meters_per_px: Optional[float] = None) -> dict:
    """
    Full analysis: spin + trajectory + composite metrics.

    Returns a JSON-serializable dict. This is the single entry point the
    server, the CLI ingester, and future batch tools all call — keep it
    the only place that knows how the sub-analyzers compose.
    """
    spin: SpinResult = estimate_session_spin(frames, fps=fps)
    traj: TrajectoryResult = analyze_trajectory(
        spin.detections, fps=fps, meters_per_px=meters_per_px
    )

    ki = knuckle_index(traj.speed_mph, spin.rpm, traj.wobble_px)
    serve_type = classify_serve(spin.rpm, traj.speed_mph)

    payload = {
        "analysis_version": 2,
        "fps": fps,
        # Headline metrics
        "rpm": spin.rpm,
        "rpm_std": spin.rpm_std,
        "spin_direction": spin.direction,
        "speed_mph": traj.speed_mph,
        "peak_speed_mph": traj.peak_speed_mph,
        "speed_m_s": traj.speed_m_s,
        "break_in": traj.break_in,
        "break_m": traj.break_m,
        "wobble_px": traj.wobble_px,
        "knuckle_index": ki,
        "ki_formula_version": KI_FORMULA_VERSION,
        "serve_type": serve_type,
        # Sub-reports
        "spin": {k: v for k, v in asdict(spin).items() if k != "detections"},
        "trajectory": {k: v for k, v in asdict(traj).items() if k != "path"},
        "path": traj.path,
        # Full detections for frontend overlays
        "detections": [asdict(d) if d else None for d in spin.detections],
        "notes": spin.notes + traj.notes,
    }
    return payload
