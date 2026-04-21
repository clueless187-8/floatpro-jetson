"""
Synthetic mock camera — no hardware required.

Generates deterministic frames with a moving "ball" so the rest of the
pipeline (ring buffer, save logic, eventually YOLO + spin) can be
exercised on any machine. Especially useful for CI and for hacking on
the dashboard code on a laptop.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .base import Camera, CameraConfig, CameraInfo


class MockCamera(Camera):
    BACKEND = "mock"
    MODEL = "Synthetic 120fps"

    def __init__(self, config: CameraConfig):
        self.config = config
        self._t_start = 0.0
        self._frame_i = 0
        self._running = False

    def start(self) -> CameraInfo:
        self._t_start = time.monotonic()
        self._frame_i = 0
        self._running = True
        return CameraInfo(
            backend=self.BACKEND,
            model=self.MODEL,
            serial="mock0",
            width=self.config.width,
            height=self.config.height,
            fps=float(self.config.fps),
            pixel_format=self.config.pixel_format or "mono8",
            is_color=False,
            is_global_shutter=True,
        )

    def read(self):
        if not self._running:
            return False, None, 0.0

        # Pace ourselves to the target framerate so callers see realistic timing
        target_t = self._t_start + self._frame_i / self.config.fps
        now = time.monotonic()
        if now < target_t:
            time.sleep(target_t - now)

        W, H = self.config.width, self.config.height
        frame = np.full((H, W), 40, dtype=np.uint8)  # dark gym floor

        # Ball trajectory: parabolic toss across the frame
        t = self._frame_i / self.config.fps
        x = int(0.1 * W + (0.8 * W) * (t % 2.0) / 2.0)
        y = int(H * 0.3 + 300 * np.sin(np.pi * (t % 2.0) / 2.0))
        y = max(20, min(H - 20, y))

        # Simulated ball: bright disc with an ASYMMETRIC seam pattern that
        # rotates with the spin. Asymmetry matters because rotationally
        # symmetric patterns (e.g. 3 evenly spaced lines with 120° symmetry)
        # make phase correlation pick the smallest-magnitude rotation
        # equivalent modulo the symmetry period, which creates aliasing.
        R = 30
        cv2.circle(frame, (x, y), R, 255, -1)
        angle_deg = (self._frame_i * 7) % 360  # ground-truth spin rate

        def rot(px, py, angle_d):
            """Rotate point around origin by angle_d degrees."""
            a = np.deg2rad(angle_d)
            return (px * np.cos(a) - py * np.sin(a),
                    px * np.sin(a) + py * np.cos(a))

        # Pattern defined in ball-local coordinates (relative to center, 0°)
        # then rotated by the ball's current spin angle and translated to
        # the ball's screen position. No rotational symmetry anywhere.
        strokes = [
            # One long curved seam from upper-left to lower-right
            [(-R * 0.9, -R * 0.3), (-R * 0.3, -R * 0.2),
             (R * 0.2, R * 0.1), (R * 0.7, R * 0.5)],
            # A short arc on the upper right
            [(R * 0.2, -R * 0.7), (R * 0.5, -R * 0.5), (R * 0.7, -R * 0.2)],
            # A dot on the lower left
        ]
        for stroke in strokes:
            pts = []
            for (sx, sy) in stroke:
                rx, ry = rot(sx, sy, angle_deg)
                pts.append((int(x + rx), int(y + ry)))
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], 50, 3)

        # Asymmetric dot: the key anti-symmetry feature
        dot_x, dot_y = rot(-R * 0.5, R * 0.6, angle_deg)
        cv2.circle(frame, (int(x + dot_x), int(y + dot_y)), 5, 30, -1)

        # Court line for homography testing
        cv2.line(frame, (0, int(H * 0.9)), (W, int(H * 0.9)), 180, 2)

        ts = time.monotonic()
        self._frame_i += 1
        return True, frame, ts

    def stop(self):
        self._running = False
