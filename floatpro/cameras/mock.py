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

        # Draw "ball" with a rotating pattern so spin-detection tests
        # have something to latch onto
        cv2.circle(frame, (x, y), 18, 255, -1)
        angle = (self._frame_i * 7) % 360  # simulated spin
        for i in range(3):
            rad = np.deg2rad(angle + i * 120)
            px = int(x + 12 * np.cos(rad))
            py = int(y + 12 * np.sin(rad))
            cv2.circle(frame, (px, py), 3, 100, -1)

        # Court line for homography testing
        cv2.line(frame, (0, int(H * 0.9)), (W, int(H * 0.9)), 180, 2)

        ts = time.monotonic()
        self._frame_i += 1
        return True, frame, ts

    def stop(self):
        self._running = False
