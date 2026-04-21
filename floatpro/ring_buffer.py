"""
Threaded ring buffer that drains any Camera backend into a deque.

Separated from the UI so it can be reused by headless capture, unit
tests, and the future ML pipeline.
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Optional

import numpy as np

from .cameras.base import Camera


class RingBuffer:
    def __init__(self, camera: Camera, capacity_frames: int):
        self.camera = camera
        self.capacity = capacity_frames
        self.buffer: collections.deque = collections.deque(maxlen=capacity_frames)
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.running = False

        self.info = None
        self.frame_count = 0
        self.drop_count = 0
        self.fps_actual = 0.0

    # --- Lifecycle ----------------------------------------------------------
    def start(self):
        self.info = self.camera.start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.camera.stop()

    # --- Capture loop -------------------------------------------------------
    def _loop(self):
        last_fps_t = time.monotonic()
        fps_frames = 0
        while self.running:
            ok, frame, ts = self.camera.read()
            if not ok:
                self.drop_count += 1
                time.sleep(0.001)
                continue
            with self.lock:
                self.buffer.append((ts, frame))
                self.frame_count += 1
            fps_frames += 1

            now = time.monotonic()
            if now - last_fps_t >= 1.0:
                self.fps_actual = fps_frames / (now - last_fps_t)
                fps_frames = 0
                last_fps_t = now

    # --- Accessors ----------------------------------------------------------
    def snapshot(self) -> list[tuple[float, np.ndarray]]:
        """Return a shallow copy of the buffer — safe to iterate without
        holding the lock."""
        with self.lock:
            return list(self.buffer)

    def latest(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.buffer[-1][1] if self.buffer else None

    @property
    def buffer_len(self) -> int:
        with self.lock:
            return len(self.buffer)
