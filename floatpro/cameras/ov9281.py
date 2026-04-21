"""
Arducam OV9281 on Jetson via GStreamer.

Monochrome global shutter, 1280x800 @ 120fps max, MIPI CSI.
Driver: Arducam Jetvariety (install_full.sh -m ov9281).
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .base import Camera, CameraConfig, CameraInfo


class OV9281Camera(Camera):
    BACKEND = "ov9281_csi"
    MODEL = "Arducam OV9281"

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None

    def _build_pipeline(self) -> str:
        device = self.config.device or "/dev/video0"
        # OV9281 is mono — we request GRAY8 explicitly. videoconvert handles
        # any downstream format tweaks. drop=1 / max-buffers=2 prevents
        # appsink from queuing stale frames if Python falls behind.
        return (
            f"v4l2src device={device} ! "
            f"video/x-raw,format=GRAY8,"
            f"width={self.config.width},height={self.config.height},"
            f"framerate={self.config.fps}/1 ! "
            f"videoconvert ! "
            f"appsink drop=1 max-buffers=2 sync=false"
        )

    def start(self) -> CameraInfo:
        pipeline = self._build_pipeline()
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"OV9281: failed to open pipeline. Check:\n"
                f"  - Arducam driver installed: ./install_full.sh -m ov9281\n"
                f"  - Device exists: ls {self.config.device or '/dev/video0'}\n"
                f"  - OpenCV has GStreamer: "
                f"python3 -c 'import cv2; print(cv2.getBuildInformation())' | grep GStreamer\n"
                f"Pipeline was:\n  {pipeline}"
            )
        return CameraInfo(
            backend=self.BACKEND,
            model=self.MODEL,
            serial="csi0",
            width=self.config.width,
            height=self.config.height,
            fps=float(self.config.fps),
            pixel_format="mono8",
            is_color=False,
            is_global_shutter=True,
        )

    def read(self):
        if self.cap is None:
            return False, None, 0.0
        ok, frame = self.cap.read()
        ts = time.monotonic()
        # OpenCV GStreamer path may up-convert GRAY8 to BGR by default;
        # normalize to single-channel mono if that happens.
        if ok and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return ok, frame, ts

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
