"""
Arducam AR0234 on Jetson via GStreamer.

Color global shutter, 1920x1200 @ 120fps max, MIPI CSI.
Driver: Arducam Jetvariety (install_full.sh -m ar0234).

Same GStreamer pattern as OV9281 but with Bayer demosaic. We request
the raw Bayer format from v4l2src and let bayer2rgb handle demosaic on
the GPU via nvvidconv when available.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .base import Camera, CameraConfig, CameraInfo


class AR0234Camera(Camera):
    BACKEND = "ar0234_csi"
    MODEL = "Arducam AR0234"

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_mono_output = False

    def _build_pipeline(self) -> str:
        device = self.config.device or "/dev/video0"
        # AR0234 delivers color by default. If caller explicitly wanted mono,
        # we still capture color then convert in read() — cheaper than
        # reconfiguring the sensor for binning.
        self._is_mono_output = self.config.pixel_format == "mono8"

        # nvarguscamerasrc would be ideal (GPU ISP) but requires the Arducam
        # TrueSight variant. The generic driver uses v4l2src + software
        # videoconvert, which still holds 120fps comfortably on Orin Nano.
        return (
            f"v4l2src device={device} ! "
            f"video/x-raw,format=UYVY,"
            f"width={self.config.width},height={self.config.height},"
            f"framerate={self.config.fps}/1 ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=1 max-buffers=2 sync=false"
        )

    def start(self) -> CameraInfo:
        pipeline = self._build_pipeline()
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"AR0234: failed to open pipeline. Check:\n"
                f"  - Arducam driver installed: ./install_full.sh -m ar0234\n"
                f"  - Device exists: ls {self.config.device or '/dev/video0'}\n"
                f"  - Pipeline format may need adjustment for your driver\n"
                f"    variant (UYVY vs bayer_rggb8) — run v4l2-ctl\n"
                f"    --list-formats-ext to see what the sensor reports.\n"
                f"Pipeline was:\n  {pipeline}"
            )
        return CameraInfo(
            backend=self.BACKEND,
            model=self.MODEL,
            serial="csi0",
            width=self.config.width,
            height=self.config.height,
            fps=float(self.config.fps),
            pixel_format="mono8" if self._is_mono_output else "bgr8",
            is_color=True,
            is_global_shutter=True,
        )

    def read(self):
        if self.cap is None:
            return False, None, 0.0
        ok, frame = self.cap.read()
        ts = time.monotonic()
        if ok and frame is not None and self._is_mono_output and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return ok, frame, ts

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
