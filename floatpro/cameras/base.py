"""
Camera abstraction layer.

Every capture backend implements the same interface, so downstream code
(ring buffer, YOLO inference, spin estimation, UI) doesn't care which
sensor is attached.

Frame format contract
---------------------
All backends yield numpy arrays shaped (H, W, C) where C is 1 (mono) or
3 (BGR, OpenCV convention). dtype is always uint8 after any pre-
processing. Downstream code that needs mono will convert if necessary;
the spin estimator in particular wants mono regardless of sensor.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CameraConfig:
    """Target capture settings. A backend may coerce these to the nearest
    supported mode and will log what it actually selected."""
    width: int = 1280
    height: int = 800
    fps: int = 120
    pixel_format: str = "auto"     # "mono8", "bgr8", or "auto" (backend picks)
    exposure_us: Optional[int] = None   # None = auto-exposure
    gain_db: Optional[float] = None     # None = auto-gain
    device: str = ""               # /dev/video0, serial number, etc. — backend-specific
    extra: dict = field(default_factory=dict)   # backend-specific knobs


@dataclass
class CameraInfo:
    """What the backend actually ended up with after start()."""
    backend: str
    model: str
    serial: str
    width: int
    height: int
    fps: float
    pixel_format: str
    is_color: bool
    is_global_shutter: bool


class Camera(abc.ABC):
    """Abstract camera backend."""

    @abc.abstractmethod
    def start(self) -> CameraInfo:
        """Open the device and begin streaming. Returns the negotiated config."""

    @abc.abstractmethod
    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        """
        Fetch the next frame.

        Returns
        -------
        ok : bool
            True if a frame was obtained.
        frame : np.ndarray | None
            Shape (H, W) for mono, (H, W, 3) for color. dtype uint8.
        timestamp : float
            Capture timestamp in seconds (time.monotonic() or equivalent
            host-side stamp — not sensor hardware timestamp unless the
            backend explicitly supports it).
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop streaming and release the device."""

    # Optional capabilities ---------------------------------------------------

    def set_exposure(self, exposure_us: int) -> bool:
        """Set exposure in microseconds. Return True if applied."""
        return False

    def set_gain(self, gain_db: float) -> bool:
        """Set analog gain in dB. Return True if applied."""
        return False

    def trigger_software(self) -> bool:
        """Fire a software trigger (only meaningful on industrial cams in
        triggered mode)."""
        return False
