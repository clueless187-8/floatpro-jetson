"""
Camera factory.

    from floatpro.cameras import make_camera, CameraConfig
    cam = make_camera("ov9281", CameraConfig(fps=120))
    info = cam.start()

Supported backend names (case-insensitive):
    ov9281       Arducam OV9281 mono global shutter (CSI)
    ar0234       Arducam AR0234 color global shutter (CSI)
    flir         FLIR Blackfly S via Spinnaker/PySpin (USB3)
    basler       Basler ace via Pylon/pypylon (USB3)
    mock         Synthetic camera for testing without hardware
"""
from __future__ import annotations

from .base import Camera, CameraConfig, CameraInfo
from .ov9281 import OV9281Camera
from .ar0234 import AR0234Camera
from .mock import MockCamera

# Industrial backends have heavy SDK deps — guard their imports so the
# rest of the package works on a machine without Spinnaker / Pylon.
try:
    from .flir_spinnaker import FlirSpinnakerCamera
    _FLIR_AVAILABLE = True
except Exception:
    FlirSpinnakerCamera = None  # type: ignore
    _FLIR_AVAILABLE = False

try:
    from .basler_pylon import BaslerPylonCamera
    _BASLER_AVAILABLE = True
except Exception:
    BaslerPylonCamera = None  # type: ignore
    _BASLER_AVAILABLE = False


# Recommended defaults per camera — capture.py falls back to these when
# the user doesn't pin resolution/fps explicitly. Each represents the
# practical sweet spot for volleyball serve analysis on that sensor.
PRESETS = {
    "ov9281": CameraConfig(width=1280, height=800,  fps=120),
    "ar0234": CameraConfig(width=1920, height=1200, fps=120),
    "flir":   CameraConfig(width=720,  height=540,  fps=240, pixel_format="mono8"),
    "basler": CameraConfig(width=720,  height=540,  fps=240, pixel_format="mono8"),
    "mock":   CameraConfig(width=1280, height=800,  fps=120),
}


def make_camera(name: str, config: CameraConfig | None = None) -> Camera:
    """Construct a camera backend by short name."""
    key = name.lower().strip()
    if config is None:
        if key not in PRESETS:
            raise ValueError(
                f"No preset for '{name}'. Supply an explicit CameraConfig."
            )
        config = PRESETS[key]

    if key == "ov9281":
        return OV9281Camera(config)
    if key == "ar0234":
        return AR0234Camera(config)
    if key == "flir":
        if not _FLIR_AVAILABLE:
            raise RuntimeError(
                "FLIR backend requires Spinnaker SDK + PySpin. See "
                "floatpro/cameras/flir_spinnaker.py for install notes."
            )
        return FlirSpinnakerCamera(config)
    if key == "basler":
        if not _BASLER_AVAILABLE:
            raise RuntimeError(
                "Basler backend requires Pylon SDK + pypylon. See "
                "floatpro/cameras/basler_pylon.py for install notes."
            )
        return BaslerPylonCamera(config)
    if key == "mock":
        return MockCamera(config)

    raise ValueError(
        f"Unknown camera backend '{name}'. "
        f"Supported: ov9281, ar0234, flir, basler, mock"
    )


def _probe_sdk(module_name: str) -> bool:
    """Check whether a vendor SDK is actually importable right now.
    The backend files import SDKs inside try/except, so module-level
    import success doesn't imply SDK presence — we probe directly."""
    import importlib
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def available_backends() -> dict[str, bool]:
    """Report which backends can actually be instantiated right now."""
    return {
        "ov9281": True,       # CSI/OpenCV — always importable
        "ar0234": True,
        "flir":   _FLIR_AVAILABLE and _probe_sdk("PySpin"),
        "basler": _BASLER_AVAILABLE and _probe_sdk("pypylon"),
        "mock":   True,
    }


__all__ = [
    "Camera",
    "CameraConfig",
    "CameraInfo",
    "make_camera",
    "available_backends",
    "PRESETS",
]
