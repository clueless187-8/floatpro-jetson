"""
FLIR Blackfly S (Spinnaker / PySpin).

USB3 industrial. 240fps+ depending on model (BFS-U3-04S2M-CS is the
classic high-speed pick for ball-tracking). Requires Spinnaker SDK and
the PySpin Python bindings to be installed separately — they're NOT
pip-installable; you download them from FLIR's site after registering.

https://www.flir.com/products/spinnaker-sdk/

Install on Jetson: use the ARM64 Ubuntu 20.04 or 22.04 build, run
install_spinnaker_arm.sh, then pip install the matching PySpin .whl
from the Spinnaker download bundle.

This backend degrades gracefully: if PySpin isn't importable, the
class still loads but start() raises with an actionable error. That
lets the rest of the code import the module on any Jetson without
crashing at module-load time.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base import Camera, CameraConfig, CameraInfo

try:
    import PySpin  # type: ignore
    _PYSPIN_OK = True
    _PYSPIN_ERR = None
except Exception as e:  # ImportError, or SDK load failure
    _PYSPIN_OK = False
    _PYSPIN_ERR = str(e)


class FlirSpinnakerCamera(Camera):
    BACKEND = "flir_spinnaker"
    MODEL = "FLIR Blackfly S"

    def __init__(self, config: CameraConfig):
        if not _PYSPIN_OK:
            raise RuntimeError(
                f"PySpin not available ({_PYSPIN_ERR}).\n"
                f"Install the Spinnaker SDK + PySpin wheel from "
                f"https://www.flir.com/products/spinnaker-sdk/"
            )
        self.config = config
        self.system = None
        self.cam_list = None
        self.cam = None
        self.nodemap = None

    def start(self) -> CameraInfo:
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() == 0:
            self._cleanup_system()
            raise RuntimeError("FLIR: no cameras detected on USB3 bus.")

        # Pick by serial if given, else take the first enumerated camera
        if self.config.device:
            try:
                self.cam = self.cam_list.GetBySerial(self.config.device)
            except Exception as e:
                self._cleanup_system()
                raise RuntimeError(
                    f"FLIR: camera with serial {self.config.device} not found: {e}"
                )
        else:
            self.cam = self.cam_list.GetByIndex(0)

        self.cam.Init()
        self.nodemap = self.cam.GetNodeMap()

        serial = self.cam.TLDevice.DeviceSerialNumber.GetValue()
        model = self.cam.TLDevice.DeviceModelName.GetValue()

        # Continuous acquisition with configurable frame rate
        self._set_enum("AcquisitionMode", "Continuous")
        self._set_enum("ExposureAuto", "Off" if self.config.exposure_us else "Continuous")
        if self.config.exposure_us:
            self._set_float("ExposureTime", float(self.config.exposure_us))

        self._set_enum("GainAuto", "Off" if self.config.gain_db is not None else "Continuous")
        if self.config.gain_db is not None:
            self._set_float("Gain", float(self.config.gain_db))

        # Enable manual frame rate control. The sensor will coerce to the
        # nearest achievable rate given the exposure + ROI.
        try:
            self._set_bool("AcquisitionFrameRateEnable", True)
            self._set_float("AcquisitionFrameRate", float(self.config.fps))
        except Exception:
            pass  # Some models expose the node with a different name

        # Choose mono vs color pixel format
        try:
            self._set_enum("PixelFormat",
                          "Mono8" if self.config.pixel_format == "mono8" else "BayerRG8")
        except Exception:
            pass

        self.cam.BeginAcquisition()

        return CameraInfo(
            backend=self.BACKEND,
            model=model,
            serial=serial,
            width=self.config.width,
            height=self.config.height,
            fps=float(self.config.fps),
            pixel_format=self.config.pixel_format,
            is_color=self.config.pixel_format != "mono8",
            is_global_shutter=True,
        )

    def read(self):
        if self.cam is None:
            return False, None, 0.0
        try:
            img = self.cam.GetNextImage(100)  # 100ms timeout
            if img.IsIncomplete():
                img.Release()
                return False, None, time.monotonic()
            arr = img.GetNDArray()
            ts = time.monotonic()
            # GetNDArray gives us the right shape already; copy so the
            # underlying buffer can be released back to the SDK.
            frame = np.array(arr, copy=True)
            img.Release()
            return True, frame, ts
        except Exception:
            return False, None, time.monotonic()

    def stop(self):
        try:
            if self.cam is not None:
                try:
                    self.cam.EndAcquisition()
                except Exception:
                    pass
                self.cam.DeInit()
                del self.cam
                self.cam = None
        finally:
            self._cleanup_system()

    def _cleanup_system(self):
        if self.cam_list is not None:
            self.cam_list.Clear()
            self.cam_list = None
        if self.system is not None:
            self.system.ReleaseInstance()
            self.system = None

    # --- Node helpers -------------------------------------------------------
    def _set_enum(self, name, value):
        node = PySpin.CEnumerationPtr(self.nodemap.GetNode(name))
        entry = node.GetEntryByName(value)
        node.SetIntValue(entry.GetValue())

    def _set_float(self, name, value):
        node = PySpin.CFloatPtr(self.nodemap.GetNode(name))
        node.SetValue(value)

    def _set_bool(self, name, value):
        node = PySpin.CBooleanPtr(self.nodemap.GetNode(name))
        node.SetValue(value)
