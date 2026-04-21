"""
Basler ace (Pylon / pypylon).

USB3 industrial. 240-750fps depending on model (acA640-750um is the
classic speed demon). Requires Pylon SDK; pypylon is pip-installable
once the SDK is on the system.

https://www.baslerweb.com/en/downloads/software-downloads/

Install on Jetson:
    1. Download pylon ARM64 .deb from Basler, install
    2. pip install pypylon

Degrades gracefully if pypylon isn't present, same pattern as the FLIR
backend.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base import Camera, CameraConfig, CameraInfo

try:
    from pypylon import pylon  # type: ignore
    _PYLON_OK = True
    _PYLON_ERR = None
except Exception as e:
    _PYLON_OK = False
    _PYLON_ERR = str(e)


class BaslerPylonCamera(Camera):
    BACKEND = "basler_pylon"
    MODEL = "Basler ace"

    def __init__(self, config: CameraConfig):
        if not _PYLON_OK:
            raise RuntimeError(
                f"pypylon not available ({_PYLON_ERR}).\n"
                f"Install Pylon SDK from Basler then: pip install pypylon"
            )
        self.config = config
        self.cam = None
        self.converter = None

    def start(self) -> CameraInfo:
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("Basler: no cameras detected on USB3 bus.")

        # Pick by serial if given
        target = None
        if self.config.device:
            for d in devices:
                if d.GetSerialNumber() == self.config.device:
                    target = d
                    break
            if target is None:
                raise RuntimeError(
                    f"Basler: camera with serial {self.config.device} not found"
                )
        else:
            target = devices[0]

        self.cam = pylon.InstantCamera(tl_factory.CreateDevice(target))
        self.cam.Open()

        # Pixel format
        try:
            if self.config.pixel_format == "mono8":
                self.cam.PixelFormat.SetValue("Mono8")
            else:
                self.cam.PixelFormat.SetValue("BayerRG8")
        except Exception:
            pass

        # Exposure / gain
        try:
            if self.config.exposure_us is not None:
                self.cam.ExposureAuto.SetValue("Off")
                self.cam.ExposureTime.SetValue(float(self.config.exposure_us))
            else:
                self.cam.ExposureAuto.SetValue("Continuous")
        except Exception:
            pass

        try:
            if self.config.gain_db is not None:
                self.cam.GainAuto.SetValue("Off")
                self.cam.Gain.SetValue(float(self.config.gain_db))
            else:
                self.cam.GainAuto.SetValue("Continuous")
        except Exception:
            pass

        # Frame rate (Basler calls it AcquisitionFrameRate on newer, ...Abs on older)
        try:
            self.cam.AcquisitionFrameRateEnable.SetValue(True)
            self.cam.AcquisitionFrameRate.SetValue(float(self.config.fps))
        except Exception:
            try:
                self.cam.AcquisitionFrameRateAbs.SetValue(float(self.config.fps))
            except Exception:
                pass

        # Output conversion: we always hand numpy BGR or Mono8 to downstream
        self.converter = pylon.ImageFormatConverter()
        if self.config.pixel_format == "mono8":
            self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        else:
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        return CameraInfo(
            backend=self.BACKEND,
            model=target.GetModelName(),
            serial=target.GetSerialNumber(),
            width=self.config.width,
            height=self.config.height,
            fps=float(self.config.fps),
            pixel_format=self.config.pixel_format,
            is_color=self.config.pixel_format != "mono8",
            is_global_shutter=True,
        )

    def read(self):
        if self.cam is None or not self.cam.IsGrabbing():
            return False, None, 0.0
        try:
            result = self.cam.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
            ts = time.monotonic()
            if not result.GrabSucceeded():
                result.Release()
                return False, None, ts
            img = self.converter.Convert(result)
            arr = img.GetArray()
            frame = np.array(arr, copy=True)
            result.Release()
            return True, frame, ts
        except Exception:
            return False, None, time.monotonic()

    def stop(self):
        if self.cam is not None:
            try:
                if self.cam.IsGrabbing():
                    self.cam.StopGrabbing()
                self.cam.Close()
            except Exception:
                pass
            self.cam = None
