# FloatPro — Jetson Capture Rig

Phase-1 hardware validation for **Volleyball FloatPro**. This repo proves the
camera pipeline works: Arducam OV9281 global shutter → Jetson Orin Nano Super
→ 120fps ring buffer → save-on-trigger to disk.

No ML, no UI, no spin math yet. Just clean frames at full frame rate with
zero drops. Once this is solid, we layer YOLO ball detection and spin
estimation on top of the same pipeline.

## Hardware

- NVIDIA Jetson Orin Nano Super Developer Kit
- Arducam OV9281 MIPI CSI module (mono global shutter, 1280×800 @ 120fps)
- M12 lens — 6mm for wide scene, 8mm for tighter ball crop
- CSI ribbon cable (included with Arducam module)
- Tripod + ball head, 8ft minimum height
- HDMI display or SSH/Tailscale for headless operation

## Software

- JetPack 6.x (Ubuntu 22.04 base)
- Arducam Jetvariety kernel driver
- OpenCV with GStreamer support (from `apt`, **not** pip)
- Python 3.10+

## Setup

### 1. System packages

```bash
chmod +x setup.sh
./setup.sh
```

### 2. Arducam driver

The Arducam driver is kernel-level and JetPack-specific, so we don't pin a URL
here — it goes stale. Follow the current instructions at:

  <https://docs.arducam.com/Nvidia-Jetson-Camera/Jetvariety-Camera/Quick-Start-Guide/>

The short version is usually:

```bash
wget -O install_full.sh https://github.com/ArduCAM/MIPI_Camera/releases/latest/download/install_full.sh
chmod +x install_full.sh
./install_full.sh -m ov9281
sudo reboot
```

### 3. Connect the camera

Power the Jetson down before plugging/unplugging CSI cables. Use CSI port 0
(the one closest to the power jack on Orin Nano).

### 4. Verify

```bash
python3 check_camera.py
```

Expected: five PASS checks. If any fail, the script tells you what to fix.

## Running capture

```bash
python3 capture.py
```

A preview window opens showing a live HUD with measured FPS, buffer fill, and
drop count. The buffer holds the **last 5 seconds** of frames continuously.

- `SPACE` — dump the last 5s to `captures/YYYYMMDD_HHMMSS_mmm/`
- `q` — quit

Each saved session contains:

```
captures/20260419_143022_817/
├── frame_00000.png          # 1280x800 grayscale PNG, one per frame
├── frame_00001.png
├── ...
└── metadata.json            # per-frame timestamps + timing diagnostics
```

## Performance notes

- **RAM budget**: 5s × 120fps × 1280 × 800 = ~614 MB. Orin Nano Super's 8 GB
  handles this with plenty of headroom.
- **Disk**: 600 PNGs at ~400 KB each = ~240 MB per session. Save to an NVMe
  SSD; an SD card will bottleneck.
- **Display**: preview is scaled to 70% so the GUI doesn't starve the capture
  thread. Full-res frames still land in the ring buffer.
- **Power**: run `sudo nvpmodel -m 0 && sudo jetson_clocks` before testing. At
  default power mode the Jetson downclocks and you'll see interval jitter.

## Interpreting results

After a save, `metadata.json` reports `effective_fps`, `avg_interval_ms`,
`min_interval_ms`, `max_interval_ms`.

Healthy run at 120fps:
- avg ≈ 8.33ms
- stdev < 0.5ms
- max < 12ms

If max > 20ms you're dropping frames — investigate before moving to the ML
stage. Spin estimation is sensitive to missing frames because it assumes
uniform sampling.

## Repo layout

```
floatpro-jetson/
├── README.md
├── setup.sh            # apt deps + Jetson perf mode
├── check_camera.py     # smoke test: driver, fps, frame content
├── capture.py          # main ring-buffer capture + trigger save
└── captures/           # session outputs (gitignored)
```

## What's next (phase 2)

Once the pipeline is validated:

1. Label ~500 frames of volleyball footage → YOLOv8n training
2. Export to TensorRT for Jetson acceleration
3. Run YOLO on each frame in the ring buffer, save bounding boxes to metadata
4. Build the spin estimator (log-polar + phase correlation on ball crops)
5. Calibration flow: 4-point court homography for velocity + landing

All of that slots into the same ring-buffer architecture. The save step
becomes: "save frames + detections + spin + velocity" instead of just frames.
