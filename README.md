# FloatPro — Jetson Capture Rig

Phase-1 hardware validation for **Volleyball FloatPro**. Captures clean
high-framerate video on a Jetson Orin Nano Super with a backend-agnostic
pipeline that supports four cameras today and is trivial to extend to
more.

No ML, no UI, no spin math yet. Just ring-buffered capture at full frame
rate with zero drops. Once this is solid across the camera options, we
layer YOLO ball detection and spin estimation on top of the same
pipeline.

## Supported cameras

| Backend  | Sensor                       | Max FPS (preset) | Interface | Shutter    | Color  | Notes                                |
|----------|------------------------------|------------------|-----------|------------|--------|--------------------------------------|
| `ov9281` | Arducam OV9281               | 120 @ 1280×800   | CSI       | Global     | Mono   | Best price/pixel, mono-only          |
| `ar0234` | Arducam AR0234               | 120 @ 1920×1200  | CSI       | Global     | Color  | Production sweet spot                |
| `flir`   | FLIR Blackfly S              | 240+ @ 720×540   | USB3      | Global     | Either | Industrial, hardware trigger         |
| `basler` | Basler ace                   | 240+ @ 720×540   | USB3      | Global     | Either | Industrial, highest FPS ceiling      |
| `mock`   | Synthetic                    | 120 @ 1280×800   | —         | N/A        | Mono   | No hardware, for dev/CI              |

The presets in `floatpro/cameras/__init__.py` pick the sweet-spot mode
for each sensor. Override at the command line with `--width / --height
/ --fps / --pixel-format`.

## Hardware

- NVIDIA Jetson Orin Nano Super Developer Kit
- One of the supported cameras above
- M12 lens for CSI cameras — 6mm wide, 8mm for tighter ball crop
- CSI ribbon cable (included with Arducam modules)
- Tripod + ball head, 8ft minimum height
- HDMI display, or SSH/Tailscale for headless operation

## Software prerequisites

- JetPack 6.x (Ubuntu 22.04 base)
- OpenCV with GStreamer support — from `apt`, **not** pip (see below)
- Python 3.10+
- Camera-specific driver or SDK (see each backend's section below)

## Installation

### 1. System packages

```bash
chmod +x setup.sh
./setup.sh
```

This installs v4l-utils, GStreamer plugins, and `python3-opencv`. It
also bumps the Jetson to max performance mode so fps tests are
meaningful.

### 2. Install the package

```bash
pip install -e .
```

No hard runtime deps — OpenCV and numpy come from the system `apt`
install deliberately. On a dev laptop without apt, `pip install
opencv-python numpy` separately.

### 3. Install the driver/SDK for your camera

#### OV9281 / AR0234 (Arducam CSI)

```bash
wget -O install_full.sh https://github.com/ArduCAM/MIPI_Camera/releases/latest/download/install_full.sh
chmod +x install_full.sh
./install_full.sh -m ov9281      # or -m ar0234
sudo reboot
```

Power the Jetson down before plugging/unplugging CSI cables. CSI is not
hot-pluggable. Use CSI port 0 (closest to the power jack).

#### FLIR Blackfly S

1. Register at <https://www.flir.com/products/spinnaker-sdk/> and
   download the ARM64 Ubuntu 22.04 build.
2. Run the included `install_spinnaker_arm.sh`.
3. Install the matching PySpin wheel from the same bundle:
   `pip install spinnaker_python-*-linux_aarch64.whl`

#### Basler ace

1. Download Pylon ARM64 `.deb` from
   <https://www.baslerweb.com/en/downloads/software-downloads/>
2. `sudo dpkg -i pylon_*.deb`
3. `pip install pypylon` (or `pip install -e '.[basler]'`)

#### Mock

No install required. `--backend mock` is always available.

## Verify

```bash
python3 check_camera.py --backend ov9281
```

Expected: five PASS checks. Script probes SDK availability, v4l2
devices (CSI backends), OpenCV GStreamer support (CSI backends),
actual framerate over 300 frames, and frame content sanity.

Check which backends are installable right now:

```bash
python3 capture.py --list-backends
```

## Running capture

```bash
# CSI cameras with preset resolution/fps
python3 capture.py --backend ov9281
python3 capture.py --backend ar0234

# Industrial cameras with explicit exposure/gain control
python3 capture.py --backend flir   --exposure 500 --gain 6
python3 capture.py --backend basler --fps 240 --width 720 --height 540

# Headless over SSH (no preview window)
python3 capture.py --backend ov9281 --no-preview

# Smoke-test the pipeline with no hardware
python3 capture.py --backend mock
```

A preview window shows a live HUD with measured FPS, buffer fill, and
drop count. The ring buffer continuously holds the last 5 seconds.

- `SPACE` — dump the last 5s to `captures/YYYYMMDD_HHMMSS_mmm/`
- `q` — quit

Each saved session contains:

```
captures/20260419_143022_817/
├── frame_00000.png          # one PNG per frame
├── frame_00001.png
├── ...
└── metadata.json            # camera info, per-frame timestamps, timing stats
```

## Architecture

```
┌─────────────────────────────────────┐
│  capture.py  (CLI + HUD)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  floatpro.ring_buffer.RingBuffer    │
│  - threaded capture loop            │
│  - deque snapshot + save hand-off   │
└──────────────┬──────────────────────┘
               │  reads from
               ▼
┌─────────────────────────────────────┐
│  floatpro.cameras.Camera (ABC)      │
│  start() → read() → stop()          │
└───┬───────┬───────┬───────┬─────────┘
    │       │       │       │
    ▼       ▼       ▼       ▼
 OV9281  AR0234   FLIR    Basler    Mock
 (CSI)   (CSI)   (USB3)   (USB3)
```

The abstraction means downstream code — ring buffer, future YOLO
inference, spin estimator, eventually the React Native app talking to
the Jetson over Tailscale — never knows or cares which camera is
attached. Swap the sensor, keep the whole stack.

## Performance notes

- **RAM budget**: 5s × 120fps × 1920 × 1200 × 3 (AR0234 color) = ~4.1 GB.
  Watch RAM on the Orin Nano Super's 8 GB — reduce `--buffer` to 3
  seconds or drop to mono if you see swapping.
- **Disk**: save destination matters. Use an NVMe SSD; SD cards
  bottleneck below 100fps sustained write.
- **Power mode**: run `sudo nvpmodel -m 0 && sudo jetson_clocks` or the
  Jetson will downclock and you'll see interval jitter. `setup.sh` does
  this for you but only until reboot.
- **Don't pip install opencv-python on Jetson** — those wheels are
  built without GStreamer on ARM64 and CSI capture will silently fail.
  Use `apt install python3-opencv`.

## Interpreting results

After a save, `metadata.json` reports `effective_fps`, min/avg/max
interval in ms.

Healthy run at 120fps:
- avg ≈ 8.33ms
- stdev < 0.5ms
- max < 12ms

Healthy run at 240fps:
- avg ≈ 4.17ms
- stdev < 0.3ms
- max < 6ms

If max > 2× the target interval, you're dropping frames. Fix before
moving to the ML stage — spin estimation assumes uniform sampling.

## Repo layout

```
floatpro-jetson/
├── README.md
├── pyproject.toml          # pip install -e .  (extras: [server], [basler])
├── setup.sh                # apt deps + Jetson perf mode
├── check_camera.py         # backend-aware smoke test
├── capture.py              # CLI app: ring buffer + SPACE to save
├── ingest_video.py         # import MP4/MOV → captures/ layout
├── floatpro/
│   ├── __init__.py
│   ├── ring_buffer.py      # threaded ring buffer, camera-agnostic
│   ├── spin_estimator.py   # ORB + RANSAC primary, log-polar fallback
│   ├── server.py           # FastAPI: sessions, analyze, frames, dashboard
│   ├── static/
│   │   └── dashboard.html  # single-file browser UI
│   └── cameras/
│       ├── __init__.py     # factory: make_camera() + available_backends()
│       ├── base.py         # Camera ABC, CameraConfig, CameraInfo
│       ├── ov9281.py       # Arducam OV9281 via GStreamer
│       ├── ar0234.py       # Arducam AR0234 via GStreamer
│       ├── flir_spinnaker.py   # FLIR Blackfly S via PySpin
│       ├── basler_pylon.py     # Basler ace via pypylon
│       └── mock.py         # synthetic, no hardware
├── cloudflare/
│   ├── README.md           # tunnel setup walkthrough
│   ├── config.yml.template # cloudflared ingress config
│   └── floatpro-server.service  # systemd unit for the API server
├── tests/
│   ├── test_spin_estimator.py   # validates against mock ground truth
│   ├── test_server.py           # end-to-end API integration test
│   └── debug_*.py               # diagnostic scripts
└── captures/               # session outputs (gitignored)
```

## Running the server

```bash
pip install -e '.[server]'
python3 -m floatpro.server --port 8080
# open http://localhost:8080 in a browser
```

The dashboard lists captured sessions, runs spin estimation on demand,
and lets you scrub through annotated frames. Wire a Cloudflare Tunnel in
front of it (see below) to make it reachable from anywhere.

## Ingesting phone / camera video

`ingest_video.py` converts any MP4/MOV/AVI into the same `captures/`
layout the rest of the pipeline consumes. This is the fastest path to
validate the spin estimator on real-world footage before committing to
hardware.

```bash
# Quick-look: probe the video + sample 5 frames for detection
python3 ingest_video.py serve.mp4 --preview

# Ingest the whole clip
python3 ingest_video.py serve.mp4

# Trim + downscale 4K → 1080p + analyze in one pass
python3 ingest_video.py serve.mp4 --start 2.5 --end 5.0 --scale 0.5 --analyze

# Override the FPS when phone metadata lies (common on slow-mo modes)
python3 ingest_video.py serve.mp4 --fps 240
```

The ingested session appears in the dashboard at `http://localhost:8080`
like any other capture. You can scrub frames, run analysis, and fetch
results through the same API. Metadata carries a `source.type =
video_file` tag so downstream tools can tell it apart from live
hardware captures.

Tips for phone footage:

- **Side view** gives velocity + contact height in one frame
- **Landscape orientation**, 1080p minimum, 240fps if the phone supports it
- **Bright ball, dark backdrop** — a gym floor works; a bleacher crowd does not
- **Lock exposure** if your phone allows it — auto-exposure mid-flight smears the ball
- **Short clips (3–5 s)** keep extracted frame count manageable; trim with `--start` / `--end`

Routes:

```
GET  /                                          # dashboard HTML
GET  /health                                    # liveness check
GET  /api/status                                # service info
GET  /api/sessions                              # list all captures
GET  /api/sessions/{id}                         # session metadata
POST /api/sessions/{id}/analyze                 # run spin estimation
GET  /api/sessions/{id}/result                  # cached analysis
GET  /api/sessions/{id}/frames/{n}              # raw PNG
GET  /api/sessions/{id}/frames/{n}/annotated    # PNG with overlay
```

## Serve metrics (analysis v2)

`POST /analyze` (or `ingest_video.py --analyze`) runs the full pipeline
in `floatpro/analysis.py` and reports:

| Metric | How it's computed | Notes |
|---|---|---|
| **Spin (RPM)** | ORB feature matching + RANSAC rotation between consecutive ball patches, median + MAD aggregation | Validated 4.9–7.4% error vs mock ground truth |
| **Speed (mph)** | Smoothed central-difference velocity, ball-diameter calibration (volleyball = 21 cm → meters/px from detected radius) | 0.0% error vs mock; no court points needed |
| **Break (in)** | Max perpendicular deviation from the launch line (TLS fit on first 30% of flight) | v1 limitation: side view conflates gravity drop with aerodynamic break |
| **Wobble (px)** | RMS second-difference of lateral offset | The knuckling signature — high-frequency direction changes a ballistic arc doesn't have |
| **Knuckle Index** | `100·(speed_norm·spin_norm·move_norm)^(1/3)` — geometric mean of speed (sat. 50 mph), low-spin (0 at 120 RPM), and wobble (sat. 3 px) | 0–100 composite float quality; formula versioned in payload for re-scoring later |
| **Serve type** | rpm < 60 → float · < 180 → jump_float/hybrid by speed · ≥ 180 → topspin | v1 heuristic |

Calibration accuracy degrades if the ball moves significantly
toward/away from the camera (radius CV > 15% triggers a warning note).
Court-homography calibration is the phase-2 fix.

## Windows quickstart (laptop analysis, no Jetson)

The analysis pipeline runs anywhere Python does — useful for analyzing
phone clips on a laptop before the camera hardware arrives.

```powershell
git clone https://github.com/clueless187-8/floatpro-jetson.git
cd floatpro-jetson          # ← must run everything from the repo root

python -m pip install opencv-python numpy fastapi "uvicorn[standard]" httpx
python -m pip install -e .

# Sanity check
python -c "import cv2; print(cv2.__version__)"

# Analyze a phone clip (drag the file into the terminal to paste its path)
python ingest_video.py "C:\path\to\serve.mp4" --preview
python ingest_video.py "C:\path\to\serve.mp4" --start 1.0 --end 4.0 --analyze

# Dashboard
python -m floatpro.server --port 8080   # → http://localhost:8080
```

Windows gotchas:
- Use `python`, not `python3` (PowerShell's `python3` may open the
  Microsoft Store).
- Run from the repo root — `pip install -e .` and the scripts both
  expect it.
- `pyproject.toml` deliberately does **not** depend on OpenCV (Jetson
  must use the apt build for GStreamer), so install `opencv-python`
  explicitly as above.

## What's next (phase 2)

Once the pipeline is validated on at least one real camera:

1. Label ~500 frames of volleyball footage → YOLOv8n ball detector
   (replaces `detect_ball_simple`) — training harness scaffolded in
   `ml/`, see `ml/README.md`
2. Export to TensorRT for Jetson acceleration
3. Court homography calibration — 4-point tap → landing zones +
   depth-robust velocity (ball-diameter calibration already gives
   velocity today)
4. Rear-view or dual-camera capture to separate aerodynamic break from
   gravity drop
5. Toss-consistency tracking (pre-contact ball path)
6. React Native app reading from the same FastAPI server for in-gym use

## Remote access (Cloudflare Tunnel)

The FastAPI server is designed to be fronted by
[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
so coaches, parents, and remote collaborators can reach the Jetson at a
public HTTPS hostname — with optional Zero Trust Access gating — without
opening any inbound ports on your network.

Full walkthrough: [`cloudflare/README.md`](cloudflare/README.md).

Short version:

```bash
# On the Jetson
curl -L -o cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb
cloudflared tunnel login
cloudflared tunnel create floatpro-jetson
cloudflared tunnel route dns floatpro-jetson floatpro.yourdomain.com
cp cloudflare/config.yml.template ~/.cloudflared/config.yml
# edit UUID / USER / HOSTNAME in the config
sudo cloudflared --config ~/.cloudflared/config.yml service install
sudo systemctl start cloudflared
```

Gate with Zero Trust Access from the Cloudflare dashboard (policy on
the app hostname → allow specific emails or SSO). No auth code in the
Python server — authentication happens at the Cloudflare edge.

## Adding a new camera

Subclass `floatpro.cameras.base.Camera`, implement `start() / read() /
stop()`, add a preset and a factory entry in
`floatpro/cameras/__init__.py`. That's it — `capture.py`,
`check_camera.py`, the ring buffer, and everything downstream will
pick it up.
