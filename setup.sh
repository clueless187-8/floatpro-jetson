#!/usr/bin/env bash
# FloatPro — Jetson setup for Arducam OV9281
# Run once on a fresh Jetson Orin Nano Super (JetPack 6.x).
#
# This installs:
#   - v4l-utils, GStreamer tooling
#   - System python3-opencv (pre-built with GStreamer support on Jetson)
#   - Maxes out Jetson clocks for consistent framerate testing
#
# It does NOT install the Arducam kernel driver — that step is board-specific
# and Arducam's own installer handles it. Follow the current guide at:
#
#   https://docs.arducam.com/Nvidia-Jetson-Camera/Jetvariety-Camera/Quick-Start-Guide/
#
# Briefly: Arducam publishes an install_full.sh for each JetPack release.
# Run it with -m ov9281 to select the OV9281 sensor. A reboot is required.

set -e

echo "=============================================="
echo "  FloatPro Jetson setup"
echo "=============================================="

# --- Sanity checks ----------------------------------------------------------
if [ ! -f /etc/nv_tegra_release ]; then
    echo "WARNING: /etc/nv_tegra_release not found — this may not be a Jetson."
    read -p "Continue anyway? [y/N] " yn
    [[ "$yn" == "y" || "$yn" == "Y" ]] || exit 1
fi

echo ""
echo "Jetson release:"
cat /etc/nv_tegra_release 2>/dev/null || echo "(unknown)"

# --- System packages --------------------------------------------------------
echo ""
echo "Installing system packages..."
sudo apt update
sudo apt install -y \
    v4l-utils \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    python3-pip \
    python3-opencv \
    python3-numpy

# --- Verify OpenCV has GStreamer -------------------------------------------
echo ""
echo "Verifying OpenCV GStreamer support..."
python3 -c "
import cv2
info = cv2.getBuildInformation()
gst = [l for l in info.split('\n') if 'GStreamer' in l]
for l in gst: print('  ' + l.strip())
if not any('YES' in l for l in gst):
    print('WARNING: OpenCV lacks GStreamer. Do not use pip opencv-python on Jetson.')
"

# --- Jetson performance mode -----------------------------------------------
echo ""
echo "Setting Jetson to max performance mode (persists until reboot)..."
sudo nvpmodel -m 0 || echo "  nvpmodel failed (non-fatal)"
sudo jetson_clocks  || echo "  jetson_clocks failed (non-fatal)"

echo ""
echo "=============================================="
echo "  System setup complete."
echo ""
echo "  NEXT STEPS:"
echo "  1. Install Arducam driver (see README.md section 'Driver install')"
echo "  2. Reboot"
echo "  3. Connect the OV9281 via CSI"
echo "  4. Run: python3 check_camera.py"
echo "=============================================="
