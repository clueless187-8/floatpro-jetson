#!/usr/bin/env python3
"""
FloatPro capture — threaded ring buffer for Arducam OV9281 on Jetson Orin Nano.

Maintains a rolling N-second window of frames in RAM. SPACE dumps the current
window to disk as timestamped PNGs + metadata JSON. This is the "smoke test"
rig: proves the camera streams cleanly at 120fps and the ring buffer pattern
works before we layer in YOLO / spin estimation.

Controls:
    SPACE   save the last BUFFER_SECONDS of frames
    q       quit
"""

import collections
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

# ---- Config ----------------------------------------------------------------
DEVICE          = "/dev/video0"
WIDTH           = 1280
HEIGHT          = 800
FPS             = 120
BUFFER_SECONDS  = 5
OUTPUT_DIR      = Path("captures")

# GStreamer pipeline for OV9281 on Jetson.
# drop=1 / max-buffers=2 keeps appsink from ballooning if Python falls behind.
PIPELINE = (
    f"v4l2src device={DEVICE} ! "
    f"video/x-raw,format=GRAY8,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! "
    f"videoconvert ! "
    f"appsink drop=1 max-buffers=2 sync=false"
)


class RingBufferCapture:
    """Captures frames into a fixed-length deque on a background thread."""

    def __init__(self, pipeline: str, capacity_frames: int):
        self.pipeline = pipeline
        self.buffer = collections.deque(maxlen=capacity_frames)
        self.running = False
        self.frame_count = 0
        self.drop_count = 0
        self.fps_actual = 0.0
        self.lock = threading.Lock()
        self.cap = None
        self.thread = None

    def start(self):
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Failed to open camera. Check:\n"
                "  1. Arducam driver installed (run install_driver.sh)\n"
                "  2. /dev/video0 exists (ls /dev/video*)\n"
                "  3. GStreamer support in OpenCV "
                "(python3 -c 'import cv2; print(cv2.getBuildInformation())' | grep GStreamer)"
            )
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[capture] streaming started")

    def _loop(self):
        last_fps_t = time.time()
        fps_frames = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.drop_count += 1
                time.sleep(0.001)
                continue
            ts = time.time()
            with self.lock:
                self.buffer.append((ts, frame))
                self.frame_count += 1
            fps_frames += 1

            now = time.time()
            if now - last_fps_t >= 1.0:
                self.fps_actual = fps_frames / (now - last_fps_t)
                fps_frames = 0
                last_fps_t = now

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()

    def snapshot(self):
        """Return a shallow copy of the buffer — safe to iterate without holding lock."""
        with self.lock:
            return list(self.buffer)

    def latest(self):
        with self.lock:
            return self.buffer[-1][1] if self.buffer else None

    @property
    def buffer_len(self):
        with self.lock:
            return len(self.buffer)


def save_buffer(frames, output_root: Path = OUTPUT_DIR):
    """Write frames + metadata to disk. Called from a worker thread so the
    capture loop never stalls on I/O."""
    if not frames:
        print("[save] empty buffer, nothing to write")
        return

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    session_dir = output_root / ts_str
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"[save] {len(frames)} frames -> {session_dir}")
    t0 = time.time()

    meta = {
        "session_id": ts_str,
        "frame_count": len(frames),
        "resolution": [WIDTH, HEIGHT],
        "target_fps": FPS,
        "frames": [],
    }

    for i, (frame_ts, frame) in enumerate(frames):
        fname = f"frame_{i:05d}.png"
        cv2.imwrite(str(session_dir / fname), frame)
        meta["frames"].append({
            "i": i,
            "t": round(frame_ts, 6),
            "file": fname,
        })

    # Per-frame timing diagnostics — if intervals drift, the camera dropped frames
    if len(frames) > 1:
        intervals = [frames[i + 1][0] - frames[i][0] for i in range(len(frames) - 1)]
        avg_ms = 1000 * sum(intervals) / len(intervals)
        max_ms = 1000 * max(intervals)
        min_ms = 1000 * min(intervals)
        meta["timing"] = {
            "avg_interval_ms": round(avg_ms, 3),
            "min_interval_ms": round(min_ms, 3),
            "max_interval_ms": round(max_ms, 3),
            "effective_fps": round(1000 / avg_ms, 2),
        }
        print(f"[save] effective fps={meta['timing']['effective_fps']}  "
              f"interval avg={avg_ms:.2f}ms min={min_ms:.2f}ms max={max_ms:.2f}ms")

    with open(session_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[save] done in {time.time() - t0:.1f}s")


def draw_hud(frame, fps_actual, buf_len, buf_cap, drops):
    """Overlay live stats so we can see frame rate / drops in real time."""
    if len(frame.shape) == 2:
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        display = frame.copy()

    # Scale down for display (keep full res in buffer)
    h, w = display.shape[:2]
    scale = 0.7
    display = cv2.resize(display, (int(w * scale), int(h * scale)))

    hud1 = f"FPS: {fps_actual:5.1f}  |  Buffer: {buf_len}/{buf_cap}  |  Drops: {drops}"
    hud2 = "SPACE = save last 5s   Q = quit"

    cv2.putText(display, hud1, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(display, hud2, (10, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return display


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("FloatPro Capture")
    print("=" * 60)
    print(f"Pipeline    : {PIPELINE}")
    print(f"Buffer      : {BUFFER_SECONDS}s @ {FPS}fps = {BUFFER_SECONDS * FPS} frames")
    print(f"RAM est.    : ~{BUFFER_SECONDS * FPS * WIDTH * HEIGHT / 1024 / 1024:.0f} MB")
    print(f"Output dir  : {OUTPUT_DIR.absolute()}")
    print("=" * 60)

    cap = RingBufferCapture(PIPELINE, BUFFER_SECONDS * FPS)
    cap.start()

    save_threads = []

    try:
        while True:
            frame = cap.latest()
            if frame is not None:
                hud = draw_hud(frame, cap.fps_actual, cap.buffer_len,
                               cap.buffer.maxlen, cap.drop_count)
                cv2.imshow("FloatPro - OV9281", hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Snapshot + hand off to a save thread so the capture loop
                # never blocks on disk I/O
                frames = cap.snapshot()
                t = threading.Thread(target=save_buffer, args=(frames,), daemon=False)
                t.start()
                save_threads.append(t)

    except KeyboardInterrupt:
        print("\n[main] interrupt")
    finally:
        print("[main] stopping capture...")
        cap.stop()
        cv2.destroyAllWindows()
        for t in save_threads:
            t.join()
        print(f"[main] total frames: {cap.frame_count}  drops: {cap.drop_count}")


if __name__ == "__main__":
    main()
