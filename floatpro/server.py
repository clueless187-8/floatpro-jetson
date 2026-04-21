"""
FloatPro FastAPI server.

Runs on the Jetson. Exposes:

  GET  /                     → liveness + system status
  GET  /health               → simple health check
  GET  /api/sessions         → list captured sessions
  GET  /api/sessions/{id}    → session metadata
  POST /api/sessions/{id}/analyze → run spin estimation and cache result
  GET  /api/sessions/{id}/frames/{n} → serve a single PNG frame
  GET  /api/sessions/{id}/result → cached analysis JSON

Intended to be fronted by cloudflared so that a remote browser (or the
future React Native app) can reach the Jetson without port-forwarding.
See README section "Remote access via Cloudflare Tunnel".

Security
--------
This server does NOT implement auth itself. Auth lives at the
Cloudflare edge — Zero Trust Access rules gate the tunnel hostname.
If you expose this server on a LAN IP, put it behind something (even
just basic-auth in a reverse proxy) before trusting it. The server
assumes its network is trusted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from floatpro.spin_estimator import estimate_session_spin
from dataclasses import asdict


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAPTURES_DIR = Path("captures").resolve()
STATIC_DIR = Path(__file__).parent / "static"
RESULT_FILENAME = "spin_result.json"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FloatPro",
    description="Volleyball serve analysis — Jetson-side API",
    version="0.3.0",
)

# Permissive CORS so the browser dashboard (on Cloudflare Pages or
# localhost during dev) can hit this API. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_dir(session_id: str) -> Path:
    # Defend against path traversal: session_id must be a single path
    # component. Reject anything with slashes, "..", or leading dots.
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        raise HTTPException(400, "invalid session id")
    d = CAPTURES_DIR / session_id
    if not d.exists() or not d.is_dir():
        raise HTTPException(404, "session not found")
    return d


def _load_session_metadata(session_dir: Path) -> dict:
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, "session metadata missing")
    return json.loads(meta_path.read_text())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    # Serve the dashboard if present, else return service info JSON.
    dash = STATIC_DIR / "dashboard.html"
    if dash.exists():
        return FileResponse(dash, media_type="text/html")
    return JSONResponse({
        "service": "FloatPro",
        "version": app.version,
        "captures_dir": str(CAPTURES_DIR),
        "captures_exists": CAPTURES_DIR.exists(),
    })


@app.get("/api/status")
def status():
    return {
        "service": "FloatPro",
        "version": app.version,
        "captures_dir": str(CAPTURES_DIR),
        "captures_exists": CAPTURES_DIR.exists(),
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/sessions")
def list_sessions():
    """List captured sessions, newest first."""
    if not CAPTURES_DIR.exists():
        return {"sessions": []}
    out = []
    for d in sorted(CAPTURES_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        result_path = d / RESULT_FILENAME
        out.append({
            "id": d.name,
            "frame_count": meta.get("frame_count"),
            "camera": meta.get("camera", {}).get("model"),
            "fps": meta.get("timing", {}).get("effective_fps"),
            "analyzed": result_path.exists(),
        })
    return {"sessions": out}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    d = _session_dir(session_id)
    meta = _load_session_metadata(d)
    # Don't ship the full per-frame list by default; it can be huge.
    summary = {
        "id": session_id,
        "camera": meta.get("camera"),
        "frame_count": meta.get("frame_count"),
        "timing": meta.get("timing"),
    }
    return summary


@app.post("/api/sessions/{session_id}/analyze")
def analyze_session(session_id: str, force: bool = False):
    """Run spin estimation on a session. Caches the result on disk."""
    d = _session_dir(session_id)
    meta = _load_session_metadata(d)
    cached_path = d / RESULT_FILENAME

    if cached_path.exists() and not force:
        return json.loads(cached_path.read_text())

    fps = (meta.get("timing") or {}).get(
        "effective_fps", meta.get("camera", {}).get("fps", 120)
    )

    frames = []
    for entry in meta.get("frames", []):
        p = d / entry["file"]
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)

    if not frames:
        raise HTTPException(500, "no frames could be loaded")

    result = estimate_session_spin(frames, fps=float(fps))
    # SpinResult has Detection objects; convert to dict form for JSON
    payload = asdict(result)
    # Strip full detections list from JSON response (keep count); cache on
    # disk has the full list if needed for frontend overlay rendering.
    cached_path.write_text(json.dumps(payload, indent=2, default=str))
    slim = {k: v for k, v in payload.items() if k != "detections"}
    slim["detection_count"] = sum(1 for x in payload.get("detections") or [] if x)
    return slim


@app.get("/api/sessions/{session_id}/result")
def get_result(session_id: str):
    d = _session_dir(session_id)
    p = d / RESULT_FILENAME
    if not p.exists():
        raise HTTPException(404, "not analyzed yet — POST /analyze first")
    return json.loads(p.read_text())


@app.get("/api/sessions/{session_id}/frames/{n}")
def get_frame(session_id: str, n: int):
    """Serve a single frame PNG. Useful for scrubbing in the dashboard."""
    d = _session_dir(session_id)
    fname = f"frame_{n:05d}.png"
    p = d / fname
    if not p.exists():
        raise HTTPException(404, "frame not found")
    return FileResponse(p, media_type="image/png")


@app.get("/api/sessions/{session_id}/frames/{n}/annotated")
def get_frame_annotated(session_id: str, n: int):
    """Serve a frame with the detection + angle overlay drawn on it.
    Requires the session to have been analyzed."""
    d = _session_dir(session_id)
    result_path = d / RESULT_FILENAME
    if not result_path.exists():
        raise HTTPException(404, "not analyzed yet — POST /analyze first")
    fname = f"frame_{n:05d}.png"
    p = d / fname
    if not p.exists():
        raise HTTPException(404, "frame not found")

    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(500, "failed to load frame")
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result = json.loads(result_path.read_text())
    dets = result.get("detections") or []
    if n < len(dets) and dets[n]:
        det = dets[n]
        cx, cy, r = int(det["cx"]), int(det["cy"]), int(det["r"])
        cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)

    rpm = result.get("rpm")
    if rpm is not None:
        label = f"RPM: {rpm:.0f}"
        if result.get("direction"):
            label += f" ({result['direction']})"
        cv2.putText(vis, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    ok, buf = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(content=buf.tobytes(), media_type="image/png")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0",
                   help="Bind address (0.0.0.0 for all interfaces)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--captures", default="captures",
                   help="Captures directory")
    args = p.parse_args()

    global CAPTURES_DIR
    CAPTURES_DIR = Path(args.captures).resolve()

    print(f"FloatPro server on {args.host}:{args.port}")
    print(f"Captures: {CAPTURES_DIR}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
