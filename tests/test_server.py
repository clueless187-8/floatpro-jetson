"""End-to-end: capture a mock session, call the server's API, fetch results."""
import sys, os, json, shutil, tempfile
sys.path.insert(0, ".")

from fastapi.testclient import TestClient
import cv2

from floatpro.cameras import make_camera, CameraConfig
from floatpro import server


def test_end_to_end():
    # Use a temp captures dir so we don't pollute the real one
    tmpdir = tempfile.mkdtemp(prefix="floatpro_test_")
    try:
        captures = os.path.join(tmpdir, "captures")
        session_id = "20260421_120000_000"
        session_dir = os.path.join(captures, session_id)
        os.makedirs(session_dir)

        # Capture 40 frames from mock and write to disk in the layout
        # capture.py produces
        cam = make_camera("mock", CameraConfig(width=640, height=480, fps=120))
        info = cam.start()
        frames_meta = []
        for i in range(40):
            ok, frame, ts = cam.read()
            if not ok:
                continue
            fname = f"frame_{i:05d}.png"
            cv2.imwrite(os.path.join(session_dir, fname), frame)
            frames_meta.append({"i": i, "t": round(ts, 6), "file": fname})
        cam.stop()

        metadata = {
            "session_id": session_id,
            "camera": {
                "backend": info.backend, "model": info.model,
                "serial": info.serial, "width": info.width,
                "height": info.height, "fps": info.fps,
                "pixel_format": info.pixel_format,
                "is_color": info.is_color,
                "is_global_shutter": info.is_global_shutter,
            },
            "frame_count": len(frames_meta),
            "frames": frames_meta,
            "timing": {"effective_fps": 120.0, "avg_interval_ms": 8.33,
                       "min_interval_ms": 8.0, "max_interval_ms": 9.0},
        }
        with open(os.path.join(session_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Point server at our temp captures dir
        from pathlib import Path
        server.CAPTURES_DIR = Path(captures).resolve()

        client = TestClient(server.app)

        # Test 1: health
        r = client.get("/health")
        assert r.status_code == 200 and r.json() == {"ok": True}
        print("  PASS /health")

        # Test 2: list sessions
        r = client.get("/api/sessions")
        assert r.status_code == 200
        j = r.json()
        assert len(j["sessions"]) == 1
        assert j["sessions"][0]["id"] == session_id
        assert j["sessions"][0]["analyzed"] is False
        print(f"  PASS /api/sessions -> {len(j['sessions'])} session(s)")

        # Test 3: session details
        r = client.get(f"/api/sessions/{session_id}")
        assert r.status_code == 200
        assert r.json()["frame_count"] == 40
        print("  PASS /api/sessions/{id}")

        # Test 4: analyze
        r = client.post(f"/api/sessions/{session_id}/analyze")
        assert r.status_code == 200, r.text
        result = r.json()
        print(f"  PASS /analyze -> rpm={result.get('rpm'):.1f} "
              f"direction={result.get('direction')} "
              f"method={result.get('method')}")
        assert result["rpm"] is not None
        assert 110 < result["rpm"] < 170, f"RPM out of range: {result['rpm']}"

        # Test 5: cached result
        r = client.get(f"/api/sessions/{session_id}/result")
        assert r.status_code == 200
        assert r.json()["rpm"] is not None
        print("  PASS /result (cached)")

        # Test 6: list shows analyzed=True now
        r = client.get("/api/sessions")
        assert r.json()["sessions"][0]["analyzed"] is True
        print("  PASS analyzed flag updated")

        # Test 7: raw frame
        r = client.get(f"/api/sessions/{session_id}/frames/5")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert len(r.content) > 100
        print(f"  PASS /frames/5 -> {len(r.content)} bytes")

        # Test 8: annotated frame
        r = client.get(f"/api/sessions/{session_id}/frames/5/annotated")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        print(f"  PASS /frames/5/annotated -> {len(r.content)} bytes")

        # Test 9: dashboard HTML
        r = client.get("/")
        assert r.status_code == 200
        assert "FloatPro" in r.text
        print("  PASS /  (dashboard HTML)")

        # Test 10: path traversal defense
        r = client.get("/api/sessions/..%2Fsecret")
        assert r.status_code in (400, 404)
        print(f"  PASS path traversal blocked ({r.status_code})")

        # Test 11: nonexistent session
        r = client.get("/api/sessions/does_not_exist")
        assert r.status_code == 404
        print("  PASS missing session -> 404")

        print("\n  ALL SERVER TESTS PASSED")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("\n[RUN] test_server_end_to_end")
    try:
        test_end_to_end()
        print("[PASS]")
        sys.exit(0)
    except AssertionError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
