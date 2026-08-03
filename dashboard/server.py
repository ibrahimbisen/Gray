"""Serve the dashboard. Python standard library only - no npm, no build step.

    python run.py

Two pages:
    /           the training monitor - what is training now, and the plan
    /summary    the project summary  - the full curriculum and every scoring term

Everything they show comes from three places, none of which this file invents:
dashboard/plan.py for the plan, dashboard/runs.py for what training wrote to disk,
and tools/check_urdf.py run live against the current model.
"""

from __future__ import annotations

import http.server
import json
import mimetypes
import socketserver
import sys
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, unquote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import plan, runs  # noqa: E402
from tools import check_urdf  # noqa: E402

HERE = Path(__file__).resolve().parent
URDF = ROOT / "sim" / "models" / "gray.urdf"
MEDIA = ROOT / "progress"


def model_status() -> dict:
    """Run the URDF checks now, so the page is never showing a stale result."""
    blank = {"urdf": "sim/models/gray.urdf", "exists": False, "passed": 0,
             "total": 11, "checks": []}
    if not URDF.exists():
        return blank
    try:
        checks = check_urdf.run_checks(URDF)
    except Exception as exc:  # noqa: BLE001
        return {**blank, "exists": True, "error": str(exc)}
    out = check_urdf.as_dict(URDF, checks)
    out["urdf"] = str(URDF.relative_to(ROOT)).replace("\\", "/")
    return out


def summary_state() -> dict:
    return {
        "goal": plan.GOAL,
        "stages": plan.STAGES,
        "scoring_intro": plan.SCORING_INTRO,
        "rewards": plan.REWARDS,
        "feasibility": plan.FEASIBILITY,
        "model": model_status(),
    }


def monitor_state() -> dict:
    all_runs = runs.all_runs()
    return {
        "runs": all_runs,
        "active": runs.active(all_runs),
        "phases": plan.PHASES,
        "next_up": plan.NEXT_UP,
        "stages": plan.STAGES,
        "rewards": plan.REWARDS,
        "model": model_status(),
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def _send(self, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if unquote(self.path.split("?")[0]) != "/api/pose/limits":
            return self.send_error(404)
        from dashboard import poser  # noqa: PLC0415

        n = int(self.headers.get("Content-Length", 0))
        try:
            saved = poser.save_limits(json.loads(self.rfile.read(n) or b"{}"))
        except Exception as exc:  # noqa: BLE001
            return self._send(json.dumps({"error": str(exc)}).encode(), "application/json")
        return self._send(json.dumps(saved).encode(), "application/json")

    def do_GET(self):  # noqa: N802, C901
        path = unquote(self.path.split("?")[0])
        query = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}

        if path == "/pose":
            return self._send((HERE / "pose.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/pose/config":
            from dashboard import poser  # noqa: PLC0415
            return self._send(json.dumps(poser.defaults()).encode(), "application/json")
        if path == "/api/pose":
            import base64  # noqa: PLC0415

            from dashboard import poser  # noqa: PLC0415
            angles = json.loads(query.get("angles", ["{}"])[0])
            png, facts = poser.pose_report(
                angles,
                azimuth=float(query.get("az", [125])[0]),
                distance=float(query.get("dist", [1.05])[0]),
            )
            return self._send(json.dumps({
                "png": base64.b64encode(png).decode(), "facts": facts,
            }).encode(), "application/json")

        if path == "/api/monitor":
            return self._send(json.dumps(monitor_state()).encode(), "application/json")
        if path == "/api/state":
            return self._send(json.dumps(summary_state()).encode(), "application/json")

        if path in ("/", "/index.html", "/monitor"):
            return self._send((HERE / "monitor.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/summary", "/summary.html", "/plan"):
            return self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")

        # Videos and images produced by training. Served with range support, which
        # is what lets a browser scrub through a video instead of downloading it all.
        if path.startswith("/media/"):
            return self._serve_media(path[len("/media/"):])

        self.send_error(404)
        return None

    def _serve_media(self, rel: str):
        target = (MEDIA / rel).resolve()
        try:
            target.relative_to(MEDIA.resolve())  # refuse to serve outside progress/
        except ValueError:
            return self.send_error(403)
        if not target.is_file():
            return self.send_error(404)

        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        size = target.stat().st_size
        rng = self.headers.get("Range")
        if not rng or not rng.startswith("bytes="):
            with target.open("rb") as fh:
                return self._send(fh.read(), ctype, cache=True)

        start_s, _, end_s = rng[len("bytes="):].partition("-")
        start = int(start_s or 0)
        end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        with target.open("rb") as fh:
            fh.seek(start)
            chunk = fh.read(end - start + 1)
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)
        return None

    def log_message(self, *a):  # keep the terminal quiet
        pass


def serve(port: int = 8000, open_browser: bool = True) -> None:
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"Dashboard: {url}")
        print(f"Pose:      {url}pose")
        print(f"Summary:   {url}summary")
        print("Ctrl-C to stop.")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
