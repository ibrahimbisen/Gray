"""Serve the dashboard. Python standard library only - no npm, no build step.

    python run.py

Everything it shows comes from two places: dashboard/plan.py for the plan, and
tools/check_urdf.py run live against the current model.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import sys
import webbrowser
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import plan  # noqa: E402
from tools import check_urdf  # noqa: E402

HERE = Path(__file__).resolve().parent
URDF = ROOT / "sim" / "models" / "gray.urdf"


def model_status() -> dict:
    """Run the URDF checks now, so the page is never showing a stale result."""
    if not URDF.exists():
        return {
            "urdf": str(URDF.relative_to(ROOT)),
            "exists": False,
            "passed": 0,
            "total": 10,
            "checks": [],
        }
    try:
        checks = check_urdf.run_checks(URDF)
    except Exception as exc:  # noqa: BLE001
        return {
            "urdf": str(URDF.relative_to(ROOT)),
            "exists": True,
            "passed": 0,
            "total": 10,
            "error": str(exc),
            "checks": [],
        }
    out = check_urdf.as_dict(URDF, checks)
    out["urdf"] = str(URDF.relative_to(ROOT))
    return out


def state() -> dict:
    return {
        "goal": plan.GOAL,
        "stages": plan.STAGES,
        "scoring_intro": plan.SCORING_INTRO,
        "rewards": plan.REWARDS,
        "feasibility": plan.FEASIBILITY,
        "model": model_status(),
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] == "/api/state":
            body = json.dumps(state()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *a):  # keep the terminal quiet
        pass


def serve(port: int = 8000, open_browser: bool = True) -> None:
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"Dashboard: {url}")
        print("Ctrl-C to stop.")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
