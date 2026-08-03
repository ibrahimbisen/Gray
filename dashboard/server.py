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

from dashboard import plan, progress, queue, runs, skills  # noqa: E402
from tools import check_urdf  # noqa: E402

# Modules whose contents the pages are built from. The HTML is re-read on every
# request, but these are imported once at startup - so editing a reward
# description or a status rule and seeing no change on the page is a trap that
# looks like a bug in the page. Reload them when the file on disk moves.
_WATCHED = (skills, plan, runs, queue, progress)
_MTIMES: dict[str, float] = {}


def _reload_if_edited() -> None:
    import importlib  # noqa: PLC0415

    for module in _WATCHED:
        path = Path(module.__file__)
        try:
            stamp = path.stat().st_mtime
        except OSError:
            continue
        if _MTIMES.get(module.__name__, stamp) != stamp:
            importlib.reload(module)
        _MTIMES[module.__name__] = stamp

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
    _reload_if_edited()
    return {
        "goal": plan.GOAL,
        "project": plan.PROJECT,
        "loop": plan.LOOP,
        "stages": plan.STAGES,
        "scoring_intro": plan.SCORING_INTRO,
        # Read off gray/tasks/ rather than hand-written here. Slow the first time
        # (it imports torch) and cached against the task files' mtime after that.
        "rewards": plan.rewards(),
        "feasibility": plan.FEASIBILITY,
        "model": model_status(),
    }


def stage_state(n: int) -> dict | None:
    """One project stage, with anything live it depends on folded in."""
    _reload_if_edited()
    if n == 1:
        return {**plan.STAGE1, "model": model_status()}
    if n == 2:
        return plan.stage2_state()
    if n == 3:
        return plan.STAGE3
    return None


def monitor_state() -> dict:
    """The training centre's poll. Deliberately light.

    No metric rows: this is fetched every few seconds and used to grow forever,
    because it carried every metric row of every run. At sixteen runs that was
    1.6 MB per poll and rising. The curves for the ONE run being looked at come
    from /api/run/<id> instead.

    No reward table either - the monitor scores a RUN, and a run carries its own
    terms in its own metadata. Serving the plan's copy as well would be a second
    source for the same thing on the same page.
    """
    _reload_if_edited()
    summaries = runs.all_summaries()
    live = next((r for r in summaries if r["status"] == "running"), None)
    return {
        "runs": summaries,
        "active": live or (summaries[0] if summaries else None),
        "queue": queue.load(),
        "phases": plan.PHASES,
        "next_up": plan.NEXT_UP,
        "stages": plan.STAGES,
        "model": model_status(),
        "metrics_available": runs.metrics_available(),
    }


def overview_state() -> dict:
    """The project against its bars. Reuses one walk of progress/runs/.

    all_summaries() is the only disk read here; everything progress.overview()
    does is arithmetic over dicts already in memory. Sharing the list matters -
    computing it twice would double the cost of the page that is meant to be the
    cheap one.
    """
    _reload_if_edited()
    summaries = runs.all_summaries()
    live = next((r for r in summaries if r["status"] == "running"), None)
    # The run list, trimmed. The overview links INTO runs, so it needs enough to
    # label and rank them - but never their metric rows, which is what made the
    # monitor's payload 1.6 MB.
    keep = ("id", "number", "variant", "name", "task", "status", "verdict", "started_age",
            "duration", "iterations_done", "iterations_target", "progress",
            "videos", "verdict_structured")
    return {
        "progress": progress.overview(summaries),
        "runs": [{k: r.get(k) for k in keep} | {"reward": (r.get("latest") or {}).get("reward")}
                 for r in summaries],
        "queue": queue.load(),
        "running": live,
        "loop": plan.LOOP,
        "stage1_open": plan.STAGE1["open"],
        "next_up": plan.NEXT_UP,
        "phases": plan.PHASES,
        "blockers": progress.blockers(),
        "startable": progress.startable(),
        "model": model_status(),
        "feasibility": plan.FEASIBILITY,
    }


def run_detail(run_id: str, points: int = 0) -> dict | None:
    _reload_if_edited()
    return runs.detail(run_id, points=points)


def compare_state(run_ids: list[str], metric: str) -> dict:
    _reload_if_edited()
    return {
        "metric": metric,
        "available": runs.metrics_available(),
        "series": runs.series(run_ids, metric),
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def handle_one_request(self):
        """Never let a handler die without answering.

        socketserver closes the connection on an unhandled exception, and the
        browser sees a network error with nothing in it. Every page then reads
        "Could not load" with no clue which request failed or why. Answering with
        the traceback turns a silent dead page into a message that names the bug.
        """
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True        # the browser went away; normal
        except Exception as exc:  # noqa: BLE001
            import traceback  # noqa: PLC0415

            detail = traceback.format_exc()
            print(f"[server] {self.path}\n{detail}", flush=True)
            try:
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}",
                                   "where": self.path,
                                   "traceback": detail}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass
            self.close_connection = True

    def _send(self, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload) -> None:
        self._send(json.dumps(payload).encode(), "application/json")

    def _queue_post(self, path: str, body: dict):
        """Queue control. Every one of these ends by returning the whole queue,
        so the page never has to guess what its edit did - it redraws from the
        server's answer rather than from an optimistic local copy."""
        _reload_if_edited()
        job_id = str(body.get("id", ""))
        if path == "/api/queue/add":
            queue.add(body.get("job") or body, position=body.get("position", "end"))
        elif path == "/api/queue/edit":
            queue.edit(job_id, body.get("job") or {})
        elif path == "/api/queue/remove":
            queue.remove(job_id)
        elif path == "/api/queue/move":
            queue.move(job_id, int(body.get("delta", 0)))
        elif path == "/api/queue/duplicate":
            queue.duplicate(job_id)
        elif path == "/api/queue/pause":
            queue.set_paused(bool(body.get("paused", True)))
        elif path == "/api/queue/clear":
            queue.clear_finished()
        else:
            return self.send_error(404)
        return self._json(queue.load())

    def do_POST(self):  # noqa: N802
        path = unquote(self.path.split("?")[0])
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json({"error": "body was not JSON"})
        if not isinstance(body, dict):
            body = {}

        if path.startswith("/api/queue/"):
            try:
                return self._queue_post(path, body)
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": f"{type(exc).__name__}: {exc}"})

        if path not in ("/api/pose/limits", "/api/pose/directions"):
            return self.send_error(404)
        from dashboard import poser  # noqa: PLC0415

        try:
            saved = (poser.save_limits(body) if path.endswith("limits")
                     else poser.save_directions(body.get("invert", [])))
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc)})
        return self._json(saved)

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
            png, facts = poser.pose_report(
                json.loads(query.get("angles", ["{}"])[0]),
                azimuth=float(query.get("az", [125])[0]),
                elevation=float(query.get("el", [-12])[0]),
                distance=float(query.get("dist", [1.05])[0]),
                invert=json.loads(query.get("invert", ["{}"])[0]),
            )
            return self._send(json.dumps({
                "png": base64.b64encode(png).decode(), "facts": facts,
            }).encode(), "application/json")

        if path == "/api/monitor":
            return self._send(json.dumps(monitor_state()).encode(), "application/json")
        if path == "/api/state":
            return self._send(json.dumps(summary_state()).encode(), "application/json")
        if path == "/api/overview":
            return self._send(json.dumps(overview_state()).encode(), "application/json")
        if path == "/api/queue":
            _reload_if_edited()
            return self._send(json.dumps(queue.load()).encode(), "application/json")
        if path.startswith("/api/run/"):
            # points= thins the curve for drawing. 0 means every sample, which is
            # what the table twin needs - a value must never be reachable only by
            # hovering a chart.
            pts = int(query.get("points", ["0"])[0] or 0)
            state = run_detail(unquote(path[len("/api/run/"):]), points=pts)
            if state is None:
                return self.send_error(404)
            return self._send(json.dumps(state).encode(), "application/json")
        if path == "/api/compare":
            ids = [r for r in (query.get("runs", [""])[0]).split(",") if r]
            metric = query.get("metric", ["reward"])[0]
            return self._send(
                json.dumps(compare_state(ids, metric)).encode(), "application/json")
        if path == "/api/job-log":
            # The runner's own output for one job. This is where "why did that
            # fail" actually lives, and it is otherwise only in a terminal that
            # may have been closed.
            name = query.get("id", [""])[0]
            log = ROOT / "progress" / "jobs" / f"{name}.log"
            if not name or ".." in name or "/" in name or not log.is_file():
                return self._send(json.dumps({"text": ""}).encode(), "application/json")
            text = log.read_text(encoding="utf-8", errors="replace")
            return self._send(
                json.dumps({"text": text[-40000:]}).encode(), "application/json")
        if path.startswith("/api/stage/"):
            try:
                state = stage_state(int(path.rsplit("/", 1)[1]))
            except ValueError:
                state = None
            if state is None:
                return self.send_error(404)
            return self._send(json.dumps(state).encode(), "application/json")

        if path == "/page.css":
            return self._send((HERE / "page.css").read_bytes(), "text/css; charset=utf-8")
        if path == "/page.js":
            return self._send((HERE / "page.js").read_bytes(),
                              "text/javascript; charset=utf-8")

        if path in ("/", "/index.html", "/overview"):
            return self._send((HERE / "overview.html").read_bytes(),
                              "text/html; charset=utf-8")
        if path in ("/runs", "/monitor", "/training"):
            return self._send((HERE / "monitor.html").read_bytes(),
                              "text/html; charset=utf-8")
        if path in ("/summary", "/summary.html", "/plan"):
            return self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        # The three project stages, each its own page.
        if path in ("/stage1", "/stage2", "/stage3"):
            return self._send((HERE / f"{path[1:]}.html").read_bytes(),
                              "text/html; charset=utf-8")

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

        # A malformed or multi-range header used to raise straight out of the
        # handler, killing the request with no response. Anything we cannot read
        # falls back to serving the whole file, which is always a valid answer.
        try:
            first = rng[len("bytes="):].split(",")[0]
            start_s, _, end_s = first.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
        except ValueError:
            with target.open("rb") as fh:
                return self._send(fh.read(), ctype, cache=True)
        end = min(end, size - 1)
        if start < 0 or start > end:
            # Out of range. 416 is the honest answer; serving the whole file with
            # a nonsense Content-Range makes a video player seek to garbage.
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
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
        print(f"Overview:  {url}            everything, at a glance")
        print(f"Runs:      {url}runs        one run: curves, films, rewards")
        print(f"Summary:   {url}summary")
        print(f"  stage 1: {url}stage1     Prepare - the model (provisional)")
        print(f"  stage 2: {url}stage2     Train - 200 skills, three training runs")
        print(f"  stage 3: {url}stage3     Deploy - the real robot (start now)")
        print(f"Pose:      {url}pose")
        print("Ctrl-C to stop.")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
