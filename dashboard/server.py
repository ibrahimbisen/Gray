"""Gray dashboard - tiny local web server.

Standard library only (no Flask, no FastAPI). Serves three things:

    GET /              the dashboard page (dashboard/index.html)
    GET /api/status    everything the page needs, as JSON, from dashboard.collect
    GET /media/<path>  a picture or video from the repo folder
    GET /videos/<path> a walk video from progress/videos/

Run it with:

    .venv\\Scripts\\python.exe dashboard\\server.py

then leave the browser tab open - the page refreshes itself while training runs.
"""

from __future__ import annotations

import argparse
import errno
import http.server
import json
import mimetypes
import re
import socketserver
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

# --------------------------------------------------------------------------------------
# Where everything lives
# --------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent          # <repo>/dashboard
REPO_ROOT = HERE.parent                         # <repo>
VIDEO_DIR = REPO_ROOT / "progress" / "videos"
INDEX_HTML = HERE / "index.html"

# How long a built status snapshot is reused before it is rebuilt. The TensorBoard
# event files grow to hundreds of KB, so re-reading them on every browser poll is
# wasteful. Five seconds is far quicker than training produces new numbers anyway.
STATUS_CACHE_SECONDS = 5.0

# Only these file types are ever sent to the browser. Anything else is refused,
# so a stray URL can never hand out source code, checkpoints or config files.
ALLOWED_MEDIA_EXTENSIONS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".svg": "image/svg+xml",
}

# Byte range header, e.g. "bytes=0-1023", "bytes=2048-", "bytes=-4096".
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$", re.IGNORECASE)


# --------------------------------------------------------------------------------------
# Loading the data collector
# --------------------------------------------------------------------------------------

def _import_collect():
    """Find dashboard.collect.collect, whichever way the server was started.

    Returns the function, or None if the collector file is not there yet.
    """
    for path in (str(REPO_ROOT), str(HERE)):
        if path not in sys.path:
            sys.path.insert(0, path)

    # Started as `python -m dashboard.server` (repo root on the path).
    try:
        from dashboard.collect import collect  # type: ignore
        return collect
    except Exception:
        pass

    # Started as `python dashboard/server.py` (dashboard folder on the path).
    try:
        from collect import collect  # type: ignore
        return collect
    except Exception:
        return None


def _json_safe(obj):
    """Last-resort converter so one odd value can never break the whole page."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="seconds")
    item = getattr(obj, "item", None)          # numpy / torch scalars
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    tolist = getattr(obj, "tolist", None)      # numpy arrays
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass
    return str(obj)


class StatusCache:
    """Builds the status dict at most once every STATUS_CACHE_SECONDS."""

    def __init__(self, repo_root: Path, ttl: float = STATUS_CACHE_SECONDS):
        self.repo_root = repo_root
        self.ttl = ttl
        self._lock = threading.Lock()
        self._payload: bytes | None = None
        self._built_at = 0.0
        self._ok = True
        self._collect = _import_collect()

    @property
    def collector_available(self) -> bool:
        return self._collect is not None

    def get(self, force_fresh: bool = False) -> tuple[bytes, bool, bool]:
        """Return (json bytes, ok, was_cached)."""
        with self._lock:
            now = time.monotonic()
            fresh_enough = (
                self._payload is not None
                and not force_fresh
                and (now - self._built_at) < self.ttl
            )
            if fresh_enough:
                return self._payload, self._ok, True

            payload, ok = self._build()
            self._payload = payload
            self._ok = ok
            self._built_at = now
            return payload, ok, False

    def _build(self) -> tuple[bytes, bool]:
        if self._collect is None:
            # The collector may have been written after the server started, so keep
            # trying rather than failing forever.
            self._collect = _import_collect()

        if self._collect is None:
            return self._error_payload(
                "dashboard/collect.py was not found, so there is no data to show yet."
            ), False

        try:
            data = self._collect(repo_root=str(self.repo_root))
        except Exception:
            detail = traceback.format_exc(limit=8).strip().splitlines()[-1]
            return self._error_payload(
                f"Reading the training data failed: {detail}"
            ), False

        try:
            return json.dumps(data, default=_json_safe).encode("utf-8"), True
        except Exception:
            detail = traceback.format_exc(limit=4).strip().splitlines()[-1]
            return self._error_payload(
                f"The collected data could not be turned into JSON: {detail}"
            ), False

    @staticmethod
    def _error_payload(message: str) -> bytes:
        """A valid, correctly shaped status dict that simply says what went wrong.

        The page can render this without special-casing anything - every key it
        expects is present, just empty.
        """
        skeleton = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "goal": {
                "headline": "",
                "explain": [],
                "baseline": {},
                "targets": [],
                "rewards": [],
                "randomization": [],
                "constraints": [],
            },
            "training": {
                "running": False,
                "run_name": "",
                "iteration": 0,
                "total_iterations": 0,
                "num_envs": 0,
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
                "series": {},
                "latest": {},
            },
            "walks": [],
            "baseline_walk": None,
            "robot": {
                "total_mass_kg": 0.0,
                "servo": {},
                "servo_placement": {},
                "links": [],
                "legs": [],
                "facts": [],
            },
            "media": {"cad": [], "build": [], "sim": [], "gifs": []},
            "timeline": [],
            "errors": [message],
        }
        return json.dumps(skeleton).encode("utf-8")


# --------------------------------------------------------------------------------------
# Path safety
# --------------------------------------------------------------------------------------

class PathRefused(Exception):
    """Raised when a requested file is outside the folder we are allowed to serve."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def resolve_inside(base: Path, decoded_tail: str) -> Path:
    """Turn an already-URL-decoded path into a real file inside `base`, or refuse.

    Refusals: 403 for anything that tries to climb out of the folder or asks for a
    file type we do not serve; 404 if the file simply is not there.
    """
    raw = decoded_tail.replace("\\", "/").lstrip("/")

    if not raw:
        raise PathRefused(404, "No file was asked for.")

    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise PathRefused(403, "That path is not allowed.")
    if any(":" in p for p in parts):                    # C:/... or a Windows data stream
        raise PathRefused(403, "That path is not allowed.")
    if not parts:
        raise PathRefused(404, "No file was asked for.")

    base_resolved = base.resolve()
    candidate = (base_resolved / Path(*parts))

    # Resolve symlinks and any remaining trickery, then prove the result is still
    # inside the folder we are allowed to serve from.
    try:
        real = candidate.resolve()
    except OSError:
        raise PathRefused(404, "That file could not be found.")

    if real != base_resolved and base_resolved not in real.parents:
        raise PathRefused(403, "That path is outside the allowed folder.")

    if real.suffix.lower() not in ALLOWED_MEDIA_EXTENSIONS:
        raise PathRefused(403, "That file type is not served.")

    if not real.is_file():
        raise PathRefused(404, "That file could not be found.")

    return real


def content_type_for(path: Path) -> str:
    return ALLOWED_MEDIA_EXTENSIONS.get(
        path.suffix.lower(),
        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    )


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Work out which slice of a file the browser asked for.

    Returns (start, end) inclusive, or None if the header is not a simple single
    range (in which case the whole file is sent, which is always allowed).
    """
    match = _RANGE_RE.match(header.strip())
    if not match:
        return None

    first, last = match.group(1), match.group(2)

    if first == "" and last == "":
        return None
    if first == "":                                    # last N bytes
        length = int(last)
        if length <= 0:
            raise ValueError("empty range")
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(first)
        end = int(last) if last else size - 1
        end = min(end, size - 1)

    if start >= size or start > end:
        raise ValueError("range out of bounds")
    return start, end


# --------------------------------------------------------------------------------------
# The request handler
# --------------------------------------------------------------------------------------

class GrayHandler(http.server.BaseHTTPRequestHandler):
    server_version = "GrayDashboard/1.0"
    protocol_version = "HTTP/1.1"

    status_cache: StatusCache = None      # type: ignore[assignment]  (set on the server)
    quiet: bool = False

    # -- logging -----------------------------------------------------------------

    def log_request(self, code="-", size="-"):        # noqa: D102  (framework hook)
        pass                                          # we log once, ourselves, below

    def log_error(self, fmt, *args):                  # noqa: D102
        self._log_line(fmt % args if args else fmt)

    def log_message(self, fmt, *args):                # noqa: D102
        self._log_line(fmt % args if args else fmt)

    def _log_line(self, text: str) -> None:
        if not self.quiet:
            print(f"  {datetime.now():%H:%M:%S}  {text}", flush=True)

    def _note(self, status: int, detail: str = "") -> None:
        started = getattr(self, "_t0", None)
        ms = f"{(time.monotonic() - started) * 1000:6.0f} ms" if started else ""
        path = getattr(self, "path", "-")
        if len(path) > 60:
            path = path[:57] + "..."
        extra = f"  {detail}" if detail else ""
        self._log_line(f"{status}  {self.command:4s} {path:<60s} {ms}{extra}")

    # -- verbs -------------------------------------------------------------------

    def do_GET(self):                                  # noqa: N802  (framework hook)
        self._handle(include_body=True)

    def do_HEAD(self):                                 # noqa: N802
        self._handle(include_body=False)

    def _handle(self, include_body: bool) -> None:
        self._t0 = time.monotonic()
        try:
            split = urlsplit(self.path)
            # Decode exactly once, here. Everything downstream works on plain text,
            # so "%2e%2e" can never sneak past the traversal check by being decoded
            # a second time later on.
            route = unquote(split.path).replace("\\", "/")
            query = split.query

            if route.rstrip("/") in ("", "/index.html", "/dashboard"):
                self._serve_index(include_body)
            elif route.rstrip("/") == "/api/status":
                self._serve_status(include_body, fresh="fresh=1" in query)
            elif route.rstrip("/") == "/healthz":
                self._send_bytes(200, b"ok", "text/plain; charset=utf-8", include_body)
                self._note(200)
            elif route.startswith("/media/"):
                self._serve_file(REPO_ROOT, route[len("/media/"):], include_body)
            elif route.startswith("/videos/"):
                self._serve_file(VIDEO_DIR, route[len("/videos/"):], include_body)
            else:
                self._send_text(404, "Not found. This server only has /, /api/status, "
                                     "/media/<file> and /videos/<file>.", include_body)
                self._note(404)
        except PathRefused as refused:
            self._send_text(refused.status, refused.message, include_body)
            self._note(refused.status)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # The browser walked away mid-download (very normal while scrubbing a
            # video). Nothing to do, and definitely nothing to crash over.
            self._note(499, "client went away")
        except Exception:
            detail = traceback.format_exc(limit=8)
            self._log_line("unhandled error while serving " + str(self.path))
            for line in detail.strip().splitlines()[-4:]:
                self._log_line("    " + line)
            try:
                self._send_text(500, "Something went wrong on the server. "
                                     "The details are in the terminal window.",
                                include_body)
                self._note(500)
            except Exception:
                pass

    # -- individual routes -------------------------------------------------------

    def _serve_index(self, include_body: bool) -> None:
        if not INDEX_HTML.is_file():
            self._send_bytes(
                404,
                (
                    "<h1>Gray dashboard</h1>"
                    "<p>The page file <code>dashboard/index.html</code> is not there "
                    "yet. The data is ready though - open "
                    "<a href='/api/status'>/api/status</a> to see it.</p>"
                ).encode("utf-8"),
                "text/html; charset=utf-8",
                include_body,
            )
            self._note(404, "index.html missing")
            return

        body = INDEX_HTML.read_bytes()
        self._send_bytes(200, body, "text/html; charset=utf-8", include_body,
                         extra={"Cache-Control": "no-store"})
        self._note(200, f"{len(body) / 1024:.0f} kB")

    def _serve_status(self, include_body: bool, fresh: bool) -> None:
        payload, ok, cached = self.status_cache.get(force_fresh=fresh)
        self._send_bytes(
            200 if ok else 500,
            payload,
            "application/json; charset=utf-8",
            include_body,
            extra={"Cache-Control": "no-store"},
        )
        tag = "cached" if cached else "rebuilt"
        if not ok:
            tag += ", collector failed"
        self._note(200 if ok else 500, f"{len(payload) / 1024:.0f} kB {tag}")

    def _serve_file(self, base: Path, url_tail: str, include_body: bool) -> None:
        real = resolve_inside(base, url_tail)
        size = real.stat().st_size
        ctype = content_type_for(real)

        range_header = self.headers.get("Range")
        if range_header:
            try:
                rng = parse_range(range_header, size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self._note(416, "bad range")
                return
        else:
            rng = None

        if rng is None:
            self._send_file_slice(real, 0, size - 1, size, ctype, 200, include_body)
            self._note(200, f"{real.name}  {size / 1024:.0f} kB")
        else:
            start, end = rng
            self._send_file_slice(real, start, end, size, ctype, 206, include_body)
            self._note(206, f"{real.name}  bytes {start}-{end}/{size}")

    # -- low level senders --------------------------------------------------------

    def _send_file_slice(self, path: Path, start: int, end: int, size: int,
                         ctype: str, status: int, include_body: bool) -> None:
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "public, max-age=60")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        if not include_body:
            return

        remaining = length
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _send_bytes(self, status: int, body: bytes, ctype: str, include_body: bool,
                    extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def _send_text(self, status: int, message: str, include_body: bool) -> None:
        self._send_bytes(status, message.encode("utf-8"),
                         "text/plain; charset=utf-8", include_body)


class GrayServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Threaded so a large video download cannot block the status polling."""

    daemon_threads = True
    # On Windows SO_REUSEADDR lets two servers grab the same port, which would break
    # the "that port is busy, using the next one" logic. Off there, on elsewhere.
    allow_reuse_address = not sys.platform.startswith("win")


# --------------------------------------------------------------------------------------
# Start-up
# --------------------------------------------------------------------------------------

def bind_server(host: str, port: int, attempts: int = 11) -> GrayServer:
    """Try `port`, then the next ten, so a leftover server never blocks a restart."""
    busy = {errno.EADDRINUSE, errno.EACCES, 10013, 10048}   # 100xx are the Windows codes
    last_error: OSError | None = None
    for candidate in range(port, port + attempts):
        try:
            return GrayServer((host, candidate), GrayHandler)
        except OSError as exc:
            last_error = exc
            if exc.errno not in busy:
                raise
            continue
    raise SystemExit(
        f"Could not start: ports {port} to {port + attempts - 1} are all busy "
        f"({last_error}). Pick another one with --port."
    )


def banner(url: str, wanted_port: int, actual_port: int,
           collector_ok: bool, index_ok: bool) -> None:
    line = "=" * 66
    print()
    print(line)
    print("  Gray - training dashboard")
    print(line)
    print(f"  Open this in your browser:   {url}")
    if actual_port != wanted_port:
        print(f"  (port {wanted_port} was already in use, so it is on "
              f"{actual_port} instead)")
    print(f"  Repo folder:                 {REPO_ROOT}")
    print(f"  Videos folder:               {VIDEO_DIR}"
          f"{'' if VIDEO_DIR.is_dir() else '   (not created yet)'}")
    if not collector_ok:
        print("  NOTE: dashboard/collect.py is missing, so the page will show an "
              "explanation instead of data.")
    if not index_ok:
        print("  NOTE: dashboard/index.html is missing, so only /api/status works "
              "for now.")
    print(f"  Live data is rebuilt at most every {STATUS_CACHE_SECONDS:.0f} seconds.")
    print("  Press Ctrl+C to stop.")
    print(line)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local web dashboard for the Gray quadruped."
    )
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to listen on (default 8080).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Address to listen on (default 127.0.0.1, this "
                             "computer only).")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not open a browser window automatically.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print a line for every request.")
    args = parser.parse_args(argv)

    cache = StatusCache(REPO_ROOT)
    GrayHandler.status_cache = cache
    GrayHandler.quiet = args.quiet

    httpd = bind_server(args.host, args.port)
    actual_port = httpd.server_address[1]
    shown_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0", "::") else args.host
    url = f"http://{shown_host}:{actual_port}/"

    banner(url, args.port, actual_port, cache.collector_available, INDEX_HTML.is_file())

    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.\n")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
