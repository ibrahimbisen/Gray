#!/usr/bin/env python3
"""Start Gray's dashboard. This is the one file you need.

    python run.py

Opens the dashboard in your browser and starts TensorBoard alongside it, so the
plain-English view and the raw training charts are both up. Ctrl+C stops both.

    python run.py --port 9000      use a different port
    python run.py --no-tensorboard just the dashboard
    python run.py --no-open        do not open a browser

It re-launches itself under .venv automatically if you started it with the wrong
Python, because the system interpreter on this machine is 3.14 and the simulation
stack does not run there - which otherwise produces a confusing ImportError rather
than a useful message.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TENSORBOARD_PORT = 6006
LOG_DIR = ROOT / "logs" / "rsl_rl"


def venv_python() -> Path | None:
    """The project interpreter, if it exists."""
    candidates = [
        ROOT / ".venv" / "Scripts" / "python.exe",   # Windows
        ROOT / ".venv" / "bin" / "python",           # macOS / Linux
    ]
    return next((p for p in candidates if p.exists()), None)


def reexec_under_venv() -> None:
    """Restart under .venv unless we are already there.

    Guarded by an environment variable rather than by comparing paths, so a symlinked
    or copied interpreter cannot send this into an infinite respawn loop.
    """
    if os.environ.get("GRAY_RELAUNCHED"):
        return
    target = venv_python()
    if target is None or Path(sys.executable).resolve() == target.resolve():
        return
    env = {**os.environ, "GRAY_RELAUNCHED": "1"}
    raise SystemExit(subprocess.call([str(target), __file__, *sys.argv[1:]], env=env))


def start_tensorboard(python: str) -> subprocess.Popen | None:
    """Launch TensorBoard in the background, or explain why it did not start."""
    if not LOG_DIR.exists():
        print("  TensorBoard  skipped - no training logs yet (logs/rsl_rl/)")
        return None
    try:
        proc = subprocess.Popen(
            [python, "-m", "tensorboard.main", "--logdir", str(LOG_DIR),
             "--port", str(TENSORBOARD_PORT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT),
        )
    except OSError as exc:
        print(f"  TensorBoard  could not start: {exc}")
        return None

    time.sleep(2.0)
    if proc.poll() is not None:
        # Almost always means port 6006 is already taken by an earlier run, which is
        # fine - that instance serves the same logs.
        print(f"  TensorBoard  already running on :{TENSORBOARD_PORT} (or failed to bind)")
        return None
    print(f"  TensorBoard  http://localhost:{TENSORBOARD_PORT}   (raw training charts)")
    return proc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--no-tensorboard", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    try:
        from dashboard.server import main as serve
    except ImportError as exc:
        print(f"Could not load the dashboard: {exc}\n")
        print("The dependencies are probably not installed. From the repo root:")
        print("    uv venv --python 3.13")
        print('    uv pip install -e ".[sim,tools,dev]"')
        return 1

    print("\nGray")
    print("-" * 52)
    tb = None
    if not args.no_tensorboard:
        tb = start_tensorboard(sys.executable)

    argv = ["--port", str(args.port), "--host", args.host]
    if args.no_open:
        argv.append("--no-open")

    try:
        return serve(argv)
    except KeyboardInterrupt:
        return 0
    finally:
        if tb is not None and tb.poll() is None:
            tb.terminate()
            print("\nTensorBoard stopped.")


if __name__ == "__main__":
    reexec_under_venv()
    raise SystemExit(main())
