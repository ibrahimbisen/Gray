#!/usr/bin/env python
"""Open the dashboard.

    python run.py

Finds a Python that can actually import this project's packages and re-runs
itself under it. There are two of them, in order of preference:

  1. .venv/Scripts/python.exe - the normal case.
  2. The interpreter .venv was built from, with .venv's site-packages on the
     path. Needed because Windows Application Control sometimes refuses to run
     the small launcher executable uv puts in .venv/Scripts, while happily
     running the real interpreter it points at.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
WIN = os.name == "nt"
VENV_PY = VENV / ("Scripts/python.exe" if WIN else "bin/python")
SITE_PACKAGES = VENV / ("Lib/site-packages" if WIN else "lib/python3/site-packages")


def base_interpreter() -> Path | None:
    """The real interpreter .venv was created from, per pyvenv.cfg."""
    cfg = VENV / "pyvenv.cfg"
    if not cfg.exists():
        return None
    for line in cfg.read_text().splitlines():
        if line.split("=")[0].strip() == "home":
            home = Path(line.split("=", 1)[1].strip())
            exe = home / ("python.exe" if WIN else "python3")
            return exe if exe.exists() else None
    return None


def runs(exe: Path, env: dict | None = None) -> bool:
    try:
        return subprocess.run(
            [str(exe), "-c", "import sys"],
            capture_output=True,
            env=env,
            timeout=30,
        ).returncode == 0
    except OSError:
        return False


def relaunch() -> int | None:
    """Re-run this script under a working interpreter. None if we already are."""
    here = Path(sys.executable).resolve()

    if VENV_PY.exists() and here == VENV_PY.resolve():
        return None
    if os.environ.get("GRAY_RELAUNCHED"):
        return None  # already tried; do not loop

    env = dict(os.environ, GRAY_RELAUNCHED="1")

    if VENV_PY.exists() and runs(VENV_PY, env):
        return subprocess.call([str(VENV_PY), __file__, *sys.argv[1:]], env=env)

    base = base_interpreter()
    if base is not None:
        fallback = dict(env)
        existing = fallback.get("PYTHONPATH", "")
        fallback["PYTHONPATH"] = str(SITE_PACKAGES) + (os.pathsep + existing if existing else "")
        if runs(base, fallback):
            if VENV_PY.exists():
                print(f"note: {VENV_PY} would not run - using {base} instead.")
            return subprocess.call([str(base), __file__, *sys.argv[1:]], env=fallback)

    return None  # nothing better available; carry on with whatever we have


def main() -> int:
    code = relaunch()
    if code is not None:
        return code

    sys.path.insert(0, str(ROOT))
    try:
        from dashboard.server import monitor_state, serve
    except ImportError as exc:
        print(f"Cannot import the dashboard: {exc}")
        print(f"Running under: {sys.executable}")
        print("Try:  .venv\\Scripts\\python.exe run.py" if WIN else "Try:  .venv/bin/python run.py")
        return 1

    args = [a for a in sys.argv[1:] if a != "--check"]
    if "--check" in sys.argv[1:]:
        s = monitor_state()
        print(f"interpreter : {sys.executable}")
        print(f"runs        : {len(s['runs'])}")
        print(f"phases      : {len(s['phases'])}")
        print(f"stages      : {len(s['stages'])}")
        print(f"model       : {s['model']['passed']}/{s['model']['total']} checks pass")
        return 0

    serve(int(args[0]) if args else 8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
