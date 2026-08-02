#!/usr/bin/env python
"""Open the dashboard.

    python run.py

Re-runs itself inside .venv if you started it with the wrong Python, because the
system Python on this machine is 3.13/3.14 and does not have the packages.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main() -> int:
    if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve():
        return subprocess.call([str(VENV_PY), __file__, *sys.argv[1:]])

    sys.path.insert(0, str(ROOT))
    from dashboard.server import serve

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    serve(port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
