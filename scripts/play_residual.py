#!/usr/bin/env python3
"""Watch a trained Phase 3 policy walk.

    python scripts/play_residual.py Gray-Residual-Flat

Same shim as train_residual.py: import train.tasks to register the task, then hand
over to mjlab's viewer. Play mode disables observation noise and the random pushes,
and runs an effectively unbounded episode, so it shows the gait rather than a
stress test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train.tasks  # noqa: F401,E402  (import registers Gray-Residual-Flat)
from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
  main()
