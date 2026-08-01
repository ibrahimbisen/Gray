#!/usr/bin/env python3
"""Render a video for each new checkpoint, while training is still running.

    python scripts/render_watcher.py

Training saves a checkpoint every 50 rounds. Left alone, videos only got made once
everything had finished, so the dashboard gallery sat empty for two hours and there
was no way to see what the robot could currently do. This watches for new
checkpoints and renders each one shortly after it appears, so the newest clip is
always roughly "the robot, now".

run.py starts this automatically. It is deliberately a thin loop around
scripts/make_progress_videos.py rather than a second copy of that logic - that
script already skips clips it has already rendered, so re-running it is cheap and
there is only one place where a rollout is defined.

GPU cost is small but not zero: one 12 s clip renders in a few seconds, against a
checkpoint every ~2 minutes. --interval trades freshness for that cost, and
--nice-seconds keeps it from rendering a burst of backlog at full tilt while
training is competing for the card.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def count_checkpoints() -> int:
    return len(list((ROOT / "logs" / "rsl_rl" / "gray_residual").glob("*/model_*.pt")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=90.0,
                    help="seconds between checks (default 90)")
    ap.add_argument("--nice-seconds", type=float, default=3.0,
                    help="pause between clips, to leave the GPU to training")
    ap.add_argument("--once", action="store_true", help="render what is pending, then exit")
    args = ap.parse_args()

    python = sys.executable
    script = str(ROOT / "scripts" / "make_progress_videos.py")
    seen = -1

    print(f"[watcher] watching for new checkpoints every {args.interval:.0f}s")
    while True:
        try:
            found = count_checkpoints()
            if found != seen:
                seen = found
                # make_progress_videos skips anything already rendered, so this only
                # does work when there is genuinely something new.
                subprocess.run([python, script], cwd=str(ROOT), check=False)
                time.sleep(args.nice_seconds)
        except Exception as exc:                       # never let the loop die
            print(f"[watcher] {exc!r}")

        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[watcher] stopped")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
