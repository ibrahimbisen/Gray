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

ONLY ONE OF THESE MAY RUN AT A TIME, and that is enforced here rather than trusted
to whoever launches it. Restarting run.py without stopping the previous copy left
two watchers rendering simultaneously against a live trainer, and the trainer then
died with `Warp CUDA error 700: an illegal memory access`, losing a 3000-round run
at round 108. Three processes on one consumer GPU is the condition that produced
it. The guard is a listening socket rather than a PID file because a socket cannot
go stale: if this process dies for any reason - including being killed - the OS
drops the port, whereas a PID file survives a kill and then blocks every future
start until someone deletes it by hand.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Arbitrary, in the IANA dynamic range, and never connected to - only bound. It is a
# lock, not a service.
LOCK_PORT = 49411


def acquire_single_instance_lock() -> socket.socket | None:
    """Bind the lock port, or return None if another watcher already holds it.

    The socket is returned so the caller can keep a reference: letting it be garbage
    collected would close it and silently release the lock mid-run.

    SO_REUSEADDR is deliberately NOT set. On Windows it lets two sockets bind the
    same port, which would defeat the entire purpose of this function.
    """
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", LOCK_PORT))
        lock.listen(1)
        return lock
    except OSError:
        lock.close()
        return None


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

    lock = acquire_single_instance_lock()
    if lock is None:
        # Not an error, and exit 0 on purpose: every launcher now starts a watcher
        # unconditionally, so "one is already running" is the normal outcome of
        # starting training while the dashboard is open. Failing here would make
        # train_supervised.py report a problem where none exists.
        print("[watcher] another watcher is already running; leaving it to it")
        return 0

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
