"""Copy mjlab's training clips into progress/runs, at the FILM_EVERY cadence.

    python tools/import_clips.py            once, over every run
    python tools/import_clips.py --watch    keep doing it while a run trains

scripts/train.py does this itself now. This exists for the case that fix cannot
cover: a run that STARTED before the fix landed is executing the old code from
memory, and cannot pick up an edit mid-flight - so its clips pile up in
logs/rsl_rl and never reach the dashboard. Point this at it and they appear.

CPU and disk only. It never opens the simulator, so it is safe to run beside
training - unlike scripts/film_checkpoints.py, which is a second process on the
graphics card and is not.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STEPS_PER_ITER = 24
FILM_EVERY = 100


def once(verbose: bool = True) -> int:
    copied = 0
    for log in (ROOT / "logs" / "rsl_rl").glob("*/*"):
        run = ROOT / "progress" / "runs" / log.name
        if not run.is_dir():
            continue
        for clip in log.glob("videos/train/rl-video-step-*.mp4"):
            m = re.fullmatch(r"rl-video-step-(\d+)", clip.stem)
            if not m:
                continue
            it = int(m.group(1)) // STEPS_PER_ITER
            if it and it % FILM_EVERY:
                continue
            out = run / "videos" / f"iter_{it:04d}.mp4"
            size = clip.stat().st_size
            # Size, not existence: the recorder writes as it goes, so a clip
            # copied mid-write would stay truncated forever.
            if size and (not out.exists() or out.stat().st_size != size):
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(clip, out)
                copied += 1
                if verbose:
                    print(f"  {log.name}  iter_{it:04d}")
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", action="store_true", help="keep going, every 60 s")
    args = ap.parse_args()
    while True:
        n = once()
        print(f"[{time.strftime('%H:%M:%S')}] copied {n}", flush=True)
        if not args.watch:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
