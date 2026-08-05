"""Queue PLAN.md 1.2 batch 6 - untangle three changes that all went in at once.

    python run.py scripts/queue_12g.py --dry-run     show what it would add
    python run.py scripts/queue_12g.py               add it

WHERE THIS BATCH COMES FROM. Batch 5 ran the mix fix with and without the new
`off_track` input, on two seeds each, and both halves failed:

    m1, m2   mix fix only        turn 0.260, 0.308   crab 5.06, 5.77
    o1, o2   mix fix + off_track turn 0.269, 0.210   crab 32.5, 52.9

TWO THINGS THAT SAYS, and the second one corrects an explanation I gave.

1. `off_track` DID NOT MAKE THE ROBOT STEER BADLY. It made it nearly stop. On
   o1 the ground covered fell from 6.47 m to 2.65 and the speed error went from
   0.036 to 0.209 m/s. A drift angle is atan2(off the line, distance along it),
   so 26 and 32 degrees is what the angle does when the distance under it
   collapses. The input was handed in raw and unbounded, at a moment when
   `wandering` charges -3.0 a step for the very quantity it reports. It is
   clipped to +/-0.5 m now, ten times the free allowance.

2. RESTORING THE GENERAL POOL DID NOT BRING TURN BACK. That was the whole point
   of the mix fix, and it failed: the pool went back to 41% of draws against the
   42.5% it had before the crab share existed, and turn error stayed at 0.26 to
   0.31 rather than returning to 0.13. So the crab share does not cost turn
   accuracy by crowding turns out of the draw. It costs it some other way -
   most likely because 15% of commands now demand sideways travel with the yaw
   held at zero, and a quadruped steps sideways by placing its feet asymmetric,
   which is the same thing it does to turn.

   My earlier explanation was wrong. Exposure was the obvious reading and it did
   not survive the measurement.

THE FOUR RUNS, one seed each, all on 1301 so every number here is a paired
comparison rather than a fresh sample. This is a screen, not the gate.

    n1   crab 0.00   off_track off   wandering -3.0   THE CONTROL
    n2   crab 0.15   off_track ON    wandering -1.0
    n3   crab 0.00   off_track ON    wandering -1.0
    n4   crab 0.15   off_track off   wandering -1.0

n1 answers "is the crab share really what broke turn" - it is the draw mix as it
stood before 5 Aug, and turn should come back to about 0.13. If it does not, the
cause is something else again and nothing else in this batch means much.

n2 and n4 differ only in the input, so that pair is the input's own test, now
that it is clipped and the penalty it feeds is no longer three times what it was
when `off_track` did not exist.

n3 is the one worth hoping for: the input WITHOUT the crab share. If crab drift
comes down and turn stays at 0.13, then the input was the answer all along and
the crab share was a detour that cost a night.

WHY `wandering` DROPS TO -1.0 IN THREE OF FOUR. It was raised to -3.0 in batch 3
on a gap of 0.20 deg between corners - and the seed spread on crab drift turned
out to be over 1 deg, so that gap was never evidence. It is also the term that
charges for exactly what `off_track` reports, so the weight it wants with the
input is not the weight it wanted without it.

5000 robots and 550 iterations, as the whole of 1.2 has been.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import queue  # noqa: E402

ROBOTS = 5000
ITERATIONS = 550
SEED = 1301

# (name, crab share, off_track on, wandering weight, what it answers)
_RUNS = [
    ("n1_control", 0.0, False, -3.0,
     "the control - the draw mix and the weights as they stood before the crab "
     "share went in on 5 Aug. Turn should come back to about 0.13 from 0.26. If "
     "it does not, the crab share is not what broke it and the rest of this "
     "batch is reading the wrong variable"),
    ("n2_track", 0.15, True, -1.0,
     "the input, with the crab share kept. Read against n4, which is the same "
     "run without the input - that pair is the input's own test, now that it is "
     "clipped to +/-0.5 m and `wandering` is no longer 3x what it was when the "
     "input did not exist"),
    ("n3_track_nocrab", 0.0, True, -1.0,
     "the input WITHOUT the crab share. The one worth hoping for: if crab drift "
     "comes down and turn holds at 0.13, the input was the answer and the crab "
     "share was a detour"),
    ("n4_w1", 0.15, False, -1.0,
     "wandering at -1.0 with no input, to find out whether -3.0 was ever "
     "earned. It was chosen on a 0.20 deg gap, and the seed spread turned out "
     "to be over 1 deg"),
]

BATCH = [
    {"name": name, "seed": SEED, "crab_share": crab,
     "rewards": {"wandering": wander},
     **({} if track else {"no_off_track": True}),
     "note": f"PLAN 1.2 batch 6, {what}. crab share {crab}, off_track "
             f"{'ON' if track else 'off'}, wandering {wander}. Seed {SEED} "
             f"throughout the batch, so every row is a paired comparison. The "
             f"same seed measured turn 0.260 and crab 5.06 on the mix fix "
             f"alone, and 0.140 turn before the crab share existed at all."}
    for name, crab, track, wander, what in _RUNS
]


def measured_minutes() -> float | None:
    """How long a walk run has actually taken, off the runs on disk."""
    from datetime import datetime  # noqa: PLC0415
    try:
        from dashboard import runs  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None

    mins = []
    for r in runs.all_summaries():
        done = r.get("iterations_done") or 0
        if r.get("task") != "Gray-Walk" or done < 200:
            continue
        try:
            a = datetime.fromisoformat(r["started"])
            b = datetime.fromisoformat(r["finished"])
        except (KeyError, TypeError, ValueError):
            continue
        secs = (b - a).total_seconds()
        if secs > 0:
            mins.append(secs / 60 * (ITERATIONS / done))
    if not mins:
        return None
    mins.sort()
    return mins[len(mins) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and add nothing")
    args = ap.parse_args()

    per = measured_minutes()
    if per:
        print(f"{len(BATCH)} runs, about {len(BATCH) * per / 60:.1f} hours at "
              f"{per:.0f} min each, measured off the runs already on disk.\n")
    else:
        print(f"{len(BATCH)} runs. No finished walk run to time them against.\n")

    for job in BATCH:
        spec = {"task": "Gray-Walk", "film": True, "verify": True,
                "num_envs": ROBOTS, "iterations": ITERATIONS, **job}
        cleaned = queue._clean(dict(queue.DEFAULTS, **spec))
        print(f"  {job['name']:18} {queue.command_line(cleaned)}")
        if not args.dry_run:
            queue.add(spec)

    if args.dry_run:
        print("\nNothing added. Drop --dry-run to queue it.")
    else:
        waiting = sum(1 for j in queue.load()["jobs"]
                      if j.get("state") == "queued")
        print(f"\nQueued. {waiting} jobs waiting.")


if __name__ == "__main__":
    main()
