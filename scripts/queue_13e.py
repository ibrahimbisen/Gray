"""Queue PLAN 1.3.2 batch 3 - the slope ladder. At what angle does it stop walking?

    python run.py scripts/queue_13e.py --dry-run     show what it would add
    python run.py scripts/queue_13e.py               add it

THE WORLD. --slope-deg replaces the flat floor with a 16 m pyramid at one
angle, every robot on its own copy, spawned on the flat 1.5 m platform at the
top. Commands resample every 5-10 s, so an episode works uphill, downhill and
across. The height terms and `collapsed` read the ground by raycast since
6 Aug, so they mean the same thing on the hill as on the floor - without
that, walking one metre downhill read as a collapse.

THE LADDER. Four angles, one per run, seed 2207, 550 iterations, everything
else the task as it stands - which since 6 Aug includes the gait winners:
swing target 50 mm, the nose_dived termination, the 10% spin share.

    run          angle    grade
    s1_4deg       4 deg    7%     a ramp a wheelchair can take
    s2_7deg       7 deg   12%     a steep street
    s3_10deg     10 deg   18%     a steep driveway
    s4_14deg     14 deg   25%     a serious hill

Why these four: this is a 3.1 kg robot whose knee servos idle at 55% of
stall just standing, with 50 mm of trained foot clearance and 0.35 m/s of
top speed. go1-class robots train to ~25 deg; Gray is not go1-class. If
s4 walks, a second ladder can climb higher; if s1 fails, the gait batch
gets reopened before any more terrain is bought.

THE READ, per run: falls and ground_covered from training metrics (a robot
that cannot hold the hill says so in the fell_over column), then verify
--gait-diag --slope-deg N on the checkpoint for swing peaks, dives and
signed pitch ON the hill. The stop row picks the angle batch 4 holds.

550 AND ONE SEED, per the ladder rule: raise one number until it fails,
write down where. The confirming runs at 1500 x 3 seeds come after, at the
picked angle.

5000 robots, 550 iterations, four runs. About 1.3 hours.
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
SEED = 2207
ANGLES = (4.0, 7.0, 10.0, 14.0)

WHY = ("PLAN 1.3.2 batch 3, the slope ladder: one angle per run, everything "
       "else the task as it stands with the gait winners landed. Read falls "
       "and ground_covered at 550, then verify --gait-diag --slope-deg on "
       "the checkpoint. The stop row picks the angle batch 4 holds at "
       "1500 x 3 seeds.")

BATCH = [
    {"name": f"s{n + 1}_{int(deg)}deg_s{SEED}", "slope_deg": deg,
     "note": f"Rung {n + 1} of 4: a {deg:g} degree slope, about "
             f"{deg * 1.75:.0f}% grade. " + WHY}
    for n, deg in enumerate(ANGLES)
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and add nothing")
    args = ap.parse_args()

    print(f"{len(BATCH)} runs, {ITERATIONS} iterations each, seed {SEED}, "
          f"angles {', '.join(f'{a:g}' for a in ANGLES)} deg.\n")

    for job in BATCH:
        spec = {"task": "Gray-Walk", "film": True, "verify": True,
                "num_envs": ROBOTS, "iterations": ITERATIONS, "seed": SEED,
                # Off for the same reason every ladder turns it off: an early
                # stop would make run length a second variable between rungs.
                "stop_at": 0.0, **job}
        cleaned = queue._clean(dict(queue.DEFAULTS, **spec))
        print(f"  {job['name']:16} {queue.command_line(cleaned)}")
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
