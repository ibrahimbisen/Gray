"""Queue PLAN 1.3.2 batch 4 - the slope confirm, at the angle the ladder picked.

    python run.py scripts/queue_13f.py --dry-run     show what it would add
    python run.py scripts/queue_13f.py               add it

WHAT THE LADDER SAID. Four rungs, 550 warm iterations each, measured on
their own hills (64 robots, 30 s, four directions):

    angle   alive    walked    ground_covered   worst pass
     4 deg   97%     2.35 m        0.56         backward
     7 deg   95%     3.60 m        0.50         backward
    10 deg   88%     1.64 m        0.40         sideways, 8 of 64 down
    14 deg   97%     1.39 m        0.37         turning

IT NEVER STOPS WALKING - it stops CLIMBING. Nothing falls off the hill;
distance degrades smoothly and the worst direction is always backward or
sideways, never forward. The cliff this batch was meant to find is past
14 degrees, or past 550 iterations, and a ladder at one run length cannot
tell those apart.

10 DEGREES, and the reason is strain rather than failure: it is the only
rung under 90% alive, on a sideways traverse, which is where a longer run
has something to buy. 18% grade is more than any ground a 3.1 kg robot
gets asked to work on. 14 deg reads EASIER than 10 at this length, which
is a reason to distrust 550-iteration terrain numbers, not a reason to
believe 14 is safe.

WHAT THIS BATCH ASKS. Whether the eleven criteria - measured on the FLAT,
which is where the bar lives - survive a policy hardened on a 10 degree
hill, and whether the hill numbers themselves keep improving with three
times the training. Three fresh seeds, 1500 iterations, warm-started from
the flat winner exactly as the ladder was.

FILMS BACK ON. The ladder ran film-free because a heightfield frame costs
about 1.4 s and it only needed numbers. Three runs is a small enough
render bill for the thing that closes a step, and nobody should land a
terrain policy without watching it walk.

5000 robots, 1500 iterations, three runs. About 3 hours with film.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import queue  # noqa: E402

ROBOTS = 5000
ITERATIONS = 1500
SEEDS = (1301, 4507, 8821)
SLOPE_DEG = 10.0
INIT_FROM = "2026-08-06_04-43-00_gc_1301"

WHY = (f"PLAN 1.3.2 batch 4, the slope confirm at {SLOPE_DEG:g} deg - the "
       f"angle the ladder picked as the one that strains without breaking "
       f"(88% alive, 8 of 64 down on a sideways traverse; every other rung "
       f"cleared 95%). Warm-started from the flat winner {INIT_FROM}, like "
       f"every rung. Read the eleven criteria on the FLAT, where the bar "
       f"lives, and the hill numbers with verify --gait-diag --slope-deg "
       f"{SLOPE_DEG:g}. Films on: three runs is a small render bill for the "
       f"thing that closes a step.")

BATCH = [
    {"name": f"sc_{seed}", "slope_deg": SLOPE_DEG, "init_from": INIT_FROM,
     "note": f"Seed {seed}. " + WHY}
    for seed in SEEDS
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and add nothing")
    args = ap.parse_args()

    print(f"{len(BATCH)} runs, {ITERATIONS} iterations each, "
          f"{SLOPE_DEG:g} deg, seeds "
          f"{', '.join(str(s) for s in SEEDS)}.\n")

    for job in BATCH:
        spec = {"task": "Gray-Walk", "film": True, "verify": True,
                "num_envs": ROBOTS, "iterations": ITERATIONS,
                "seed": int(job["name"].split("_")[1]), **job}
        cleaned = queue._clean(dict(queue.DEFAULTS, **spec))
        print(f"  {job['name']:10} {queue.command_line(cleaned)}")
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
