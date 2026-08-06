"""Queue the gait confirm - the three winning levers together, three seeds.

    python run.py scripts/queue_13d.py --dry-run     show what it would add
    python run.py scripts/queue_13d.py               add it

WHAT THE PROBES SAID (batch 2b, five 550-iteration runs, seed 2207, read with
verify --ladder --seconds 30 on 64 robots - the diagnostics carry the files).

    probe        lever                swing peak     dives
                                      at 0.25/0.35
    g0_control   none                 26.4/26.9 mm   none in this window
    g1_spin      spin_share 0.10      24.4/26.1      one, at 0.45, LOCKED 29 s
    g2_dive      dive_ends            26.4/26.9      one, at 0.45, 11 STEPS
    g3_swing50   swing_target 0.05    38.4/37.9 mm   NONE, at any speed
    g4_swingw    swing_height -1.0    22.5/27.4      one at 0.35, the only
                                                     in-box FALL of the batch

THE PICKS, one line each:

    g3 IN. The one lever that moved the feet - +11 mm, past the old 35 mm
       target - and with it every dive disappeared, in and past the box.
    g2 IN. The habit it removes is the one the films show: a robot that digs
       its nose in and STAYS there. With the termination trained in, a dive
       lasts 11 steps and resolves, instead of 29 seconds of ploughing. It
       cost nothing anywhere else, and on rough ground - where clearance
       alone cannot cover a hole - it is the insurance.
    g1 IN. Turn wander 0.218 -> 0.158 m and forward drift 5.63 -> 4.42 deg
       against the control. The turn RATE did not move at 550 - whether it
       moves at 1500 is one of the things this batch reads. Nothing got
       worse, and the owner's call on 6 Aug was to train the spin.
    g4 OUT. No lift, and the only in-box fall of the batch.

CAVEAT, kept honest: the control showed no in-box dive in ITS 30 s windows,
while the 60 s ladders on f0 (narrow, 550) and d2 (wide, 1500) each caught
robots locked at -27 to -33 deg at 0.35 m/s. The dive is a sometimes-thing
per window - which is exactly why the lock-duration numbers (11 steps against
1449) are the read, not the dive counts alone.

THREE SEEDS, 1500 ITERATIONS - the confirm rules from 1.2.5: crab drift is
only measurable at 1500, and a config is only judged on seeds that did not
choose it. against the eleven criteria plus the gait numbers. If it holds,
the three levers land in the task as defaults and 1.3.2 (slopes) begins.

5000 robots, 1500 iterations, three runs. About 2.5 hours.
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

WHY = ("The gait confirm, 6 Aug 2026: spin_share 0.10 + dive_ends + "
       "swing_target 0.05 together, picked from the 2b probes. g3 lifted the "
       "feet 11 mm and removed every dive; g2 turned the 29-second dive-lock "
       "into an 11-step stumble; g1 cut turn wander 28% for free. Read all "
       "eleven criteria plus verify --gait-diag. Fresh seeds, full length, "
       "because that is what a confirm is.")

BATCH = [
    {"name": f"gc_{seed}", "spin_share": 0.10, "dive_ends": True,
     "swing_target": 0.05,
     "note": f"Seed {seed}. " + WHY}
    for seed in SEEDS
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and add nothing")
    args = ap.parse_args()

    print(f"{len(BATCH)} runs, {ITERATIONS} iterations each, seeds "
          f"{', '.join(str(s) for s in SEEDS)}.\n")

    for job in BATCH:
        spec = {"task": "Gray-Walk", "film": True, "verify": True,
                "num_envs": ROBOTS, "iterations": ITERATIONS,
                "seed": int(job["name"].split("_")[1]), **job}
        cleaned = queue._clean(dict(queue.DEFAULTS, **spec))
        print(f"  {job['name']:12} {queue.command_line(cleaned)}")
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
