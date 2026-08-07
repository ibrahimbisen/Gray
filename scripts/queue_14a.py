"""The straightness bisect - which of our own changes cost 1.5 degrees?

    python scripts/queue_14a.py --dry-run     show what it would add
    python scripts/queue_14a.py               add it

THE FINDING THAT CAUSED THIS, read on 7 Aug 2026 out of every run.json on disk.
The owner asked whether more iterations would straighten the walk. The record
says no, twice over.

    LENGTH DOES NOT HELP
        550 iterations      drift about 3.2 deg
        1500 iterations     3.4 to 5.5 deg
        5500 iterations     6.49 deg

    AND WE MADE IT WORSE OURSELVES. Same seed 1301, same 1500 iterations, both
    from scratch:
        r1_crab, 5 Aug      3.72 deg mean,  9.8 deg worst robot
        gc_1301, 6 Aug      5.20 deg mean, 21.9 deg worst robot

The worst robot more than doubled. It is the same seed, so it is not the draw.
Between those two runs the gait confirm landed three things at once - the spin
share, the swing target, and the dive termination. Three suspects, and the batch
that landed them was measuring gait, not straightness. Nothing was watching the
thing that broke.

1500 ITERATIONS, FROM SCRATCH, AND THE FIRST ATTEMPT GOT THIS WRONG. The first
version of this batch ran 750 iterations from scratch. b0_control trained
cleanly - reward 168 and rising, full episodes, ground_covered healthy - and
then covered 0.06 m in verify. It could stand, and it could not yet walk. The
mistake was reading the A-series drift numbers as if they were comparable: A1
to A4 are 500-iteration FINE-TUNES of L1_1301, a 5500-iteration policy, and
`init_from` is the field that says so. Measured on this same card, from scratch:

        750 iterations      0.06 m covered     not a robot
        1500 iterations     6.18 - 6.44 m      a robot

So the arms are 1500 from scratch, which is exactly what r1_crab and gc_1301
were. A bisect has to reproduce the conditions of the thing it is bisecting.

WHY NOT 500, the other half of the same mistake. `veering` - the term that
charges heading error, the fault being hunted - is RAMPED, and its stages land
at iteration 0, 100, 250 and 500:

        -0.2  ->  -0.5  ->  -1.2  ->  -2.0

Every straightness probe run before this was 500 or 550 iterations, so each one
ENDED at the moment straightness started to be charged at full price. A1, A3 and
A4 landed within 0.9 deg of each other and winners were read off that; they were
three robots that had never trained under the setting they were judged on. 1500
gives 1000 iterations at full weight.

THE ARMS. Seed 1301, 5000 robots, 1500 iterations, flat ground, from scratch,
with `even_stance` switched off - the term is three failed attempts old, it
postdates the regression, and the owner reports the robot drives worse with it.
b0 is today's task with nothing else changed; each other run differs from b0 in
EXACTLY ONE THING.

    b0_control      today's task                    the number the rest are read against
    b1_nospin       spin share 0.10 -> 0            10% of practice moved off straight travel
    b2_oldswing     swing target 50 -> 35 mm        higher feet, less contact time
    b3_nodive       nose_dived removed              a dive ends the attempt

The speed box is the fourth suspect and it is NOT here. It moved after gc_1301,
so it cannot explain a regression measured before it, and it is the one change
the owner actively wants - speed is his first priority. It gets its own batch,
where a slower box is a cost to weigh rather than a bug to remove.

NO FILM. Both switches, because they are two switches and setting only `film`
has cost this project three separate batches at 30 s an iteration. The
checkpoints are still written, so any of these four can still be driven.

WHAT COUNTS AS AN ANSWER. Drift mean AND the worst robot, from the run's own
verify. The owner drives ONE robot, so the worst is the number that matches what
he feels; the mean has been flattering us for days because the robots scatter
both ways and cancel. If no arm beats b0 by more than the seed noise, none of
the three is the cause, and the `veering` ramp itself is the next place to look.

5000 robots, 1500 iterations, four runs. About two hours fifty.
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
SEED = 1301

WHY = ("Straightness bisect, 7 Aug 2026. Forward drift went 3.72 -> 5.20 deg "
       "and the worst robot 9.8 -> 21.9 deg between r1_crab (5 Aug) and "
       "gc_1301 (6 Aug) - same seed, same length, both from scratch. Three "
       "levers landed together in between. Each run here differs from "
       "b0_control in exactly one of them. 1500 iterations from scratch "
       "because that is what both reference runs were, and because 750 from "
       "scratch covers 0.06 m - measured, on the first attempt at this batch.")

# even_stance off in all four. It postdates the regression, so leaving it in
# would put a fourth difference between this batch and the 5 Aug reference it
# is compared against - and the owner reports the robot drives worse with it.
OFF = {"even_stance": 0.0}

BATCH = [
    {"name": "b0_control", "rewards": OFF,
     "note": "The control. Today's task, even_stance off, nothing else "
             "changed. Every other run in the batch is read against this "
             "one. " + WHY},
    {"name": "b1_nospin", "rewards": OFF, "spin_share": 0.0,
     "note": "Spin share 0.10 -> 0. The suspect I would bet on: it moves a "
             "tenth of all practice off straight travel and onto turning on "
             "the spot, which is the skill that competes with holding a "
             "heading. " + WHY},
    {"name": "b2_oldswing", "rewards": OFF, "swing_target": 0.035,
     "note": "Swing target 50 -> 35 mm. Higher feet mean less time with a "
             "foot on the ground, and a foot on the ground is the only thing "
             "that can correct a heading. " + WHY},
    {"name": "b3_nodive", "rewards": OFF, "no_dive_ends": True,
     "note": "The nose_dived termination removed. It ends attempts early, so "
             "it changes which parts of an episode the policy ever practises "
             "- and a shorter episode is a shorter line to hold. " + WHY},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and add nothing")
    args = ap.parse_args()

    print(f"{len(BATCH)} runs, {ITERATIONS} iterations each, seed {SEED}, "
          f"{ROBOTS} robots, from scratch, no film.\n")

    for job in BATCH:
        spec = {"task": "Gray-Walk", "film": False, "no_video": True,
                "verify": True, "num_envs": ROBOTS, "iterations": ITERATIONS,
                "seed": SEED, **job}
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
