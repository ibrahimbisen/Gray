"""Straightness, attacked from the policy that actually walks.

    python scripts/queue_14b.py --dry-run     show what it would add
    python scripts/queue_14b.py               add it

WHY THIS BATCH REPLACES queue_14a.py. 14a bisected the 5 Aug -> 6 Aug drift
regression with from-scratch runs, because both reference runs (r1_crab at 3.72
deg, gc_1301 at 5.20) were from scratch. It cannot be run that way any more:

    b0_control, today's task, 1500 iterations from scratch, seed 1301
        distance 0.07 m      speed error 0.347 m/s      drift 17.18 deg

    gc_1301, 6 Aug task, 1500 iterations from scratch, same seed, same robots
        distance 6.18 m      speed error 0.051 m/s      drift  5.20 deg

Training looked healthy both times - reward 168 and rising, full episodes,
ground_covered normal. The robot simply learns to stand. TODAY'S TASK CANNOT
LEARN TO WALK FROM SCRATCH, and nothing noticed because every run since 6 Aug
has been a warm start: L1 continued gc, A1 continued L1, A2-A4 chained on. The
leading suspect is the speed box, widened to -0.45/0.7 after gc - the owner
found the 0.9 box teaching a freeze from the driving seat, and 0.7 may simply be
a slower version of the same trap. f1_box35 in this batch tests exactly that.

That finding does not block the night. It renames the question: from-scratch
training is broken, and every policy the owner has ever driven is a warm start,
so the straightness work continues on warm starts and loses nothing.

THE BASE IS A1_flat_1301. It is 500 iterations of flat fine-tuning on top of
L1_1301, it walks (6.29 m, speed error 0.048), and it is the run the owner drove
and complained about - drift 5.33 deg mean, 15.3 deg on its worst robot. Every
arm below is that policy plus 750 more iterations with ONE thing changed, so
every arm is comparable to the others AND to a robot the owner has felt.

    w0_control    nothing changed          separates "750 more iterations" from "the lever"
    w1_nospin     spin share 0.10 -> 0     10% of practice moved off straight travel
    w2_oldswing   swing target -> 35 mm    higher feet, less time with a foot down
    w3_nodive     nose_dived removed       shorter episodes are shorter lines to hold
    w4_veerhard   veering ramp doubled     the owner's own pick: go at heading error directly

w0 IS NOT OPTIONAL and it is the arm 14a's first attempt lacked. Without it, an
arm that improves cannot be told from a policy that simply trained longer.

w4 IS THE ONE NEW IDEA. `veering` charges heading error and is RAMPED - it only
reaches its full -2.0 at iteration 500, which is why every 500-iteration probe
so far ended at the moment straightness started to be charged properly. This arm
starts it at -2.0 and takes it to -4.0. The risk is named in advance: a robot
that is charged heavily for heading error can pay zero by not moving, so if w4
comes back with a short distance, that is the answer and not a surprise.

NO FILM. Both switches - they are two switches, and setting only `film` has cost
this project three batches at 30 s an iteration. The checkpoints are still
written, so every arm here is drivable.

Five arms at 750 warm iterations, about 22 minutes each. Then f1_box35: 1500
from scratch at the OLD speed box, which is the one test that names the freeze.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import queue  # noqa: E402

ROBOTS = 5000
WARM = 750
SEED = 1301
BASE = "2026-08-07_01-59-55_A1_flat_1301"

WHY = ("Straightness, 7 Aug 2026, continued from A1_flat_1301 - the policy the "
       "owner drove and called crooked (drift 5.33 deg mean, 15.3 deg on its "
       "worst robot). From-scratch training is broken on today's task, so "
       "every arm is a warm start, which is what every policy he has driven "
       "always was. One change per arm against w0_control.")

# even_stance off everywhere. Three failed attempts, and the owner reports the
# robot drives worse with it. It is not a variable in this batch.
OFF = {"even_stance": 0.0}

BATCH = [
    {"name": "w0_control", "rewards": OFF,
     "note": "The control: A1 plus 750 iterations, nothing changed. Tells "
             "more training apart from the lever. " + WHY},
    {"name": "w1_nospin", "rewards": OFF, "spin_share": 0.0,
     "note": "Spin share 0.10 -> 0. Moves a tenth of all practice off "
             "straight travel and onto turning on the spot, which is the "
             "skill that competes with holding a heading. " + WHY},
    {"name": "w2_oldswing", "rewards": OFF, "swing_target": 0.035,
     "note": "Swing target 50 -> 35 mm. A foot on the ground is the only "
             "thing that can correct a heading, and higher feet mean less of "
             "that. " + WHY},
    {"name": "w3_nodive", "rewards": OFF, "no_dive_ends": True,
     "note": "The nose_dived termination removed. It ends attempts early, so "
             "it changes which parts of an episode are ever practised. " + WHY},
    {"name": "w4_veerhard", "rewards": OFF,
     "ramps": {"veering": [-2.0, -2.0, -3.0, -4.0]},
     "note": "Heading error charged at full weight from step 0 and taken to "
             "-4.0, double the task's ceiling. The owner's own pick - go "
             "after the heading error directly. Named risk: a robot charged "
             "heavily for heading error can pay nothing by standing still, so "
             "read distance before drift on this one. " + WHY},
]

# The freeze test, and it is not a straightness arm. From scratch, 1500
# iterations, the only change being the speed box back to the width gc_1301
# trained in. gc walked; today's task does not. If this one walks, the box is
# the cause and the owner's speed ceiling has a measured price on it.
FREEZE = {
    "name": "f1_box35", "rewards": OFF, "speed_hi": 0.35, "speed_lo": -0.35,
    "iterations": 1500, "init_from": "",
    "note": "THE FREEZE TEST, not a straightness arm. b0_control - today's "
            "task, 1500 iterations from scratch - covered 0.07 m while "
            "gc_1301 covered 6.18 under the same seed and length on 6 Aug. "
            "The speed box widened to -0.45/0.7 in between. This run is from "
            "scratch at the old +/-0.35 box and nothing else. If it walks, "
            "the box is what broke from-scratch training, and the owner's "
            "first priority - speed - has a measured price attached to it.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and add nothing")
    args = ap.parse_args()

    print(f"{len(BATCH)} warm arms at {WARM} iterations from {BASE}, "
          f"then 1 freeze test at 1500 from scratch.\n")

    for job in [*BATCH, FREEZE]:
        spec = {"task": "Gray-Walk", "film": False, "no_video": True,
                "verify": True, "num_envs": ROBOTS, "iterations": WARM,
                "seed": SEED, "init_from": BASE, **job}
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
