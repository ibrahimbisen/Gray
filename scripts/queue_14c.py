"""What to do about it: the centre of mass, the speed, and the hills.

    python scripts/queue_14c.py --dry-run     show what it would add
    python scripts/queue_14c.py               add it

THE MEASUREMENT THIS BATCH ANSWERS TO, taken 7 Aug 2026 on w0_control with
verify --gait-diag and no training at all. Same policy, same test, one dial left
free at a time:

    dial left free            drift mean    worst robot
    ------------------------  ----------    -----------
    none - every body alike       2.60 deg      5.1 deg
    foot grip                     2.86          6.2
    gearbox drag                  2.99          6.1
    servo strength                2.98          9.3
    how heavy                     3.54          7.8
    WHERE THE WEIGHT IS           4.90         11.4
    all five                      4.77         12.2

The centre of mass alone reproduces the whole spread. The other four together
add almost nothing. That is why five reward attempts in two days moved nothing:
the robot is not walking crookedly, it is compensating - differently every
attempt - for a trunk balance point it is never told. A reward term cannot fix
an input the policy has not got.

AND IT IS A CONSTANT, ON THE REAL ROBOT. With one fixed body, 63 of 64 robots
lean the SAME way, sd 1.2 to 1.5 deg. The real robot has one centre of mass, so
its error is a fixed lean and not a wander. A fixed lean trims out.

    w4_veerhard   4.28 deg mean, 11.26 worst, distance 6.42 m held
    w0_control    5.48            14.06

w4 is the owner's own pick - charge heading error directly - and the best of the
five arms. Its named risk did not happen: a robot charged -4.0 for heading error
could have paid nothing by standing still, and this one kept its distance.

THE FIVE RUNS.

    n1_narrowcom   the centre-of-mass draw put back to the narrow range, in
                   TRAINING. The one lever the measurement points at. The real
                   robot has one balance point and the owner is remodelling the
                   CAD with the pot mounts in it, so a tight range is a fact
                   about the robot rather than a cheat. If this halves the
                   worst robot, the straightness question is answered.
    n2_veer4507    w4_veerhard again on a second seed. One seed is not a
                   result, and this project has read winners off single seeds
                   three times this week.
    n3_both        veerhard AND the narrow centre of mass. If both work they
                   should stack - one removes the disturbance, the other pays
                   more for correcting what is left.
    s1_fast        the speed box to 0.9 m/s, which is 3.2 km/h. The owner's
                   FIRST priority. 0.9 taught a from-scratch policy to freeze
                   once already, so this is warm-started from a policy that
                   walks and the read is DISTANCE before anything else.
    t1_mixed       every ground at once, warm. The hills the owner asked for.

All warm starts from A1_flat_1301 at 750 iterations, about 22 minutes each -
except the speed and ground arms, which change what the robot is asked to do
rather than how it is scored, and get the same length for comparability.

NO FILM, both switches. The checkpoints are still written and still drivable.
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
VEER = [-2.0, -2.0, -3.0, -4.0]

WHY = ("7 Aug 2026. Measured with verify --vary-only: the centre-of-mass draw "
       "alone reproduces the whole heading spread (4.90 deg mean, 11.4 worst) "
       "while every body alike gives 2.60 and 5.1. The other four dials add "
       "almost nothing. w4_veerhard, charging heading error to -4.0, was the "
       "best of five reward arms at 4.28 and 11.26 with distance held.")

OFF = {"even_stance": 0.0}

BATCH = [
    {"name": "n1_narrowcom", "rewards": OFF,
     "narrow_dials": "where_the_weight_is",
     "note": "The centre-of-mass draw back to the narrow range, in training. "
             "The one lever the measurement points at. " + WHY},
    {"name": "n2_veer4507", "rewards": OFF, "ramps": {"veering": VEER},
     "seed": 4507,
     "note": "w4_veerhard on a second seed. One seed is not a result, and "
             "this week has read winners off single seeds three times. " + WHY},
    {"name": "n3_both", "rewards": OFF, "ramps": {"veering": VEER},
     "narrow_dials": "where_the_weight_is",
     "note": "Both together - remove the disturbance AND pay more for "
             "correcting what is left. They should stack. " + WHY},
    {"name": "s1_fast", "rewards": OFF, "speed_hi": 0.9,
     "note": "The speed box to 0.9 m/s - 3.2 km/h, the owner's first "
             "priority. 0.9 taught a from-scratch policy to freeze once, so "
             "this is warm-started from a policy that already walks. READ "
             "DISTANCE FIRST: if it drops, the box is past what this robot "
             "can do and 0.7 stands. " + WHY},
    {"name": "t1_mixed", "rewards": OFF, "mixed_ground": True,
     "note": "Every ground at once - flat, hills, rough, waves - warm. The "
             "hills the owner asked for, on top of the flat policy rather "
             "than instead of it. " + WHY},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and add nothing")
    args = ap.parse_args()

    print(f"{len(BATCH)} warm arms at {WARM} iterations from {BASE}.\n")

    for job in BATCH:
        spec = {"task": "Gray-Walk", "film": False, "no_video": True,
                "verify": True, "num_envs": ROBOTS, "iterations": WARM,
                "seed": SEED, "init_from": BASE, **job}
        cleaned = queue._clean(dict(queue.DEFAULTS, **spec))
        print(f"  {job['name']:13} {queue.command_line(cleaned)}")
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
