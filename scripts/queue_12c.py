"""Queue batch 3 of PLAN.md step 1.2 - straightness, now that crab steps have it.

    python run.py scripts/queue_12c.py --dry-run     show what it would add
    python run.py scripts/queue_12c.py               add it

WHAT CHANGED IN THE CODE FIRST. Until 4 Aug 2026 `_straight_now()` required the
commanded travel to point FORWARD - forward speed above the gate, sideways
below it. So on a crab command all three of the terms that hold a line switched
off at once:

    veering            not charged for turning off the line
    wandering          not charged for ending up off it
    off_line           not even TOLD which way it was off

The policy was asked to crab in a straight line while being neither shown its
error nor charged for missing it. It came in 19 to 41 degrees off, against 3.6
degrees walking forwards - and forward walking had been in exactly that state
until it was given a heading to steer by on 4 Aug.

The gate is now "moving in some direction, and not told to turn". Two changes go
with it. `wandering` rotates by the COURSE rather than the facing, because on a
crab step those are 90 degrees apart and the old version measured drift along
the very axis the robot had been told to travel down - charging a perfect crab
step for the whole distance it was asked to cover. And the line's course is
pinned alongside its heading, so a robot that drifts round is still measured
against the line it was originally sent along.

The observation count does NOT change. `off_line` was one number and still is;
it is simply non-zero on more of the draws, so every existing checkpoint still
loads.

WHAT THIS BATCH ASKS. The fix hands the policy a signal and a penalty it has
never had on crab commands. The question is whether the stock weights are the
right size for a job they have never been asked to do:

    veering     -0.2 stock, -0.6
    wandering   -1.0 stock, -3.0

A 2x2 rather than one at a time, for the same reason 1.1 round 1 was a
factorial: these two terms have pulled against each other before. `wandering`
charges for where the robot ENDED UP and `veering` for which way it is POINTING,
and the cheapest way to stop paying one has historically been to do the thing
the other charges for.

Plus a straight repeat of the stock corner on a second seed, because the whole
batch is read on a number - crab drift - whose run-to-run noise has never been
measured. Forward drift's is 0.72 deg. Nothing says crab's is the same, and the
gap between corners here may well be smaller than the gap this pair shows.

WHAT TO WATCH. Forward drift has been sitting on its 4.0 deg bar - 4.12, 4.31,
4.19 across batches 1 and 2, passing and failing on luck. Both these weights
should help it, and if they do not while crab improves, the two directions are
being traded against each other and the answer is not a bigger weight.

A SMOKE TEST GOES FIRST because the task changed: a gate, a rotation and a
pinned course, all in the reward path. Three iterations says whether it builds,
draws a command, steps and scores, in about a minute rather than 20 minutes from
now.

track_turn is 5.0 in the task config since batch 2, so every run here inherits
the turn fix rather than overriding it.

5000 robots and 550 iterations, as before.
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
# `veering` IS NOT A FLAT WEIGHT. It is driven by `ease_in_straightness`, a
# four-stage curriculum that lands at roughly iterations 0, 100, 250 and 500 -
# so its weight at the end of a run is -2.0, not the -0.2 it starts at. It is
# ramped last and slowest on purpose: charging for heading drift before the
# robot can hold a direction just fines it for falling over in a way that
# happens to rotate it.
#
# train.py REFUSES `--reward veering=`, because a curriculum would overwrite it
# within seconds and the run would record a weight it never trained on. That
# refusal killed the first attempt at this batch at launch, before any GPU time,
# which is the check doing exactly its job. Use --ramp.
#
# The hard level is 2x the stock ramp - the same VEER_HARD that 1.1's round 1
# factorial used, so the two batches can be read against each other.
VEER_STOCK = [-0.2, -0.5, -1.2, -2.0]
VEER_HARD = [-0.4, -1.0, -2.4, -4.0]
# `wandering` is flat, so it stays a plain weight override.
WANDER_STOCK, WANDER_HARD = -1.0, -3.0

SMOKE = {
    "name": "c00_smoke", "task": "Gray-Walk", "iterations": 3,
    "num_envs": 512, "film": False, "verify": False,
    "note": "Smoke test, 3 iterations. Not an experiment - it only has to build "
            "the task, draw a command, step and score. Worth a job of its own "
            "because the TASK changed: `_on_a_line` replaced `_straight_now` "
            "and now accepts sideways travel, `wandering` rotates by a pinned "
            "course instead of the facing, and both feed the reward path. If "
            "any of that is broken this says so in a minute.",
}

_CORNERS = [
    ("c1_fix", VEER_STOCK, WANDER_STOCK, None,
     "the control - the crab fix with the weights exactly as 1.1 left them. "
     "Everything else is read against this"),
    ("c2_veer", VEER_HARD, WANDER_STOCK, None,
     "veering ramp x2. The learnable half of straightness: the policy can SEE "
     "its heading error, so this is the term it can actually act on"),
    ("c3_wander", VEER_STOCK, WANDER_HARD, None,
     "wandering x3. Charges for where it ended up, which the robot cannot "
     "sense directly - it can only steer, so this asks more of less information"),
    ("c4_both", VEER_HARD, WANDER_HARD, None,
     "both up. If only this corner works the two need each other; if it is "
     "worse than either alone they are fighting, which they have before"),
    ("c5_fix_s7", VEER_STOCK, WANDER_STOCK, 7,
     "IDENTICAL to c1_fix but seed 7. Not an experiment - it is the noise floor "
     "for crab drift, which has never been measured on any run"),
]

BATCH = [
    {"name": name, "ramps": {"veering": veer}, "rewards": {"wandering": wander},
     **({"seed": seed} if seed else {}),
     "note": f"PLAN 1.2 batch 3, {what}. veering ramp {veer}, wandering "
             f"{wander}. "
             f"First runs with the crab-straightness fix: `_on_a_line` accepts "
             f"sideways travel, so `veering`, `wandering` and the `off_line` "
             f"input apply to a crab command for the first time, and "
             f"`wandering` measures perpendicular to the COURSE rather than the "
             f"facing. Crab drift was 19.5 to 20.9 deg against a 4.0 bar on the "
             f"batch 2 winners. Watch forward drift too - it sits on its bar at "
             f"4.1 to 4.3 and these weights should help it, not cost it."}
    for name, veer, wander, seed, what in _CORNERS
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
    ap.add_argument("--no-smoke", dest="smoke", action="store_false",
                    help="skip the 3-iteration check in front of the queue")
    args = ap.parse_args()

    jobs = ([SMOKE] if args.smoke else []) + BATCH

    per = measured_minutes()
    if per:
        print(f"{len(BATCH)} runs, about {len(BATCH) * per / 60:.1f} hours at "
              f"{per:.0f} min each, measured off the runs already on disk"
              + (", plus a 1-minute smoke test in front.\n" if args.smoke
                 else ".\n"))
    else:
        print(f"{len(jobs)} jobs. No finished walk run to time them against.\n")

    for job in jobs:
        spec = {"task": "Gray-Walk", "film": True, "verify": True,
                "num_envs": ROBOTS, "iterations": ITERATIONS, **job}
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
