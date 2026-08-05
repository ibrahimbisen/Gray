"""Queue batch 4 of PLAN.md step 1.2 - close the gate on unseen seeds.

    python run.py scripts/queue_12d.py --dry-run     show what it would add
    python run.py scripts/queue_12d.py               add it

WHAT IS BEING CLOSED. All eleven criteria, on seeds the config has never been
trained with. Same shape as 1.1.5, which closed 1.1 on seeds 99, 314 and 2718 -
and the same reason: a config that passes on the seed it was tuned with has
shown one result, not a property.

WHERE THE CONFIG CAME FROM. Two measured changes, both now in
gray/tasks/walk_env_cfg.py rather than passed as overrides, so these runs test
the task AS IT STANDS:

    track_turn   1.0 -> 5.0     batch 2, 8 runs. Turn error 0.476 -> 0.141
                                against a 0.20 bar, confirmed at 0.135 on a
                                second seed. The band was not the problem:
                                0.40 and 1.20 were both worse than 0.80.

    wandering   -1.0 -> -3.0    batch 3, 5 runs. Best of the four corners on
                                crab drift, forward drift AND turn error at
                                once. Heavier `veering` made all three worse.

plus the code change that made batch 3 possible at all: `_on_a_line` accepts
sideways travel, so `veering`, `wandering` and the `off_line` input apply to a
crab command, and `wandering` measures perpendicular to the COURSE rather than
the facing. That alone took crab drift from 19.5 deg to 3.7 - the policy was
never bad at crabbing, it was never told which way it was off.

THE MARGIN THIS IS TESTING. c3_wander passed at 3.53 deg against a 4.0 bar.
That is 0.47 of room on a number whose run-to-run noise is 0.07, measured off
two seeds of the stock corner. Comfortable, but measured on ONE seed of this
exact config, which is precisely what three unseen seeds are for.

THE FOURTH RUN IS NOT PART OF THE GATE. `wandering` improved every straightness
number when it went from -1.0 to -3.0, and nothing has tested past -3.0, so the
trend has no known end. -6.0 says whether it keeps going. It is deliberately
NOT in the gate: 1.2 closes on passing, and finding the best value is 1.3's
sweep. What this buys is the range that sweep should cover, for one run.

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

# Seeds the walking config has never been trained with. 1.1 closed on 99, 314
# and 2718, so those are used up - a "fresh" seed that already picked a winner
# once is not fresh.
SEEDS = (1301, 4507, 8821)

# RUN TWICE. The first attempt got one seed in - g1_close_s1301, which passed
# all eleven - and was then stopped on purpose, because the owner spotted the
# robot walking slightly nose-down in the checkpoint films and that turned out
# to be a sign error in POSE_PITCH. Nose down is NEGATIVE, measured in the sim;
# three comments in the task said positive, and the commanded range had been
# written to match them. It asked for up to 15 deg of nose-down into 10 deg of
# available travel, never asked for 12 of the 20 deg of nose-up it has, and put
# the average draw 3.4 deg nose-down.
#
# Finishing the old gate would have certified a task about to change by 23 deg
# of commanded pitch, so it was cancelled with two runs to go rather than spend
# an hour proving something that could not carry forward. g1's pass still says
# the reward config generalises past its tuning seed; it does not say the task
# was right.

BATCH = [
    {"name": f"g{n + 1}_close_s{seed}", "seed": seed,
     "note": f"PLAN 1.2 gate, seed {seed}. No overrides - the task exactly as "
             f"it stands, so what passes here is the config and not a command "
             f"line. All eleven criteria: the six that closed 1.1, three for "
             f"crabbing and two for turning. The reference is c3_wander, which "
             f"passed all eleven at crab drift 3.53 deg of 4.0, forward 2.96 "
             f"of 4.0 and turn 0.106 of 0.20 - on one seed. Seeds 99, 314 and "
             f"2718 are used up; they closed 1.1."}
    for n, seed in enumerate(SEEDS)
]

BATCH.append({
    "name": "g4_wander6", "rewards": {"wandering": -6.0},
    "note": "PLAN 1.2, NOT part of the gate - read it as a measurement. "
            "`wandering` improved crab drift, forward drift and turn error all "
            "at once going from -1.0 to -3.0, and nothing has tested past -3.0, "
            "so the trend has no known end. This says whether it keeps going. "
            "Deliberately outside the gate because 1.2 closes on PASSING and "
            "finding the best value is 1.3's sweep - what this buys, for one "
            "run, is the range that sweep should cover.",
})


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
