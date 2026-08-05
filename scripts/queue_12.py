"""Queue batch 1 of PLAN.md step 1.2, "the whole command box".

    python run.py scripts/queue_12.py --dry-run     show what it would add
    python run.py scripts/queue_12.py               add it

WHAT THIS BATCH ASKS. One question: does the robot turn at the rate it is told
to if turning is actually paid for?

WHERE THE QUESTION CAME FROM. `scripts/verify.py` learned to test sideways and
turning on 4 Aug 2026. Until then it pinned lin_vel_y and ang_vel_z to zero and
ran forward and backward only - so two of the four things the policy is
commanded to do had been trained for two days and graded by nothing. Scored on
the three unseen-seed runs that closed 1.1, with --no-record:

    r5a  r5b  r5c
    ----------------------------------------------------------------
    0.510  0.437  0.453   rad/s of turn-rate error, commanded 1.0
    0.05   0.03   0.03    m of wander while spinning - this part is fine
    19.9   34.1   29.4    deg off the line while crabbing sideways

Half rate, on every seed. Told to turn at 1.0 rad/s it turns at about 0.5, and
the worst robot in each run is 1.15-1.32 rad/s of error, which is a robot
turning the WRONG WAY.

WHY IT UNDER-TURNS, AND WHY THIS IS THE SAME TRAP TWICE. `track_turn` pays
exp(-error^2 / std^2) and TURN_STD is 0.80. At the 0.5 rad/s error the robot
actually makes, that pays 0.68 out of 1.0 - so getting it right is worth 0.32,
against terms like `effort`, `shaking`, `joint_shock` and `rocking` that all
charge more when it turns harder. Under-turning is the cheaper answer and the
policy found it.

0.80 was itself a FIX, made when the error was 1.5 rad/s and a std of 0.30 paid
1.4e-11 - a dead term with no gradient. The file's own lesson from that day was
"a tracking band has to be set against the error the robot MAKES". The error it
makes is now 0.5, not 1.5, so the band is wrong again, in the other direction.

THE FOUR CORNERS. Two factors, both levels of each:

    turn_std       0.80 stock, 0.40 - a real slope where the robot now operates
    track_turn     1.0 stock, 3.0   - simply paying more for it

A factorial and not one-at-a-time because the two are not independent: a wider
band makes the weight matter less, so testing either alone cannot say which of
them is the thing that moved. What each pays at the current 0.5 rad/s error:

    std 0.80   err 0.5 -> 0.68,  err 0.2 -> 0.94    slope 0.26 over that range
    std 0.40   err 0.5 -> 0.21,  err 0.2 -> 0.78    slope 0.57

WHAT THIS BATCH DOES NOT TOUCH. Crab drift - 20 to 34 degrees off the line - is
NOT a tuning problem and is not in here. `wandering` and `veering`, the two
terms that hold a line, are gated on `_straight_now()`, and so is the `off_line`
observation. On a sideways command all three switch off, so the policy is being
asked to hold a line it is neither shown nor charged for missing. That is the
same fault forward walking had before it could sense its heading, it needs a
code change and an extra observation, and it is PLAN.md 1.2.1. Mixing it into
this batch would change the observation width at the same time as two reward
numbers, and nothing could be attributed to anything.

5000 robots and 550 iterations, on the owner's call.
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

# The stock levels, spelled out rather than imported. Importing walk_env_cfg
# pulls in torch, and this script is meant to run in a second - the same reason
# queue_week.py lists VEER_BASE by hand.
STD_STOCK, STD_NARROW = 0.80, 0.40
PAY_STOCK, PAY_HIGH = 1.0, 3.0

_CORNERS = [
    ("t1_stock", STD_STOCK, PAY_STOCK,
     "the control. Neither factor moved, so it is what the other three are "
     "read against"),
    ("t2_band", STD_NARROW, PAY_STOCK,
     "band only. Does a real gradient where the robot operates fix it, with no "
     "extra money on the table"),
    ("t3_pay", STD_STOCK, PAY_HIGH,
     "money only. Does paying three times as much fix it while the slope stays "
     "as flat as it is"),
    ("t4_band_pay", STD_NARROW, PAY_HIGH,
     "both. If this is the only corner that turns, the two factors need each "
     "other and neither alone is the answer"),
]

BATCH = [
    {"name": name, "turn_std": std, "rewards": {"track_turn": pay},
     "num_envs": ROBOTS, "iterations": ITERATIONS,
     "note": f"PLAN 1.2 batch 1, {what}. turn_std {std}, track_turn {pay}. "
             f"Measured on r5a/r5b/r5c, which PASSED all six straight-line "
             f"bars: told to turn at 1.0 rad/s they turn at about 0.5, with "
             f"0.44-0.51 rad/s of mean error on every seed. The new bar is "
             f"0.20 rad/s, a fifth of the command, matching what the forward "
             f"speed bar asks. Watch the six straight-line bars as well as the "
             f"turn one - paying three times as much for turning is exactly the "
             f"kind of change that buys it by giving up walking straight."}
    for name, std, pay, what in _CORNERS
]


def measured_minutes() -> float | None:
    """How long a walk run has actually taken, off the runs on disk.

    Scaled to this batch's iteration count, not a hardcoded per-run estimate. An
    estimate nobody can trace is worse than no estimate, because it gets planned
    against. Runs at a different robot count still count: the per-iteration cost
    moves with robots, and this is a rough hours figure, not a promise.
    """
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
              f"{per:.0f} min each, measured off the runs already on disk.")
        print("Verify is longer than those runs saw - four 25 s passes now, "
              "not two - so read that as a floor.\n")
    else:
        print(f"{len(BATCH)} runs. No finished walk run to time them against, "
              f"so there is no honest estimate of how long.\n")

    for job in BATCH:
        spec = {"task": "Gray-Walk", "film": True, "verify": True, **job}
        cleaned = queue._clean(dict(queue.DEFAULTS, **spec))
        print(f"  {job['name']:14} {queue.command_line(cleaned)}")
        if not args.dry_run:
            queue.add(spec)

    if args.dry_run:
        print("\nNothing added. Drop --dry-run to queue it.")
    else:
        waiting = sum(1 for j in queue.load()["jobs"]
                      if j.get("state") == "queued")
        print(f"\nQueued. {waiting} jobs waiting.")
        print("Start the runner in a second terminal:  run.bat --runner")


if __name__ == "__main__":
    main()
