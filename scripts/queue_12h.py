"""Queue PLAN.md 1.2 batch 7 - confirm the revert reproduces the best config.

    python run.py scripts/queue_12h.py --dry-run     show what it would add
    python run.py scripts/queue_12h.py               add it

WHAT HAPPENED OVERNIGHT, and why this batch is a step backwards on purpose.

Crab drift failed the gate at 4.33, 4.79 and 4.95 deg against a 4.0 bar, with
the other ten criteria passing on every seed. Three things were tried against
it. All three lost ground, measured on ONE SEED, 1301, one change per row:

    straight  crab  extra          turn    crab drift
    0.50      0.00  -              0.140      4.33     <- the best config
    0.50      0.15  -              0.277      4.37
    0.35      0.15  -              0.260      5.06
    0.35      0.00  -              0.190      5.22
    0.35      0.00  off_track      0.249      5.56

The pure-sideways share DOUBLED turn error and moved crab drift by 0.04 deg,
which is nothing against a seed spread of over 1 deg. Taking its 15% out of the
straight share rather than the general pool did not help and made crab worse.
The cross-track input made both worse still, and unbounded it nearly stopped the
robot walking - 6.47 m of ground covered down to 2.65.

So the task is back to the top row. `rel_crab_envs` and `off_track` are still in
the code, off by default, with what they measured written beside them.

WHAT THIS BATCH IS FOR. Two things, and neither is an experiment.

1. The command term was REFACTORED while all that was going on - `_resample_command`
   now splits one uniform draw between the straight and crab shares through a
   shared `_pin`. With the crab share at 0 that should behave exactly as the old
   code did, and "should" is not a measurement. If these three seeds do not
   reproduce 4.33, 4.79 and 4.95, the refactor changed something and that is a
   bug to find before anything else is decided.

2. It re-establishes the fallback cleanly. Whatever is decided about the crab
   bar, the decision should be made against numbers from today's code rather
   than against a run from before three days of edits.

The same three seeds, so every number is a paired before-and-after.

WHAT IS LEFT AFTER THIS, and it is not mine to choose. Ten of eleven criteria
pass. Crab drift is the one that does not, and weights, exposure and a new input
have each been tried and failed. That leaves the bar itself. At 0.20 m/s a
4 degree error is 0.014 m/s of unwanted fore-aft speed, while the speed bar
beside it allows 0.05 m/s - so the sideways direction is held to a standard 3.5
times tighter than the direction the robot is actually asked to travel in, and
it is the direction the robot has least authority over. That is an argument for
re-cutting the bar in the same units as the speed bar. It is the owner's call.

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

# What each seed scored on the same three criteria before the three failed
# changes went in, so a job carries its own before-numbers.
BEFORE = {1301: (4.33, 0.140, 3.17),
          4507: (4.79, 0.134, 3.37),
          8821: (4.95, 0.123, 2.95)}

BATCH = [
    {"name": f"p{n + 1}_revert_s{seed}", "seed": seed,
     "note": f"PLAN 1.2 batch 7, seed {seed}. No overrides - the task with the "
             f"crab share and the cross-track input both reverted, which is the "
             f"config that scored best on every criterion. This seed measured "
             f"crab {crab} deg, turn {turn} rad/s and forward drift {fwd} deg "
             f"before any of the three failed changes. This run should "
             f"reproduce those. It is a CHECK ON THE REFACTOR, not an "
             f"experiment: `_resample_command` was rewritten to share a `_pin` "
             f"between the straight and crab shares, and with the crab share at "
             f"0 it should behave exactly as the old code did. If these numbers "
             f"do not come back, the refactor changed something."}
    for n, (seed, (crab, turn, fwd)) in enumerate(BEFORE.items())
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
