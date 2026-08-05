"""Queue PLAN.md 1.3 batch 1 - the five dials, made wider.

    python run.py scripts/queue_13a.py --dry-run     show what it would add
    python run.py scripts/queue_13a.py               add it

WHAT 1.3 IS. Every one of these five numbers is a fact about a robot that is not
built, weighed or wired yet. The policy must not depend on any particular value
of any of them, so it trains across a band. This step makes the bands wider. It
adds no new skill - the policy reads the same 49 numbers, in a different pattern.

WHAT CHANGED, and all five moved at once:

    dial                    was            now
    foot grip           0.4  to 1.2    0.25 to 1.4
    how heavy           0.8  to 1.2    0.7  to 1.3
    where the weight is +/-15 mm       +/-25 mm
    servo strength      0.7  to 1.3    0.6  to 1.4
    gearbox drag        0.005 to 0.03  0.003 to 0.05

All five together on purpose. This batch asks ONE question - does the config
that closed 1.2 still pass in a wider world - and the cheapest answer is yes.
Only if it says no does the next batch take them one at a time to find which.
Widening them one per batch would cost five batches to learn what one batch
probably learns for free.

WHAT IS BEING CARRIED IN. The 1.2 config, unchanged: a pure sideways share of
0.15, `track_turn` at 5.0, `wandering` at -3.0, and eleven bars with crab drift
at 5.0 deg. It scored crab 3.74 and 4.25, turn 0.083 and 0.136, on a flat floor
in the narrower world.

WHY 1500 ITERATIONS. Crab drift does not settle below it. At 550 the same config
on the same seed gave 2.69 one run and 9.06 another - see 1.2.5 - so a short run
here would produce three numbers that cannot be compared with the three above.
`stop_at` stays at 0.965, so a run that reaches its ceiling early still stops.

WHAT TO WATCH. Crab drift has the least room of the eleven: it clears 5.0 by
about 0.8 deg, and a wider world is exactly the kind of thing that spends a
margin like that. Forward drift is second, at 3.7 of 4.0. Turn has room to
spare, at 0.1 of 0.2.

THE NUMBERS ABOVE ARE JUDGEMENT, and that is worth saying plainly. Nothing has
been measured on a machine that does not exist. Each is one step out from what
1.1 and 1.2 trained on, with the reasoning written beside it in
gray/tasks/walk_env_cfg.py. Batches 3 to 6 turn terrain, payload and a failed
servo into ladders with measured answers; these five do not have that yet.

5000 robots, 1500 iterations, three seeds. About 2.5 hours.
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

# Seeds never used to train a walking policy. 1301, 4507 and 8821 chose the 1.2
# config, so they cannot judge it; 99, 314 and 2718 closed 1.1.
SEEDS = (2207, 6653, 9410)

BATCH = [
    {"name": f"d{n + 1}_wide_s{seed}", "seed": seed, "iterations": ITERATIONS,
     "note": f"PLAN 1.3.1 batch 1, seed {seed}. All five world dials widened at "
             f"once - foot grip 0.25 to 1.4, mass +/-30%, centre of mass "
             f"+/-25 mm, servo strength +/-40%, gearbox drag 0.003 to 0.05. "
             f"Nothing else changes: the 1.2 config is carried in whole. The "
             f"question is whether it still passes all eleven in a wider world. "
             f"Crab drift is the row with the least room - it cleared its 5.0 "
             f"bar by about 0.8 deg on a flat floor - and a wider world is what "
             f"spends a margin like that. A fresh seed, because 1301, 4507 and "
             f"8821 chose this config and cannot judge it."}
    for n, seed in enumerate(SEEDS)
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
                "num_envs": ROBOTS, **job}
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
