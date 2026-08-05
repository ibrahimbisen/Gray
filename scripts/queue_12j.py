"""Queue PLAN.md 1.2 batch 9 - the sideways share, re-tried where it can be judged.

    python run.py scripts/queue_12j.py --dry-run     show what it would add
    python run.py scripts/queue_12j.py               add it

WHY RE-RUN SOMETHING ALREADY REJECTED. Because the reason it was rejected does
not survive.

The sideways share was tried on 5 Aug at 550 iterations and dropped, on two
counts: it did not improve crab drift, and it doubled turn error. Run length
then turned out to be the thing that decides whether either claim can be made at
all:

                       crab drift            turn
    550 iterations    mean 5.11, SPREAD 6.37   0.132
    1500 iterations   mean 4.95, spread 0.82   0.099

At 550 the same config on the same seed produced 2.69 one time and 9.06 another.
Nothing measured there was worth anything, and the sideways share was judged on a
gap of 0.04. At 1500 the spread is 0.82 and crab drift is a number again.

THE SECOND REASON ALSO WEAKENS. The share doubled turn error from 0.132 to about
0.27, which cleared a 0.20 bar and failed. At 1500 iterations turn converges to
0.099 with margin to spare, so the same doubling lands near 0.20 rather than well
past it. That is not a prediction - it is why the test is worth an hour.

WHAT IS AT STAKE. The reverted config now fails ONE criterion, reliably, by about
a degree: crab drift 4.60, 4.82, 5.42 against 4.0. Everything else passes. If the
sideways share closes that without pushing turn over its bar, 1.2 closes on the
bars as written. If it does not, the bar itself is the only thing left, and that
decision belongs to the owner rather than to another batch.

THE THREE RUNS. Same three seeds as the batch above, same 1500 iterations, one
change: `rel_crab_envs` from 0.0 to 0.15. A paired comparison against q1, q2 and
q3, which is the whole point of reusing the seeds.

    watch crab drift   does it drop below 4.0, on all three
    watch turn         does it stay under 0.20 - it was 0.083 to 0.113 without
                       the share, so there is about 0.09 of room
    watch forward      q3 came in at 4.13 against its own 4.0 bar, so forward
                       drift is not comfortable either at this length

About 50 minutes a run, so about 2.5 hours for the batch.
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
CRAB_SHARE = 0.15

# What each seed scored at 1500 iterations WITHOUT the sideways share, so each
# job carries the number it has to beat.
WITHOUT = {1301: (4.60, 0.100, 2.75),
           4507: (4.82, 0.113, 3.66),
           8821: (5.42, 0.083, 4.13)}

BATCH = [
    {"name": f"r{n + 1}_crab_s{seed}", "seed": seed, "iterations": ITERATIONS,
     "crab_share": CRAB_SHARE,
     "note": f"PLAN 1.2 batch 9, seed {seed}, sideways share {CRAB_SHARE} at "
             f"{ITERATIONS} iterations. Paired against q{n + 1}_long_s{seed}, "
             f"which is the identical run without the share: crab {crab} deg, "
             f"turn {turn} rad/s, forward drift {fwd} deg. The share was tried "
             f"and dropped on 5 Aug, but at 550 iterations, where the same "
             f"config on the same seed gave crab drift anywhere from 2.69 to "
             f"9.06 - so it was judged on a gap of 0.04 inside a spread of 6.4. "
             f"At 1500 the spread is 0.82 and the comparison is worth making. "
             f"Turn is the risk: the share roughly doubled it before, and 0.083 "
             f"to 0.113 doubled lands near the 0.20 bar rather than well past "
             f"it."}
    for n, (seed, (crab, turn, fwd)) in enumerate(WITHOUT.items())
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
