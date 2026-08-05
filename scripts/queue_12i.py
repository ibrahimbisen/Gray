"""Queue PLAN.md 1.2 batch 8 - the one dial never turned: run length.

    python run.py scripts/queue_12i.py --dry-run     show what it would add
    python run.py scripts/queue_12i.py               add it

THE FINDING THAT PRODUCED THIS BATCH, and it invalidates a night of comparisons.

The reverted config was run again on the three gate seeds, to check a refactor.
Turn came back exactly as expected. Crab drift did not:

    seed    turn   was      crab   was
    1301   0.125  0.140     4.86   4.33
    4507   0.152  0.134     2.69   4.79
    8821   0.120  0.123     9.06   4.95

Same config, same seed, and crab drift moved by up to 4.1 degrees against a bar
of 4.0. Training is not bit-reproducible on this card - GPU atomics are not
associative and the ordering changes run to run - and over 550 iterations that
compounds into a different policy.

IT IS NOT THE TEST. Each verify measures its own mean to within about 1 degree:
the 64 per-robot readings behind those three numbers have standard errors of
0.37, 0.27 and 1.04. The test is fine. The POLICY is different each time.

SO EVERY CRAB COMPARISON MADE ON ONE RUN IS WORTHLESS, including the table this
project spent the night building. Turn survives it - 0.12 to 0.15 without the
sideways share against 0.25 to 0.31 with it is a separation far wider than
anything seen here, on three seeds each way. Crab drift does not survive it. The
sideways share and the cross-track input were both judged on differences of
0.04 to 1.2 degrees, inside a spread that reaches 6.4.

WHAT THAT POINTS AT. A skill the policy learns reliably does not come out at 2.7
one time and 9.1 the next. Crab is being learned unreliably, and the dial that
has never once been turned in the whole of 1.2 is HOW LONG A RUN IS. Every run
from batch 1 to batch 7 was 550 iterations, chosen on 4 Aug alongside the move
to 5000 robots. Turn is paid 5.0 and practised constantly, and it converges;
crab is rare in the draw and may simply not be finished at 550.

THE THREE RUNS. Same config, same three seeds, 1500 iterations instead of 550.
Nothing else changes, so the comparison is clean against the rows above.

    if crab drops AND tightens   it was under-trained, and length is the answer
    if crab stays wide           it is not a training-length problem, and the
                                 bar has to be judged against a spread rather
                                 than against a number

Either result settles something that has been guessed at all night. `stop_at`
stays at 0.965 - RULES.md rule 1 - so a run that reaches its ceiling early still
stops early, and the extra iterations cost nothing when they are not needed.

About 45 minutes a run at 5000 robots, so about 2.2 hours for the batch.
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

# Both readings of each seed at 550 iterations, because the pair IS the finding:
# same config, same seed, and this much apart.
AT_550 = {1301: (4.33, 4.86), 4507: (4.79, 2.69), 8821: (4.95, 9.06)}

BATCH = [
    {"name": f"q{n + 1}_long_s{seed}", "seed": seed, "iterations": ITERATIONS,
     "note": f"PLAN 1.2 batch 8, seed {seed}, {ITERATIONS} iterations instead "
             f"of 550. Nothing else changes. This seed produced crab drift "
             f"{a} deg and then {b} deg from the IDENTICAL config at 550 - "
             f"training is not bit-reproducible on this card, and 550 "
             f"iterations is apparently not long enough for the policy to "
             f"settle on a way of crabbing. Turn converges fine at 550 and "
             f"reproduced to within 0.02 rad/s, so this is specific to crab. "
             f"Watch whether crab drift comes DOWN and whether the three seeds "
             f"come closer TOGETHER - the second matters more, because a bar "
             f"cannot be judged against a number that moves by 4 deg on a "
             f"re-run. stop_at is still 0.965, so this costs nothing if the "
             f"run was already finished at 550."}
    for n, (seed, (a, b)) in enumerate(AT_550.items())
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
