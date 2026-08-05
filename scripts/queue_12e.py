"""Queue the PLAN.md 1.2 gate again, with the pure-sideways share in.

    python run.py scripts/queue_12e.py --dry-run     show what it would add
    python run.py scripts/queue_12e.py               add it

WHAT FAILED. The gate ran on three unseen seeds and failed the same single row
on all three, passing the other ten every time:

    seed    crab drift    forward drift    turn
    1301       4.33            3.17        0.140
    4507       4.79            3.37        0.134
    8821       4.95            2.95        0.123
                bar 4.0         bar 4.0    bar 0.20

Forward walking clears the same 4.0 deg bar at 2.9 to 3.4, with the same reward
terms, the same policy and the same test length. So 4.0 is reachable - crab is
not failing because the bar is wrong.

WEIGHT TUNING WAS ALREADY FINISHED as a lever. `wandering` at -3.0 was the best
of four corners in batch 3, and g4_wander6 then measured 4.98 at -6.0 - no
better. Nothing left to buy with a bigger number.

WHAT WAS ACTUALLY WRONG. The three velocities are drawn INDEPENDENTLY, so a pure
crab - sideways, no forward, no turn - needs |vx| under 0.05 out of +/-0.35 and
|wz| under 0.05 out of +/-1.0 at the same moment. That is 14% x 5%, about one
draw in 350 of the general pool. And verify.py scores crab on nothing else, at
0.20 m/s, the hard edge of the box.

The robot was graded, at the edge, on the one command it almost never got. The
gap was exposure, not ability.

THE FIX. `rel_crab_envs`, at 0.15 - a pure sideways command gets its own share of
the draws, exactly as `rel_straight_envs` has given one to a straight line since
3 Aug. Checked before spending any GPU on it: 4000 draws came out 50.5% pure
straight, 15.7% pure crab, none under the moving gate, and `_on_a_line` agreed
with all 626 - so the straightness terms really do apply to the share added to
train them.

The share comes out of the general pool, so diagonals and turns drop from about
34% of draws to 20%. Turn rate has the room - it passes at 0.12 to 0.14 against
0.20 - but it is the number to watch here, alongside the crab row.

THE SAME THREE SEEDS, on purpose. They are not spent: they failed, and the task
then changed for a reason that had nothing to do with what they measured. Re-using
them makes this a paired before-and-after on the one number in question rather
than a fresh sample that has to be compared across seeds as well as across
configs.

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

# What each seed scored on crab drift before the sideways share went in, so the
# note on the job carries its own before-number rather than sending the reader
# to a table somewhere else.
BEFORE = {1301: 4.33, 4507: 4.79, 8821: 4.95}

BATCH = [
    {"name": f"h{n + 1}_gate_s{seed}", "seed": seed,
     "note": f"PLAN 1.2 gate, seed {seed}, with rel_crab_envs at 0.15. No "
             f"overrides - the task exactly as it stands. This seed measured "
             f"{BEFORE[seed]} deg of crab drift against a 4.0 bar without the "
             f"sideways share, and passed the other ten criteria. Same seed on "
             f"purpose, so the crab row is a paired before-and-after. Watch the "
             f"turn row too: the sideways share is taken from the general pool, "
             f"which is the only share that ever gets a turn, and that pool "
             f"drops from about 34% of draws to 20%."}
    for n, seed in enumerate(BEFORE)
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
        print(f"  {job['name']:14} {queue.command_line(cleaned)}")
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
