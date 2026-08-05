"""Queue PLAN.md 1.2 batch 5 - the mix fix, and the cross-track input.

    python run.py scripts/queue_12f.py --dry-run     show what it would add
    python run.py scripts/queue_12f.py               add it

TWO RESULTS SET THIS BATCH UP, and read together they say more than either does
alone. Both are about EXPOSURE - how often the robot is given a command:

    turn draws   CUT ~30%    turn error  0.13 -> 0.27 rad/s, all 3 seeds, bar 0.20
    crab draws   RAISED 50x  crab drift  4.69 -> 4.32 deg,   spread over 2 deg

The turn row is not subtle: take away a third of the practice and the error
doubles on every seed. So exposure works, and it works hard.

Which makes the crab row the interesting one. Fifty times the practice bought
almost nothing. That is not what a shortage of practice looks like. It is what a
MISSING INPUT looks like - the robot cannot correct an error it cannot sense, and
no amount of rehearsal teaches it to.

WHY THE TURN ROW BROKE, since it was self-inflicted. The crab share was taken out
of the general pool, which fell from about 42% of draws to 30%. Straight, crab
and standing commands all zero the yaw, so the general pool is the ONLY one that
ever asks for a turn. It now comes out of the straight share instead: 0.50 ->
0.35. Forward walking is the one with room to give, at 2.6 to 3.7 deg against a
4.0 bar. Checked on the card: straight 30.7%, crab 13.3%, general 41.0%, stop
15.1%.

WHAT `off_track` IS. The 50th input: how far off the line the robot has ended up,
and which side. Heading error and cross-track error are the same problem walking
FORWARD - it is off the line because it points wrong, and steering fixes both.
They come apart the moment it CRABS: the robot holds its heading and drifts fore
and aft, so `off_line` reads near zero while `wandering` charges up to -3.0 a
step for the distance. Corrupted to match an integrated estimate, the same way
`off_line` is - 0.034 m of error against a 0.05 m allowance, measured.

Every checkpoint before today is 49 wide and cannot be loaded into this task.

THE FOUR RUNS. Two factors would normally be a 2x2, but the mix fix is a REPAIR
and not an experiment - it puts back something that was measurably broken. So it
goes in all four, and the input is the only thing that varies:

    m1, m2      mix fix only          does turn recover to about 0.13
    o1, o2      mix fix + off_track   does crab drift close

Two seeds each, and that is thin on purpose rather than by accident: this is a
screen, not the gate. Crab drift's seed spread is over 1 deg - the 0.07 an early
pair of seeds suggested was luck - so read these as a direction, and let the gate
do the deciding on three fresh seeds afterwards.

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
SEEDS = (1301, 4507)

# What each seed scored last time, so a job carries its own before-number.
BEFORE = {1301: (4.37, 0.277), 4507: (3.21, 0.270)}

BATCH = []
for n, seed in enumerate(SEEDS):
    crab, turn = BEFORE[seed]
    BATCH.append({
        "name": f"m{n + 1}_mix_s{seed}", "seed": seed, "no_off_track": True,
        "note": f"PLAN 1.2 batch 5, mix fix only, seed {seed}. The crab share "
                f"now comes out of the straight share (0.50 -> 0.35) instead of "
                f"the general pool, which is the only pool that draws a turn. "
                f"This seed measured turn {turn} rad/s against a 0.20 bar with "
                f"the pool cut, and 0.140 before the cut. The question is "
                f"whether turn comes back. Crab was {crab} deg and is NOT "
                f"expected to move - `off_track` is switched off here, so the "
                f"policy still cannot sense the cross-track error it is fined "
                f"for.",
    })
for n, seed in enumerate(SEEDS):
    crab, turn = BEFORE[seed]
    BATCH.append({
        "name": f"o{n + 1}_track_s{seed}", "seed": seed,
        "note": f"PLAN 1.2 batch 5, mix fix AND `off_track`, seed {seed}. The "
                f"50th input: how far off the line the robot has ended up, and "
                f"which side. Read against m{n + 1}_mix_s{seed}, which is the "
                f"same run without it - that pair is the whole experiment. This "
                f"seed measured crab drift {crab} deg against a 4.0 bar. "
                f"Fifty times more crab practice barely moved that number, "
                f"while a 30% cut in turn practice doubled turn error, so the "
                f"case is that crab is short of a SIGNAL rather than short of "
                f"rehearsal.",
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
