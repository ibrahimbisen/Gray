"""Queue batch 2 of PLAN.md step 1.2 - how far does paying for turning go?

    python run.py scripts/queue_12b.py --dry-run     show what it would add
    python run.py scripts/queue_12b.py               add it

WHAT BATCH 1 SAID. Four corners, turn_std crossed with track_turn's weight,
scored on mean turn-rate error against a 1.0 rad/s command. The bar is 0.20.

    band   pay    turn error
    -----------------------------------------------------------------
    0.80   1.0    0.476    t1_stock, the control
    0.40   1.0    0.736    t2_band      - WORSE, and it failed crab distance
    0.80   3.0    0.251    t3_pay       - best of the four
    0.40   3.0    0.323    t4_band_pay

The money is what moved it. The band did not, and narrowing it alone made
turning WORSE - which is the opposite of the hypothesis batch 1 was built on,
and worth writing down rather than quietly dropping.

Why narrowing backfired, most likely: the weight sets how much the term is worth
in total, and the band only sets where its slope is. Shrinking the band at
weight 1.0 shrinks the reward actually collected across the whole region the
robot operates in, so the term contributes LESS overall and the penalties that
charge for turning hard - effort, shaking, joint_shock, rocking - win by more
than they did before. Slope is worth nothing if there is no money on it.

WHAT THIS BATCH ASKS. t3 came in at 0.251 against a 0.20 bar, so the answer is
close. Two questions, both about how far the same lever goes:

    does more money keep working          pay 5.0 and 8.0 at the stock band
    is a WIDER band better still          band 1.20 at pay 5.0

and one about whether any of this is real:

    how noisy is the turn number          a straight repeat on a second seed

That last one is not padding. Every turn-rate number on record is a single seed.
The 0.476 -> 0.251 gap is large enough to believe, but 0.251 vs 0.20 is not, and
the next decision is made on exactly that margin. 1.1 measured its noise floor
before it compared anything; this is the same rule applied to a new metric.

WHAT TO WATCH BESIDES THE TURN ROW. t3 failed forward drift at 4.124 deg against
4.0 - marginal, and inside the 0.72 deg run-to-run noise, so not yet a finding.
But paying three times as much for turning is exactly the change that buys
turning by giving up walking straight, and 5.0 and 8.0 push harder on the same
lever. If forward drift walks up with the weight across these four, that is the
trade showing itself and the answer is not more money.

Crab drift is untouched here and is expected to stay at 29-41 deg. It is not a
tuning problem - `wandering`, `veering` and the `off_line` input are all gated
on `_straight_now()`, so a sideways command is asked to hold a line it is
neither shown nor charged for missing. That is PLAN.md 1.2.3 and it needs code.

5000 robots and 550 iterations, same as batch 1, so the two can be read together.
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
STD_STOCK, STD_WIDE = 0.80, 1.20

BATCH = [
    {"name": "t5_pay5", "turn_std": STD_STOCK, "rewards": {"track_turn": 5.0},
     "note": "PLAN 1.2 batch 2. Does the money keep working past 3.0? t3 at "
             "weight 3.0 gave 0.251 against a 0.20 bar, down from 0.476 at "
             "weight 1.0. This is the next step on the only lever that moved."},
    {"name": "t6_pay8", "turn_std": STD_STOCK, "rewards": {"track_turn": 8.0},
     "note": "PLAN 1.2 batch 2. Weight 8.0 - four times what track_speed is "
             "paid. Deliberately past where it should be needed, so the point "
             "where more money stops helping is INSIDE the batch rather than "
             "somewhere beyond it. If 8.0 is no better than 5.0, the lever is "
             "spent and 1.2.3's code change is the remaining answer."},
    {"name": "t7_wide5", "turn_std": STD_WIDE, "rewards": {"track_turn": 5.0},
     "note": "PLAN 1.2 batch 2. Band 1.20, WIDER not narrower. Batch 1 showed "
             "0.40 is worse than 0.80 at both weights, so the band's trend "
             "points the other way and nothing has tested past 0.80. Read "
             "against t5_pay5, which is the same weight at the stock band."},
    {"name": "t8_pay5_s7", "turn_std": STD_STOCK, "rewards": {"track_turn": 5.0},
     "seed": 7,
     "note": "PLAN 1.2 batch 2. IDENTICAL to t5_pay5 but seed 7. Not an "
             "experiment - it is the noise floor for the turn-rate number, "
             "which has never been measured. Every turn reading on record is a "
             "single seed, and the next decision turns on whether 0.251 beats "
             "0.20, a margin far smaller than the gap this pair will show."},
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
        print(f"{len(BATCH)} runs. No finished walk run to time them against, "
              f"so there is no honest estimate of how long.\n")

    for job in BATCH:
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
