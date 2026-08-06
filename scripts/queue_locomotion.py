"""Queue THE locomotion batch - everything the dog has to do, in one world.

    python run.py scripts/queue_locomotion.py --dry-run     show what it would add
    python run.py scripts/queue_locomotion.py               add it

WHAT THIS IS. The owner's decision on 6 Aug 2026, after driving the robot and
reading the bars: stop running one probe per question. Train everything at
once, measure afterwards, and if it fails, find the cause by MEASURING the
failed policy condition by condition rather than by training four more
batches.

WHAT IS IN THE WORLD:

    speed        -0.5 to 0.9 m/s, up from -0.35 to 0.35. About 3 km/h at the
                 top. The owner asked for 7 km/h; the servos cannot do it -
                 the knee already holds 55% of its stall torque standing, and
                 1.94 m/s needs a gallop at 13 steps a second against a 50 Hz
                 control loop. 0.9 is the honest stretch, and the speed
                 ladder afterwards says what it really reached.
    ground       flat, gentle hills, steep hills, a bowl, rough ground and
                 waves, all in one world - 5000 robots spread across every
                 patch at once, so ONE policy learns all of it.
    gravel       the SHAPE of it - rough patches with bumps up to 40 mm -
                 against a grip dial that already draws anything from ice
                 (0.25) to dry concrete (1.4). Stones that roll and slide
                 under a foot are NOT simulated. Only the real robot
                 settles that one, and it should be said out loud rather
                 than assumed away.
    payload      0 to 1.0 kg on the trunk. The torque arithmetic says about
                 1 kg is what the servos have left, so nothing in the batch
                 asks for a weight that cannot physically be held. The true
                 limit gets MEASURED on the finished policy, which takes
                 minutes and needs no training.

WHAT IS NOT IN IT: stairs, obstacles to step over, and getting up after a
fall. Those are their own policies and their own decisions.

5500 iterations, no early stop - the stop was deleted from the codebase on
6 Aug 2026, see RULES.md rule 1. Checkpoints are saved throughout, so the
morning read can score the policy at 3000 AND at 5500 and say whether the
last 2500 bought anything.

THREE SEEDS, each continuing its own parent from the gait confirm, so they
are three genuinely different attempts rather than three copies of one.

5000 robots - the ceiling in dashboard/queue.py, and the owner asked for
5500. The cap stands at 5000 because it is already 11% more card memory
than every run before 4 Aug used, the terrain adds more on top, and the
failure mode is an out-of-memory crash at 3 am.

3 runs, 5500 iterations. About 15 hours.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import queue  # noqa: E402

ROBOTS = 5000
ITERATIONS = 5500

# seed -> the gait-confirm run it continues from.
PARENTS = {
    1301: "2026-08-06_04-43-00_gc_1301",
    4507: "2026-08-06_05-34-25_gc_4507",
    8821: "2026-08-06_06-28-32_gc_8821",
}

WHY = ("THE locomotion batch, 6 Aug 2026. One world with everything in it: "
       "speed to 0.9 m/s, hills up and down, rough ground, gravel by its "
       "effects, and 0 to 1 kg of payload. 5500 iterations, no early stop. "
       "Read against the owner's revised bars - forward drift 7 deg, crab "
       "drift 15 deg, a full turn in 12 s, 50 cm of turn wander - plus a "
       "speed ladder, a slope test and a payload ladder measured on the "
       "finished policy.")

BATCH = [
    {"name": f"L1_{seed}", "seed": seed, "init_from": parent,
     "note": f"Seed {seed}, continuing {parent}. " + WHY}
    for seed, parent in PARENTS.items()
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and add nothing")
    args = ap.parse_args()

    print(f"{len(BATCH)} runs, {ITERATIONS} iterations each, {ROBOTS} robots, "
          f"seeds {', '.join(str(s) for s in PARENTS)}.\n")

    for n, job in enumerate(BATCH):
        job = dict(job)
        job["name"] = f"L{n + 1}_{job['seed']}"
        spec = {"task": "Gray-Walk", "film": False, "verify": True,
                "num_envs": ROBOTS, "iterations": ITERATIONS,
                "mixed_ground": True, "payload_kg": 1.0, **job}
        cleaned = queue._clean(dict(queue.DEFAULTS, **spec))
        print(f"  {job['name']:10} {queue.command_line(cleaned)}")
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
