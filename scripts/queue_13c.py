"""Queue the gait batch - one lever per run, against the faults the films show.

    python run.py scripts/queue_13c.py --dry-run     show what it would add
    python run.py scripts/queue_13c.py               add it

RESTART THE RUNNER FIRST. spin_share, dive_ends and swing_target were added to
dashboard/queue.py on 6 Aug 2026, and a runner started before that still holds
the old module. The argv guard turns that into a failed job rather than a run
that silently trains the wrong thing - but a failed job at 2 am is still a
wasted night.

WHAT THIS BATCH IS FOR. On 6 Aug the owner's films beat the instruments. The
robot turns well - about three quarters of any rate it is asked, measured at
0.5, 0.7 and 1.0 rad/s - but the feet barely lift, the nose points down in a
forward walk, and it usually collapses nose-down after a short walk. Not one
of those three was measured by anything: error_pitch throws the sign away,
the reward records the cost of a swing and not its height, and nothing writes
down WHEN a robot fell, only whether. verify --gait-diag measures all of it
now, and this batch is the first to be read with it.

WHY THESE LEVERS.

    g1  the turn bar is a spin on the spot at 1.00 rad/s, and an independent
        draw produces a pure spin about once in 80 - the same exposure fault
        the crab share fixed. Decided with the owner: train the case, keep
        the bar.
    g2  a nose-down collapse costs the policy nothing today: no sensor on the
        trunk, an episode only ends past 45.8 deg of tilt or under 111 mm.
        The trunk contact sensor is now always on; --dive-ends makes it
        terminal, and fell_over pays its -40 for it.
    g3  the swing target is 35 mm, chosen when 35 mm matched the stage 3 bar.
        A foot with 35 mm in hand has nothing left for ground that is not a
        floor - and 1.3.2 is slopes and rough ground.
    g4  the other way to lift the feet: leave the target alone and make the
        peak error worth chasing. swing_height earns about 0.4% of the reward
        at -0.25, and the project's own rule says a term under 1% cannot
        change a gait.

    run                 one change                       read against g0
    g0_control          none - the task as it stands
    g1_spin             spin_share 0.10                  turn_error
    g2_dive             dive_ends                        dives, collapses
    g3_swing50          swing_target 0.05                swing peaks
    g4_swingw           swing_height -1.0                swing peaks

WHY A CONTROL, when f0_control_s2207 exists at 550. f0 ran with all five world
dials NARROW - it was batch 2's control, not this batch's. These five run the
task as it stands, dials wide, and a comparison that differs in the dials as
well as the lever is not a comparison.

WHAT GETS READ, AND WHAT DOES NOT. At 550, turn error settles; swing peaks,
pitch and dive counts come from verify --gait-diag on the finished checkpoint
- run it by hand on each, it is minutes. Crab drift does NOT settle by 550
(1.2.5: same config, same seed, 2.69 one run and 9.06 another) and is not
read here. The winning combination goes to 3 x 1500 on fresh seeds, and THAT
batch decides.

`stop_at` IS OFF, same reason as batch 2: a run that stops early differs in
length as well as in the lever.

ONE SEED, 2207 - the seed both earlier batches ran, so the g0 control also
reads against f0_control_s2207 for a wide-against-narrow number at equal
length, for free.

5000 robots, 550 iterations, five runs. About 1.5 hours.
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
SEED = 2207

WHY = ("The gait batch, 6 Aug 2026. The films show three faults no number "
       "recorded: feet that barely lift, a nose that points down walking "
       "forward, a nose-down collapse after a short walk. One lever per run. "
       "Turn error is readable at 550; the gait numbers come from verify "
       "--gait-diag on the checkpoint; crab drift is not readable at 550 and "
       "is not read.")

BATCH = [
    {"name": f"g0_control_s{SEED}",
     "note": "The control: the task exactly as it stands, dials wide, no "
             "lever. The four probes read against this - f0_control_s2207 "
             "was batch 2's control and ran with all five dials NARROW, so "
             "it cannot be this batch's baseline. " + WHY},
    {"name": f"g1_spin_s{SEED}", "spin_share": 0.10,
     "note": "One lever: 10% of command draws become a PURE spin, the same "
             "exposure fix the crab share was. An independent draw makes a "
             "pure spin about once in 80, and the turn bar tests nothing "
             "else. Read turn_error against g0_control. " + WHY},
    {"name": f"g2_dive_s{SEED}", "dive_ends": True,
     "note": "One lever: the trunk touching anything ends the attempt, and "
             "fell_over pays -40 for it. Today a nose-down collapse costs "
             "nothing - no trunk sensor, termination only past 45.8 deg or "
             "under 111 mm. Read dive counts and first-fall times from "
             "--gait-diag against g0_control. " + WHY},
    {"name": f"g3_swing50_s{SEED}", "swing_target": 0.05,
     "note": "One lever: the swing target up from 35 mm to 50 mm, on both "
             "dragging and swing_height. Read the swing peaks from "
             "--gait-diag against g0_control, and speed tracking for what "
             "the higher step costs. " + WHY},
    {"name": f"g4_swingw_s{SEED}", "rewards": {"swing_height": -1.0},
     "note": "One lever: swing_height's weight from -0.25 to -1.0, target "
             "left at 35 mm. The other way to lift the feet - make the peak "
             "error worth chasing; at -0.25 it earns about 0.4% of the "
             "reward, under the project's own 1% floor for changing a gait. "
             "Read the swing peaks against g0_control and g3_swing50. " + WHY},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and add nothing")
    args = ap.parse_args()

    print(f"{len(BATCH)} runs, {ITERATIONS} iterations each, seed {SEED}, "
          f"stop_at off so they all train the same length.\n")

    for job in BATCH:
        spec = {"task": "Gray-Walk", "film": True, "verify": True,
                "num_envs": ROBOTS, "iterations": ITERATIONS, "seed": SEED,
                **job}
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
