"""Queue PLAN.md 1.3 batch 2 - one dial at a time, to find which one broke it.

    python run.py scripts/queue_13b.py --dry-run     show what it would add
    python run.py scripts/queue_13b.py               add it

WHAT BATCH 1 SAID. All five dials were widened at once, on three fresh seeds,
1500 iterations each. All three failed, and they failed on the SAME four
criteria:

    criterion            bar      d1      d2      d3     narrow, before
    forward drift       4.0 deg  4.35    5.40    4.83    about 3.7
    sideways drift      5.0 deg  5.59    5.83    5.04    3.74 to 4.25
    turn rate         0.20 rad/s 0.335   0.247   0.253   0.083 to 0.152
    turn wander         0.10 m   0.35    0.25    0.26    passing

    distance, speed, uprightness and trunk height: all four passed, on all
    three, with room to spare. 6.23 to 6.32 m walked against a 5.0 m bar.

THE ROBOT DID NOT LOSE ITS WALK. IT LOST ITS HEADING. Every criterion that
failed is a direction, and every criterion that passed is a distance or a
speed. Three seeds out of three is not a bad draw, so the dials caused it.

WHAT THIS BATCH ASKS. Which of the five. Six runs, one seed, 550 iterations:
one run per dial with that dial WIDE and the other four back at the Gray-Push
values, plus a control with all five narrow.

    run              wide            narrow
    f0_control       none            all five
    f1_grip          foot grip       the other four
    f2_mass          how heavy       the other four
    f3_com           where the weight is
    f4_servo         servo strength  the other four
    f5_drag          gearbox drag    the other four

WHY A CONTROL RUN, when the narrow numbers above are already measured. Because
they were measured at 1500 iterations and this batch runs at 550. A comparison
across two run lengths is not a comparison. f0 costs 17 minutes and it is the
only run here that makes the other five mean anything.

WHY 550 AND NOT 1500. Turn rate is the discriminator and turn rate settles by
550 - it has landed between 0.083 and 0.152 on every narrow run ever made,
which makes it the most repeatable number in the project. Sideways drift does
NOT settle by 550 (see 1.2.5: the same config, the same seed, gave 2.69 one run
and 9.06 another), so this batch cannot read the drift rows and does not try
to. It reads turn, and it reads it against f0.

`stop_at` IS SWITCHED OFF, and that is a change from every batch before it.
RULES.md rule 1 stops a run at 96.5% of the reward ceiling, which is right when
the question is "does this pass". It is wrong here: a run that stops at 420
iterations and a run that goes to 550 differ in how long they trained as well
as in the dial, and there would be no way to tell the two apart afterwards. All
six train exactly 550.

ONE SEED, NOT THREE. This batch does not decide anything on its own - it points
at a dial. Whatever it points at gets narrowed and re-run on three seeds, and
THAT run decides. Three seeds here would cost 5 hours to make a pointer more
confident.

Seed 2207 because it was the worst of the three in batch 1, at 0.335 rad/s -
the largest signal to work against.

5000 robots, 550 iterations, six runs. About 1.7 hours.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import queue  # noqa: E402

ROBOTS = 5000
ITERATIONS = 550
SEED = 2207


def dial_names() -> list[str]:
    """The five dial names, read out of WIDE_DIALS in walk_env_cfg.py.

    PARSED, not imported. `import gray.tasks.walk_env_cfg` pulls in mjlab, and
    a queue script runs in the plain interpreter where mjlab is not installed -
    queue_13a.py imports nothing but dashboard.queue for the same reason.

    Read rather than typed out, because a renamed dial has to fail HERE. It is
    caught either way: train.py refuses an unknown name. But there it is caught
    six times, once per run, after the queue has been left alone for the night.
    """
    src = (ROOT / "gray" / "tasks" / "walk_env_cfg.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "WIDE_DIALS"
                        for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            return [k.value for k in node.value.keys]
    raise SystemExit(
        "gray/tasks/walk_env_cfg.py has no WIDE_DIALS dict at module level. "
        "This batch narrows four of the five dials per run and cannot name "
        "them without it.")


DIALS = dial_names()

PLAIN = {
    "ground_grip": ("grip", "foot grip, 0.25 to 1.4 instead of 0.4 to 1.2"),
    "how_heavy": ("mass", "mass and inertia, +/-30% instead of +/-20%"),
    "where_the_weight_is": ("com", "the trunk centre of mass, +/-25 mm "
                                   "instead of +/-15"),
    "servo_strength": ("servo", "the servo gains, +/-40% instead of +/-30%"),
    "gearbox_drag": ("drag", "joint friction, 0.003 to 0.05 instead of "
                             "0.005 to 0.03"),
}

WHY = ("PLAN 1.3.1 batch 2. Batch 1 widened all five dials at once and failed "
       "3 seeds out of 3, on the same four criteria - forward drift, sideways "
       "drift, turn rate and turn wander. Distance, speed and uprightness all "
       "passed. The robot did not lose its walk, it lost its heading. This "
       "batch finds which dial did it: one dial wide, the other four back at "
       "the Gray-Push values. Turn rate is what gets read - it settles by 550 "
       "and has landed between 0.083 and 0.152 on every narrow run. Sideways "
       "drift does not settle by 550 and is not read here.")

BATCH = [
    {"name": f"f0_control_s{SEED}", "narrow_dials": "all",
     "note": "The control: all five dials narrow, the world 1.2 closed in. "
             "It exists because the narrow numbers on record were measured at "
             "1500 iterations and this batch runs at 550, and a comparison "
             "across two run lengths is not a comparison. " + WHY},
]
BATCH += [
    {"name": f"f{n + 1}_{PLAIN[dial][0]}_s{SEED}",
     "narrow_dials": ",".join(d for d in DIALS if d != dial),
     "note": f"Only one dial is wide: {PLAIN[dial][1]}. The other four are "
             f"back at the Gray-Push values. Read the turn rate against "
             f"f0_control_s{SEED}. " + WHY}
    for n, dial in enumerate(DIALS)
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
