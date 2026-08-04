"""Queue the week's experiment programme.

    python run.py scripts/queue_week.py --dry-run     show what it would add
    python run.py scripts/queue_week.py               add it

PLAN.md step 1.1, "make it walk". Rounds 0 to 2 are its substeps 1.1.2 to 1.1.4:
fifteen runs, plus a three-iteration smoke test in front of them. Round 3 (1.1.5)
is designed after these are read, because it starts from whichever config won.

How long it takes is measured off the runs already on disk rather than asserted
here. A hardcoded estimate goes stale silently and then gets planned against.

Every job carries a `note` saying what it is testing, because in three days a
queue of fifteen near-identical walk runs is unreadable without one.

Naming is `w<round><n>_<what>`, so the run list sorts into rounds by itself and
a name says what varied without opening anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import queue  # noqa: E402

# The stock veering ramp, from walk_env_cfg.py's `ease_in_straightness`. Listed
# rather than imported because importing it pulls in torch, and this script is
# meant to run in a second.
VEER_BASE = [-0.2, -0.5, -1.2, -2.0]
VEER_HARD = [-0.4, -1.0, -2.4, -4.0]

# Round 0. One config, three seeds. This is the only round whose result is not a
# ranking - it is a measurement of how much two identical runs differ, which
# every later comparison has to beat.
ROUND_0 = [
    {"name": f"w0{i}_seedfloor", "seed": seed,
     "note": f"PLAN 1.1.2 noise floor, seed {seed}. Identical config to the "
             f"other two - whatever these three disagree by is the number a real "
             f"effect has to beat. It matters more than usual this time, because "
             f"EIGHT things changed on 3 Aug 2026 and nothing before these runs "
             f"is comparable to anything after: the standing pose is the owner's "
             f"at 202.4 mm; the command box went from (0.15, 0.35) forward to "
             f"(-0.35, 0.35) with sideways +/-0.20 and turn +/-1.0; height, "
             f"pitch and roll became commands, so the observation is 48 numbers "
             f"not 45; wandering is measured in the sent-heading frame; "
             f"_going_straight gates on abs(vx); mjlab's forward-only clamp is "
             f"gone; ground_covered is measured along the commanded direction "
             f"instead of world X; and stepping no longer pays a standing robot "
             f"to march. Every bar the old runs were scored against was scored "
             f"on a different task."}
    for i, seed in enumerate((42, 7, 1234), start=1)
]

# Round 1. Full factorial on the three terms that hold a line. A factorial and
# not one-at-a-time because these interact: wandering and veering were pulling
# against each other until today's fix, and OFAT cannot see that.
_R1_FACTORS = [
    ("wandering", (-1.0, -3.0)),
    ("veering", (None, "hard")),
    ("skidding", (-0.5, -2.0)),
]

ROUND_1 = []
for n in range(8):
    wander = _R1_FACTORS[0][1][(n >> 2) & 1]
    veer_hard = bool((n >> 1) & 1)
    skid = _R1_FACTORS[2][1][n & 1]
    tag = ("W" if wander == -3.0 else "w") + ("V" if veer_hard else "v") + \
          ("S" if skid == -2.0 else "s")
    ROUND_1.append({
        "name": f"w1{n + 1}_{tag}",
        "rewards": {"wandering": wander, "skidding": skid},
        "ramps": {"veering": VEER_HARD} if veer_hard else {},
        "note": f"R1 {tag}: wandering {wander}, veering "
                f"{'x2' if veer_hard else 'stock'}, skidding {skid}. "
                f"Corner {n + 1} of 8 in the straightness factorial. On the WIDE box, so "
                f"these terms are now being asked to hold a line backward and "
                f"sideways as well as forward - which they were never tested on.",
    })

# Round 2. The speed bar. It was failing at 0.071 against 0.05 - but that number
# was measured on the OLD box, where mjlab's forward-only clamp forced four
# attempts in five to at least 0.3 m/s, three quarters of them landing exactly on
# it. The policy was never trained across a speed range at all. So the 0.071 is
# not a starting point any more; round 0 measures what it actually is now, and
# this round may turn out to be tuning something that already passes.
ROUND_2 = [
    {"name": f"w2{n + 1}_spd{ts}x{gc}",
     "rewards": {"track_speed": ts, "ground_covered": gc},
     "note": f"R2: track_speed {ts}, ground_covered {gc}. Does paying more for "
             f"speed close the speed-tracking gap, or just trade drift for it? "
             f"Both terms changed on 3 Aug 2026: the speed range is real now "
             f"rather than clamped to one value, and ground_covered is measured "
             f"along the commanded direction rather than world X - so it can pay "
             f"for backward and sideways, which it never could. Read round 0 "
             f"first; the 0.071 vs 0.05 this round was designed around was "
             f"measured on a task that no longer exists."}
    for n, (ts, gc) in enumerate(((2.0, 1.0), (4.0, 1.0), (2.0, 2.0), (4.0, 2.0)))
]


# Three iterations of walking, in front of everything else. It is worth a
# job of its own whenever the MODEL or the TASK has changed under the runs, and
# on 3 Aug 2026 both did: a new standing pose, a wider command box, three new
# commanded numbers, and three reward terms rewritten. If any of that is broken
# this says so in about a minute instead of six hours from now.
SMOKE = {
    "name": "w00_smoke", "task": "Gray-Walk", "iterations": 3,
    "film": False, "verify": False,
    "note": "Smoke test, 3 iterations. Not an experiment - it only has to build "
            "the task, draw a command, step, and score. Gray-Walk rather than "
            "Gray-Stand on purpose: the standing pose moved on 3 Aug 2026 AND "
            "the walk task gained a second command term for height, pitch and "
            "roll. Gray-Stand has no commands at all, so it would exercise none "
            "of the new code and pass while the walk task was broken.",
}


def measured_minutes() -> float | None:
    """How long a walk run has actually taken, off the runs on disk.

    This used to be a hardcoded 87 minutes multiplied by the job count. Nothing
    produced that number and nothing checked it, so it went stale silently - and
    an estimate nobody can trace is worse than no estimate, because it gets
    planned against. If there is no finished walk run to measure, say so.
    """
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
        # A run that was stopped early still times honestly per iteration, so it
        # counts - scaled up to the 3000 these jobs ask for.
        try:
            a = datetime.fromisoformat(r["started"])
            b = datetime.fromisoformat(r["finished"])
        except (KeyError, TypeError, ValueError):
            continue
        secs = (b - a).total_seconds()
        if secs > 0:
            mins.append(secs / 60 * (3000 / done))
    if not mins:
        return None
    mins.sort()
    return mins[len(mins) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and add nothing")
    ap.add_argument("--rounds", default="0,1,2",
                    help="which rounds to queue, comma separated")
    ap.add_argument("--no-smoke", dest="smoke", action="store_false",
                    help="skip the 3-iteration stand check in front of the queue")
    args = ap.parse_args()

    want = {r.strip() for r in args.rounds.split(",") if r.strip()}
    batches = [("0", ROUND_0), ("1", ROUND_1), ("2", ROUND_2)]
    jobs = [j for name, batch in batches if name in want for j in batch]
    if not jobs:
        raise SystemExit(f"no rounds selected from {args.rounds!r}")

    if args.smoke:
        jobs = [SMOKE, *jobs]

    per = measured_minutes()
    if per:
        print(f"{len(jobs)} runs, about {len(jobs) * per / 60:.0f} hours "
              f"at {per:.0f} min each, measured off the runs already on disk\n")
    else:
        print(f"{len(jobs)} runs. No finished walk run to time them against yet, "
              f"so there is no honest estimate of how long.\n")

    for job in jobs:
        spec = {"task": "Gray-Walk", "film": True, "verify": True, **job}
        cleaned = queue._clean(dict(queue.DEFAULTS, **spec))
        print(f"  {job['name']:22} {queue.command_line(cleaned)}")
        if not args.dry_run:
            queue.add(spec)

    if args.dry_run:
        print("\nNothing added. Drop --dry-run to queue it.")
    else:
        # The job dict's key is `state`. It was `status` here, which is the key
        # the DASHBOARD renames it to - so this line raised KeyError after every
        # job had already been added, and the script looked like it had failed
        # when it had actually worked.
        waiting = sum(1 for j in queue.load()["jobs"] if j.get("state") == "queued")
        print(f"\nQueued. {waiting} jobs waiting.")
        print("Start the runner in a second terminal:  run.bat --runner")


if __name__ == "__main__":
    main()
