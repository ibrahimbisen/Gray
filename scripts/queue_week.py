"""Queue the week's experiment programme.

    python run.py scripts/queue_week.py --dry-run     show what it would add
    python run.py scripts/queue_week.py               add it

Rounds 0 to 2, fifteen runs, about 23 hours. Round 3 is designed after these are
read, because it starts from whichever config won.

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
     "note": f"R0 noise floor, seed {seed}. Identical config to the other two - "
             f"whatever these three disagree by is the number a real effect has "
             f"to beat. Also the first run with wandering measured in the "
             f"sent-heading frame."}
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
                f"Corner {n + 1} of 8 in the straightness factorial.",
    })

# Round 2. The speed bar, which is failing at 0.071 against 0.05. Independent of
# round 1, so it is queued alongside rather than after it.
ROUND_2 = [
    {"name": f"w2{n + 1}_spd{ts}x{gc}",
     "rewards": {"track_speed": ts, "ground_covered": gc},
     "note": f"R2: track_speed {ts}, ground_covered {gc}. Testing whether paying "
             f"more for speed closes the 0.071 vs 0.05 gap, or just trades drift "
             f"for it."}
    for n, (ts, gc) in enumerate(((2.0, 1.0), (4.0, 1.0), (2.0, 2.0), (4.0, 2.0)))
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and add nothing")
    ap.add_argument("--rounds", default="0,1,2",
                    help="which rounds to queue, comma separated")
    args = ap.parse_args()

    want = {r.strip() for r in args.rounds.split(",") if r.strip()}
    batches = [("0", ROUND_0), ("1", ROUND_1), ("2", ROUND_2)]
    jobs = [j for name, batch in batches if name in want for j in batch]
    if not jobs:
        raise SystemExit(f"no rounds selected from {args.rounds!r}")

    print(f"{len(jobs)} runs, about {len(jobs) * 87 / 60:.0f} hours\n")
    for job in jobs:
        spec = {"task": "Gray-Walk", "film": True, "verify": True, **job}
        cleaned = queue._clean(dict(queue.DEFAULTS, **spec))
        print(f"  {job['name']:22} {queue.command_line(cleaned)}")
        if not args.dry_run:
            queue.add(spec)

    if args.dry_run:
        print("\nNothing added. Drop --dry-run to queue it.")
    else:
        state = queue.load()
        waiting = sum(1 for j in state["jobs"] if j["status"] == "queued")
        print(f"\nQueued. {waiting} jobs waiting.")


if __name__ == "__main__":
    main()
