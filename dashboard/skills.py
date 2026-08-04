"""The skill library, read and counted. The CSV says what everything is.

**This file no longer classifies anything.** It used to: 230 lines of lookup
tables deciding which subsection each row belonged to, what covered it, and what
it was blocked on - and about 40% of rows had their filed category contradicted
by one of those tables. One comment in it read "not a jump, whatever the category
says", which is Python arguing with data.

The library now states all of that in its own columns, so this file reads them:

    id          the row's number
    name        one canonical phrasing
    kind        run | axis | constraint | test | parked
    group       R1 R2 R3 D1 C1 T1 T2 P1 P2
    coverage    command | condition | emerges | pilot | blocked | measured
    blocked_on  height | range | foot | camera | flight, and only when blocked
    dial        a numeric range, or empty
    note        why, in one sentence
    covers      which of the original 200 rows this one stands in for

Anything that does not parse goes in `faults` and is shown on the page. A row this
file quietly guesses at is a row nobody ever finds out is wrong.

**A skill library is not a work list.** The unit of work is a **training run**,
because a run is what produces a policy file. Everything else is a number inside
one, a property checked across all of them, or a measurement taken afterwards. So
each row is one of five kinds:

    run         a training run. Produces a policy file. There are two.
    axis        a dial turned during a run. Not a run of its own.
    constraint  a property every run has to hold. Checked, never trained.
    test        measured after training. Nothing new is learned.
    parked      in the plan, not being worked. Each one carries its trigger.

**The test for whether two skills share a run** is not what they look like. It is
whether the *reward function* would have to change. Walking sideways is the same
equation with a different setpoint, so it shares a run. Getting up off its back has
no commanded velocity to track at all, so it needs its own.

That test is what collapses the library into two runs. Deduplicating on top of it
took 200 rows to 60: the old file spent 15 rows on jumping, 20 on the command box,
16 on ride height, and 4 on one drop height. `covers` records where every one of
the original 200 went.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# One file, named once. This used to be a list with a browser-download name at
# the front of it - "gray_skill_library (1).csv" - which would have silently
# shadowed the real library the first time a copy landed in the repo root.
LIBRARY = "gray_skill_library.csv"

# The five kinds, in the order they appear on the page. `blurb` is what the kind
# means; it is shown once, above the items, so each item does not have to re-argue
# why it is not a training run.
KINDS = [
    {"key": "run", "name": "Training runs",
     "blurb": "Each one produces a policy file. This is the only kind that is a "
              "piece of work in its own right, and there are two: locomotion, and "
              "getting up. PLAN.md steps 1 and 2."},
    {"key": "axis", "name": "Dials on a run",
     "blurb": "Conditions varied during a run, not runs of their own. The reward "
              "does not change and neither does the observation - only the world "
              "does. This is domain randomisation, and it is where most of the "
              "library lives."},
    {"key": "constraint", "name": "Constraints",
     "blurb": "Properties every run has to hold, checked continuously. Not a stage "
              "and never finished."},
    {"key": "test", "name": "Tests",
     "blurb": "Measured after a run. Nothing is trained here - these say whether "
              "what was trained is any good."},
    {"key": "parked", "name": "Parked",
     "blurb": "In the plan, not being worked. Each one names the thing that "
              "un-parks it, so 'later' means something."},
]

# What a subsection is, in the order it appears on the page.
SUBSECTIONS = [
    # ---- runs ------------------------------------------------------------
    {"id": "R1", "key": "walk", "kind": "run", "name": "Locomotion",
     "rule": "one policy, six commands",
     "what": "Everything the robot does on its feet, standing included. Forward, "
             "backward, sideways, diagonal, turning, arcs, sprints, dead stops, "
             "crouching, crawling, sit to stand, holding a height, pitching and "
             "rolling on the spot, seven named gaits, stepping stones, walking to "
             "a point and stopping on a heading.",
     "why": "The largest group in the library and ONE training run, because every row "
            "in it is the same reward with a different setpoint. Walk backward is a "
            "negative number in the same command. Standing still is that command at "
            "zero. Holding a height is a seventh number alongside it. None of that "
            "changes the equation being optimised, so none of it needs its own "
            "policy - and splitting it would mean learning balance twice and then "
            "needing a switcher, which is where separately-trained policies fall "
            "over. Holding a height would be a fourth number alongside the three, "
            "not a fourth policy.",
     # What the vector ACTUALLY holds today, read against walk_env_cfg.py's
     # UniformVelocityCommandCfg. It used to read "vx, vy, yaw, height, pitch,
     # roll", which described a proposal as if it had shipped - directly beside
     # ten rows listed as blocked on exactly those three.
     "commands": "vx, vy, yaw",
     "gap": "Thirteen of these rows are blocked on a command that does not exist. "
            "Ten want a height or an attitude - crouching, sitting, leaning, "
            "'hold any commanded height' - and folding height, pitch and roll into "
            "the vector covers all ten at once. Three want a single leg addressed "
            "on its own, which is a bigger change nobody has costed. Both grow the "
            "observation, so both mean every policy file is retrained from zero.",
     "state": "in progress"},
    {"id": "R2", "key": "recover", "kind": "run", "name": "Getting up",
     "rule": "starts lying on the ground",
     "what": "Up from the belly, from either side, from fully upside down, on a "
             "slope, in a tight space. Surviving drops. Powering on in a random "
             "pose. Being picked up and set down somewhere else.",
     "why": "The one group that genuinely fails the shared-reward test. There is no "
            "commanded velocity to track when the robot is on its back, so "
            "track_speed, stepping, dragging and wandering are all meaningless. "
            "Different start state, different reward, different terminations - "
            "therefore a different policy file, and a rule at runtime that hands "
            "control back to R1 once the robot is upright.",
     "commands": "none - the goal is fixed: get upright",
     "state": "not started"},
    {"id": "R3", "key": "tool", "kind": "parked", "name": "Using a foot on something",
     "rule": "the goal is the object, not the robot",
     "what": "Push an object, drag it backward, press a pedal, kick, dig, paw or tap.",
     "why": "Fails the shared-reward test in the other direction: success is "
            "measured on something that is not the robot. It needs objects in the "
            "scene, which nothing else does, and a reward that reads the object's "
            "state.",
     "commands": "where to move the object",
     "trigger": "None. PLAN.md shelves it outright - nothing in the project "
                "depends on it, so there is no event that makes it due.",
     "state": "parked"},

    # ---- dial ------------------------------------------------------------
    {"id": "D1", "key": "robust", "kind": "axis", "name": "A harder world",
     "rule": "no new skill - a number changed",
     "what": "Grass, gravel, sand, wet tile, rubble, slopes, stairs, wobbly boards. "
             "Shoves and sustained leans. Payloads to +2 kg, off-centre and "
             "sloshing. Weak servos, dead servos, low battery, worn feet. Wind. "
             "Being handled.",
     "why": "Second only to R1 in size, and not a piece of work at all. None of it "
            "teaches a new movement: the policy never sees the word 'gravel', "
            "it sees the same thirty numbers arriving in a different pattern. These "
            "rows are the RANGE the run is sampled over, and the DOE rule applies - "
            "inside the range sampled, trusted; outside it, nothing. Never vary "
            "friction below 0.3 and the robot has no answer for ice. Already partly "
            "on: friction, mass, centre of mass, PD gains and joint friction are "
            "randomised on every attempt in push_env_cfg.py.",
     "state": "partly - already on in R1"},

    # ---- constraint ------------------------------------------------------
    {"id": "C1", "key": "quality", "kind": "constraint", "name": "Fit for the real robot",
     "rule": "a property of every run, not a run",
     "what": "Only observe what Gray can actually measure. Only ask for motion 1.96 "
             "N-m at 50 Hz can produce. Model the sensor noise, the drift, the "
             "command delay. Behave sensibly when given an impossible order.",
     "why": "What killed the previous attempt at this project: the policy read joint "
            "angles the hardware could not measure, so it could never have run at "
            "all. Cheap to check as you go, fatal to discover at the end. Not a "
            "stage and never finished.",
     "state": "ongoing"},

    # ---- tests -----------------------------------------------------------
    {"id": "T1", "key": "chain", "kind": "test", "name": "Joining them up",
     "rule": "a handover test",
     "what": "Fall, get up, keep walking. Walk, jump, land, keep walking. Sprint, "
             "stop, turn 180, sprint back. Stairs to flat to stairs. Freeze "
             "mid-stride and resume. Emergency stop from any state.",
     "why": "Nothing is trained here. These check that R1 and R2 hand over to each "
            "other cleanly, which is exactly where a set of separately-trained "
            "policies usually falls apart. The handover is the risk, not either "
            "policy on its own.",
     "state": "not started"},
    {"id": "T2", "key": "measure", "kind": "test", "name": "Speed and smoothness",
     "rule": "measured, never trained",
     "what": "Find top speed and the speed ladder. Acceleration. Sustained run. Top "
             "speed per surface, backward, sideways, with payload. Ten minutes "
             "continuous. Efficient gait at each speed. Same quality at low battery. "
             "Keep an object on its back.",
     "why": "Every row here is a number you read off a trained policy, not a target "
            "you train toward. 'Find top speed' has no reward term - you raise the "
            "command until it fails and write down where. Filed as work, these look "
            "like eighteen more things to do; filed as tests, they are an afternoon "
            "with a stopwatch.",
     "state": "not started"},

    # ---- parked ----------------------------------------------------------
    {"id": "P1", "key": "flight", "kind": "parked", "name": "Jumping",
     "rule": "all four feet leave the ground",
     "what": "Jumps in place, over gaps, up and down steps, mid-air twists, two "
             "jumps back to back, landing on a narrow target - and the gaits with a "
             "flight phase: pronk, gallop, three-legged hop.",
     "why": "Every other run assumes at least one foot is down, and every "
            "contact-based reward quietly stops working the moment none is. It also "
            "needs more torque and far more joint SPEED than anything else here. "
            "Rough arithmetic puts a push-off at roughly 60 degrees of knee travel "
            "in about 0.13 s, which is at or past the DS3218's no-load rating - but "
            "that rating is a datasheet number, and datasheet numbers are exactly "
            "what this project is removing. Answering it now would mean spending a "
            "calculation on a guessed input.",
     "trigger": "Stage 3.3 measures the real servo speed under load. That "
                "measurement answers this for free. Un-park it then, not before.",
     "state": "parked"},
    {"id": "P2", "key": "see", "kind": "parked", "name": "Seeing",
     "rule": "needs a camera that does not exist",
     "what": "Spot an obstacle and step over it, avoid a hole, find stairs from a "
             "metre away, follow a path, react to something appearing, cope with low "
             "light, a reflective floor, a dirty lens.",
     "why": "The only group that changes the OBSERVATION rather than the command or "
            "the world - it adds a heightmap to the input list, so the contract "
            "changes and the policy retrains from scratch. It is also the hardest "
            "sim-to-real jump there is: in simulation the heightmap is perfect and "
            "free, and on the robot it is a noisy 30 Hz depth camera full of holes. "
            "Reacting to an obstacle is blind and already covered by D1 - step, feel "
            "the contact, correct. Only ANTICIPATING one needs eyes.",
     "trigger": "A camera is fitted and a depth-to-heightmap pipeline exists. Its "
                "own project, after the robot walks.",
     "state": "parked"},
]

# HOW a skill is covered, which is a different question from WHERE it is filed.
#
# The owner's question, and it is the right one: "if one reward function does all
# 69, why are we training it to back into a tight spot?" We are not. Nothing in
# the code mentions any of these rows. They are covered - or not - in five very
# different ways, and lumping them together is what makes a 69-row list look like
# 69 pieces of work.
#
#   command   set a number and it happens. Already covered by the reward.
#   condition a dial in the scene. Same reward, different world.
#   emerges   nobody asks for it; it falls out of the reward terms.
#   blocked   no amount of reward tuning produces it. NOT "impossible" and NOT
#             "the hardware will never exist" - the owner is fitting cameras and
#             distance sensors. It means the fix is a new input, a new command or
#             a separate run, never a new weight in the existing one.
#   measured  read off a trained policy. Never trained toward.
#   pilot     a human at the controls supplies it. Added 3 Aug 2026 on the
#             owner's question - "why does it need to know where it is, I am the
#             one driving it with a camera and a controller?" He is right, and it
#             is a real sixth answer rather than a softer word for blocked. A row
#             that a pilot handles is not missing anything: the robot does not
#             need to know where it is because somebody who does is steering.
COVERAGE = {
    "command":   "set a command and it happens - already covered",
    "condition": "a dial in the scene - same reward, different world",
    "emerges":   "nobody asks for it; it falls out of the reward",
    "pilot":     "the human at the controls does it - nothing is missing",
    "blocked":   "no weight in the current reward reaches it - it needs a new "
                 "input, a new command, or its own run",
    "measured":  "read off a trained policy, never trained toward",
}

# WHY a blocked row is blocked, which is three different problems that were
# previously wearing one word. Splitting them matters because the fix, the cost
# and the person who unblocks it are different in each case.
BLOCKED_CLASSES = {
    "input": {
        "label": "The policy cannot KNOW it",
        "why": "The policy reads a fixed list of numbers, and these are not on it. "
               "Today that list is: which way is down and how fast the trunk is "
               "turning (the IMU), twelve joint angles, twelve joint speeds, the "
               "twelve targets it sent last step, and the three command numbers. "
               "Nothing else reaches it.",
        "fix": "Fitting the sensor is necessary and not sufficient. The reading has "
               "to be added to that list, which changes its length - so it is a "
               "different policy and a retrain from zero. The simulator also has to "
               "produce a fake version of the sensor for every robot at 50 Hz, and "
               "for a camera that is the expensive part, not the hardware.",
    },
    "command": {
        "label": "The policy cannot be TOLD to",
        "why": "Nothing is missing from what the robot senses. There is simply no "
               "way to ask. The command is three numbers - forward, sideways, turn "
               "- and a request that is not one of those three cannot be made, so "
               "it can never be scored either.",
        "fix": "Add the numbers to the command. Cheap to write, and it still grows "
               "the input list, so it is still a retrain.",
    },
    "sampled": {
        "label": "The range was never sampled",
        "why": "The command exists and the reward already scores it. The policy "
               "simply never once saw that value. A policy knows the BOX it was "
               "sampled over and nothing outside it, so a corner that was never "
               "visited is a corner it cannot do - and it does not refuse, it "
               "guesses. Measured on run #25: asked to walk backward it stood "
               "still for 8 s; asked for a 45 degree diagonal, which is outside "
               "the sampled range, it managed 26 degrees.",
        "fix": "One line in walk_env_cfg.py and a retrain. By far the cheapest "
               "class on this page - no hardware, no new input, no new reward "
               "term. The cost is dilution: the sample budget is fixed, so a "
               "wider box means fewer draws everywhere, and these corners are "
               "harder than straight-ahead rather than equal to it.",
    },
    "reward": {
        "label": "The reward stops working",
        "why": "Not a sensing problem at all - detection here is already perfect. "
               "The contact sensor reports zero feet down today. What breaks is the "
               "SCORING: foot_clearance, stepping, dragging and ground_covered are "
               "each multiplied by foot contact, so the moment all four feet leave "
               "the ground every one of them returns zero and the robot is given no "
               "signal for the whole airborne part.",
        "fix": "A different reward function for the airborne part, which by this "
               "page's own test means a separate run.",
    },
}

# What every blocked row is waiting on. "19 blocked" is only a useful number if
# it says blocked on WHAT - and the answer is four missing capabilities, not
# nineteen separate problems. Grouped this way, the list becomes four things to
# build rather than nineteen things to worry about.
BLOCKED_NEEDS = {
    "height": {"label": "A height and attitude command", "class": "command",
               "why": "The command is vx, vy, yaw. Nothing in it asks for a ride "
                      "height or a lean, so a crouch cannot be commanded and 'hold "
                      "any commanded height' cannot even be tested.",
               "cost": "The one blocker with a decision already on the table: fold "
                       "height, pitch and roll into the command. Ten rows at once. "
                       "It grows the input list by three, so every existing policy "
                       "file becomes unreadable and all three runs restart."},
    "range":  {"label": "A wider command range", "class": "sampled",
               "why": "WALK_SPEED is (0.15, 0.35) and WALK_SIDE is (-0.10, 0.10). "
                      "vx is never zero and never negative, so backward and pure "
                      "sideways were not sampled once in 864,000 draws. Confirmed "
                      "by driving run #25: backward moved 0.00 m in 8 s.",
               "cost": "Widen WALK_SPEED to (-0.35, 0.35) and let vy stand without "
                       "vx. Cheap to write, and it invalidates every current run. "
                       "TRAP FIRST: _going_straight() gates on `command[:, 0] > "
                       "MOVING`, which is positive-only - so `veering` and "
                       "`wandering`, the two penalties that hold a line, switch "
                       "off entirely for a backward command. Widen the range "
                       "without fixing that gate to abs(vx) and backward walking "
                       "trains with no straightness penalty at all."},
    "foot":   {"label": "A per-foot command", "class": "command",
               "why": "Nothing in the command addresses one leg, so a three-legged "
                      "stance or a raised foot cannot be asked for.",
               "cost": "Bigger than the height change and nobody has costed it - "
                       "four more numbers, and terms that currently score the robot "
                       "as a whole would have to score it per leg."},
    "camera": {"label": "A camera", "class": "input",
               "why": "Anticipating an obstacle rather than feeling it. Reacting to "
                      "one is already covered - step, feel the contact, correct.",
               "cost": "The camera is the easy half. Behind it: a depth-to-heightmap "
                       "pipeline, a much longer input list, and a simulator that has "
                       "to render a fake depth image for 4500 robots at 50 Hz. It is "
                       "also the hardest sim-to-real jump there is, because the fake "
                       "heightmap is perfect and free and the real one is a noisy "
                       "30 Hz camera full of holes. Its own project."},
    # Retired 3 Aug 2026. Kept so the concept is documented and so a future row
    # can be filed here deliberately, but nothing is currently blocked on it:
    # the owner flies Gray with a camera and a controller, so a human supplies
    # the position and the seven rows that needed it are `pilot`, not blocked.
    "where":  {"label": "Knowing where it is", "class": "input",
               "why": "Where it stands on the floor, and where the walls are.",
               "cost": "Retired as a blocker. A pilot supplies it. Only comes back "
                       "if Gray is ever asked to do something with nobody on the "
                       "sticks - and then it is UWB anchors in the room, because "
                       "range finders give range, not position."},
    "flight": {"label": "A flight phase", "class": "reward",
               "why": "Detection is NOT the problem - the contact sensor already "
                      "reports zero feet down, today. The reward is the problem: "
                      "every contact-gated term returns zero while the robot is in "
                      "the air, so it is scored nothing for the part that matters.",
               "cost": "A different reward for the airborne part, which means its "
                       "own run. And first the physical question: whether DS3218MGs "
                       "can push 3.1 kg off the ground at all. Stage 3.3 measures "
                       "the real servo speed under load and answers it for free."},
    # A blocked row that names no blocker is a hole in THIS file, not a fact
    # about the robot. Shown rather than defaulted into someone else's pile.
    "unsaid": {"label": "Not yet said what", "class": "command",
               "why": "Marked blocked, but nothing here says what it is waiting on.",
               "cost": "That is a gap in this page, not a fact about the robot. Give "
                       "it a blocked_on in the CSV, or stop calling it blocked."},
}

# Kept for anything still importing it. `kind == "run"` is the real answer now.
TEACHES_NEW = tuple(s["key"] for s in SUBSECTIONS if s["kind"] == "run")

_KIND_OF = {s["id"]: s["kind"] for s in SUBSECTIONS}
_NAME_OF = {s["id"]: s["name"] for s in SUBSECTIONS}
_KEY_OF = {s["id"]: s["key"] for s in SUBSECTIONS}

VALID_COVERAGE = frozenset(COVERAGE)
VALID_NEEDS = frozenset(BLOCKED_NEEDS)


def _csv_path() -> Path | None:
    p = ROOT / LIBRARY
    return p if p.is_file() else None


def _group_needs(items: list[dict]) -> list[dict]:
    """Blocked rows, grouped by the one capability each is waiting on."""
    seen: dict[str, list] = {}
    for i in items:
        if i["needs"]:
            seen.setdefault(i["needs"], []).append(i)
    return [{"key": k, **BLOCKED_NEEDS[k], "count": len(rows),
             "rows": [{"n": r["n"], "name": r["name"], "why": r["coverage_why"]}
                      for r in sorted(rows, key=lambda r: r["n"])]}
            for k, rows in sorted(seen.items(), key=lambda kv: -len(kv[1]))]


def _empty() -> dict:
    return {"found": False, "path": "", "total": 0, "from_rows": 0,
            "subsections": [], "kinds": [], "runs": 0, "dialled": 0,
            "collapsed": [], "faults": [], "coverage_words": COVERAGE,
            "blocked_classes": BLOCKED_CLASSES, "needs": [], "run_totals": {}}


def load() -> dict:
    """Read the library and sort it. Every row lands in exactly one subsection.

    There is no classification logic left in this file. The CSV states each row's
    kind, group, coverage and blocker in its own columns, so this function reads
    them and counts. Anything it cannot make sense of goes in `faults` and is
    shown on the page, because a row this file quietly guesses at is a row nobody
    ever finds out is wrong.
    """
    path = _csv_path()
    if path is None:
        return _empty()

    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("id") or "").strip().isdigit()]

    by_group: dict[str, list] = {s["id"]: [] for s in SUBSECTIONS}
    faults: list[dict] = []

    def fault(n, name, what):
        faults.append({"n": n, "name": name, "what": what})

    for r in rows:
        n = int(r["id"])
        name = (r.get("name") or "").strip()
        group = (r.get("group") or "").strip().upper()
        cover = (r.get("coverage") or "").strip().lower()
        needs = (r.get("blocked_on") or "").strip().lower()
        kind = (r.get("kind") or "").strip().lower()

        if group not in by_group:
            fault(n, name, f"group {group!r} is not one of {', '.join(by_group)}")
            continue
        if cover not in VALID_COVERAGE:
            fault(n, name, f"coverage {cover!r} is not one of "
                           f"{', '.join(sorted(VALID_COVERAGE))}")
            cover = "condition"
        # Only blocked rows are waiting on anything, and a blocked row that names
        # no blocker is a hole in the CSV rather than a fact about the robot.
        if cover == "blocked" and not needs:
            fault(n, name, "blocked, but blocked_on is empty")
            needs = "unsaid"
        elif cover != "blocked" and needs:
            fault(n, name, f"blocked_on is {needs!r} but the row is not blocked")
            needs = ""
        if needs and needs not in VALID_NEEDS:
            fault(n, name, f"blocked_on {needs!r} is not one of "
                           f"{', '.join(sorted(VALID_NEEDS))}")
            needs = "unsaid"
        if kind and kind != _KIND_OF[group]:
            fault(n, name, f"kind {kind!r} but {group} is a {_KIND_OF[group]}")

        old = [int(t) for t in (r.get("covers") or "").split() if t.isdigit()]
        by_group[group].append({
            "n": n,
            "name": name,
            "dial": (r.get("dial") or "").strip(),
            "note": (r.get("note") or "").strip(),
            "coverage": cover,
            # The note doubles as the reason a blocked row is blocked, which is
            # what the grouped-blockers table wants to show.
            "coverage_why": (r.get("note") or "").strip(),
            "needs": needs,
            "covers": old,
        })

    out = []
    for s in SUBSECTIONS:
        items = sorted(by_group[s["id"]], key=lambda i: i["n"])
        # How this subsection's rows are covered, counted. This is the number
        # that answers "what is it actually being trained for" - a run of 20 rows
        # of which 15 are commands and 5 are blocked is a very different job from
        # 20 things to teach.
        cover: dict[str, int] = {}
        for i in items:
            cover[i["coverage"]] = cover.get(i["coverage"], 0) + 1
        out.append({**s, "count": len(items), "skills": items,
                    "coverage": cover,
                    "blocked": [i for i in items if i["coverage"] == "blocked"],
                    "needs": _group_needs(items)})

    # Counts are reported PER KIND and never summed across kinds. One combined
    # "teaches something new" number is what made the old page misleading: it put
    # a training run and a friction value in the same total.
    kinds = []
    for k in KINDS:
        items = [s for s in out if s["kind"] == k["key"]]
        kinds.append({**k, "items": [s["id"] for s in items],
                      "subsections": len(items),
                      "count": sum(s["count"] for s in items)})

    every = [i for s in out for i in s["skills"]]
    # What the collapse did, row by row. The old library said the same thing many
    # times over - five rows for one drop height, fifteen for one parked idea -
    # and `covers` records exactly which of the original 200 each row absorbed.
    # This is the table to read when a row looks like it went missing.
    collapsed = sorted(
        ({"n": i["n"], "name": i["name"], "group": g["id"],
          "kind": g["kind"], "was": len(i["covers"]), "covers": i["covers"],
          "note": i["note"]}
         for g in out for i in g["skills"] if len(i["covers"]) > 1),
        key=lambda c: (-c["was"], c["n"]))

    dialled = sum(1 for i in every if i["dial"])
    run_rows = [i for s in out if s["kind"] == "run" for i in s["skills"]]
    run_totals: dict[str, int] = {"rows": len(run_rows)}
    for i in run_rows:
        run_totals[i["coverage"]] = run_totals.get(i["coverage"], 0) + 1

    return {
        "found": True,
        "path": path.name,
        "total": len(every),
        # How many rows of the original library these stand in for. Printed beside
        # the total so "60 skills" never reads as "the list got shorter".
        "from_rows": sum(len(i["covers"]) for i in every),
        "subsections": out,
        "kinds": kinds,
        "runs": sum(1 for s in out if s["kind"] == "run"),
        "dialled": dialled,
        "collapsed": collapsed,
        "faults": sorted(faults, key=lambda f: f["n"]),
        "coverage_words": COVERAGE,
        "blocked_classes": BLOCKED_CLASSES,
        "run_totals": run_totals,
        # Every blocked row in the library, however it is filed.
        "needs": _group_needs(every),
    }
