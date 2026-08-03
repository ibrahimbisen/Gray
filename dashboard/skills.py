"""The skill library, sorted by what each row actually *is*.

The owner's CSV is the source. This file only decides where each skill belongs -
it never invents a skill and never stores a copy of one. Edit the CSV and the page
changes.

**A skill library is not a work list.** The earlier version of this file sorted 200
rows into nine subsections and gave them equal billing, which made them look like
nine comparable pieces of work. They are not. Four different kinds of thing were
wearing one label, and the count at the top - "128 skills teach something new" -
implied 128 units of work when the real number is three.

The unit of work is a **training run**, because a training run is what produces a
policy file. Everything else is a number inside one, a property checked across all
of them, or a measurement taken afterwards. So each row lands in one of five kinds:

    run         a training run. Produces a policy file. There are three.
    axis        a dial turned during a run. Not a run of its own.
    constraint  a property every run has to hold. Checked, never trained.
    test        measured after training. Nothing new is learned.
    parked      in the plan, not being worked. Each one carries its trigger.

**The test for whether two skills share a run** is not what they look like. It is
whether the *reward function* would have to change. Walking sideways is the same
equation with a different setpoint, so it shares a run. Getting up off its back has
no commanded velocity to track at all, so it needs its own.

That single test is what collapses 200 rows into three runs.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The owner adds to this file. The newest one wins.
CANDIDATES = ("gray_skill_library (1).csv", "gray_skill_library.csv")

# The five kinds, in the order they appear on the page. `blurb` is what the kind
# means; it is shown once, above the items, so each item does not have to re-argue
# why it is not a training run.
KINDS = [
    {"key": "run", "name": "Training runs",
     "blurb": "Each one produces a policy file. This is the only kind that is a "
              "piece of work in its own right, and there are three."},
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
            "over.",
     "commands": "vx, vy, yaw, height, pitch, roll",
     "gap": "Three-legged stance, two-legged diagonal stance and single-leg reach "
            "are filed here but are NOT in the command vector above. They need a "
            "per-foot command that does not exist yet. Listed rather than buried.",
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
    {"id": "R3", "key": "tool", "kind": "run", "name": "Using a foot on something",
     "rule": "the goal is the object, not the robot",
     "what": "Push an object, drag it backward, press a pedal, kick, dig, paw or tap.",
     "why": "Fails the shared-reward test in the other direction: success is "
            "measured on something that is not the robot. It needs objects in the "
            "scene, which nothing else does, and a reward that reads the object's "
            "state. Optional - nothing else in the project depends on it.",
     "commands": "where to move the object",
     "state": "not started"},

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
#   blocked   needs a capability that does not exist yet - a camera, a
#             per-foot command, an object in the scene. No amount of reward
#             tuning produces it.
#   measured  read off a trained policy. Never trained toward.
COVERAGE = {
    "command":   "set a command and it happens - already covered",
    "condition": "a dial in the scene - same reward, different world",
    "emerges":   "nobody asks for it; it falls out of the reward",
    "blocked":   "needs a capability that does not exist yet",
    "measured":  "read off a trained policy, never trained toward",
}

_COVERAGE_BY_CATEGORY = {
    "Walking": "command", "Turning": "command", "Body control": "command",
    "Body positions": "command", "Gaits": "command",
    "Terrain": "condition", "Payload": "condition", "Disturbance": "condition",
    "Degraded hardware": "condition", "People": "condition",
    "Hard variants": "condition", "Stuck": "condition",
    "Feet": "emerges", "Foot movements": "emerges",
    "Falls & recovery": "command", "Startup": "command",
    "Athletic": "blocked", "Air movements": "blocked",
    # These need to know where the robot IS, or where the walls are. That is
    # localisation and perception - a layer above the policy that does not
    # exist. Filing them as training targets overstates what the reward can do.
    "Perception": "blocked", "Precision": "blocked", "Path shapes": "blocked",
    "Speed": "measured", "Smoothness": "measured", "Self-check": "measured",
    "Chaining": "measured",
}

# Rows whose coverage is not what their category implies. Same idea as
# _OVERRIDE, and each one carries why.
_COVERAGE_OVERRIDE = {
    8:  ("blocked", "standing on three legs needs a per-foot command; the vector "
                    "has vx, vy and yaw and nothing per leg"),
    9:  ("blocked", "same - two diagonal legs is a per-foot command"),
    10: ("blocked", "holding one leg out is a per-foot command"),
    21: ("blocked", "placing a foot on a marked target needs to see the target"),
    22: ("emerges", "a low object is felt through the leg, not seen"),
    37: ("emerges", "a cable or rug edge is felt on contact"),
    59: ("emerges", "an unseen hole is exactly that - felt, not seen"),
    99: ("blocked", "'see an obstacle' is the whole point - needs a camera"),
    130: ("blocked", "stepping stones need to see where the stones are"),
}

# Where a category goes by default...
_DEFAULT = {
    # Everything on its feet is one run. Body control and body positions used to
    # be their own subsection; they are commands into R1, not a separate policy.
    "Body control": "walk", "Body positions": "walk",
    "Turning": "walk", "Feet": "walk", "Walking": "walk", "Precision": "walk",
    "Gaits": "walk", "Path shapes": "walk", "Foot movements": "walk",
    "Athletic": "flight", "Air movements": "flight",
    "Falls & recovery": "recover", "Startup": "recover",
    "Disturbance": "robust", "Terrain": "robust", "Payload": "robust",
    "Degraded hardware": "robust", "Stuck": "robust", "Hard variants": "robust",
    "People": "robust",
    # Read off a trained policy rather than trained toward.
    "Speed": "measure", "Smoothness": "measure",
    "Self-check": "quality",
    "Chaining": "chain",
    "Perception": "see",
}

# ...and the skills whose category is not where they belong. Each one is a
# judgement, so each one carries its reason.
_OVERRIDE = {
    23: ("tool", "the goal is the object, not the robot"),
    38: ("robust", "a treadmill is a condition, not a skill"),
    89: ("flight", "taller than a leg - there is no way over without a push-off"),
    90: ("walk", "squeezing under is crouched walking"),
    91: ("walk", "walking, plus a decision"),
    95: ("quality", "IMU noise is simulator honesty, not damage"),
    96: ("quality", "command delay is simulator honesty, not damage"),
    106: ("robust", "depends what the event is; mostly it is a shove"),
    107: ("recover", "felt through the IMU - no eyes needed"),
    108: ("robust", "a tug is just a force"),
    125: ("quality", "how it behaves, not what the world does"),
    134: ("quality", "quiet is a penalty on effort, not something measured after"),
    137: ("recover", "it gets set down in an unknown pose"),
    138: ("see", "it has to see the foot coming"),
    142: ("see", "there has to be a camera for it to be blocked"),
    143: ("see", "degraded vision"),
    158: ("robust", "tripping is a disturbance, not a speed measurement"),
    162: ("flight", "a pronk pushes off with all four - there is a flight phase"),
    163: ("flight", "a gallop has a flight phase"),
    164: ("flight", "a hop has a flight phase"),
    169: ("recover", "a roll starts on the ground"),
    170: ("walk", "crawling still travels"),
    177: ("quality", "a power state, not a movement"),
    180: ("tool", "the goal is the object"),
    181: ("tool", "the goal is the object"),
    182: ("tool", "the goal is the object"),
    183: ("tool", "the goal is the object"),
    184: ("tool", "the goal is the object"),
}

# Kept for anything still importing it. `kind == "run"` is the real answer now.
TEACHES_NEW = tuple(s["key"] for s in SUBSECTIONS if s["kind"] == "run")

_KIND_OF = {s["key"]: s["kind"] for s in SUBSECTIONS}


def _csv_path() -> Path | None:
    for name in CANDIDATES:
        p = ROOT / name
        if p.is_file():
            return p
    return None


def _empty() -> dict:
    return {"found": False, "path": "", "total": 0, "subsections": [],
            "kinds": [], "runs": 0, "dialled": 0, "moved": []}


def load() -> dict:
    """Read the library and sort it. Every skill lands in exactly one subsection."""
    path = _csv_path()
    if path is None:
        return _empty()

    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("#") or "").strip().isdigit()]

    by_key: dict[str, list] = {s["key"]: [] for s in SUBSECTIONS}
    moved = []
    for r in rows:
        n = int(r["#"])
        category = (r.get("Category") or "").strip()
        if n in _OVERRIDE:
            key, reason = _OVERRIDE[n]
            moved.append({"n": n, "skill": r["Skill / Event"], "from": category,
                          "to": key, "reason": reason})
        else:
            key = _DEFAULT.get(category)
        if key is None:            # a category nobody has classified yet
            key = "robust"
        if n in _COVERAGE_OVERRIDE:
            cover, cover_why = _COVERAGE_OVERRIDE[n]
        else:
            cover = _COVERAGE_BY_CATEGORY.get(category, "condition")
            cover_why = ""
        by_key[key].append({
            "n": n,
            "name": (r.get("Skill / Event") or "").strip(),
            "dial": (r.get("Difficulty dial") or "").strip().lstrip("-").strip(),
            "category": category,
            "coverage": cover,
            "coverage_why": cover_why,
        })

    out = []
    for s in SUBSECTIONS:
        items = sorted(by_key[s["key"]], key=lambda i: i["n"])
        # How this subsection's rows are covered, counted. This is the number
        # that answers "what are we actually training for" - a run with 69 rows
        # of which 40 are commands and 9 are blocked is a very different job
        # from 69 things to teach.
        cover: dict[str, int] = {}
        for i in items:
            cover[i["coverage"]] = cover.get(i["coverage"], 0) + 1
        out.append({**s, "count": len(items), "skills": items,
                    "coverage": cover,
                    "blocked": [i for i in items if i["coverage"] == "blocked"],
                    "categories": sorted({i["category"] for i in items})})

    # Counts are reported PER KIND and never summed across kinds. One combined
    # "teaches something new" number is what made the old page misleading: it put
    # a training run and a friction value in the same total.
    kinds = []
    for k in KINDS:
        items = [s for s in out if s["kind"] == k["key"]]
        kinds.append({**k, "items": [s["id"] for s in items],
                      "subsections": len(items),
                      "count": sum(s["count"] for s in items)})

    dialled = sum(1 for r in rows if (r.get("Difficulty dial") or "").strip() not in ("-", ""))
    for m in moved:
        m["to_name"] = next(s["name"] for s in SUBSECTIONS if s["key"] == m["to"])
        m["to_kind"] = _KIND_OF[m["to"]]
    return {
        "found": True,
        "path": path.name,
        "total": len(rows),
        "subsections": out,
        "kinds": kinds,
        "runs": sum(1 for s in out if s["kind"] == "run"),
        "dialled": dialled,
        "moved": sorted(moved, key=lambda m: m["n"]),
    }
