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

# A row that _OVERRIDE relocated must take its coverage from where it LANDED, not
# from where it was filed. Without this, "Squeeze under something" - filed under
# Athletic, moved to R1 because squeezing under is crouched walking - still read
# as blocked on a flight phase, and "IMU drift and noise" read as blocked on a
# camera. Both are nonsense, and both were on the page.
_COVERAGE_BY_SUBSECTION = {
    "walk": "command", "recover": "command", "tool": "emerges",
    "robust": "condition", "quality": "condition",
    "measure": "measured", "chain": "measured",
    "flight": "blocked", "see": "blocked",
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
    # Height and attitude. The command vector is vx, vy, yaw - there is no ride
    # height in it and no lean, so none of these can be ASKED for, whatever the
    # policy is capable of. This is the same fact UNVERIFIED_CLAUSES states in
    # prose for R1's bar; listing the rows makes the count agree with the prose.
    1:  ("blocked", "sit to stand is a height command, and there is no height in "
                    "the command vector"),
    2:  ("blocked", "a commanded height"),
    3:  ("blocked", "a commanded height"),
    4:  ("blocked", "a commanded height"),
    5:  ("blocked", "this row IS the missing command, named"),
    6:  ("blocked", "pitch and roll are not in the command vector"),
    8:  ("blocked", "standing on three legs needs a per-foot command; the vector "
                    "has vx, vy and yaw and nothing per leg"),
    9:  ("blocked", "same - two diagonal legs is a per-foot command"),
    10: ("blocked", "holding one leg out is a per-foot command"),
    21: ("blocked", "placing a foot on a marked target needs to see the target"),
    22: ("emerges", "a low object is felt through the leg, not seen"),
    37: ("emerges", "a cable or rug edge is felt on contact"),
    59: ("emerges", "an unseen hole is exactly that - felt, not seen"),
    90: ("blocked", "clearance to 60% body height is a crouch, and a crouch is a "
                    "height command - not a jump, whatever the category says"),
    91: ("pilot", "the pilot can see the corner and steers out of it"),
    99: ("blocked", "'see an obstacle' is the whole point - needs a camera"),
    # ---- the localisation re-audit, 3 Aug 2026 -----------------------------
    # These thirteen were all filed as blocked on "knowing where it is". Six of
    # them were simply misfiled and are not about position at all; the other
    # seven are, and a human at the controls supplies it. Each is named rather
    # than swept, because "13 blocked" was wrong in two different directions.
    126: ("pilot", "the pilot decides where the point is and stops there"),
    127: ("pilot", "the pilot is looking at it and lets go on the heading wanted"),
    128: ("pilot", "reversing into a space is what a human on sticks is for"),
    129: ("command", "walk straight is vx with vy and yaw at zero - and holding "
                     "that line for 3 m IS R1's existing drift bar, measured in "
                     "simulation where position is free"),
    130: ("blocked", "stepping stones need to see where the stones are"),
    131: ("pilot", "lining up on a doorway is a steering job"),
    193: ("pilot", "a figure eight is a sequence of turn commands, flown"),
    194: ("command", "a constant forward speed with a constant turn rate IS a "
                     "circle - two command numbers held, nothing more"),
    195: ("pilot", "backing out of a dead end is steering"),
    196: ("emerges", "a wall is felt through the leg when it pushes against it, "
                     "not seen - the same way a low obstacle already is"),
    197: ("condition", "a steep slope is terrain, which is a dial in the scene"),
    198: ("condition", "same - the slope is the condition, backward is a command"),
    199: ("blocked", "skipping a stair means knowing where the next one is, which "
                     "is the camera, not position on the floor"),
    170: ("blocked", "belly near the ground is the bottom of the height range"),
    174: ("blocked", "leaning with the feet planted is a pitch and roll command"),
    175: ("blocked", "scanning with the feet planted is a yaw attitude, not a yaw rate"),
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
                       "it an entry in _NEED_OVERRIDE or _NEED_BY_CATEGORY."},
}

# Precision and Path shapes used to map to "where". Both are gone: every row in
# them is now named individually, and a NEW row in either should surface as a
# gap ("unsaid") rather than be quietly filed against a retired blocker.
_NEED_BY_CATEGORY = {
    "Athletic": "flight", "Air movements": "flight",
    "Perception": "camera",
    "Body control": "height", "Body positions": "height",
}

# A relocated row's blocker follows where it landed, same rule as its coverage.
_NEED_BY_SUBSECTION = {"flight": "flight", "see": "camera"}

# Rows whose blocker is not what their category implies.
_NEED_OVERRIDE = {8: "foot", 9: "foot", 10: "foot",
                  21: "camera", 99: "camera", 130: "camera", 199: "camera",
                  90: "height"}

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


def _by_category(items: list[dict]) -> list[dict]:
    """Category by coverage, biggest first.

    This is the table that answers "what is it actually being trained for" one
    level down from the totals: 22 rows of Walking that are all one command is a
    very different thing from 6 rows of Precision that no reward can reach.
    """
    seen: dict[str, dict] = {}
    for i in items:
        row = seen.setdefault(i["category"] or "(none)",
                              {"category": i["category"] or "(none)", "count": 0,
                               "coverage": {}})
        row["count"] += 1
        row["coverage"][i["coverage"]] = row["coverage"].get(i["coverage"], 0) + 1
    return sorted(seen.values(), key=lambda r: (-r["count"], r["category"]))


def _empty() -> dict:
    return {"found": False, "path": "", "total": 0, "subsections": [],
            "kinds": [], "runs": 0, "dialled": 0, "moved": [],
            "coverage_words": COVERAGE, "needs": [], "run_totals": {}}


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
        # Coverage follows where the row LANDED when it was relocated, and its
        # filed category otherwise. A named override beats both.
        moved_here = n in _OVERRIDE
        if n in _COVERAGE_OVERRIDE:
            cover, cover_why = _COVERAGE_OVERRIDE[n]
        elif moved_here:
            cover = _COVERAGE_BY_SUBSECTION.get(key, "condition")
            cover_why = ""
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
            # Only blocked rows are waiting on anything. Everything else is
            # already reachable, so `needs` stays empty rather than inventing a
            # blocker for a row that has none.
            "needs": (_NEED_OVERRIDE.get(n)
                      or (_NEED_BY_SUBSECTION.get(key) if moved_here else None)
                      or _NEED_BY_CATEGORY.get(category)
                      or "unsaid") if cover == "blocked" else "",
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
                    "needs": _group_needs(items),
                    "by_category": _by_category(items),
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
    # The three runs added up, which is the only cross-subsection total that is
    # honest: they are the same kind of thing. Kinds are still never summed.
    run_rows = [i for s in out if s["kind"] == "run" for i in s["skills"]]
    run_totals: dict[str, int] = {"rows": len(run_rows)}
    for i in run_rows:
        run_totals[i["coverage"]] = run_totals.get(i["coverage"], 0) + 1

    return {
        "found": True,
        "path": path.name,
        "total": len(rows),
        "subsections": out,
        "kinds": kinds,
        "runs": sum(1 for s in out if s["kind"] == "run"),
        "dialled": dialled,
        "moved": sorted(moved, key=lambda m: m["n"]),
        "coverage_words": COVERAGE,
        "blocked_classes": BLOCKED_CLASSES,
        "run_totals": run_totals,
        # Every blocked row in the library, however it is filed.
        "needs": _group_needs([i for s in out for i in s["skills"]]),
    }
