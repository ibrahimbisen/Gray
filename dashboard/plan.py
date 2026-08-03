"""The plan, as data.

Everything the dashboard shows lives here, not in the HTML. To change what the
robot is being taught, or how it is scored, edit this file. The pages are only
renderers.

The project has three stages, and they are a LOOP rather than a line:

    1  PREPARE   a digital Gray worth trusting      provisional
    2  TRAIN     teach it, in simulation            here, and for a long time
    3  DEPLOY    put it on the real robot           start now, in parallel
       |                                                  |
       +------- real masses, real servos, real delay -----+

Stage 1 is not finished and saying it is was the plan contradicting itself: three
of the numbers in it have no datasheet and can only come off the physical robot,
which is stage 3. Stage 3 therefore does not wait for stage 2 - it feeds stage 1.

Stage 2 is the one with depth. It is broken up in dashboard/skills.py, which sorts
the owner's skill library live from the CSV - by what each row actually IS, not by
what it looks like. Three training runs, one dial, one constraint, two test groups,
two parked.

The reward terms on the page are NOT written here. They are read off the task
configs in gray/tasks/, so the page cannot drift from the code the way the old
hand-written list did.
"""

from __future__ import annotations

from pathlib import Path

from dashboard import skills

ROOT = Path(__file__).resolve().parent.parent

# The files rewards() reads. Listed as paths so their mtimes can be checked without
# importing torch, which is the whole reason that import is lazy.
_TASK_FILES = tuple(
    ROOT / "gray" / "tasks" / f"{name}_env_cfg.py"
    for name in ("stand", "push", "walk")
)

# ---------------------------------------------------------------------------
# What we are training for
# ---------------------------------------------------------------------------

GOAL = {
    "one_line": "Teach Gray to walk, and then walk on the real robot.",
    "paragraphs": [
        "Gray is a 12-DOF quadruped: four legs, three joints each. A controller has "
        "to decide, fifty times a second, what angle every one of those twelve joints "
        "should be at. Nobody can write those numbers by hand for every situation, so "
        "we train a program to produce them.",
        "What training produces is not a recording of a good walk. It is a single "
        "function - sensors and a command in, twelve joint angles out - and the "
        "command is what makes one function cover forward, backward, sideways, "
        "turning and standing still. Change the command, not the file.",
        "The training happens in a physics simulator, not on the real robot. Thousands "
        "of copies of Gray run at once on the graphics card, each in a different "
        "situation. Each copy tries something, gets a score, and the ones that score "
        "better shape what all of them try next. After a few hours this produces a "
        "single small file - a few hundred kilobytes - that goes on the Raspberry Pi "
        "and drives the real robot.",
        "The whole game is making the simulator honest enough that a controller which "
        "works in it also works outside it. Where a number cannot be measured, it gets "
        "randomised instead - a policy that survives the whole range survives the real "
        "value, whatever it turns out to be.",
    ],
    "success": "Gray walks across a floor, in a straight line, without falling, "
               "driven by a policy trained entirely in simulation.",
}

# ---------------------------------------------------------------------------
# The three stages. A loop, not a line - see LOOP below.
# ---------------------------------------------------------------------------

PROJECT = [
    {
        "n": 1,
        "name": "Prepare",
        "state": "provisional",
        "one_line": "A digital Gray worth trusting.",
        "summary": "Rebuild the robot in SolidWorks, export it, and turn it into "
                   "something a physics simulator can run - then prove it matches "
                   "the real machine rather than assuming it does.",
        "why": "Everything in stage 2 is measured inside the simulator. If the "
               "simulated robot is the wrong mass, or its joints bend further than "
               "the real ones can, then every hour of training after that teaches a "
               "robot that does not exist. This is the stage the last attempt got "
               "wrong, and it cost the entire project.",
        "headline": "11 of 11 checks pass. 2030 g, 13 links, 12 joints, travel limits "
                    "verified in the simulator. Not done: the mass is a parts-list "
                    "estimate until the parts session, and three numbers have no "
                    "datasheet at all.",
        "page": "/stage1",
    },
    {
        "n": 2,
        "name": "Train",
        "state": "in progress",
        "one_line": "Teach it, in simulation.",
        "summary": "200 skills in the library, sorted by what each row actually is. "
                   "Three training runs, one randomisation dial, one constraint, two "
                   "groups of tests, two parked.",
        "why": "This is where the project lives for a long time, and it is far smaller "
               "than 200 rows makes it look. The unit of work is a training run, not a "
               "skill: the largest group of rows is ONE run with different commands, "
               "the next largest is the range that run is sampled over, and the rest "
               "are checked or measured rather than trained. Live counts are on the "
               "stage 2 page - they are read from the CSV, never written down here.",
        "headline": "R1 locomotion is training. R2 getting up and R3 foot-on-object "
                    "not started. Jumping and seeing are parked, each with a trigger.",
        "page": "/stage2",
    },
    {
        "n": 3,
        "name": "Deploy",
        "state": "start now",
        "one_line": "Put it on the real Gray.",
        "summary": "Reassemble, fit the potentiometers and the ADC, calibrate every "
                   "sensor, measure the real servos, and run the policy on the Pi at "
                   "50 Hz.",
        "why": "It does not depend on stage 2 and it should not wait for it. It is the "
               "long pole - the robot is in pieces and no parts are bought - and step "
               "3.3 produces the three numbers stage 1 is currently guessing. Waiting "
               "means finding out at skill 200 what could have been found out at "
               "skill 1.",
        "headline": "Nothing started. Now the parallel track rather than the last "
                    "stage, because it is what closes the loop back to stage 1.",
        "page": "/stage3",
    },
]

# The feedback edge, stated as data so the page cannot draw a line while the text
# argues for a loop. That contradiction is what this replaces.
LOOP = {
    "one_line": "Stage 3 measures what stage 1 is guessing. That is why it starts now.",
    "edges": [
        {"from": "3.3", "to": "Stage 1",
         "what": "Real servo stiffness and speed under load",
         "why": "kp = 21 and kv = 0.6 were chosen so the worst joint sags about 1.5 "
                "degrees. A hobby servo's internal controller is sealed and "
                "unpublished, so no datasheet can settle this. Until then it is "
                "randomised +/-30%."},
        {"from": "3.3", "to": "Stage 1",
         "what": "True loop latency",
         "why": "Sensor read, network, servo response. Guessed today. Measurable in "
                "an afternoon once the ADC and pots are on."},
        {"from": "3.1", "to": "Stage 1",
         "what": "Backlash and slop, and weighed parts",
         "why": "Ball joints, gear trains and printed fits all have play, and the "
                "simulator has none. Weighing one calf, one thigh and one hip during "
                "reassembly also confirms the trunk-to-legs split, which has already "
                "been wrong twice."},
        {"from": "3.3", "to": "Stage 2 / P1",
         "what": "Whether jumping is physically possible",
         "why": "The measured servo speed answers it for free. Until then, spending "
                "the calculation would mean spending it on a datasheet guess."},
    ],
    "cost": "A retrain is about 3000 iterations - a few hours. Changing the CAD and "
            "re-exporting is cheap too. Neither is a reason to delay starting.",
}

# ---------------------------------------------------------------------------
# Stage 1 - the model.
# ---------------------------------------------------------------------------

STAGE1 = {
    "lede": "A simulator is only ever as honest as the model inside it. This is "
            "what the model is, where every number in it came from, and which ones "
            "are still guesses.",
    "story": [
        "The previous attempt at this project was built on a SolidWorks export that "
        "was wrong at the source: Y-up instead of Z-up, no joint limits at all, servo "
        "geometry fused into the link meshes, and joint axes up to 0.9 degrees off "
        "true. Every one of those faults was patched downstream in Python, so the "
        "whole software stack rested on corrections to corrections. It was archived to "
        "the git tag archive/attempt-1 and nothing from it carried forward.",
        "This time the robot was remodelled in SolidWorks from scratch and exported "
        "properly. What follows is not the export - it is what had to be done to the "
        "export before it described the real machine.",
        "This stage is marked PROVISIONAL rather than done, and that is not "
        "pessimism. Eleven of eleven checks pass, which says the model is "
        "self-consistent. It does not say the model is the robot. Three numbers in it "
        "have no datasheet and can only come off the physical machine.",
    ],
    "facts": [
        {"k": "Mass", "v": "2030 g", "n": "parts-list estimate - the parts session replaces this"},
        {"k": "Links / joints", "v": "13 / 12", "n": "trunk, four hips, four thighs, four calves"},
        {"k": "Ride height", "v": "190 mm", "n": "70% of the 272 mm it can reach"},
        {"k": "Footprint", "v": "280 x 186 mm", "n": "feet 34 mm outboard of each hip"},
        {"k": "Torque margin", "v": "4.18x", "n": "worst joint needs 0.47 of 1.96 N-m"},
        {"k": "Control rate", "v": "50 Hz", "n": "hard limit - the servo PWM period is 20 ms"},
    ],

    # The session that takes stage 1 from four guesses to three.
    "parts": {
        "intro": "The largest uncertainty in stage 1 is not hard to remove - it just "
                 "has not been done. Most of the mass build-up is a datasheet lookup "
                 "away, and the printed plastic, which is the biggest unknown of all, "
                 "is a number the owner's slicer already knows.",
        "why_slicer": "The slicer is not an estimate. It knows the real toolpath at "
                      "the real infill and wall count, and reports actual grams per "
                      "part. That single number replaces the whole 'PLA costed at 22% "
                      "infill rather than solid' guess, which is where most of the "
                      "error in 2030 g lives.",
        "needed": [
            {"item": "Servos", "ask": "Confirm 12 x DS3218MG. Any joint different?"},
            {"item": "Battery", "ask": "Exact pack - mAh, cell count, brand and model"},
            {"item": "Compute", "ask": "Which Pi, and the power board feeding it"},
            {"item": "Servo driver", "ask": "Which PCA9685 breakout"},
            {"item": "ADC", "ask": "Which 12-channel chip is planned - placeholder mass if unbought"},
            {"item": "Potentiometers", "ask": "Rotary model, linear model, and how many of each"},
            {"item": "IMU", "ask": "Which part, and confirm three on the trunk"},
            {"item": "Printed parts", "ask": "The slicer's reported grams per part, at the real "
                                            "infill and wall count. Plus material and printer."},
            {"item": "Hardware", "ask": "Ball joints, pushrods, bearings, fastener sizes and counts"},
            {"item": "Wiring", "ask": "Rough gauge and total length"},
        ],
        "output": "gray/config/parts.yaml, and a mass split that is looked up rather "
                  "than assumed. The split between trunk and legs matters more than "
                  "the total - it is what a swing costs and what a shove has to stop, "
                  "and it has already been wrong twice.",
        "cannot_fix": "Servo gains, backlash and loop latency. No datasheet exists for "
                      "any of them. They stay randomised, which is the correct handling "
                      "and not a compromise - see 'Still assumed'.",
    },

    "mass": {
        "intro": "The CAD masses were dropped, because measuring the exported meshes "
                 "against them showed two faults pulling in opposite directions. What "
                 "replaced them is a parts list, which is better - and which the parts "
                 "session replaces again with datasheet and slicer numbers.",
        "rows": [
            {"part": "Trunk", "g": 914, "n": "holds 4 of the 12 servos"},
            {"part": "Hip (x4)", "g": 165, "n": "two servos each - thigh drive and calf drive"},
            {"part": "Thigh (x4)", "g": 65, "n": "no servo; a printed arm and a pushrod"},
            {"part": "Calf (x4)", "g": 49, "n": "no servo"},
        ],
        "total": 2030,
        "notes": [
            "The thigh and calf came out of SolidWorks at almost exactly 1.0 g/cm3 - "
            "water. That is the exporter writing volume in cm3 into the mass field, so "
            "those numbers were not masses at all.",
            "The hip and trunk were denser than water and split into 85 and 551 "
            "separate closed bodies, so those did contain servos and fasteners as real "
            "geometry - but their printed plastic was still priced as solid, and a part "
            "printed at 22% infill weighs roughly 45% of solid.",
            "So the CAD was simultaneously too heavy and too light. Adding up known "
            "part weights avoids both: 12 servos at 60 g, a 3S LiPo, the Pi, the "
            "boards, the pots, the wiring, and printed PLA costed at infill.",
            "The split matters more than the total. It has already been wrong twice - "
            "once putting a knee servo on each thigh (legs came out 67% of the robot), "
            "once reading the build photos as 8 servos in the body (42%). The owner's "
            "answer - 4 in the body, 2 in each hip, none in the thigh or calf - puts "
            "it at 55%, and leg mass is what a swing costs and what a shove has to stop.",
            "Every number above is still an estimate. The parts session is what turns "
            "the printed-plastic rows into slicer output and the rest into datasheet "
            "lookups, and weighing three parts during reassembly checks the answer.",
        ],
    },
    "limits": {
        "intro": "Measured by the owner in the pose editor, against the CAD, and then "
                 "verified by turning each joint in the simulator and reading back "
                 "where it actually stopped. All twelve match.",
        "rows": [
            {"joint": "Hip", "range": "-65 to +40 deg", "n": "+ swings the leg out, away from the body"},
            {"joint": "Thigh", "range": "-35 to +45 deg", "n": "+ swings the leg forward, towards the nose"},
            {"joint": "Calf", "range": "-85 to +1 deg", "n": "+ lifts the foot up"},
        ],
    },
    "fixed": [
        {"t": "Inertial frames were still Y-up",
         "d": "The exporter converts joint origins to Z-up but leaves inertial origins "
              "in the old frame. base_link's centre of mass came out at x = -2.39 m - "
              "two and a half metres behind the robot - and it toppled backwards "
              "instantly. Fixed by checking each centre of mass against its own mesh "
              "bounding box and rotating only the ones that land outside it."},
        {"t": "Two legs exported 180 degrees rotated",
         "d": "The front-right and back-left thighs came out upside down. Fixed by "
              "trying each thigh both ways and keeping whichever puts the knee below "
              "the hip."},
        {"t": "Decimating the meshes shredded the trunk",
         "d": "An STL gives every triangle its own copy of its vertices, so the "
              "simplifier had no idea which triangles were neighbours and tore the "
              "shape apart. Merging duplicate vertices first took the error from "
              "4.51 mm down to 0.08 mm."},
        {"t": "MuJoCo welds the root link to the world",
         "d": "Silently. The robot cannot move and the trunk's mass vanishes - 914 g "
              "of 2030 g. Giving base_link a free joint fixes it, and the model check "
              "now confirms the simulator sees the full mass."},
        {"t": "Geoms had no names",
         "d": "The URDF importer leaves them unnamed, so nothing that selects bodies "
              "by pattern could match anything. Named on the way through."},
    ],

    # Three, not four. The mass one is being closed by the parts session, so it is
    # listed as work rather than as an unknown.
    "open": [
        {"q": "The servo gains have never been measured.",
         "n": "kp = 21 N-m/rad and kv = 0.6 were chosen so the worst-loaded joint sags "
              "about 1.5 degrees, which is roughly what these servos do. A hobby "
              "servo's internal controller is sealed and unpublished, so no research "
              "can settle it - only the physical robot can. Handled by randomising "
              "+/-30% around the guess, which is the correct answer rather than a "
              "workaround: a policy that survives the range survives the truth. "
              "Measured in stage 3.3."},
        {"q": "Backlash and slop are not modelled.",
         "n": "Ball joints, gear trains and printed parts all have play. The simulator "
              "has none, so the policy never learns to expect it. This is the usual "
              "reason a controller that works in simulation does not work outside it. "
              "Joint friction is randomised as a partial stand-in; the real figure "
              "needs the assembled robot."},
        {"q": "The true loop latency is a guess.",
         "n": "Sensor read, ADC conversion, network, servo response. Randomised as "
              "command delay today. Measurable in an afternoon once the pots and the "
              "ADC are fitted, in stage 3.3."},
        {"q": "bl_hip's inertia is assumed, and the masses are not weighed yet.",
         "n": "The exporter gave bl_hip no mass, so its tensor is copied from another "
              "hip and mirrored - same part, so it is close. The mass build-up is "
              "being replaced by the parts session, and weighing one calf, one thigh "
              "and one hip during reassembly confirms it. This is the one item on this "
              "list with a fix that needs no hardware."},
    ],
}

# ---------------------------------------------------------------------------
# Stage 2 - the subsections come from skills.py, which reads the owner's CSV.
# What lives here is only what each one has to PASS.
# ---------------------------------------------------------------------------

STAGE2 = {
    "lede": "200 skills, sorted by what each row actually IS - three training runs, "
            "one dial, one constraint, two groups of tests, two parked.",
    "intro": [
        "The library is the owner's. This page does not invent skills or store a copy "
        "of them; it reads the CSV and sorts it. Edit the CSV and this page changes.",
        "The unit of work is a TRAINING RUN, because a run is what produces a policy "
        "file. Counting skills instead makes the project look forty times bigger than "
        "it is: 'walk forward' and 'walk backward' are the same run with a sign "
        "flipped, and 'gravel' is not a skill at all - it is a number in the scene.",
        "The test for whether two rows share a run is whether the REWARD FUNCTION "
        "would have to change. Walking sideways is the same equation with a different "
        "setpoint, so it shares a run. Getting up off its back has no commanded "
        "velocity to track, so every tracking term is meaningless and it needs its "
        "own. That test is not a judgement call - write the reward down and look.",
        "Applied to 200 rows it gives three runs. Everything else is a dial turned "
        "during a run, a property checked across all of them, or a measurement taken "
        "afterwards - and each of those is cheap in a way a run is not.",
    ],
    "bars": {
        "walk": "Walks 5 m without falling, holds commanded speed within 0.05 m/s, "
                "drifts under 100 mm sideways over 20 s - and holds any commanded "
                "height and attitude at zero speed for 30 s, trunk within 5 mm of "
                "target, uprightness above 0.99.",
        "recover": "Stands up from 9 of 10 random ground poses in under 3 s, and "
                   "hands control back to R1 upright and stable.",
        "tool": "Moves the target object by the commanded amount without falling.",
        "robust": "Every bar above still holds with the dials at full.",
        "quality": "The observation space contains nothing Gray cannot measure, and "
                   "no commanded motion exceeds what 1.96 N-m at 50 Hz can produce.",
        "chain": "Runs the whole sequence end to end without a fall or a stall, "
                 "including every R1 to R2 handover.",
        "measure": "Not a bar - a number written down. These are read off a trained "
                   "policy, not passed.",
        "flight": "Parked. First it has to be shown the servos can leave the ground "
                  "at all, and that needs the measured servo speed from stage 3.3.",
        "see": "Parked. Not definable - Gray has no cameras fitted.",
    },
    "order": [
        "R1 first and for a long time. Everything else either leans on it or is "
        "measured against it, and it is the only run on the critical path to a robot "
        "that walks on a floor.",
        "D1 runs alongside R1, not after it. It is already on - ground friction, mass, "
        "centre of mass, servo gains and joint friction are randomised on every "
        "attempt in push_env_cfg.py. Terrain, payload and damaged servos come next.",
        "C1 is checked continuously and never finished. It is what killed the previous "
        "attempt, and it is cheap while it is a habit and fatal when it is a discovery.",
        "R2 before anything that involves leaving the ground. Jumping means falling, "
        "and a robot should be able to get up before it is taught to jump.",
        "T1 only once R1 and R2 both exist - it tests the handover between them, which "
        "cannot be tested before there are two things to hand over.",
        "R3 last of the runs. It needs objects in the scene, which nothing else does, "
        "and nothing depends on it.",
    ],
}

# ---------------------------------------------------------------------------
# Stage 3 - the parallel track, and the thing that closes the loop.
# ---------------------------------------------------------------------------

STAGE3 = {
    "lede": "Mostly hardware, none of it started, and it is the parallel track rather "
            "than the last stage.",
    "intro": [
        "The robot is in pieces. No potentiometers fitted, no ADC bought, the three "
        "IMUs not wired. That work does not need stage 2 to be finished, and it is "
        "the long pole - if it waits for training to be done, it becomes the thing "
        "that holds up the project.",
        "This stage is not the end of a line. Step 3.3 measures the servo stiffness, "
        "the servo speed and the true loop latency - the three numbers stage 1 is "
        "currently guessing - and those go back into the model. Everything trained "
        "afterwards is more honest for it. That loop is the point of deploying early, "
        "with one skill riding on it, rather than at the end with all of them.",
        "It also settles a stage 2 question for free: whether jumping is physically "
        "possible. That answer is a measured servo speed, not a calculation on a "
        "datasheet number.",
    ],
    "steps": [
        {"n": "3.1", "name": "Reassemble and fit the sensors",
         "d": "Rotary pots on the rotating joints, a linear pot on each knee pushrod, "
              "three IMUs on the trunk, and a 12-channel ADC - the Pi has no analog "
              "inputs, so that chip is a hard requirement. Weigh one calf, one thigh "
              "and one hip on the way past; it costs a minute and checks the mass "
              "split that has already been wrong twice."},
        {"n": "3.2", "name": "Calibrate",
         "d": "Raw ADC counts to radians, per joint, stored in a file. The model is "
              "only as good as that mapping."},
        {"n": "3.3", "name": "Measure the real servos and feed it back",
         "d": "Stiffness, speed under load, and the true loop latency. These replace "
              "the three guesses in the model, which means going back to stage 1 and "
              "retraining - about three thousand iterations, a few hours. The same "
              "measurement decides whether jumping is possible."},
        {"n": "3.4", "name": "Run the policy on the Pi",
         "d": "Export the network and its observation normaliser, and hold 50 Hz on "
              "a Raspberry Pi with no GPU."},
        {"n": "3.5", "name": "Walk on the floor",
         "d": "The whole point."},
    ],
    "software_first": [
        "One observation builder, shared by the simulator and the robot. The policy "
        "reads about 45 numbers every 20 ms; if one joint is in a different order, or "
        "a sign is flipped, the robot thrashes and nothing says why. Two "
        "implementations that agree today will not agree in a month.",
        "The observation normaliser ships with the network. The policy learned a "
        "running mean and standard deviation for those 45 numbers - export the network "
        "without it and the robot does nothing sensible.",
        "A safety layer before first power-on: angle and rate limits, a watchdog, a "
        "soft start, a kill switch.",
        "The knee linkage as a function, both directions. The policy commands a knee "
        "JOINT angle and the servo needs a SERVO angle; the pushrod means they differ. "
        "The linear pot reads the true knee angle, so the measurement is solved - the "
        "command is not.",
        "Logging on the robot, so a real run can be laid over a simulated one. That "
        "comparison is what turns 'it walks badly' into 'the knee is 8 degrees off "
        "under load'.",
    ],
}

# ---------------------------------------------------------------------------
# The pipeline the training monitor shows. Built from the same subsections, so
# there is exactly one plan in this repo rather than two that drift apart.
# ---------------------------------------------------------------------------

STAGES = [
    {
        "n": i + 1,
        "id": s["id"],
        "name": f"{s['id']}  {s['name']}",
        "kind": s["kind"],
        "goal": s["what"],
        "why": s["why"],
        "teaches": [],
        "rewarded": [],
        "bar": STAGE2["bars"].get(s["key"], ""),
        "effort": s["rule"],
        "state": s["state"],
    }
    for i, s in enumerate(skills.SUBSECTIONS)
]

# ---------------------------------------------------------------------------
# How points are given.
#
# The terms are NOT listed here. A hand-written copy drifted from the code - it
# still named track_velocity, foot_clearance, drift and joint_limit long after the
# code had renamed them and added four more - so it was deleted. `rewards()` reads
# the live config instead, and a term with no written explanation shows up on the
# page as a gap rather than vanishing.
# ---------------------------------------------------------------------------

SCORING_INTRO = [
    "The proper name for this is the reward function. Its parts are reward terms, "
    "the multipliers on them are weights, and a term with a negative weight is a "
    "penalty. Those are the words used everywhere else, so they are worth "
    "knowing - but nothing below depends on them.",
    "Fifty times a second, the robot is given a score for what it just did. Good "
    "behaviour adds points, bad behaviour subtracts them. The scores are added up "
    "over an attempt, and the training process pushes the controller toward whatever "
    "scored highest.",
    "That is the entire mechanism. There is no other way to tell it what you want. "
    "If a behaviour is not in this table, the robot has no reason to do it - and if "
    "something unwanted scores well by accident, it will do that instead. Last time a "
    "reward was written badly enough that standing perfectly still scored better than "
    "walking, and the robot dutifully stood still.",
    "Notice what is NOT in the table: there is no 'gravel' term, no 'sideways' term, "
    "no 'recover from a shove' term. The reward says what good MEANS; the "
    "randomisation says which situations you have to be good in. Skills come out of "
    "the second one, not the first. That is why 200 rows collapse into three runs.",
    "Most terms use the same shape: score = exp(-(error squared) / tolerance). That "
    "gives 1.0 for perfect, about 0.6 when the error equals the tolerance, and "
    "approaches 0 beyond it. It is smooth, which matters - a reward that jumps gives "
    "the training process nothing to follow.",
    "One reward function per RUN, not one for the project. Standing still and walking "
    "share theirs, which is why they share a policy. Getting up will need its own.",
]

# The tasks whose reward tables the page shows, in the order they were built. Each
# one inherits from the one before it, so the walking table already contains the
# terms the earlier tasks added.
_REWARD_TASKS = (
    {"key": "stand", "title": "Smoke test - stand still",
     "note": "Not a shipped policy. It answers 'is the model even right?' in minutes "
             "rather than finding out six hours into a walking run. Kept for exactly "
             "that."},
    {"key": "walk", "title": "R1 - locomotion",
     "note": "The run that matters. Inherits every term from standing and shoving, "
             "drops the four that only made sense while standing still, and adds the "
             "gait terms on top."},
)

_REWARDS_CACHE: dict | None = None
_REWARDS_STAMP: float = -1.0


def _tasks_mtime() -> float:
    """Newest mtime across the task files, or -1 if none of them exist."""
    stamps = [f.stat().st_mtime for f in _TASK_FILES if f.is_file()]
    return max(stamps) if stamps else -1.0


def rewards() -> dict:
    """Read the reward terms straight off the task configs in gray/tasks/.

    Imported lazily and cached: this pulls in torch and mjlab, which is slow and
    which the rest of the dashboard does not need. If the import fails - no venv,
    a syntax error mid-edit - the page says so instead of showing a stale list,
    which is the failure this replaced.

    The cache is keyed on the task files' mtime, so editing a weight or a note in
    gray/tasks/ shows up on the next page load. Without that, this file would be
    a cached copy of the code - which is the same drift in a new hiding place.
    """
    global _REWARDS_CACHE, _REWARDS_STAMP  # noqa: PLW0603
    stamp = _tasks_mtime()
    if _REWARDS_CACHE is not None and stamp == _REWARDS_STAMP:
        return _REWARDS_CACHE

    out: dict = {"tasks": [], "error": "",
                 "source": "gray/tasks/*.py, read live - never hand-copied"}
    try:
        import importlib  # noqa: PLC0415

        # import_module by full dotted path, NOT `from gray.tasks import
        # stand_env_cfg`. gray/tasks/__init__.py re-exports the stand_env_cfg
        # FUNCTION under that same name, so the `from` form hands back the
        # function and every attribute lookup below fails.
        stand, push, walk = (
            importlib.import_module(f"gray.tasks.{name}_env_cfg")
            for name in ("stand", "push", "walk")
        )
        # Already-imported modules are stale after an edit. Reload in dependency
        # order - walk builds on push, which builds on stand.
        if _REWARDS_CACHE is not None:
            for module in (stand, push, walk):
                importlib.reload(module)
    except Exception as exc:  # noqa: BLE001
        out["error"] = (f"Could not import the task configs ({type(exc).__name__}: "
                        f"{exc}). The reward terms are read from them, so nothing is "
                        f"shown rather than something out of date.")
        return out

    notes = {**stand.REWARD_NOTES, **push.PUSH_NOTES, **walk.WALK_NOTES}
    builders = {"stand": stand.stand_env_cfg, "walk": walk.walk_env_cfg}

    for spec in _REWARD_TASKS:
        entry = {**spec, "terms": [], "error": "", "ceiling": 0.0}
        try:
            cfg = builders[spec["key"]]()
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
            out["tasks"].append(entry)
            continue

        for name, term in (cfg.rewards or {}).items():
            weight = float(getattr(term, "weight", 0.0) or 0.0)
            entry["terms"].append({
                "name": name,
                "label": name.replace("_", " "),
                "sign": "+" if weight >= 0 else "-",
                "weight": abs(weight),
                "note": notes.get(name, ""),
                # A term nobody has explained is worth seeing. Silently dropping it
                # is how the old list ended up describing four terms that no longer
                # existed while missing four that did.
                "described": name in notes,
            })
        # Rewards first, then penalties; biggest weight first inside each.
        entry["terms"].sort(key=lambda t: (t["sign"] != "+", -t["weight"]))
        entry["ceiling"] = round(
            sum(t["weight"] for t in entry["terms"] if t["sign"] == "+"), 2)
        entry["undescribed"] = sum(1 for t in entry["terms"] if not t["described"])
        out["tasks"].append(entry)

    _REWARDS_CACHE = out
    _REWARDS_STAMP = stamp
    return out


# ---------------------------------------------------------------------------
# The front page's four-line version
# ---------------------------------------------------------------------------

PHASES = [
    {
        "n": 1, "name": "Prepare", "state": "provisional",
        "one_line": "A digital Gray worth trusting.",
        "detail": "Rebuilt from SolidWorks. 11 of 11 checks pass: 2030 g, joint axes "
                  "exact, and the owner's measured travel limits on all twelve joints, "
                  "verified in the simulator. Still provisional - the mass is a "
                  "parts-list estimate and three numbers have no datasheet.",
        "needs_robot": False,
    },
    {
        "n": 2, "name": "Train", "state": "next",
        "one_line": "Three training runs, not 200 skills.",
        "detail": "R1 locomotion is training - one policy covering standing, walking, "
                  "turning and strafing. R2 getting up and R3 foot-on-object are not "
                  "started. Jumping and seeing are parked with triggers.",
        "needs_robot": False,
    },
    {
        "n": 3, "name": "Deploy", "state": "start now",
        "one_line": "Reassemble, fit the potentiometers, calibrate, run it on the Pi.",
        "detail": "The parallel track, not the last stage. It is the long pole, and "
                  "step 3.3 measures the three numbers stage 1 is guessing.",
        "needs_robot": True,
    },
]

NEXT_UP = {
    "title": "R1  Locomotion",
    "why": "The largest group in the library, and one policy file. Standing, walking, "
           "turning and strafing are the same reward with a different setpoint, so "
           "they train together. It is the largest single piece of the project and "
           "the only one on the critical path to a robot walking on a floor.",
    "before": [
        {"task": "Hold the R1 bar", "who": "code",
         "note": "5 m without falling, commanded speed within 0.05 m/s, under 100 mm "
                 "of sideways drift over 20 s."},
        {"task": "Widen the command vector", "who": "code",
         "note": "Add height, pitch and roll alongside vx, vy and yaw. This is what "
                 "folds the old balance subsection in. The observation grows by three, "
                 "so the contract changes and the current policy is superseded - a few "
                 "hours of retraining."},
        {"task": "Turn the D1 dials up", "who": "code",
         "note": "Friction, mass, centre of mass, servo gains and joint friction are "
                 "already randomised. Terrain, payload and damaged servos come next."},
        {"task": "Sit down and list the parts", "who": "owner",
         "note": "Slicer grams per printed part, plus the exact battery, Pi, boards, "
                 "pots and fasteners. Closes the largest guess in stage 1 and needs no "
                 "hardware."},
        {"task": "Start the hardware in parallel", "who": "owner",
         "note": "Pots, ADC and IMUs. Nothing in training is blocked on it, and "
                 "everything in stage 3 is."},
    ],
    "open": [
        {"q": "Three of stage 1's numbers have no datasheet.",
         "note": "Servo gains, backlash and true loop latency. All three are "
                 "randomised rather than guessed, which is the right handling - but "
                 "the range is itself a guess until stage 3.3 measures them."},
        {"q": "The mass is a parts-list estimate.",
         "note": "2030 g, built from 12 servos at 60 g, a 3S LiPo, the Pi and boards, "
                 "and PLA costed at 22% infill. The parts session replaces the plastic "
                 "rows with slicer output; weighing one calf, one thigh and one hip "
                 "during reassembly checks the split."},
        {"q": "Three-legged stance is filed under R1 but is not commandable.",
         "note": "Standing on three legs, on two diagonal legs, and holding one leg "
                 "out need a per-foot command that the vector does not have. Listed on "
                 "the R1 page rather than buried."},
    ],
}

FEASIBILITY = {
    "verdict": "Yes, and it is measured, not assumed. The graphics card is saturated "
               "and the model is the limiting factor.",
    "facts": [
        {"k": "Graphics card", "v": "NVIDIA RTX 4070 Ti", "note": "12.3 GB, measured on this machine"},
        {"k": "Simulator", "v": "MuJoCo 3.10 via mjlab", "note": "runs the physics on the GPU"},
        {"k": "Robots trained at once", "v": "6,400", "note": "11.5 GB - probed, not guessed"},
        {"k": "Throughput", "v": "~90,000 steps/s", "note": "GPU 94% busy, CPU 5.9%"},
        {"k": "Control rate", "v": "50 Hz", "note": "hard limit, set by the servo PWM period"},
        {"k": "Time for one run", "v": "1 - 6 hours", "note": "R1 locomotion is 3000 iterations"},
    ],
    "risks": [
        {"risk": "Every number the robot is trained against is an estimate.",
         "detail": "Mass is a parts list. Servo stiffness is a guess. Pot calibration "
                   "does not exist yet. Train all 200 skills before touching hardware "
                   "and you find out at skill 200 what you could have found out at "
                   "skill 1. Deploy R1 early, while only one policy rides on it. This "
                   "is why stage 3 starts now rather than last.",
         "level": "high"},
        {"risk": "Backlash and slop are not in the simulator.",
         "detail": "Ball joints, servo gear trains and printed parts all have play. The "
                   "simulator has none, so the controller never learns to expect it. "
                   "This is the usual reason a policy that works in simulation does not "
                   "work outside it.",
         "level": "medium"},
        {"risk": "A reward that pays for the wrong thing.",
         "detail": "mjlab's default velocity tolerance would have paid this robot 78% "
                   "of full marks for standing perfectly still, because it is tuned for "
                   "robots that walk five times faster. Caught by arithmetic before the "
                   "run, not after it. Every new reward gets the same treatment.",
         "level": "medium"},
        {"risk": "The observation contract is one shared list, or it is nothing.",
         "detail": "The policy reads about 45 numbers every 20 ms, and the simulator "
                   "and the robot must build that list identically - same order, same "
                   "units, same scaling. One flipped sign and the robot thrashes with "
                   "nothing in the logs to say why. Write it once and share it; two "
                   "implementations that agree today will not agree in a month.",
         "level": "high"},
        {"risk": "Ten skills need cameras that do not exist yet.",
         "detail": "Reacting to something is blind; anticipating it needs eyes. Gray can "
                   "cross a 20 mm block by feel but cannot avoid a hole it has not "
                   "stepped in. Vision also changes the observation list, so it is a "
                   "retrain from scratch rather than an addition. Parked on purpose.",
         "level": "medium"},
    ],
}


def stage2_state() -> dict:
    """Stage 2 with the live skill library folded in."""
    lib = skills.load()
    for s in lib["subsections"]:
        s["bar"] = STAGE2["bars"].get(s["key"], "")
    return {**STAGE2, **lib}
