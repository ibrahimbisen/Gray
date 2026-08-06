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

Stage 2 is the one with depth, and PLAN.md is its plan: five steps, agreed 3 Aug
2026. FORWARD below is that plan as data, and it is the only plan on the page -
the six A-F phases and the nine subsections it used to show are gone.

Stage 2 is also broken up in dashboard/skills.py, which sorts the owner's skill
library live from the CSV - by what each row actually IS, not by what it looks
like. Two training runs, one dial, one constraint, two test groups, three parked.

The reward terms on the page are NOT written here. They are read off the task
configs in gray/tasks/, so the page cannot drift from the code the way the old
hand-written list did.
"""

from __future__ import annotations

import re
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
        "headline": "11 of 11 checks pass. 3100 g, 13 links, 12 joints, travel limits "
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
        "summary": "PLAN.md's five steps: locomotion, getting up, joining them, then "
                   "the sensors twice - once to keep what you had, once to use them. "
                   "Two training runs, one randomisation dial, one constraint, two "
                   "groups of tests, three parked.",
        "why": "This is where the project lives for a long time, and it is far smaller "
               "than 200 rows makes it look. The unit of work is a training run, not a "
               "skill: the largest group of rows is ONE run with different commands, "
               "the next largest is the range that run is sampled over, and the rest "
               "are checked or measured rather than trained. Live counts are on the "
               "stage 2 page - they are read from the CSV, never written down here.",
        "headline": "At PLAN.md 1.2.7/1.3.1 - it walks straight and turns at three "
                    "quarters of the rate it is told; the open faults are foot lift "
                    "and a nose-down collapse. R2 getting up not started. R3 "
                    "foot-on-object, jumping and seeing are shelved, each with the "
                    "thing that un-shelves it.",
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
        {"k": "Mass", "v": "3100 g", "n": "2030 g robot + 1070 g payload, rounded up"},
        {"k": "Links / joints", "v": "13 / 12", "n": "trunk, four hips, four thighs, four calves"},
        {"k": "Ride height", "v": "190 mm", "n": "70% of the 272 mm it can reach"},
        {"k": "Footprint", "v": "280 x 186 mm", "n": "feet 34 mm outboard of each hip"},
        {"k": "Torque margin", "v": "2.73x", "n": "worst joint needs 0.72 of 1.96 N-m"},
        {"k": "Leg mass", "v": "36%", "n": "was 55% - the payload is all trunk, so legs got lighter"},
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
            {"part": "Trunk", "g": 1984, "n": "714 structural + 4 servos + 1070 payload"},
            {"part": "Hip (x4)", "g": 165, "n": "two servos each - thigh drive and calf drive"},
            {"part": "Thigh (x4)", "g": 65, "n": "no servo; a printed arm and a pushrod"},
            {"part": "Calf (x4)", "g": 49, "n": "no servo"},
        ],
        "total": 3100,
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
            "THE PAYLOAD, added 3 Aug 2026. A Jetson Orin Nano dev kit (~250 g), "
            "twelve 18650 cells in 3S4P (540 g, 89 Wh, about 1.5 h of walking), five "
            "cameras (~50 g) and 200 g besides. The 200 g 3S LiPo it replaces comes "
            "out, so the net is +840 g - rounded up to a 3100 g total on the owner's "
            "call, because shaving mass off a model later is far cheaper than finding "
            "the real robot heavier than everything was trained against.",
            "It is all on the trunk as one lump, because none of it is in the CAD yet. "
            "That is honest now and wrong later: a battery underneath and a Jetson on "
            "top are not the same as 1070 g at the trunk's centre, and the difference "
            "shows up in roll inertia. Redistribute when the mounts are modelled.",
            "The good news is where it landed. Every gram is trunk, so leg mass fell "
            "from 55% of the robot to 36% - and leg mass is what a swing costs and "
            "what a shove has to arrest. Torque margin went 4.18x to 2.73x, still "
            "comfortable. The cost is to dynamics and servo speed under load, not to "
            "standing up: stand and push both passed their bars on the heavier robot "
            "the same night the mass changed.",
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
    "lede": "200 skills, sorted by what each row actually IS - two training runs, "
            "one dial, one constraint, two groups of tests, three parked.",
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
        "Applied to 200 rows it gives two runs - locomotion, and getting up. They are "
        "PLAN.md steps 1 and 2. Everything else is a dial turned during a run, a "
        "property checked across all of them, or a measurement taken afterwards - and "
        "each of those is cheap in a way a run is not.",
    ],
    "bars": {
        "walk": "Walks 5 m without falling, forwards AND backwards, holds commanded "
                "speed within 0.05 m/s, stays within 4 deg of the line it was sent "
                "along - and holds any commanded height and attitude at zero speed "
                "for 30 s, trunk within 5 mm of target, uprightness above 0.99.",
        "recover": "Stands up from 9 of 10 random ground poses in under 3 s, and "
                   "hands control back to R1 upright and stable.",
        "tool": "Shelved. It would be: moves the target object by the commanded "
                "amount without falling. Nothing depends on it, so nothing makes it "
                "due.",
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
        "cannot be tested before there are two things to hand over. PLAN.md step 3. "
        "The preparation for it happens inside R1 and R2, at 1.1.6 and 2.1.2, not "
        "here.",
        "R3 is shelved, not queued last. It needs objects in the scene, which nothing "
        "else does, and nothing depends on it - so there is no event that makes it "
        "due.",
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
        "reads 49 numbers every 20 ms; if one joint is in a different order, or "
        "a sign is flipped, the robot thrashes and nothing says why. Two "
        "implementations that agree today will not agree in a month.",
        "The observation normaliser ships with the network. The policy learned a "
        "running mean and standard deviation for those 49 numbers - export the network "
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
# What the training monitor shows for the skill library: one row per group in
# skills.SUBSECTIONS, with the bar each one has to pass.
#
# This is NOT the plan. The plan is PLAN.md, and FORWARD below is the only copy
# of it in this file. These nine rows are a sort of the owner's CSV - what each
# library row IS - and they were being read as a nine-step plan, which is one of
# the five competing structures this file used to ship.
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
    "the second one, not the first. That is why 200 rows collapse into two runs.",
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
        "detail": "Rebuilt from SolidWorks. 11 of 11 checks pass: 3100 g with the "
                  "payload, joint axes exact, and the owner's measured travel limits "
                  "on all twelve joints, verified in the simulator. Still provisional "
                  "- the mass is a parts-list estimate and three numbers have no "
                  "datasheet.",
        "needs_robot": False,
    },
    {
        "n": 2, "name": "Train", "state": "next",
        "one_line": "Five steps, two training runs, not 200 skills.",
        "detail": "PLAN.md: locomotion, getting up, join them up, then the sensors "
                  "twice. At 1.2.2 today - 1.1 passed on three unseen seeds, and the "
                  "first measurement of the rest of the command box says it turns at "
                  "half the rate it is told. R2 getting up is not started. R3 "
                  "foot-on-object, jumping and seeing are shelved, each with the thing "
                  "that un-shelves it.",
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
    "title": "PLAN.md 1.2.2  Make it turn at the rate it is told",
    "why": "1.1 passed on 4 Aug on three seeds it had never seen, so the robot "
           "walks straight, forward and backward. The same day verify learned to "
           "test the other half of the command box - crabbing sideways and turning "
           "on the spot - which the policy had been TRAINED on since 3 Aug and which "
           "nothing had ever graded. Those three passing seeds turn at about half the "
           "rate they are told to: 0.44 to 0.51 rad/s of error against a 1.0 rad/s "
           "command, against a new 0.20 bar. That is what this step closes.",
    "before": [
        {"task": "1.2.1  Grade sideways and turning - DONE 4 Aug", "who": "code",
         "note": "Verify runs four passes now, not two, and carries five new bars. It "
                 "cost nothing to find because it needs no training: the three "
                 "finished 1.1 policies were re-scored in minutes, with --no-record so "
                 "the measurement could not overwrite the verdicts that closed 1.1."},
        {"task": "1.2.2  Batch 1, the turn band and weight", "who": "code",
         "note": "4 runs, 5000 robots, 550 iterations. turn_std 0.80 or 0.40 crossed "
                 "with track_turn 1.0 or 3.0. At std 0.80 the 0.5 rad/s error the "
                 "robot actually makes still pays 0.68 of 1.0, so turning correctly is "
                 "worth 0.32 against terms that all charge more for turning harder - "
                 "under-turning is simply cheaper. This is the same trap as TRACK_STD, "
                 "in the other direction, for the second time."},
        {"task": "1.2.3  Score the path on a sideways command", "who": "code",
         "note": "Crab comes in 20 to 34 deg off the line. `wandering`, `veering` and "
                 "the `off_line` input are all gated on `_straight_now()`, so on a "
                 "crab command the policy is asked to hold a line it is neither shown "
                 "nor charged for missing - exactly what forward walking was doing "
                 "before it could sense its heading. Needs a code change and a wider "
                 "observation, so it is its own batch."},
        {"task": "Close step 1.2's gate", "who": "code",
         "note": "All eleven criteria pass, on unseen seeds. NINE of eleven today - "
                 "the six straight-line ones, crab distance, crab speed and turn "
                 "wander. The two that fail are crab drift, at 19.9 to 34.1 deg "
                 "against 4.0, and turn rate, at 0.44 to 0.51 rad/s against 0.20. Two "
                 "faults, two different fixes, one batch each. Only then does 1.3 turn "
                 "the dials up."},
        {"task": "Sit down and list the parts", "who": "owner",
         "note": "Slicer grams per printed part, plus the exact battery, Pi, boards, "
                 "pots and fasteners. Closes the largest guess in stage 1 and needs no "
                 "hardware."},
        {"task": "Start the hardware in parallel", "who": "owner",
         "note": "Pots, ADC and IMUs. Step 4 fits them, and stage 3.3 measures the "
                 "three numbers the model is guessing. Nothing in training is blocked "
                 "on it, and everything in stage 3 is."},
    ],
    "open": [
        {"q": "Three of stage 1's numbers have no datasheet.",
         "note": "Servo gains, backlash and true loop latency. All three are "
                 "randomised rather than guessed, which is the right handling - but "
                 "the range is itself a guess until stage 3.3 measures them."},
        {"q": "The mass is a parts-list estimate.",
         "note": "3100 g: a 2030 g robot built from 12 servos at 60 g, the Pi and "
                 "boards and PLA costed at 22% infill, plus a 1070 g payload of "
                 "Jetson, 18650 pack and cameras that is not in the CAD yet. The parts "
                 "session replaces the plastic rows with slicer output; weighing one "
                 "calf, one thigh and one hip during reassembly checks the split."},
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
        # Settled at 4500 after measuring the throughput curve: 6400 runs 1.87 s
        # per iteration against 3072's 1.52, because the card is already at 100%
        # and more robots buy sub-linear steps. 4500 is the ceiling in
        # dashboard/queue.py, and asking for more is clamped with the reason.
        {"k": "Robots trained at once", "v": "4,500", "note": "measured, and the queue's cap"},
        {"k": "Throughput", "v": "~76,000 steps/s", "note": "GPU saturated, ~1.9 s per iteration"},
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
         "detail": "The policy reads 49 numbers every 20 ms, and the simulator "
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


# ---------------------------------------------------------------------------
# Which verifier tasks feed which subsection.
#
# This CANNOT be derived from anything on disk and has to be written by hand.
# run.json carries a `stage` number, but it is legacy from scripts/train.py -
# Stand is 1, Push is 2, Walk is 5 - and in today's STAGES, built from
# skills.SUBSECTIONS, index 5 is C1 "fit for the real robot". Joining on it
# would file every walking run under a constraint. Join on `task`.
#
# All three current tasks feed R1, because standing, being shoved and walking
# are the same policy at different commands - which is the whole argument for
# folding them into one run in the first place.
# ---------------------------------------------------------------------------

SUBSECTION_TASKS = {
    "walk": ["Gray-Stand", "Gray-Push", "Gray-Walk"],
    "recover": [],   # no verifier written yet
    "tool": [],
    "robust": [],    # a dial, not a run - see skills.py
    "quality": [],   # a constraint, checked continuously, never "passed"
    "chain": [],
    "measure": [],   # its bar is explicitly "not a bar, a number written down"
    "flight": [],    # parked
    "see": [],       # parked
}

# Clauses of a bar that NOTHING measures, listed so the page can show them as
# unknown rather than quietly dropping them. An unlisted clause is worse than a
# failing one: it makes the count look complete.
UNVERIFIED_CLAUSES = {
    "walk": [
        "holds any commanded height and attitude at zero speed for 30 s - the "
        "command vector has no height, pitch or roll in it yet, so nothing can "
        "ask for this",
        "three-legged and two-legged diagonal stances, and holding one leg out - "
        "these need a per-foot command that does not exist",
    ],
}


# ---------------------------------------------------------------------------
# How much ONE training run covers.
#
# The question this answers: "if one reward function covers 40 skills, do we run
# 40 simulations?" No - one, and the reason is arithmetic rather than an opinion,
# so the arithmetic is shown instead of asserted.
#
# Every input is READ, never written here: four out of the source line that sets
# them, one out of the queue's live default. A regex over the file rather than an
# import, because importing the task configs pulls in torch and this page is meant
# to open instantly. A read that misses names itself on the page - a missing
# number is recoverable, a number that used to be true is not.
# ---------------------------------------------------------------------------

# The single idea this whole page rests on, written down once.
BOX = {
    "lede": "The policy works inside the range it was sampled on, and not "
            "outside it.",
    "paras": [
        "The command is three numbers, each with a range. Every attempt draws a "
        "random point inside those ranges. So 'walk forward slow to sprint' and "
        "'walk diagonally' are not separate jobs - they are different points in "
        "the same range, already being drawn from.",
        "The policy has never been given the exact command you give it, and does "
        "not need to be. It has never been asked for 0.23 m/s with 0.07 sideways. "
        "It handles that because it drew thousands of points nearby.",
        "Outside the range it does not refuse. It guesses, and the guess is not "
        "reliable. Both cases were measured on run #25. Asked for a 45 degree "
        "diagonal, past the sampled edge, it walked 26 degrees. Asked to walk "
        "backward, which is outside the range entirely, it moved 0.00 m in eight "
        "seconds.",
        "The number of draws is fixed at 864,000 whether the range is narrow or "
        "wide, so widening it lowers the density everywhere. Backward and "
        "sideways also need more draws than straight ahead, not the same number. "
        "This is why rel_forward_envs is 0.8: four in five draws go to straight "
        "ahead, because that is the case being solved now.",
        "The same applies to everything else that is randomised - ground "
        "friction, mass, centre of mass, servo stiffness, joint friction. That is "
        "what D1 is. Friction is never sampled below 0.3, so the robot has no "
        "response to ice regardless of how well it walks.",
    ],
}

_SAMPLING_READS = (
    {"key": "iterations", "label": "Iterations in the run", "unit": "",
     "file": "gray/tasks/walk_env_cfg.py", "re": r"max_iterations\s*=\s*(\d+)",
     "note": "R1's own target, set in its config"},
    {"key": "num_steps_per_env", "label": "Steps per iteration", "unit": "",
     "file": "gray/tasks/stand_env_cfg.py", "re": r"num_steps_per_env\s*=\s*(\d+)",
     "note": "control steps each robot takes before the policy is updated"},
    {"key": "control_hz", "label": "Control rate", "unit": "Hz",
     "file": "gray/config/robot.yaml", "re": r"control_hz:\s*(\d+)",
     "note": "the PWM period is 20 ms, so this is a hard ceiling"},
    {"key": "hold_lo", "label": "Command held for, shortest", "unit": "s",
     "file": "gray/tasks/walk_env_cfg.py",
     "re": r"resampling_time_range\s*=\s*\(\s*([\d.]+)\s*,",
     "note": "then the robot is handed a fresh random command"},
    {"key": "hold_hi", "label": "Command held for, longest", "unit": "s",
     "file": "gray/tasks/walk_env_cfg.py",
     "re": r"resampling_time_range\s*=\s*\(\s*[\d.]+\s*,\s*([\d.]+)\s*\)",
     "note": "drawn uniformly between the two"},
)


def _fmt(x: float) -> str:
    """Whole numbers without a trailing .0, and thousands separated."""
    return f"{int(round(x)):,}" if abs(x - round(x)) < 1e-9 else f"{x:,.1f}"


def sampling() -> dict:
    """One run, in numbers: how many different commands it actually tries."""
    from dashboard import queue as job_queue  # noqa: PLC0415

    inputs, missing, val = [], [], {}
    for spec in _SAMPLING_READS:
        path = ROOT / spec["file"]
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        hit = re.search(spec["re"], text)
        if hit is None:
            missing.append(f"{spec['label']} - not found in {spec['file']}")
            continue
        val[spec["key"]] = float(hit.group(1))
        inputs.append({**spec, "value": _fmt(val[spec["key"]]),
                       "source": spec["file"]})

    # The box itself, read off the three module constants that define it. Shown
    # beside the arithmetic because the count of commands tried is meaningless
    # without the region they were drawn from.
    walk_src = (ROOT / "gray/tasks/walk_env_cfg.py")
    src = walk_src.read_text(encoding="utf-8") if walk_src.is_file() else ""
    axes = []
    for const, name, unit, note in (
        ("WALK_SPEED", "forward", "m/s", "never zero and never negative"),
        ("WALK_SIDE", "sideways", "m/s", "only ever a lean on forward motion"),
        ("WALK_TURN", "turn", "rad/s", "symmetric, and the widest axis"),
    ):
        hit = re.search(rf"{const}\s*=\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)", src)
        if hit:
            axes.append({"axis": name, "const": const, "unit": unit, "note": note,
                         "lo": float(hit.group(1)), "hi": float(hit.group(2))})

    envs = float(job_queue.DEFAULTS.get("num_envs") or 0)
    inputs.insert(0, {
        "key": "num_envs", "label": "Robots at once", "unit": "", "value": _fmt(envs),
        "source": "dashboard/queue.py", "file": "dashboard/queue.py",
        "note": "the queue's default, capped at what the card measured"})
    if not envs:
        missing.append("Robots at once - the queue has no default")

    if missing:
        return {"error": "Could not read: " + "; ".join(missing),
                "inputs": inputs, "derived": [], "axes": axes, "headline": {}}

    step_dt = 1.0 / val["control_hz"]
    seconds = val["iterations"] * val["num_steps_per_env"] * step_dt
    hold = (val["hold_lo"] + val["hold_hi"]) / 2.0
    per_robot = seconds / hold
    total = per_robot * envs

    derived = [
        {"label": "Simulated time, per robot", "value": _fmt(seconds), "unit": "s",
         "how": f"{_fmt(val['iterations'])} iterations "
                f"× {_fmt(val['num_steps_per_env'])} steps ÷ {_fmt(val['control_hz'])} Hz",
         "aside": f"{seconds / 60:.0f} minutes of walking"},
        {"label": "Commands tried, per robot", "value": _fmt(per_robot), "unit": "",
         "how": f"{_fmt(seconds)} s ÷ {hold:g} s average hold",
         "aside": "a fresh speed, direction and turn rate each time"},
        {"label": "Commands tried, whole run", "value": _fmt(total), "unit": "",
         "how": f"{_fmt(per_robot)} × {_fmt(envs)} robots",
         "aside": "this is the number that covers the skill list"},
        {"label": "Robot-hours of walking", "value": _fmt(seconds * envs / 3600.0),
         "unit": "h", "how": f"{_fmt(seconds)} s × {_fmt(envs)} robots",
         "aside": "in roughly 1.5 hours of wall clock"},
    ]
    return {"error": "", "inputs": inputs, "derived": derived, "axes": axes,
            "headline": {"sims": 1, "commands": _fmt(total), "envs": _fmt(envs)}}


# ---------------------------------------------------------------------------
# What Gray could be given to sense.
#
# The owner's question: "we are fitting cameras and sensors, so why is anything
# blocked?" The answer is a menu with three separate costs on it, and it lives in
# gray/config/sensors.yaml rather than here - it is a fact about the robot, not
# about the dashboard. Read live, and the row counts each sensor would clear are
# joined from skills.py so the two can never disagree.
# ---------------------------------------------------------------------------

_SENSORS_FILE = ROOT / "gray" / "config" / "sensors.yaml"

_STATUS_ORDER = {"committed": 0, "candidate": 1, "not_recommended": 2}


def sensors() -> dict:
    """The sensor menu, with each entry joined to what it would unblock."""
    try:
        import yaml  # noqa: PLC0415

        raw = yaml.safe_load(_SENSORS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not read gray/config/sensors.yaml "
                         f"({type(exc).__name__}: {exc})",
                "sensors": [], "observation_now": {}, "actuators": {}}

    # How many library rows each blocker actually holds up, counted live rather
    # than written into the yaml where it would go stale the moment a row moves.
    held = {n["key"]: n for n in skills.load()["needs"]}

    out = []
    for s in raw.get("sensors", []):
        un = s.get("unblocks") or {}
        need = held.get(un.get("need"))
        out.append({
            **s,
            "unblocks_need": un.get("need", ""),
            "unblocks_extent": un.get("extent", ""),
            "unblocks_label": (need or {}).get("label", ""),
            # "partly" never claims the whole pile. An unqualified count beside
            # a partial fix is the kind of number that gets believed.
            "unblocks_rows": (need or {}).get("count", 0) if need else 0,
        })
    # .get, not [key]: sensors.yaml is hand-written, and a sensor added without a
    # status used to raise here and 500 both /api/stage/2 and /api/state.
    out.sort(key=lambda s: (_STATUS_ORDER.get(s.get("status"), 9), -s["unblocks_rows"]))

    # What the input list becomes if everything already decided goes in. This is
    # the number that decides WHEN they go in: they all land in one batch,
    # because each one on its own would cost the same full retrain.
    now = raw.get("observation_now", {}).get("total", 0)
    adding = [s for s in out
              if s["status"] == "committed" and isinstance(s.get("adds"), int)
              and s["adds"] > 0]
    added = sum(s["adds"] for s in adding)

    # Mounting requirements, joined to the sensor they belong to so the page can
    # show them together. Ordered by how much the position matters, because that
    # is the only thing separating "get this right in CAD" from "put it anywhere".
    rank = {"critical": 0, "moderate": 1, "low": 2, "none": 3}
    by_key = {x["key"]: x for x in out}
    fitting = sorted(
        ({**m, "name": by_key.get(m.get("key"), {}).get("name", m.get("key", "")),
          "count": by_key.get(m.get("key"), {}).get("count", 1),
          "status": by_key.get(m.get("key"), {}).get("status", "")}
         for m in raw.get("mounting", [])),
        key=lambda m: rank.get(m.get("matters"), 9))

    return {"error": "", "sensors": out,
            "fitting": fitting,
            "wiring": raw.get("wiring", {}),
            "observation_now": raw.get("observation_now", {}),
            "batch": {
                "sensors": len(adding),
                "added": added,
                "after": now + added,
                "growth": round((now + added) / now, 2) if now else 0,
                "rows": [{"name": s["name"], "adds": s["adds"]} for s in adding],
            },
            "actuators": raw.get("actuators", {}),
            "source": "gray/config/sensors.yaml"}


# ---------------------------------------------------------------------------
# What a policy actually did when it was told to.
#
# Every other number about R1 on this page is a claim about what the reward and
# the command CAN cover. This one is a measurement of what one trained file DID,
# and the two disagreed - which is the reason the section exists. Read live out
# of the newest drive.json rather than typed here, because a hand-copied
# measurement is a measurement that will be wrong by next week.
# ---------------------------------------------------------------------------

_LOG_ROOT = ROOT / "logs" / "rsl_rl"


def driven() -> dict:
    """The most recent scripts/drive.py result, if there is one."""
    files = sorted(_LOG_ROOT.glob("*/*/drive/drive.json"),
                   key=lambda p: p.stat().st_mtime) if _LOG_ROOT.is_dir() else []
    if not files:
        return {"exists": False, "cases": []}
    import json  # noqa: PLC0415

    newest = files[-1]
    try:
        data = json.loads(newest.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"exists": False, "cases": [],
                "error": f"{type(exc).__name__}: {exc}"}

    # The run's number, so the page can say "#25" rather than a timestamp. Same
    # rule as dashboard/runs.py - oldest is 1.
    runs_dir = ROOT / "progress" / "runs"
    names = sorted(d.name for d in runs_dir.iterdir() if d.is_dir()) \
        if runs_dir.is_dir() else []
    number = names.index(data["run"]) + 1 if data.get("run") in names else None

    cases = []
    for c in data.get("cases", []):
        # drive.json is written by a separate script and read here. A case that
        # is missing a half is skipped, not raised on: this function feeds
        # /api/stage/2 and /api/state, so one short case used to take out the
        # Train page and the Summary page together.
        told = c.get("commanded")
        got = c.get("walked")
        if not isinstance(told, dict) or not isinstance(got, dict):
            continue
        moved = abs(got.get("along_m") or 0) + abs(got.get("across_m") or 0)
        cases.append({
            **c,
            "number": number,
            # A robot that did not move has no heading, and printing the angle it
            # "walked" would be reporting the direction of its own noise.
            "moved": moved,
            "went_nowhere": moved < 0.10,
            "asked_deg": told.get("angle_deg"),
            "got_deg": got.get("angle_deg"),
            "sd_deg": got.get("angle_sd_deg"),
            "lo_deg": got.get("angle_min_deg"),
            "hi_deg": got.get("angle_max_deg"),
        })
    return {"exists": True, "run": data.get("run"), "number": number,
            "checkpoint": data.get("checkpoint"), "seconds": data.get("seconds"),
            "robots": data.get("robots"), "cases": cases,
            "source": str(newest.relative_to(ROOT)).replace("\\", "/")}


# ---------------------------------------------------------------------------
# Shelved. Not forgotten, not blocked - decided against, each with the one thing
# that un-shelves it. Straight out of PLAN.md's shelved table.
#
# "The real robot" is on this list, and it is the expensive one. Saying so
# plainly is the point of the `cost` note: deferring it is a choice the owner
# made, and a choice with a price is not the same thing as a dependency.
# ---------------------------------------------------------------------------

SHELVED = {
    "lede": "Decided against for now. Each row names the thing that un-shelves "
            "it, so 'later' means something.",
    "rows": [
        {"what": "R3 - using a foot on an object",
         "trigger": "None. Nothing depends on it, and it needs objects in the "
                    "scene nothing else does."},
        {"what": "Jumping",
         "trigger": "Stage 3.3 measures the real servo SPEED under load. Torque "
                    "is not the blocker - higher-torque hobby servos are usually "
                    "slower."},
        {"what": "Seeing",
         "trigger": "A camera the POLICY reads, plus a heightmap pipeline and the "
                    "worst sim-to-real gap on the list. Not the camera used to "
                    "drive the robot."},
        {"what": "The real robot",
         "trigger": "Deferred by choice, not blocked by anything."},
    ],
    "cost": "On the last one. Stage 3.3 measures three numbers the model is "
            "guessing - servo gains, backlash, loop latency. Every policy trained "
            "before it is trained against those guesses. One policy depends on "
            "them today. After steps 1 to 5, two do, hardened and tuned against a "
            "robot that may not be the real one. That is the cost of deferring "
            "it. It is a real cost and it is the owner's call.",
}


# ---------------------------------------------------------------------------
# The whole thing, forward. PLAN.md, agreed 3 Aug 2026, as data.
#
# One plan, not five. This used to ship six A-F phases, three stages, four
# rounds, nine subsections and a thirteen-section paper layout at the same
# time, and none of them was the plan that was agreed. They are gone.
#
# The shape is GATES rather than dates: a step is over when a number is met,
# not when a week is up, which is the only honest way to schedule work whose
# duration nobody knows. Two substeps measure rather than train, and those two
# carry no gate at all rather than an invented one.
#
# `here` is where the project is. It is 1.2.2 and it is stated once, at the top
# and on the step, so the page cannot show two different answers.
# ---------------------------------------------------------------------------

FORWARD = {
    "one_line": "Five steps, each ending on a number rather than a date.",
    "here": "1.2.7",
    "lede": "One pattern, applied twice: train it, harden it, measure it. "
            "Locomotion first, then getting up. Then join them. Then the "
            "sensors, twice - once to keep what you had, once to use them. "
            "Where we are: the 1.2.7 gate is open, and 1.3.1 batches 1 and 2 "
            "are answered - the dials are innocent, and the robot turns at "
            "three quarters of the rate it is told. The open faults are the "
            "gait's: feet that barely lift, and a nose-down collapse after a "
            "short walk.",
    "shape": [
        "Two things get tuned, and they fight each other. The REWARD says what "
        "good means. The RANGE says where it has to be good. Widening the range "
        "makes the same reward harder to satisfy, so they cannot be finished one "
        "after the other.",
        "That is why the range stays pinned at its narrowest until 1.1 passes. "
        "You cannot tell whether a wider range broke something if the narrow one "
        "was already broken - and until 3 Aug it was: `wandering` was measuring "
        "drift against a different line from the one the bar scores.",
        "Every step ends on a number, not a date, because nobody knows how long "
        "a reward takes to get right. Two substeps below - 1.4 and 2.3 - have no "
        "gate at all. They measure a trained policy and train nothing, so there "
        "is nothing for them to pass.",
    ],
    "phases": [
        {"id": "1", "name": "Locomotion", "state": "now", "here": "1.2",
         "gate": "Three gates, one per substep. 1.1 was six criteria on a "
                 "straight line. 1.2 is ELEVEN - the same six, plus three for "
                 "crabbing sideways and two for turning on the spot, added "
                 "4 Aug 2026 once the test could measure them at all. 1.3 is "
                 "those eleven again with the dials at full. 1.4 has no gate - "
                 "it measures.",
         "what": "One policy for everything the robot does on its feet. Train it "
                 "until it walks, widen what it can be told to do, turn the "
                 "world's dials up, then measure it.",
         "steps": [
             "1.1  Make it walk. PASSED 4 Aug on three unused seeds - DONE",
             "1.1.1  Fix `wandering` to measure the line it was sent along "
             "- done, 3 Aug",
             "1.1.2  Round 0, the noise floor. 3 seeds, one config - done 4 Aug. "
             "Reward noise 0.7 points. Drift noise is 0.72 deg sd, so two "
             "configs under about 2 deg apart read the same on one run each",
             "1.1.3  Round 1, straightness. WAS an 8-run factorial over the "
             "straightness weights; became 6 runs plus 6 once the cause was "
             "measured rather than guessed - done 4 Aug",
             "1.1.4  Round 2, the speed terms. 4 runs",
             "1.1.5  Round 3, the winner, long, on unseen seeds. 3 runs",
             "1.1.6  Add handover-shaped resets to R1's mix. Preparation for "
             "step 3",
             "1.2  The whole command box. Gate: eleven criteria, across the "
             "full range",
             "1.2.0  Widen the box and fix `_going_straight` to gate on "
             "abs(vx) - done, 3 Aug. WALK_SPEED went to (-0.35, 0.35), "
             "sideways to +/-0.20, turn to +/-1.0",
             "1.2.1  Grade sideways and turning at all - done, 4 Aug. Verify "
             "ran two passes, forward and backward, with sideways and turn "
             "pinned to zero, so half of what the policy is commanded to do "
             "was measured by nothing. Four passes now, and five new bars",
             "1.2.2  Make it turn at the rate it is told. Measured 0.44-0.51 "
             "rad/s of error against a 1.0 rad/s command, on all three of the "
             "seeds that PASSED 1.1 - about half rate. WHERE WE ARE",
             "1.2.3  Score the path on a sideways command. `wandering`, "
             "`veering` and the `off_line` input are all gated on "
             "`_straight_now()`, so a crab command is asked to hold a line it "
             "is neither shown nor charged for missing - and it comes in 20 to "
             "34 deg off",
             "1.2.4  Retune, and close the box on unseen seeds",
             "1.3  Turn the dials up. Gate: the same eleven criteria as 1.2, "
             "dials at full",
             "1.3.1  Widen what is already on - friction, mass, centre of mass, "
             "servo gains, joint friction",
             "1.3.2  Terrain. Slopes first, then uneven ground",
             "1.3.3  Payload, to +2 kg, off-centre",
             "1.3.4  Degraded hardware - weak servo, dead servo, low battery",
             "1.4  Measure. NO GATE: numbers written down, nothing trained",
             "1.4.1  Top speed, and the speed ladder",
             "1.4.2  Acceleration, sustained run",
             "1.4.3  Smoothness, read off the trained policy",
         ],
         "cost": "18 runs in 1.1 alone. 1.2 needs more than 1.1, because the "
                 "range is wider and the number of draws is fixed at 864,000.",
         "blocked_by": "",
         "note": "The box was widened on 3 Aug and 1.1 closed on it on 4 Aug, "
                 "so the policy has been TRAINED on backward, sideways and "
                 "turning since then. What had never happened was measuring "
                 "any of it: verify pinned sideways and turn to zero and ran "
                 "forward and backward only. It measures all four since 4 Aug, "
                 "and the first reading says the straight-line half is done and "
                 "the other half is not - turning at half the commanded rate, "
                 "crabbing 20 to 34 deg off the line. 1.3 turns the dials UP, "
                 "it does not switch them on - foot friction is already "
                 "randomised 0.4 to 1.2 on every attempt, alongside +/-20% mass, "
                 "+/-15 mm centre of mass and +/-30% servo stiffness. None of it "
                 "is a new skill: the policy never sees the word gravel, it sees "
                 "the same 49 numbers in a different pattern, and that is also "
                 "where sim-to-real robustness comes from. 1.4 trains nothing - "
                 "'find top speed' has no reward term, you raise the command "
                 "until it fails and write down where.",
         "subs": [
             {"id": "1.1", "name": "Make it walk",
              "goal": "Get the config right, so a policy can move at all.",
              # Only a CLOSED step has anything to carry, so this is set per-sub
              # rather than being a page in the nav. A global link would sit
              # there being wrong for the eight steps that have not finished.
              "carry": "/carry",
              "gate": "PASSED 4 Aug 2026. All six criteria, three unused seeds "
                      "(99, 314, 2718), one config, no flags: 100% stayed up, "
                      "trunk 8.9-14.0 mm of 40, uprightness 0.998+, 6.55-6.72 m "
                      "of 5, speed error 0.032-0.034 of 0.05, and 3.24-3.82 deg "
                      "off the line against a 4 deg bar.",
              "why": "Round 0 looks like the least and matters the most. Three "
                     "identical configs on three seeds; whatever they disagree "
                     "by is the noise floor, and every later difference has to "
                     "beat it before it means anything.",
              "items": [
                  {"id": "1.1.1", "state": "done",
                   "what": "Fix `wandering` to measure the line it was sent along",
                   "goal": "Score drift against the line it was given, not world Y.",
                   "gets": "A drift penalty that stops fighting `veering`."},
                  {"id": "1.1.2", "state": "done",
                   "what": "Round 0, the noise floor. 3 seeds, one config",
                   "goal": "Find how much two identical runs disagree.",
                   "gets": "0.72 deg drift, 0.7 reward points. Nothing smaller "
                           "than that counts as a result."},
                  {"id": "1.1.3", "state": "done",
                   "what": "Round 1, straightness - done 4 Aug",
                   "goal": "Find why it will not walk straight, and fix it.",
                   "gets": "A 49th input - heading error. Drift 19.5 -> 3.6 deg. "
                           "Blind it again and drift is 23-27, so the trunk "
                           "gyros are now required hardware."},
                  {"id": "1.1.4", "state": "done",
                   "what": "Round 2, smoothness - done 4 Aug",
                   "goal": "Stop the twitching and the trunk shake.",
                   "gets": "Final weights for twitching, jitter, rocking, "
                           "shaking, and the tilt tolerance."},
                  {"id": "1.1.5", "state": "done",
                   "what": "Round 3, the winner, on unseen seeds - done 4 Aug",
                   "goal": "Prove the config, not one lucky run.",
                   "gets": "One config passing all six bars on 3 seeds. This is "
                           "what closes 1.1."},
                  {"id": "1.1.6", "state": "done",
                   "what": "Add handover-shaped resets to R1's mix - done 4 Aug",
                   "goal": "Let R1 start from the pose R2 will hand it.",
                   "gets": "A reset mix step 3 can hand over into. No runtime "
                           "rule can add this later."},
              ]},
             {"id": "1.2", "name": "The whole command box",
              "goal": "Make it obey every command, not just straight ahead.",
              "gate": "Eleven criteria, across the full range: the six that "
                      "closed 1.1, three for crabbing sideways, two for turning "
                      "on the spot.",
              "why": "The box was widened on 3 Aug 2026 and 1.1 closed on it on "
                     "4 Aug, so the policy has trained on backward, sideways "
                     "and turning for two days. None of it was ever MEASURED: "
                     "verify pinned sideways and turn to zero and ran forward "
                     "and backward only. It runs four passes since 4 Aug, and "
                     "the first reading on the three seeds that PASSED 1.1 says "
                     "the straight-line half is genuinely done and the other "
                     "half is not - 0.44 to 0.51 rad/s of turn error against a "
                     "1.0 rad/s command, and 20 to 34 deg off the line while "
                     "crabbing.",
              "items": [
                  {"id": "1.2.0", "state": "done",
                   "what": "Widen the box, and gate straightness on abs(vx) "
                           "- done 3 Aug",
                   "goal": "Put backward, sideways and turning inside the "
                           "range at all.",
                   "gets": "WALK_SPEED (-0.35, 0.35), sideways +/-0.20, turn "
                           "+/-1.0, with the straightness terms still on for a "
                           "backward command."},
                  {"id": "1.2.1", "state": "done",
                   "what": "Grade sideways and turning at all - done 4 Aug",
                   "goal": "Stop half the command box being scored by nothing.",
                   "gets": "Four verify passes instead of two, and five new "
                           "bars: crab distance 3.75 m, crab speed 0.05 m/s, "
                           "crab drift 4.0 deg, turn rate 0.20 rad/s, turn "
                           "wander 0.10 m."},
                  {"id": "1.2.2", "state": "done",
                   "what": "Make it turn at the rate it is told - done 4 Aug",
                   "goal": "Close a 0.5 rad/s gap on a 1.0 rad/s command.",
                   "gets": "track_turn raised from 1.0 to 5.0, the best-paid "
                           "term in the task. 0.476 -> 0.141 rad/s of error "
                           "against a 0.20 bar, confirmed at 0.135 on a second "
                           "seed. 8 runs. The BAND was not the problem: 0.40 "
                           "was worse at both weights and 1.20 was worse than "
                           "0.80, so 0.80 is a measured optimum in both "
                           "directions rather than an untested default."},
                  {"id": "1.2.3", "state": "done",
                   "what": "Score the path on a sideways command - done 4 Aug",
                   "goal": "Stop straightness scoring switching off for crab "
                           "steps and turns.",
                   "gets": "`_on_a_line` replaced `_straight_now`: it asks for "
                           "travel in ANY direction, not a forward component, "
                           "so `veering`, `wandering` and the `off_line` input "
                           "apply to a crab command. `wandering` measures "
                           "perpendicular to the pinned COURSE, not the facing "
                           "- 90 degrees apart on a crab step, so the old "
                           "version charged a perfect crab for the whole "
                           "distance it was told to cover. Crab drift 19.5 -> "
                           "3.7 deg on the stock weights alone. Then wandering "
                           "-1.0 -> -3.0, the best of four corners on crab "
                           "drift, forward drift and turn at once; heavier "
                           "`veering` made all three worse."},
                  {"id": "1.2.4", "state": "done",
                   "what": "Three attempts at crab drift, all reverted "
                           "- 5 Aug",
                   "goal": "Close a 4.3 to 5.0 deg crab drift against a 4.0 "
                           "bar, with the other ten criteria passing.",
                   "gets": "A measured dead end, and the config that was "
                           "already best. Tried, one change per row, all on "
                           "seed 1301: a pure-sideways share of the draws "
                           "(turn 0.140 -> 0.277, crab 4.33 -> 4.37); taking "
                           "that share out of the straight draws instead (crab "
                           "-> 5.06); and `off_track`, a 50th input reporting "
                           "cross-track distance (crab -> 5.56, turn -> 0.249, "
                           "and unbounded it nearly stopped the robot - 6.47 m "
                           "of ground covered down to 2.65). Every step away "
                           "from the original config lost ground on BOTH "
                           "numbers. Both are kept in the code, off by default, "
                           "with what they measured written beside them."},
                  {"id": "1.2.5", "state": "done",
                   "what": "Find out whether crab drift is measurable at all "
                           "- done 5 Aug",
                   "goal": "Stop comparing configs on a number that moves more "
                           "than the differences between them.",
                   "gets": "THE FINDING THAT UNDOES A NIGHT OF COMPARISONS. The "
                           "reverted config was re-run on the same three gate "
                           "seeds. Turn reproduced - 0.140, 0.134, 0.123 came "
                           "back 0.125, 0.152, 0.120. Crab drift did not: 4.33, "
                           "4.79, 4.95 came back 4.86, 2.69, 9.06. Same config, "
                           "same seed, up to 4.1 deg apart against a 4.0 bar, "
                           "because training is not bit-reproducible on this "
                           "card. It is NOT the test - the 64 per-robot "
                           "readings give standard errors of 0.37, 0.27 and "
                           "1.04. So every crab comparison made on a single run "
                           "means nothing, the sideways share and the "
                           "cross-track input included; only the TURN result "
                           "survives, and it survives comfortably. THE FIX WAS "
                           "RUN LENGTH, the one dial never turned in the whole "
                           "of 1.2: at 1500 iterations the same three seeds "
                           "give 4.60, 4.82 and 5.42 - a spread of 0.82 against "
                           "6.37 - and turn tightens to 0.099. The MEAN barely "
                           "moves, 5.11 to 4.95. Length did not make the robot "
                           "better at crabbing; it made the number trustworthy. "
                           "Note what that says about the 500-iteration cap: it "
                           "was set from the REWARD curve going flat, and a "
                           "flat reward does not mean every criterion has "
                           "settled."},
                  {"id": "1.2.6", "state": "done",
                   "what": "Decide the crab bar - done 5 Aug, bar moved to 5.0",
                   "goal": "Close 1.2, or say plainly what it cannot pass.",
                   "gets": "The reverted config fails exactly one criterion, "
                           "reliably: crab drift 4.60 to 5.42 against 4.0, with "
                           "the other ten passing and turn at 0.099 of 0.20. "
                           "One batch left to try - the sideways share at 1500 "
                           "iterations, where it can finally be judged; it was "
                           "dropped on a gap of 0.04 measured inside a spread "
                           "of 6.37. If that does not close it, the bar is what "
                           "is left. At 0.20 m/s a 4 deg error is 0.014 m/s of "
                           "unwanted fore-aft speed while the speed bar beside "
                           "it allows 0.05, so the sideways direction is held "
                           "3.5x tighter than the direction the robot is asked "
                           "to travel in - and lateral steps come from one hip "
                           "servo a leg against two for fore-aft. Re-cutting it "
                           "in the speed bar's units is the honest option. It "
                           "is the owner's call, not the agent's."},
                  {"id": "1.2.7", "state": "now",
                   "what": "Close the box - NOT closed, reopened 5 Aug",
                   "goal": "Make the wider range pass all eleven, as a "
                           "property and not one result.",
                   "gets": "NOT YET A CLOSED GATE, and it was marked closed for "
                           "about an hour on two seeds of three. The third had "
                           "not finished. It came in at crab drift 7.68 deg "
                           "against 5.0, turn 0.279 against 0.20 and forward "
                           "drift 5.45 against 4.0 - three criteria failing, "
                           "not one, where the other two seeds scored 3.74 and "
                           "4.25 with everything passing. So the config is 3.74, "
                           "4.25, 7.68: a spread of 3.94 deg, and no bar drawn "
                           "above the mean covers it. Seed 8821 is the same seed "
                           "that gave 9.06 at 550 iterations and 5.42 without "
                           "the sideways share, so it is consistently the worst "
                           "of the three rather than a one-off. THE OPEN "
                           "QUESTION is whether that seed is an unlucky draw or "
                           "whether this config fails about one time in three. "
                           "Three fresh seeds are running against the wider "
                           "1.3.1 dials and will answer it either way."},
              ]},
             {"id": "1.3", "name": "Turn the dials up",
              "goal": "Make it survive every number we guessed being wrong.",
              "gate": "The same eleven criteria as 1.2, dials at full.",
              # Written in Simplified Technical English, per CLAUDE.md. Short
              # sentences, active voice, one word for one thing.
              "why": "The dials are already on. Every attempt draws foot grip "
                     "from 0.4 to 1.2, mass to +/-20%, the centre of mass to "
                     "+/-15 mm, and servo strength to +/-30%. This step makes "
                     "those ranges wider. It does not add a new skill. The "
                     "policy never reads the word gravel. It reads the same 49 "
                     "numbers in a different pattern. This step is also where "
                     "sim-to-real strength comes from. A policy that walks "
                     "across the whole range also walks at the real value, and "
                     "nobody can measure that value yet.",
              "items": [
                  {"id": "1.3.1", "state": "now",
                   "what": "Make the five dials wider",
                   "goal": "Cover the real robot, not our estimate of it.",
                   "gets": "Foot grip, mass, where the weight sits, servo "
                           "strength and gearbox drag, each over a range wide "
                           "enough to hold whatever the built machine is."},
                  {"id": "1.3.2", "state": "later",
                   "what": "Terrain. Slopes first, then rough ground",
                   "goal": "Get the robot off a flat floor.",
                   "gets": "A slope angle it can walk, and a roughness it can "
                           "walk. Low grip and rough ground together are what "
                           "gravel is; the policy is never told the word."},
                  {"id": "1.3.3", "state": "later",
                   "what": "Payload, up to 2 kg, off centre",
                   "goal": "Find what the robot can carry.",
                   "gets": "A weight limit, and how far off centre that weight "
                           "may sit."},
                  {"id": "1.3.4", "state": "later",
                   "what": "Damaged hardware. A weak servo, a dead servo, a "
                           "flat battery",
                   "goal": "Know what the robot does when a part fails.",
                   "gets": "The behaviour on a failed joint, written down "
                           "before it happens on the bench. This substep has no "
                           "gate. The robot is not asked to survive a dead "
                           "servo; we only record what it does."},
              ],
              # The batch plan. `stop` marks a row where a person must read a
              # number and choose - they are listed as rows because a plan that
              # shows only the work reads as if it runs itself.
              #
              # Two lengths, on purpose. Use 550 iterations to find where
              # something BREAKS: a robot that falls over says so in 17 minutes.
              # Use 1500 only to confirm it PASSES, because crab drift does not
              # settle below that - see 1.2.5. That split halves the cost of the
              # whole step.
              "batches": [
                  {"n": "1", "stage": "1.3.1", "runs": "3 x 1500",
                   "code": "make the five dials wider",
                   "asks": "does the 1.2 config still pass in a wider world",
                   "done": "All 3 seeds failed, on the same 4 criteria - "
                           "forward drift, sideways drift, turn rate, turn "
                           "wander. Read at the time as the wider world "
                           "costing the robot its heading. THAT READ WAS "
                           "WRONG; batch 2 disproved it."},
                  {"stop": "did it hold, or which dial broke it"},
                  {"n": "2", "stage": "1.3.1", "runs": "6 x 550",
                   "code": "--narrow-dials, to set a dial back per run",
                   "asks": "which of the five costs the heading. One seed, "
                           "one dial wide per run, plus an all-narrow control",
                   "done": "NONE OF THEM. The all-narrow control failed too, "
                           "at 0.292 rad/s of turn error - the same as the "
                           "five wide runs, which spread 0.264 to 0.336. The "
                           "dials are innocent. The turn numbers from before "
                           "1.2 closed were measured by a verify that pinned "
                           "the turn command to ZERO, so they never tested "
                           "turning at all: q3_long_s8821 is on record at "
                           "0.083 and measures 0.280 under today's verify, "
                           "from the same checkpoint. The first read of that "
                           "was 'turning has never passed'. WRONG WAY UP. The "
                           "robot turns - at about three quarters of any rate "
                           "it is asked, measured at three points (told 0.5 "
                           "it does 0.38, told 0.7 it does 0.50-0.55, told "
                           "1.0 about 0.72), both ways, evenly, and the owner "
                           "confirmed it on the stick on 6 Aug. What was "
                           "never tested is THE BAR's one case: a spin on "
                           "the spot at 1.00 rad/s, a command the training "
                           "draw produces about once in 80 attempts. An "
                           "unpractised case, not a lost skill."},
                  {"stop": "the turn bar tests 1.00 rad/s, the very edge of "
                           "the trained box - widen the box, or move the bar. "
                           "DECIDED 6 Aug: neither. Keep the bar and train "
                           "the case - a pure-spin share in the command draw, "
                           "the same fix the crab share was. That is the g1 "
                           "probe of the gait batch below."},
                  {"n": "3", "stage": "1.3.2", "runs": "4 x 550",
                   "code": "ground that tilts",
                   "asks": "at what slope does it stop walking"},
                  {"stop": "which angle to hold for the confirming runs"},
                  {"n": "4", "stage": "1.3.2", "runs": "3 x 1500",
                   "code": "slope held at that angle",
                   "asks": "all eleven criteria, on a slope"},
                  {"n": "5", "stage": "1.3.2", "runs": "4 x 550, 3 x 1500",
                   "code": "the height under each foot varies",
                   "asks": "how rough before it fails, then confirm"},
                  {"stop": "which roughness to keep"},
                  {"n": "6", "stage": "1.3.3", "runs": "4 x 550",
                   "code": "mass hung off the trunk, off centre",
                   "asks": "the weight limit, and how far off centre"},
                  {"stop": "the payload limit to write down"},
                  {"n": "7", "stage": "1.3.4", "runs": "3 x 550",
                   "code": "a joint weakened, then killed, mid-attempt",
                   "asks": "what it does with a failed servo. No gate"},
                  {"n": "8", "stage": "gate", "runs": "3 x 1500",
                   "code": "nothing - the task as it stands",
                   "asks": "all eleven criteria, every dial at full"},
                  {"stop": "1.3 closes, or one dial needs another pass"},
                  {"n": "9", "stage": "sweep", "runs": "64 x 550",
                   "code": "nothing",
                   "asks": "cut 6 reward numbers down to a shortlist"},
                  {"stop": "which settings go through - a script can pick"},
                  {"n": "10", "stage": "sweep", "runs": "18 x 1500",
                   "code": "nothing",
                   "asks": "6 survivors on 3 seeds each - the real ranking"},
                  {"stop": "which two or three get the long build"},
                  {"n": "11", "stage": "final", "runs": "3 x long",
                   "code": "stop_at switched off",
                   "asks": "the policy that goes on the robot"},
              ],
              "batches_note":
                  "About 116 runs and 53 hours of card time. Batches 1 to 8 are "
                  "about 15 hours and end with a robot that PASSES. Batches 9 "
                  "to 11 are about 38 hours and end with the BEST one; they are "
                  "optional, and they only pay if you want margin rather than a "
                  "pass. Batch 9 cannot pick a winner - at 550 iterations crab "
                  "drift is not measurable, so it only throws away the clearly "
                  "bad half. Batches 3, 5 and 6 are ladders: raise one number "
                  "until it fails, write down where, and no decision is needed "
                  "inside them. Batch 2 was the one that might never happen. "
                  "Batch 1 failed 3 seeds out of 3, so it happens.",
              },
             {"id": "1.4", "name": "Measure",
              "goal": "Write down what it can actually do. Nothing is trained.",
              "gate": "", "no_gate": True,
              "why": "No gate. Numbers written down, nothing trained. 'Find top "
                     "speed' has no reward term - you raise the command until it "
                     "fails and write down where. The trained half of smoothness "
                     "- twitching, rocking, shaking, joint_shock, landing_speed "
                     "- is already ramping in during 1.1.",
              "items": [
                  {"id": "1.4.1", "state": "later",
                   "what": "Top speed, and the speed ladder",
                   "goal": "Find where it stops keeping up.",
                   "gets": "A top speed, and how well it holds each one below."},
                  {"id": "1.4.2", "state": "later",
                   "what": "Acceleration, sustained run",
                   "goal": "Find how fast it can change speed, and for how long.",
                   "gets": "An acceleration figure and a battery-realistic "
                           "endurance."},
                  {"id": "1.4.3", "state": "later",
                   "what": "Smoothness, read off the trained policy",
                   "goal": "Measure the gait rather than the reward.",
                   "gets": "The operating envelope - the sheet the real robot is "
                           "driven by."},
              ]},
         ]},

        {"id": "2", "name": "Getting up", "state": "later",
         "gate": "Two gates: up from 9 of 10 random ground poses in under 3 s "
                 "(2.1), and still 9 of 10 with the dials at full (2.2). 2.3 has "
                 "no gate - it measures.",
         "what": "The second policy file, and the one group that genuinely needs "
                 "its own. There is no commanded velocity to track when the "
                 "robot is on its back, so track_speed, stepping, dragging and "
                 "wandering are all meaningless.",
         "steps": [
             "2.1  Make it stand up. Gate: up from 9 of 10 random ground poses, "
             "in under 3 s",
             "2.1.1  Write the recover task - start poses, reward, terminations",
             "2.1.2  Reward it for ENDING WHERE R1 STARTS: stable, at ride "
             "height, low joint velocity. Preparation for step 3",
             "2.1.3  Train it",
             "2.2  Turn the dials up. Gate: still 9 of 10, dials at full",
             "2.2.1  The same dials as 1.3, applied to R2",
             "2.3  Measure. NO GATE",
             "2.3.1  Time to stand, per start pose",
         ],
         "cost": "A new task written from scratch, so slower than a retune.",
         "blocked_by": "Step 1, so the handover has something to hand to. It can "
                       "otherwise overlap 1.2 and 1.3.",
         "note": "2.1.2 is not really part of getting up. It is step 3's "
                 "preparation, and it is paid for here because that is the only "
                 "place it can be paid for - what R2 finishes in is what R1 has "
                 "to be able to start from.",
         "subs": [
             {"id": "2.1", "name": "Make it stand up",
              "goal": "Get it off the floor. Needs its own policy - no walking "
                      "reward means anything on its back.",
              "gate": "Up from 9 of 10 random ground poses, in under 3 s.",
              "why": "The one group that genuinely needs its own policy file. "
                     "There is no commanded velocity to track when the robot is "
                     "on its back, so track_speed, stepping, dragging and "
                     "wandering are all meaningless.",
              "items": [
                  {"id": "2.1.1", "state": "later",
                   "what": "Write the recover task - start poses, reward, "
                           "terminations",
                   "goal": "Define what getting up means.",
                   "gets": "The recover task file."},
                  {"id": "2.1.2", "state": "later",
                   "what": "Reward it for ending where R1 starts",
                   "goal": "Finish upright, at ride height, joints slow.",
                   "gets": "A handover state step 3 can use."},
                  {"id": "2.1.3", "state": "later", "what": "Train it",
                   "goal": "Get it standing from 9 of 10 poses.",
                   "gets": "The second policy file."},
              ]},
             {"id": "2.2", "name": "Turn the dials up",
              "goal": "Make getting up survive the same wrong guesses as 1.3.",
              "gate": "Still 9 of 10, dials at full.",
              "why": "",
              "items": [
                  {"id": "2.2.1", "state": "later",
                   "what": "The same dials as 1.3, applied to R2",
                   "goal": "Get up on any robot, on any ground.",
                   "gets": "Getting up, still working at full randomisation."},
              ]},
             {"id": "2.3", "name": "Measure",
              "goal": "Write down how long it takes to stand. Nothing trained.",
              "gate": "", "no_gate": True,
              "why": "No gate. A number written down.",
              "items": [
                  {"id": "2.3.1", "state": "later",
                   "what": "Time to stand, per start pose",
                   "goal": "Know the worst case.",
                   "gets": "Seconds to stand, per pose."},
              ]},
         ]},

        {"id": "3", "name": "Join them up", "state": "later",
         "gate": "100 handovers with no stall and no fall-loop.",
         "what": "R2 stands the robot up and hands it to R1. Only the switch "
                 "rule and the test are here - the preparation happened inside "
                 "steps 1 and 2, at 1.1.6 and 2.1.2.",
         "steps": [
             "3.1  The switch rule - hysteresis and a settle time, so control "
             "cannot chatter at the boundary",
             "3.2  The test - 100 drops: R2 stands it up, hands over, R1 walks "
             "5 m",
         ],
         "cost": "Small if 1.1.6 and 2.1.2 were done. Two retrains if they were "
                 "not.",
         "blocked_by": "Steps 1 and 2. There is nothing to hand over until both "
                       "policies exist.",
         "note": "The failure this prevents. R2 finishes and says 'upright, take "
                 "it.' R1 receives a robot mid-wobble, with joint velocities and "
                 "a trunk height it has never seen, because R1's resets always "
                 "start it clean: nudge_base is x, y +/-0.01 m, yaw +/-0.1 rad, "
                 "velocity_range {}, and the joint reset is velocity +/-0.05 "
                 "rad/s. That is a robot standing still, and R1 has trained on "
                 "nothing else. Hand it an out-of-distribution state and it "
                 "falls; R2 picks it up, hands over, it falls again. No runtime "
                 "rule fixes that loop. Which is why 1.1.6 and 2.1.2 exist: the "
                 "preparation happens during training, in the two steps above. "
                 "Skip it and this step is where you find out.",
         "subs": [
             {"id": "3.1", "name": "The switch rule",
              "goal": "Decide when control passes between the two policies.",
              "gate": "", "why": "",
              "items": [
                  {"id": "3.1", "state": "later",
                   "what": "Hysteresis and a settle time",
                   "goal": "Stop control chattering at the boundary.",
                   "gets": "The handover rule and its thresholds."},
              ]},
             {"id": "3.2", "name": "The test",
              "goal": "Prove the two work as one robot, not two that each work.",
              "gate": "", "why": "",
              "items": [
                  {"id": "3.2", "state": "later",
                   "what": "100 drops: stand up, hand over, walk 5 m",
                   "goal": "Catch the fall-loop before the hardware does.",
                   "gets": "A pass rate for the whole sequence, end to end."},
              ]},
         ]},

        {"id": "4", "name": "Sensors - keep what you had", "state": "later",
         "gate": "Every bar that passed before still passes, with 99 inputs.",
         "what": "Ten sensors decided; seven add inputs. 49 numbers become 99. "
                 "They go in as one batch because each one alone costs the same "
                 "full retrain as all of them together.",
         "steps": [
             "4.1  Fit them. CAD positions needed for the three where the "
             "mounting is in the maths",
             "4.2  Model each one in the simulator, including how it fails",
             "4.3  Warm-start - grow the network, copy the old weights across, "
             "initialise the new input columns to ZERO, so day one behaves "
             "identically to the best policy",
             "4.4  Retrain R1 and R2 from that warm start",
         ],
         "cost": "One retrain of both policies, from a warm start rather than "
                 "from zero.",
         "blocked_by": "The mechanical work. Worth doing after steps 1 to 3, so "
                       "the retrain is spent once on something already working.",
         "note": "The three where mounting position is a coefficient rather than "
                 "documentation: the five optical flow units (each reads "
                 "v + w x r), the middle IMU (offset from the centre of mass), "
                 "and the six range finders (origin and aim vector, which the "
                 "simulator turns into raycasts). THIS GATE IS A REGRESSION "
                 "CHECK. It proves nothing broke. It does not say the sensors "
                 "were worth fitting - that is step 5, and that is why step 5 "
                 "exists.",
         "subs": [
             {"id": "4", "name": "Keep what you had",
              "goal": "Fit the sensors and prove nothing broke. No new ability "
                      "on purpose.",
              "gate": "Every bar that passed before still passes, with 99 "
                      "inputs. A regression check - it proves nothing broke, not "
                      "that the sensors were worth fitting.",
              "why": "Ten sensors decided; seven add inputs. 49 to 99 numbers, "
                     "which is why they go in as one batch: each one alone costs "
                     "the same full retrain as all of them together.",
              "items": [
                  {"id": "4.1", "state": "later",
                   "what": "Fit them",
                   "goal": "Get the sensors on the robot.",
                   "gets": "CAD positions for the three where the mounting is "
                           "in the maths."},
                  {"id": "4.2", "state": "later",
                   "what": "Model each one, including how it fails",
                   "goal": "Simulate a broken sensor, not just a working one.",
                   "gets": "Seven sensor models the policy can be trained "
                           "against."},
                  {"id": "4.3", "state": "later",
                   "what": "Warm-start from the current weights",
                   "goal": "Do not throw away a working policy to add inputs.",
                   "gets": "A 99-input network that behaves identically on day "
                           "one."},
                  {"id": "4.4", "state": "later",
                   "what": "Retrain R1 and R2 from that warm start",
                   "goal": "Prove the wider policy still passes.",
                   "gets": "Every old bar, met again."},
              ]},
         ]},

        {"id": "5", "name": "Sensors - use them", "state": "later",
         "gate": "Numbers that were out of reach before now pass.",
         "what": "Step 4 proved nothing broke. This is the step that says the "
                 "sensors were worth fitting: tighter bars, and three reward "
                 "terms that could not be written before.",
         "steps": [
             "5.1  Tighten the bars. They were set for a blind policy",
             "5.2  Write reward terms that were impossible before",
             "5.2  slip - five flow units against the IMU's yaw rate. Nothing "
             "else on the robot can see a slip",
             "5.2  landing - load cells turn 'did it slam' from a guess into a "
             "number",
             "5.2  anticipation - range finders make stepping over something "
             "before touching it reachable at all",
             "5.3  Retune against the tighter bars",
         ],
         "cost": "A retune rather than a rewrite - but against bars that have "
                 "moved, so runs get longer before they pass.",
         "blocked_by": "Step 4. Nothing can score a sensor the policy does not "
                       "read.",
         "note": "The policy currently cannot measure its own speed. "
                 "base_lin_vel sits in the critic group, which is thrown away "
                 "after training, precisely because the real robot has no way to "
                 "measure it - so the policy infers its speed while track_speed, "
                 "its largest reward term, scores exactly that. Optical flow "
                 "measures it directly. 5.2 also unblocks library rows: some of "
                 "the rows filed under 'needs a camera' are reachable with range "
                 "finders instead, which is the cheap half of seeing.",
         "subs": [
             {"id": "5", "name": "Use them",
              "goal": "Spend the sensors. Every bar so far was set for a blind "
                      "robot.",
              "gate": "Numbers that were out of reach before now pass.",
              "why": "The policy currently cannot measure its own speed - "
                     "base_lin_vel sits in the critic group, thrown away after "
                     "training, because the real robot has no way to measure it. "
                     "Optical flow measures it directly.",
              "items": [
                  {"id": "5.1", "state": "later",
                   "what": "Tighten the bars",
                   "goal": "Ask for what a sighted robot can do.",
                   "gets": "New pass numbers, set against the sensors."},
                  {"id": "5.2a", "state": "later",
                   "what": "slip - optical flow against the IMU's yaw rate",
                   "goal": "Notice a foot sliding.",
                   "gets": "A slip term. Nothing on the robot can see one "
                           "today."},
                  {"id": "5.2b", "state": "later",
                   "what": "landing - load cells",
                   "goal": "Measure the slam instead of inferring it.",
                   "gets": "Landing force as a number."},
                  {"id": "5.2c", "state": "later",
                   "what": "anticipation - range finders",
                   "goal": "See an obstacle before touching it.",
                   "gets": "Stepping over things, reachable for the first "
                           "time."},
                  {"id": "5.3", "state": "later",
                   "what": "Retune against the tighter bars",
                   "goal": "Make the new numbers pass.",
                   "gets": "The policy that goes on the robot."},
              ]},
         ]},
    ],
    "loop": "It does not close in simulation. Stage 3.3 measures the real "
            "servos - stiffness, backlash, loop latency - and those three "
            "numbers change the model every policy was trained against, so some "
            "of steps 1 to 5 gets done again against a robot the simulator "
            "finally describes. The real robot is shelved by choice, so that "
            "loop has not started, and what deferring it costs is on the shelved "
            "list.",
    # The page renders this; SHELVED is the source. One list, two shapes, so a
    # row cannot be un-shelved in one place and still shown in the other.
    "parked": [{"what": r["what"], "why": r["trigger"]} for r in SHELVED["rows"]],
    "shelved_cost": SHELVED["cost"],
}


# ---------------------------------------------------------------------------
# The rounds inside PLAN.md step 1.1. NOT a second plan.
#
# These four rounds ARE substeps 1.1.2 to 1.1.5, and each one carries the
# substep number it is. The page used to show them as a separate "week" beside
# a separate A-F phase list, which is how the repo ended up with two plans that
# disagreed. There is one plan - FORWARD - and this is the detail of one leaf
# of it.
#
# The honest version of "leave it running for a week" is not one long run: a
# run always starts from scratch, so repeating a config gives the same answer
# twice. It is a TEST MATRIX. Every run changes one declared thing, gets scored
# against the same bar, and lands in a table you can read.
#
# What makes it work unattended is that the runner already claims jobs one at a
# time and verify.py already scores each one. What it cannot do is design the
# next round. So the machine runs continuously and the thinking happens between
# rounds - which is what a DOE is, and the reason rounds get smaller as they go.
# ---------------------------------------------------------------------------

_WEEK_ROUNDS = [
        {
            "id": "R0", "step": "1.1.2",
            "name": "The fix, and the noise floor", "runs": 3,
            "hours": 4.5,
            "asks": "Did fixing `wandering` move drift at all - and by more than "
                    "two identical runs differ from each other?",
            "design": "One config, three seeds. Nothing varies but the random "
                      "number stream.",
            "reads": "If the three seeds land within a few hundred mm of each "
                     "other and all well under 2,125 mm, the fix worked and the "
                     "spread between them is the noise floor. If they scatter "
                     "wildly, the noise floor is too high to sweep anything and "
                     "the next job is reducing variance, not tuning weights.",
        },
        {
            "id": "R1", "step": "1.1.3",
            "name": "The two straightness terms", "runs": 8,
            "hours": 12.0,
            "asks": "Now that both terms measure the same line, how hard should "
                    "each one push?",
            "design": "Full 2x2x2 on wandering, the veering ramp, and skidding. "
                      "A factorial rather than one-at-a-time, because these three "
                      "interact - wandering and veering were fighting each other "
                      "until today, and one-at-a-time cannot see that.",
            "reads": "Which corner has the lowest drift, and whether any pair "
                     "matters more together than separately.",
        },
        {
            "id": "R2", "step": "1.1.4",
            "name": "The speed bar", "runs": 4,
            "hours": 6.0,
            "asks": "Speed error is 0.071 m/s against a 0.05 bar. Does paying "
                    "more for speed close it, or does it just trade drift for "
                    "speed?",
            "design": "2x2 on track_speed and ground_covered. Independent of "
                      "round 1, so it is queued alongside rather than after.",
            "reads": "Whether the two bars are in tension. If every run that "
                     "fixes speed worsens drift, the reward needs a different "
                     "shape rather than different weights.",
        },
        {
            "id": "R3", "step": "1.1.5",
            "name": "The winner, long and repeated", "runs": 3,
            "hours": 9.0, "queued": False,
            "asks": "Does the best config hold up at full length, and on seeds "
                    "it has not seen?",
            "design": "Best settings from rounds 0 to 2, 6,000 iterations, three "
                      "seeds. Designed after the earlier rounds are read.",
            "reads": "If all three clear the bar, 1.1 is done and 1.2 starts. If "
                     "one does, it was luck.",
        },
]


def _week_glance(rounds: list[dict]) -> list[dict]:
    """The week's headline figures, counted off the rounds table above.

    Every figure here used to be typed by hand - "~87 min", "15 runs queued
    now, about 23 hours", "a week holds ~115 runs" - and none of them was
    produced by any code. Two of them also contradicted the "~1.9 s per
    iteration" on the feasibility page. A number nobody computes is a number
    that goes stale silently, so the ones that could not be counted were
    deleted rather than re-guessed.
    """
    total = sum(r["runs"] for r in rounds)
    queued = sum(r["runs"] for r in rounds if r.get("queued", True))
    return [
        {"k": "Where we are", "v": "1.1.2",
         "n": "round 0, the noise floor"},
        {"k": "Rounds", "v": str(len(rounds)),
         "n": "PLAN.md 1.1.2 to 1.1.5, one per substep"},
        {"k": "Runs in them", "v": str(total),
         "n": f"{queued} designed now; the last round is designed from the "
              f"ones above it"},
        {"k": "Needs a person", "v": "between rounds",
         "n": "to read the table and design the next one"},
    ]


WEEK = {
    "lede": "PLAN.md step 1.1, make it walk. Four rounds - 1.1.2 to 1.1.5 - run "
            "back to back as a designed experiment rather than one long run.",
    "step": "1.1",
    "why": [
        "A run always starts from scratch. There is no warm start yet, so "
        "queueing the same config twice gives the same answer twice and a week "
        "of it gives nothing a single run would not have.",
        "So step 1.1 is a test matrix. Every run changes one declared thing, "
        "every run is scored by verify.py against the same bar, and the result "
        "is a table rather than an impression.",
        "The one thing that has to be measured first is the NOISE FLOOR, which "
        "is 1.1.2 and where the project is. Round 0 runs the same config on "
        "three seeds. Whatever those three disagree by is the number every later "
        "difference has to beat before it means anything. Without it a sweep is "
        "just reading noise.",
    ],
    "at_a_glance": _week_glance(_WEEK_ROUNDS),
    "rounds": _WEEK_ROUNDS,
    "rules": [
        "One training process at a time. RULES.md rule 4, and the queue makes it "
        "structural rather than something to remember.",
        "Every run is verified when it finishes. A run nobody scored is a run "
        "that did not happen.",
        "Nothing in gray/tasks/ changes while a round is in flight. A weight "
        "edited mid-round makes every run before it unreadable.",
        "Films stay on. They cost about 4% of throughput at one clip per 100 "
        "iterations, and they are the only way to see a robot doing something "
        "the numbers do not describe.",
    ],
    "not_doing": [
        {"what": "Stage 3.3, measuring the real servos",
         "why": "Shelved on the owner's call, not blocked by anything. It stays "
                "the thing that settles the three guessed numbers, and it is "
                "still the trigger that un-shelves jumping. What deferring it "
                "costs is written out on the shelved list."},
        {"what": "The CAD rebuild and the sensor mounts",
         "why": "PLAN.md step 4, and nothing in step 1 waits on it. The sensors "
                "are decided and recorded, and they go into the observation as "
                "one batch of 49 to 99 numbers when the mechanical work "
                "happens."},
        {"what": "Widening the command range to backward and sideways",
         "why": "PLAN.md 1.2, and it comes after 1.1 for a reason. Drift is "
                "failing on the narrowest range there is; widening it spreads a "
                "fixed number of draws over a harder problem and makes the "
                "failing number worse. 1.2.1 also has to fix the "
                "_going_straight gate to abs(vx) first, or backward walking "
                "trains with no straightness penalty at all."},
    ],
}


def stage2_state() -> dict:
    """Stage 2 with the live skill library folded in."""
    lib = skills.load()
    for s in lib["subsections"]:
        s["bar"] = STAGE2["bars"].get(s["key"], "")
        s["unverified"] = UNVERIFIED_CLAUSES.get(s["key"], [])
    return {**STAGE2, **lib, "sampling": sampling(), "sensors": sensors(),
            "driven": driven(), "box": BOX}


# ---------------------------------------------------------------------------
# The dials
# ---------------------------------------------------------------------------
#
# Everything that gets varied during a run, in one place. There are two kinds and
# the difference is the whole point of the page that shows them:
#
#   the robot READS it   - the command. Three numbers, handed in every step.
#   the WORLD has it     - domain randomisation. Rolled fresh each attempt, and
#                          the robot is never told the value. It finds out.
#
# Same rule governs both: inside the range that was swept, it copes; outside it,
# it guesses. Never sweep foot grip below 0.4 and ice is a stranger, exactly as
# never sweeping forward speed below 0.15 made backward a stranger.
#
# Every number below is READ OUT of the task files. Nothing here is typed, so a
# range that changes in the code changes on the page in the same edit.

# Values a real floor or a real fault actually takes, for marking against the
# swept band. These are references, not measurements off this robot - they are
# here to answer "is the band wide enough?", which a band with no scale beside
# it cannot.
_REFERENCES = {
    "ground_grip": [
        {"at": 0.05, "what": "wet ice"},
        {"at": 0.15, "what": "wet tile"},
        {"at": 0.35, "what": "smooth vinyl"},
        {"at": 0.7, "what": "carpet"},
        {"at": 1.0, "what": "dry concrete"},
    ],
    "servo_strength": [
        {"at": 0.5, "what": "a servo at half strength"},
        {"at": 0.2, "what": "a flat battery"},
    ],
}

# Where the legs actually run out, solved against the owner's stance on
# 3 Aug 2026: find_stance puts every foot back on its own print with the trunk
# moved, and reports where the joints hit their stops. (low, high, why it stops)
_REACH = {
    "POSE_HEIGHT": (0.12, 0.27,
                    "measured: it holds anywhere from 120 to 270 mm, at 1.59x "
                    "the servo's stall torque at the lowest"),
    # (-0.17, 0.35), not (-0.35, 0.17). NOSE DOWN IS NEGATIVE - measured in the
    # sim on 5 Aug 2026, and the opposite of what three comments in the task
    # claimed. So the 10 deg of nose-down travel is the NEGATIVE bound and the
    # 20 deg of nose-up is the positive one. Swapped, this page showed the
    # commanded range using half the reach in the direction the robot has most
    # of, and overrunning it in the direction it has least.
    "POSE_PITCH": (-0.17, 0.35,
                   "measured: nose up 20 deg, nose down only 10, because the "
                   "stance rakes the legs forward and spends the travel "
                   "nose-down would need"),
    "POSE_ROLL": (-0.52, 0.52,
                  "measured to +/- 30 deg, where the sweep stopped rather than "
                  "where the robot did - it never found a limit"),
}

_COMMAND_DIALS = (
    ("WALK_SPEED", "forward", "m/s",
     "How fast the body travels nose-first. Negative is backward."),
    ("WALK_SIDE", "sideways", "m/s",
     "How fast it slides left or right, still facing the same way."),
    ("WALK_TURN", "turn", "rad/s",
     "How fast it spins on the spot."),
    # Added 3 Aug 2026. Where the trunk should BE, as opposed to where it should
    # GO - which is what unblocked sit, crouch, crawl, bow, stretch and lean.
    ("POSE_HEIGHT", "height", "m",
     "How far off the ground to hold the trunk."),
    # Both signs MEASURED on 6 Aug 2026 by scripts/measure_pitch_sign.py: a
    # known attitude is written into the sim and proven with foot heights,
    # then read back. This row said the opposite of both until then.
    ("POSE_PITCH", "pitch", "rad",
     "Nose down is negative. Limited by joint travel, and lopsided: the stance "
     "rakes the legs forward, which spends travel that nose-down needs."),
    ("POSE_ROLL", "roll", "rad",
     "Right side down is negative."),
)

# (event name in the task, label, unit, what it is, how to read the range out)
_WORLD_DIALS = (
    ("ground_grip", "foot grip", "", "How much the feet hold. Low is ice.",
     r'"ground_grip".*?"ranges":\s*\(([-\d.]+),\s*([-\d.]+)\)'),
    ("how_heavy", "how heavy", "x", "Every part's mass and inertia, scaled.",
     r'"how_heavy".*?"alpha_range":\s*\(([-\d.]+),\s*([-\d.]+)\)'),  # noqa: E501
    ("where_the_weight_is", "where the weight sits", "m",
     "The trunk's centre of mass, moved.",
     r'"where_the_weight_is".*?"ranges":\s*\(([-\d.]+),\s*([-\d.]+)\)'),
    ("servo_strength", "servo strength", "x",
     "How hard every servo pulls toward the angle it was told.",
     r'"servo_strength".*?"kp_range":\s*\(([-\d.]+),\s*([-\d.]+)\)'),
    ("gearbox_drag", "gearbox drag", "N-m",
     "Stiction in the gear train.",
     r'"gearbox_drag".*?"ranges":\s*\(([-\d.]+),\s*([-\d.]+)\)'),
)


def _read_pair(src: str, pattern: str) -> tuple[float, float] | None:
    hit = re.search(pattern, src, re.S)
    return (float(hit.group(1)), float(hit.group(2))) if hit else None


def dials() -> dict:
    """Every dial in use, read out of the task files."""
    walk = (ROOT / "gray/tasks/walk_env_cfg.py")
    push = (ROOT / "gray/tasks/push_env_cfg.py")
    wsrc = walk.read_text(encoding="utf-8") if walk.is_file() else ""
    psrc = push.read_text(encoding="utf-8") if push.is_file() else ""
    missing = []

    # ---- the three the robot reads ----
    command = []
    for const, name, unit, what in _COMMAND_DIALS:
        pair = _read_pair(wsrc, rf"{const}\s*=\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")
        if pair is None:
            missing.append(f"{const} in gray/tasks/walk_env_cfg.py")
            continue
        lo, hi = pair
        # What to draw the swept band against. Two different questions, and they
        # need two different envelopes:
        #
        #   speed, sideways, turn - nothing has measured a limit, so the envelope
        #     is the dial mirrored about zero and the gap means NEVER SWEPT.
        #     PLAN.md 1.4.1 is where a real top speed gets measured.
        #   height, pitch, roll - the limit IS measured (find_stance solves for
        #     where the legs run out of travel), so the envelope is that, and the
        #     gap means THERE IS NO MORE DIAL. Mirroring height about zero would
        #     draw a robot standing 250 mm underground.
        reach = _REACH.get(const)
        span = max(abs(lo), abs(hi))
        command.append({
            "key": const, "name": name, "unit": unit, "what": what,
            "lo": lo, "hi": hi,
            "full_lo": reach[0] if reach else -span,
            "full_hi": reach[1] if reach else span,
            "measured_limit": bool(reach),
            "limit_note": reach[2] if reach else "",
            "symmetric": abs(lo + hi) < 1e-9,
            "source": "gray/tasks/walk_env_cfg.py",
        })

    # ---- how the three are drawn ----
    mix = {}
    for key, pattern in (
        ("standing", r"rel_standing_envs\s*=\s*([\d.]+)"),
        # `rel_straight_envs`, not mjlab's `rel_forward_envs` - which is pinned
        # to 0 because it forced the speed positive and clamped it up to 0.3.
        # See gray/tasks/walk_command.py.
        ("forward_only", r"rel_straight_envs\s*=\s*([\d.]+)"),
        # The pure-sideways share, added 5 Aug 2026. It belongs on this page for
        # the same reason the other two do: the draw mix is WHY a command works
        # or does not. Crab failed its bar on three seeds while forward passed,
        # and the whole explanation was that a pure crab turned up about once in
        # 350 draws. A share the page does not show is a share nobody checks.
        ("crab", r"rel_crab_envs\s*=\s*([\d.]+)"),
    ):
        hit = re.search(pattern, wsrc)
        mix[key] = float(hit.group(1)) if hit else None
        if not hit:
            missing.append(f"{key} in gray/tasks/walk_env_cfg.py")
    hold = _read_pair(wsrc, r"resampling_time_range\s*=\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")
    mix["hold_lo"], mix["hold_hi"] = hold if hold else (None, None)
    if not hold:
        missing.append("resampling_time_range in gray/tasks/walk_env_cfg.py")

    # ---- the ones the world has ----
    world = []
    for key, name, unit, what, pattern in _WORLD_DIALS:
        # THE WALK TASK FIRST, then push. All five are DEFINED in push_env_cfg,
        # and since PLAN.md 1.3.1 the walk task overrides them with wider ones -
        # widening them at the source would silently re-score Gray-Push, which
        # has its own bar and its own passing runs.
        #
        # Reading only push was this page showing 0.4 to 1.2 for foot grip while
        # every walk run was training on 0.25 to 1.4. A dial page that reports
        # the range the robot is NOT being trained on is worse than no dial page,
        # because the whole point of it is to say what the policy has seen.
        pair = _read_pair(wsrc, pattern) or _read_pair(psrc, pattern)
        if pair is None:
            missing.append(f"{key} in gray/tasks/push_env_cfg.py")
            continue
        lo, hi = pair
        if key == "how_heavy":       # written as a +/- fraction, shown as a scale
            lo, hi = 1 + lo, 1 + hi
        refs = _REFERENCES.get(key, [])
        pad = (hi - lo) * 0.35 or 1.0
        outs = [r["at"] for r in refs]
        full_lo = min([lo - pad, *outs])
        # Grip, strength and drag cannot be negative, so an envelope that runs
        # below zero draws a stretch of dial that does not exist - and puts a
        # zero mark on a scale that has no meaningful zero on it.
        if lo >= 0:
            full_lo = max(full_lo, 0.0)
        world.append({
            "key": key, "name": name, "unit": unit, "what": what,
            # What to print beside the name when there is no unit symbol. A
            # friction coefficient is a number, not a multiplier, and calling it
            # one was wrong on the page.
            "units_note": {"ground_grip": "friction coefficient",
                           "how_heavy": "of the modelled mass",
                           "servo_strength": "of the modelled gain"}.get(key, ""),
            "lo": lo, "hi": hi,
            "full_lo": full_lo, "full_hi": max([hi + pad, *outs]),
            "refs": refs,
            "outside": [r for r in refs if not lo <= r["at"] <= hi],
            "source": "gray/tasks/push_env_cfg.py",
        })

    # ---- the shove, which is an event rather than a dial ----
    #
    # Every key is always present, and a read that failed says so in `missing`.
    # A key that simply vanished from the payload reached the page as
    # `undefined[0]` and killed the whole of /dials - beside a banner that was
    # ready to name the cause but never got drawn. The rule for this whole
    # function: the shape never changes, only the values and `missing`.
    shove: dict = {}
    for key, const in (("speed", "PUSH_MS"), ("spin", "PUSH_SPIN"),
                       ("every", "PUSH_EVERY_S")):
        # No `^` anchor: _read_pair searches with re.S, where `^` only matches
        # the very start of the file. `NAME = (` is unique enough on its own -
        # the other mentions of these constants pass them, they do not assign.
        pair = _read_pair(psrc, rf"{const}\s*=\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")
        if pair:
            shove[key] = list(pair)
        else:
            shove[key] = None
            missing.append(f"{const} in gray/tasks/push_env_cfg.py")
    shove["on_in_walk"] = 'cfg.events.pop("shove"' not in wsrc

    return {
        "command": command,
        "mix": mix,
        "world": world,
        "shove": shove,
        "missing": missing,
    }


# =========================================================== what 1.1 left ===
#
# WHAT THIS IS FOR. A step's runs are thrown away. Its policy is thrown away.
# What survives is a configuration and a short list of things that turned out to
# be true, and until this existed both were recoverable only by reading the code
# and the run history together.
#
# The split is deliberate. Anything with a NUMBER in it that the code also holds
# - weights, ramps, tolerances, bars, inputs - is read live in carry_over() and
# is NOT written here. What is written here is the part no file holds: what was
# measured, what was tried and rejected, and what any of it means. That is the
# half that goes stale, so it is the half that carries a date.
CARRY = {
    "step": "1.1  Make it walk",
    "closed": "4 Aug 2026",
    "verdict": "All six criteria, three seeds never used before (99, 314, 2718), "
               "one config, no command-line overrides.",
    "lede": [
        "The robot walked 19.5 degrees off a straight line in the morning and "
        "3.2-3.8 in the evening, forwards and backwards, on seeds it had never "
        "trained against. It never falls, holds its ride height inside a quarter "
        "of the allowance, and tracks the commanded speed with a third to spare.",
        "The cause was not a reward weight. The policy could feel how fast it was "
        "turning and never which way it was POINTING, so it was being fined for a "
        "quantity it could not sense. No weight fixes that. A 49th input did.",
    ],

    # Measured, not argued. Every row cost at least one run.
    "learned": [
        {"fact": "The trunk gyros are required hardware",
         "number": "3.6 deg sighted, 23-27 blind",
         "why": "Three runs with the heading input removed and nothing else "
                "changed. The robot cannot hold a line without integrated gyro "
                "yaw, so that signal and its calibration are part of the build, "
                "not an accuracy improvement."},
        {"fact": "Run-to-run noise on drift",
         "number": "0.72 deg sd",
         "why": "Five runs, identical config AND identical seed. Two configs "
                "closer than about 2 deg apart cannot be told apart by one run "
                "each. Found by accident, from six runs a stale runner turned "
                "into six replicates."},
        {"fact": "Run-to-run noise on reward",
         "number": "0.7 points",
         "why": "Round 0, three seeds. The floor every later comparison is read "
                "against."},
        {"fact": "500 iterations is a whole run",
         "number": "99.1% of best",
         "why": "Measured across three runs; on two of them 500 beat the number "
                "250 iterations later. Everything past it buys less than the "
                "noise, for a third of the wall clock."},
        {"fact": "Drift is a curve, not a slide",
         "number": "R2 0.98, slope 1.0",
         "why": "Cross-track offset is explained almost entirely by the heading "
                "turning: 19-31 deg of curve against 2-4 deg of crabwise slip. "
                "That is what said a heading sensor would fix it."},
        {"fact": "The randomised body causes the scatter, not the lean",
         "number": "spread 23 deg -> 1 deg",
         "why": "With friction, mass, centre of mass and servo gains held fixed, "
                "the spread across 64 robots collapses and a systematic lean "
                "remains. Two different problems; the heading input fixes both."},
        {"fact": "A term under ~1% of earned reward cannot change a gait",
         "number": "3.86 points earned per episode",
         "why": "twitching at 0.037 would have been traded away entirely for 2% "
                "more speed reward, and was. Same shape as effort, track_turn's "
                "tolerance and hard_landing. Measure a term's share before "
                "arguing about its weight."},
    ],

    # Tried, measured, NOT kept. Worth more than the successes: each one is a run
    # nobody has to spend again.
    "rejected": [
        {"what": "Sharpen track_turn, 0.80 -> 0.15 rad/s",
         "result": "3.49 deg vs 3.64 - inside the noise",
         "verdict": "No effect. The turning error was unobservable, and no price "
                    "fixes an error the policy cannot sense."},
        {"what": "Sharpen upright, 0.45 -> 0.15 rad",
         "result": "tilt 2.1 -> 1.25 deg, drift 3.42 -> 4.37",
         "verdict": "Rejected. It does halve the wobble, and it costs 0.9 deg of "
                    "straightness on both seeds - enough to fail the bar. Level "
                    "and straight spend the same budget. Revisit inside 1.2."},
        {"what": "Smoothness at 15x instead of 5x",
         "result": "roughness -78%, speed error 0.045 of a 0.05 bar",
         "verdict": "Rejected. Smoother, and one seed from failing speed. 5x is "
                    "the setting that costs nothing measurable."},
        {"what": "The 8-run straightness factorial",
         "result": "never run",
         "verdict": "Cancelled. Guess-and-check over the straightness weights, "
                    "designed before the cause was known. Diagnosis turned it "
                    "into 8 hours that would have confirmed nothing."},
    ],

    # Process, not physics. Each cost real time to learn.
    "rules": [
        {"rule": "Restart the runner after editing dashboard/queue.py",
         "cost": "6 runs, 5 hours",
         "detail": "A running Python process holds the module it imported. The "
                   "runner stripped every experimental flag and trained one "
                   "config six times under six different names. train_argv now "
                   "refuses a job whose settings it cannot pass on, and add() "
                   "refuses a spec it would silently narrow."},
        {"rule": "Check the LIVE process, not the queue file",
         "cost": "the same 6 runs",
         "detail": "command_line() read from a fresh interpreter showed the "
                   "flags. The runner never saw that interpreter. The only "
                   "honest check is the command line of the process actually "
                   "running."},
        {"rule": "A bar in the wrong unit is not a bar",
         "cost": "about a day",
         "detail": "'under 100 mm sideways' is 0.94 deg of heading held for 25 "
                   "seconds over the distance walked. Nothing could pass it, and "
                   "the failure was read as a tuning problem for weeks."},
        {"rule": "Keep confirmation runs independent of the experiment",
         "cost": "saved 3 runs",
         "detail": "The three final seeds deliberately did NOT carry the tilt "
                   "change being tested alongside them. The tilt change failed; "
                   "the finals passed. Bundled, all three would have failed and "
                   "the step would not have closed."},
        {"rule": "Training-only events must be gated on `play`",
         "cost": "caught before it ran",
         "detail": "The handover resets would otherwise have fired during "
                   "verification, putting 15% of test robots into a wobble and "
                   "moving every bar for a reason invisible on the page."},
    ],
}


def carry_over() -> dict:
    """CARRY, joined with the numbers read live from the code.

    Everything with a number the code also holds - weights, ramps, tolerances,
    the observation, the command box - comes from gray/tasks/ on every call,
    never from the dict above. Same rule the per-run tables follow, for the same
    reason: a hand-written copy of a weight is a weight that will be wrong within
    a week.
    """
    out = {**CARRY, "terms": [], "ramps": [], "tolerances": [], "observes": [],
           "commands": [], "training": [], "error": ""}
    try:
        import importlib  # noqa: PLC0415

        stand = importlib.import_module("gray.tasks.stand_env_cfg")
        push = importlib.import_module("gray.tasks.push_env_cfg")
        walk = importlib.import_module("gray.tasks.walk_env_cfg")
        for module in (stand, push, walk):
            importlib.reload(module)
        notes = {**stand.REWARD_NOTES, **push.PUSH_NOTES, **walk.WALK_NOTES}
        cfg = walk.walk_env_cfg()
        agent = walk.walk_ppo_cfg()
    except Exception as exc:  # noqa: BLE001
        out["error"] = (f"Could not read gray/tasks/ ({type(exc).__name__}: {exc}). "
                        f"Nothing is shown rather than something out of date.")
        return out

    ramp = {c.params["reward_name"]: [s["weight"] for s in c.params["stages"]]
            for c in (cfg.curriculum or {}).values() if c.params.get("reward_name")}
    for name, term in cfg.rewards.items():
        end = ramp[name][-1] if name in ramp else term.weight
        out["terms"].append({"name": name, "start": term.weight, "end": end,
                             "ramped": name in ramp,
                             "sign": "+" if end > 0 else "-",
                             "note": notes.get(name, "")})
    out["terms"].sort(key=lambda t: -abs(t["end"]))
    out["ramps"] = [{"name": k, "stages": v} for k, v in sorted(ramp.items())]

    walk_cmd = cfg.commands.get("walk")
    posture = cfg.commands.get("posture")
    if walk_cmd is not None:
        r = walk_cmd.ranges
        out["commands"] = [
            {"name": "forward", "range": f"{r.lin_vel_x[0]} to {r.lin_vel_x[1]}",
             "unit": "m/s"},
            {"name": "sideways", "range": f"{r.lin_vel_y[0]} to {r.lin_vel_y[1]}",
             "unit": "m/s"},
            {"name": "turn", "range": f"{r.ang_vel_z[0]} to {r.ang_vel_z[1]}",
             "unit": "rad/s"},
            {"name": "told to stand still",
             "range": f"{walk_cmd.rel_standing_envs:.0%}", "unit": "of draws"},
            {"name": "drawn as a straight line",
             "range": f"{walk_cmd.rel_straight_envs:.0%}", "unit": "of the rest"},
        ]
    if posture is not None:
        out["commands"] += [
            {"name": "ride height",
             "range": f"{posture.ranges.height[0]} to {posture.ranges.height[1]}",
             "unit": "m"},
            {"name": "pitch",
             "range": f"{posture.ranges.pitch[0]} to {posture.ranges.pitch[1]}",
             "unit": "rad"},
            {"name": "roll",
             "range": f"{posture.ranges.roll[0]} to {posture.ranges.roll[1]}",
             "unit": "rad"},
        ]

    for term, unit, why in (
            ("track_turn", "rad/s", "how sharply yaw rate is scored"),
            ("upright", "rad", "how sharply trunk tilt is scored. 0.45 rad is 26 deg"),
            ("track_speed", "m/s", "how sharply speed is scored"),
            ("ride_height", "m", "how sharply ride height is scored")):
        t = cfg.rewards.get(term)
        if t is not None and "std" in (t.params or {}):
            out["tolerances"].append({"term": term, "value": t.params["std"],
                                      "unit": unit, "why": why})
    if walk_cmd is not None:
        for attr, unit, why in (
                ("gyro_bias_rad", "rad",
                 "fixed error on the heading the POLICY reads, per episode"),
                ("gyro_walk_rad_per_s", "rad/rt-s",
                 "how far that reading wanders. 0 is a gyro the robot has not got")):
            if hasattr(walk_cmd, attr):
                out["tolerances"].append({"term": "heading sensor",
                                          "value": getattr(walk_cmd, attr),
                                          "unit": unit, "why": why})

    group = cfg.observations.get("actor")
    out["observes"] = sorted(getattr(group, "terms", {}) or {})
    out["training"] = [
        {"name": "iterations", "value": str(agent.max_iterations),
         "why": "99.1% of best reward; past it buys less than the noise"},
        {"name": "robots at once", "value": str(cfg.scene.num_envs),
         "why": "the card's measured ceiling"},
        {"name": "episode length", "value": f"{cfg.episode_length_s} s",
         "why": "1000 control steps at 50 Hz"},
        {"name": "control rate", "value": "50 Hz",
         "why": "the servo's PWM period. Not negotiable"},
    ]
    return out
