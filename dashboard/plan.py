"""The plan, as data.

Everything the dashboard shows lives here, not in the HTML. To change what the
robot is being taught, or how it is scored, edit this file. The page is only a
renderer.
"""

from __future__ import annotations

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
        "The training happens in a physics simulator, not on the real robot. Thousands "
        "of copies of Gray run at once on the graphics card. Each copy tries something, "
        "gets a score, and the ones that score better shape what all of them try next. "
        "After a few hours this produces a single small file - a few hundred kilobytes - "
        "that maps what the robot senses to what the joints should do.",
        "That file then goes on the Raspberry Pi and drives the real robot. The whole "
        "game is making the simulator honest enough that a controller which works in it "
        "also works outside it.",
    ],
    "success": "Gray walks across a floor, in a straight line, without falling, "
               "driven by a policy trained entirely in simulation.",
}

# ---------------------------------------------------------------------------
# The curriculum - what gets taught, in order
# ---------------------------------------------------------------------------
#
# Each stage is only started once the one before it passes its bar. That
# discipline is the thing that was missing last time.

STAGES = [
    {
        "n": 0,
        "name": "A model worth trusting",
        "kind": "setup",
        "goal": "Get a URDF out of SolidWorks that describes the real robot.",
        "why": "Every stage below is measured inside a simulator. If the simulator's "
               "robot is the wrong mass, or its joints bend further than the real ones "
               "can, then everything learned on top of it is learned about a robot that "
               "does not exist.",
        "teaches": [],
        "rewarded": [],
        "bar": "All 10 checks in tools/check_urdf.py pass.",
        "effort": "CAD work. No training.",
    },
    {
        "n": 1,
        "name": "Stand still",
        "kind": "train",
        "goal": "Hold a stable standing pose, feet under the hips, and do not fall over.",
        "why": "This is the cheapest possible test of whether the model is right. If "
               "the robot cannot stand, the mass is wrong, or the joint limits are "
               "wrong, or the servos are not strong enough - and it is far better to "
               "find that out in ten minutes than after a six-hour training run.",
        "teaches": [
            "Push against gravity without collapsing",
            "Keep the trunk level",
            "Hold a target ride height",
        ],
        "rewarded": ["upright", "height", "alive", "joint_limit", "effort", "body_contact"],
        "bar": "30 s without falling. Trunk height within 5 mm of target. Uprightness above 0.99.",
        "effort": "Minutes.",
    },
    {
        "n": 2,
        "name": "Take a push",
        "kind": "train",
        "goal": "Stay standing when shoved, and when the floor is not flat or not grippy.",
        "why": "This is the first task that cannot be solved by memorising a pose. "
               "Standing still can be: a policy could ignore every sensor, replay one "
               "set of angles forever, and nobody would know. A push breaks that - the "
               "robot has to notice it moved and do something about it. So this is "
               "also the first real use of the potentiometers, and it proves the "
               "observation space works before anything harder depends on it.",
        "why_now": "Walking needs exactly this reflex. Every step is a disturbance the "
                   "robot creates itself - weight shifts, a foot lands early, the trunk "
                   "rocks - and a policy that only knows one still pose has no answer "
                   "when that happens. The previous attempt hit this precisely: the "
                   "hand-written gait walked but wandered about 140 mm, and the note "
                   "recorded that it needed active balance.\n\n"
                   "It is also where domain randomisation goes cheaply. Friction, mass, "
                   "servo lag and uneven ground can all be varied while the task is "
                   "still simple, for minutes of GPU time; learning the same robustness "
                   "during walking costs hours.\n\n"
                   "It is not strictly required. Stand then walk would work, and pushes "
                   "could be added later. But then a fall while walking has two possible "
                   "causes - the gait or the balance - and there is no way to tell which. "
                   "Doing it now rules one of them out.",
        "teaches": [
            "React to a disturbance it did not cause",
            "Use measured joint angles, not just commanded ones",
            "Widen the stance when it needs to",
        ],
        "rewarded": ["upright", "height", "alive", "effort", "action_rate", "body_contact"],
        "bar": "Survives a 30 N-s shove from any direction, 9 times out of 10, "
               "on ground friction anywhere from slippery to grippy.",
        "effort": "About an hour.",
    },
    {
        "n": 3,
        "name": "Lift one foot",
        "kind": "train",
        "goal": "Shift weight onto three legs, lift the fourth foot 30 mm, hold it, put it back.",
        "why": "This is the atom of walking. Every gait, no matter how complicated, is "
               "this move repeated in a sequence. If the robot can do it cleanly on all "
               "four legs it can be made to walk; if it cannot, no amount of gait tuning "
               "will help.",
        "teaches": [
            "Move its centre of mass deliberately",
            "Load and unload a leg without lurching",
            "Place a foot where it intended to",
        ],
        "rewarded": ["foot_target", "upright", "height", "slip", "alive", "body_contact"],
        "bar": "All four legs, 20 lifts each, no fall, trunk moves less than 10 mm.",
        "effort": "One to two hours.",
    },
    {
        "n": 4,
        "name": "Step in place",
        "kind": "train",
        "goal": "Cycle all four legs continuously while staying on the same spot.",
        "why": "Walking is the atom from stage 3 run in a loop with the timing right. "
               "Doing it without going anywhere separates the timing problem from the "
               "travelling problem, so when it does go wrong there is only one thing "
               "it can be.",
        "teaches": [
            "Keep a rhythm",
            "Never lift two feet that share a corner",
            "Land softly instead of dropping",
        ],
        "rewarded": ["gait_phase", "foot_clearance", "soft_landing", "upright", "slip", "drift"],
        "bar": "20 s of stepping, drifts less than 50 mm from where it started.",
        "effort": "Two to three hours.",
    },
    {
        "n": 5,
        "name": "Walk forward",
        "kind": "train",
        "goal": "Travel forward at a commanded speed, in a straight line.",
        "why": "This is the headline goal. The bar is set against the hand-written gait "
               "from the previous attempt, which managed 675 mm in 12 seconds - so there "
               "is a real number to beat rather than a vague sense of progress.",
        "teaches": [
            "Match a speed it is told to travel at",
            "Go straight despite the legs not being perfectly symmetric",
            "Keep walking when the ground changes under it",
        ],
        "rewarded": ["track_velocity", "gait_phase", "foot_clearance", "soft_landing",
                      "upright", "drift", "slip", "effort", "action_rate"],
        "bar": "Beats 56 mm/s, veers less than 50 mm over 8 s, and does not fall - "
               "with the simulator's mass, friction and servo strength randomised.",
        "effort": "Three to six hours.",
    },
    {
        "n": 6,
        "name": "Steer",
        "kind": "train",
        "goal": "Follow commanded forward speed, sideways speed and turn rate.",
        "why": "A robot that can only walk in one straight line is not much use. This is "
               "also what a gamepad would eventually plug into.",
        "teaches": ["Turn on the spot", "Crab sideways", "Blend all three at once"],
        "rewarded": ["track_velocity", "track_yaw", "gait_phase", "foot_clearance",
                      "upright", "slip", "effort"],
        "bar": "Tracking error under 20% across the whole commanded range.",
        "effort": "Three to six hours.",
    },
    {
        "n": 7,
        "name": "Get up",
        "kind": "train",
        "goal": "From lying on its belly, return to standing.",
        "why": "This is how every real session starts, and how every fall ends. Without "
               "it somebody has to pick the robot up by hand every single time.",
        "teaches": ["Fold the legs underneath itself", "Push up without tipping over",
                     "Recover from any starting pose"],
        "rewarded": ["height", "upright", "alive", "effort", "joint_limit"],
        "bar": "Stands up from 9 out of 10 random belly poses, in under 3 s.",
        "effort": "Two to four hours.",
    },
    {
        "n": 8,
        "name": "Rough ground",
        "kind": "train",
        "goal": "Walk over slopes, steps and loose ground without falling.",
        "why": "Real floors are not the flat plane in the simulator. This is the stage "
               "that decides whether the robot works only on a desk.",
        "teaches": ["Cope with a foot landing early or late", "Recover mid-stride",
                     "Slow down when it needs to"],
        "rewarded": ["track_velocity", "upright", "soft_landing", "foot_clearance",
                      "body_contact", "slip"],
        "bar": "Crosses 20 mm steps and 10 degree slopes with under 10% falls.",
        "effort": "Six hours or more.",
    },
    {
        "n": 9,
        "name": "The real robot",
        "kind": "deploy",
        "goal": "Run the trained controller on Gray itself.",
        "why": "Everything above is preparation for this. It needs the robot "
               "reassembled, the potentiometers fitted and calibrated, and the power "
               "system built.",
        "teaches": [],
        "rewarded": [],
        "bar": "Gray walks across the floor.",
        "effort": "Hardware work. Blocked until the robot is rebuilt.",
    },
]

# ---------------------------------------------------------------------------
# How points are given
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
    "Most terms use the same shape: score = exp(-(error squared) / tolerance). That "
    "gives 1.0 for perfect, about 0.6 when the error equals the tolerance, and "
    "approaches 0 beyond it. It is smooth, which matters - a reward that jumps gives "
    "the training process nothing to follow.",
]

REWARDS = [
    {"key": "track_velocity", "sign": "+", "name": "Match the commanded speed",
     "measures": "Difference between how fast the trunk is actually moving and how fast it was told to move.",
     "note": "Measured on a smoothed velocity, not the raw one. Each footfall makes the "
             "trunk surge; that ripple was four times bigger than the average speed and "
             "it flattened this reward to zero last time."},
    {"key": "track_yaw", "sign": "+", "name": "Match the commanded turn rate",
     "measures": "Difference between actual and commanded rotation about the vertical.", "note": ""},
    {"key": "upright", "sign": "+", "name": "Stay upright",
     "measures": "How close the trunk's up direction is to the world's up direction. 1.0 is perfectly level.", "note": ""},
    {"key": "height", "sign": "+", "name": "Hold ride height",
     "measures": "Trunk height above the ground against its target.", "note": ""},
    {"key": "foot_target", "sign": "+", "name": "Put the foot where intended",
     "measures": "Distance between where a swinging foot is and where it was supposed to be.", "note": ""},
    {"key": "foot_clearance", "sign": "+", "name": "Lift the feet clear",
     "measures": "Peak height of a foot during its swing.",
     "note": "Without this the robot learns to scuff its feet along the floor, which "
             "works in simulation and trips on any real surface."},
    {"key": "soft_landing", "sign": "+", "name": "Land softly",
     "measures": "Downward speed of a foot at the moment it touches.",
     "note": "Hard landings shake the trunk and, on the real robot, break printed parts."},
    {"key": "gait_phase", "sign": "+", "name": "Keep the rhythm",
     "measures": "Whether each foot is on the ground when the gait says it should be.", "note": ""},
    {"key": "alive", "sign": "+", "name": "Still standing",
     "measures": "A small fixed bonus for every moment it has not fallen.",
     "note": "Stops the robot from ending an attempt early to escape a negative score."},
    {"key": "body_contact", "sign": "-", "name": "Something other than a foot touched the ground",
     "measures": "Any contact involving the trunk, a thigh or a shank.", "note": ""},
    {"key": "slip", "sign": "-", "name": "Foot slipped",
     "measures": "A foot sliding sideways while it is carrying weight.", "note": ""},
    {"key": "drift", "sign": "-", "name": "Went the wrong way",
     "measures": "Sideways or rotational motion that was never commanded.",
     "note": "This is the term that pays for walking straight."},
    {"key": "joint_limit", "sign": "-", "name": "Hit the end stop",
     "measures": "How far into the last few degrees of a joint's travel it went.",
     "note": "On the real robot this is a servo straining against a hard stop."},
    {"key": "effort", "sign": "-", "name": "Worked too hard",
     "measures": "Total torque across all twelve joints.",
     "note": "Keeps the gait within what a 20 kg-cm servo can actually deliver, and "
             "saves battery."},
    {"key": "action_rate", "sign": "-", "name": "Twitchy commands",
     "measures": "How much the commanded angles changed from one moment to the next.",
     "note": "Smooth commands are the single biggest factor in whether a simulated "
             "controller survives contact with real hardware."},
]

# ---------------------------------------------------------------------------
# Can this machine actually train it
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The plan, in four lines. This is the front page - the long version is the
# curriculum above, on the summary page.
# ---------------------------------------------------------------------------

PHASES = [
    {
        "n": 1,
        "name": "Digital twin",
        "state": "done",
        "one_line": "A simulated Gray that matches the real one.",
        "detail": "Rebuilt from SolidWorks. 11 of 11 checks pass: 2378.70 g, joint "
                  "axes exact, and the owner's measured travel limits on all twelve "
                  "joints, verified by turning each one in the simulator.",
        "needs_robot": False,
    },
    {
        "n": 2,
        "name": "Move it by hand",
        "state": "next",
        "one_line": "Work out where a foot is from three angles, and back again. "
                    "Then a hand-written walk.",
        "detail": "No learning yet. This is the thing the learned controller is "
                  "measured against, so it has to exist first. Ends with a number: "
                  "mm/s, and whether it walks straight.",
        "needs_robot": False,
    },
    {
        "n": 3,
        "name": "Teach it",
        "state": "later",
        "one_line": "Stand still, stand up, take a push, then walk.",
        "detail": "Each stage has to pass its bar before the next starts. Cheapest "
                  "failures first: standing still takes minutes and answers whether "
                  "the servos can hold the robot up at all.",
        "needs_robot": False,
    },
    {
        "n": 4,
        "name": "The real robot",
        "state": "blocked",
        "one_line": "Reassemble, fit the potentiometers, calibrate, deploy.",
        "detail": "Everything above needs no hardware. This needs the robot back "
                  "together and the position sensors fitted.",
        "needs_robot": True,
    },
]

# What has to be true before the next stage can start. Short, checkable, honest.
NEXT_UP = {
    "title": "Stand still",
    "why": "Can twelve servos at 1.96 N-m hold up 2.378 kg? Minutes of training, "
           "and it decides whether anything else is possible.",
    "before": [
        {"task": "Static torque check", "who": "code",
         "note": "Work out what each joint needs to hold a standing pose, against "
                 "the servo's 1.96 N-m. A calculation, not a training run."},
        {"task": "Actuators in the simulation", "who": "code",
         "note": "The model has twelve free-swinging joints and nothing driving "
                 "them. Position servos at 50 Hz."},
        {"task": "A standing pose", "who": "code",
         "note": "The exported zero is the sprawl the CAD was posed in - feet 557 mm "
                 "apart front to back."},
    ],
    "open": [
        {"q": "bl_hip's inertia is the one assumed number in the model.",
         "note": "The exporter gave that link no mass, so its tensor is copied from "
                 "another hip and mirrored. Same part, so it is close."},
        {"q": "The masses are CAD numbers, not scale readings.",
         "note": "2378.70 g comes from SolidWorks. Weigh the real parts during "
                 "reassembly and replace them."},
    ],
}

FEASIBILITY = {
    "verdict": "Yes. The hardware is not the limiting factor - the model is.",
    "facts": [
        {"k": "Graphics card", "v": "NVIDIA RTX 4070 Ti", "note": "measured on this machine"},
        {"k": "Simulator", "v": "MuJoCo 3.10 via mjlab", "note": "runs the physics on the GPU"},
        {"k": "Robots trained at once", "v": "8,000 - 16,000", "note": "estimate, to be measured"},
        {"k": "Control rate", "v": "50 Hz", "note": "hard limit, set by the servo PWM period"},
        {"k": "Time for one stage", "v": "1 - 6 hours", "note": "estimate, from the previous attempt"},
        {"k": "Size of the trained file", "v": "a few hundred KB", "note": "small enough to keep every one"},
    ],
    "risks": [
        {"risk": "The knee actuator is too slow.",
         "detail": "A knee has to move roughly 40 degrees in 0.2 s during a swing - "
                   "about 90 mm/s of travel. Typical hobby linear actuators manage "
                   "5-20 mm/s. If the chosen actuator is in that range the robot "
                   "cannot swing a leg fast enough to walk at all, no matter how good "
                   "the controller is. This needs a number before anything is modelled.",
         "level": "high"},
        {"risk": "The mass in the CAD is not the mass of the robot.",
         "detail": "Everything the simulator predicts about torque and balance scales "
                   "with mass. The previous model was out by 30%. Put the real robot on "
                   "a scale during reassembly and match it.",
         "level": "medium"},
        {"risk": "Backlash and slop are not in the simulator.",
         "detail": "Ball joints, servo gear trains and printed parts all have play. The "
                   "simulator has none, so the controller never learns to expect it. "
                   "This is the usual reason a policy that works in simulation does not "
                   "work outside it.",
         "level": "medium"},
        {"risk": "A reward that pays for the wrong thing.",
         "detail": "Three training runs were lost to this last time. Every new reward "
                   "gets checked against a recorded walk before it is trusted.",
         "level": "medium"},
    ],
}
