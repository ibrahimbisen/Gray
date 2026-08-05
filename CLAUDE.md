# CLAUDE.md

## How to talk to the user
Always no matter what use ASD-STE100 Simplified Technical English (STE).

Load the `ste` skill at the start of every session, before you write the first reply. It
holds the rules, the word substitutions, and the self-check.

**The rule again, because it is the one that matters: always use ASD-STE100 Simplified Technical English (STE)**

---

## What Gray is

A 12-DOF 3D-printed quadruped robot. Designed and built by the user in 2021–22. The
mechanical work is done. The robot is currently disassembled, all parts accounted for.

The goal is to make it walk: correct digital model → classical gait → reinforcement
learning on top → deploy to the real robot.

## Where the project stands

**Full reset, Aug 2026.** The previous software stack was built on a bad CAD export and
has been archived to the git tag `archive/attempt-1`. Nothing from it carries forward.

The rebuild starts in SolidWorks: remodel the robot, export a correct URDF, and only then
write software on top of it.

## Hardware facts that do not change

These are facts about the physical robot. They stay true no matter what the CAD says.
Each one cost real time to discover.

- **12 joints.** Currently DS3218MG 270° hobby servos on a PCA9685 board over I²C.
- **Where the 12 servos physically are.** Four are embedded in the body, one per leg,
  and swing the hip out and in. The other eight are in the hip modules, two per leg:
  one drives the thigh, the other drives the calf through the pushrod. **The thigh and
  the calf contain no servo at all** — the thigh carries only the rod and its ball
  joints. This is what sets how heavy a leg is to swing and how much a shove has to
  arrest, so it drives the mass model in `gray/config/robot.yaml`. It has been guessed
  wrong twice from photos and CAD; the layout above came from the owner and is the
  one to trust.
- **Three IMUs on the trunk: front, exact middle, and back.** The policy reads two
  IMU signals every single step — which way is down, and how fast the trunk is
  turning. Without an IMU no trained policy can run on the real robot at all, so
  this is not an accuracy improvement, it is the thing that makes deployment
  possible. The middle unit is the one the simulation models, because it sits
  closest to the trunk's centre of mass, which is where `projected_gravity` and
  `base_ang_vel` are defined. The front and back pair are worth more than spares:
  differencing them measures pitch directly, and averaging all three cuts the
  vibration noise a single unit would feed straight into the policy.
- **Every joint will have position feedback, from potentiometers the owner installs.**
  Rotary pots on the rotating joints, linear pots on the pushrod. This is load-bearing —
  design for it from the start. Measured joint angles are allowed in the observation space,
  and closed-loop control is on the table. The old design banned both because the hardware
  could not report position. That restriction is gone.
- **Potentiometers are analog and the Raspberry Pi has no analog inputs.** An external ADC
  chip is required — 12 channels of it. This is a hard requirement, not an optimisation.
- **PWM period is 20 ms, so 50 Hz is still the control ceiling.** The servos are still
  driven by PWM. Feedback changes what can be sensed, not how fast commands go out.
- **The knee is a pushrod linkage, not direct drive.** A thigh-mounted servo drives the
  shank through a ball-jointed metal rod. Servo angle ≠ joint angle. A linear pot on the
  rod measures the true knee angle directly and removes this problem.
- **Every pot needs calibration.** Raw ADC counts → radians, per joint, stored in a file.
  The model is only as good as that mapping.

## Rules of the repo

- **The CAD rebuild must include mounts for the potentiometers.** Pot bodies, shaft
  couplings, clearance, and wire routing. They are part of the robot's mass and geometry.
  Adding them later means redoing the model.
- `robot/` is the original SolidWorks CAD. Never edit, move, or delete anything in it.
- `Overview/` is build photos. Same rule. They are often the only record of how the real
  robot is put together.
- **The training monitor is updated in the same change that alters what it shows.**
  Whoever makes the change — the owner or the agent. Change a reward weight, a stage,
  a rule, a mass, a bar to pass; the dashboard reflects it before the change is
  finished. Not afterwards, not in the next commit.

  This is not tidiness. A dashboard that is out of date is worse than no dashboard,
  because it is trusted: a stale reward table means a run gets read against a scoring
  function it was never trained on, and hours are spent explaining a result that was
  never real. That has already happened once here, which is why `verify.py` warns when
  a policy is scored against a changed task.

  What that means in practice:
  - Per-run tables come from the run's own `run.json`, written by `scripts/train.py`
    from the live config. These are correct by construction — do not hand-edit them.
  - Any new reward term needs a plain-English note in `REWARD_NOTES` or `PUSH_NOTES`
    in the same edit. `train.py` refuses to start if one is missing. Do not work
    around that check; it is the rule enforcing itself.
  - The `/summary` page reads `dashboard/plan.py`, which is written by hand. It is
    the part that goes stale. If a weight, stage or bar moves, `plan.py` moves too.
- Nothing else is sacred. Delete freely.
