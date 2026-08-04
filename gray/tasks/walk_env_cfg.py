"""Stage 5 - walk.

Stages 3 and 4 (lift one foot, step in place) are skipped on purpose: the owner
wants to see how much a policy picks up in a fixed amount of training, and going
straight from "recovers from a shove" to "walks" is the shortest way to find out.
Built on the push task, so it keeps that task's domain randomisation, its contact
sensors and its anti-skid term - but the shoves are switched off, because a first
walk has enough to deal with.

The scoring here follows the stage 5 table in docs/REWARDS.md. Three things in it
are mjlab defaults that are wrong for THIS robot rather than wrong in general, and
all three fail silently - the run trains, the curves look plausible, and a term
that was supposed to be doing the work is quietly worth nothing.

**1. The tracking band, sigma.** Tracking pays `exp(-error^2 / sigma^2)`. mjlab
ships sigma = 0.5 m/s, tuned for ANYmal and Go1, which walk at 0.6-1.5 m/s. Gray
walks at 0.25. Put those together and a robot that never moves at all collects

    exp(-0.25^2 / 0.5^2) = 0.78

- 78% of full marks for standing still. There is then almost nothing to gain by
walking and a great deal to lose by falling over, so the policy learns to stand.
That is what the archived attempt at this project did, and IsaacLab issue #458
reports the same thing independently. At sigma = 0.15 the same robot collects
0.06, and walking becomes the only way to score.

**2. Speed gates set for a faster robot.** Four separate terms multiply themselves
by zero unless the commanded speed clears a threshold, and the defaults are set
where a Go1 lives, not where Gray does:

    feet_air_time      command_threshold  0.5    Gray never exceeds 0.35
    feet_swing_height  command_threshold  0.5
    variable_posture   walking_threshold  0.5    -> Gray is always "standing"
    variable_posture   running_threshold  1.5

On defaults, the term that pays for picking a foot up is dead for the whole run,
and the posture term holds Gray to its tight standing tolerance while asking it to
walk. Every threshold is retuned below.

**3. Terms that need a sensor we do not have.** mjlab's `feet_clearance` and
`feet_swing_height` both require a `TerrainHeightSensor`. On a flat floor, height
above the environment origin is the same number, so both are rewritten here
against the foot sites already in the model. On rough terrain (stage 8) they must
go back to the real sensor.

**Smoothness is ramped, not switched on.** `twitching` rises from -0.01 to -0.05
over the first 250 iterations. docs/REWARDS.md is emphatic about this and so is
our own history: push_v3 set a large smoothness penalty from step 0 and the robot
stopped being able to catch itself. RMA reports the same failure and the same fix.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.tasks.velocity import mdp as vmdp

from mjlab.managers import SceneEntityCfg

from gray.tasks.posture_command import PostureCommandCfg
from gray.tasks.walk_command import StraightLineVelocityCommandCfg
from gray.tasks.push_env_cfg import FOOT_SITES, push_env_cfg, push_ppo_cfg
from gray.tasks.stand_env_cfg import ALL_JOINTS, _stance

# The trunk, named explicitly. mjlab's default SceneEntityCfg("robot") resolves
# to every body, and a term expecting one trunk then gets 13 of them - a shape
# mismatch that names neither the term nor the reason.
ROBOT = SceneEntityCfg("robot", body_names=("base_link",))

# The two joints that make a stride look like a stride. The hip only swings the
# leg sideways, so animating it does nothing for how the walk reads.
SWING_JOINTS = SceneEntityCfg("robot", joint_names=(".*thigh", ".*calf"))

# How fast it is asked to walk. Slow, because 1.96 N-m servos at 50 Hz are not
# going to run, and because a speed the robot cannot reach is a reward it can
# never earn.
# THE BOX. Every command the policy will ever be trusted on is drawn from inside
# these, and nothing outside them means anything: the policy interpolates between
# things it has seen and produces nonsense past the edge, the same way a curve fit
# does. Widened 3 Aug 2026 - see the note under WALK_SPEED.
WALK_SPEED = (-0.35, 0.35)    # m/s, negative is backward
WALK_SIDE = (-0.20, 0.20)     # m/s sideways
WALK_TURN = (-1.00, 1.00)     # rad/s about the vertical

# What the old box cost. Until today WALK_SPEED was (0.15, 0.35) - never zero,
# never negative - so across roughly 900,000 draws a backward command was never
# issued once. Driving run #25 by hand confirmed it: told to walk backward it
# covered 0.00 m in 8 seconds, and told to crab sideways, 3 cm. Four rows of the
# skill library were filed as "a command" when the command had never been asked.
#
# WALK_SIDE was symmetric but only ever drawn ON TOP of forward motion of at
# least 0.15 m/s, so pure sideways never happened either. It happens now because
# vx can be drawn near zero.
#
# THE TRAP THIS DEPENDS ON: _going_straight() gated on `command[:, 0] > MOVING`,
# positive-only. Widening through zero without fixing that to abs() would have
# switched `veering` and `wandering` off for every backward command, and backward
# walking would have trained with no straightness penalty while the terms quietly
# reported zero. Fixed in the same change; do not separate them.

# Where the trunk should BE. Measured on the owner's stance by solving for the
# joint angles that keep every foot on its own print while the trunk moves, and
# finding where the legs run out of travel:
#
#     height   120 to 270 mm    holds throughout, 1.59x the servo at the lowest
#     pitch    nose up 20 deg, nose DOWN only 10 deg
#     roll     +/- 30 deg, and the sweep never found the limit
#
# Commanded ranges sit inside those, because a limit measured standing still is
# not a limit while walking: a leg already at its end stop has nothing left to
# swing with. Pitch stays asymmetric because the geometry is - the stance rakes
# the legs forward, which spends travel that nose-down needs. Squaring it off to
# +/-10 would throw away half the nose-up travel for tidiness.
POSE_HEIGHT = (0.15, 0.25)    # m, trunk off the ground
POSE_PITCH = (-0.26, 0.14)    # rad, nose down positive: 15 deg up, 8 deg down
POSE_ROLL = (-0.35, 0.35)     # rad, right side down positive: +/- 20 deg

# The band the tracking reward falls off over. See the module docstring - this is
# the single most important number in the file. docs/REWARDS.md puts it at
# 0.3-0.5x the target speed, against mjlab's borrowed 0.5.
TRACK_STD = 0.15

# The turning band. This is the same trap as the tracking band above, in the
# other direction, and it cost three runs to find.
#
# track_angular_velocity pays exp(-(yaw error squared + xy rotation) / std^2).
# At std = 0.30, the yaw error the robot ACTUALLY makes - measured at 1.5 rad/s,
# flat across all 3000 iterations of walk_m3100_a and _b - pays:
#
#     exp(-1.5^2 / 0.30^2) = 1.4e-11
#
# That is not a small reward, it is a dead one. The term is saturated at zero
# across the whole region the policy operates in, so it has no gradient and the
# policy was never told to stop spinning. Both runs walked over 5 m without
# falling and then drifted 3.5-4 m sideways, and this is why. Raising the WEIGHT
# does nothing either: twice zero is zero.
#
# At std = 0.80 the same error pays 0.030, rising to 0.68 at 0.5 rad/s and 0.94
# at 0.2 - a real slope the whole way down. The reward stays capped at 1.0 so the
# ceiling, and RULES.md rule 1's stop threshold, are unchanged.
#
# The lesson generalises: a tracking band has to be set against the error the
# robot MAKES, not only against the command it is given. Both numbers here have
# now been wrong in opposite directions for the same reason.
TURN_STD = 0.80

# The lowest command that still counts as "being asked to move". Every gated term
# uses this instead of mjlab's 0.5, which Gray never reaches.
MOVING = 0.05

# How high a foot should be at the top of its swing. mjlab's default target is
# 0.1 m, set for a Go1; Gray rides at 0.19 m, so a tenth of a metre is most of
# its ride height. 35 mm matches the lift the stage 3 bar asks for.
SWING_TARGET = 0.035

# How many control steps the trunk speed is averaged over before it is scored.
# At 50 Hz, 12 steps is 0.24 s, about half a stride. Every footfall makes the
# trunk surge, and that ripple was measured at four times the mean speed - scored
# raw it collapses the tracking reward at every step, no matter how well the
# robot is actually walking. Nothing in mjlab does this; it has to be written.
FILTER_STEPS = 12

# How far a thigh or calf should swing either side of its own average, in radians.
# 0.15 rad is 8.6 degrees of typical deviation, so a stride spanning roughly 17
# degrees at each of those joints. Below that the robot creeps along on stiff legs
# and technically walks; the owner's word for what is wanted instead is
# "animated". Capped, because uncapped this buys flailing.
SWING_SPREAD = 0.15

# How far sideways the robot may wander before it costs, in metres, when it has
# been told to go straight. The previous attempt's hand-written gait walked but
# drifted about 140 mm, which is the failure this exists to price.
DRIFT_FREE_M = 0.05

WALK_NOTES = {
    "track_speed": "Moving at the speed it was told to, measured on a trunk speed "
                   "averaged over about half a stride rather than the raw one. This "
                   "is the term that pays for walking at all.",
    "track_turn": "Turning at the rate it was told to, and not turning when it was "
                  "not asked to.",
    "stepping": "Feet spending a sensible time in the air - long enough to be a "
                "step, short enough not to be a hop. This is what turns 'move "
                "forwards' into 'walk' rather than 'shuffle'. Pays nothing when "
                "the robot was told to stand still: it had no such gate until "
                "3 Aug 2026, and since every term that would charge for marching "
                "is itself switched off below MOVING, jogging on the spot was the "
                "cheapest way to obey 'stop'.",
    "ride_height": "Trunk at the height it was TOLD to hold. Without it the "
                   "cheapest way to move is to sink onto its belly and crawl. "
                   "Scored against a fixed 202 mm until 3 Aug 2026, when height "
                   "became something you can ask for - so this is now what makes "
                   "'crouch' and 'stand tall' commands rather than accidents.",
    "upright": "Trunk leaning the way it was TOLD to lean, measured off gravity - "
               "so it needs no outside reference and works on the real robot from "
               "its IMU alone. Scored one attitude, level, until 3 Aug 2026, which "
               "meant a commanded lean was a thing the robot got punished for. "
               "Level is still here; it is this command at zero.",
    "posture": "Joints near their default pose, but with the tolerance widened once "
               "the robot is asked to move. A fixed tolerance is the thing that "
               "stops a gait developing at all - it pays the robot to keep its legs "
               "where they started, which is the opposite of walking.",
    "dragging": "A foot at the wrong height while it is travelling. This is what "
                "stops the dragging leg that shows up in every walking run without "
                "it, and it is scored continuously rather than only at touchdown.",
    "swing_height": "How wrong the top of a swing was, scored when the foot lands. "
                    "Catches a foot that skims the floor and one that is thrown "
                    "needlessly high, neither of which survives a real floor.",
    "landing_speed": "How fast a foot is still falling at the moment it touches. "
                     "hard_landing charges for the force of the impact; this "
                     "charges for the approach, which is what a policy can "
                     "actually do something about - it has to slow the foot "
                     "BEFORE contact, not react afterwards. Ramped in rather than "
                     "switched on, because a large landing penalty from step 0 "
                     "teaches a robot not to put its feet down at all.",
    "hard_landing": "Slamming a foot down. Printed PLA with no suspension, so this "
                    "is set harder than the reference value.",
    "joint_shock": "Joint acceleration. Footfalls spike it, and a servo gearbox is "
                   "what absorbs that spike on the real robot.",
    "ground_covered": "Ground actually put behind it, along the direction it was "
                      "told to go. Speed tracking alone can be satisfied by rocking "
                      "forwards and backwards, which averages out to the right speed "
                      "and gets nowhere; this only pays for net progress, so that "
                      "trick earns nothing. Rewritten 3 Aug 2026 - it measured world "
                      "X against the magnitude of the command, and so could not pay "
                      "for backward at all, paid nothing for a crab step, paid "
                      "nothing to a robot walking perfectly while turned 90 degrees, "
                      "and paid FULL MARKS to a robot told to stand still that "
                      "drifted forwards.",
    "leg_swing": "How far the thighs and calves swing either side of their own "
                 "average. This is what makes the walk look like a dog rather than "
                 "a table sliding along - without it the cheapest gait is stiff "
                 "legs and tiny steps. Capped, so it buys a stride and not a flail.",
    "wandering": "Drifting sideways off the line it was sent along. Only charged "
                 "when it was told to go straight, and measured on where the robot "
                 "has actually ended up rather than which way it is pointing. "
                 "Measured PERPENDICULAR TO THE HEADING IT WAS GIVEN, not against "
                 "world Y - the world-Y version charged a robot spawned at 0.1 rad "
                 "for 499 mm of drift it never committed, and fought 'veering' for "
                 "it.",
    "shaking": "How hard the trunk is being jolted about, measured as its own "
               "acceleration. Smooth joints can still add up to a trunk that "
               "hammers, and the trunk is where the IMU and the electronics live.",
    "rocking": "Trunk rolling and pitching. Priced on its own rather than lumped "
               "in with turning, so rocking costs whether or not the robot is "
               "also holding its heading.",
    "veering": "Turning off the heading it was sent along. Different from "
               "'wandering': a robot can hold a perfectly straight line while "
               "slowly rotating to face sideways, and this is the term that "
               "catches that. Yaw RATE is already scored - this catches the drift "
               "that a near-zero rate hides, because a tenth of a degree per step "
               "is invisible in the rate and 20 degrees off after twenty seconds.",
}


# ---------------------------------------------------------------------------
# rewards that mjlab does not have, or has only in a form that needs a sensor
# this scene does not carry
# ---------------------------------------------------------------------------


def _filtered_speed(env) -> torch.Tensor:
    """Trunk velocity, smoothed over about half a stride.

    Called from exactly ONE reward term, so the average advances once per step.
    If a second term ever needs this, cache it against a step counter first -
    otherwise the average runs at a multiple of the real rate and the effective
    filter gets shorter without anything looking wrong.
    """
    v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    buf = getattr(env, "_gray_vel_avg", None)
    if buf is None or buf.shape != v.shape:
        buf = v.clone()
    # A reset means a new episode: start the average again rather than carrying
    # the last one's speed across, which would score the first tenth of a second
    # of every episode against the previous episode's motion.
    fresh = (env.episode_length_buf <= 1).unsqueeze(1)
    buf = torch.where(fresh, v, buf + (v - buf) / FILTER_STEPS)
    env._gray_vel_avg = buf.detach()
    return buf


def track_speed(env, std: float, command_name: str = "walk"):
    """How close the smoothed trunk speed is to the commanded one."""
    command = env.command_manager.get_command(command_name)
    err = torch.sum(torch.square(command[:, :2] - _filtered_speed(env)), dim=1)
    return torch.exp(-err / (std * std))


def stepping(env, sensor_name: str = "feet", lo: float = 0.10, hi: float = 0.45,
             command_name: str = "walk"):
    """Fraction of the four feet that are mid-step, rather than planted or hopping.

    Divided by four so the term tops out at 1.0 like every other reward here. The
    mjlab original returns 0-4, which would quietly make this worth four times its
    stated weight and break the reward ceiling RULES.md rule 1 depends on.

    GATED ON BEING TOLD TO MOVE, since 3 Aug 2026. It was not, and about one
    attempt in seven is commanded to stand still - where planted feet paid 0 and
    marching on the spot paid 1.0. Every term that would have charged for the
    marching is itself gated off below MOVING, so nothing anywhere objected: the
    cheapest way to obey "stand still" was to jog on the spot.
    """
    command = env.command_manager.get_command(command_name)
    told_to_move = torch.norm(command[:, :3], dim=1) > MOVING
    air = env.scene[sensor_name].data.current_air_time
    mid = torch.sum(((air > lo) & (air < hi)).float(), dim=1) / 4.0
    return mid * told_to_move.float()


def _foot_height(env, asset_cfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return (asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
            - env.scene.env_origins[:, 2].unsqueeze(1))


def foot_clearance(env, target: float, command_name: str = "walk",
                   asset_cfg=FOOT_SITES):
    """Penalise a foot at the wrong height while it is travelling.

    Weighted by how fast the foot is moving, so a planted foot costs nothing and a
    foot skimming the floor on its way forward costs a lot. mjlab's version reads
    a TerrainHeightSensor; on flat ground the height above the environment origin
    is the same number and needs no extra sensor.
    """
    command = env.command_manager.get_command(command_name)
    moving = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > MOVING)
    asset = env.scene[asset_cfg.name]
    speed = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
    wrong = torch.abs(_foot_height(env, asset_cfg) - target)
    return torch.sum(wrong * speed, dim=1) * moving.float()


def swing_height(env, target: float, sensor_name: str = "feet",
                 command_name: str = "walk", asset_cfg=FOOT_SITES):
    """How wrong the top of each swing was, charged at the moment the foot lands.

    Clearance above scores every step of the swing; this scores the peak once. The
    two catch different faults - a foot can average the right height and still
    never actually clear the ground.
    """
    contact = env.scene[sensor_name]
    height = _foot_height(env, asset_cfg)
    peak = getattr(env, "_gray_swing_peak", None)
    if peak is None or peak.shape != height.shape:
        peak = torch.zeros_like(height)

    in_air = contact.data.found == 0
    peak = torch.where(in_air, torch.maximum(peak, height), peak)

    landed = contact.compute_first_contact(dt=env.step_dt)
    command = env.command_manager.get_command(command_name)
    moving = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > MOVING)
    cost = torch.sum(torch.square(peak / target - 1.0) * landed.float(), dim=1)

    # Reset a foot's peak once it has been charged for, and whenever the episode
    # restarts - otherwise a peak carries across a reset and is billed twice.
    fresh = (env.episode_length_buf <= 1).unsqueeze(1)
    env._gray_swing_peak = torch.where(landed | fresh,
                                       torch.zeros_like(peak), peak).detach()
    return cost * moving.float()


def touchdown_speed(env, sensor_name: str = "feet", asset_cfg=FOOT_SITES):
    """How fast a foot is still travelling downward at the instant it lands.

    The owner's observation: a leg should be slowing as it arrives, the way a
    servo eases into the end of its travel, rather than running at speed into the
    floor. This is that, priced.

    **Why speed and not force.** `hard_landing` already charges for contact FORCE
    at touchdown, and it is set to -1e-4, which is off. Turning it up would work,
    but force in the simulator is a function of the contact stiffness - a
    parameter nobody measured, on a robot whose mass just rose 53%. Optimising
    hard against a guessed parameter is how a gait comes out smooth in simulation
    and harsh on a real floor. Foot velocity is plain kinematics with no guess in
    the path, and it is the one of the two that could be checked on the real
    robot: there are no force sensors, but the potentiometers give joint rates.

    **Why only at touchdown.** Charging for deceleration in general would tax the
    fast corrections that catch a stumble - which is exactly what push_v3 did
    with a large smoothness penalty, and the robot stopped being able to catch
    itself. `landed` restricts this to the single step a foot arrives on.

    Downward only: `clamp(-vz, min=0)`. A foot moving UP as it grazes something
    is not a hard landing and should not be billed as one.

    compute_first_contact is a pure read of the air-time state, so calling it
    here as well as in swing_height advances nothing twice - unlike the running
    averages in this file, which must each have exactly one caller.
    """
    contact = env.scene[sensor_name]
    landed = contact.compute_first_contact(dt=env.step_dt)
    asset = env.scene[asset_cfg.name]
    vz = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 2]
    return torch.sum(torch.clamp(-vz, min=0.0) * landed.float(), dim=1)


def _going_straight(env, command_name: str = "walk") -> torch.Tensor:
    """True where the robot has been told to hold a line and not to turn.

    `abs`, not `>`. This read `command[:, 0] > MOVING` until 3 Aug 2026, which is
    positive-only - so the moment the command range was widened through zero to
    negative, `veering` and `wandering`, the two penalties that hold a line,
    would have switched off entirely for every backward command. Backward walking
    would have trained with no straightness penalty at all, and nothing would have
    reported it: the terms return zero, which looks exactly like passing.
    """
    command = env.command_manager.get_command(command_name)
    return ((torch.abs(command[:, 0]) > MOVING)
            & (torch.abs(command[:, 2]) < 0.05)
            & (torch.abs(command[:, 1]) < 0.05))


def commanded_height(env, std: float, command_name: str = "posture",
                     asset_cfg: SceneEntityCfg = ROBOT):
    """1.0 when the trunk is at the height it was told, falling off over `std`.

    Replaces the fixed-target `ride_height`. Same shape, same tolerance - the
    only change is that the target now comes from the command instead of from
    stance.yaml, which is what makes "crouch" something you can ask for.
    """
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    want = env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-torch.square(height - want) / (std * std))


def commanded_attitude(env, std: float, command_name: str = "posture",
                       asset_cfg: SceneEntityCfg = ROBOT):
    """1.0 when the trunk is leaning the way it was told.

    Replaces `upright`, which scored one thing: level. Level is now just the
    commanded attitude with both numbers at zero, so nothing is lost - and a
    commanded lean stops being a thing the robot is punished for.
    """
    from gray.tasks.posture_command import trunk_pitch_roll  # noqa: PLC0415

    pitch, roll = trunk_pitch_roll(env.scene[asset_cfg.name])
    want = env.command_manager.get_command(command_name)
    err = torch.square(pitch - want[:, 1]) + torch.square(roll - want[:, 2])
    return torch.exp(-err / (std * std))


def ground_covered(env, command_name: str = "walk"):
    """Net forward progress, as a fraction of the distance it was asked to cover.

    Speed tracking is scored on velocity, and velocity can be faked: a robot
    rocking forward and back at the right amplitude averages to the commanded
    speed and travels nowhere. This measures where the robot actually is compared
    to where it was, so only real progress counts.

    Clamped to [0, 1] per step. Unclamped it would be an unbounded linear reward
    and the ceiling that RULES.md rule 1 depends on would stop meaning anything.

    REWRITTEN 3 Aug 2026. It used to measure world X against the 2-D magnitude of
    the command, and that was wrong in four separate ways - all of which returned
    a number rather than an error, so nothing ever reported them:

      backward   progress is negative, the clamp takes it to 0. A backward
                 command could lose the point and never win it. Harmless while
                 the box was (0.15, 0.35); fatal now that half of it is negative.
      standing   `asked` is 0, clamped up to 1e-6, so any forward drift divides
                 to something enormous and clamps to 1.0. A robot told to stand
                 still scored FULL MARKS for wandering off.
      sideways   `asked` was the magnitude of (vx, vy) but progress was X only,
                 so a crab step was asked for and paid nothing.
      turning    world X, not the robot's own heading. A robot facing 90 degrees
                 off and walking perfectly earned zero.

    Now: progress is measured along the direction it was TOLD to go, in its own
    frame, and compared against the distance that direction implies. Same idea,
    same clamp, same ceiling - it just answers the question it always claimed to.
    """
    robot = env.scene["robot"]
    pos = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    was = getattr(env, "_gray_fwd_xy", None)
    if was is None or was.shape != pos.shape:
        was = pos.clone()
    fresh = env.episode_length_buf <= 1
    moved = torch.where(fresh.unsqueeze(-1), torch.zeros_like(pos), pos - was)
    env._gray_fwd_xy = pos.detach()

    command = env.command_manager.get_command(command_name)
    asked_speed = torch.norm(command[:, :2], dim=1)

    # The commanded direction, in the world, using the heading the robot has now.
    # command[:, 0] is along its nose and command[:, 1] is to its left, so the
    # commanded direction rotates with the robot rather than being pinned to the
    # arena - which is what makes this work while turning, and what makes a
    # negative vx count as progress instead of as failure.
    heading = robot.data.heading_w
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    want_x = command[:, 0] * cos_h - command[:, 1] * sin_h
    want_y = command[:, 0] * sin_h + command[:, 1] * cos_h
    unit = torch.stack((want_x, want_y), dim=1) / torch.clamp(
        asked_speed.unsqueeze(-1), min=1e-6)

    progress = (moved * unit).sum(dim=1)
    asked = asked_speed * env.step_dt

    # Told to stand still, there is no distance to cover and no progress to pay
    # for. Zero, not a division by an epsilon - that epsilon is what paid a
    # standing robot full marks for drifting.
    moving = asked_speed > MOVING
    ratio = progress / torch.clamp(asked, min=1e-6)
    return torch.where(moving, torch.clamp(ratio, 0.0, 1.0),
                       torch.zeros_like(ratio))


def leg_swing(env, target: float, command_name: str = "walk",
              asset_cfg=SWING_JOINTS):
    """How far the thighs and calves swing either side of their own average.

    Measured against a running average of each joint rather than its default
    pose, so this pays for movement and not for sitting at an offset. Capped at
    `target`: past that there is nothing more to earn, which is what stops it
    turning into a flail. The smoothness terms handle the other failure - a joint
    could score here by buzzing, and `twitching`, `jitter` and `joint_shock` all
    charge for that.
    """
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    avg = getattr(env, "_gray_joint_avg", None)
    if avg is None or avg.shape != q.shape:
        avg = q.clone()
    fresh = (env.episode_length_buf <= 1).unsqueeze(1)
    avg = torch.where(fresh, q, avg + (q - avg) / FILTER_STEPS)
    env._gray_joint_avg = avg.detach()

    spread = torch.mean(torch.abs(q - avg), dim=1)
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return torch.clamp(spread / target, max=1.0) * (moving > MOVING).float()


def shaking(env):
    """How hard the trunk is being jolted, as its own acceleration in m/s^2.

    Nothing else here measures this. The smoothness terms all watch the COMMANDS
    or the joints; a policy can issue perfectly smooth joint targets and still
    produce a trunk that hammers, because the jolt comes from the feet hitting
    the floor rather than from the servos. This is the thing three trunk-mounted
    IMUs would actually read.
    """
    v = env.scene["robot"].data.root_link_lin_vel_b
    was = getattr(env, "_gray_prev_vel", None)
    if was is None or was.shape != v.shape:
        was = v.clone()
    fresh = (env.episode_length_buf <= 1).unsqueeze(1)
    accel = torch.where(fresh, torch.zeros_like(v), (v - was) / env.step_dt)
    env._gray_prev_vel = v.detach()
    return torch.sum(torch.square(accel), dim=1)


def veering(env, command_name: str = "walk"):
    """How far the trunk has turned off the heading it was sent along.

    `track_turn` scores yaw RATE against the commanded rate, and a rate near zero
    looks perfect while the heading quietly walks away: a tenth of a degree per
    step is invisible in the rate and puts the robot 20 degrees off after twenty
    seconds. This scores the accumulated angle instead.

    The reference heading follows the robot while it is being told to turn, and
    locks the moment a straight command starts - so a legitimate turn is never
    charged as drift.
    """
    h = env.scene["robot"].data.heading_w
    straight = _going_straight(env, command_name)
    ref = getattr(env, "_gray_heading_ref", None)
    if ref is None or ref.shape != h.shape:
        ref = h.clone()
    hold = straight & (env.episode_length_buf > 1)
    ref = torch.where(hold, ref, h)
    env._gray_heading_ref = ref.detach()
    return torch.square(vmdp.wrap_to_pi(h - ref)) * straight.float()


def wandering(env, free: float, command_name: str = "walk"):
    """Sideways drift off the line it was told to walk.

    Scored on displacement, not on sideways velocity. A robot can have zero
    lateral velocity at every instant this reward is sampled and still be a foot
    off course, which is exactly what the previous attempt did - it walked, and
    wandered about 140 mm doing it.

    Measured PERPENDICULAR TO THE HEADING IT WAS SENT ALONG. This used to be
    |world Y - spawn Y|, and that was the drift bug. Reset nudges each spawn yaw
    by up to +/-0.1 rad, so a robot placed at +0.1 rad and walking a perfect
    straight line accumulates 5.0 * sin(0.1) = 499 mm of world-Y offset over one
    20 s episode - charged against a 50 mm allowance, at weight -1.0, every step,
    for doing exactly the right thing.

    Its only way to stop paying was to curve back toward world Y = 0, which means
    turning off its own heading, which is what `veering` charges for at up to
    -2.0. Two terms pulling opposite ways, with the random spawn yaw deciding
    which won - so a robot spawned at 0 rad had no conflict and one at +/-0.1 rad
    had the most, in opposite directions. That is the -7 to +18 degree spread
    measured across 64 robots given the identical command, and verify.py was
    scoring in the robot's own start-heading frame the whole time. Trained
    against one line, scored against another.

    The line is locked the moment a straight command begins: where the robot was
    AND which way it pointed. A heading with no origin does not define a line.
    """
    robot = env.scene["robot"]
    heading = robot.data.heading_w
    pos = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    straight = _going_straight(env, command_name)

    ref_pos = getattr(env, "_gray_line_pos", None)
    ref_head = getattr(env, "_gray_line_head", None)
    if ref_pos is None or ref_pos.shape != pos.shape:
        ref_pos = pos.clone()
    if ref_head is None or ref_head.shape != heading.shape:
        ref_head = heading.clone()
    hold = straight & (env.episode_length_buf > 1)
    ref_pos = torch.where(hold.unsqueeze(-1), ref_pos, pos)
    ref_head = torch.where(hold, ref_head, heading)
    env._gray_line_pos = ref_pos.detach()
    env._gray_line_head = ref_head.detach()

    # Cross-track distance, the same rotation verify.py scores drift with.
    moved = pos - ref_pos
    off = torch.abs(-moved[:, 0] * torch.sin(ref_head)
                    + moved[:, 1] * torch.cos(ref_head))
    return torch.clamp(off - free, min=0.0) * straight.float()


# ---------------------------------------------------------------------------


def walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = push_env_cfg(play=play)

    # A first walk has enough to deal with. The randomised ground, mass and servo
    # gains stay; only the shoving stops.
    cfg.events.pop("shove", None)

    cfg.commands = {
        "walk": StraightLineVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(5.0, 10.0),
            # About one attempt in seven commands a full stop. Without them the
            # policy never learns that zero means stand, and a robot that cannot
            # stop is not much use.
            rel_standing_envs=0.15,
            # Half get a straight line and nothing else - sideways and turn both
            # zero, forward speed as drawn INCLUDING backward. `wandering` and
            # `veering`, the two terms that hold a line, only apply here, so
            # without a decent share of these there is nothing to train
            # straightness on: a uniform draw lands inside their +/-0.05 gates
            # about once in eighty.
            #
            # It was 0.8 on mjlab's own rel_forward_envs until 3 Aug 2026, which
            # forced the speed positive and up to at least 0.3 - see
            # gray/tasks/walk_command.py. Half, not four fifths, because the
            # other half now has a much bigger box to cover.
            rel_forward_envs=0.0,
            rel_straight_envs=0.5,
            straight_min_speed=0.10,
            ranges=vmdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=WALK_SPEED, lin_vel_y=WALK_SIDE, ang_vel_z=WALK_TURN),
        ),
        # Where the trunk should BE, as opposed to where it should GO. Three more
        # numbers, on the same clock as the velocity command so one draw is one
        # coherent instruction rather than two that change at different moments.
        "posture": PostureCommandCfg(
            entity_name="robot",
            resampling_time_range=(5.0, 10.0),
            nominal_height=_stance()[1],
            rel_nominal_envs=0.25,
            ranges=PostureCommandCfg.Ranges(
                height=POSE_HEIGHT, pitch=POSE_PITCH, roll=POSE_ROLL),
        ),
    }
    # The policy has to be told where it is being sent, or the command is noise.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["command"] = ObservationTermCfg(
            func=vmdp.generated_commands, params={"command_name": "walk"})
        # And where the trunk should be. Three more numbers, so the observation
        # goes 45 -> 48 and every policy trained before today is unreadable by
        # this task - a saved file is a fixed-size mapping, and the size changed.
        cfg.observations[group].terms["posture"] = ObservationTermCfg(
            func=vmdp.generated_commands, params={"command_name": "posture"})

    # Terms that were the whole point of standing still and are now in the way.
    #   still        - it is being asked to move
    #   joint_speed  - same; legged_gym disables this one for walking too
    #   foot_lift    - 'stepping' and 'swing_height' are the gait versions of this
    #   spinning     - it fines ALL trunk rotation, including the yaw the robot has
    #                  just been ordered to produce. track_turn already pays for the
    #                  commanded yaw and fines roll and pitch, so keeping both means
    #                  paying and fining the same motion at once.
    #   tilt         - replaced by 'upright', the same measurement scored the way
    #                  the walking references score it
    #   posture      - replaced below by the speed-dependent version
    for gone in ("still", "joint_speed", "foot_lift", "spinning", "tilt", "posture"):
        cfg.rewards.pop(gone, None)

    # --- the task itself ---
    cfg.rewards["track_speed"] = RewardTermCfg(
        func=track_speed, weight=2.0, params={"std": TRACK_STD})
    cfg.rewards["track_turn"] = RewardTermCfg(
        func=vmdp.track_angular_velocity, weight=1.0,
        params={"std": TURN_STD, "command_name": "walk"})
    # Ground actually covered, which velocity tracking alone does not guarantee.
    cfg.rewards["ground_covered"] = RewardTermCfg(func=ground_covered, weight=1.0)
    # And covered in a straight line. Two different failures: 'wandering' is
    # ending up off the line, 'veering' is turning off it. A robot can do either
    # without the other - slide sideways while pointing straight ahead, or hold a
    # perfect line while slowly rotating to face the wrong way.
    cfg.rewards["wandering"] = RewardTermCfg(
        func=wandering, weight=-1.0, params={"free": DRIFT_FREE_M})
    cfg.rewards["veering"] = RewardTermCfg(func=veering, weight=-0.2)

    # --- staying up while doing it ---
    # asset_cfg must name the trunk explicitly. mjlab's default resolves to every
    # body in the robot, and `upright` then hands 13 quaternions per robot to a
    # function expecting one - it fails with a shape mismatch that names neither
    # this term nor the reason.
    # Both of these now score against the POSTURE COMMAND rather than against a
    # fixed pose. `upright` scored one attitude - level - so a commanded lean was
    # a thing the robot got punished for. `ride_height` scored one height, so a
    # crouch could not be asked for at all. Level and ride height are still there;
    # they are the command at zero.
    cfg.rewards["upright"] = RewardTermCfg(
        func=commanded_attitude, weight=1.0,
        params={"std": 0.45, "command_name": "posture", "asset_cfg": ROBOT})
    cfg.rewards.pop("height", None)
    cfg.rewards["ride_height"] = RewardTermCfg(
        func=commanded_height, weight=1.0,
        params={"std": 0.05, "command_name": "posture", "asset_cfg": ROBOT})

    # Tolerance that widens once the robot is asked to move. The thresholds are
    # the whole point: mjlab's 0.5 / 1.5 are Go1 speeds, and Gray's total command
    # tops out near 0.85, so on the defaults it would sit in the tight standing
    # band for the entire run while being asked to walk.
    cfg.rewards["posture"] = RewardTermCfg(
        func=vmdp.variable_posture, weight=1.0,
        params={
            "asset_cfg": ALL_JOINTS,
            "command_name": "walk",
            "walking_threshold": MOVING,
            "running_threshold": 0.60,
            "std_standing": {".*hip": 0.10, ".*thigh": 0.10, ".*calf": 0.15},
            "std_walking": {".*hip": 0.25, ".*thigh": 0.35, ".*calf": 0.45},
            "std_running": {".*hip": 0.25, ".*thigh": 0.35, ".*calf": 0.45},
        })

    # --- what makes it a gait rather than a shuffle ---
    cfg.rewards["stepping"] = RewardTermCfg(func=stepping, weight=1.0)
    # The one that decides whether this looks like a dog. Everything else in this
    # file is happy with a robot creeping along on stiff legs; this is the only
    # term that pays for the thighs and calves actually swinging through a stride.
    cfg.rewards["leg_swing"] = RewardTermCfg(
        func=leg_swing, weight=1.0, params={"target": SWING_SPREAD})
    cfg.rewards["dragging"] = RewardTermCfg(
        func=foot_clearance, weight=-2.0, params={"target": SWING_TARGET})
    cfg.rewards["swing_height"] = RewardTermCfg(
        func=swing_height, weight=-0.25, params={"target": SWING_TARGET})
    cfg.rewards["hard_landing"] = RewardTermCfg(
        func=vmdp.soft_landing, weight=-1e-4,
        params={"sensor_name": "feet", "command_name": "walk",
                "command_threshold": MOVING})

    # Sliding still costs. docs/REWARDS.md puts the reference at -0.1, but the
    # owner's complaint about the push runs was specifically skidding, and the
    # simulator's friction is a guess a policy should not be allowed to lean on.
    # -0.5 is five times the reference and half what the push task used, now that
    # clearance and swing height are also pushing toward picking feet up.
    cfg.rewards["skidding"].weight = -0.5

    # --- how steady the trunk is ---
    # `upright` above scores the trunk's ANGLE off level. These two score how
    # violently it is getting there. A robot can average perfectly level while
    # rocking and hammering the whole way, and nothing else in this file notices.
    # Both start near zero and ramp - see the curriculum below. A robot that has
    # not yet learned to stand rocks hard, and charged at full weight from step 0
    # these two are worth more per step than staying alive is. legged_gym clips
    # the summed reward at zero for exactly this reason: under a net-negative
    # score, ending the episode early beats carrying on.
    cfg.rewards["rocking"] = RewardTermCfg(
        func=vmdp.body_angular_velocity_penalty, weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))})
    cfg.rewards["shaking"] = RewardTermCfg(func=shaking, weight=-0.001)

    # --- protecting the hardware ---
    cfg.rewards["joint_shock"] = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # Arriving slowly. Ramped in with the other tidiness terms rather than set
    # from step 0 - a robot that cannot yet walk lands hard on every step, and
    # charging full price for that before it has a gait just teaches it not to
    # pick its feet up.
    cfg.rewards["landing_speed"] = RewardTermCfg(func=touchdown_speed, weight=-0.1)

    # Everything that asks the robot to be TIDY is ramped in, rather than set at
    # full weight from step 0. docs/REWARDS.md is emphatic about this and so is
    # our own history: push_v3 set one large smoothness penalty from the start and
    # the robot lost the ability to catch itself. RMA reports the same failure and
    # the same fix. First learn to walk, then learn to walk neatly.
    #
    # common_step_counter counts env steps, and at 24 per iteration these land at
    # roughly iteration 100, 250 and 500.
    def _ramp(name, weights):
        return CurriculumTermCfg(
            func=vmdp.reward_curriculum,
            params={"reward_name": name,
                    "stages": [{"step": s, "weight": w} for s, w in
                               zip((0, 2400, 6000, 12000), weights)]})

    cfg.curriculum = {
        # commanded angles jumping about
        "ease_in_smoothness": _ramp("twitching", (-0.01, -0.025, -0.05, -0.05)),
        # trunk rolling and pitching
        "ease_in_steadiness": _ramp("rocking", (-0.02, -0.05, -0.12, -0.2)),
        # trunk being hammered by its own footfalls
        "ease_in_shake": _ramp("shaking", (-0.001, -0.003, -0.006, -0.01)),
        # heading drifting off the line. Last and slowest: it is meaningless until
        # the robot can actually hold a direction, and charging for it early just
        # fines a robot for falling over in a way that happens to rotate it.
        "ease_in_straightness": _ramp("veering", (-0.2, -0.5, -1.2, -2.0)),
        # arriving slowly rather than running into the floor
        "ease_in_landing": _ramp("landing_speed", (-0.1, -0.3, -0.6, -0.8)),
    }

    if not play:
        cfg.episode_length_s = 20.0
    return cfg


def walk_ppo_cfg():
    cfg = push_ppo_cfg()
    cfg.experiment_name = "gray_walk"
    cfg.max_iterations = 3000
    return cfg
