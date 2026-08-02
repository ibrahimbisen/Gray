"""Phase 3 environment: learn a residual on Gray's classical crawl gait.

Built to the spec already written in docs/PROJECT_NOTES.md. The three decisions worth
restating, because each is driven by hardware rather than by RL convention:

CONTROL RATE IS 50 Hz AND NOT NEGOTIABLE. The PCA9685 drives the servos with a 20 ms
PWM period; nothing faster can reach the robot. Timestep 0.005 s x decimation 4 = 50 Hz
exactly, which also lands every gait phase on a lookup-table grid point.

THE POLICY NEVER SEES A MEASURED JOINT ANGLE. DS3218MG servos take a position command
and report nothing back. A policy trained on measured joint positions would learn to
rely on a signal that does not exist on the robot, and would fail the moment it was
deployed. The actor therefore observes only what the IMU can supply, plus quantities it
computes itself (its own last action, the gait phase, the commanded velocity, and the
gait's current targets - a command, not a measurement). The critic is allowed the true
joint state as privileged information: it is discarded after training and never shipped.

REWARDS ARE SHAPED AROUND BRITTLE PARTS. Gray's legs are SLA resin, which cracks under
repeated impact, and walking is nothing but repeated impact. Contact impulse and joint
acceleration carry heavier penalties than a standard velocity task would use, so a soft
gait is learned from the start rather than retrofitted after parts break.

REWARDS MAY READ ANYTHING THE SIMULATOR KNOWS. The no-feedback rule above constrains
OBSERVATIONS only, because only observations ship to the robot. True joint angles, foot
positions and contact forces are all fair game in a reward and several of the terms
below depend on exactly that. Do not let the hardware constraint talk you out of a
reward it does not apply to.

READ THE FREEZE ARITHMETIC ABOVE THE REWARD DICT BEFORE CHANGING ANY WEIGHT. It is the
one piece of this file that has already been paid for in a lost training run.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from train.gray_robot import FOOT_GEOM_REGEX, get_gray_robot_cfg
from train.residual_action import (
  ResidualGaitActionCfg,
  gait_nominal,
  gait_phase,
)
from train.rewards import (
  BaseHeightFloor,
  CrossTrackDrift,
  ExcessTouchdowns,
  FallPenalty,
  ForwardProgress,
  PitchRegulation,
  PreClipActionMagnitude,
  ServoTrackingError,
  StanceFootAnchoring,
  SupportCount,
  SwingClearance,
  TrackFilteredAngularVelocity,
  TrackFilteredLinearVelocity,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor


##
# Speed envelope.
#
# The Phase 2 crawl walks 52.9 mm/s at stride scale 1.0, and above ~2 strides/s the
# feet skid and the robot goes BACKWARD. Commanding metres per second at Go1 scale
# (+/-1 m/s) would ask for roughly twenty times what this robot can physically do, and
# the tracking reward would saturate into a constant. These are Gray's real limits.
##
MAX_FORWARD_MS = 0.080
MAX_BACKWARD_MS = -0.040

# Tracking is scored against a stride-averaged velocity, not the instantaneous one -
# see train/rewards.py for the measurements that forced that.
#
# TRACK_LIN_STD was 0.035 in the first training run and that was too tight: once the
# policy drifted off speed the exponential went flat, leaving only rewards that are
# maximised by standing still, and it duly learned to stand still (675 mm -> 65 mm).
# Widened so the term still has slope across the whole plausible error range, and
# paired with the linear ForwardProgress term which has slope everywhere.
TRACK_LIN_STD = 0.070
# WAS 0.050, AND THAT WAS TIGHTER THAN THE NOISE FLOOR. 0.200 rad/s (inherited from
# Go1) really is far too loose for a robot that walks at 56 mm/s, and 0.050 was the
# overcorrection. The signal this term scores is already filtered at tau = one gait
# period, and even filtered the yaw rate of a steady crawl has sd 0.080 rad/s (raw
# 0.532). So at std 0.050 a PERFECTLY STRAIGHT walk scores exp(-0.080^2/0.050^2) =
# 0.08, and an excellent one reaches only ~0.28: the exponential is flat across the
# entire range the robot can actually occupy, and a flat exponential has no gradient.
# That is Failure #2 - the stride ripple swamping an instantaneous signal - recurring
# in the yaw channel instead of the forward one.
#
# At 0.150 the same straight walk scores 0.75, which is on the shoulder of the
# exponential where its slope is largest. Straightness is now paid for properly by
# CROSS_TRACK_STD below, which is a POSITION error and cannot be swamped by ripple at
# all; this term drops to being a damper on gross yaw, and its weight halves to match.
TRACK_ANG_STD = 0.150
TRACK_TAU_S = 0.6

# Perpendicular distance from the straight line the robot was commanded to walk along.
# Immune to the stride ripple BY CONSTRUCTION rather than by tuning: position is the
# integral of velocity, and a zero-mean ripple integrates away to nothing. No filter,
# no tau, nothing to check against the ripple.
#
# WAS 0.03, AND THAT WAS THE TOO-TIGHT-STD FAILURE FOR THE THIRD TIME. First it was
# TRACK_LIN_STD at 0.035, then TRACK_ANG_STD at 0.050; here it was 0.03 m, and a
# walking policy scored +0.035 out of a possible +0.50 - seven percent of the term's
# ceiling, with no slope anywhere the robot can actually be.
#
# THE DERIVATION, and it is a derivation rather than a preference.
#
# What this term samples is not the drift at the END of a run - it is evaluated every
# step, on a cross-track distance that starts at zero and grows. So the quantity that
# has to sit on the slope is the PER-STEP MEAN |cross-track| over an episode, and
# that is now measured directly rather than inferred from an end-of-run figure:
#
#     classical crawl, per-step mean |cross-track| over 10 s, 1024 robots
#       nominal robot (s = 0)              37.4 mm
#       start-of-training fleet (s = 0.3)  68.9 mm
#       full envelope (s = 1.0)           118.6 mm
#
# (For reference, the end-of-run drift these replace: -52.6 mm mean, sd 162.1 over
# 6 seeds. The reason that number produced a bad std is that it is roughly twice the
# per-step figure - drift accumulates over the episode - and the term is charged on
# the per-step one.)
#
# exp(-d^2/std^2) has its steepest slope at d = std/sqrt(2). Putting that point on the
# envelope training actually STARTS at, 68.9 mm, gives std = sqrt(2) x 0.0689 =
# 0.097 m. Rounded to 0.10.
#
# THE CHECK THE OTHER TWO NEVER GOT - what the term is worth across the range the
# policy occupies, at std = 0.10:
#
#      0 mm  1.00      the target
#     37 mm  0.87      classical gait on a nominal robot
#     69 mm  0.62      classical gait at the start of the ramp   <- max slope
#    100 mm  0.37
#    119 mm  0.24      classical gait under the full envelope
#    200 mm  0.02      a robot that has wandered off
#
# Nearly the whole 0-1 range is in use and the derivative is largest exactly where the
# untrained policy sits. Compare std = 0.03, where the same six rows read 1.00 / 0.22 /
# 0.005 / 1e-5 / 1e-7 / 0: everything past 40 mm was indistinguishable from everything
# else, which is a flat exponential, which has no gradient.
#
# MEASURED WALKING vs FROZEN at the start of training is in THE FREEZE ARITHMETIC in
# train/rewards.py - this term is the one that fails the freeze test on its own and is
# capped by achieved progress to contain it.
CROSS_TRACK_STD = 0.10

# Load threshold, in newtons, for calling a foot "in stance".
#
# NOT "contact > 0", and NOT the commanded gait phase. The measured contact channel
# CHATTERS: the gait commands 80 footfalls in 12 s and the robot makes 262, i.e. 3.3x
# the stride rate. A boolean gate would read every one of those bounces as a stance
# foot, so a stance-slip penalty gated that way would spend most of its budget scoring
# bounce rather than skid, and a support-count reward gated that way would pay for
# chatter. 1 N is ~6% of the robot's 16 N weight: far above a graze, far below the
# ~4 N a genuinely loaded foot of a 1.625 kg quadruped on three feet carries.
STANCE_FORCE_N = 1.0

# Swing clearance target, ground to the underside of the foot, in metres.
#
# The gait COMMANDS 34 mm of lift (GaitParams.step_height is 0.035; the Bezier peaks
# just under its control point). Measured: the front feet reach 21.6 mm and the BACK
# FEET 0.8 mm - the back feet are dragging, not swinging at all. 25 mm is
# deliberately BELOW the command, for a reason that matters: the Phase 2 Bezier is
# near-vertical at lift-off and touchdown (chosen for low touchdown speed on brittle
# resin), so a swinging foot is legitimately close to the ground at both ends of its
# arc. Only the middle of the swing should be held to a clearance, and only to a
# height the leg can actually reach.
SWING_CLEARANCE_M = 0.025

# The foot collision geom is a sphere of radius 12 mm (see sim/models/gray.xml), so
# its centre sits at z = 0.012 with the foot resting on flat ground. Gray has no foot
# sites and no terrain height sensor, so foot clearance has to be computed from the
# geom position minus this radius.
FOOT_RADIUS_M = 0.012

# One-sided floor on trunk height, in metres - penalised only from BELOW.
#
# Measured over the classical crawl in plain MuJoCo the trunk sits at z mean 0.1661,
# min 0.1600 (GaitParams.stance_height is 0.16). The floor is set 10 mm below the
# observed MINIMUM, not at the mean, because it is not a ride-height regulator: it
# exists solely to block the crouch-hack that stance_foot_anchoring and
# servo_tracking_error otherwise open up - both of those are cheapest with the legs
# folded and the belly on the floor.
#
# At weight -2000 (a plain quadratic on the shortfall, units m^2) a 10 mm violation
# costs 0.20/step and a 25 mm crouch costs 1.25/step, which exceeds the entire penalty
# budget below. Ordinary servo sag under the widened kp randomisation stays inside the
# 10 mm of headroom and costs exactly nothing.
BASE_HEIGHT_FLOOR_M = 0.150

# Resin density +/-40%, as e^(2*alpha) per dr.pseudo_inertia. Covers the open question
# of whether the SLA parts were printed solid (2.00 kg) or hollowed (1.625 kg).
DENSITY_ALPHA = (math.log(0.6) / 2.0, math.log(1.4) / 2.0)

FOOT_CONTACT_SENSOR = "feet_ground_contact"


##
# THE RANDOMISATION ENVELOPE, AND THE RAMP THAT LETS THE GAIT SURVIVE IT.
#
# =================================================================================
# THE MEASUREMENT THAT FORCED THIS, AND IT WAS MEASURED, NOT GUESSED. Classical
# crawl, zero residual, 1 s settle + 10 s measured, 1024 robots per point, commanded
# 52.9 mm/s, everything else exactly as training runs it (pushes on). The envelope
# below scaled by a single factor s:
#
#      s     mean mm    median mm    % of s=0    tipped in first 1 s
#     0.00     425.6      436.3        100%          3.5%
#     0.10     429.3      437.8        101%          4.1%
#     0.20     355.2      353.4         84%          5.7%
#     0.25     316.4      304.8         74%          6.4%
#     0.30     286.2      270.6         67%          7.4%
#     0.40     227.1      180.3         53%          8.5%
#     0.50     180.5      125.1         42%          8.3%
#     0.75      96.1        2.9         23%         11.4%
#     1.00      47.3       -0.2         11%         13.0%
#
# THERE IS NO CLIFF, which is exactly why testing the events one at a time found no
# culprit: the losses compound. The envelope is flat to s = 0.1 and then costs about
# 1.4% of the gait's distance per 0.01 of s, all the way down.
#
# WATCH THE MEDIAN, NOT THE MEAN. Up to s = 0.3 the two track each other, so the
# TYPICAL robot walks. From s = 0.4 the median falls away from the mean (180 vs 227,
# then 125 vs 181, then 3 vs 96), which is the fleet splitting into robots that still
# walk and robots that have stopped - and the mean is then reporting a population that
# no longer exists. Under the full envelope the median robot travels -0.2 mm: it does
# not walk at all.
#
# s = 0.3 IS THE STARTING POINT. It keeps 67% of the gait's distance and a median
# robot moving at 27 mm/s, it is the last scale at which the fleet is still unimodal,
# and it matches the "~0.3x" the research specified. Anything past 0.4 hands the
# policy a starting point that does not walk, which is the entire failure this ramp
# exists to prevent.
# =================================================================================
#
# Each entry is (VALUE WITH NO RANDOMISATION, FULL RANGE). At scale s each bound is
# interpolated linearly from the first toward the second, so s = 0 is a fleet of
# identical nominal robots and s = 1 is the envelope as designed. Linear rather than
# log interpolation, which is the plain reading of "scale the envelope" and keeps the
# nominal value inside the range at every s; for the multiplicative entries a
# geometric interpolation would also be defensible and gives a slightly narrower band
# at intermediate s.
#
# THE NOMINAL VALUES ARE NOT FREE PARAMETERS. 1.0 for anything applied with
# operation="scale", 0.0 for anything additive (COM shift, servo zero offset, and the
# pseudo-inertia alpha, which enters as e^(2*alpha)), and 0.8 for foot friction
# because that is the absolute value train/gray_robot.py compiles into the MJCF. Get
# one of these wrong and s = 0 is not "no randomisation", it is a silent constant
# offset applied to every robot in the fleet.
DR_ENVELOPE: dict[str, dict[str, tuple[float, tuple[float, float]]]] = {
  "resin_density_legs": {"alpha_range": (0.0, DENSITY_ALPHA)},
  "trunk_inertia": {
    "alpha_range": (0.0, DENSITY_ALPHA),
    "t_range": (0.0, (-0.02, 0.02)),
  },
  "foot_friction": {"ranges": (0.8, (0.4, 1.2))},
  "servo_armature": {"ranges": (1.0, (0.35, 3.0))},
  "knee_pushrod_inertia": {"ranges": (1.0, (0.224, 4.69))},
  "servo_gains": {"kp_range": (1.0, (0.5, 4.0)), "kd_range": (1.0, (0.5, 4.0))},
  "servo_zero_offset": {"bias_range": (0.0, (-0.03, 0.03))},
}

# Where the ramp starts, and when it reaches full width.
#
# START comes from the table above: 0.3 is the largest scale at which the fleet is
# still unimodal - where the MEDIAN robot walks, not just the mean - i.e. the largest
# envelope that still leaves something for the residual to be a residual OF. It is not
# a taste judgement and it should be re-derived, not nudged, if the gait or the servo
# model changes; the throwaway that produced the table is a loop over
# `scaled_dr_params` and a distance measurement.
#
# RAMP_END_STEPS is one third of training. train/tasks.py runs max_iterations 3000 x
# num_steps_per_env 24 = 72000 environment steps, and `env.common_step_counter` counts
# exactly those. So the envelope reaches full width at step 24000 and THE FINAL TWO
# THIRDS OF TRAINING RUN AT FULL RANDOMISATION, which is the part that is not
# negotiable: a policy polished at reduced randomisation looks brilliant in sim and
# falls over on the floor. There is deliberately no anneal back down, and nothing here
# ever lowers the scale - it is a max() against the previous value away from being
# monotone, and it is monotone by construction because it reads only the step counter.
DR_SCALE_START = 0.3
DR_RAMP_END_STEPS = 24_000


def scaled_dr_params(scale: float) -> dict[str, dict[str, tuple[float, float]]]:
  """The randomisation ranges at envelope scale ``scale``, keyed by event term name.

  scale = 0 collapses every range to its nominal value (a fleet of identical nominal
  robots); scale = 1 returns the ranges exactly as `DR_ENVELOPE` declares them.
  """
  out: dict[str, dict[str, tuple[float, float]]] = {}
  for term_name, params in DR_ENVELOPE.items():
    out[term_name] = {
      param: (
        nominal + scale * (full[0] - nominal),
        nominal + scale * (full[1] - nominal),
      )
      for param, (nominal, full) in params.items()
    }
  return out


def randomisation_ramp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  start: float = DR_SCALE_START,
  full_at_step: int = DR_RAMP_END_STEPS,
) -> dict[str, torch.Tensor]:
  """Widen the domain-randomisation envelope from ``start`` to 1.0, then leave it.

  WHY A CURRICULUM AND NOT A NARROWER ENVELOPE. Measured, the full envelope destroys
  96% of the classical gait before a single gradient step is taken, and the whole
  premise of residual RL is that training starts from something that already walks. But
  the envelope is also what makes the policy work on the real robot, and
  docs/PROJECT_NOTES.md sets the bar at "beat Phase 2 UNDER FULL RANDOMISATION". Both
  are true at once, so the envelope has to move: the policy learns to walk first, then
  learns to walk on any robot.

  WHY THE DR EVENTS ARE NOW "reset" AND NOT "startup". A startup event fires exactly
  once, in `ManagerBasedRlEnv.__init__`, before any curriculum term has ever been
  computed (manager_based_rl_env.py: `event_manager.apply(mode="startup")` at the end
  of `__init__`; `curriculum_manager.compute` only ever runs inside `_reset_idx`).
  Mutating a startup term's params afterwards therefore does nothing at all - it would
  be a ramp that does not ramp. In "reset" mode the same events are re-drawn per
  episode and the curriculum lands. `_reset_idx` computes the curriculum BEFORE it
  applies reset events, so the ranges written here are the ranges the next draw uses.

  This is safe to do per episode because every `dr.*` function used here samples
  against the model's DEFAULTS rather than its current values (`_select_default_values`
  in mjlab/envs/mdp/dr/_core.py, and the `operation="scale"` path in `pd_gains`), so
  repeated application does not compound. The one real cost is that `pseudo_inertia`
  and `joint_armature` trigger a `recompute_constants` on every reset; mjlab's own
  docs name "startup or reset" as the modes to use for exactly these terms.

  It also changes the character of the randomisation slightly, for the better: each
  environment is now a fresh robot every episode rather than one fixed robot for the
  whole run, so the policy meets far more of the envelope than 16384 fixed draws.
  """
  del env_ids  # this term is global; it edits configs, not per-env state.
  progress = min(1.0, env.common_step_counter / max(1, full_at_step))
  scale = start + (1.0 - start) * progress
  for term_name, params in scaled_dr_params(scale).items():
    env.event_manager.get_term_cfg(term_name).params.update(params)
  return {"scale": torch.tensor(scale)}


##
# Training-time metrics. These are diagnostics, not rewards - no weight, no dt scaling,
# nothing reaches the policy through them. They exist because both of the failures this
# project has already paid for were invisible in the logs until after the fact.
##


def clip_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Fraction of raw policy outputs that the residual clip is throwing away.

  THE SINGLE MOST DIAGNOSTIC NUMBER FOR THIS ARCHITECTURE, and until now it was not
  logged at all. ResidualGaitAction computes clamp(action * 0.2, -0.2, +0.2), so the
  clip binds exactly when |action| > 1.0 and everything past that is discarded.

  Reconstructed after the fact, the last run went 1.7% at round 500 -> 20.2% at 950 ->
  27.9% at 1100 while Policy/mean_std climbed 0.101 -> 0.313 and never turned over.
  That is a runaway: a clipped sample produces no gradient, so nothing pushes the
  distribution back in, while the entropy bonus keeps widening it. Nobody saw it
  because the only visible symptom was action_rate, and action_rate was by then
  measuring its own sampling noise (-0.104 implies 0.415 of step-to-step change per
  joint; pure random sampling at that std gives 0.44).

  Watch it against these thresholds: under ~5% is healthy, over ~15% means the policy
  is spending its output range on samples that never reach the robot. The fixes are in
  train/tasks.py (entropy_coef) and the pre-clip action magnitude penalty below.
  """
  return (env.action_manager.action.abs() > 1.0).float().mean(dim=-1)


def stance_slip_speed(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = STANCE_FORCE_N,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Mean horizontal speed of the feet that are actually carrying load, in m/s.

  The baseline to read this against is 68 mm/s - the measured skid of the classical
  crawl. It is logged because stance_foot_anchoring has an obvious failure mode: the
  policy can cut its slip penalty either by GRIPPING BETTER (what we want) or by
  SHORTENING ITS STRIDE (cheating, and it also cuts forward_progress, but a policy
  will happily take a bad trade if the penalty is over-weighted). Slip falling while
  travel also falls is the cheat; slip falling with travel held is the win. Read this
  next to Episode_Reward/forward_progress, never on its own.

  Gated on measured force, not on the contact boolean, for the reason given at
  STANCE_FORCE_N. Uses the foot geoms because Gray has no foot sites; this assumes the
  contact sensor's slot order matches the resolved geom order, which is the same
  assumption mjlab's own ``feet_slip`` makes.
  """
  asset = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force = sensor.data.force                                          # [B, F, 3]
  loaded = (torch.norm(force, dim=-1) > force_threshold).float()      # [B, F]
  foot_vel = asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2]     # [B, F, 2]
  speed = torch.norm(foot_vel, dim=-1)                                # [B, F]
  return (speed * loaded).sum(dim=1) / loaded.sum(dim=1).clamp(min=1.0)


def gray_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Gray walking on flat ground, tracking a commanded forward velocity."""

  ##
  # Sensors. No terrain scanning and no foot sites: the robot is flat-ground only for
  # now and has no cameras, so the feet are tracked by their collision geoms.
  ##

  feet_contact = ContactSensorCfg(
    name=FOOT_CONTACT_SENSOR,
    primary=ContactMatch(mode="geom", pattern=(FOOT_GEOM_REGEX,), entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  ##
  # Observations.
  ##

  # What the real robot can actually produce. Note the absence of joint_pos/joint_vel.
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "gait_phase": ObservationTermCfg(func=gait_phase),
    "gait_targets": ObservationTermCfg(func=gait_nominal),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  # Privileged. Used only to fit the value function during training, then thrown away,
  # so relying on unmeasurable quantities here is safe.
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact, params={"sensor_name": FOOT_CONTACT_SENSOR}
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces, params={"sensor_name": FOOT_CONTACT_SENSOR}
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Actions: a bounded correction on the classical gait. See train/residual_action.py.
  ##

  actions: dict[str, ActionTermCfg] = {
    "residual": ResidualGaitActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      max_residual=0.2,
      gait_pattern="crawl",
    )
  }

  ##
  # Commands. Lateral and yaw are commanded to zero rather than omitted: the classical
  # gait cannot strafe or turn, so asking for either would be asking the residual to
  # invent a gait. Commanding zero yaw instead makes the tracking reward pay for
  # walking STRAIGHT, which is exactly the few-mm CAD asymmetry that Phase 2 could not
  # tune out and that docs/PROJECT_NOTES.md expects the RL layer to absorb.
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 8.0),
      rel_standing_envs=0.1,
      heading_command=False,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(MAX_BACKWARD_MS, MAX_FORWARD_MS),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
      ),
    )
  }

  ##
  # Events - the domain randomisation that makes sim-to-real possible.
  ##

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    # NARROWED FROM +/-0.05 rad TO +/-0.02, AND IT WAS DOING MORE DAMAGE THAN THE
    # ENTIRE DOMAIN-RANDOMISATION ENVELOPE. Measured, classical gait, 1 s settle +
    # 10 s, 512 robots, NO domain randomisation at all:
    #
    #     +/-0.050 rad   37.5% past 60 deg of tilt within 1 s,  277 mm walked
    #     +/-0.030 rad   18.4%                                   356 mm
    #     +/-0.020 rad    3.3%                                   424 mm
    #     +/-0.010 rad    0.0%                                   443 mm
    #     +/-0.000 rad    0.0%                                   443 mm
    #
    # +/-0.05 rad is 2.9 degrees per joint. It has no business tipping a quadruped over,
    # and on its own it does not - THE ROOT CAUSE IS A SPAWN TRANSIENT THAT THIS FILE
    # CANNOT FIX. mjlab spawns Gray in `GaitGenerator.stand()` (INIT_STATE in
    # train/gray_robot.py) but `ResidualGaitAction.reset` randomises the gait phase, so
    # the very first control step commands every joint to jump 0.2-0.5 rad at once.
    # Twelve servos saturate together, the trunk is thrown ~330 mm BACKWARD and rises
    # from 175 mm to 250 mm, and the robot then has to land. The landing is marginal:
    # with no per-joint offset every environment is near-identical and all of them
    # survive it, and any asymmetry decides it. The joint offset is not the disease, it
    # is what made the disease visible.
    #
    # THE REAL FIX is to spawn in the pose the gait commands at the drawn phase, which
    # means INIT_STATE in train/gray_robot.py or the phase draw in
    # train/residual_action.py - neither of which this change owns. Until then +/-0.02
    # rad is the largest offset that keeps 96% of the gait's distance at a single-digit
    # tip rate. It is not redundant at that size - it is roughly a servo's own
    # repeatability - and the gait phase is already drawn uniformly over the whole
    # cycle, which is far more initial-state diversity than this term ever supplied.
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.02, 0.02),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Legs: density only. Physics-consistent - mass AND inertia scale together, which
    # dr.body_mass would not do.
    "resin_density_legs": EventTermCfg(
      mode="reset",
      func=dr.pseudo_inertia,
      params={
        "alpha_range": DENSITY_ALPHA,
        "asset_cfg": SceneEntityCfg("robot", body_names=("^(fl|fr|br|bl)_.*",)),
      },
    ),
    # Trunk: density plus a COM shift, in one call. Two pseudo_inertia events on the
    # same body would fight over the same fields. The trunk carries the battery, Pi
    # and electronics - 447 g that robot.yaml models as a lump - so where its COM
    # actually sits is genuinely uncertain until the robot is reassembled.
    "trunk_inertia": EventTermCfg(
      mode="reset",
      func=dr.pseudo_inertia,
      params={
        "alpha_range": DENSITY_ALPHA,
        "t_range": (-0.02, 0.02),
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
    # PER-LEG, not shared. shared_random=True drew ONE friction coefficient and gave
    # it to all four feet, which is the one case that never happens on a real floor:
    # the interesting failure is a robot with three grippy feet and one slippery one,
    # because that is what produces the drift the RL layer is meant to absorb, and a
    # shared draw cannot generate it at all. It also interacts with the per-leg CAD
    # asymmetry that Phase 2 could not tune out.
    "foot_friction": EventTermCfg(
      mode="reset",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(FOOT_GEOM_REGEX,)),
        "operation": "abs",
        "ranges": (0.4, 1.2),
        "shared_random": False,
      },
    ),
    # Armature is an ESTIMATE (0.003 kg.m^2, from a ~6e-8 rotor behind ~245:1) and it
    # dominates the ~8e-5 link inertia, so it is randomised hard - a factor of 3 each
    # way rather than a few percent.
    #
    # Split hip/thigh from knee so the knees can also carry the pushrod ratio; see
    # below. The joint sets are disjoint, which matters: dr's "scale" operation
    # multiplies the DEFAULT field rather than the current one, so two events covering
    # the same joint would overwrite each other instead of composing.
    "servo_armature": EventTermCfg(
      mode="reset",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=("^(fl|fr|br|bl)_(hip|top)$",)
        ),
        "operation": "scale",
        "ranges": (0.35, 3.0),
      },
    ),
    # KNEES ONLY, and wider, because the knee is not direct drive. A thigh-mounted
    # DS3218 drives the shank through a ball-jointed pushrod (confirmed from the build
    # photos), and THE RATIO OF THAT LINKAGE HAS NEVER BEEN MEASURED - it cannot be,
    # until the robot is reassembled. It is a named Phase 4 deployment blocker.
    #
    # A ratio error r multiplies reflected rotor inertia at the joint by r^2, so a
    # plausible r in 0.8-1.25 contributes a factor of 0.64-1.56 on top of the armature
    # estimate's own 0.35-3.0: (0.35 x 0.64, 3.0 x 1.5625) = (0.224, 4.69).
    #
    # BE CLEAR ABOUT WHAT THIS DOES NOT COVER. A transmission ratio changes three
    # things - reflected inertia (r^2), effective stiffness and torque at the joint
    # (also r^2 and r), and the servo-angle-to-joint-angle MAPPING. Only the first is
    # expressible here. The second is absorbed by the widened kp/kd band below, whose
    # 0.5-4.0 span comfortably contains 0.64-1.56. The third is not modelled by
    # anything and is not modellable in mjlab 1.5.3: dr has no actuator_gear
    # randomiser, and dr.pd_gains / dr.effort_limits select by actuator CONFIG rather
    # than by joint - Gray has exactly one config covering all twelve joints, so
    # passing actuator_names=(".*_bottom",) resolves to indices [2, 5, 8, 11] into a
    # list of length 1 and raises IndexError. Enabling knee-only gains would mean
    # splitting SERVO_ACTUATOR in train/gray_robot.py into a knee config and a
    # hip/thigh config, which this change does not own. Until then the mapping error
    # stays a deployment blocker rather than a solved problem.
    "knee_pushrod_inertia": EventTermCfg(
      mode="reset",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=("^(fl|fr|br|bl)_bottom$",)),
        "operation": "scale",
        "ranges": (0.224, 4.69),
      },
    ),
    # WIDENED FROM 0.7-1.4 TO 0.5-4.0, and the old range was buying nothing.
    #
    # The servo model is kp = 20 N.m/rad against a 1.96 N.m effort limit, which is a
    # proportional band of only 5.6 degrees - past that the servo is simply saturated.
    # In sim the hip-pitch servo sits at its torque limit 72.6% of the time and ends 25
    # degrees from commanded. Scaling kp by 0.7-1.4 moves the band to 4.0-8.0 degrees,
    # which is still inside the saturated regime for every draw: the policy therefore
    # never saw a servo that behaves differently, and the randomisation was decorative.
    # 0.5-4.0 spans a 2.8-degree band up to a 22-degree one, so a real fraction of
    # environments actually track their commands and the policy has to work in both
    # regimes.
    #
    # THE ROBOT IS DISASSEMBLED, so none of this can be bench-checked. Widening the
    # band is containment, not a fix, and it should not be mistaken for one: if the
    # real servo turns out to be far outside 0.5-4.0 the policy still will not transfer.
    "servo_gains": EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "kp_range": (0.5, 4.0),
        "kd_range": (0.5, 4.0),
      },
    ),
    # Not an encoder - Gray has none. This is the per-servo zero-offset error from
    # horn spline misalignment: real, permanent, and roughly one spline tooth.
    "servo_zero_offset": EventTermCfg(
      mode="reset",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.03, 0.03),
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(3.0, 6.0),
      params={"velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
    ),
  }

  ##
  # Rewards.
  ##

  # ==================================================================================
  # THE FREEZE ARITHMETIC - MEASURED AT THE START OF TRAINING, NOT ESTIMATED
  # ==================================================================================
  # Failure #1 of this project was a reward set every term of which was maximised by
  # standing still. The policy duly stood still and 675 mm of travel became 65 mm. The
  # check that matters is therefore not per-term but on the SUM, and it has to be taken
  # at the point training actually begins, on the fleet training actually begins with.
  #
  # 1024 environments x 1200 steps, per-step reward RATE (value x weight, before the
  # manager's dt scaling), curriculum active so the envelope is at its 0.3 start, and
  # the actions are the STARTING POLICY's - samples from N(0, 0.10), the init_std in
  # train/tasks.py - not zeros.
  #
  #   WALKING = the classical crawl with that exploration noise on top.
  #   FROZEN  = identical, with the gait's stride scale forced to ~0 so the robot
  #             marches on the spot. It is NOT commanded to stand: the command still
  #             asks for forward motion and every reward still demands it. That is
  #             failure #1 exactly as it happened.
  #
  #                              WALKING     FROZEN      delta
  #     forward_progress         +0.8903    -0.2254    +1.1157
  #     track_linear_velocity    +0.9290    +0.8377    +0.0913
  #     track_angular_velocity   +0.3903    +0.4109    -0.0205
  #     upright                  +0.4699    +0.4668    +0.0031
  #     cross_track_drift        +0.1779    +0.2150    -0.0371
  #     support_count            +0.0337    +0.0366    -0.0029
  #     soft_landing             -0.1826    -0.1692    -0.0134
  #     joint_acc                -0.1150    -0.0974    -0.0176
  #     stance_foot_anchoring    -0.3338    -0.1979    -0.1359
  #     swing_clearance          -0.1011    -0.0831    -0.0181
  #     servo_tracking_error     -0.2303    -0.2224    -0.0079
  #     pitch_regulation         -0.0052    -0.0051    -0.0001
  #     base_height_floor        -0.0006    -0.0010    +0.0003
  #     action_rate              -0.0120    -0.0120     0.0000
  #     action_acc               -0.0036    -0.0036     0.0000
  #     preclip_action_mag       -0.0002    -0.0002     0.0000
  #     fall_penalty             -0.0348    -0.0385    +0.0037
  #                              -------    -------    -------
  #     penalties only           -1.0194    -1.0558
  #     TOTAL                    +1.8718    +0.9112    +0.9606
  #
  #   WALKING WINS BY +0.96, i.e. it scores 2.05x what freezing scores.
  #
  # WHAT THAT NUMBER REPLACES. Under the previous configuration - full randomisation
  # from step 0, CROSS_TRACK_STD 0.03, excess_touchdowns priced alongside soft_landing -
  # the same arithmetic INVERTED: frozen beat walking by 0.09, because the envelope had
  # destroyed 89% of the gait before learning began and there was no walking left to
  # reward. No reward weight can fix that; the reward was correctly reporting that the
  # robot was not walking. The fix is the randomisation ramp above, not a weight.
  #
  # ONE TERM CARRIES THE MARGIN, exactly as before. forward_progress supplies +1.12 of
  # the +0.96 gap and every other term nets out NEGATIVE against walking by -0.155
  # combined. Standing still genuinely is better on almost everything else. That is why
  # its weight is 3.0 and why nothing may be added to this set without re-running this
  # table.
  #
  # WHAT IS STILL WRONG AND IS NOT A REWARD PROBLEM. forward_progress reads +0.89 of a
  # possible +3.00. Two causes, both outside this file: the fleet at scale 0.3 walks at
  # roughly two thirds of nominal by design, and every episode opens with a spawn
  # transient that throws the trunk ~330 mm BACKWARD in its first half second (see the
  # note at reset_robot_joints). Fixing the transient is worth more to this table than
  # any reward change available here.
  #
  # THE RULE, and it is a rule rather than a guideline:
  #
  #     sum of all penalties at good behaviour  <  0.35 x the forward_progress weight
  #
  # i.e. under 1.05 here. Past that, the cheapest available way to raise the return is
  # to stop moving, and Failure #1 gets rebuilt out of new parts.
  #
  #   MEASURED, ON THE UNTRAINED GAIT: -1.019, against the 1.05 ceiling. It fits, with
  #   3% of headroom, and it fits only because the excess_touchdowns/soft_landing double
  #   charge was resolved: with both priced the same measurement gives -1.42, which is
  #   35% over. Note this is the STARTING figure - every large item in it
  #   (stance_foot_anchoring -0.33, servo_tracking_error -0.23, soft_landing -0.18) is
  #   something training should reduce - so good behaviour will sit below it, which is
  #   the direction the rule wants.
  #
  # cross_track_drift IS NOW COUNTED IN BOTH COLUMNS rather than excluded. It is still
  # the one term a statue scores perfectly, and it still fails the freeze test - +0.178
  # walking against +0.215 frozen - but at std 0.10 it is a live term worth 5x what it
  # was worth at 0.03, and the progress cap in train/rewards.py holds the frozen column
  # to +0.215 instead of the full +0.50. Counting it costs 0.037 of margin and the table
  # says so rather than hiding it.
  #
  # A WARNING ABOUT UNITS, because it is the likeliest way this table is wrong. The ten
  # units are no longer a risk in the table itself: every row above came out of
  # RewardManager._step_reward on a running environment, so it is the implementation
  # that was measured and not a description of it. The risk that remains is DRIFT -
  # change a weight, a std, the randomisation ramp or the gait and the whole table is
  # stale. RE-RUN IT, do not patch a row. Then check Episode_Reward/* against it inside
  # the first 50 iterations of any run.
  # ==================================================================================
  rewards = {
    # Linear in achieved speed: unlike the exponential, its gradient never dies, so
    # a stalled policy is always told which way to go. This is the term that makes
    # standing still a losing strategy.
    "forward_progress": RewardTermCfg(
      func=ForwardProgress,
      weight=3.0,       # 2.0 -> 3.0. See the freeze arithmetic above: this term is not
                        # merely the largest contributor to the walking-vs-frozen gap,
                        # it is the ONLY positive one. Measured, it supplies +1.12 of a
                        # +0.96 gap while every other term nets -0.155 against walking.
                        # At weight 2.0 the gap would be +0.66.
      params={"command_name": "twist", "tau": TRACK_TAU_S},
    ),
    "track_linear_velocity": RewardTermCfg(
      func=TrackFilteredLinearVelocity,
      weight=1.5,
      params={"command_name": "twist", "std": TRACK_LIN_STD, "tau": TRACK_TAU_S},
    ),
    # DEMOTED, 1.0 -> 0.5, and widened 0.050 -> 0.150. It used to be the straightness
    # term; it was bad at it, for the reason spelled out at TRACK_ANG_STD - at 0.050 a
    # perfectly straight walk scored 0.08 because the std was tighter than the filtered
    # yaw noise itself. Straightness moves to cross_track_drift below, which measures a
    # POSITION and so cannot be swamped by ripple. What is left here is a damper on
    # gross yaw, and half the weight is the right price for that. A stationary robot
    # still scores this perfectly, which is counted against it in the arithmetic above.
    "track_angular_velocity": RewardTermCfg(
      func=TrackFilteredAngularVelocity,
      weight=0.5,
      params={"command_name": "twist", "std": TRACK_ANG_STD, "tau": TRACK_TAU_S},
    ),
    # Also halved, for the same reason - standing still is perfectly upright. Falling
    # over is already punished by the fell_over termination, which ends the episode
    # and forfeits all future reward; this term only needs to discourage leaning.
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=0.5,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
    # --- straightness, measured as a position rather than a rate -------------------
    # THE term that replaces track_angular_velocity's failed attempt at this. Scores
    # exp(-d^2/std^2) on the perpendicular distance from the line the robot was told to
    # walk along. It needs no filter and no check against the stride ripple, because a
    # zero-mean ripple integrates to nothing over a position - it is ripple-immune by
    # construction rather than by tuning, which is the whole reason it works where the
    # yaw-rate version did not.
    #
    # FREEZE TEST: THIS ONE FAILS. A robot that never moves is never off the line, and
    # scores 1.0 forever. Gating on commanded speed (below) stops it paying during
    # deliberate stand commands but does NOT stop it paying a policy that has frozen
    # while still being told to walk. It must additionally be capped by achieved
    # progress in train/rewards.py; see the note in the arithmetic above.
    "cross_track_drift": RewardTermCfg(
      func=CrossTrackDrift,
      weight=0.5,
      params={
        "command_name": "twist",
        "std": CROSS_TRACK_STD,
        "command_threshold": 0.005,
      },
    ),
    # --- the brittle-parts group. Every one of these is minimised by not moving, so
    # the SUM is held under 0.35 x the forward_progress weight - see the budget above.
    # Individually they are now well above a standard velocity task, which is the point:
    # the parts are SLA resin and walking is nothing but repeated impact. ---
    #
    # 26x heavier: -1e-4 was described in the devlog as "a judgement, not a measurement"
    # and at that weight it was contributing essentially nothing. The measurement that
    # settles it is 262 footfalls against the 80 the gait commands - the robot is
    # landing three times more often than it means to, and every unplanned landing is
    # an impact on a brittle part. Still the first knob to back off if it tiptoes.
    #
    # THIS IS THE SURVIVOR OF THE excess_touchdowns / soft_landing PAIR. Both priced the
    # same physical event and the two together measured -0.56 of a -1.05 budget. The
    # argument is not close once you look at what each one computes:
    #
    #   soft_landing        = sum over feet of (contact force x first_contact)
    #   excess_touchdowns   = count of first_contacts past one per commanded gait cycle
    #
    # soft_landing ALREADY CONTAINS THE COUNT. It fires on exactly the same event and
    # sums over every one of them, so a robot landing 262 times instead of 80 pays it
    # 3.3x over. What it adds is the severity of each event, which excess_touchdowns
    # cannot see at all: a 0.5 N re-graze and a 40 N stomp both cost it exactly 0.5.
    # The reverse containment does not exist.
    #
    # And severity is what actually breaks the parts. Fatigue life of a brittle
    # thermoset goes as roughly S^-m with m of order 10, so halving the landing force
    # buys three orders of magnitude of life while halving the number of landings buys
    # a factor of two. "Fails on repeated impact" is true, but the repetition only
    # matters at a given stress amplitude, and amplitude is the term that dominates.
    # Pricing by force is pricing the physics; pricing by count is pricing a proxy that
    # is blind to the variable the S-N curve is steep in.
    #
    # WHAT IS GIVEN UP, stated rather than hidden: excess_touchdowns PASSED the freeze
    # test (-0.3506 walking against -0.3757 frozen) and soft_landing FAILS it weakly
    # (-0.2095 against -0.1784). Dropping the anti-freeze term of the pair and keeping
    # the freeze-shaped one costs 0.025 of walking-vs-frozen margin, which is 1% of the
    # measured gap. That is the price of pricing the right quantity, and it is cheap.
    # The count is not lost, only unpriced - `excess_touchdowns` is still computed and
    # logged as Episode_Metrics/excess_touchdowns below, so a policy that starts
    # bouncing is still visible, it just is not charged for the same event twice.
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-0.02,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "command_name": "twist",
        "command_threshold": 0.005,
      },
    ),
    # ANSWERING THE ARGUMENT FOR KEEPING excess_touchdowns AS A PRICED TERM: "a policy
    # can pass soft_landing by touching down gently 262 times where the gait commands
    # 80, and only excess_touchdowns notices." It cannot. soft_landing is
    # sum(force x first_contact) summed over feet AND accumulated every step, not a
    # per-landing average - 262 gentle landings cost it 3.3x what 80 identical ones do.
    # The scenario the count term is supposed to catch is already charged, three times
    # over, by the force term. What it is NOT charged for is landing gently, which is
    # the behaviour we want.
    # KEPT AS A PRICED TERM. This has now been proposed for demotion to a metric three
    # times, on the grounds that it double-charges soft_landing, and the answer is the
    # same each time: it is not dead - it logged -0.2225 in the run this was last
    # decided from - and the two terms charge different things. soft_landing prices how
    # HARD a foot lands; this prices how OFTEN one lands that should not have. A policy
    # can satisfy the first by touching down gently 262 times in 12 s where the gait
    # commands 80, and only this term notices. Dropping a penalty that protects brittle
    # SLA parts is a design change, not a bug fix. If it is to go, it goes on its own
    # evidence in its own commit, not folded into unrelated work.
    "excess_touchdowns": RewardTermCfg(
      func=ExcessTouchdowns,
      weight=-0.5,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "action_name": "residual",
        "force_threshold": STANCE_FORCE_N,
      },
    ),
    "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-3e-7),
    "action_acc": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.005),
    # Kept at -0.05, but read its logged value with suspicion until the noise runaway
    # is confirmed dead. It contributed -0.104 last run, which implies 0.415 of
    # step-to-step change per joint - and pure random sampling at that run's std gives
    # 0.44. The term was measuring the policy's own exploration noise, not its
    # behaviour. It only becomes a smoothness signal again once Policy/mean_std turns
    # over; see train/tasks.py.
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
    # --- foot behaviour: where the commanded motion is actually being lost ----------
    # THE BIGGEST ITEM IN THIS SET. Penalises the squared horizontal speed of any foot
    # carrying load, targeting the measured 68 mm/s of skid.
    #
    # Gated on MEASURED CONTACT FORCE above 1 N - not on commanded gait phase, and not
    # on "contact > 0". The contact channel chatters at 3.3x the stride rate, so an
    # ungated version would spend its budget scoring bounce instead of skid.
    #
    # HONEST SCOPE: skid is the SMALLER of the two losses. The chain is 267 mm/s
    # commanded at the foot -> 163 mm/s the leg actually sweeps -> 68 mm/s skidding ->
    # 57 mm/s of body travel. This term addresses the 68. The larger loss, 267 -> 163,
    # is the leg failing to achieve its commanded sweep at all, and this term does
    # nothing about it; that is probably the servo model (kp 20 N.m/rad against a
    # 1.96 N.m limit is a 5.6 degree proportional band, and the hip-pitch servo sits at
    # its torque limit 72.6% of the time), which is what servo_tracking_error below and
    # the widened gain randomisation are for.
    #
    # FREEZE TEST: fails weakly - a stationary foot has zero slip. That weak failure is
    # part of why forward_progress went to 3.0.
    # RISK: over-weighted, the cheap way to cut slip is to SHORTEN THE STRIDE rather
    # than grip better. Episode_Metrics/stance_slip_speed logs the mean slip against
    # the 68 mm/s baseline so that trade is visible; read it next to distance.
    "stance_foot_anchoring": RewardTermCfg(
      func=StanceFootAnchoring,
      weight=-8.0,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "force_threshold": STANCE_FORCE_N,
        "asset_cfg": SceneEntityCfg("robot", geom_names=(FOOT_GEOM_REGEX,)),
      },
    ),
    # The back feet lift 0.8 mm against 34 mm commanded - they are dragging, not
    # swinging. Prerequisite for anything on terrain, and a dragging foot is also a
    # foot that trips.
    #
    # Gated on COMMANDED GAIT PHASE here, deliberately opposite to the terms above: a
    # foot that is meant to be swinging but is on the ground registers as contact, so
    # gating on measured contact would switch this term off in exactly the case it
    # exists to catch. The chatter argument does not apply because the gait phase is
    # noise-free by construction.
    # FREEZE TEST: passes cleanly - a frozen robot never enters swing, and the terms
    # that pay for entering swing are unaffected.
    "swing_clearance": RewardTermCfg(
      func=SwingClearance,
      weight=-0.6,
      params={
        "action_name": "residual",
        "target_height": SWING_CLEARANCE_M,
        "foot_radius": FOOT_RADIUS_M,
        "asset_cfg": SceneEntityCfg("robot", geom_names=(FOOT_GEOM_REGEX,)),
      },
    ),
    # The stability the crawl gait was chosen for in the first place, and it currently
    # holds only 36% of the time. Worth restating how bad the rest of the picture is:
    # the body centre is OUTSIDE the support triangle at 18 of 20 sampled phases, by up
    # to 29.8 mm, so the gait is not statically stable even when three feet are down.
    # Measured contacts, force-gated, for the usual chatter reason.
    "support_count": RewardTermCfg(
      func=SupportCount,
      weight=0.25,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "force_threshold": STANCE_FORCE_N,
        "min_contacts": 3,
      },
    ),
    # --- posture: the two terms that stop the other penalties being gamed -----------
    # ONE-SIDED, penalises being below target height only. Without it, both
    # stance_foot_anchoring and servo_tracking_error are cheapest with the legs folded
    # and the body on the floor, and the policy would find that. Never fires above the
    # floor, so it costs nothing at good behaviour.
    # FREEZE TEST: passes - it is indifferent between standing and walking at height.
    "base_height_floor": RewardTermCfg(
      func=BaseHeightFloor,
      weight=-2000.0,      # units m^-2; a 10 mm shortfall costs 0.20/step
      params={
        "target_height": BASE_HEIGHT_FLOOR_M,
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
    # Raises the back feet WITHOUT demanding more servo travel, which matters because
    # the servos are already saturated 72.6% of the time - levelling the trunk buys
    # rear clearance for free where asking the rear knees for more lift does not.
    # Distinct from `upright`, which scores tilt in any direction and therefore lets a
    # steady nose-up pitch hide inside a good score.
    # FREEZE TEST: passes.
    "pitch_regulation": RewardTermCfg(
      func=PitchRegulation,
      weight=-1.5,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
    ),
    # --- sim-to-real and policy hygiene ---------------------------------------------
    # The gap between commanded joint angle and achieved joint angle. This is the term
    # that decides whether ANY of this transfers: it pushes the policy into the part of
    # the envelope where the actuator is still linear, and the linear part is the only
    # part where the simulator and the hardware can be expected to agree. Outside it,
    # sim is extrapolating a servo model that CANNOT BE BENCH-CHECKED because the robot
    # is disassembled. In sim the hip-pitch servo currently ends 25 degrees from
    # commanded.
    #
    # Reading true joint angles is fine here. Rewards never ship to the robot - only
    # observations are constrained by the no-feedback rule, and this is a reward.
    "servo_tracking_error": RewardTermCfg(func=ServoTrackingError, weight=-0.20),
    # Penalises the RAW action BEFORE the +/-0.2 rad clip. Restores gradient in exactly
    # the region the clip flattens: a clipped sample produces no gradient at all, so
    # once the distribution has drifted out past the clip nothing pulls it back while
    # the entropy bonus keeps pushing it wider. That is the mechanism behind
    # Policy/mean_std going 0.101 -> 0.313 without ever turning over, and behind the
    # clip fraction reaching 27.9%. Small on purpose - it is a restoring force on the
    # distribution, not a smoothness penalty. Watch it together with
    # Episode_Metrics/clip_fraction.
    "preclip_action_magnitude": RewardTermCfg(
      func=PreClipActionMagnitude, weight=-0.002
    ),
    # One-shot, on the terminating step only. REPLACES ANY SURVIVAL BONUS, and that
    # substitution is the point: a survival bonus at a normal weight of 0.5 pays
    # 0.5 x 0.02 s x 600 steps = +6.0 over an episode of standing perfectly still,
    # which is Failure #1 rebuilt out of new parts. This pays nothing for existing and
    # charges only for falling.
    #
    # Note the manager's dt scaling: reward = value x weight x dt, so -250 lands as
    # -5.0 on the step it fires, against a well-walked episode's total return of about
    # +4.65 x 0.02 x 600 = +56. That is deliberately a kick and not a catastrophe - the
    # real cost of falling is the forfeited remainder of the episode, and this only has
    # to make sure an early fall is not accidentally cheaper than a bad walk.
    "fall_penalty": RewardTermCfg(func=FallPenalty, weight=-250.0),
  }
  # DELETED IN THIS REVISION, and why, so they do not get re-added:
  #
  #   joint_torques (-1e-4). Contributed 0.05% of total reward and pointed the wrong
  #     way: on a position-commanded hobby servo that is saturated most of the time,
  #     the route to lower torque is to stop pushing, which on this robot means to stop
  #     walking. Cost of transport is out for the same reason and must NOT come back as
  #     a reward - a DS3218MG's current is dominated by holding torque, so a stalled
  #     servo would score as nearly free. Keep it as a logged metric if it is wanted.
  #
  #   dof_pos_limits (-1.0). STRUCTURALLY DEAD - verified in the code, not assumed.
  #     ResidualGaitAction.process_actions clamps nominal + residual to
  #     data.soft_joint_pos_limits before the target ever reaches the actuator
  #     (train/residual_action.py, the "second clamp"), so the COMMAND can never leave
  #     the soft band. The reward reads the MEASURED joint position rather than the
  #     command, so strictly it is not impossible for it to fire - but the margins make
  #     it moot: the joints range +/-2.356 rad, the 0.95 soft factor puts the band at
  #     +/-2.238 rad, and the crawl gait's joint angles peak at 0.894 rad even at stride
  #     scale 1.6 (above the 1.51 the maximum command produces). Add the 0.2 rad
  #     residual and the largest reachable command is ~1.094 rad. A joint would have to
  #     be driven 1.14 rad past where it was told to go for this term to produce a
  #     single non-zero value, and the servos are lagging BEHIND their targets by 25
  #     degrees, not overshooting them. It has never fired and cannot.
  #
  # CONSIDERED AND DELIBERATELY NOT ADDED:
  #
  #   feet_air_time - designed for trot and run, where the failure mode is scuffing.
  #     Gray runs a duty-0.75 crawl that WANTS feet down; rewarding air time inverts
  #     the gait that was chosen on purpose, and mjlab's version is anyway gated on a
  #     command threshold of 0.5 m/s, six times Gray's top speed.
  #   any survival / is_alive bonus - see fall_penalty above.
  #   cost of transport as a REWARD - wrong physics for a hobby servo; see above.
  #   gait symmetry - the RL layer exists partly to ABSORB the few-mm per-leg CAD
  #     asymmetry that Phase 2 could not tune out. Penalising asymmetry asks it to
  #     recreate the problem.
  #   posture regularisation - on a residual architecture the gait IS the nominal, so
  #     a pull toward the default pose is a pull toward standing. It is a freeze term
  #     wearing a different name.

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(60.0)},
    ),
  }

  metrics = {
    "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
    # The two numbers that were invisible during the last run. Neither is a reward;
    # both are read while training rather than after it. See the docstrings.
    "clip_fraction": MetricsTermCfg(func=clip_fraction),
    "stance_slip_speed": MetricsTermCfg(
      func=stance_slip_speed,
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "force_threshold": STANCE_FORCE_N,
        "asset_cfg": SceneEntityCfg("robot", geom_names=(FOOT_GEOM_REGEX,)),
      },
    ),
    # DEMOTED FROM A REWARD, not deleted. It priced the same physical event as
    # soft_landing - see the note above soft_landing for why force survived and count
    # did not - but the count is still the clearest single read on whether the robot is
    # bouncing, and losing sight of it would mean losing the 262-against-80 measurement
    # that motivated both terms. Baseline: 0.30 excess touchdowns per step implied by
    # the classical gait's 262 footfalls against the 80 it commands in 12 s; measured on
    # the untrained gait it is 0.80.
  }

  ##
  # Curriculum. Exactly one term, and it is load-bearing rather than a refinement:
  # under the full envelope the classical crawl covers 47 mm of the 426 mm it walks on
  # a nominal robot, and residual RL on a starting point that cannot walk has nothing
  # to be a residual OF. See `randomisation_ramp` above for the mechanism and
  # DR_ENVELOPE for the measurement.
  #
  # NOTE WHAT THE EVENT TERMS ABOVE ARE CONFIGURED WITH: the FULL ranges. The
  # curriculum narrows them for the first third of training and then hands them back.
  # That ordering is deliberate - anything that builds this config and ignores the
  # curriculum (play mode below, scripts/eval_policy.py, a bare env in a notebook) gets
  # full randomisation by default rather than a quietly easier robot.
  ##
  curriculum = {
    "randomisation_ramp": CurriculumTermCfg(
      func=randomisation_ramp,
      params={"start": DR_SCALE_START, "full_at_step": DR_RAMP_END_STEPS},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_gray_robot_cfg()},
      sensors=(feet_contact,),
      num_envs=1,
      env_spacing=1.0,
      extent=1.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=1.0,
      elevation=-15.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        # sim/models/gray.xml carries <option integrator="implicitfast"
        # cone="elliptic" impratio="10"/>, but MjSpec.attach() does NOT propagate
        # <option> into an mjlab scene - it only warns. Without these three lines
        # training silently ran on pyramidal friction with impratio 1 while
        # scripts/walk.py ran elliptic with impratio 10, so the gait was being tuned
        # against one contact model and learned against another.
        integrator="implicitfast",
        cone="elliptic",
        impratio=10.0,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    # 0.005 s x 4 = 50 Hz, the PWM ceiling. Note walk.py steps at 0.002 s x 10 for
    # the same 50 Hz: the timestep deliberately differs so that scripts/eval_policy.py
    # re-scoring a checkpoint in plain MuJoCo is a genuinely independent check, which
    # is the point of the dual-sim validation in docs/PROJECT_NOTES.md.
    decimation=4,
    episode_length_s=12.0,     # same window Phase 2 was measured over
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # NO RAMP IN PLAY MODE, and this is not an oversight. common_step_counter starts
    # at zero in a fresh viewer, so leaving the curriculum in would show every
    # checkpoint - including a finished one - on a 0.3x fleet and flatter it. The
    # events keep the full ranges they are configured with, so play shows the robot
    # the policy actually has to survive.
    cfg.curriculum = {}

  return cfg
