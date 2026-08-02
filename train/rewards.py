"""Gray's reward terms: velocity tracking, foot mechanics and brittle-part protection.

WHY mjlab's STOCK TRACKING REWARD DOES NOT WORK HERE
----------------------------------------------------
`mdp.track_linear_velocity` scores exp(-|v_cmd - v|^2 / std^2) against the trunk's
INSTANTANEOUS body-frame velocity. That is fine for a 12 kg Go1 cruising at 1-2 m/s.
Gray averages 0.055 m/s, and its instantaneous trunk velocity was measured over a
steady crawl as:

    vx  mean +0.0549   sd 0.2050
    vy  mean -0.0084   sd 0.1430

The stride ripple is nearly **four times the mean**. Every footfall of a stiff,
position-controlled hobby servo jolts a 1.6 kg trunk, so the signal the reward is
meant to measure is buried under its own gait. Choosing std to match the command range
drives the exponent to about -6 and the reward, and its gradient, to zero - which is
exactly what the first smoke run showed: `track_linear_velocity: 0.0000` while every
other term was alive.

Raising std until the reward responds would just make it a ripple-smoothness reward,
paying the policy to fight the Bezier profile Phase 2 deliberately chose.

THE FIX
-------
Track a low-pass-filtered velocity instead. "Walking at 55 mm/s" is a statement about
average velocity, not instantaneous. A first-order filter at tau = one gait period
attenuates the ripple 12x while leaving the mean untouched (measured):

    raw          vx sd 0.2050   vy sd 0.1430
    tau = 0.6 s  vx sd 0.0162   vy sd 0.0218      mean unchanged at 0.0566

so std can be tight enough to actually discriminate 40 mm/s from 60 mm/s.

The filter is per-environment state, so these are stateful ManagerTermBase classes
rather than plain functions - mjlab instantiates a term whose `func` is a class.

THE SAME PROBLEM IN THE YAW CHANNEL, AND THEN AGAIN IN THE CROSS-TRACK ONE
--------------------------------------------------------------------------
Yaw was measured the same way: raw sd 0.532 rad/s, filtered at tau = 0.6 s sd 0.080.
`TrackFilteredAngularVelocity` was originally configured with std = 0.050, which is
TIGHTER THAN THE RESIDUAL NOISE - a perfectly straight walk still scores only
exp(-0.080^2/0.050^2) = 0.08, and a very good stride reaches ~0.28. The term spends
its whole life in the flat part of the exponential, which is failure mode #2 recurring
in a second channel. std is now 0.150, and straightness is paid for by `CrossTrackDrift`
below, which is immune to ripple by construction rather than by tuning.

It then happened a THIRD time in `CrossTrackDrift` itself, at std 0.03 m against a
per-step cross-track distance of 37-119 mm depending on the randomisation - a walking
policy scored 7% of the term's ceiling. std is now 0.10 m, derived from the measured
per-step distance rather than from an end-of-run drift figure. The pattern is worth
naming, because it has now cost three terms: THE STD MUST BE SIZED AGAINST THE SPREAD
OF THE QUANTITY THE TERM ACTUALLY SAMPLES, at the point in training where the term has
to have slope. Sizing it against the target, or against a summary statistic collected
somewhere else, produces a dead term every time.

WHAT ELSE IS IN THIS FILE
-------------------------
Everything below the velocity terms is about the four feet, and every number quoted in
those docstrings was measured on the classical crawl in this simulator:

    commanded foot sweep rel. body    267 mm/s
    sweep the leg actually achieves    163 mm/s
    foot skidding on the ground         68 mm/s
    body travelling forward              57 mm/s   (21% of commanded)
    foot lift commanded 34 mm; measured front 21.6 mm, BACK FEET 0.8 mm
    "three feet down at all times" holds only 36% of the time
    the gait commands 80 footfalls in 12 s; the robot makes 262
    classical baseline over 6 seeds: 711.0 +/- 86.3 mm, drift -52.6 +/- 162.1 mm
      (the single "675.4 mm / 33.8 mm" figure quoted elsewhere is ONE LUCKY DRAW
      and is not the benchmark)

REWARDS MAY READ ANYTHING THE SIMULATOR KNOWS. The "no feedback ever" rule that keeps
measured joint positions out of the observation space applies to OBSERVATIONS only -
those have to survive on a Raspberry Pi driving twelve dumb servos. Rewards are consumed
by the optimiser during training and are never shipped, so contact forces, true joint
angles and foot positions are all fair game and are used freely below.

THE FREEZE TEST
---------------
Failure #1 on this project was a reward set every term of which was maximised by
standing still: the policy abandoned a gait that walked 675 mm and learned to stand
(65 mm). Every term added here therefore carries an explicit answer to one question,
IN ITS DOCSTRING:

    does standing still score STRICTLY better on this term than doing the task well?

"Strictly" matters. A one-sided or target-seeking penalty whose optimum is reachable
*while walking* (base height, pitch) ties at zero and passes. A penalty that is only
ever zero when nothing moves (slip, servo gap) fails weakly and has to be paid for out
of the forward_progress budget. Terms that fail are documented as failing, not quietly
dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from gray.gait import DEFAULT_DUTY, GAIT_PATTERNS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# One full gait cycle. Long enough to average a stride, short enough that the reward
# still reacts within an episode.
DEFAULT_TAU_S = 0.6

# A foot is "carrying load" above this. NOT `contact > 0`: the contact channel chatters
# at 3.3x the stride rate (262 detected footfalls against 80 commanded in 12 s), so an
# ungated term scores bounce instead of scoring stance. 1 N is ~6% of the robot's 16 N
# weight - well below what a genuinely loaded foot carries, well above a graze.
# train/gray_env.py owns the configured value (STANCE_FORCE_N); this is the fallback.
CONTACT_FORCE_N = 1.0

# Radius of the foot collision spheres in sim/models/gray.xml (`*_bottom_collision`,
# size="0.012"). Ground clearance is the geom centre height minus this; on the flat
# terrain used in Phase 3 the ground is z = 0.
FOOT_RADIUS_M = 0.012

##
# THE FREEZE ARITHMETIC - MEASURED AT THE START OF TRAINING, NOT ESTIMATED.
#
# Per-step reward RATE (value x weight, before the reward manager's dt scaling), 1024
# envs x 1200 steps, in the training environment exactly as train/gray_env.py configures
# it: commands drawn from the real range, pushes on, CURRICULUM ACTIVE so the
# randomisation envelope is at its 0.3 starting scale, and the actions are the STARTING
# POLICY's rather than zeros - samples from N(0, 0.10), the init_std in train/tasks.py.
#
#   WALKING = the classical crawl with that exploration noise on top.
#   FROZEN  = identical, with the stride scale forced to ~0. Note that this is NOT a
#             statue: `foot_targets` scales step_LENGTH only, so a zero-stride robot
#             still lifts each foot the full 35 mm and sets it down in place. It marches
#             on the spot, under a command that still says walk. That is failure #1 as
#             it actually happened (675 mm -> 65 mm with the legs still cycling), and it
#             is the only freeze this architecture can reach, since a +/-0.2 rad
#             residual cannot cancel the gait.
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
#   WALKING WINS BY +0.96, scoring 2.05x what freezing scores.
#
# ONE TERM CARRIES IT. forward_progress supplies +1.12 of the +0.96 gap; every other
# term in the set nets out NEGATIVE against walking, by -0.155 combined. Standing still
# genuinely is better on almost everything else, exactly as failure #1 predicted. The
# 2.0 -> 3.0 weight change on that one term is load-bearing, not cosmetic: at 2.0 the
# gap is +0.66.
#
# WHAT THIS REPLACES, AND WHY THE COMPARISON MATTERS. Under the previous configuration -
# full randomisation from step 0, CROSS_TRACK_STD 0.03 m, excess_touchdowns priced
# alongside soft_landing, reset joint offset +/-0.05 rad - the same arithmetic INVERTED:
# frozen beat walking. Not because the reward set was wrong but because the randomisation
# envelope had destroyed 89% of the classical gait before the first gradient step, so
# there was no walking left to reward and the reward was correctly reporting it. The fix
# was the randomisation ramp in train/gray_env.py, not a weight.
#
# THE PENALTY BUDGET FITS, AND ONLY JUST. -1.019 against the 1.05 ceiling, 3% of
# headroom, measured on the untrained gait. It fits because the double charge was
# resolved: with excess_touchdowns priced alongside soft_landing the same measurement
# gives -1.42, 35% over. Every large item left (stance_foot_anchoring -0.33,
# servo_tracking_error -0.23, soft_landing -0.18) is something training should reduce,
# so good behaviour sits below the starting figure.
#
# THE LARGEST REMAINING DEFECT IS NOT IN THIS FILE. forward_progress reads +0.89 of a
# possible +3.00 partly because every episode opens with a spawn transient that throws
# the trunk ~330 mm BACKWARD in its first half second: mjlab spawns Gray in the standing
# pose while `ResidualGaitAction.reset` randomises the gait phase, so step 1 commands
# every joint to jump 0.2-0.5 rad at once and twelve saturated servos launch the robot.
# See the note at reset_robot_joints in train/gray_env.py. Fixing that is worth more to
# this table than any reward change available here.
#
# RULE, unchanged: keep the sum of all penalties at good behaviour under 0.35 x the
# forward_progress weight - at weight 3.0, under 1.05.
#
# DELIBERATELY NOT ADDED, and why:
#   feet_air_time      - designed for trot/run, where feet SHOULD be off the ground. A
#                        duty-0.75 crawl wants feet DOWN; rewarding air time inverts
#                        the gait that was chosen for stability in the first place.
#   any survival bonus - at a typical weight it pays about +6.0 for a full episode of
#                        standing perfectly still. That is failure #1 rebuilt from
#                        parts. The one-shot `FallPenalty` below replaces it.
#   cost of transport  - wrong physics for a hobby servo, whose current is dominated by
#                        holding torque. A stalled servo draws its maximum and would
#                        score as free. Keep it as a LOGGED METRIC, never a reward.
#   gait symmetry      - the RL layer exists partly to absorb the few-mm per-leg CAD
#                        asymmetry Phase 2 could not tune out. Paying for symmetry pays
#                        the policy to stop doing its job.
#   posture regularisation - on a residual architecture the classical gait IS the
#                        nominal pose, so "stay near default" drags every joint toward
#                        the standing pose. It is a freeze term wearing a different
#                        name. (`PreClipActionMagnitude` gives the useful half of this:
#                        it pulls toward a ZERO RESIDUAL, i.e. toward the gait that
#                        already walks, not toward standing.)
##


class _StatelessTerm(ManagerTermBase):
  """A term with no per-environment state that still wants to be a class.

  mjlab instantiates a class-valued `func` as `func(cfg=term_cfg, env=env)`, but
  `ManagerTermBase.__init__` takes only `env`. These terms could be plain functions;
  they are classes so that the whole reward set is imported under one naming
  convention, and so that any of them can grow state later without a signature change
  rippling into train/gray_env.py.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    self.cfg = cfg


class _EmaVelocity(ManagerTermBase):
  """Shared first-order filter over a body-frame velocity signal."""

  _WIDTH = 1

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    self.cfg = cfg
    self._filtered = torch.zeros(env.num_envs, self._WIDTH, device=env.device)
    self._step_dt = float(env.step_dt)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    # Start from rest: a reset robot really is stationary, and the filter reaches
    # steady state well inside one 12 s episode.
    self._filtered[env_ids] = 0.0

  def _update(self, sample: torch.Tensor, tau: float) -> torch.Tensor:
    alpha = self._step_dt / (tau + self._step_dt)
    self._filtered += alpha * (sample - self._filtered)
    return self._filtered


class TrackFilteredLinearVelocity(_EmaVelocity):
  """Reward tracking the commanded planar velocity, averaged over a stride."""

  _WIDTH = 2

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    tau: float = DEFAULT_TAU_S,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # Deliberately planar. The stock reward also penalises vertical velocity, but
    # Gray's trunk bob is prescribed by the Bezier swing profile - penalising it
    # would be paying the policy to undo Phase 2's touchdown shaping, which exists
    # to protect brittle SLA parts.
    filtered = self._update(asset.data.root_link_lin_vel_b[:, :2], tau)
    error = torch.sum(torch.square(command[:, :2] - filtered), dim=1)
    return torch.exp(-error / std**2)


class ForwardProgress(_EmaVelocity):
  """Reward for actually covering ground, with a gradient that never dies.

  WHY THIS EXISTS
  ---------------
  The exponential tracking reward above has a fatal property when it is the *only*
  velocity signal: exp(-e^2/std^2) is flat and almost zero once the error is a few
  std out, so a policy that has stopped walking gets no gradient telling it which way
  to go. Meanwhile `upright`, `track_angular_velocity` and every smoothness penalty
  are all *maximised* by standing perfectly still. Standing is therefore a local
  optimum with a wide basin, and the first training run walked straight into it:

      round   0   568 mm      (starts at the classical gait, as intended)
      round  50   623 mm
      round 100   150 mm
      round 150   -39 mm      (walking backwards)
      round 450    65 mm      (slowly relearning, uprightness pinned at 0.996)

  It abandoned a working gait to stand still, which is exactly what residual RL is
  supposed to make impossible.

  This term is linear in achieved velocity, so its gradient is constant and points
  forward everywhere - including from a dead stop, where the exponential says
  nothing. Projecting onto the command direction means it also cannot be farmed by
  running off sideways, and clamping at 1.0 means there is no prize for exceeding
  the commanded speed, only for reaching it.

  Its weight is 3.0, not 2.0. See THE FREEZE ARITHMETIC at the top of this file, which
  is measured rather than estimated: this term supplies +2.90 of the +2.55 walking-vs-
  frozen gap, and every other term in the set nets out NEGATIVE against walking by -0.35
  combined. It is not merely the largest contributor, it is the only one. At weight 2.0
  the gap falls to +1.38, worse than the reward set this one replaces.
  """

  _WIDTH = 2

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    tau: float = DEFAULT_TAU_S,
    command_threshold: float = 0.005,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    filtered = self._update(asset.data.root_link_lin_vel_b[:, :2], tau)

    speed = torch.norm(command, dim=1)
    # velocity projected onto the commanded direction, as a fraction of it: 1.0 means
    # exactly the commanded speed, negative means going the wrong way.
    projected = (filtered * command).sum(dim=1) / speed.clamp(min=1e-6).square()
    # When told to stand, this term says nothing - the exponential term scores that.
    return torch.where(
      speed > command_threshold, projected.clamp(-1.0, 1.0),
      torch.zeros_like(projected),
    )


class TrackFilteredAngularVelocity(_EmaVelocity):
  """Reward tracking commanded yaw rate, averaged over a stride.

  Commanded yaw is always zero (see train/gray_env.py), so in principle this is what
  pays for walking STRAIGHT - absorbing the few-mm per-leg CAD asymmetry that Phase 2
  could not tune out. In practice it was bad at that job, and the job has moved.

  RIPPLE CHECK - and why std went 0.050 -> 0.150.
  Yaw rate has the same problem as forward velocity, only worse: raw sd 0.532 rad/s,
  filtered at tau = 0.6 s sd 0.080 rad/s. A std of 0.050 is TIGHTER THAN THE FILTERED
  NOISE FLOOR, so a robot walking dead straight still scores only
  exp(-0.080^2/0.050^2) = 0.08, wandering up to ~0.28 on a good stride. The term lives
  entirely in the flat part of the exponential - the same pathology that made the stock
  linear tracker read 0.0000, one channel over. At 0.150 a straight walk scores 0.75,
  on the shoulder of the exponential where the slope is largest.

  Its weight also halves, 1.0 -> 0.5. Straightness moves to `CrossTrackDrift`, which
  measures the quantity actually cared about (metres off a line) and cannot be swamped
  by ripple at all. What is left here is a damper on gross yaw.
  """

  _WIDTH = 1

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    tau: float = DEFAULT_TAU_S,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    filtered = self._update(asset.data.root_link_ang_vel_b[:, 2:3], tau)
    error = torch.square(command[:, 2:3] - filtered).squeeze(-1)
    return torch.exp(-error / std**2)


##
# Shared machinery for the foot terms.
##


class _FootTerm(ManagerTermBase):
  """Resolves the four feet once, and lines the geoms up with the contact-sensor slots.

  Gray's MJCF has no foot sites and no terrain height sensor, so a foot is identified by
  its COLLISION GEOM (`^(fl|fr|br|bl)_bottom_collision$`) and its load is read from the
  contact sensor train/gray_env.py builds over those same geoms.

  Foot ORDER is taken from `ContactSensor.primary_names` when a sensor is configured,
  and from the resolved `asset_cfg.geom_names` otherwise. Either way the geom row
  indices are looked up back through `Entity.find_geoms(..., preserve_order=True)` on
  those exact names, so sensor slot i and geom row i are the same foot by construction
  rather than by coincidence - mjlab's own `feet_slip` merely assumes the two orderings
  agree.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    self.cfg = cfg

    self._sensor_name: str | None = cfg.params.get("sensor_name")
    asset_cfg = cfg.params.get("asset_cfg") or SceneEntityCfg("robot")
    self._asset_name: str = asset_cfg.name
    entity: Entity = env.scene[self._asset_name]

    if self._sensor_name is not None:
      names = list(env.scene[self._sensor_name].primary_names)
    else:
      names = list(asset_cfg.geom_names or ())
      if not names:
        raise ValueError(
          f"{type(self).__name__} needs either a 'sensor_name' param or an "
          "'asset_cfg' whose geom_names select the foot collision geoms."
        )

    geom_ids, found = entity.find_geoms(names, preserve_order=True)
    if found != names:
      raise RuntimeError(
        f"foot names {names} do not resolve to geoms on entity "
        f"'{self._asset_name}' (got {found}). This term indexes geom rows by foot "
        "slot and cannot proceed without that alignment."
      )
    self._foot_names: tuple[str, ...] = tuple(names)
    self._num_feet = len(names)
    self._foot_geom_ids = torch.tensor(geom_ids, device=env.device, dtype=torch.long)

  # Accessors. Methods rather than cached tensors because mjlab hands the env to
  # __call__, not to __init__.

  def _sensor(self, env: ManagerBasedRlEnv):
    assert self._sensor_name is not None
    return env.scene[self._sensor_name]

  def _asset(self, env: ManagerBasedRlEnv) -> Entity:
    return env.scene[self._asset_name]

  def _loaded(
    self, env: ManagerBasedRlEnv, force_threshold: float = CONTACT_FORCE_N
  ) -> torch.Tensor:
    """[B, F] bool - feet carrying more than `force_threshold` newtons.

    The sensor is configured `reduce="netforce"`, so `force` is one net wrench per foot
    in the global frame and its magnitude is the load that foot is taking.
    """
    force = self._sensor(env).data.force
    assert force is not None, (
      f"contact sensor '{self._sensor_name}' must include 'force' in its fields"
    )
    return torch.norm(force, dim=-1) > force_threshold

  def _foot_speed_xy(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """[B, F] horizontal speed of each foot in the WORLD frame, m/s."""
    vel = self._asset(env).data.geom_lin_vel_w[:, self._foot_geom_ids, :2]
    return torch.norm(vel, dim=-1)

  def _foot_speed_xy_rel_body(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """[B, F] horizontal foot speed relative to the trunk, m/s - the achieved sweep."""
    asset = self._asset(env)
    vel = asset.data.geom_lin_vel_w[:, self._foot_geom_ids, :2]
    trunk = asset.data.root_link_lin_vel_w[:, None, :2]
    return torch.norm(vel - trunk, dim=-1)

  def _foot_clearance(
    self, env: ManagerBasedRlEnv, foot_radius: float = FOOT_RADIUS_M
  ) -> torch.Tensor:
    """[B, F] height of the UNDERSIDE of each foot above the ground plane, m."""
    z = self._asset(env).data.geom_pos_w[:, self._foot_geom_ids, 2]
    return z - foot_radius


class _GaitPhasedFootTerm(_FootTerm):
  """A foot term that also knows where each leg is in the COMMANDED gait cycle.

  The residual action term owns the gait clock (`ResidualGaitAction.phase`, one scalar
  per env in [0, 1)). Per-leg phase is that clock plus the leg's offset, read straight
  out of `gray.gait.GAIT_PATTERNS` so there is no second copy of the gait definition to
  drift out of sync. `gray.gait.foot_offset` puts stance at leg-phase < duty and swing
  at leg-phase >= duty, so touchdown is the phase wrapping 1 -> 0.

  Reading COMMANDED phase rather than measured contact is the entire point for
  `SwingClearance`: the gait clock advances no matter what the robot does, so a robot
  that has stopped lifting its feet is still being measured against a schedule it is
  failing.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    params = cfg.params
    self._action_name: str = params.get("action_name", "residual")
    pattern: str = params.get("gait_pattern", "crawl")
    if pattern not in GAIT_PATTERNS:
      raise ValueError(f"unknown gait pattern {pattern!r}")
    duty = params.get("duty")
    self._duty = float(DEFAULT_DUTY[pattern] if duty is None else duty)

    offsets = GAIT_PATTERNS[pattern]
    per_foot = [offsets[name.split("_")[0]] for name in self._foot_names]
    self._offsets = torch.tensor(
      per_foot, device=env.device, dtype=torch.float
    ).unsqueeze(0)  # [1, F]

  def _leg_phase(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """[B, F] commanded phase of each leg, in [0, 1)."""
    phase = env.action_manager.get_term(self._action_name).phase
    return (phase.unsqueeze(1) + self._offsets) % 1.0

  def _swing_fraction(self, env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """([B, F] bool in-swing, [B, F] progress through swing in [0, 1))."""
    leg_phase = self._leg_phase(env)
    in_swing = leg_phase >= self._duty
    frac = (leg_phase - self._duty).clamp(min=0.0) / (1.0 - self._duty)
    return in_swing, frac


##
# 1. Stance foot anchoring - the biggest single item.
##


class StanceFootAnchoring(_FootTerm):
  """Penalise a LOADED foot sliding along the ground. Weight -8.0.

  WHAT IT IS FOR
  --------------
  The measured chain of losses through one stride:

      commanded foot sweep relative to the body    267 mm/s
      sweep the leg actually achieves              163 mm/s   (-104)
      foot skidding on the ground                   68 mm/s
      body travelling forward                       57 mm/s   (21% of commanded)

  This term attacks the 68 mm/s of skid, and only that. BE HONEST ABOUT THE REST: the
  larger loss is the first one, 267 -> 163 mm/s, the leg simply not achieving the sweep
  it was told to make, and NOTHING HERE ADDRESSES IT. It may not even be real - the
  servo model puts 20 N.m/rad of stiffness against a 1.96 N.m limit, a 5.6 degree
  proportional band, and in sim the hip-pitch servo sits at its torque limit 72.6% of
  the time. That shortfall is at least partly an artefact of the actuator model, and
  the robot is disassembled so it cannot be bench-checked. `ServoTrackingError` below
  contains it; this term does not pretend to fix it.

  GATING
  ------
  Contact is gated on MEASURED FORCE > 1 N, not on `contact > 0` and not on commanded
  gait phase. The contact channel chatters at 3.3x the stride rate (262 detected
  footfalls against 80 commanded in 12 s), so a term gated on the raw flag spends most
  of its budget scoring bounce. Gating on commanded phase would be worse still: it
  would charge for skid on a foot the gait BELIEVES is planted while it is airborne.

  RIPPLE CHECK
  ------------
  This reads an instantaneous velocity, which failure #2 says to distrust. It is safe
  here for a reason that does not extend to trunk velocity: the quantity being scored
  is not a mean with ripple on top, it is a magnitude whose target is ZERO. Filtering a
  signal whose target is zero only blurs which foot and which instant was at fault, and
  a squared, non-negative cost cannot be cancelled by a zero-mean ripple the way a
  signed velocity error can. No filter.

  FREEZE TEST: FAILS WEAKLY, as expected, and by more than the design budget allowed.
  Measured -0.3207 walking against -0.1411 frozen. The budget predicted -0.11 from
  8.0 x 3 loaded feet x the mean 0.068 m/s squared; the real figure is 3x that, because
  slip is heavy-tailed - mean slip measures 0.087 m/s but the squared cost implies an
  RMS several times higher, so a few badly-planted feet dominate. Squaring is still the
  right choice (it is what makes those feet visible at all), but size the weight against
  the RMS, not the mean.

  RISK, AND HOW TO SEE IT
  -----------------------
  Over-weighted, the cheapest way to cut slip is to stop trying to push - shorten the
  stride, take smaller bites, walk slower. In the reward curve that is indistinguishable
  from gripping better. `MeanFootSlip` and `MeanStanceSweep` are logged for exactly
  this: slip falling (baseline 68 mm/s) while sweep holds (baseline 163 mm/s) is the
  win; both falling together is the cheat, and distance travelled will confirm it.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = CONTACT_FORCE_N,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    del sensor_name, asset_cfg  # resolved once in __init__.
    loaded = self._loaded(env, force_threshold).float()  # [B, F]
    speed = self._foot_speed_xy(env)  # [B, F]

    log = env.extras.get("log")
    if log is not None:
      n_loaded = loaded.sum()
      denom = n_loaded.clamp(min=1.0)
      # Baselines: slip 0.068 m/s, sweep 0.163 m/s. Read them together - see above.
      log["Metrics/stance_slip_mean"] = (speed * loaded).sum() / denom
      log["Metrics/stance_sweep_mean"] = (
        self._foot_speed_xy_rel_body(env) * loaded
      ).sum() / denom
      log["Metrics/stance_feet_mean"] = n_loaded / env.num_envs

    return torch.sum(torch.square(speed) * loaded, dim=1)


##
# 2. Cross-track drift - the straightness term that actually works.
##


class CrossTrackDrift(ManagerTermBase):
  """Reward staying on the straight line the robot was told to walk. Weight +0.5.

  exp(-d^2 / std^2) on d, the perpendicular distance from the line through the pose the
  robot reset into, along the heading it reset with. std 0.10 m - see the derivation in
  train/gray_env.py at CROSS_TRACK_STD, and the paragraph on 0.03 m below.

  WHY POSITION AND NOT YAW RATE
  -----------------------------
  Straightness is the thing actually wanted, and it is a statement about POSITION. The
  classical gait's drift over 6 seeds is -52.6 mm with sd 162.1 and a range of +33.8 to
  -358.7 mm; that spread is what the RL layer is supposed to absorb, and it is measured
  in millimetres off a line, not in rad/s.

  RIPPLE CHECK: IMMUNE BY CONSTRUCTION, not by tuning. Position is the integral of
  velocity, and a zero-mean ripple integrates to zero. The lateral velocity ripple that
  makes instantaneous vy useless (sd 0.143 against a mean of -0.008) contributes exactly
  nothing to d. This is the correct escape from failure #2: pick a signal the ripple
  cannot reach, instead of filtering one it dominates.

  Any constant offset between world and env-local coordinates cancels in `pos - origin`,
  so this works whatever frame `root_link_pos_w` is expressed in.

  FREEZE TEST: FAILS ON ITS OWN, and there is no honest way to dress that up. A robot
  that never moves is never off the line and would collect the full +0.50 forever. Two
  things are done about it:

    1. GATED on COMMANDED speed, so it pays nothing during the 10% of episodes
       commanded to stand. This alone does NOT fix the freeze case - a frozen robot
       under a live command still scores perfectly.
    2. CAPPED by achieved along-track progress, which does. The cap is the fraction of
       the commanded distance the robot has actually covered along the line,
       clamp(|along| / (commanded_speed x elapsed), 0, 1). Both halves are integrals of
       velocity, so the cap inherits the same ripple immunity as d itself - no filter is
       needed for it either. A robot that has gone nowhere multiplies its perfect
       straightness by zero, and a robot that walks 200 mm and then stops watches its
       cap decay as the denominator keeps growing.

  THE std WAS 0.03 m AND THAT WAS FAILURE #2 IN A THIRD CHANNEL. First TRACK_LIN_STD at
  0.035, then TRACK_ANG_STD at 0.050, then this: a std tighter than the quantity it is
  measuring, leaving the exponential flat exactly where it needs slope. At 0.03 m a
  walking policy scored +0.035 of a possible +0.50 - seven percent - because the drift
  the robot actually produces (64 mm on the run that was measured) scores exp(-4.5) =
  0.011. There was no gradient toward straightness until the policy was already straight.

  std IS NOW 0.10 m, DERIVED FROM WHAT THIS TERM ACTUALLY SAMPLES. It is charged every
  step, on a cross-track distance that starts at zero and grows, so the quantity that
  has to sit on the slope is the PER-STEP MEAN |cross-track| over an episode, not the
  end-of-run drift. Measured on the classical crawl over 10 s, 1024 robots:

      nominal robot                        37.4 mm
      the fleet training starts on         68.9 mm
      the full randomisation envelope     118.6 mm

  exp(-d^2/std^2) is steepest at d = std/sqrt(2), so putting the maximum-slope point on
  the envelope training begins with gives std = sqrt(2) x 68.9 mm = 97 mm. Rounded to
  0.10 m, the term reads 0.87 / 0.62 / 0.24 across those three rows and 1.00 at zero
  drift - nearly the whole range in use, with the steepest part where the untrained
  policy sits. At 0.03 m the same rows read 0.22 / 0.005 / 1e-7.

  MEASURED WALKING vs FROZEN, at the start of training and at std 0.10: +0.187 walking
  against +0.221 frozen. It is worth 5x what it was worth at 0.03 (+0.035 / +0.083) and
  the freeze deficit is SMALLER in absolute terms (-0.034 against -0.048), so widening
  the std bought slope without buying freeze incentive. It is still a weak failure and
  is still counted as one; the progress cap is what keeps it weak.

  The origin and heading are latched lazily on the first call after a reset rather than
  inside reset(), so this does not depend on whether the reset events have already
  written the new root state by the time the managers are reset.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    self.cfg = cfg
    self._origin = torch.zeros(env.num_envs, 2, device=env.device)
    self._dir = torch.zeros(env.num_envs, 2, device=env.device)
    self._expected = torch.zeros(env.num_envs, device=env.device)
    self._latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._step_dt = float(env.step_dt)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._latched[env_ids] = False
    self._expected[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.03,
    command_threshold: float = 0.005,
    progress_floor: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.root_link_pos_w[:, :2]
    heading = asset.data.heading_w
    facing = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)

    # Branchless latch: no host sync, no .any() on the GPU.
    latched = self._latched.unsqueeze(1)
    self._origin = torch.where(latched, self._origin, pos)
    self._dir = torch.where(latched, self._dir, facing)
    self._latched = torch.ones_like(self._latched)

    delta = pos - self._origin
    # 2D cross product: signed perpendicular distance from the commanded line. Commands
    # reverse sign (-0.04 to +0.08 m/s) but the LINE is the same line either way, which
    # is why the direction vector never needs re-latching on a command resample.
    cross = delta[:, 0] * self._dir[:, 1] - delta[:, 1] * self._dir[:, 0]
    along = (delta * self._dir).sum(dim=1).abs()

    speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    self._expected = self._expected + speed * self._step_dt
    # progress_floor keeps the ratio finite in the first few steps, where both terms
    # are ~0; it makes the cap ramp in from zero rather than start at one, which is the
    # conservative direction.
    cap = (along / self._expected.clamp(min=progress_floor)).clamp(0.0, 1.0)

    log = env.extras.get("log")
    if log is not None:
      log["Metrics/cross_track_abs_mean"] = cross.abs().mean()
      log["Metrics/along_track_fraction"] = cap.mean()

    reward = torch.exp(-torch.square(cross) / std**2) * cap
    return torch.where(speed > command_threshold, reward, torch.zeros_like(reward))


##
# 3. Servo saturation / tracking error.
##


class ServoTrackingError(_StatelessTerm):
  """Penalise the gap between the angle a servo was commanded and the angle it reached.

  Weight -0.20. Sum over the twelve joints of
  clamp(|target - achieved| - deadband, min=0)^2.

  WHY THIS DECIDES WHETHER ANY OF THIS TRANSFERS
  ----------------------------------------------
  The servo model is 20 N.m/rad of stiffness against a 1.96 N.m effort limit. Divide:
  the actuator is linear only within a 0.098 rad (5.6 degree) proportional band, and
  outside it the servo is a constant-torque source that no longer behaves like the
  position device the policy believes it is commanding. In sim the hip-pitch servo is
  AT ITS TORQUE LIMIT 72.6% OF THE TIME and finishes 25 degrees (0.44 rad) from where
  it was told to go.

  A policy trained in that regime is not learning to control a servo. It is learning the
  saturation behaviour of one particular MuJoCo actuator model - the least trustworthy
  part of this simulation, and the part that CANNOT BE BENCH-CHECKED because the robot
  is disassembled. Pushing the policy back into the linear band pushes it into the only
  region where sim and hardware have a chance of agreeing. The contingency for the model
  being wrong is wider randomisation (train/gray_env.py already spreads armature
  0.35-3.0x and gains 0.7-1.4x); this term is not a fix for it.

  WHAT IS COMPARED
  ----------------
  `joint_pos_target` is what `ResidualGaitAction.apply_actions` wrote: gait + residual,
  clamped to the soft limits, MINUS the randomised per-servo zero-offset. The joint's
  true angle is `joint_pos`. Both are in the actuator's own frame, so that zero-offset
  cancels exactly and this term never punishes the policy for a miscalibrated horn it
  has no way to observe. The 2-8 physics-step command latency does not cancel, so a
  small part of this reading is transport lag rather than saturation; that is bounded
  and harmless.

  DEADBAND: default 0.0, so every departure from the commanded angle costs something,
  including the unavoidable gravity-holding deflection. Setting it to 0.098
  (= 1.96 / 20, the proportional band) turns this into a pure saturation penalty that
  ignores the linear region entirely. That is the first knob to reach for if the policy
  starts refusing to load its legs.

  RIPPLE CHECK: reads positions, not velocities. Not applicable.

  FREEZE TEST: FAILS WEAKLY, but only just. Measured -0.2493 walking against -0.2238
  frozen: holding a static stance against gravity costs almost as much as walking does,
  so the freeze incentive here is small (-0.026) even though the term itself is large.
  Paid for out of forward_progress, like the other weak failures.

  MEASURED SATURATION at the classical gait: mean |gap| 0.166 rad (9.5 degrees) against
  a 0.098 rad proportional band, worst joint 1.09 rad (62 degrees). The servos are
  outside their linear region on average, not occasionally - which is the whole reason
  this term exists, and a standing reminder that the actuator model is the least
  trustworthy thing in this simulation.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    deadband: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    gap = (
      asset.data.joint_pos_target[:, asset_cfg.joint_ids]
      - asset.data.joint_pos[:, asset_cfg.joint_ids]
    )
    log = env.extras.get("log")
    if log is not None:
      # Baseline: the hip-pitch joint currently ends 0.44 rad from commanded, against
      # a 0.098 rad proportional band.
      log["Metrics/servo_gap_abs_mean"] = gap.abs().mean()
      log["Metrics/servo_gap_abs_max"] = gap.abs().max()
    excess = (gap.abs() - deadband).clamp(min=0.0)
    return torch.sum(torch.square(excess), dim=1)


##
# 4. Swing clearance.
##


class SwingClearance(_GaitPhasedFootTerm):
  """Penalise a foot COMMANDED to be mid-swing that is not off the ground. Weight -0.6.

  Cost per swinging foot is the squared NORMALISED height deficit,
  clamp(1 - h/target, 0, 1)^2. Normalising keeps it dimensionless and makes a foot at
  zero clearance cost exactly 1.0 whatever `target_height` is set to; an absolute
  squared-metres version would be ~5e-4 at this scale and effectively invisible next to
  every other term.

  THE MEASUREMENT THAT MOTIVATES IT
  ---------------------------------
      foot lift commanded          34 mm   (peak of the Bezier swing arc)
      measured, front feet       21.6 mm
      measured, BACK FEET         0.8 mm

  The back feet are not swinging. They are being dragged. That is a direct contributor
  to the 68 mm/s of skid and to the 262 detected footfalls against 80 commanded, and it
  makes anything on terrain impossible - a foot with 0.8 mm of clearance cannot cross a
  1 mm pebble.

  WHY THE GATE IS COMMANDED PHASE, NOT MEASURED CONTACT
  -----------------------------------------------------
  The opposite choice from `StanceFootAnchoring`, deliberately. Measured contact cannot
  be used to define swing here, because "the foot is still touching when it should not
  be" is precisely the failure being measured - gate on contact and the term switches
  itself off in exactly the case it exists to catch. The gait clock is an open-loop
  schedule that advances regardless, which is what gives this term its teeth. The
  chatter argument that forces force-gating elsewhere does not apply, because the
  commanded phase is noise-free by construction.

  Only the middle of swing is scored (`swing_window`, default the central half). At the
  ends of the swing the Bezier arc is legitimately near the ground - it is shaped for a
  near-vertical, low-speed touchdown to protect SLA resin - so demanding clearance
  there would pay the policy to slam the foot down, which is what `soft_landing` exists
  to prevent. Across the default window the commanded height runs from 22 mm at the
  edges to 34 mm at the apex, so the 25 mm target asks for less than the gait already
  commands.

  RIPPLE CHECK: reads a position, not a velocity. Not applicable.

  FREEZE TEST: FAILS WEAKLY. This was specified as passing cleanly, on the reasoning
  that the gait clock keeps commanding swings whether or not the robot obliges, so a
  frozen robot would take the full -0.60. MEASUREMENT SAYS OTHERWISE: -0.1395 walking
  against -0.0919 frozen, so the frozen robot scores BETTER.

  The reasoning was right about the clock and wrong about the robot.
  `GaitGenerator.foot_targets` scales step_LENGTH by the stride scale and leaves
  step_HEIGHT alone, so a robot that has stopped translating is still lifting each foot
  the full 35 mm - it marches on the spot. Its feet then go straight up and down instead
  of arcing forward, which actually clears MORE reliably than a real stride does. There
  is no reachable state in this architecture where the gait clock runs and the feet stay
  down, so the mechanism that was supposed to make this term freeze-proof does not
  exist.

  It is still worth its weight - it is the only term that pays for the back feet leaving
  the ground at all - but it belongs in the "fails weakly, paid for out of
  forward_progress" group with slip and servo gap, not in a class of its own.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    target_height: float = 0.025,
    swing_window: tuple[float, float] = (0.25, 0.75),
    foot_radius: float = FOOT_RADIUS_M,
    sensor_name: str | None = None,
    action_name: str = "residual",
    gait_pattern: str = "crawl",
    duty: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    del sensor_name, action_name, gait_pattern, duty, asset_cfg  # resolved in __init__.

    in_swing, frac = self._swing_fraction(env)
    lo, hi = swing_window
    scored = in_swing & (frac >= lo) & (frac < hi)

    clearance = self._foot_clearance(env, foot_radius)  # [B, F]
    deficit = (1.0 - clearance / target_height).clamp(0.0, 1.0)

    log = env.extras.get("log")
    if log is not None:
      n = scored.float().sum().clamp(min=1.0)
      # Baselines: front feet 0.0216 m, back feet 0.0008 m, commanded 0.034 m.
      log["Metrics/swing_clearance_mean"] = (clearance * scored.float()).sum() / n

    return torch.sum(torch.square(deficit) * scored.float(), dim=1)


##
# 5. Excess touchdowns.
##


class ExcessTouchdowns(_GaitPhasedFootTerm):
  """Count footfalls the gait never asked for. NOW A METRIC, NOT A REWARD.

  DEMOTED, and the reason is the whole point of the term. It and `soft_landing` priced
  the SAME PHYSICAL EVENT - a foot's first contact - and between them measured -0.40 of
  a -1.05 penalty budget. `soft_landing` sums (contact force x first_contact) over the
  feet, so it already fires on every one of these events and already charges 3.3x for a
  robot that lands 262 times instead of 80. What it adds is the SEVERITY of each
  landing, which this term cannot see at all: a 0.5 N re-graze and a 40 N stomp both
  cost it exactly 0.5. The containment runs one way only.

  Severity is also the quantity that actually breaks the parts. Fatigue life of a
  brittle thermoset goes roughly as S^-m with m of order 10, so halving the landing
  force buys three orders of magnitude of life where halving the number of landings
  buys a factor of two. The count matters, but only at a given stress amplitude, and
  amplitude is where the S-N curve is steep. Pricing force is pricing the physics.

  It is still COMPUTED, as Episode_Metrics/excess_touchdowns, because the count is the
  clearest single read on whether the robot is bouncing and the 262-against-80
  measurement is what motivated both terms in the first place. Everything below still
  describes what it measures; only its weight is gone.

  The crawl commands exactly one touchdown per leg per cycle: 4 legs x 20 cycles = 80
  footfalls in a 12 s episode. The robot makes 262. Each of those extra 182 is an
  unplanned impact on a brittle SLA resin part.

  HOW "EXCESS" IS DEFINED, AND WHY IT MATTERS
  -------------------------------------------
  Not as a rate. Subtracting a flat 80/600 touchdowns per step would charge the policy
  for its legitimate footfalls too, leaving a permanent negative floor that a robot
  could escape only by never stepping again - a freeze term in disguise.

  Instead each leg carries a budget of ONE touchdown per COMMANDED gait cycle. The
  counter clears when that leg's commanded phase wraps, which is its scheduled touchdown
  instant (`gray.gait.foot_offset` starts stance at leg-phase 0). The first measured
  touchdown after that is free; the second and every one after it inside the same
  commanded cycle is charged. A gait that lands each foot once per cycle scores exactly
  ZERO, so this penalty TIES with standing still instead of losing to it.

  Gated on measured force > 1 N so that a graze is not counted as a footfall.

  KNOWN LIMITATION, stated rather than hidden: the contact sensor's air-time bookkeeping
  resets on `found > 0` regardless of force, so if a sub-newton graze immediately
  precedes a real landing, that landing is no longer the foot's "first contact" and goes
  uncounted. The force gate can remove events, not resurrect ones the sensor has already
  swallowed. The term therefore under-counts if it errs, which is the safe direction for
  a penalty.

  RIPPLE CHECK: counts discrete events, reads no velocity. Not applicable.

  FREEZE TEST: PASSED, and that is what it cost to drop it. Measured -0.3506 walking
  against -0.3757 frozen, so the frozen robot scored WORSE - a robot marching on the
  spot re-lands each foot in the same place and bounces more, not less. `soft_landing`,
  the term that survives, fails the freeze test weakly instead (-0.2095 against
  -0.1784). Removing the anti-freeze half of the pair and keeping the freeze-shaped half
  costs about 0.025 of walking-vs-frozen margin, roughly 1% of the measured gap. That is
  the price of pricing the right quantity.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    shape = (env.num_envs, self._num_feet)
    self._touchdowns = torch.zeros(shape, device=env.device)
    self._prev_phase = torch.zeros(shape, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._touchdowns[env_ids] = 0.0
    # ResidualGaitAction randomises the phase on reset, so at worst this fires one
    # spurious budget-clear on the first step of an episode - which only ever makes the
    # term more lenient.
    self._prev_phase[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = CONTACT_FORCE_N,
    action_name: str = "residual",
    gait_pattern: str = "crawl",
    duty: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    del sensor_name, action_name, gait_pattern, duty, asset_cfg  # resolved in __init__.

    leg_phase = self._leg_phase(env)  # [B, F]
    # A commanded touchdown is the leg phase wrapping 1 -> 0. Clear the budget there.
    wrapped = leg_phase < self._prev_phase
    self._prev_phase = leg_phase
    self._touchdowns = torch.where(
      wrapped, torch.zeros_like(self._touchdowns), self._touchdowns
    )

    first_contact = self._sensor(env).compute_first_contact(dt=env.step_dt)
    landed = (first_contact & self._loaded(env, force_threshold)).float()  # [B, F]

    over_budget = (self._touchdowns >= 1.0).float()
    excess = landed * over_budget
    self._touchdowns = self._touchdowns + landed

    log = env.extras.get("log")
    if log is not None:
      # Baselines per step over 600 steps: 0.133 commanded, 0.437 measured.
      log["Metrics/touchdowns_per_step"] = landed.sum() / env.num_envs
      log["Metrics/excess_touchdowns_per_step"] = excess.sum() / env.num_envs

    return torch.sum(excess, dim=1)


##
# 6. Fall penalty.
##


class FallPenalty(_StatelessTerm):
  """One-shot cost on the step an episode ends for falling over. Weight -250.

  Returns 1.0 for environments that terminated - the `fell_over` bad-orientation check
  in train/gray_env.py - and 0.0 for those that merely timed out. mjlab computes
  terminations before rewards within the same step, so the flag is fresh when this runs.

  WHY NOT A SURVIVAL BONUS, WHICH IS THE USUAL WAY TO DO THIS
  -----------------------------------------------------------
  Because a survival bonus is failure #1 rebuilt from parts. At a typical +0.5/step
  weight it pays roughly +6.0 over a 12 s episode for standing perfectly still and never
  attempting anything - a large, unconditional, freeze-shaped income stream. Moving the
  same information into a one-shot terminal cost buys the identical "do not fall" signal
  with none of the per-step incentive to stop moving.

  The weight is large because the event is rare and terminal. It passes through the
  reward manager's dt scaling like every other term, so -250 lands as about -5.0 of
  actual return on the terminating step, against a walking rate of roughly +4.55/s.
  Falling therefore costs a little over one second of good walking, PLUS every step of
  the episode forfeited by ending early - and the forfeited remainder is the real
  deterrent. The one-shot kick only has to ensure that an early fall is never
  accidentally cheaper than a bad walk.

  There is a second, mechanical reason to keep this weight high that has nothing to do
  with RL: on the real robot a fall is 1.6 kg of brittle SLA resin hitting the floor.

  RIPPLE CHECK: reads a boolean, not a velocity. Not applicable.

  FREEZE TEST: FAILS in the trivial sense that a robot standing still never falls. It
  cannot be FARMED, though - it pays nothing per step and can only be avoided, never
  collected, so it adds no per-step income to the frozen column of the arithmetic at the
  top of this file. That is the entire reason it is shaped as a penalty rather than as
  its algebraically equivalent bonus.
  """

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    terminated = env.termination_manager.terminated
    time_out = env.termination_manager.time_outs
    fell = (terminated & ~time_out).float()
    log = env.extras.get("log")
    if log is not None:
      log["Metrics/fall_rate"] = fell.mean()
    return fell


##
# 7. Pre-clip action magnitude.
##


class PreClipActionMagnitude(_StatelessTerm):
  """Penalise the RAW policy output, before the residual clip flattens it. Weight -0.002.

  Sum over the twelve joints of clamp(|a| - deadband, min=0)^2 on
  `env.action_manager.action`, which is the policy's output before any per-term scaling
  or clamping.

  THE PROBLEM IT SOLVES
  ---------------------
  `ResidualGaitAction` computes clamp(a * 0.2, -0.2, +0.2), so the clip binds at
  |a| = 1 and everything beyond it is discarded. Anything discarded produces NO
  GRADIENT: once a sample lands outside the clip, no term in the reward can tell the
  policy which way to move it, because moving it does not change the robot's behaviour
  at all. Meanwhile PPO's entropy bonus keeps pushing the distribution wider. The result
  is a one-way ratchet, and the last run's logs show it:

      Policy/mean_std   0.101 -> 0.191 (r300) -> 0.286 (r700) -> 0.313 (r1150)
      fraction of actions on the clip   1.7% (r500) -> 20.2% (r950) -> 27.9% (r1100)

  never turning over. It also poisoned the diagnostics: `action_rate` at -0.104 implies
  0.415 of step-to-step change per joint, and pure random sampling at that noise level
  gives 0.44 - the smoothness penalty had stopped measuring the policy and started
  measuring its own exploration noise.

  This term restores a gradient in the flat region. It is deliberately tiny - at
  |a| ~ 0.3 across twelve joints it costs about -0.002/step, invisible beside
  forward_progress at +3.00 - because its job is to supply a DIRECTION, not a force.
  Read it next to Episode_Metrics/clip_fraction; neither is meaningful alone.

  DEADBAND: default 0.0, a plain magnitude penalty with slope everywhere. Set it to 1.0
  to charge only for the part of the action the clip discards, leaving the interior of
  the residual box completely free.

  FREEZE TEST: PASSES, and this is where the residual architecture pays a dividend. The
  term is minimised at a = 0, and A ZERO RESIDUAL IS THE PHASE 2 CRAWL GAIT, which walks
  711 mm. Its optimum is a walking robot, not a standing one. This is exactly why plain
  posture regularisation is NOT in this file: the same idea applied in joint space has
  the standing pose as its optimum instead.

  RIPPLE CHECK: reads the action, not a physical velocity. Not applicable.
  """

  def __call__(self, env: ManagerBasedRlEnv, deadband: float = 0.0) -> torch.Tensor:
    action = env.action_manager.action
    excess = (action.abs() - deadband).clamp(min=0.0)
    log = env.extras.get("log")
    if log is not None:
      log["Metrics/action_abs_mean"] = action.abs().mean()
    return torch.sum(torch.square(excess), dim=1)


##
# 8. Support count.
##


class SupportCount(_FootTerm):
  """Reward having at least three feet genuinely loaded. Weight +0.25.

  The crawl gait was chosen over the trot for exactly one reason: at duty 0.75 it keeps
  three feet on the ground at all times, which makes the robot statically stable and
  hard to tip. That property is currently fiction. Measured on the classical gait:

      "three feet down at all times" holds only 36% of the time
      body centre OUTSIDE the support triangle at 18 of 20 sampled phases,
        by up to 29.8 mm

  So Phase 2 is paying the speed cost of a crawl and collecting a third of the stability
  it was supposed to buy. This term pays for the property directly.

  Counted on MEASURED FORCE > 1 N per foot, not on the contact flag: a foot resting with
  a fraction of a newton on it is not carrying the robot, and the flag chatters (262
  detected footfalls against 80 commanded).

  The reward is an indicator, so it has no gradient within a single step. That is normal
  and fine for a contact statistic - PPO optimises an expectation over sampled
  trajectories, and the policy's influence on how OFTEN three feet are loaded is
  perfectly learnable even when the per-step signal is a step function.

  RIPPLE CHECK: counts contacts, reads no velocity. Not applicable.

  FREEZE TEST: FAILS, as designed-for. Measured +0.0339 walking against +0.0585 frozen.
  It is deliberately the smallest positive weight in the set for that reason.

  MEASURED, THE PROBLEM IS WORSE THAN THE 36% QUOTED. With the 1 N load gate the
  classical crawl has three or more feet genuinely loaded only 13.6% of the time, and
  averages 1.67 loaded feet at any instant. A duty-0.75 crawl should average 3.0. The
  robot is spending most of its stride on two feet, which is a trot it never asked for -
  and it explains the skid and the 262 footfalls at the same time. This term has a great
  deal of room to earn its weight.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    min_contacts: int = 3,
    force_threshold: float = CONTACT_FORCE_N,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    del sensor_name, asset_cfg  # resolved once in __init__.
    count = self._loaded(env, force_threshold).sum(dim=1)
    supported = (count >= min_contacts).float()
    log = env.extras.get("log")
    if log is not None:
      # Baseline to beat: 0.36.
      log["Metrics/support_fraction"] = supported.mean()
      log["Metrics/loaded_feet_mean"] = count.float().mean()
    return supported


##
# 9. Base height floor.
##


class BaseHeightFloor(_StatelessTerm):
  """Penalise the trunk sinking BELOW a target height. One-sided. Weight -2000 (m^-2).

  cost = clamp(target_height - z, min=0)^2, silent at or above the target and quadratic
  below it. On the flat terrain used in Phase 3 the ground is z = 0, so z is read
  straight off the trunk body.

  WHY IT IS HERE
  --------------
  It closes a hole that terms 1 and 3 open between them. `StanceFootAnchoring` is
  minimised by feet that do not move and `ServoTrackingError` is minimised by joint
  angles the servo can comfortably hold, and there is one posture that scores well on
  both: crouch. Fold the legs, drop the trunk, shuffle. It looks like progress on two
  reward curves while being useless on the robot and, on the real machine, holding every
  servo in a high-torque pose indefinitely.

  The weight looks enormous because the units are m^-2. A 10 mm shortfall costs
  2000 x 0.01^2 = 0.20/step; a 25 mm crouch costs 1.25/step, which exceeds the entire
  penalty budget at the top of this file. That sharp knee is the point - the term should
  be invisible until the crouch starts and expensive immediately afterwards.

  ONE-SIDED ON PURPOSE: no penalty for riding high. Standing taller costs servo torque
  and the policy will learn that from `ServoTrackingError`; this term has no business
  having an opinion about it.

  RIPPLE CHECK: reads a position, not a velocity. The trunk does bob with every footfall
  - that bob is prescribed by the Bezier profile and is deliberately not penalised
  anywhere - so the floor must sit BELOW the observed minimum trunk height, not at the
  mean, or it will charge for the gait's own designed motion. train/gray_env.py sets it
  at 0.150 against a measured minimum of 0.160.

  FREEZE TEST: PASSES. Its optimum, zero, is reached by any robot at or above the floor,
  walking or standing. It does not score standing STRICTLY better than walking, so it
  contributes nothing to the frozen column.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.body_ids, slice):
      height = asset.data.root_link_pos_w[:, 2]
    else:
      height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].mean(dim=1)
    log = env.extras.get("log")
    if log is not None:
      # Baseline: 0.1661 m mean, 0.1600 m minimum over the classical crawl.
      log["Metrics/base_height_mean"] = height.mean()
    return torch.square((target_height - height).clamp(min=0.0))


##
# 10. Pitch regulation.
##


class PitchRegulation(_StatelessTerm):
  """Penalise trunk pitch. Weight -1.5.

  Scored as (sin(pitch) - sin(target_pitch))^2, read off the x component of gravity
  projected into the trunk frame - which is -sin(pitch) for a body-x-forward convention.
  No atan2 and no gimbal case, and it is the same quantity the IMU-derived
  `projected_gravity` observation already carries, so nothing here needs information the
  real robot could not supply.

  WHY PITCH SPECIFICALLY
  ----------------------
  The back feet clear 0.8 mm against 34 mm commanded. There are two ways to fix that:
  ask the rear legs for more travel, which means more servo angle in exactly the joints
  already at their torque limit 72.6% of the time; or level the trunk, which raises the
  rear of the robot and hands the back feet their clearance for free. This term buys the
  second. It is a cheaper lever on the same problem `SwingClearance` attacks head-on,
  and the two are meant to be read together.

  DISTINCT FROM `upright`, which scores tilt in ANY direction through a single
  exponential and therefore lets a steady nose-up pitch hide inside a perfectly
  respectable score. This one is axis-specific.

  `target_pitch` is a parameter rather than a constant because a small nose-down trim
  may turn out to be the right answer - it lifts the rear further still. Leave it at 0.0
  until there is a measurement that says otherwise.

  RIPPLE CHECK: reads an orientation, not a velocity. Trunk pitch does oscillate with
  the stride, so this is a mild penalty on the gait's own designed motion - which is why
  the weight is small and the term is quadratic in the sine rather than in the angle.

  FREEZE TEST: PASSES. A level trunk scores zero walking or standing, so its optimum is
  reachable while doing the task and it adds nothing to the frozen column. (Strictly, a
  standing robot holds level more easily than a walking one, so a fraction of the
  -0.05/step budget is freeze-shaped. That is small, bounded and accounted for.)
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    target_pitch: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.body_ids, slice):
      projected_gravity_b = asset.data.projected_gravity_b
    else:
      quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
      projected_gravity_b = quat_apply_inverse(quat_w, asset.data.gravity_vec_w)
    sin_pitch = -projected_gravity_b[:, 0]
    error = sin_pitch - torch.sin(
      torch.as_tensor(target_pitch, device=sin_pitch.device, dtype=sin_pitch.dtype)
    )
    log = env.extras.get("log")
    if log is not None:
      log["Metrics/pitch_abs_mean"] = torch.asin(sin_pitch.clamp(-1.0, 1.0)).abs().mean()
    return torch.square(error)


##
# Metrics. No weight, no dt scaling - wire these into `metrics` in train/gray_env.py as
# MetricsTermCfg so they land in TensorBoard as Episode_Metrics/*.
#
# THESE ARE NOT OPTIONAL. `StanceFootAnchoring` can be satisfied two ways - by gripping
# better, or by taking smaller steps - and the reward curve looks identical either way.
# Slip alone cannot tell them apart. Slip AND sweep can:
#
#     slip down, sweep held      -> it grips better.            This is the win.
#     slip down, sweep down too  -> it shortened its stride.    This is the cheat.
#
# Baselines, both measured on the classical crawl: slip 0.068 m/s, sweep 0.163 m/s
# (against 0.267 m/s commanded at the foot).
##


class MeanFootSlip(_FootTerm):
  """Mean world-frame horizontal speed of the loaded feet, m/s. Baseline 0.068.

  Per environment, averaged over whichever feet are carrying more than the force
  threshold this step. Environments with no loaded foot report 0.0 - rare enough on a
  duty-0.75 crawl not to bias the episode average, and honest in the sense that there
  was nothing to measure.

  This is `StanceFootAnchoring`'s cost with the square removed and the sum turned into a
  mean, so it reads directly in millimetres per second against the number that motivated
  the term.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = CONTACT_FORCE_N,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    del sensor_name, asset_cfg  # resolved once in __init__.
    loaded = self._loaded(env, force_threshold).float()
    speed = self._foot_speed_xy(env)
    return (speed * loaded).sum(dim=1) / loaded.sum(dim=1).clamp(min=1.0)


class MeanStanceSweep(_FootTerm):
  """Mean speed of the loaded feet RELATIVE TO THE TRUNK, m/s. Baseline 0.163.

  The other half of the pair, and the half that makes the first one interpretable. This
  is the sweep the leg actually achieves - the middle number in the 267 / 163 / 68 / 57
  mm/s chain - and it is what distinguishes a policy that has learned to grip from one
  that has learned to mince. If this falls while `MeanFootSlip` falls, the stride
  shortened, and distance travelled will confirm it.

  It is also the cheapest available read on the 267 -> 163 mm/s loss that
  `StanceFootAnchoring` explicitly does NOT address, so if the servo model is ever
  corrected against real hardware, this is the curve that should move.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = CONTACT_FORCE_N,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    del sensor_name, asset_cfg  # resolved once in __init__.
    loaded = self._loaded(env, force_threshold).float()
    sweep = self._foot_speed_xy_rel_body(env)
    return (sweep * loaded).sum(dim=1) / loaded.sum(dim=1).clamp(min=1.0)
