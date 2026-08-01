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
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
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
  TrackFilteredAngularVelocity,
  TrackFilteredLinearVelocity,
)

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
# see train/rewards.py for the measurements that forced that. After filtering at
# tau = 0.6 s a well-tracking robot still shows ~0.027 m/s of residual ripple, so std
# is set just above it: tight enough to tell 40 mm/s from 60 mm/s, loose enough that
# the exponent does not collapse.
TRACK_LIN_STD = 0.035
TRACK_ANG_STD = 0.200
TRACK_TAU_S = 0.6

# Resin density +/-40%, as e^(2*alpha) per dr.pseudo_inertia. Covers the open question
# of whether the SLA parts were printed solid (2.00 kg) or hollowed (1.625 kg).
DENSITY_ALPHA = (math.log(0.6) / 2.0, math.log(1.4) / 2.0)

FOOT_CONTACT_SENSOR = "feet_ground_contact"


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
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Legs: density only. Physics-consistent - mass AND inertia scale together, which
    # dr.body_mass would not do.
    "resin_density_legs": EventTermCfg(
      mode="startup",
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
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "alpha_range": DENSITY_ALPHA,
        "t_range": (-0.02, 0.02),
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(FOOT_GEOM_REGEX,)),
        "operation": "abs",
        "ranges": (0.4, 1.2),
        "shared_random": True,
      },
    ),
    # Armature is an ESTIMATE (0.003 kg.m^2, from a ~6e-8 rotor behind ~245:1) and it
    # dominates the ~8e-5 link inertia, so it is randomised hard - a factor of 3 each
    # way rather than a few percent.
    "servo_armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "scale",
        "ranges": (0.35, 3.0),
      },
    ),
    # Hobby servos vary part to part, and gains drift as the battery sags.
    "servo_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "kp_range": (0.7, 1.4),
        "kd_range": (0.7, 1.4),
      },
    ),
    # Not an encoder - Gray has none. This is the per-servo zero-offset error from
    # horn spline misalignment: real, permanent, and roughly one spline tooth.
    "servo_zero_offset": EventTermCfg(
      mode="startup",
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

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=TrackFilteredLinearVelocity,
      weight=2.0,
      params={"command_name": "twist", "std": TRACK_LIN_STD, "tau": TRACK_TAU_S},
    ),
    # Commanded yaw is always zero, so this term is what pays for walking straight.
    "track_angular_velocity": RewardTermCfg(
      func=TrackFilteredAngularVelocity,
      weight=1.0,
      params={"command_name": "twist", "std": TRACK_ANG_STD, "tau": TRACK_TAU_S},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
    # --- the brittle-parts group. Heavier than a standard velocity task. ---
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-4,          # 10x mjlab's default; first knob to tune if it tiptoes
      params={
        "sensor_name": FOOT_CONTACT_SENSOR,
        "command_name": "twist",
        "command_threshold": 0.005,
      },
    ),
    "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-1e-6),
    "action_acc": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.01),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.1),
    # --- everything else ---
    "joint_torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-1e-4),
    "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
  }

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

  metrics = {"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)}

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
    curriculum={},
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

  return cfg
