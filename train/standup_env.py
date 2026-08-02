"""Stage 1: start lying flat, learn to sit up and stay there.

The owner's plan, in his words: "lets first reliably learn how to get up and get into
position and then we can learn how to walk and then everything else". So this task does
ONE thing. No velocity command, no walking, no turning. An unused command is an
observation the policy learns to ignore, and a tracking reward commanded to zero is a
reward for standing still - which is the failure that has already cost this project two
runs.

    START   resting   hips +55 out, thighs -21, knees +39    trunk  42 mm
    GOAL    sitting   hips   0,     thighs -22, knees +39    trunk 110 mm

Both poses were set by the owner by hand in tools/pose_editor.py and checked for
self-collision at triangle level against the real CAD meshes. See reference/POSES.md.

## WHY THIS TASK EXISTS AT ALL, WHICH IS A FAIR QUESTION

A SCRIPT ALREADY DOES THIS PERFECTLY. gray/standup.py interpolates resting -> sitting
-> standing and was measured at 60 out of 60 successes on randomly built robots across
the full training envelope, landing within 167-180 mm every time. Standing up is a move
between two known poses with the floor holding the robot the whole way; it never has to
react to anything, so there is nothing for feedback to feed back.

This task is therefore NOT expected to beat the script at standing up on flat ground.
It exists to answer one question cheaply: CAN A BLIND POLICY LEARN A WHOLE-BODY MOVE ON
THIS ROBOT AT ALL. Gray's servos report nothing back, ever, so the policy works from
body tilt and its own past commands and nothing else. Every other quadruped doing this
gets joint feedback. If blind learning does not work, it is far better to discover that
on a 7-second movement than after a month on walking.

The script is the benchmark. Beating it is not required; getting anywhere near it while
blind is the result.

## ACTIONS

Direct joint targets, all twelve, scaled into the OWNER'S measured travel rather than
the DS3218MG's own 270 degrees:

    hip   -82 to +92     thigh  -35 to +65     knee  -40 to +39

These are hardware stops on the assembled robot and cannot be derived from the CAD. The
CAD independently agrees for thigh and knee - swept across their whole range and past
it, neither self-collides - and disagrees for the hip only in a pose-dependent way: with
the legs hanging straight the two front feet meet at -28 degrees.

ACTION ZERO IS THE RESTING POSE, not the joint's midpoint. That matters: an untrained
policy outputs near-zero, so at reset it commands almost exactly the pose the robot is
already in. The alternative - zero meaning some arbitrary mid-range pose - commands a
large jump on the first control step of every episode, which is precisely the spawn
transient that put a third of the residual task's episodes on the floor.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from train.gray_env import FOOT_CONTACT_SENSOR
from train.gray_robot import FOOT_GEOM_REGEX, get_gray_robot_cfg
from train.standup_rewards import HoldSitting, PostureMatch, RiseToSitting

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

POSES_JSON = "progress/poses.json"

# The owner's measured hardware travel, in the PHYSICAL convention (hip + is out,
# thigh + is forward, knee + is foot up). Converted to raw per-joint limits below,
# because the legs are mirrored and the raw signs are not uniform.
OWNER_LIMITS_DEG = {"hip": (-82.0, 92.0), "top": (-35.0, 65.0), "bottom": (-40.0, 39.0)}

# Long enough to sit up several times over. The script takes 3.6 s to reach sitting; at
# 8 s a policy that dawdles still gets there, and one that arrives early is paid for
# every extra step it holds the pose.
EPISODE_S = 8.0


def _load_poses() -> tuple[dict, dict]:
  """(resting, sitting) as raw joint angles in radians, keyed by joint name."""
  if not os.path.exists(POSES_JSON):
    raise SystemExit(
      f"{POSES_JSON} is missing. It is written by tools/pose_editor.py and holds the "
      f"poses the owner set by hand; this task has no meaning without them."
    )
  with open(POSES_JSON, encoding="utf-8") as fh:
    poses = json.load(fh)
  for need in ("resting_limp", "sitting_ready"):
    if need not in poses:
      raise SystemExit(f"{POSES_JSON} has no '{need}' pose.")
  return poses["resting_limp"]["angles_rad"], poses["sitting_ready"]["angles_rad"]


def _raw_limits(resting: dict) -> dict[str, tuple[float, float]]:
  """The owner's physical limits turned into raw per-joint ranges.

  The sign that makes "+ is out / forward / up" true differs per leg, and not in one
  pattern: the hips mirror front/back while the thighs and knees mirror left/right.
  Rather than re-deriving that here, the sign is taken from the RESTING pose, whose
  physical values are known (+55 out, -21 forward, +39 up) - so the raw angle's sign
  against the known physical sign gives the per-joint mapping directly.
  """
  physical_resting = {"hip": 55.0, "top": -21.0, "bottom": 39.0}
  out: dict[str, tuple[float, float]] = {}
  for joint, raw_rad in resting.items():
    seg = joint.split("_", 1)[1]
    phys = physical_resting[seg]
    sign = 1.0 if (raw_rad >= 0) == (phys >= 0) else -1.0
    lo_p, hi_p = OWNER_LIMITS_DEG[seg]
    a, b = sign * lo_p, sign * hi_p
    out[joint] = (min(a, b), max(a, b))
  return out


def gray_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  resting, sitting = _load_poses()
  limits = _raw_limits(resting)

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
  # Observations. THE ACTOR NEVER SEES A JOINT ANGLE OR A JOINT VELOCITY. DS3218MG
  # servos report nothing back, so a policy that uses them cannot be deployed - the
  # whole reason Phase 3 leaned on a gait for rhythm. Do not add joint_pos, joint_vel
  # or base_lin_vel to the actor group; base_lin_vel in particular is not available
  # from an IMU without drift.
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    # Its own last command. With position servos this is the only proxy the robot has
    # for where its joints are - imperfect, because a loaded servo lags its command,
    # but it is what the real machine will have.
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
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
  # Actions. Joint targets offset from the RESTING pose so action zero is where the
  # robot already is - see the module docstring on why the alternative launches it.
  ##

  names = list(resting)
  offset = {n: float(resting[n]) for n in names}
  # Scale so an action of +/-1 spans the owner's travel from the resting pose. Taken as
  # the larger side so the full range stays reachable; the action manager clips to the
  # joint limits anyway.
  import math

  scale = {}
  for n in names:
    lo, hi = limits[n]
    reach = max(abs(math.radians(hi) - offset[n]), abs(offset[n] - math.radians(lo)))
    scale[n] = max(reach, 1e-3)

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": envs_mdp.JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=scale,
      offset=offset,
      use_default_offset=False,
    )
  }

  ##
  # Events. The randomisation envelope and its ramp are IMPORTED from train/gray_env.py
  # rather than copied, so a correction to the envelope reaches both tasks.
  ##

  events: dict[str, EventTermCfg] = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    # Start in the resting pose, jittered by about a servo horn's repeatability. Any
    # wider and the task changes from "sit up" to "recover from wherever you landed".
    "reset_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.02, 0.02),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }
  # DOMAIN RANDOMISATION IS DELIBERATELY NOT ON YET. This task exists to answer one
  # question - can a blind policy learn a whole-body move on this robot at all - and
  # the cheapest way to get a wrong answer is to ask it on a fleet of robots that are
  # each built differently. Nominal first. If it learns, the envelope and its ramp are
  # imported from train/gray_env.py and switched on for the run that matters.

  rewards: dict[str, RewardTermCfg] = {
    # The two that make the robot try. Both are zero for a robot that does not move,
    # which is what makes lying still a losing strategy - see the freeze arithmetic in
    # train/standup_rewards.py.
    "rise": RewardTermCfg(func=RiseToSitting, weight=4.0),
    "posture": RewardTermCfg(
      func=PostureMatch, weight=1.5,
      params={"target": sitting, "start": resting},
    ),
    # Arriving is not sitting. This pays only while near the height AND nearly still.
    "hold": RewardTermCfg(func=HoldSitting, weight=1.0),
    # Small on purpose: lying flat is already almost upright, so this cannot be the
    # term that decides the run. It is here to stop the robot sitting up crooked.
    # std 0.5 because the robot STARTS nearly upright lying flat (its trunk is a
    # flat slab, so gravity reads about 0.99 down). A tight std would be nearly
    # satisfied at step one and would have no slope left to give.
    # asset_cfg NAMES THE TRUNK. Left at its default the term resolves to all 13
    # bodies and dies with "size of tensor a (104) must match b (8)" - 13 bodies
    # times 8 envs against 8 root quaternions.
    "upright": RewardTermCfg(
      func=vel_mdp.upright, weight=0.5,
      params={"std": 0.5,
              "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])},
    ),
    # Set for servo gear wear, not for brittle parts - the owner built the robot and
    # says the resin is stronger than the old task assumed, where impact penalties
    # were about 18% of the whole penalty budget and produced a tentative gait.
    # CUT 25x from -2.5e-7. At that weight the smoothness penalty measured -3.98
    # against a +2.12 gain for moving, so it alone made lying still the winning
    # strategy. Sitting up is a big fast movement of the whole body; a weight
    # sized for a walking gait forbids it.
    "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-1e-8),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.005),
    "joint_torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-5e-5),
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits, weight=-1.0
    ),
  }

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    # Only for a robot that has genuinely tipped over. Not a height floor: the robot
    # STARTS on the floor, so a height termination would end every episode at step one.
    "flipped": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": 1.9}
    ),
  }

  # NO CURRICULUM, because there is no domain randomisation for it to ramp. The
  # randomisation_ramp term walks the DR event terms' parameters upward, and with those
  # events absent it fails outright: "Event term 'resin_density_legs' not found in
  # active terms." A curriculum that ramps nothing is not harmless, it is a crash.
  curriculum: dict = {}

  # START LYING DOWN. get_gray_robot_cfg() spawns the robot STANDING, at the Phase 2
  # gait's stance height - correct for a walking task and completely wrong here, where
  # the whole point is to begin on the floor. Measured before this override: the robot
  # reset at 175 mm, so the "rise from 42 mm" reward was already fully paid at step one
  # and there was nothing left to learn.
  #
  # The joint angles come from the owner's own resting pose, and the spawn height is
  # just above where that pose settles (39-42 mm), so the robot drops a millimetre and
  # is on the floor rather than being placed inside it.
  robot_cfg = get_gray_robot_cfg()
  robot_cfg.init_state = dataclasses.replace(
    robot_cfg.init_state,
    # 45 mm: the resting pose settles with its trunk at 42.5 mm, so this drops it
    # 2 mm onto the floor rather than dropping it 33 mm, which was a fall at the
    # start of every episode.
    pos=(0.0, 0.0, 0.045),
    joint_pos=dict(resting),
  )

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
      sensors=(feet_contact,),
      num_envs=1,
      env_spacing=1.0,
      extent=1.0,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=0.9,
      elevation=-15.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=300,
      mujoco=MujocoCfg(timestep=0.005),
    ),
    decimation=4,
    episode_length_s=EPISODE_S,
  )
  return cfg
