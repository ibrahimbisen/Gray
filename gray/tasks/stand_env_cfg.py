"""Smoke test - stand still.

The robot starts in the standing pose and has to stay there: trunk level, trunk
at the stance height (202.4 mm, read from progress/stance/stance.yaml rather than
written here, so the two cannot disagree), feet where they were put, not falling
over.

This is NOT a shipped policy and it is not a subsection of the plan. Standing is
the zero-velocity corner of R1 locomotion - same reward, same observation, so the
same policy file. What this task is, and all it is, is a check that the model is
right before six hours are spent finding out otherwise.

Why this is the first thing trained, before anything that moves: it is the
cheapest possible test of whether the model is right. A static torque check says
twelve servos at 1.96 N-m hold 3100 g in this pose with 1.82x to spare, and the drop test
says the robot holds itself up for four seconds without a controller at all - so
if a policy cannot learn to stand, the fault is in the reward or the training
setup, not in the robot. Minutes to find that out here; six hours to find it out
in a walking run.

The observations deliberately include measured joint positions. The previous
attempt at this project could not use them, because the servos had no feedback
and a policy that reads a joint angle it cannot actually measure will not
transfer. The potentiometers being fitted change that.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import torch
import yaml

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.rl.config import RslRlModelCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactMatch
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

ROOT = Path(__file__).resolve().parents[2]
MJCF = ROOT / "sim" / "models" / "gray.xml"
STANCE = ROOT / "progress" / "stance" / "stance.yaml"
CONFIG = ROOT / "gray" / "config" / "robot.yaml"

ROBOT = SceneEntityCfg("robot")
ALL_JOINTS = SceneEntityCfg("robot", joint_names=(".*",))

# What each scoring term is actually paying for, in plain words. The dashboard
# shows these beside the weights so a run can be read without opening this file -
# and so a term that is quietly dominating the total is obvious.
REWARD_NOTES = {
    "tilt": "Trunk off level. Measured off gravity, so it needs no outside reference.",
    "upright": "Trunk off level. (Older runs called this term 'upright'.)",
    "height": "Trunk at the ride height. 1.0 dead on, falling away over 30 mm.",
    "still": "Trunk not moving. Standing still means still, not slowly drifting.",
    "posture": "Joints near the stance they started in, rather than any pose that "
               "happens to stay up.",
    "alive": "A small amount for every moment it has not fallen. Stops it ending "
             "an attempt early to escape a bad score.",
    "fell_over": "Fell. By far the largest single number here, and the one thing "
                 "that ends an attempt.",
    "effort": "Total torque across the twelve joints. Keeps the pose inside what a "
              "20 kg-cm servo can hold, and saves battery.",
    "joint_speed": "Joints moving at all. A robot standing still should be still.",
    "twitching": "How much the commanded angles jump between one moment and the "
                 "next. Smooth commands are the single biggest factor in whether "
                 "this survives contact with a real servo.",
    "end_stops": "Driving a joint into the last few degrees of its travel. On the "
                 "real robot that is a servo straining against a hard stop.",
}


def _stance() -> tuple[dict[str, float], float]:
    """The standing pose, straight out of scripts/find_stance.py.

    Since 3 Aug 2026 those twelve angles are the owner's, stated by hand in
    gray/config/robot.yaml, and find_stance.py only measures them. The pose it
    used to solve for planted each foot under its own hip; this one slides them
    forward and outward, which puts the footprint under the centre of mass
    instead of behind it. Worst lean margin goes 94.6 mm -> 106.0 mm, and the
    knees pay for it: 0.72 N-m -> 1.08 N-m, 1.82x the servo's stall instead of
    2.74x.
    """
    st = yaml.safe_load(STANCE.read_text())
    # stance.yaml names joints fr_hip; the model calls them frhip.
    pose = {name.replace("_", ""): float(deg) * 3.141592653589793 / 180.0
            for name, deg in st["angles_deg"].items()}
    return pose, float(st["trunk_height_m"])


def _servo() -> dict:
    return yaml.safe_load(CONFIG.read_text())["servo"]


# ---------------------------------------------------------------------------
# rewards that mjlab does not already have
# ---------------------------------------------------------------------------


def base_height(env, target: float, std: float, asset_cfg: SceneEntityCfg = ROBOT):
    """1.0 at the target ride height, falling off over `std` metres."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-torch.square(height - target) / (std * std))


def base_still(env, std: float, asset_cfg: SceneEntityCfg = ROBOT):
    """1.0 when the trunk is not moving. Standing still means still."""
    asset = env.scene[asset_cfg.name]
    speed = torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1)
    return torch.exp(-speed / (std * std))


def base_height_obs(env, asset_cfg: SceneEntityCfg = ROBOT):
    asset = env.scene[asset_cfg.name]
    return (asset.data.root_link_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3])


# ---------------------------------------------------------------------------


def _robot_cfg() -> EntityCfg:
    pose, height = _stance()

    def spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(MJCF))

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, height),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=pose,
            joint_vel={".*": 0.0},
        ),
        spec_fn=spec,
        articulation=EntityArticulationInfoCfg(
            # The XML's twelve <position> actuators are used as they are: the
            # 1.96 N-m ceiling and the gains all live in sim/models/gray.xml,
            # written there by tools/make_mjcf.py from gray/config/robot.yaml.
            actuators=(XmlActuatorCfg(target_names_expr=(".*",)),),
            soft_joint_pos_limit_factor=0.95,
        ),
    )


def stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    servo = _servo()
    _, height = _stance()

    # What the robot can actually sense: the IMU, and now the potentiometers.
    actor_terms = {
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "last_action": ObservationTermCfg(func=mdp.last_action),
    }
    # The critic is thrown away after training, so it may see things the robot
    # cannot - its own true speed and height. That makes the value estimate far
    # better without the policy ever depending on it.
    critic_terms = {
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "base_height": ObservationTermCfg(func=base_height_obs),
    }

    rewards = {
        # What we want.
        "height": RewardTermCfg(func=base_height, weight=2.0,
                                params={"target": height, "std": 0.03}),
        "still": RewardTermCfg(func=base_still, weight=1.0, params={"std": 0.25}),
        "posture": RewardTermCfg(func=mdp.posture, weight=1.5,
                                 params={"asset_cfg": ALL_JOINTS, "std": {".*": 0.35}}),
        "alive": RewardTermCfg(func=mdp.is_alive, weight=0.5),
        # What we do not. Named for the thing being punished, so the dashboard's
        # "loses points for" column reads as what it is - "upright" in a penalty
        # column reads backwards.
        "tilt": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-2.0),
        "fell_over": RewardTermCfg(func=mdp.is_terminated, weight=-20.0),
        "effort": RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.0002),
        "joint_speed": RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001),
        # Smooth commands are the single biggest factor in whether a policy
        # trained in simulation survives contact with a real servo.
        "twitching": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "end_stops": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "tipped_over": TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": 0.8}),
        "collapsed": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": height * 0.55}),
    }

    events = {
        "reset": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
        # Start each attempt slightly off, so the policy learns to correct rather
        # than to memorise one pose it was handed.
        "nudge_pose": EventTermCfg(
            func=mdp.reset_joints_by_offset, mode="reset",
            params={"position_range": (-0.05, 0.05),
                    "velocity_range": (-0.05, 0.05),
                    "asset_cfg": ALL_JOINTS}),
        "nudge_base": EventTermCfg(
            func=mdp.reset_root_state_uniform, mode="reset",
            params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01),
                                   "yaw": (-0.1, 0.1)},
                    "velocity_range": {}}),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": _robot_cfg()},
            # Four foot contacts, the same thing a switch in each foot would
            # report on the real robot. Free in simulation - MuJoCo already
            # knows which geoms are touching - and it needs nothing added to
            # the CAD. track_air_time also gives how long each foot has been
            # up or down, which is what every gait reward is built on.
            sensors=(ContactSensorCfg(
                name="feet",
                primary=ContactMatch(mode="geom", pattern=(".*_calf",),
                                     entity="robot"),
                fields=("found", "force"),
                track_air_time=True,
            ),),
            num_envs=4096,
            env_spacing=1.0,
        ),
        observations={
            "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
            "critic": ObservationGroupCfg(critic_terms),
        },
        actions={
            # The policy nudges the stance, it does not invent a pose. A quarter
            # of a radian either way is plenty to stand, and it means a
            # half-trained policy degrades toward a stance that already works.
            "joints": JointPositionActionCfg(
                entity_name="robot", actuator_names=(".*",),
                scale=0.25, use_default_offset=True),
        },
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot", body_name="base_link",
            distance=1.2, elevation=-12.0, azimuth=125.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=1.0 / (servo["control_hz"] * 4),
                             iterations=10, ls_iterations=20),
        ),
        # 50 Hz control. Not negotiable - it is the servo's PWM period.
        decimation=4,
        episode_length_s=10.0,
    )

    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False
        cfg.scene.num_envs = 16
    return cfg


def stand_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        max_iterations=600,
        save_interval=25,
        experiment_name="gray_stand",
        logger="tensorboard",   # wandb needs a login this project does not have
        # The actor has to be stochastic: PPO learns from the probability it
        # assigned to the action it took, and a deterministic model has none.
        # Without this the run dies on the first step with
        # "'NoneType' object has no attribute 'log_prob'".
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution",
                              "init_std": 1.0, "std_type": "scalar"},
        ),
        # The critic only predicts a number, so it stays deterministic.
        critic=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu",
                             obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.005,
            desired_kl=0.01,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            clip_param=0.2,
        ),
    )
