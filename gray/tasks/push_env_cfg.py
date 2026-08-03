"""Stage 2 - take a push.

Built on stage 1 rather than beside it: the same stance, the same rewards, the
same terminations. What is added is a world that will not hold still.

**Why this before walking.** Standing still can be solved without looking at
anything. A policy could ignore every sensor, replay one set of angles forever,
and nothing in the score would give it away. A shove breaks that - the robot has
to notice it moved and do something. So this is the first task that genuinely
uses the potentiometers, and it proves the observation space works before
anything harder rests on it.

Walking needs the same reflex. Every step is a disturbance the robot creates for
itself: weight shifts, a foot lands early, the trunk rocks. A policy that only
knows one still pose has no answer when that happens. The previous attempt hit
exactly this - the hand-written gait walked, but wandered about 140 mm, and the
note recorded that it needed active balance.

It is also where domain randomisation is cheap. Friction, mass, servo gains and
the ground itself can all be varied while the task is still simple, for minutes
of GPU time. Learning the same robustness during walking costs hours, and a fall
there has two possible causes - the gait or the balance - with no way to tell
which. This rules one of them out.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg

from gray.tasks.stand_env_cfg import ALL_JOINTS, ROBOT, stand_env_cfg, stand_ppo_cfg

# How hard the shoves are, as a change in trunk velocity. 0.6 m/s applied to
# 2.379 kg is about 1.4 N-s - a firm nudge with a finger, not a kick. The bar
# talks in impulse; this is the mass-independent equivalent mjlab provides.
PUSH_MS = 0.6
PUSH_EVERY_S = (2.0, 4.0)

FEET = SceneEntityCfg("robot", geom_names=(".*calf.*",))

PUSH_NOTES = {
    "spinning": "Trunk rotating. A shove usually sets the robot turning as well as "
                "sliding, and killing that rotation is most of recovering from it. "
                "The stand task had no term for this because nothing ever span it.",
}


def trunk_spin(env, asset_cfg=ROBOT):
    """How fast the trunk is rotating, however it was set off."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)


def push_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = stand_env_cfg(play=play)

    # A shove every two to four seconds, from any direction, including spin.
    # Applied as an instant change in trunk velocity, which is the cheap
    # mass-independent disturbance and the standard one for locomotion.
    cfg.events["shove"] = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=PUSH_EVERY_S,
        params={"velocity_range": {
            "x": (-PUSH_MS, PUSH_MS),
            "y": (-PUSH_MS, PUSH_MS),
            "yaw": (-1.0, 1.0),
        }},
    )

    # A world that is not the same every time. Each of these is a number we do
    # not actually know about the real robot, so the policy should not be allowed
    # to depend on any particular value of it.
    cfg.events["ground_grip"] = EventTermCfg(
        func=mdp.dr.geom_friction, mode="reset",
        params={"ranges": (0.4, 1.2), "operation": "abs", "asset_cfg": FEET})
    cfg.events["how_heavy"] = EventTermCfg(
        # Mass is a CAD number, not a scale reading, so train across being wrong
        # about it by +/-20%. pseudo_inertia rather than body_mass: body_mass
        # changes the mass and leaves the inertia tensor alone, which is a robot
        # that got heavier without getting harder to turn - not a thing. This
        # scales both, the way a density change actually would.
        func=mdp.dr.pseudo_inertia, mode="reset",
        params={"alpha_range": (-0.2, 0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=(".*",))})
    cfg.events["where_the_weight_is"] = EventTermCfg(
        # The battery and electronics are modelled where the CAD says they sit,
        # which is not necessarily where they end up once it is wired.
        func=mdp.dr.body_com_offset, mode="reset",
        params={"ranges": (-0.015, 0.015),
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))})
    cfg.events["servo_strength"] = EventTermCfg(
        # A hobby servo's internal gains are sealed and unpublished, so the ones
        # in robot.yaml are a guess. Train across a band around them.
        func=mdp.dr.pd_gains, mode="reset",
        params={"kp_range": (0.7, 1.3), "kd_range": (0.7, 1.3),
                "operation": "scale", "asset_cfg": ALL_JOINTS})
    cfg.events["gearbox_drag"] = EventTermCfg(
        func=mdp.dr.joint_friction, mode="reset",
        params={"ranges": (0.005, 0.03), "operation": "abs", "asset_cfg": ALL_JOINTS})

    # A shove sets the robot turning as well as sliding, and the stand task never
    # needed a term for that because nothing ever span it. The existing height,
    # tilt and still terms already pay for the rest of recovering.
    cfg.rewards["spinning"] = RewardTermCfg(func=trunk_spin, weight=-0.05)

    # Falling costs more here. The whole point is that being disturbed and
    # recovering beats being disturbed and going over.
    cfg.rewards["fell_over"].weight = -40.0

    # Long enough to be shoved several times in one attempt, so recovering once
    # is not enough.
    if not play:
        cfg.episode_length_s = 20.0
    return cfg


def push_ppo_cfg():
    cfg = stand_ppo_cfg()
    cfg.experiment_name = "gray_push"
    # Harder task, and the randomisation means every attempt is a different
    # world - it needs longer than standing did.
    cfg.max_iterations = 1500
    return cfg
