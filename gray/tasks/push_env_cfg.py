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

# How hard the shoves are, as an instant change in trunk speed. On 2.379 kg,
# 1.2 m/s is about 2.9 N-s - a proper shove rather than a nudge.
PUSH_MS = (0.4, 1.2)
PUSH_SPIN = (-1.5, 1.5)      # rad/s about the vertical, either way
PUSH_EVERY_S = (2.0, 4.0)

FEET = SceneEntityCfg("robot", geom_names=(".*calf.*",))
FOOT_SITES = SceneEntityCfg("robot", site_names=(".*_foot",))

PUSH_NOTES = {
    "spinning": "Trunk rotating. A shove usually sets the robot turning as well as "
                "sliding, and killing that rotation is most of recovering from it. "
                "The stand task had no term for this because nothing ever span it.",
    "skidding": "A foot sliding along the floor while it is carrying weight. Nothing "
                "stopped the robot recovering by scuffing its feet across the ground, "
                "and that does not survive contact with a real floor: the simulator's "
                "friction is a guess, and a policy that leans on sliding is leaning "
                "on the guess.",
    "foot_lift": "Picking a foot up and putting it down. Once sliding is expensive, "
                 "the only way left to move a foot is to lift it - this pays for "
                 "doing that properly rather than dragging.",
}


def foot_slip(env, sensor_name: str = "feet", asset_cfg=FOOT_SITES):
    """How fast the feet are sliding while they are actually touching the ground.

    Contact comes from the contact sensor rather than being inferred from height.
    A height threshold is a guess that gets it wrong exactly when it matters -
    a foot skimming a millimetre above the floor is not carrying weight, and one
    pressed into a slope can be higher than the threshold while fully loaded.
    """
    asset = env.scene[asset_cfg.name]
    contact = env.scene[sensor_name]
    vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    down = (contact.data.found > 0).float()
    return torch.sum(torch.sum(torch.square(vel), dim=-1) * down, dim=1)


def foot_lift(env, target: float = 0.04, asset_cfg=FOOT_SITES):
    """Reward a foot that is genuinely off the ground, up to a sensible height.

    Capped on purpose. Rewarding height without a ceiling buys a robot that
    stands on three legs waving the fourth, which scores well and is useless.
    """
    asset = env.scene[asset_cfg.name]
    height = (asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
              - env.scene.env_origins[:, 2].unsqueeze(1))
    return torch.sum(torch.clamp(height, max=target), dim=1) / target


def trunk_spin(env, asset_cfg=ROBOT):
    """How fast the trunk is rotating, however it was set off."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)


def shove_from_any_angle(env, env_ids, speed_range, spin_range, asset_cfg=ROBOT):
    """Shove the trunk in a direction picked uniformly from the whole circle.

    Sampling x and y independently, which is what mjlab's own push event does,
    draws from a SQUARE: a diagonal shove comes out 1.41 times harder than a
    sideways one, and the robot gets pushed towards its corners more often than
    along its axes. Picking an angle and a magnitude separately gives every
    direction the same weight and the same range of strengths.
    """
    from mjlab.envs.mdp.events import resolve_env_ids  # noqa: PLC0415

    env_ids = resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)

    angle = torch.rand(n, device=env.device) * (2 * torch.pi)
    speed = (torch.rand(n, device=env.device)
             * (speed_range[1] - speed_range[0]) + speed_range[0])
    spin = (torch.rand(n, device=env.device)
            * (spin_range[1] - spin_range[0]) + spin_range[0])

    vel = asset.data.root_link_vel_w[env_ids].clone()
    vel[:, 0] += speed * torch.cos(angle)
    vel[:, 1] += speed * torch.sin(angle)
    vel[:, 5] += spin
    asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


def push_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = stand_env_cfg(play=play)

    # A shove every two to four seconds, from a direction drawn uniformly from
    # the whole circle, with a spin on top.
    cfg.events["shove"] = EventTermCfg(
        func=shove_from_any_angle,
        mode="interval",
        interval_range_s=PUSH_EVERY_S,
        params={"speed_range": PUSH_MS, "spin_range": PUSH_SPIN},
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

    # Scuffing a foot along the floor is the cheapest way to recover from a shove
    # and the one least likely to survive contact with a real floor. Make it
    # expensive and stepping becomes the cheaper option on its own.
    cfg.rewards["skidding"] = RewardTermCfg(func=foot_slip, weight=-2.0)
    cfg.rewards["foot_lift"] = RewardTermCfg(func=foot_lift, weight=0.3,
                                             params={"target": 0.04})

    # The stance task wanted the robot glued to one pose. Recovering by stepping
    # means leaving it, so stop paying so much for staying put.
    cfg.rewards["posture"].weight = 0.6
    cfg.rewards["still"].weight = 0.5
    cfg.rewards["joint_speed"].weight = -0.0002

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
