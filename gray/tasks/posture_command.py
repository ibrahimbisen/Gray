"""Three more numbers the robot is told: how tall to stand, and how to lean.

The velocity command says where the trunk should GO - forward, sideways, turn.
It says nothing about where the trunk should BE, so until now the robot rode at
one fixed height, level, and there was no way to ask for anything else. Nineteen
rows of the skill library were blocked on exactly that: sit, stand, crouch,
sneak, crawl, squeeze under, bow, stretch, shake off, lean, scan.

None of them is a new skill and none needs its own policy. They are one control
the command vector did not have. This adds it:

    height   how far off the ground to hold the trunk, in metres
    pitch    nose down NEGATIVE, nose up positive, in radians
    roll     right side down negative, in radians (measured 6 Aug 2026)

These two lines said "nose down positive" until 5 Aug 2026 and the code has
always done the opposite - `trunk_pitch_roll` below reads pitch as
atan2(-g_x, down), and tipping the nose forward swings gravity toward +x, so
nose down comes back negative. Measured in the sim to settle it: set the trunk
to a known 10 deg nose-down and it reports -0.175 rad.

The cost of the wrong comment was walk_env_cfg's POSE_PITCH, which was written
to match it and therefore ran backwards for two days - commanding up to 15 deg
of nose-down into 10 deg of available travel, and leaving 12 of the 20 deg of
nose-up unasked for. It also left the average commanded pitch 3.4 deg nose-down,
which is how it was eventually found: the owner noticed the robot walking with
its nose down in the checkpoint films. Nothing measured it, because
`error_pitch` is an absolute value and `upright` scores the lean the robot was
TOLD to hold.

RANGES ARE MEASURED, NOT CHOSEN. scripts/find_stance.py solves for the joint
angles that keep every foot on its own print while the trunk moves, and reports
where the legs run out of travel. Against the owner's stance, 3 Aug 2026:

    height   120 to 270 mm     holds throughout; 1.59x the servo at the lowest
    pitch    nose up 20 deg, nose DOWN only 10 deg
    roll     +/- 30 deg, and the sweep never found the limit

Pitch is lopsided because the stance already rakes the legs forward, which
spends some of the travel nose-down would need. That asymmetry is real geometry,
so the command range is asymmetric too rather than being squared off to the
smaller side and throwing away half the nose-up travel.

The ranges actually commanded sit INSIDE those limits, because a limit measured
standing still is not a limit while walking - a leg that is at its end stop has
nothing left to swing with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def trunk_pitch_roll(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """The trunk's pitch and roll, off gravity.

    Read from the gravity vector as the robot's own IMU sees it, not from a
    world-frame quaternion - because the real robot has an IMU and does not have
    a world frame. Level is (0, 0, -1); tipping forward swings it toward +x.

    Same measurement `upright` already scores, decomposed into two axes so each
    can be scored against its own commanded value.
    """
    g = robot.data.projected_gravity_b
    down = torch.clamp(-g[:, 2], min=1e-6)
    pitch = torch.atan2(-g[:, 0], down)
    roll = torch.atan2(g[:, 1], down)
    return pitch, roll


class PostureCommand(CommandTerm):
    """Height, pitch and roll, drawn uniformly and held for a few seconds."""

    cfg: PostureCommandCfg

    def __init__(self, cfg: PostureCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        # height, pitch, roll - in that order, and that order is the contract.
        # It is what the policy reads and what verify.py and drive.py index into.
        self.posture_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        r = self.cfg.ranges
        return (f"PostureCommand: height {r.height}, pitch {r.pitch}, "
                f"roll {r.roll}, {self.cfg.rel_nominal_envs:.0%} nominal")

    @property
    def command(self) -> torch.Tensor:
        return self.posture_command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        r = self.cfg.ranges
        draw = torch.empty(n, device=self.device)
        self.posture_command[env_ids, 0] = draw.uniform_(*r.height)
        self.posture_command[env_ids, 1] = draw.uniform_(*r.pitch)
        self.posture_command[env_ids, 2] = draw.uniform_(*r.roll)

        # A share are pinned to the plain standing posture. Without them, walking
        # normally is a lucky draw from a continuous range and is never actually
        # practised - the same reason rel_standing_envs exists on the velocity
        # command. This is the posture equivalent of "zero means stand".
        keep = int(self.cfg.rel_nominal_envs * n)
        if keep:
            pick = env_ids[torch.randperm(n, device=self.device)[:keep]]
            self.posture_command[pick, 0] = self.cfg.nominal_height
            self.posture_command[pick, 1:] = 0.0

    def _update_command(self) -> None:
        """Nothing to do. The command is held as drawn until it is resampled."""

    def _update_metrics(self) -> None:
        height = (self.robot.data.root_link_pos_w[:, 2]
                  - self._env.scene.env_origins[:, 2])
        pitch, roll = trunk_pitch_roll(self.robot)
        self.metrics["error_height"] = torch.abs(height - self.posture_command[:, 0])
        self.metrics["error_pitch"] = torch.abs(pitch - self.posture_command[:, 1])
        self.metrics["error_roll"] = torch.abs(roll - self.posture_command[:, 2])


@dataclass
class PostureCommandCfg(CommandTermCfg):
    entity_name: str = "robot"

    # The height that counts as "normal", and the share of attempts pinned to it
    # with the trunk level. Read off progress/stance/stance.yaml by the task, so
    # there is no second copy of the ride height to drift out of date.
    nominal_height: float = 0.2024
    rel_nominal_envs: float = 0.25

    @dataclass
    class Ranges:
        height: tuple[float, float]
        pitch: tuple[float, float]     # radians, nose down NEGATIVE
        roll: tuple[float, float]      # radians, right side down negative

    # These defaults carried the same swapped pitch signs as walk_env_cfg's
    # POSE_PITCH until 5 Aug 2026. The walk task overrides all three, so the
    # default was never what trained - but it is what anything that forgets to
    # override would get, which is how a fixed bug comes back.
    ranges: Ranges = field(
        default_factory=lambda: PostureCommandCfg.Ranges(
            height=(0.15, 0.25), pitch=(-0.14, 0.26), roll=(-0.35, 0.35)))

    def build(self, env: ManagerBasedRlEnv) -> PostureCommand:
        return PostureCommand(self, env)
