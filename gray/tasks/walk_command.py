"""The velocity command, with a straight-line share that works in both directions.

mjlab's `rel_forward_envs` was doing something worth reading carefully:

    self.vel_command_b[fwd_ids, 0] = self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
    self.vel_command_b[fwd_ids, 1] = 0.0
    self.vel_command_b[fwd_ids, 2] = 0.0

Zeroing sideways and turn is exactly what is wanted - a straight line is a line
with nothing else on it, and the two penalties that hold a line only apply there.
The problem is the first line. It takes the ABSOLUTE VALUE and then clamps it up
to 0.3 m/s.

WHAT THAT COST, ON THE OLD BOX. WALK_SPEED was (0.15, 0.35) and
rel_forward_envs was 0.8. So four attempts in five had their speed forced to at
least 0.3, and three quarters of those landed exactly ON 0.3 because that is
where the clamp puts everything below it. The policy did not train across
0.15-0.35 m/s at all. It trained at 0.30-0.35, with a spike at 0.30 - and the
speed-tracking bar has been failing at 0.071 against 0.05.

WHAT IT WOULD COST ON THE NEW BOX. `abs()` erases the sign. With WALK_SPEED
widened to (-0.35, 0.35), 80% of commands would have been flipped back to
positive - so widening the range would have bought almost nothing, and the
measurement afterwards would have said backward still does not work.

This subclass keeps the good half and drops the clamp: sideways and turn are
zeroed, and the forward speed is left exactly as it was drawn, sign and all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class StraightLineVelocityCommand(UniformVelocityCommand):
    """A share of attempts get a straight line, forwards OR backwards."""

    cfg: StraightLineVelocityCommandCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if len(env_ids) == 0 or self.cfg.rel_straight_envs <= 0.0:
            return
        # Undo the parent's forward-only handling for these ids, then redo it
        # without the abs() and the clamp. Cheaper and far less brittle than
        # copying the whole method to change two lines of it.
        pick = torch.rand(len(env_ids), device=self.device) <= self.cfg.rel_straight_envs
        ids = env_ids[pick]
        if len(ids) == 0:
            return
        # Redraw the forward speed, because the parent may already have clamped
        # it. Anything at or below MOVING is nudged clear of it - a "straight
        # line" command at 0.01 m/s is a standing command wearing the wrong hat,
        # and the straightness terms would score it as drift.
        lo, hi = self.cfg.ranges.lin_vel_x
        v = torch.empty(len(ids), device=self.device).uniform_(lo, hi)
        floor = self.cfg.straight_min_speed
        v = torch.where(v.abs() < floor, torch.sign(v + 1e-9) * floor, v)
        self.vel_command_b[ids, 0] = v
        self.vel_command_b[ids, 1] = 0.0
        self.vel_command_b[ids, 2] = 0.0
        self.vel_command_w[ids] = self.vel_command_b[ids]
        self.is_forward_env[ids] = False


@dataclass
class StraightLineVelocityCommandCfg(UniformVelocityCommandCfg):
    """`rel_forward_envs` is left at 0. Use `rel_straight_envs` instead."""

    rel_straight_envs: float = 0.0
    # Below this the command is a stop, not a direction, and the terms that hold
    # a line switch off anyway. Matches walk_env_cfg's MOVING with room to spare.
    straight_min_speed: float = 0.10

    def build(self, env: ManagerBasedRlEnv) -> StraightLineVelocityCommand:
        return StraightLineVelocityCommand(self, env)
