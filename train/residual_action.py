"""The residual action: policy output is a correction to the Phase 2 gait.

    joint target = classical_gait(phase, stride) + clip(policy_output, +/-0.2 rad)

This is the load-bearing idea of Phase 3, following D2-GMBC (arXiv 2010.12070). The
policy never chooses joint angles outright. It is handed a gait that already walks and
may only nudge it, which buys three things:

  * it converges in hours instead of days, because it starts from something that works
  * a bad or half-trained policy degrades toward the classical gait rather than flailing
  * whatever the network fails to learn, there is still a walking robot underneath

The 0.2 rad ceiling is the whole safety argument, so it is enforced twice: once on the
policy output, and again against the joint limits after the sum.

The nominal gait comes from train.gait_table - a precomputed tensor rather than a live
call into gray.gait, which would otherwise run thousands of Python IK solves per
control tick. See that module for why the precomputation is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

from gray.gait import GaitParams
from train.gait_table import GaitTable
from train.gray_robot import JOINT_ORDER

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ResidualGaitActionCfg(BaseActionCfg):
  """Configuration for the residual gait action."""

  max_residual: float = 0.2
  """Hard ceiling on the correction, in radians, per joint."""

  period: float = GaitParams.period
  """Gait cycle length in seconds. Keep the control rate an integer multiple of the
  phase grid (see gait_table.N_PHASE) so lookups stay exact."""

  speed_at_unit_stride: float = 0.0529
  """Forward speed in m/s that the Phase 2 gait reaches at stride scale 1.0, measured
  by scripts/walk.py. Converts a commanded velocity into a stride scale."""

  command_name: str = "twist"
  """Command term supplying the velocity to track."""

  gait_pattern: str = "crawl"
  """Which Phase 2 gait to sit on top of. Crawl keeps three feet down at all times."""

  def __post_init__(self) -> None:
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> ResidualGaitAction:
    return ResidualGaitAction(self, env)


class ResidualGaitAction(BaseAction):
  """Adds a bounded, learned correction to the classical gait."""

  cfg: ResidualGaitActionCfg

  def __init__(self, cfg: ResidualGaitActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg=cfg, env=env)

    params = GaitParams(pattern=cfg.gait_pattern, period=cfg.period)
    self._table = GaitTable.build(self.device, params)

    # gait_table emits columns in gray.kinematics order; mjlab resolved its own joint
    # order from the model. Permute once here rather than per step.
    perm = [JOINT_ORDER.index(name) for name in self._target_names]
    self._perm = torch.tensor(perm, device=self.device, dtype=torch.long)

    self._phase = torch.zeros(self.num_envs, device=self.device)
    self._nominal = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._dt = float(env.step_dt)

  # Exposed so observation terms can report what the gait is doing without
  # recomputing it. See train.gray_env.gait_phase / gait_nominal.

  @property
  def phase(self) -> torch.Tensor:
    """Gait phase in [0, 1), one per environment."""
    return self._phase

  @property
  def nominal(self) -> torch.Tensor:
    """Classical-gait joint targets before the residual, in mjlab joint order."""
    return self._nominal

  def process_actions(self, actions: torch.Tensor) -> None:
    """Called once per control step - this is where the gait clock advances."""
    self._raw_actions[:] = actions

    self._phase = (self._phase + self._dt / self.cfg.period) % 1.0

    command = self._env.command_manager.get_command(self.cfg.command_name)
    stride = command[:, 0] / self.cfg.speed_at_unit_stride

    self._nominal = self._table(self._phase, stride)[:, self._perm]

    residual = torch.clamp(
      actions * self.cfg.max_residual, -self.cfg.max_residual, self.cfg.max_residual
    )

    # Second clamp: the residual is bounded, but nominal + residual still has to be a
    # pose the servo can actually hold.
    limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._processed_actions = torch.clamp(
      self._nominal + residual, limits[..., 0], limits[..., 1]
    )

  def apply_actions(self) -> None:
    """Called every physics substep; the target is already computed."""
    # encoder_bias here is not an encoder at all - Gray has no feedback. It is the
    # per-servo zero-offset error from horn spline misalignment, which is real,
    # permanent, and unmeasurable without disassembly. Subtracting it means the
    # policy trains against servos that quietly disagree about where zero is.
    bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      self._processed_actions - bias, joint_ids=self._target_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    # Start each episode somewhere different in the cycle, so a batch is not all
    # lifting the same foot on the same step.
    count = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
    self._phase[env_ids] = torch.rand(count, device=self.device)


##
# Observation terms backed by the action above.
##


def gait_phase(env: ManagerBasedRlEnv, action_name: str = "residual") -> torch.Tensor:
  """Where in the gait cycle we are, as (cos, sin).

  Fed to the policy as a pair rather than a raw number so that phase 0.99 and phase
  0.01 look adjacent, which they are - a raw value would present a cliff between them.
  """
  phase = env.action_manager.get_term(action_name).phase * 2.0 * math.pi
  return torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)


def gait_nominal(env: ManagerBasedRlEnv, action_name: str = "residual") -> torch.Tensor:
  """The classical gait's current joint targets.

  This is the one joint-space quantity the policy is allowed to see. It is a
  *command*, not a measurement - the real servos can report nothing - so it stays
  available on hardware.
  """
  return env.action_manager.get_term(action_name).nominal
