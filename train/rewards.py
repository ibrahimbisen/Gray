"""Velocity-tracking rewards that work on a robot this small and this slow.

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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# One full gait cycle. Long enough to average a stride, short enough that the reward
# still reacts within an episode.
DEFAULT_TAU_S = 0.6


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

  Commanded yaw is always zero (see train/gray_env.py), so in practice this is what
  pays for walking STRAIGHT - absorbing the few-mm per-leg CAD asymmetry that Phase 2
  could not tune out.
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
