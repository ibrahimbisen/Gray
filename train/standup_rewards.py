"""Rewards for Stage 1: get off the floor and sit up.

Separate from train/rewards.py because those terms are all about WALKING - tracking a
commanded velocity, keeping feet planted, not drifting off line. Sitting up is a
different problem: move from one known pose to another and stay there.

THE FREEZE ARITHMETIC, which is the thing that has already cost this project two runs.
At the start of training the policy outputs near-random small actions and the robot is
lying flat at 42 mm. If doing nothing scores better than trying, it will learn to do
nothing, and it will look like a bug in the physics rather than a bug in the scoring.

MEASURED, 64 robots x 200 steps, episode sums. The first version of this file FAILED
this test and the numbers below are why every weight here is what it is:

                     lying still     moving randomly
    rise                   +0.11               +2.12
    posture                +2.55               +1.78     <- paid for lying there
    upright                +2.00               +1.78     <- paid for lying there
    joint_acc              -0.04               -3.98     <- 100x too harsh
    TOTAL                  +4.61               +1.56

Doing nothing won by 3x. A run started on that would have learned to lie on the floor
and it would have looked like a physics fault rather than a scoring fault. Two things
were wrong: `posture` was written as distance-to-target, which pays a large constant
for merely being in the start pose, and the smoothness penalty was sized for a walking
gait rather than for a robot heaving itself off the ground.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Where the robot starts and where it is going, in metres of trunk height. Both are
# SETTLED values measured in simulation, not commanded ones: every pose sags 14-19 mm
# under the robot's 1.625 kg, so scoring against a commanded height would be scoring
# against a number the robot can never reach. See reference/POSES.md.
RESTING_H = 0.042
SITTING_H = 0.110


class RiseToSitting(ManagerTermBase):
  """Fraction of the way from lying to sitting, by trunk height.

  LINEAR IN HEIGHT, not an exponential around the target. An exponential is flat far
  from its centre, and at the start of training the robot is as far from the target as
  it will ever get - which is exactly where it needs slope. Three terms in
  train/gray_env.py were written as exponentials with a std sized against the target
  and each one measured dead; the comments there record it.

  Clamped at 1.0 so overshooting past sitting earns nothing extra. Standing up further
  is Stage 2's job and paying for it here would teach the robot to skip the pose it is
  being asked to reach.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    self.cfg = cfg

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    h = env.scene["robot"].data.root_link_pos_w[:, 2]
    frac = (h - RESTING_H) / (SITTING_H - RESTING_H)
    return frac.clamp(0.0, 1.0)


class PostureMatch(ManagerTermBase):
  """How much of the joint-space distance from RESTING to SITTING has been closed.

  A FRACTION OF PROGRESS, NOT A DISTANCE TO TARGET, and that rewrite is the whole
  point of this class. Written the obvious way - exp(-(distance to sitting)^2) - it
  pays 0.43 for a robot lying perfectly still in the start pose, because resting is
  already fairly close to sitting in joint space. Measured over 200 steps that came to
  +2.55 for doing nothing against +1.78 for moving, so the term actively paid the
  robot to stay where it was. Combined with the other terms, lying still beat moving
  4.61 to 1.56 and the run would have learned to do nothing.

  Written as progress it is 0.0 at the start pose by construction, so it can only ever
  be earned by moving toward sitting.

  Height alone is not enough to define sitting - a robot can raise its trunk by
  flailing one leg - so this pays for the SHAPE and RiseToSitting pays for the height.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    self.cfg = cfg
    names = env.scene["robot"].joint_names
    target, start = cfg.params["target"], cfg.params["start"]
    missing = [n for n in names if n not in target or n not in start]
    if missing:
      raise KeyError(f"pose is missing angles for {missing}")
    self._target = torch.tensor([target[n] for n in names],
                                device=env.device, dtype=torch.float)
    self._start = torch.tensor([start[n] for n in names],
                               device=env.device, dtype=torch.float)
    # How far apart the two poses are, so progress is a fraction of a real distance
    # rather than of an invented scale.
    self._span = float((self._start - self._target).square().mean().sqrt())

  def __call__(self, env: ManagerBasedRlEnv, target: dict,
               start: dict) -> torch.Tensor:
    q = env.scene["robot"].data.joint_pos
    left = (q - self._target).square().mean(dim=1).sqrt()
    return (1.0 - left / self._span).clamp(0.0, 1.0)


class HoldSitting(ManagerTermBase):
  """Paid only while the robot is BOTH near sitting height and nearly still.

  Reaching the pose and falling straight back out of it is not sitting up. This term
  is what separates arriving from staying, and it is why the episode does not end on
  arrival: an episode that ends the moment the target is touched cannot tell the
  difference.

  The stillness threshold is 0.15 m/s on the trunk. Measured: a settled robot reads
  under 0.002 m/s, and the transient of arriving reads 0.3-0.9 m/s, so 0.15 sits
  clearly between the two.
  """

  _NEAR_M = 0.020
  _STILL_MS = 0.15

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    self.cfg = cfg

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    data = env.scene["robot"].data
    h = data.root_link_pos_w[:, 2]
    near = (h - SITTING_H).abs() < self._NEAR_M
    still = data.root_link_lin_vel_w.norm(dim=1) < self._STILL_MS
    return (near & still).float()
