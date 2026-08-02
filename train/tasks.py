"""Register Gray's Phase 3 task with mjlab.

Importing this module is what makes ``Gray-Residual-Flat`` visible to mjlab's train
and play entry points - the registry is a plain module-level dict, so it only has to
happen before the CLI reads it. scripts/train_residual.py does exactly that.
"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.registry import register_mjlab_task

from train.gray_env import gray_flat_env_cfg
from train.standup_env import gray_standup_env_cfg
from train.runner import GrayOnPolicyRunner

TASK_ID = "Gray-Residual-Flat"

# 4070 Ti, 12 GB. Chosen by measurement, not guesswork - throughput plateaus here:
#
#    4096 envs -> 111,550 steps/s   iter 0.88s
#    8192 envs -> 153,179 steps/s   iter 1.29s
#   16384 envs -> 174,646 steps/s   iter 2.26s   <- knee of the curve
#   24576 envs -> 175,028 steps/s   iter 3.38s   <- no faster, 50% slower per round
#
# Past 16384 the GPU is saturated, so extra environments buy nothing and only make
# each iteration longer. Gray is a 12-DOF robot with one contact pair per foot, which
# is why it takes this many environments to fill a 4070 Ti at all.
#
# The win is sample quality, not wall clock: each PPO update now averages gradients
# over 16384 x 24 = ~393k transitions instead of ~98k, so the policy sees four times
# the experience per update and the gradient is markedly less noisy. Drop this if you
# ever hit an out-of-memory error.
NUM_ENVS = 16384


def gray_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # Deliberately small. This network is the one artefact of Phase 3 that has to
      # run on a Raspberry Pi 4 at 50 Hz in Phase 5, and the task - nudging an
      # existing gait - does not need capacity. ~40k parameters is nothing on the
      # 4070 Ti and comfortably real-time on the Pi.
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # The default of 1.0 would emit residuals at the full +/-0.2 rad ceiling on
        # the very first step, which does not perturb the classical gait so much as
        # obliterate it. The premise of residual RL is to begin from something that
        # already walks, so this has to start small.
        #
        # 0.25 (residual noise ~0.05 rad = 2.9 deg on every joint, resampled every
        # tick) was still too much: the first run measured 568 mm at round 0 against
        # the gait's 675 mm, so a sixth of the gait was destroyed before learning
        # began, and it never recovered. 0.10 is ~1.1 deg, which leaves the initial
        # policy essentially the Phase 2 gait.
        #
        # DELIBERATELY UNCHANGED. The last run's noise problem was not where the
        # distribution STARTED - 0.101 at round 0 is exactly this value, so the
        # initialisation was doing its job - but that it grew from there and never came
        # back: 0.191 by round 300, 0.286 by 700, 0.313 by 1150, never turning over.
        # Lowering init_std would not have touched that. The fix is entropy_coef below
        # and the pre-clip action penalty in gray_env.py.
        "init_std": 0.10,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      # The critic is discarded after training, so it can afford to be larger.
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # STOPPING THE NOISE RUNAWAY. At 0.005 the policy's action distribution widened
      # monotonically for the whole of the last run - Policy/mean_std 0.101 -> 0.191
      # (r300) -> 0.286 (r700) -> 0.313 (r1150) - and never turned over. The mechanism
      # is specific to a CLIPPED action space: ResidualGaitAction discards anything
      # past |action| = 1.0, a clipped sample contributes no gradient, so once the
      # distribution has spread past the clip there is nothing pulling it back while
      # the entropy bonus keeps pushing it out. The fraction of actions being clipped
      # went 1.7% (r500) -> 20.2% (r950) -> 27.9% (r1100) unnoticed, because nothing
      # logged it. gray_env.py now logs it as Episode_Metrics/clip_fraction and adds a
      # small penalty on the pre-clip magnitude to restore the missing restoring force.
      #
      # THE INTENDED FIX WAS AN ANNEAL, 0.005 -> 0.0005 over the first third of the
      # run. RSL-RL 5.4.0 CANNOT DO THAT, and the API to do it does not exist rather
      # than being merely awkward: in .venv/Lib/site-packages/rsl_rl/algorithms/ppo.py
      # entropy_coef arrives as a float (l.45), is stored once (l.105) and is read once
      # in the loss (l.278). Nothing ever writes back to it, there is no callback and
      # no schedule field, and mjlab's RslRlPpoAlgorithmCfg exposes only the scalar.
      # (The `schedule="adaptive"` two lines down is the LEARNING RATE schedule and has
      # nothing to do with entropy.) Annealing would mean mutating
      # runner.alg.entropy_coef from a per-iteration hook in train/runner.py, which
      # this change does not own.
      #
      # So: a lower FIXED value, chosen nearer the anneal's endpoint than its start.
      # 0.001 is a 5x cut. Not the full 10x to 0.0005, because the run genuinely does
      # need exploration early and a fixed coefficient has to serve the whole run
      # rather than just its tail. If Policy/mean_std still fails to turn over by
      # ~round 300, drop this to 0.0005 rather than reaching for anything else -
      # mean_std and clip_fraction are now both visible, so this is a one-look decision.
      entropy_coef=0.001,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="gray_residual",
    # mjlab defaults to Weights & Biases, which needs an online account and an API
    # key. TensorBoard writes to logs/rsl_rl/ locally and needs neither.
    logger="tensorboard",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=3000,
  )


def _train_cfg():
  cfg = gray_flat_env_cfg()
  cfg.scene.num_envs = NUM_ENVS
  return cfg


def _play_cfg():
  cfg = gray_flat_env_cfg(play=True)
  cfg.scene.num_envs = 16
  return cfg


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=_train_cfg(),
  play_env_cfg=_play_cfg(),
  rl_cfg=gray_ppo_runner_cfg(),
  runner_cls=GrayOnPolicyRunner,
)


##
# Stage 1: get off the floor. A separate task, additive - Gray-Residual-Flat above is
# untouched and stays the comparison.
##

STANDUP_TASK_ID = "Gray-Standup"

# Fewer robots than the walking task's 16384, on purpose. Sitting up is an 8 s episode
# against walking's 12 s, and the whole point of this task is a FAST answer to whether a
# blind policy can learn a whole-body move at all. 8192 still gathers 196k moments a
# round, which is far more than the 51k-parameter actor needs per update.
STANDUP_NUM_ENVS = 8192


def gray_standup_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = gray_ppo_runner_cfg()
  cfg.experiment_name = "gray_standup"
  # A short, simple movement. 800 rounds is about 25 minutes and is enough to see the
  # answer; the walking task's 3000 exists because walking is hard, not because long
  # runs are good.
  cfg.max_iterations = 800
  return cfg


def _standup_train_cfg():
  cfg = gray_standup_env_cfg()
  cfg.scene.num_envs = STANDUP_NUM_ENVS
  return cfg


def _standup_play_cfg():
  cfg = gray_standup_env_cfg(play=True)
  cfg.scene.num_envs = 16
  return cfg


register_mjlab_task(
  task_id=STANDUP_TASK_ID,
  env_cfg=_standup_train_cfg(),
  play_env_cfg=_standup_play_cfg(),
  rl_cfg=gray_standup_runner_cfg(),
  runner_cls=GrayOnPolicyRunner,
)
