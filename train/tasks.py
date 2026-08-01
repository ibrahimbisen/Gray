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
        # obliterate it. Starting at 0.25 means the initial policy is the Phase 2
        # gait plus noise of a couple of degrees, which is the whole premise: begin
        # from something that already walks.
        "init_std": 0.25,
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
      entropy_coef=0.005,
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
