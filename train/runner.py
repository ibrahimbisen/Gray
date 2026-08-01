"""Checkpoint export for Gray, including ONNX metadata the Pi will need.

mjlab's stock `get_base_metadata` is hardcoded to an action term literally named
"joint_pos" that must be a `JointPositionAction`:

    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, JointPositionAction)

Gray's action term is named "residual" and is a `ResidualGaitAction`, so that raised
KeyError('joint_pos') and the runner swallowed it as
"[WARN] ONNX export failed (training continues)". The .onnx weights were still
written - only the metadata was missing, which is easy to miss and would have been
discovered in Phase 5 with a policy nobody could interpret.

The metadata matters more here than it does for a stock task, because Gray's policy
does NOT emit joint targets. It emits a *residual*, and reconstructing a joint command
on the Raspberry Pi requires knowing that, plus the gait it is a residual of. So this
records the residual limit, the gait parameters, and the observation layout, alongside
the usual joint names and gains.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner

from train.residual_action import ResidualGaitAction


def gray_metadata(env: ManagerBasedRlEnv, run_name: str = "local") -> dict:
  """Everything a deployment needs to reproduce this policy off-line."""
  robot: Entity = env.scene["robot"]

  action = env.action_manager.get_term("residual")
  assert isinstance(action, ResidualGaitAction)

  # Actuator ids in the model's natural joint order, so gains line up with joint_names.
  joint_to_ctrl = {
    a.target.split("/")[-1]: a.id for a in robot.spec.actuators
  }
  ctrl_ids = [joint_to_ctrl[j] for j in robot.joint_names if j in joint_to_ctrl]
  stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
  damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]

  obs_names = env.observation_manager.active_terms["actor"]
  scales: list = []
  clips: list = []
  flatten: list = []
  history: list = []
  for term in obs_names:
    cfg = env.observation_manager.get_term_cfg("actor", term)
    scale = cfg.scale
    if scale is None:
      scales.append(1.0)
    else:
      scales.append(scale.cpu().tolist() if isinstance(scale, torch.Tensor) else scale)
    clips.append([float("-inf"), float("inf")] if cfg.clip is None else list(cfg.clip))
    flatten.append(cfg.flatten_history_dim)
    history.append(cfg.history_length)

  return {
    "run_path": run_name,
    "joint_names": list(robot.joint_names),
    "joint_stiffness": stiffness.tolist(),
    "joint_damping": damping.tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": list(env.command_manager.active_terms),
    "observation_names": obs_names,
    "observation_terms_scale": scales,
    "observation_terms_flatten_history_dim": flatten,
    "observation_terms_history_length": history,
    "observation_terms_clip": clips,
    # --- Gray-specific: the network output is NOT a joint target ---
    "action_type": "residual_on_bezier_gait",
    "action_scale": action.cfg.max_residual,
    "residual_limit_rad": action.cfg.max_residual,
    "gait_pattern": action.cfg.gait_pattern,
    "gait_period_s": action.cfg.period,
    "speed_at_unit_stride_ms": action.cfg.speed_at_unit_stride,
    "control_hz": 1.0 / float(env.step_dt),
    "joint_order": list(action.target_names),
  }


class GrayOnPolicyRunner(MjlabOnPolicyRunner):
  """Standard runner plus an ONNX export that knows what Gray's action means."""

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      attach_metadata_to_onnx(str(onnx_path), gray_metadata(self.env.unwrapped))
    except Exception as exc:  # never let export kill a multi-hour run
      print(f"[WARN] ONNX export failed (training continues): {exc!r}")
