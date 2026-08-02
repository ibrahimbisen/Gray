#!/usr/bin/env python3
"""Film a stand-up policy and score it. The walking scorer cannot do this.

    python scripts/render_standup.py                    # newest checkpoint
    python scripts/render_standup.py --checkpoint 400   # a specific round
    python scripts/render_standup.py --all              # every saved checkpoint

WHY THIS IS A SEPARATE SCRIPT. scripts/make_progress_videos.py and train/evaluate.py
are built around WALKING: they hardcode logs/rsl_rl/gray_residual, drive the classical
gait, feed the policy 35 observations and score distance, drift and speed tracking. A
stand-up policy takes 18 observations and travels nowhere. Pointing the walking harness
at one would either crash on the shape mismatch or, worse, return a distance figure
that looked like a real measurement.

WHAT IT MEASURES, which is what stand-up actually means:

    peak height     the highest the trunk got
    final height    where it ended up, which is the one that matters - reaching
                    sitting and falling back out of it is not sitting up
    settled         whether it had stopped moving by the end
    upright         how level it was

The benchmark to beat is gray/standup.py, the hand-written sequence, measured at 60 of
60 successes on randomly built robots.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import csv

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_DIR = "logs/rsl_rl/gray_standup"
OUT_DIR = "progress/runs"
SITTING_MM = 110.0


def _newest_mtime(directory: str) -> float:
  newest = 0.0
  for root, _dirs, files in os.walk(directory):
    for name in files:
      try:
        newest = max(newest, os.path.getmtime(os.path.join(root, name)))
      except OSError:
        continue
  return newest


def newest_run() -> str:
  """The run written to most recently.

  BY MTIME, NOT BY NAME, so this agrees with the dashboard. Sorting by name picked a
  different run when two were live three seconds apart: clips landed under
  ..._03-33-59_standup while the page was displaying ..._03-33-56_standup, so the
  videos existed and the page correctly reported none.
  """
  runs = [r for r in glob.glob(os.path.join(LOG_DIR, "*")) if os.path.isdir(r)]
  if not runs:
    raise SystemExit(f"no stand-up runs in {LOG_DIR}")
  return max(runs, key=_newest_mtime)


def checkpoints(run_dir: str) -> list[tuple[int, str]]:
  out = []
  for path in glob.glob(os.path.join(run_dir, "model_*.pt")):
    m = re.search(r"model_(\d+)\.pt$", path)
    if m:
      out.append((int(m.group(1)), path))
  return sorted(out)


class Policy:
  """The trained actor, run on CPU. Deliberately rebuilt by hand from the saved
  weights rather than by importing rsl_rl: the shapes are in the checkpoint, and this
  keeps the renderer independent of the training stack."""

  def __init__(self, path: str):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["actor_state_dict"]
    self.mean = sd["obs_normalizer._mean"].numpy().astype(np.float64)
    self.std = sd["obs_normalizer._std"].numpy().astype(np.float64)
    self.layers = []
    i = 0
    while f"mlp.{i}.weight" in sd:
      self.layers.append((sd[f"mlp.{i}.weight"].numpy().astype(np.float64),
                          sd[f"mlp.{i}.bias"].numpy().astype(np.float64)))
      i += 2
    self.iteration = int(blob.get("iter", -1))

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    x = (obs - self.mean[0]) / np.maximum(self.std[0], 1e-2)
    for k, (w, b) in enumerate(self.layers):
      x = x @ w.T + b
      if k < len(self.layers) - 1:
        # elu, matching the trained network
        x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
    return x


def rollout(policy: Policy | None, seed: int = 0, seconds: float = 8.0,
            video: str | None = None, randomise: bool = False) -> dict:
  """One episode. `policy=None` runs gray/standup.py instead, as the benchmark."""
  import mujoco
  from gray.standup import JOINT_ORDER, StandUp

  rng = np.random.default_rng(seed)
  model = mujoco.MjModel.from_xml_path("sim/models/gray.xml")
  if randomise:
    for b in range(1, model.nbody):
      f = float(np.exp(rng.uniform(np.log(0.6), np.log(1.4))))
      model.body_mass[b] *= f
      model.body_inertia[b] *= f
    for g in range(model.ngeom):
      model.geom_friction[g, 0] = float(rng.uniform(0.4, 1.2))
    kp = float(rng.uniform(0.5, 4.0))
    model.actuator_gainprm[:, 0] *= kp
    model.actuator_biasprm[:, 1] *= kp
    model.dof_armature[6:] *= float(rng.uniform(0.35, 3.0))

  data = mujoco.MjData(model)
  seq = StandUp()
  qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
          for j in JOINT_ORDER]
  dadr = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
          for j in JOINT_ORDER]
  aid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
         for j in JOINT_ORDER]
  trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

  rest = seq.at(0.0)
  for i, q in enumerate(qadr):
    data.qpos[q] = rest[i] + rng.uniform(-0.02, 0.02)
  data.qpos[2] = 0.045
  mujoco.mj_forward(model, data)

  # The action term's scale and offset, so the policy's [-1,1] output means the same
  # joint angles here as it did in training.
  from train.standup_env import OWNER_LIMITS_DEG, _load_poses, _raw_limits
  resting, _ = _load_poses()
  limits = _raw_limits(resting)
  import math
  offset = np.array([resting[j] for j in JOINT_ORDER])
  scale = np.array([
    max(abs(math.radians(limits[j][1]) - resting[j]),
        abs(resting[j] - math.radians(limits[j][0])), 1e-3)
    for j in JOINT_ORDER
  ])

  renderer = frames = None
  if video:
    renderer = mujoco.Renderer(model, height=540, width=840)
    frames = []
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.azimuth, cam.elevation = 0.95, 125, -14
    cam.lookat[:] = [0, 0, 0.09]

  last_action = np.zeros(12)
  peak = 0.0
  t = 0.0
  step = 0
  while t < seconds:
    if policy is None:
      target = seq.at(t)
    else:
      # EXACTLY the actor observation, in the order train/standup_env.py builds it:
      # base_ang_vel(3), projected_gravity(3), last_action(12).
      R = data.xmat[trunk].reshape(3, 3)
      ang = R.T @ data.qvel[3:6]
      grav = R.T @ np.array([0.0, 0.0, -1.0])
      obs = np.concatenate([ang, grav, last_action])
      action = np.clip(policy(obs), -100.0, 100.0)
      last_action = action
      target = offset + scale * action
    for i, a in enumerate(aid):
      data.ctrl[a] = target[i]
    for _ in range(10):
      mujoco.mj_step(model, data)
    peak = max(peak, data.xpos[trunk][2] * 1000)
    if frames is not None and step % 2 == 0:
      renderer.update_scene(data, cam)
      frames.append(renderer.render())
    t += 0.02
    step += 1

  if frames:
    import imageio.v2 as iio
    os.makedirs(os.path.dirname(video), exist_ok=True)
    iio.mimsave(video, frames, fps=25, macro_block_size=1)

  h = data.xpos[trunk][2] * 1000
  return {
    "final_mm": float(h),
    "peak_mm": float(peak),
    "upright": float(data.xmat[trunk].reshape(3, 3)[2, 2]),
    "settled": bool(np.linalg.norm(data.qvel[:3]) * 1000 < 20),
    "reached": bool(abs(h - SITTING_MM) < 25),
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--checkpoint", type=int, default=None)
  ap.add_argument("--all", action="store_true")
  ap.add_argument("--trials", type=int, default=8)
  ap.add_argument("--randomise", action="store_true")
  args = ap.parse_args()

  run_dir = newest_run()
  run_name = os.path.basename(run_dir)
  found = checkpoints(run_dir)
  if not found:
    raise SystemExit(f"no checkpoints in {run_dir} yet")
  if args.checkpoint is not None:
    found = [c for c in found if c[0] == args.checkpoint] or found[-1:]
  elif not args.all:
    found = found[-1:]

  print(f"run {run_name}")
  print(f"target: sitting at {SITTING_MM:.0f} mm\n")

  base = [rollout(None, seed=s, randomise=args.randomise)
          for s in range(args.trials)]
  ok = sum(1 for r in base if r["reached"])
  print(f"{'hand-written script':22s} final {np.mean([r['final_mm'] for r in base]):6.1f} mm   "
        f"reached sitting {ok}/{args.trials}")

  rows = []
  for it, path in found:
    pol = Policy(path)
    vid = os.path.join(OUT_DIR, run_name, "videos", f"iter_{it:04d}.mp4")
    res = [rollout(pol, seed=s, video=vid if s == 0 else None,
                   randomise=args.randomise)
           for s in range(args.trials)]
    ok = sum(1 for r in res if r["reached"])
    final = float(np.mean([r["final_mm"] for r in res]))
    print(f"round {it:<16d} final {final:6.1f} mm   "
          f"peak {np.mean([r['peak_mm'] for r in res]):6.1f}   "
          f"reached sitting {ok}/{args.trials}   -> {vid}")
    rows.append({
      "iteration": it,
      # THE DASHBOARD READS THIS FILE. Its columns are named for walking, because that
      # is what it was built for, so the honest mapping is: height_mm is the real
      # measurement, upright_min is real, and the columns that only mean something for
      # a walking robot are left EMPTY rather than filled with a plausible-looking
      # zero. A blank shows as "-"; a fake zero would show as a measurement.
      "height_mm": round(final, 1),
      "upright_min": round(float(np.mean([r["upright"] for r in res])), 3),
      "fell": 0 if ok else 1,
      "distance_mm": "", "speed_mms": "", "drift_mm": "",
      "foot_force_p99_n": "", "joint_acc_rms": "",
      "power_mean_w": "", "cost_of_transport": "",
      "video": vid.replace(os.sep, "/"),
    })

  base_row = {
    "iteration": "baseline",
    "height_mm": round(float(np.mean([r["final_mm"] for r in base])), 1),
    "upright_min": round(float(np.mean([r["upright"] for r in base])), 3),
    "fell": 0,
    "distance_mm": "", "speed_mms": "", "drift_mm": "",
    "foot_force_p99_n": "", "joint_acc_rms": "",
    "power_mean_w": "", "cost_of_transport": "", "video": "",
  }
  csv_path = os.path.join(OUT_DIR, run_name, "summary.csv")
  os.makedirs(os.path.dirname(csv_path), exist_ok=True)
  fields = list(base_row)
  with open(csv_path, "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerow(base_row)
    w.writerows(rows)
  print(f"")
  print(f"wrote {csv_path} - the dashboard reads this")


if __name__ == "__main__":
  main()
