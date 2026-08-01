"""Score a trained policy in plain MuJoCo, independently of the trainer.

WHY NOT JUST REPLAY IT IN mjlab
-------------------------------
A policy that only works in the simulator it trained in has probably learned that
simulator's quirks rather than how to walk, and it will fall over on the real robot.
docs/PROJECT_NOTES.md therefore requires dual-sim validation before any hardware
rollout: re-score the checkpoint in a second engine and see whether it survives.

This module is that second engine. It shares no code with the training stack:

  * plain MuJoCo stepping sim/models/gray.xml, at that file's 2 ms timestep with 10
    substeps per control tick - the trainer runs 5 ms x 4, so the discretisation
    genuinely differs
  * the real `GaitGenerator`, not train/gait_table.py, so the precomputed table is
    validated too rather than assumed
  * the policy re-implemented as a few NumPy matrix multiplies, so a bug in the
    training-side network wrapper cannot hide here

It is also deterministic. MuJoCo is deterministic given the same model and initial
state, and nothing here samples: the same checkpoint, seed and command reproduce the
same walk to the millimetre, forever. That is what makes a progress reel meaningful -
every clip starts from identical conditions, so any difference is the policy
improving and nothing else.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np

from gray.gait import GaitGenerator, GaitParams
from gray.kinematics import LEGS, joint_vector, load_legs

MJCF = "sim/models/gray.xml"
SETTLE_S = 0.5          # identical to scripts/walk.py, so numbers are comparable
CONTROL_HZ = 50.0
SEGMENTS = ("hip", "top", "bottom")
JOINT_ORDER = tuple(f"{leg}_{seg}" for leg in LEGS for seg in SEGMENTS)

# Must match train/residual_action.py and train/gray_robot.py.
MAX_RESIDUAL = 0.2
SPEED_AT_UNIT_STRIDE = 0.0529
SOFT_LIMIT_FACTOR = 0.95


# --------------------------------------------------------------------------------
# The policy, as plain NumPy.
# --------------------------------------------------------------------------------


# rsl_rl's EmpiricalNormalization divides by (std + eps), not by std, with eps = 1e-2.
# This is not a numerical nicety that can be skipped: two observation dimensions have
# exactly zero variance - lin_vel_y and ang_vel_z, which train/gray_env.py deliberately
# commands to zero - so dividing by the raw std produces inf, then NaN, and the policy
# outputs nothing at all. Recovered by diffing the exported ONNX divisor against the
# checkpoint's stored _std: the gap is 0.01 on every dimension.
NORM_EPS = 1e-2


@dataclass
class Policy:
    """The actor network: normalise, then a 4-layer ELU MLP. No torch, no GPU."""

    mean: np.ndarray
    scale: np.ndarray            # already includes NORM_EPS; divide by this directly
    layers: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    iteration: int = -1

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / self.scale
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:              # ELU on all but the output
                x = np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))
        return x


def load_policy(checkpoint: str) -> Policy:
    """Load a policy from an rsl_rl .pt checkpoint or a distilled .npz."""
    if checkpoint.endswith(".npz"):
        z = np.load(checkpoint)
        n = int(z["n_layers"])
        return Policy(
            mean=z["mean"], scale=z["scale"],
            layers=[(z[f"w{i}"], z[f"b{i}"]) for i in range(n)],
            iteration=int(z["iteration"]),
        )

    import torch  # only needed to unpickle a training checkpoint

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = blob["actor_state_dict"]
    to_np = lambda k: sd[k].detach().cpu().numpy()  # noqa: E731

    layers = []
    for i in (0, 2, 4, 6):                            # nn.Sequential with ELU between
        layers.append((to_np(f"mlp.{i}.weight"), to_np(f"mlp.{i}.bias")))

    return Policy(
        mean=to_np("obs_normalizer._mean").reshape(-1),
        scale=to_np("obs_normalizer._std").reshape(-1) + NORM_EPS,
        layers=layers,
        iteration=int(blob.get("iter", -1)),
    )


def save_policy_npz(policy: Policy, path: str) -> None:
    """Write just the actor - about 200 KB against 3 MB for a full checkpoint.

    A training checkpoint also carries the critic and the optimiser state, neither of
    which is needed to walk. Stripping them makes a policy small enough to commit, so
    the repo can hold a version history of gaits (as docs/PROJECT_NOTES.md intends)
    rather than leaving them in logs/, which .gitignore excludes and which nothing
    protects from being wiped.

    Because the rollout is deterministic, this file is all you ever need: any video
    of this policy can be regenerated from it, byte for byte.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {"mean": policy.mean.astype(np.float32),
              "scale": policy.scale.astype(np.float32),
              "n_layers": np.int32(len(policy.layers)),
              "iteration": np.int32(policy.iteration)}
    for i, (w, b) in enumerate(policy.layers):
        arrays[f"w{i}"] = w.astype(np.float32)
        arrays[f"b{i}"] = b.astype(np.float32)
    np.savez_compressed(path, **arrays)


def find_checkpoints(run_dir: str | None = None) -> list[str]:
    """Every model_<n>.pt under the newest run, sorted by iteration."""
    if run_dir is None:
        runs = sorted(glob.glob(os.path.join("logs", "rsl_rl", "gray_residual", "*")))
        runs = [r for r in runs if glob.glob(os.path.join(r, "model_*.pt"))]
        if not runs:
            return []
        run_dir = runs[-1]
    found = glob.glob(os.path.join(run_dir, "model_*.pt"))
    return sorted(found, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))


# --------------------------------------------------------------------------------
# Rollout.
# --------------------------------------------------------------------------------


def rollout(policy: Policy | None, *, speed_ms: float = 0.0529, duration: float = 12.0,
            seed: int = 0, video: str | None = None, width: int = 640,
            height: int = 480, fps: int = 25) -> dict:
    """Walk for `duration` seconds and measure it.

    `policy=None` runs the bare Phase 2 gait, which is the baseline every trained
    policy has to beat. Everything else about the rollout is identical, so the two
    are directly comparable.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)

    legs = load_legs()
    params = GaitParams()
    gait = GaitGenerator(legs, params)

    act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
           for i in range(model.nu)}
    order = [act[n] for n in JOINT_ORDER]

    # The robot is taught six things at once and they pull against each other -
    # walking faster means stomping harder, landing softly means going slower. One
    # overall score hides that, so each objective gets its own measurement here.
    foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_bottom")
                   for leg in LEGS]

    def sensor(name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = model.sensor_adr[sid]
        return data.sensordata[adr:adr + model.sensor_dim[sid]]

    # Same starting pose as scripts/walk.py.
    stand = joint_vector(gait.stand())
    data.qpos[7:] = stand
    data.qpos[2] = params.stance_height + 0.015
    data.ctrl[order] = stand
    mujoco.mj_forward(model, data)

    # Soft joint limits, as mjlab applies them during training.
    lo = model.jnt_range[1:, 0] * SOFT_LIMIT_FACTOR
    hi = model.jnt_range[1:, 1] * SOFT_LIMIT_FACTOR

    dt = model.opt.timestep
    substeps = max(1, int(round(1.0 / CONTROL_HZ / dt)))
    step_dt = substeps * dt

    rng = np.random.default_rng(seed)
    phase = float(rng.random()) if seed >= 0 else 0.0
    prev_action = np.zeros(12)

    renderer = cam = None
    frames: list = []
    if video:
        renderer = mujoco.Renderer(model, height=height, width=width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance, cam.elevation, cam.azimuth = 0.9, -12, 120
    render_every = max(1, int(round(CONTROL_HZ / fps)))

    t, fell, samples, tick = 0.0, False, [], 0
    foot_force, joint_acc, power = [], [], []
    total = SETTLE_S + duration
    while t < total:
        walking = t >= SETTLE_S
        command = np.array([speed_ms if walking else 0.0, 0.0, 0.0])
        stride = float(np.clip(command[0] / SPEED_AT_UNIT_STRIDE, -1.0, 1.0))

        nominal = joint_vector(gait.joint_angles(phase * params.period, stride))

        if policy is None:
            target = nominal
        else:
            quat = sensor("imu_quat")
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, quat)
            R = R.reshape(3, 3)
            obs = np.concatenate([
                sensor("imu_gyro"),          # base_ang_vel   (3)
                R.T @ np.array([0.0, 0.0, -1.0]),   # projected_gravity (3)
                command,                     # command        (3)
                [np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)],  # phase (2)
                nominal,                     # gait_targets   (12)
                prev_action,                 # actions        (12)
            ])
            action = policy(obs)
            prev_action = action
            residual = np.clip(action * MAX_RESIDUAL, -MAX_RESIDUAL, MAX_RESIDUAL)
            target = nominal + residual

        data.ctrl[order] = np.clip(target, lo, hi)

        for _ in range(substeps):
            mujoco.mj_step(model, data)
            if t >= SETTLE_S:
                # Sampled every PHYSICS step, not every control step: a touchdown
                # spike lasts a few milliseconds and stepping at 50 Hz would stride
                # straight over the peak that actually cracks a resin part.
                foot_force.append(
                    max(float(np.linalg.norm(data.cfrc_ext[b][3:6])) for b in foot_bodies))
                joint_acc.append(float(np.sqrt(np.mean(data.qacc[6:] ** 2))))
                power.append(float(np.abs(data.actuator_force * data.qvel[6:]).sum()))
        t += step_dt
        phase = (phase + step_dt / params.period) % 1.0

        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, data.qpos[3:7])
        up = float(Rm.reshape(3, 3)[2, 2])

        if SETTLE_S < t <= total:
            samples.append((t - SETTLE_S, data.qpos[0], data.qpos[1], data.qpos[2], up))
        if up < 0.2:
            fell = True
            break

        if renderer is not None and tick % render_every == 0:
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.08]
            renderer.update_scene(data, cam)
            frames.append(renderer.render())
        tick += 1

    if not samples:
        return {"error": "no samples"}

    s = np.array(samples)
    distance = float(s[-1, 1] - s[0, 1])
    ff = np.array(foot_force) if foot_force else np.zeros(1)
    ja = np.array(joint_acc) if joint_acc else np.zeros(1)
    pw = np.array(power) if power else np.zeros(1)

    result = {
        "iteration": policy.iteration if policy else -1,
        "fell": fell,
        "elapsed": float(s[-1, 0]),
        "distance": distance,
        "drift": float(s[-1, 2] - s[0, 2]),
        "speed": float(distance / max(s[-1, 0], 1e-9)),
        "height_mean": float(s[:, 3].mean()),
        "height_std": float(s[:, 3].std()),
        "upright_min": float(s[:, 4].min()),
        "support_min": min(gait.support_count(x)
                           for x in np.linspace(0, params.period, 200)),
        # --- one measurement per training objective ---
        # Landing softness. The 99th percentile rather than the outright max, which
        # a single solver artefact can dominate. Standing still is ~5.3 N per foot
        # (1.625 kg over three feet), so anything far above that is a stomp.
        "foot_force_p99": float(np.percentile(ff, 99)),
        "foot_force_mean": float(ff.mean()),
        # Smoothness, as felt by the gearbox.
        "joint_acc_rms": float(ja.mean()),
        # Mechanical power at the joints, and how much it costs to get anywhere.
        # Cost of transport is dimensionless, so it compares across speeds - a robot
        # that walks twice as fast for twice the power is no less efficient.
        "power_mean_w": float(pw.mean()),
        "cost_of_transport": float(
            pw.mean() / (1.6247 * 9.81 * abs(distance) / max(s[-1, 0], 1e-9))
        ) if abs(distance) > 1e-4 else float("nan"),
    }

    if video and frames:
        os.makedirs(os.path.dirname(video) or ".", exist_ok=True)
        import imageio.v2 as iio
        iio.mimwrite(video, frames, fps=fps, quality=8, macro_block_size=1)
        result["video"] = video
    return result


def format_row(label: str, r: dict) -> str:
    if "error" in r:
        return f"{label:>12s}  FAILED: {r['error']}"
    return (f"{label:>12s}  {r['distance']*1000:+8.1f} mm  "
            f"{r['speed']*1000:+7.1f} mm/s  drift {r['drift']*1000:+7.1f} mm  "
            f"upright {r['upright_min']:.3f}"
            f"{'  FELL' if r['fell'] else ''}")
