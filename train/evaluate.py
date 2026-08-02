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

WHY IT WAS REBUILT
------------------
The first version of this harness scored ONE seed, at ONE commanded speed, on the
NOMINAL robot with ZERO servo latency. It gave the wrong answer twice, and two
findings had to be retracted because of it. Three separate blind spots, each of
which is now closed:

*One seed is not a measurement.* Re-run over six seeds, the classical gait covers
711.0 mm with a standard deviation of 86.3 mm, and drifts anywhere from +33.8 to
-358.7 mm. The single "675.4 mm / 33.8 mm" figure that was quoted everywhere as
"the baseline" is one draw from that distribution, and a flattering one. Differences
smaller than about 170 mm between two single-seed runs are noise. Everything here
therefore reports mean, spread and worst case over at least five seeds, and
`measure_baseline` writes the classical gait's whole distribution to
progress/baseline.json so nothing has to hardcode a lucky number again.

*One command is structurally blind.* Scoring only 52.9 mm/s cannot see the thing
that actually distinguishes a trained policy from the hand-written gait. The
classical gait's stride scale saturates at 1.0, so it tops out near 59.7 mm/s no
matter what you ask for; a residual policy can push past that. It also cannot see a
policy that has quietly learned to stand still, because at one non-zero command a
frozen policy just looks slow. The command is now swept, and the sweep deliberately
includes 0.000 (where freezing hides) and +0.080 (where the classical gait's cap
shows up). Note that freezing loses badly on a swept score: a policy that never
moves has a mean speed error equal to the mean commanded speed, about 38 mm/s.

*Nominal dynamics is the one draw you will never get.* Training randomises mass,
COM, friction, servo gains, armature, zero offsets and 10-40 ms of command latency;
the old harness used none of it, and latency in particular was entirely absent even
though it is the single randomisation most likely to decide whether the real robot
walks. PROJECT_NOTES sets the bar at "beat Phase 2 under full randomisation", so
--full runs eight fixed draws over all of it, and worst case is reported alongside
the mean, because sim-to-real is a worst-case problem: the real robot is one draw
and you do not get to choose which one.

DETERMINISM IS PRESERVED
------------------------
MuJoCo is deterministic given the same model and initial state, and nothing here
samples at run time: the same (checkpoint, seed, command, draw) reproduces the same
walk to the millimetre, forever. A seed selects the gait phase the robot spawns at,
which is the dominant source of run-to-run spread. `draw=None` is the untouched
nominal robot, so a default rollout is bit-identical to what this file produced
before the rebuild and progress reels stay comparable across the change.
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

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

TRUNK_BODY = "base_link"
LEG_BODY_RE = re.compile(r"^(fl|fr|br|bl)_")
FOOT_GEOMS = tuple(f"{leg}_bottom_collision" for leg in LEGS)
FLOOR_GEOM = "floor"

MASS_KG = 1.6247        # robot.yaml total; used for cost of transport

##
# The evaluation grid.
#
# COMMANDS span the trained envelope from train/gray_env.py (-0.040 .. +0.080 m/s).
# 0.000 is in both grids on purpose - it is where a frozen policy hides - and so is
# +0.080, which is where the classical gait's stride-scale cap becomes visible.
# REF_COMMAND is the speed the old single-point harness used; the classic columns of
# progress/summary.csv still report it, so old rows stay comparable to new ones.
##
REF_COMMAND = 0.0529
SCREEN_COMMANDS = (0.0, 0.0529, 0.080)
FULL_COMMANDS = (-0.040, -0.020, 0.0, 0.0529, 0.080)
SCREEN_SEEDS = (0, 1, 2, 3, 4)
FULL_SEEDS = (0, 1, 2, 3, 4)
BASELINE_SEEDS = tuple(range(30))

N_DRAWS = 8
# Fixed, so every checkpoint and the baseline meet the SAME eight robots. That makes
# the comparison paired: a difference between two policies cannot be a difference in
# which robots they happened to be handed.
DRAW_SEED = 20260801

BASELINE_JSON = os.path.join("progress", "baseline.json")


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
    """The actor network: normalise, then a 4-layer ELU MLP. No torch, no GPU.

    Verified against onnxruntime on the exported graph to 9.4e-06 max absolute
    difference. Do not change the arithmetic here without re-running that check.
    """

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


def resolve_checkpoint(name: str, run_dir: str | None = None) -> str:
    """Accept a path, a bare iteration number, or a progress/policies tag.

    "950", "model_950.pt", "iter_0950.npz" and a full path all mean the same
    checkpoint, so the acceptance test can be written the way a person says it.
    """
    if os.path.exists(name):
        return name
    stem = os.path.basename(name)
    match = re.search(r"(\d+)", stem)
    if match is not None:
        want = int(match.group(1))
        for path in find_checkpoints(run_dir):
            if int(re.search(r"model_(\d+)\.pt", path).group(1)) == want:
                return path
        npz = os.path.join("progress", "policies", f"iter_{want:04d}.npz")
        if os.path.exists(npz):
            return npz
    raise FileNotFoundError(f"no checkpoint matching {name!r}")


# --------------------------------------------------------------------------------
# Domain randomisation - the same knobs train/gray_env.py turns.
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Draw:
    """One physically distinct robot: the thing the policy actually has to survive.

    Every field mirrors a randomisation in train/gray_env.py or train/gray_robot.py,
    with the same range. Nothing is invented here - if training does not randomise
    it, neither does this.

    `latency_s` is the one the old harness was missing entirely. It is the lag
    between the controller deciding a joint target and the servo receiving it over
    I2C, modelled as a delay buffer on the command (see `rollout`). Training samples
    2-8 physics steps at 5 ms, i.e. 10-40 ms, which is what robot.yaml records for
    the real bus.
    """

    name: str = "nominal"
    leg_density: float = 1.0            # mass AND inertia scale together
    trunk_density: float = 1.0
    com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    friction: float | None = None       # None -> leave the model's own value alone
    kp_scale: float = 1.0
    kv_scale: float = 1.0
    armature_scale: float = 1.0
    zero_offset: tuple[float, ...] = ()  # per-joint servo horn misalignment, rad
    latency_s: float = 0.0

    @property
    def is_nominal(self) -> bool:
        return (self.leg_density == 1.0 and self.trunk_density == 1.0
                and self.com_offset == (0.0, 0.0, 0.0) and self.friction is None
                and self.kp_scale == 1.0 and self.kv_scale == 1.0
                and self.armature_scale == 1.0 and not any(self.zero_offset)
                and self.latency_s == 0.0)

    def describe(self) -> str:
        return (f"{self.name:>8s}  legs x{self.leg_density:.2f}  "
                f"trunk x{self.trunk_density:.2f}  mu {self.friction or 1.0:.2f}  "
                f"kp x{self.kp_scale:.2f}  kv x{self.kv_scale:.2f}  "
                f"armature x{self.armature_scale:.2f}  "
                f"lag {self.latency_s*1000:.0f} ms")


NOMINAL = Draw()


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Uniform in the exponent, which is how dr.pseudo_inertia parameterises density."""
    return float(np.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_draws(n: int = N_DRAWS, seed: int = DRAW_SEED) -> tuple[Draw, ...]:
    """`n` randomised robots, drawn from the training ranges. Fixed for a given seed.

    Deliberately NOT re-drawn per policy: every checkpoint is scored on the same
    eight robots, so a difference between two checkpoints is a difference between
    the policies and not between the dice.

    No draw has zero latency. The real bus always has some, and the whole reason
    this class exists is that the old harness assumed none.
    """
    rng = np.random.default_rng(seed)
    draws = []
    for i in range(n):
        draws.append(Draw(
            name=f"d{i}",
            leg_density=_log_uniform(rng, 0.6, 1.4),        # resin density +/-40%
            trunk_density=_log_uniform(rng, 0.6, 1.4),
            com_offset=tuple(float(v) for v in rng.uniform(-0.02, 0.02, 3)),
            friction=float(rng.uniform(0.4, 1.2)),
            kp_scale=float(rng.uniform(0.7, 1.4)),
            kv_scale=float(rng.uniform(0.7, 1.4)),
            armature_scale=float(rng.uniform(0.35, 3.0)),
            zero_offset=tuple(float(v) for v in rng.uniform(-0.03, 0.03, 12)),
            latency_s=float(rng.uniform(0.010, 0.040)),
        ))
    return tuple(draws)


_PRISTINE = None            # per-process cache: compiling the MJCF costs ~40 ms


def _pristine_model():
    global _PRISTINE
    if _PRISTINE is None:
        import mujoco
        _PRISTINE = mujoco.MjModel.from_xml_path(MJCF)
    return _PRISTINE


def _model_for(draw: Draw | None):
    """A model with `draw` applied. The nominal draw returns the cached model as-is."""
    import mujoco

    base = _pristine_model()
    if draw is None or draw.is_nominal:
        return base

    model = copy.deepcopy(base)

    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if LEG_BODY_RE.match(name):
            scale = draw.leg_density
        elif name == TRUNK_BODY:
            scale = draw.trunk_density
        else:
            continue
        # Density scaling: mass and inertia both go linearly with density, which is
        # what dr.pseudo_inertia enforces and what dr.body_mass alone would not.
        model.body_mass[i] *= scale
        model.body_inertia[i] *= scale
        if name == TRUNK_BODY:
            model.body_ipos[i] += np.asarray(draw.com_offset)

    if draw.friction is not None:
        # Both sides of the contact, so the drawn value is the effective one. MuJoCo
        # pairs equal-priority geoms by taking the larger friction, so setting only
        # the foot would leave the floor's 0.8 as a floor under the draw.
        for gname in FOOT_GEOMS + (FLOOR_GEOM,):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                model.geom_friction[gid, 0] = draw.friction

    # <position kp kv> compiles to gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kv.
    model.actuator_gainprm[:, 0] *= draw.kp_scale
    model.actuator_biasprm[:, 1] *= draw.kp_scale
    model.actuator_biasprm[:, 2] *= draw.kv_scale

    model.dof_armature[6:] *= draw.armature_scale     # skip the free joint's 6 DOF

    return model


# --------------------------------------------------------------------------------
# Rollout.
# --------------------------------------------------------------------------------


def rollout(policy: Policy | None, *, speed_ms: float = REF_COMMAND,
            duration: float = 12.0, seed: int = 0, draw: Draw | None = None,
            video: str | None = None, width: int = 640, height: int = 480,
            fps: int = 25) -> dict:
    """Walk for `duration` seconds and measure it.

    `policy=None` runs the bare Phase 2 gait, which is the baseline every trained
    policy has to beat. Everything else about the rollout is identical, so the two
    are directly comparable.

    `draw=None` is the nominal robot with no command latency - identical to what
    this function did before randomisation was added, so old numbers and old videos
    remain reproducible.
    """
    import mujoco

    model = _model_for(draw)
    data = mujoco.MjData(model)
    draw = draw or NOMINAL

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

    # Servo command latency, as a delay buffer on the joint command. This is the one
    # randomisation the old harness had no model of at all, and on a 50 Hz loop it is
    # a substantial fraction of a control period: 40 ms is two whole ticks.
    lag_steps = int(round(draw.latency_s / dt))
    bias = (np.asarray(draw.zero_offset, dtype=float) if draw.zero_offset
            else np.zeros(12))
    pipe = [np.clip(stand, lo, hi) - bias] * (lag_steps + 1)

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

        # The zero offset is subtracted from the command exactly as
        # train/residual_action.apply_actions does: it is not an encoder reading -
        # Gray has none - it is the servo and the controller disagreeing about where
        # zero is, so the joint ends up `bias` away from where it was told to go.
        cmd = np.clip(target, lo, hi) - bias
        if lag_steps == 0:
            data.ctrl[order] = cmd

        for _ in range(substeps):
            if lag_steps > 0:
                pipe.append(cmd)
                data.ctrl[order] = pipe.pop(0)
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
        return {"error": "no samples", "command": speed_ms, "seed": seed,
                "draw": draw.name}

    s = np.array(samples)
    distance = float(s[-1, 1] - s[0, 1])
    ff = np.array(foot_force) if foot_force else np.zeros(1)
    ja = np.array(joint_acc) if joint_acc else np.zeros(1)
    pw = np.array(power) if power else np.zeros(1)
    elapsed = float(s[-1, 0])

    result = {
        "iteration": policy.iteration if policy else -1,
        "command": float(speed_ms),
        "seed": int(seed),
        "draw": draw.name,
        "fell": fell,
        "elapsed": elapsed,
        "distance": distance,
        "drift": float(s[-1, 2] - s[0, 2]),
        "speed": float(distance / max(elapsed, 1e-9)),
        # Speed over the whole intended window rather than the part survived. A robot
        # that falls at 3 s having covered 200 mm is doing 67 mm/s by the other
        # definition, which would let a fall flatter the tracking score. Every
        # command-tracking number below is built on this one.
        "speed_effective": float(distance / max(duration, 1e-9)),
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
            pw.mean() / (MASS_KG * 9.81 * abs(distance) / max(elapsed, 1e-9))
        ) if abs(distance) > 1e-4 else float("nan"),
    }
    result["speed_error"] = abs(result["speed_effective"] - float(speed_ms))

    if video and frames:
        os.makedirs(os.path.dirname(video) or ".", exist_ok=True)
        import imageio.v2 as iio
        iio.mimwrite(video, frames, fps=fps, quality=8, macro_block_size=1)
        result["video"] = video
    return result


# --------------------------------------------------------------------------------
# Aggregation. Mean, spread and worst case - never a single number.
# --------------------------------------------------------------------------------


def _agg(values, worst: str = "min") -> dict | None:
    """mean / sd / sem / worst over a list, skipping NaN and missing entries.

    `sd` is the spread of a single rollout, which is what tells you whether two
    checkpoints actually differ. `sem` is the uncertainty on the mean, which is what
    shrinks when you add seeds: the classical gait's distance has sd 86 mm however
    many seeds you run, but at n=5 the mean is known to about 39 mm.
    """
    clean = [float(v) for v in values
             if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return None
    arr = np.asarray(clean, dtype=float)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": float(arr.mean()),
        "sd": sd,
        "sem": sd / math.sqrt(arr.size) if arr.size > 1 else 0.0,
        "worst": float(arr.min() if worst == "min" else arr.max()),
        "n": int(arr.size),
    }


# metric -> (key in the rollout result, scale to report in, which end is "worst")
_METRICS = {
    "distance_mm":       ("distance", 1000.0, "min"),
    "speed_mms":         ("speed", 1000.0, "min"),
    "speed_error_mms":   ("speed_error", 1000.0, "max"),
    "drift_abs_mm":      (None, 1000.0, "max"),        # |drift|, computed below
    "upright_min":       ("upright_min", 1.0, "min"),
    "height_mm":         ("height_mean", 1000.0, "min"),
    "foot_force_p99_n":  ("foot_force_p99", 1.0, "max"),
    "joint_acc_rms":     ("joint_acc_rms", 1.0, "max"),
    "power_mean_w":      ("power_mean_w", 1.0, "max"),
    "cost_of_transport": ("cost_of_transport", 1.0, "max"),
}


def _trial_metric(trial: dict, name: str):
    key, scale, _ = _METRICS[name]
    if name == "drift_abs_mm":
        return abs(trial["drift"]) * scale if "drift" in trial else None
    return trial[key] * scale if key in trial else None


def _metric_values(trials: list[dict], name: str) -> list:
    values = (_trial_metric(t, name) for t in trials)
    return [v for v in values if v is not None]


def _summarise(trials: list[dict], command: float | None = None) -> dict:
    """Aggregate a set of trials. Pass `command` when they all share one.

    Distance and speed are SIGNED, so which end of their range is the bad end
    depends on which way the robot was told to walk: at -40 mm/s the worst run is
    the least negative one, not the most. Without a command - an aggregate spanning
    the whole sweep - the worst case of a signed quantity has no meaning, so read
    those two per command and use speed_error_mms for anything sweep-wide.
    """
    backwards = command is not None and command < 0.0
    out = {}
    for name, (_, _, worst) in _METRICS.items():
        if backwards and name in ("distance_mm", "speed_mms"):
            worst = "max"
        stats = _agg(_metric_values(trials, name), worst)
        if stats is not None:
            out[name] = stats
    scored = [t for t in trials if "error" not in t]
    out["fall_rate"] = (sum(1 for t in scored if t["fell"]) / len(scored)
                        if scored else float("nan"))
    out["n_trials"] = len(scored)
    out["n_failed"] = len(trials) - len(scored)
    return out


@dataclass
class Report:
    """Everything one evaluation grid produced, aggregated and still itemised."""

    label: str
    iteration: int
    commands: tuple[float, ...]
    seeds: tuple[int, ...]
    draws: tuple[str, ...]
    duration: float
    trials: list[dict]
    per_command: dict            # command -> summary dict
    overall: dict                # summary over the whole grid
    reference: dict              # summary at REF_COMMAND only, for summary.csv

    @property
    def track_mae_mms(self) -> float:
        """Mean |achieved - commanded| speed over the sweep. Lower is better.

        This is the number the acceptance test turns on, and the reason it is a
        SWEEP is that no single command can measure it: at one command a policy that
        stands still is merely slow, while across the sweep it scores the mean
        commanded speed - about 38 mm/s - and loses outright. Falling does not help
        either, because speed_effective divides by the full window.
        """
        return self.overall["speed_error_mms"]["mean"]

    @property
    def track_worst_mms(self) -> float:
        return self.overall["speed_error_mms"]["worst"]

    def top_speed_mms(self) -> dict | None:
        """What it does when asked for the most it can have."""
        return self.per_command.get(max(self.commands), {}).get("speed_mms")


def paired_delta(a: "Report", b: "Report", metric: str = "speed_error_mms",
                 lower_is_better: bool = True) -> dict | None:
    """How much better `a` is than `b`, matched trial for trial.

    Every report is run on the SAME commands, seeds and randomisation draws (see
    `sample_draws`), so the two are paired and the comparison can be made per
    matched rollout instead of between two means. That matters: on the randomised
    grid the spread ACROSS robots is far larger than the difference between two
    policies, so an unpaired comparison drowns a real 2 mm/s gap in a 10 mm/s sd
    while the paired one resolves it. This is precisely the mistake the old harness
    made in the other direction - reading noise as signal.

    Returns the mean advantage, its standard error, and how often `a` won.
    """
    def index(report):
        out = {}
        for t in report.trials:
            if "error" in t:
                continue
            out[(round(t["command"], 9), t["seed"], t["draw"])] = t
        return out

    ia, ib = index(a), index(b)
    shared = sorted(set(ia) & set(ib))
    diffs = []
    for k in shared:
        va, vb = _trial_metric(ia[k], metric), _trial_metric(ib[k], metric)
        if va is None or vb is None:
            continue
        diffs.append((vb - va) if lower_is_better else (va - vb))
    if not diffs:
        return None

    arr = np.asarray(diffs, dtype=float)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "metric": metric,
        "advantage": float(arr.mean()),      # positive means `a` is better
        "sd": sd,
        "sem": sd / math.sqrt(arr.size) if arr.size > 1 else 0.0,
        "win_rate": float((arr > 0).mean()),
        "n": int(arr.size),
    }


def format_paired(name_a: str, name_b: str, delta: dict | None,
                  unit: str = "mm/s") -> str:
    """A verdict with an error bar on it, or an honest 'cannot tell'."""
    if delta is None:
        return f"  {name_a} vs {name_b}: nothing comparable"
    adv, sem = delta["advantage"], delta["sem"]
    # 2 sem either side of the mean difference. If that interval spans zero the
    # harness has not separated them and must say so rather than pick a winner -
    # naming a winner it cannot actually resolve is how the old one went wrong.
    if delta["n"] < 2 or adv == 0.0 or abs(adv) < 2 * sem:
        verdict = "NOT SEPARATED"
    else:
        verdict = "BETTER" if adv > 0 else "WORSE"
    return (f"  {name_a} vs {name_b}: {verdict} by {adv:+.2f} +/- {sem:.2f} {unit} "
            f"on {delta['metric']}, winning {delta['win_rate']*100:.0f}% of "
            f"{delta['n']} matched rollouts")


def _ticker(enabled: bool, total: int, label: str):
    """A one-line progress counter on stderr. Silent when not attached to a terminal."""
    live = bool(enabled) and sys.stderr.isatty()
    state = {"n": 0}

    def tick(done: bool = False) -> None:
        if not live:
            return
        if done:
            sys.stderr.write("\r" + " " * 48 + "\r")
        else:
            state["n"] += 1
            sys.stderr.write(f"\r  {label} {state['n']}/{total} rollouts")
        sys.stderr.flush()

    return tick


_JOB_POLICY: Policy | None = None


def _init_worker(policy: Policy | None) -> None:
    """Hand each worker the policy once, rather than with every job."""
    global _JOB_POLICY
    _JOB_POLICY = policy


def _trial_job(args):
    """Module-level so ProcessPoolExecutor can pickle it on Windows."""
    speed_ms, duration, seed, draw = args
    return rollout(_JOB_POLICY, speed_ms=speed_ms, duration=duration, seed=seed,
                   draw=draw)


def evaluate(policy: Policy | None, *, commands=SCREEN_COMMANDS, seeds=SCREEN_SEEDS,
             draws=(NOMINAL,), duration: float = 12.0, jobs: int | None = None,
             label: str | None = None, progress: bool = False) -> Report:
    """Run the whole grid of (command x seed x draw) and aggregate it.

    Every rollout is independent and deterministic, so the grid parallelises
    perfectly; `jobs` controls how many processes. jobs=1 stays in this process,
    which is what you want under a debugger.
    """
    grid = [(float(c), duration, int(s), d)
            for c in commands for s in seeds for d in draws]

    if jobs is None:
        jobs = min(os.cpu_count() or 1, len(grid), 16)
    jobs = max(1, min(int(jobs), len(grid)))

    # Progress goes to stderr, so `eval_policy.py ... > report.txt` keeps the report
    # clean and a piped run does not end up with a line of carriage returns in it.
    tick = _ticker(progress, len(grid), label or "")

    results = []
    if jobs == 1:
        _init_worker(policy)
        for job in grid:
            results.append(_trial_job(job))
            tick()
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(policy,)) as pool:
            for r in pool.map(_trial_job, grid, chunksize=1):
                results.append(r)
                tick()
    tick(done=True)

    by_command = {}
    for c in commands:
        subset = [r for r in results if abs(r["command"] - c) < 1e-12]
        by_command[float(c)] = _summarise(subset, command=float(c))

    ref = [r for r in results if abs(r["command"] - REF_COMMAND) < 1e-12]

    iteration = policy.iteration if policy is not None else -1
    return Report(
        label=label or ("Phase 2 gait" if policy is None else f"iter {iteration}"),
        iteration=iteration,
        commands=tuple(float(c) for c in commands),
        seeds=tuple(int(s) for s in seeds),
        draws=tuple(d.name for d in draws),
        duration=duration,
        trials=results,
        per_command=by_command,
        overall=_summarise(results),
        reference=(_summarise(ref, command=REF_COMMAND) if ref
                   else _summarise(results)),
    )


# --------------------------------------------------------------------------------
# Printing.
# --------------------------------------------------------------------------------


def _fmt(stats: dict | None, digits: int = 1, width: int = 22) -> str:
    if stats is None:
        return "-".rjust(width)
    return (f"{stats['mean']:+.{digits}f} +/-{stats['sd']:.{digits}f} "
            f"[{stats['worst']:+.{digits}f}]").rjust(width)


# Which command each column is measured at is part of the column, not a footnote.
# "@52.9" is the reference command every historical number was taken at; the two
# unmarked columns are sweep-wide, because that is the only place they mean anything.
REPORT_HEADER = (f"{'policy':>14}  {'trials':>6}  "
                 f"{'speed MAE mm/s':>22}  {'distance @52.9 mm':>22}  "
                 f"{'top speed mm/s':>22}  {'|drift| @52.9 mm':>22}  {'falls':>6}")


def _pct(fraction: float, width: int = 6) -> str:
    if fraction is None or math.isnan(fraction):
        return "-".rjust(width)
    return f"{fraction * 100:.0f}%".rjust(width)


def format_report(report: Report) -> str:
    """One line: mean +/- sd [worst], for the four numbers that decide a checkpoint."""
    over = report.overall
    return (f"{report.label:>14}  {over['n_trials']:>6}  "
            f"{_fmt(over.get('speed_error_mms'))}  "
            f"{_fmt(report.reference.get('distance_mm'))}  "
            f"{_fmt(report.top_speed_mms())}  "
            f"{_fmt(report.reference.get('drift_abs_mm'))}  "
            f"{_pct(over['fall_rate'])}")


def format_sweep(report: Report) -> str:
    """The command sweep itself, which is the part a single-speed harness cannot see."""
    lines = [f"  {'commanded':>9}  {'achieved mm/s':>22}  {'|error| mm/s':>22}  "
             f"{'distance mm':>22}  {'falls':>6}"]
    for c in report.commands:
        s = report.per_command[c]
        lines.append(f"  {c*1000:>9.1f}  {_fmt(s.get('speed_mms'))}  "
                     f"{_fmt(s.get('speed_error_mms'))}  "
                     f"{_fmt(s.get('distance_mm'))}  "
                     f"{_pct(s['fall_rate'])}")
    return "\n".join(lines)


def format_row(label: str, r: dict) -> str:
    """A single rollout, for the one-off case. Kept for scripts that print one walk."""
    if "error" in r:
        return f"{label:>12s}  FAILED: {r['error']}"
    return (f"{label:>12s}  {r['distance']*1000:+8.1f} mm  "
            f"{r['speed']*1000:+7.1f} mm/s  drift {r['drift']*1000:+7.1f} mm  "
            f"upright {r['upright_min']:.3f}"
            f"{'  FELL' if r['fell'] else ''}")


# --------------------------------------------------------------------------------
# The baseline, as a distribution rather than a lucky draw.
# --------------------------------------------------------------------------------


def measure_baseline(*, seeds=BASELINE_SEEDS, commands=FULL_COMMANDS,
                     duration: float = 12.0, draws: tuple[Draw, ...] | None = None,
                     jobs: int | None = None, path: str = BASELINE_JSON,
                     progress: bool = True) -> dict:
    """Re-measure the Phase 2 gait properly and write progress/baseline.json.

    Two grids, because two different questions get asked of the baseline: what the
    hand-written gait does on the nominal robot (what every past number meant), and
    what it does across the randomised fleet (the bar PROJECT_NOTES actually sets).

    The file it writes is the single source of truth. Nothing should hardcode
    675.4 mm again - that was one seed, and the distribution it came from has a
    standard deviation of 86 mm.
    """
    if progress:
        print(f"nominal   {len(seeds)} seeds x {len(commands)} commands "
              f"= {len(seeds)*len(commands)} rollouts")
    nominal = evaluate(None, commands=commands, seeds=seeds, draws=(NOMINAL,),
                       duration=duration, jobs=jobs, label="Phase 2 gait",
                       progress=progress)

    draws = draws if draws is not None else sample_draws()
    if progress:
        print(f"randomised  {len(seeds)} seeds x {len(commands)} commands x "
              f"{len(draws)} draws = {len(seeds)*len(commands)*len(draws)} rollouts")
    randomised = evaluate(None, commands=commands, seeds=seeds, draws=draws,
                          duration=duration, jobs=jobs, label="Phase 2 gait (DR)",
                          progress=progress)

    ref_nom = nominal.reference
    blob = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "what": "The Phase 2 hand-written crawl gait, measured as a DISTRIBUTION. "
                "Read this file instead of hardcoding a number.",
        "source": "train/evaluate.py measure_baseline()",
        "duration_s": duration,
        "commands_ms": [round(c * 1000, 1) for c in commands],
        "n_seeds": len(seeds),
        "n_draws": len(draws),
        "draw_seed": DRAW_SEED,
        # Flat keys at the top level so a reader that only wants "the baseline
        # distance" does not have to know the schema. These are the NOMINAL robot at
        # the reference command, which is what every historical number meant.
        "distance_mm": round(ref_nom["distance_mm"]["mean"], 1),
        "distance_mm_sd": round(ref_nom["distance_mm"]["sd"], 1),
        "distance_mm_sem": round(ref_nom["distance_mm"]["sem"], 1),
        "distance_mm_worst": round(ref_nom["distance_mm"]["worst"], 1),
        "speed_mms": round(ref_nom["speed_mms"]["mean"], 1),
        "speed_mms_sd": round(ref_nom["speed_mms"]["sd"], 1),
        "drift_mm": round(ref_nom["drift_abs_mm"]["mean"], 1),
        "drift_mm_sd": round(ref_nom["drift_abs_mm"]["sd"], 1),
        "upright_min": round(ref_nom["upright_min"]["mean"], 3),
        "track_mae_mms": round(nominal.track_mae_mms, 1),
        "top_speed_mms": round(nominal.top_speed_mms()["mean"], 1),
        "nominal": {"per_command": {str(round(c * 1000, 1)): nominal.per_command[c]
                                    for c in nominal.commands},
                    "overall": nominal.overall,
                    "reference_command_mms": round(REF_COMMAND * 1000, 1)},
        "randomised": {"per_command": {str(round(c * 1000, 1)): randomised.per_command[c]
                                       for c in randomised.commands},
                       "overall": randomised.overall,
                       # Distance summed across a sweep that includes walking
                       # backwards is not a quantity. The reference command is.
                       "reference": randomised.reference,
                       "reference_command_mms": round(REF_COMMAND * 1000, 1)},
        "superseded": {
            "distance_mm": 675.4, "drift_mm": 33.8,
            "note": "The single-seed figure quoted in docs and dashboard/collect.py "
                    "until 2026-08-01. It is one draw from the distribution above "
                    "and is not the baseline. Do not compare anything to it.",
        },
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    os.replace(tmp, path)
    return blob


def load_baseline(path: str = BASELINE_JSON) -> dict | None:
    """progress/baseline.json, or None if it has not been measured yet."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None
