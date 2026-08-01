# Gray — project notes

> Read this first. It carries the project state, the findings that were
> expensive to discover, and the spec for the next phase.

Gray is a 12-DOF 3D-printed quadruped, designed and built from scratch by Ibrahim
Bisen in 2021–22. The mechanical work is finished; the software was never written
because the collaborator handling it left in Nov 2022. The goal is to teach it to
walk: a classical gait first, then a reinforcement learning policy layered on top,
then deployment to the real robot.

**The robot is currently disassembled** (all parts accounted for, reassembly
possible). Everything through Phase 3 is pure software and needs no hardware.

---

## Where the project stands

| Phase | What | Robot needed | Status |
|---|---|---|---|
| 1 | Digital twin — URDF repair, mass model, MuJoCo | No | **Done** |
| 2 | Classical gait — IK, Bézier trajectories, scheduler | No | **Done** |
| 3 | Residual RL training on the 4070 Ti | No | **In progress** |
| 4 | Reassemble, weigh, calibrate, build power system | Yes | Blocked |
| 5 | Sim-to-real deployment | Yes | Pending |
| 6 | Gamepad control, terrain, perception | Yes | Pending |

Phase 2 result: crawl gait walks **675 mm in 12 s (56 mm/s)**, three feet down at
all times, uprightness 0.987, no falls.

---

## The constraint that shapes everything

**12× DS3218MG 270° hobby servos on a PCA9685 over I²C.**

- Position-commanded, **not** torque-commanded.
- **No feedback.** You can never read where a joint actually is, only where you
  told it to go. This is why the observation space excludes measured joint
  positions — the hardware cannot supply them, so a policy that uses them will
  not transfer.
- PWM period is 20 ms, so **50 Hz is a hard control ceiling**. Do not design
  anything that needs faster.

This rules out the actuator-space methods used by ANYmal/Unitree/MIT Cheetah and
puts Gray in the Stanford Pupper / SpotMicro class.

---

## Hard-won findings — do not rediscover these

**The SolidWorks export is Y-up.** URDF and MuJoCo are Z-up. Loaded raw, the robot
lies on its side: two legs vertical, feet spanning 19 mm instead of 216 mm.
`tools/fix_urdf.py` rotates the base frame +90° about X. If you ever regenerate
from the source URDF, this must be applied.

**Servo geometry is baked into some link meshes.** `base_link` contains 4, each
`*_top` contains 1. Found by splitting the multi-body STLs and matching the
DS3218MG's 40×20 mm footprint. Their volume is subtracted before applying resin
density, so servos are not counted twice.

**MuJoCo silently welds the root link to the world**, discarding `base_link`'s
723 g — the model reports 901 g instead of 1625 g. `tools/make_mjcf.py` rebuilds
the trunk as a floating body. If mass ever looks wrong, check this first.

**The original 1.254 kg URDF is incomplete, not wrong.** A least-squares fit
recovers its assumptions to 0.017 g RMS: solid plastic at 1.0227 g/cm³, 64.3 g
servos, 8 of 12 modelled. Its COM and inertia axes are *good* and are reused —
mesh centroids match to 0.1 mm on every servo-free link. Only completeness was
lacking (no battery, Pi, electronics, fasteners, or hip-pitch servos).

**"Not watertight" meshes are multi-body STLs, not broken geometry.** Hole-filling
changes their volume by 0.00%. Volumes are trustworthy.

**The knee is a pushrod linkage, not direct drive.** A thigh-mounted DS3218 drives
the shank through a ball-jointed metal rod (see `Overview/photo_2022-09-18_12-46-38.jpg`
and `RObot_from_angle.JPG`). Harmless in sim — a four-bar driving a hinge is still
a hinge — but **servo angle ≠ joint angle on the real robot**. This is a Phase 4
blocker for deployment, recorded in `gray/config/robot.yaml`.

**Gait tuning limits.** Above ~2 strides/s the feet skid and the robot travels
*backward*. Speed trades against straightness: 160 mm/0.7 s gives 71.6 mm/s but
veers 65 mm, while 120 mm/0.6 s runs nearly straight at 52.8 mm/s. Defaults are the
straight one — drift comes from few-mm per-leg CAD asymmetry and is exactly what
the RL layer should absorb. Trot works but wanders ~140 mm; it needs active
balance, which is Phase 3's job, not more hand-tuning.

**Mass is estimated, not measured.** 1.625 kg assumes SLA resin hollowed at ~55%
fill; solid parts would give 2.00 kg. Randomize resin density ±40% to cover it.
Correct with a scale during Phase 4.

---

## Layout

```
gray/                  package — SHARED by simulator and Raspberry Pi
  kinematics.py        closed-form 3-DOF leg FK/IK (pure NumPy, no MuJoCo)
  gait.py              Bézier swing + stance + crawl/trot/pace schedulers
  config/robot.yaml    SINGLE SOURCE OF TRUTH: masses, inertia, leg geometry,
                       servo specs, hardware notes
sim/models/            gray.urdf, gray.xml (MJCF), decimated meshes, previews
scripts/walk.py        run and measure a gait (--sim now, --real in Phase 5)
train/                 Phase 3 RL. MAY import MuJoCo/torch/mjlab — that is why it
  gait_table.py        is not inside gray/. Precomputed gait, GPU lookup
  residual_action.py   policy output -> bounded correction on the gait
  gray_robot.py        Gray as an mjlab entity (servo specs, collisions)
  gray_env.py          obs / rewards / randomization / commands
  rewards.py           stride-averaged velocity tracking (read the docstring)
  runner.py            checkpoint + ONNX export with Gray-aware metadata
  tasks.py             registers "Gray-Residual-Flat" with mjlab
tools/                 model generation pipeline (see below)
tests/                 pytest — 9 tests, all must pass
robot/                 original CAD, untouched. Git LFS.
Overview/              build photos — genuinely useful for hardware questions
```

**Key principle:** `kinematics.py` and `gait.py` are imported unchanged by both the
simulator and the real robot. One code path, no sim/real drift. Neither may import
MuJoCo — the Pi will never have it.

---

## Commands

```bash
# regenerate the model from source CAD (in order)
python tools/estimate_masses.py --write-yaml     # masses -> robot.yaml
python tools/fix_urdf.py                         # -> sim/models/gray.urdf
python tools/make_mjcf.py --check                # -> sim/models/gray.xml + drop test
python tools/extract_kinematics.py               # leg geometry -> robot.yaml

# verify
PYTHONPATH=. python -m pytest tests/ -q          # 9 tests
python scripts/walk.py --duration 8              # expect ~450 mm, no fall
python scripts/walk.py --gait trot --duration 8
```

`tools/estimate_masses.py --fill 1.0` re-runs assuming solid parts.

```bash
# watch it walk - live 3D window, orbit with the mouse
python scripts/walk.py --view

# Phase 3: train, watch a checkpoint, score one against the Phase 2 baseline
python scripts/train_residual.py Gray-Residual-Flat
python scripts/train_residual.py Gray-Residual-Flat --env.scene.num-envs 1024
python scripts/play_residual.py Gray-Residual-Flat
tensorboard --logdir logs/rsl_rl          # training curves at localhost:6006

# check the gait lookup table still matches the real GaitGenerator
python train/gait_table.py
```

Dependencies live in `pyproject.toml`, split so `gray/` stays installable on the Pi:
core is only `numpy pyyaml`; `[sim]`, `[tools]`, `[dev]` and `[train]` add the rest.
**Python 3.10–3.13 only** — mjlab does not support 3.14. Install torch from the CUDA
index or you get a CPU-only build that silently ignores the GPU:
`uv pip install torch --index-url https://download.pytorch.org/whl/cu130`.

---

## Phase 3 — built, training

Implements **D²-GMBC** ([arXiv 2010.12070](https://arxiv.org/pdf/2010.12070)): Bézier
gait + RL modulating it + domain randomization, on **mjlab 1.5.3** (Isaac Lab API over
MuJoCo-Warp) with PPO from rsl_rl. Everything lives in `train/`, deliberately outside
`gray/` because it imports MuJoCo and torch.

**The policy emits a correction, never a joint angle:**

```
joint target = classical_gait(phase, stride) + clip(policy, ±0.2 rad)
```

so a bad or half-trained policy degrades toward a gait that already walks.

**The one non-obvious piece — `train/gait_table.py`.** Training runs thousands of
robots per GPU step, and `GaitGenerator.joint_angles` is per-leg NumPy that solves IK
four times per call. Calling it per-robot per-tick would have made the gait, not the
physics, the entire cost of training. But the Phase 2 gait is **open loop** —
`foot_targets(t, speed)` reads only `t` and `speed`, never robot state — so it is a
pure function of (phase, stride scale) and can be tabulated exactly, once, using the
unmodified `GaitGenerator`. Verified: **1.6e-07 rad** at the phases the 50 Hz
controller actually visits (`N_PHASE` is a multiple of the 30 ticks per cycle, so
every lookup lands on a grid point), 7.1e-05 rad on the interpolated speed axis. This
is a precomputation, not an approximation, and `gray/` stays the single code path
shared with the Pi.

**Task as configured** (`train/gray_env.py`):

- **50 Hz** — 0.005 s timestep × decimation 4. Not negotiable; it is the PWM ceiling.
- **Actor observes IMU only** — projected gravity, angular velocity, commanded
  velocity, gait phase as (cos, sin), the gait's current targets, and its own last
  action. **No measured joint positions**, because the servos cannot report them. The
  critic gets the true joint state as privileged information and is discarded after
  training.
- **Commands** are Gray-scale: **−0.04 to +0.08 m/s** forward, lateral and yaw
  commanded to zero. The classical gait cannot strafe or turn, so commanding zero yaw
  makes the tracking reward pay for walking *straight* — absorbing the few-mm CAD
  asymmetry Phase 2 could not tune out.
- **Randomization:** resin density ±40% via `dr.pseudo_inertia(alpha_range=...)`
  (scales mass *and* inertia — `dr.body_mass` alone would not), trunk COM ±2 cm,
  ground friction 0.4–1.2, armature ×0.35–3.0, servo gains ±30–40%, servo zero-offset
  ±0.03 rad, command latency 2–8 physics steps (10–40 ms).

**Velocity tracking had to be rewritten — see `train/rewards.py`.** mjlab's stock
`track_linear_velocity` scores *instantaneous* trunk velocity. Measured over a steady
crawl, Gray's is `vx mean 0.055, sd 0.205` — the stride ripple from each footfall is
**four times the mean**. Any std tight enough to be meaningful drove the exponent to
≈ −6 and the reward and its gradient to exactly zero, which is what the first smoke
run showed. Tracking is now scored against a velocity low-pass filtered at
τ = one gait period, which cuts the ripple 12× (sd 0.205 → 0.016) and leaves the mean
untouched. **If you add a reward here, check it against the ripple before trusting it.**

**Still to do:**

- **Dual-sim validation before any hardware rollout.** `scripts/eval_policy.py` —
  re-score the best checkpoint in plain MuJoCo and compare against the Phase 2
  baseline on the same metrics. Note the two sims deliberately differ (mjlab steps at
  0.005 s, `walk.py` at 0.002 s), which is what makes it an independent check.
- Success bar: beat Phase 2 on distance and fall rate **under full randomization**.

---

## Working setup

One repo, four machines. Mac writes code and evaluates checkpoints; **Windows
desktop (4070 Ti) trains**; Raspberry Pi 4 runs Phase 5; GitHub syncs. Trained
policies are a few hundred KB — commit them for a version history of gaits.

Minimal clone for the training box (tested — full clone is 133 MB because of
`Overview/` and non-LFS assemblies):

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/ibrahimbisen/Gray.git Gray
cd Gray && git sparse-checkout set gray sim scripts tools tests train docs
```

`sim/models/meshes/*.STL` are deliberately excluded from LFS in `.gitattributes`
so an LFS-skipped clone still loads the model. Do not undo that.

---

## Conventions

- Work lives on branch `sim-phase-1-2-digital-twin-and-gait`; `main` is untouched
  original CAD history. Branch rather than committing to `main`.
- `robot.yaml` is generated — edit the tool, not the file.
- Never let `gray/` import MuJoCo.
- Commit trained policies; do not commit logs, checkpoints or `.venv`.
- The user is a mechanical engineer, not a software person. Explain software
  concepts plainly; do not explain mechanical ones.

## Open questions for the user

1. Are the SLA parts hollowed or solid? (±15% on total mass)
2. Servo placement is confirmed from photos: 4 body / 4 hip / 4 thigh, none on the
   shank.
