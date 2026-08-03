# Reward functions — one reference for every stage

What the established implementations actually use, per curriculum stage, with real
weights. Compiled from the installed mjlab source, this project's own run history,
and the published work on legged_gym, Isaac Lab, Unitree, ANYmal, Walk-These-Ways,
MIT Cheetah, D²-GMBC and the hobby-servo quadrupeds.

**A units note that makes every number below comparable.** mjlab, legged_gym and
Isaac Lab all compute `reward = raw × weight × dt`. Gray runs at `dt = 0.02 s`
(50 Hz), the same convention, so config weights transfer between those three
directly. They do **not** transfer from numbers printed in papers, which sometimes
quote the already-dt-multiplied value or use a different formula entirely. Every
paper-sourced figure below says so.

---

## 1. Every term

### Task tracking

| Term | Measures | Typical weight | Stages | Where |
|---|---|---|---|---|
| `track_linear_velocity` | `exp(-(‖cmd_xy − v_xy‖² + v_z²)/σ²)`, σ = 0.5 m/s default | mjlab **2.0**; legged_gym / Isaac Lab **1.0**; Isaac Go2 **1.5** | 5, 6, 8 | `mjlab.tasks.velocity.mdp.rewards.track_linear_velocity` |
| `track_angular_velocity` | `exp(-((cmd_yaw−ω_z)² + ‖ω_xy‖²)/σ²)`, σ ≈ 0.71 rad/s in mjlab | mjlab **2.0**; legged_gym **0.5** | 5, 6, 8 | `...rewards.track_angular_velocity` |
| `feet_air_time` | Bonus per foot for air time in 0.05–0.5 s, gated by command | mjlab **0.0 — shipped disabled on Go1**; legged_gym **1.0**; ANYmal-flat **2.0**; Isaac Lab **0.125** | 4, 5 | `...rewards.feet_air_time` |
| Raibert foot placement | `−‖desired_xy − actual_xy‖²` per foot | Walk-These-Ways raw **−10.0** | 3 | must write — no equivalent in mjlab |
| Gait-phase / contact schedule | Gaussian-CDF desired contact vs actual force | Walk-These-Ways raw **4.0** | 4, 5, 6 | must write — mjlab has **no** phase reward |

### Posture and stability

| Term | Measures | Typical weight | Stages | Where |
|---|---|---|---|---|
| `posture` | `exp(-mean((q − q_default)²/std²))` | ours **1.5**, std 0.35 | 1, 7 | `mjlab.envs.mdp.rewards.posture` |
| `variable_posture` | Same, std widens with commanded speed | mjlab **1.0**; Go1 std 0.05→0.3 hip/thigh, 0.1→0.6 calf | 5, 6, 8 | `...velocity.mdp.rewards.variable_posture` |
| `upright` | `exp(-‖proj_gravity_xy‖²/σ²)`, σ = √0.2 | mjlab **1.0** | all | `...velocity.mdp.rewards.upright` |
| `flat_orientation_l2` | `sum(proj_gravity_xy²)`, plain L2 | ours **−2.0** ("tilt") | 1, 2 | `mjlab.envs.mdp.rewards.flat_orientation_l2` |
| `base_height` | `exp(-(z − target)²/std²)` | ours **2.0**, std 0.03 | 1, 2, 7 | ours, `gray/tasks/stand_env_cfg.py` |
| `base_still` | `exp(-‖v_lin‖²/std²)` | ours **1.0**, std 0.25 | **1 only** — actively wrong for 2–8 | ours |
| `body_angular_velocity_penalty` | `sum(ω_xy²)`, excludes yaw | mjlab **0.0** (off); ANYmal **−6·dt** | 2, 5, 6, 7 | `...velocity.mdp.rewards.body_angular_velocity_penalty` |
| `trunk_spin` | `sum(ω²)` all three axes | ours **−0.05** | 2 | ours, `push_env_cfg.py` |

### Gait and foot quality

| Term | Measures | Typical weight | Stages | Where |
|---|---|---|---|---|
| `feet_clearance` | `‖height − target‖ × ‖vel_xy‖` | mjlab **−2.0**, target 0.1 m | 4, 5, 6, 8 | `...velocity.mdp.rewards.feet_clearance` |
| `feet_swing_height` | `(peak/target − 1)²` scored at landing | mjlab **−0.25** | 4, 5, 6, 8 | `...rewards.feet_swing_height` |
| `feet_slip` | `‖vel_xy‖²` while in contact | mjlab **−0.1**; Walk-These-Ways **−0.04** | 3–8 | `...rewards.feet_slip` |
| `soft_landing` | Contact force at touchdown | mjlab **−1e-5** | 4–8 | `...rewards.soft_landing` |

### Smoothness, effort, actuator protection

| Term | Measures | Typical weight | Stages | Where |
|---|---|---|---|---|
| `action_rate_l2` | `sum((aₜ − aₜ₋₁)²)` | **−0.01** is the cross-framework anchor (legged_gym, Isaac Lab, Unitree, our stage 1). Walk-These-Ways −0.1 on PD-target delta. Mini Cheetah −2e-4. **We tried −1.5 and it broke fall-catching.** | 1, 2, 5, 6 | `mjlab.envs.mdp.rewards.action_rate_l2` |
| `action_acc_l2` | `sum((aₜ − 2aₜ₋₁ + aₜ₋₂)²)` | Walk-These-Ways equivalent **−0.1** | 2, 5, 6, 7 | `mjlab.envs.mdp.rewards.action_acc_l2` |
| `joint_acc_l2` | `sum(q̈²)` | legged_gym / Isaac Lab **−2.5e-7** | 5, 6, 7 | `mjlab.envs.mdp.rewards.joint_acc_l2` |
| `joint_torques_l2` | `sum(τ²)` | ANYmal **−1e-5**; **A1 / Go2 −0.0002**; ours **−0.0002** | 1, 2, 5–8 | `mjlab.envs.mdp.rewards.joint_torques_l2` |
| `joint_vel_l2` | `sum(q̇²)` | ours **−0.001**; legged_gym **0** (disabled) | **1 only** | `mjlab.envs.mdp.rewards.joint_vel_l2` |
| `electrical_power_cost` | `sum(clamp(τ·q̇, min=0))` | Minitaur equivalent 0.008 | 5, 8 | `mjlab.envs.mdp.rewards.electrical_power_cost` |
| `joint_pos_limits` | Soft-limit violation | mjlab **−1.0**; **A1 / Go2 −10.0** | all | `mjlab.envs.mdp.rewards.joint_pos_limits` |
| `self_collision_cost` | Contacts above a force threshold | Go1 rough **−0.1** each | 2, 5, 7, 8 | `...velocity.mdp.rewards.self_collision_cost` |
| `is_alive` / `is_terminated` | Per-step bonus / termination penalty | ours **0.5** / **−20 to −40** | all | `mjlab.envs.mdp.rewards.*` |

`joint_torques_l2` reads `actuator_force` — the simulator's computed force from the
position-PD law, not a measured torque. That is the right choice here: Gray has no
torque sensing, so this is a training-time shaping term only.

---

## 2. Per stage

### 1 — Stand still (done)

Current set matches established practice, and two of our numbers were arrived at
independently and match exactly: **−0.0002 torque is A1/Go2's own config value**
(against ANYmal's −1e-5 — a light, weak-servo robot needs ~20× the torque penalty
of an ANYmal-class one), and **−0.01 action rate is the cross-framework default**.

One change worth testing: `joint_pos_limits` is at mjlab's default −1.0, but A1
and Go2 use **−10.0**. Gray's knee has only 86° of travel with the working end
right against a stop, so try −5 to −10 before stage 3.

### 2 — Take a push (in progress)

Keep `action_rate` at **−0.1 to −0.05**. For more smoothness add `action_acc_l2`
(≈ −0.01 to −0.05) rather than scaling action-rate further: a second-order penalty
kills high-frequency jitter without taxing the large, fast, low-frequency
corrections a real recovery needs.

Add `joint_acc_l2` ≈ −2.5e-7 — impact events spike joint acceleration.

**If smoothness pressure has to rise, ramp it in.** Every source that reports this
failure fixes it the same way: RMA ramps the whole penalty vector from k=0.03 via
`k_{t+1} = k_t^0.997`, because a fixed large penalty from step 0 makes the policy
"learn to stay in place because of the penalty terms".

### 3 — Lift one foot

**No mainstream framework has this stage at all** — legged_gym, Isaac Lab and every
paper surveyed go straight from standing to velocity tracking. Build from parts:
`upright`, `base_height`, `posture` on the three stance legs, `feet_slip` on the
stance feet only, and a written `foot_target` of the form
`exp(-‖foot_pos − target‖²/σ²)`.

For the trunk: the bar says "trunk moves less than 10 mm", but stage 1's `still`
wants *zero* velocity while this stage expects a deliberate shift onto the tripod.
Score `exp(-‖displacement − expected_shift‖²/σ²)`, not all motion.

### 4 — Step in place

The standard toolkit points the **wrong way** here. mjlab's `feet_air_time`,
`feet_clearance`, `feet_swing_height` and `feet_slip` are all gated off below a
command threshold, and legged_gym ships a `stand_still` reward whose whole purpose
is to punish joint motion at zero command. Every mainstream framework wants the
robot frozen when not commanded to move.

Needs a gait-phase reward that runs **independent of the velocity gate** — which
does not exist in mjlab. Port the Walk-These-Ways contact schedule (phase
`t += f_gait·dt`, Gaussian-CDF desired contact, `kappa ≈ 0.07`) driven by a
stepping-frequency command, and un-gate `feet_clearance` / `feet_swing_height`.

### 5 — Walk forward

| Term | Weight | Why |
|---|---|---|
| `track_linear_velocity` | 1.5–2.0, **σ retuned — see below** | the task |
| `feet_clearance` | −2.0 | stops the dragging leg Rudin describes |
| `feet_swing_height` | −0.25 | wrong peak height, scored at landing |
| `feet_slip` | −0.1 | stance foot sliding under load |
| `soft_landing` | −1e-5 to −1e-4 | keep **larger** than mjlab's default: printed parts, no suspension |
| `variable_posture` | 1.0 | fixed-std posture prevents the gait moving at all |
| `upright` | 1.0 | |
| `joint_pos_limits` | −1.0 to −5.0 | tight knee |
| `action_rate_l2` | −0.05 to −0.1 | do not repeat push_v3 |
| `action_acc_l2` | −0.01 to −0.05 | |
| `joint_torques_l2` | −0.0002 to −0.001 | 1.96 N·m stall vs Go1's ~23.7 |
| Gait phase **or** `feet_air_time` | 4.0 / 1.0 | **only if there is no hand-written gait underneath** |

#### The velocity-tracking σ — the single most important finding

Neither legged_gym nor Isaac Lab filter anything: both score the raw instantaneous
root velocity every step. They get away with it because σ = 0.5 m/s is huge next to
ANYmal/Go1's 0.6–1.5 m/s targets.

**Gray is in the opposite regime, and copying that σ causes two failures, not one.**

At a 0.2–0.3 m/s target with the footfall ripple measured at 4× the mean:

- **Ripple flattens the reward.** Peak instantaneous error ≈ 1.2 m/s →
  `exp(−1.2²/0.25) ≈ 0.003`. The reward collapses at every footfall. This is the
  failure already recorded in this project's history.
- **The same loose σ pays for standing still.** Command 0.3 m/s, robot stationary →
  `exp(−0.3²/0.25) ≈ 0.70`. **A policy that never walks scores 70% of maximum.**

That is very likely the mechanism behind "standing still scored better than
walking" in the previous attempt — and the same symptom is independently reported
in IsaacLab issue #458.

Both have one cause and one two-part fix:

1. **Filter the velocity before scoring it** — EMA or gait-cycle average, time
   constant ≈ 0.15–0.25 s, about one stride. Nothing in mjlab does this; it has to
   be written. (D²-GMBC sidesteps it by rewarding raw displacement instead.)
2. **Retune σ to Gray's speeds** — roughly 0.3–0.5× the target, so σ ≈ 0.1–0.15 m/s
   for a 0.2–0.3 m/s target, not the borrowed 0.5.

### 6 — Steer

Same set as stage 5 plus `track_angular_velocity`. mjlab's `UniformVelocityCommandCfg`
already has the knobs: drop `rel_forward_envs` toward 0 and enable `heading_command`.
No separate "go straight" penalty is needed — the heading controller generates the
yaw command and the existing angular-tracking term scores it.

### 7 — Stand up

The references are the two ANYmal recovery papers. Orientation is weighted
**highest** ("recovers upright ASAP"), plus joint position, joint acceleration
("smooth motions"), contact impulse ("avoid violent motions"), slippage,
self-collision, action difference and torque.

Mapped onto what we have: `upright` 1.0–2.0, `base_height` 2.0, `posture` 1.0–1.5,
`joint_acc_l2` −2.5e-7, `feet_slip` −0.1 to −0.3, `joint_torques_l2` as stage 5,
and `self_collision_cost` **repointed at a ground-contact sensor** to penalise
slamming the trunk down — the direct analog of the papers' impulse term, and
nothing else in mjlab covers it.

Two procedural lessons both papers state outright: **randomise the starting pose
broadly** (they drop from ~0.5 m with random joint angles, not one fixed pose), and
**ramp the penalties in from near zero** — same fix as RMA, same failure otherwise.

### 8 — Rough terrain

mjlab's `unitree_go1_rough_env_cfg` is close to a complete reference: `upright`
re-pointed at the terrain normal via `terrain_sensor_names`, clearance measured
against a per-foot `TerrainHeightSensor` rather than world Z, three −0.1 contact
penalties (self, shank-ground, trunk-ground), `illegal_contact` and
`out_of_terrain_bounds` terminations replacing the orientation-based one, and the
`terrain_levels_vel` curriculum.

Lee 2020's rough-terrain weights: linear velocity 0.05, angular 0.05, base motion
0.04, foot clearance 0.01, body collision 0.02, smoothness 0.025, torque 2e-5 —
clearance and collision stay small relative to tracking even on rough ground.

---

## 3. Known failure modes

**Standing still scoring well.** Four independent mechanisms produce it. Our own
history records a reward where standing beat walking. IsaacLab #458: a Solo12 given
ANYmal's weights "preferred to stand still"; a maintainer's reason is that reused
weights do not transfer because the actuator models and rewards were tuned for
those robots. RMA hit it with a fixed-weight penalty set. And, as derived above, a
σ borrowed from a faster robot **mathematically** pays ~70% of maximum for doing
nothing — the one most likely still live here, since mjlab's defaults are Go1-speed.

**Foot scuffing without a clearance term.** Rudin 2021, from their own runs:
"artifacts in the behavior, such as a dragging leg or unreasonably high or low base
heights."

**Jitter without an action-rate penalty.** CAPS (ICRA 2021) is the citation:
oscillation causes "poor control, high power consumption, and undue system wear";
their smoothness penalty cut real quadrotor power ~80%.

**Action-rate set too high.** Our own `push_v3` at −1.5 is the sharpest evidence
available: the robot stopped being able to catch itself. RMA's naive full-strength
penalty vector caused outright paralysis. Fix is the ramp, not a lower constant.

**Rushing the fall.** legged_gym clips the summed reward at ≥0 per step, its own
comment saying this "avoids early termination problems" — under net-negative
reward a policy can end the episode early rather than keep accruing penalty. Worth
watching with `fell_over` at −20 to −40.

**Simulating servos as stronger than they are.** D²-GMBC's own hardware result:
rear 23 kgf·cm servos saturated during backward walking and degraded that gait
direction by **57.6%** — a failure the reward never saw because the sim did not
model saturation.

**The structural finding.** Every project surveyed that actually ran RL on hobby
servos either escaped the actuator class (Stanford Pupper v3, Solo12, RealAnt all
moved to brushless or Dynamixel, citing no torque limiting and no overheat
protection) or worked around it (D²-GMBC, SpotMicroAI, rex-gym all use RL as a
residual on a hand-written gait generator rather than raw joint targets).

**Gray is in that class, not in the ANYmal/Go1 class mjlab's defaults were tuned
against.** Treat those numbers as a sound template but expect every effort, torque
and action-rate weight to need retuning by more than A1's already-20×-ANYmal
adjustment, and expect the biggest sim-to-real gap to appear as torque saturation
on hard manoeuvres — fast turns, catching a big shove, standing up — rather than as
a training-time reward problem.

---

## 4. Sources

**Frameworks**: legged_gym (`legged_robot_config.py`, `a1_config.py`,
`anymal_c_rough_config.py`), Isaac Lab (`velocity_env_cfg.py`, `mdp/rewards.py`,
`go2/rough_env_cfg.py`), unitree_rl_gym, and the installed mjlab 1.5.3 source.

- Rudin et al., *Learning to Walk in Minutes*, [arXiv:2109.11978](https://arxiv.org/abs/2109.11978)
- Margolis & Agrawal, *Walk These Ways*, [arXiv:2212.03238](https://arxiv.org/abs/2212.03238)
- Margolis et al., *Rapid Locomotion via RL*, [arXiv:2205.02824](https://arxiv.org/abs/2205.02824)
- Hwangbo et al., *Learning agile and dynamic motor skills*, [arXiv:1901.08652](https://arxiv.org/abs/1901.08652)
- Lee et al., *Quadrupedal locomotion over challenging terrain*, [arXiv:2010.11251](https://arxiv.org/abs/2010.11251)
- Lee et al., *Robust Recovery Controller*, [arXiv:1901.07517](https://arxiv.org/abs/1901.07517)
- Kumar et al., *RMA: Rapid Motor Adaptation*, [arXiv:2107.04034](https://arxiv.org/abs/2107.04034)
- Rahme et al., *D²-GMBC*, [arXiv:2010.12070](https://arxiv.org/abs/2010.12070)
- Tan et al., *Sim-to-Real: Agile Locomotion*, [arXiv:1804.10332](https://arxiv.org/abs/1804.10332)
- Mysore et al., *CAPS*, [arXiv:2012.06644](https://arxiv.org/abs/2012.06644)
- [IsaacLab issue #458](https://github.com/isaac-sim/IsaacLab/issues/458) — standing-still reward hacking
- Solo12 clearance reward, [arXiv:2309.16683](https://arxiv.org/abs/2309.16683)
