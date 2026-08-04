# Reward functions — one reference for every stage

What Gray actually scores, per stage, with the real weights, next to what the established
implementations use. Compiled from the installed mjlab source, this project's own run
history, and the published work on legged_gym, Isaac Lab, Unitree, ANYmal,
Walk-These-Ways, MIT Cheetah, D²-GMBC and the hobby-servo quadrupeds.

Findings from checking this config against that literature live in
[REWARD_AUDIT.md](REWARD_AUDIT.md). This file is the reference; that one is the
argument.

**A units note that makes every number below comparable.** mjlab, legged_gym and
Isaac Lab all compute `reward = raw × weight × dt`. Gray runs at `dt = 0.02 s`
(50 Hz), the same convention, so config weights transfer between those three
directly. They do **not** transfer from numbers printed in papers, which sometimes
quote the already-dt-multiplied value or use a different formula entirely. Every
paper-sourced figure below says so.

**A second consequence of that same `dt`, which is easy to miss.** A term that fires
every step is multiplied by 0.02 fifty times a second. A term that fires **once** —
`fell_over` — is multiplied by 0.02 and then never again. They are not on the same
scale, and comparing their weights directly is wrong. See audit finding 5.

---

## 1. One reward function, not three

There is one reward set, built by inheritance:

```
stand_env_cfg.py    10 terms
      │  push_env_cfg.py adds shove + DR, adds 4 terms, re-weights 5
      ▼
push_env_cfg.py     14 terms
      │  walk_env_cfg.py drops 7, adds 17, re-weights 1
      ▼
walk_env_cfg.py     24 terms   ← the live one
```

That is the industry norm and not an accident. legged_gym, Isaac Lab and mjlab all use a
single reward set for locomotion; standing still is the zero-velocity command, not a
separate task. A genuinely separate reward is normal only for a genuinely separate task —
fall recovery is the standard example, and **Gray has not written one yet.** Stage 7
below is a plan, not a file.

---

## 2. The live walk set — all 24 terms

Weights from `progress/runs/2026-08-03_14-42-29_walk_m3100_d/run.json`, which
`scripts/train.py` writes from the live config. Terms marked **↗** are ramped by the
curriculum; the arrow shows start → end.

### Pays (9 terms, ceiling 9.5 / s)

| Term | Weight | What it measures | Where |
|---|---|---|---|
| `track_speed` | **2.0** | `exp(−‖cmd_xy − v̄_xy‖²/σ²)`, σ = 0.15, on a **12-step EMA** of trunk speed | ours, [walk_env_cfg.py:262](../gray/tasks/walk_env_cfg.py#L262) |
| `track_turn` | **1.0** | `exp(−(yaw err² + ‖ω_xy‖²)/σ²)`, σ = 0.80 | `vmdp.track_angular_velocity` |
| `ground_covered` | **1.0** | net world-X progress ÷ distance asked, clamped [0,1] | ours, [:414](../gray/tasks/walk_env_cfg.py#L414) — **see audit finding 1** |
| `upright` | **1.0** | `exp(−(pitch err² + roll err²)/σ²)`, σ = 0.45, vs the **posture command** | ours, [:398](../gray/tasks/walk_env_cfg.py#L398) |
| `ride_height` | **1.0** | `exp(−(z − cmd_z)²/σ²)`, σ = 0.05, vs the **posture command** | ours, [:384](../gray/tasks/walk_env_cfg.py#L384) |
| `posture` | **1.0** | joints near default, tolerance widening with commanded speed | `vmdp.variable_posture` |
| `stepping` | **1.0** | fraction of feet with air time in 0.10–0.45 s | ours, [:269](../gray/tasks/walk_env_cfg.py#L269) — **see audit finding 2** |
| `leg_swing` | **1.0** | mean thigh/calf excursion about a running average, capped at 0.15 rad | ours, [:439](../gray/tasks/walk_env_cfg.py#L439) |
| `alive` | **0.5** | per-step bonus while not terminated | `mdp.is_alive` |

Every one of these genuinely caps at 1.0 — six are `exp(−·)`, two are clamped, one is
binary — so the 9.5 ceiling `RULES.md` rule 1 depends on is real.

### Charges (15 terms)

| Term | Weight | What it measures | Where |
|---|---|---|---|
| `fell_over` | **−40** | termination penalty, fires once | `mdp.is_terminated` |
| `end_stops` | **−5.0** | soft joint-limit violation | `mdp.joint_pos_limits` |
| `dragging` | **−2.0** | `Σ ‖height − 0.035‖ × ‖v_xy‖` per foot, gated on MOVING | ours, [:286](../gray/tasks/walk_env_cfg.py#L286) |
| `veering` ↗ | **−0.2 → −2.0** | accumulated heading error off the commanded line, squared; caps at π² so **−19.7 at full ramp** | ours, [:482](../gray/tasks/walk_env_cfg.py#L482) |
| `wandering` | **−1.0** | cross-track displacement past 50 mm, **no upper bound** | ours, [:505](../gray/tasks/walk_env_cfg.py#L505) |
| `skidding` | **−0.5** | `Σ ‖v_xy‖²` per foot while in contact | `push_env_cfg.foot_slip` |
| `swing_height` | **−0.25** | `(peak/0.035 − 1)²`, charged at landing | ours, [:303](../gray/tasks/walk_env_cfg.py#L303) |
| `landing_speed` ↗ | **−0.1 → −0.8** | downward foot speed at the instant of touchdown | ours, [:333](../gray/tasks/walk_env_cfg.py#L333) |
| `rocking` ↗ | **−0.02 → −0.2** | `Σ ω_xy²` on the trunk only | `vmdp.body_angular_velocity_penalty` |
| `twitching` ↗ | **−0.01 → −0.05** | `Σ (aₜ − aₜ₋₁)²` | `mdp.action_rate_l2` |
| `jitter` | **−0.01** | `Σ (aₜ − 2aₜ₋₁ + aₜ₋₂)²` | `mdp.action_acc_l2` |
| `shaking` ↗ | **−0.001 → −0.01** | trunk's own linear acceleration, squared | ours, [:463](../gray/tasks/walk_env_cfg.py#L463) |
| `effort` | **−0.0002** | `Σ τ²` across the twelve joints | `mdp.joint_torques_l2` |
| `hard_landing` | **−1e-4** | contact force at touchdown | `vmdp.soft_landing` |
| `joint_shock` | **−2.5e-7** | `Σ q̈²` | `mdp.joint_acc_l2` |

Every term that pays is bounded at 1.0. `veering` and `wandering` are the only two that
charge without a useful bound, and `veering` is also the one that ramps hardest — at full
ramp it can charge more than twice the entire positive ceiling. Audit finding 6.

### The curriculum

Five terms ramp in over the first 12,000 env steps (≈ iteration 500), driven by
`vmdp.reward_curriculum` at [walk_env_cfg.py:737](../gray/tasks/walk_env_cfg.py#L737).
Steps 0 / 2400 / 6000 / 12000 land at roughly iteration 0, 100, 250 and 500.

This is not tidiness. RMA ramps its whole penalty vector from k=0.03 via
`k_{t+1} = k_t^0.997` because a fixed large penalty from step 0 makes the policy "learn to
stay in place because of the penalty terms". This project hit the same wall independently:
`push_v3` set `twitching` to −1.5 from step 0 and the robot stopped being able to catch
itself. LoComposition (2026) ramps its power penalty the same way. Three sources, one
lesson: **first learn to walk, then learn to walk neatly.**

### What walk drops, and why

| Dropped | Reason |
|---|---|
| `still` | it is being asked to move |
| `joint_speed` | same; legged_gym disables this for walking too |
| `foot_lift` | `stepping` and `swing_height` are the gait versions of it |
| `spinning` | it fined *all* trunk rotation including the commanded yaw — paying and fining the same motion |
| `tilt` | replaced by `upright`, the same measurement scored the way the walking references score it |
| `height`, fixed `posture` | replaced by the commanded-posture and variable-tolerance versions |

### The commands

Two command terms on the same 5–10 s clock, so one draw is one coherent instruction.

**`walk`** — `StraightLineVelocityCommandCfg`, `vx ∈ ±0.35`, `vy ∈ ±0.20`,
`ωz ∈ ±1.00`. 15% standing; 50% of the rest redrawn as a pure straight line, forwards
**or backwards**, with `|vx| ≥ 0.10`. The subclass exists because mjlab's
`rel_forward_envs` does `abs().clamp(min=0.3)`, which erases the sign — on the widened box
that would have flipped 80% of backward commands back to forward and the measurement
afterwards would have said backward still does not work.

**`posture`** — height 0.15–0.25 m, pitch −0.26 to +0.14 rad, roll ±0.35 rad, 25% at
nominal. This is what turns `upright` and `ride_height` from fixed targets into commands:
level and stance height are simply the command at zero, and a crouch or a lean becomes
something you can ask for instead of something the robot is punished for.

### The three numbers that were wrong and are now right

Worth keeping in this file because each cost runs to find.

1. **σ on `track_speed` = 0.15, not mjlab's 0.5.** At 0.5, a robot commanded 0.25 m/s and
   standing perfectly still collects `exp(−0.25²/0.5²) = 0.78` — 78% of full marks for
   doing nothing. At 0.15 it collects 0.06. IsaacLab issue #458 reports the same failure
   independently, and it is very likely what the archived attempt did.
2. **σ on `track_turn` = 0.80, not 0.30.** The same trap in the other direction. At 0.30
   the yaw error the robot actually made — 1.5 rad/s, flat across 3000 iterations of
   `walk_m3100_a` and `_b` — paid `1.4e-11`. Not a small reward, a **dead** one: no
   gradient, so the policy was never told to stop spinning. Raising the weight does
   nothing; twice zero is zero.
3. **MOVING = 0.05, not mjlab's 0.5.** Four terms gate themselves off below a command
   threshold, and Go1's 0.5 is above anything Gray is ever commanded. On the defaults the
   term that pays for picking a foot up is dead for the entire run.

**The generalisation, which is the actually useful part:** a tracking band has to be set
against the error the robot *makes*, not only against the command it is *given*. Both
numbers above were wrong for the same reason, in opposite directions.

---

## 3. Reference values from the established implementations

For any term you are thinking of adding or re-weighting.

### Task tracking

| Term | Typical weight | Where |
|---|---|---|
| `track_linear_velocity` | mjlab **2.0**; legged_gym / Isaac Lab **1.0**; Isaac Go2 **1.5** | `mjlab.tasks.velocity.mdp.rewards.track_linear_velocity` |
| `track_angular_velocity` | mjlab **2.0**; legged_gym **0.5** | `...rewards.track_angular_velocity` |
| `feet_air_time` | mjlab **0.0 — shipped disabled on Go1**; legged_gym **1.0**; ANYmal-flat **2.0**; Isaac Lab **0.125** | `...rewards.feet_air_time` |
| Raibert foot placement | Walk-These-Ways raw **−10.0** | must write |
| Gait-phase / contact schedule | Walk-These-Ways raw **4.0** | must write — mjlab has **no** phase reward |

### Posture and stability

| Term | Typical weight | Where |
|---|---|---|
| `posture` | ours **1.5** (stand), std 0.35 | `mjlab.envs.mdp.rewards.posture` |
| `variable_posture` | mjlab **1.0**; Go1 std 0.05→0.3 hip/thigh, 0.1→0.6 calf | `...velocity.mdp.rewards.variable_posture` |
| `upright` | mjlab **1.0**, σ = √0.2 | `...velocity.mdp.rewards.upright` |
| `flat_orientation_l2` | ours **−2.0** (stand, "tilt") | `mjlab.envs.mdp.rewards.flat_orientation_l2` |
| `body_angular_velocity_penalty` | mjlab **0.0** (off); ANYmal **−6·dt** | `...velocity.mdp.rewards.body_angular_velocity_penalty` |

### Gait and foot quality

| Term | Typical weight | Where |
|---|---|---|
| `feet_clearance` | mjlab **−2.0**, target 0.1 m | `...velocity.mdp.rewards.feet_clearance` |
| `feet_swing_height` | mjlab **−0.25** | `...rewards.feet_swing_height` |
| `feet_slip` | mjlab **−0.1**; Walk-These-Ways **−0.04** | `...rewards.feet_slip` |
| `soft_landing` | mjlab **−1e-5** | `...rewards.soft_landing` |

mjlab's `feet_clearance` and `feet_swing_height` both require a `TerrainHeightSensor`.
On a flat floor, height above the environment origin is the same number, which is why
Gray rewrites both against the foot sites. **On rough terrain (stage 8) they must go back
to the real sensor.**

### Smoothness, effort, actuator protection

| Term | Typical weight | Where |
|---|---|---|
| `action_rate_l2` | **−0.01** is the cross-framework anchor. Walk-These-Ways −0.1 on PD-target delta. Mini Cheetah −2e-4. **We tried −1.5 and it broke fall-catching.** | `mjlab.envs.mdp.rewards.action_rate_l2` |
| `action_acc_l2` | Walk-These-Ways equivalent **−0.1** | `mjlab.envs.mdp.rewards.action_acc_l2` |
| `joint_acc_l2` | legged_gym / Isaac Lab **−2.5e-7** | `mjlab.envs.mdp.rewards.joint_acc_l2` |
| `joint_torques_l2` | ANYmal **−1e-5**; **A1 / Go2 −0.0002**; ours **−0.0002** | `mjlab.envs.mdp.rewards.joint_torques_l2` |
| `joint_vel_l2` | ours **−0.001** (stand only); legged_gym **0** | `mjlab.envs.mdp.rewards.joint_vel_l2` |
| `electrical_power_cost` | Minitaur equivalent 0.008; LoComposition **0.008** ramped | `mjlab.envs.mdp.rewards.electrical_power_cost` — **not in any Gray task. Audit finding 3.** |
| `joint_pos_limits` | mjlab **−1.0**; **A1 / Go2 −10.0**; ours **−5.0** | `mjlab.envs.mdp.rewards.joint_pos_limits` |
| `self_collision_cost` | Go1 rough **−0.1** each | `...velocity.mdp.rewards.self_collision_cost` |

`joint_torques_l2` reads `actuator_force` — the simulator's computed force from the
position-PD law, not a measured torque. That is the right choice here: Gray has no torque
sensing, so this is a training-time shaping term only.

---

## 4. Per stage

### 1 — Stand still (done)

10 terms. `height` 2.0 (std 0.03), `still` 1.0 (std 0.25), `posture` 1.5 (std 0.35),
`alive` 0.5, `tilt` −2.0, `fell_over` −20, `effort` −0.0002, `joint_speed` −0.001,
`twitching` −0.01, `end_stops` −1.0.

Two of these were arrived at independently and match the references exactly:
**−0.0002 torque is A1/Go2's own config value** (against ANYmal's −1e-5 — a light,
weak-servo robot needs ~20× the torque penalty of an ANYmal-class one), and **−0.01
action rate is the cross-framework default**.

### 2 — Take a push (done)

Adds `spinning` −0.05, `skidding` −2.0, `foot_lift` +0.3, `jitter` −0.01. Re-weights
`posture` → 0.6, `still` → 0.5, `joint_speed` → −0.0002, `end_stops` → −5.0,
`fell_over` → −40.

`end_stops` at −5.0 is a step toward A1/Go2's −10.0 rather than a jump to it: Gray's knee
has 86° of travel in total and the stance already sits about 3° past the stop, so there is
no slack to spend.

`jitter` — the **second** difference of the action — is the right way to buy smoothness,
and Kim et al. use exactly this pair (first *and* second difference of the PD target) as
their single smoothness reward. `push_v3` tried to get the same thing by scaling
`twitching` 150× instead and the robot stopped catching itself: action-rate taxes speed,
and catching a fall **is** a large fast movement.

### 3 — Lift one foot (skipped)

**No mainstream framework has this stage at all** — legged_gym, Isaac Lab and every paper
surveyed go straight from standing to velocity tracking. Skipped here on purpose too.

If it is ever built: `upright`, `ride_height`, `posture` on the three stance legs,
`skidding` on the stance feet only, and a written `foot_target` of the form
`exp(−‖foot_pos − target‖²/σ²)`. For the trunk, score
`exp(−‖displacement − expected_shift‖²/σ²)` rather than all motion — the bar wants a
deliberate shift onto the tripod, not stillness.

### 4 — Step in place (skipped)

The standard toolkit points the **wrong way** here. mjlab's foot terms are all gated off
below a command threshold, and legged_gym ships a `stand_still` reward whose whole purpose
is to punish joint motion at zero command. Every mainstream framework wants the robot
frozen when not commanded to move.

Doing it properly needs a gait-phase reward that runs **independent of the velocity
gate** — which does not exist in mjlab. Port the Walk-These-Ways contact schedule
(phase `t += f_gait·dt`, Gaussian-CDF desired contact, `kappa ≈ 0.07`) driven by a
stepping-frequency command.

**Note the accident.** Gray's `stepping` term currently *is* un-gated, so this stage
partly happens by mistake inside stage 5 — for the 15% of environments commanded to stand.
That is audit finding 2, and it is a bug rather than a feature.

### 5 — Walk (live)

Section 2 above is this stage. Nothing to add.

### 6 — Steer

Same set as stage 5. mjlab's `UniformVelocityCommandCfg` already has the knobs: enable
`heading_command` and let the heading controller generate the yaw command; the existing
`track_turn` scores it. No separate "go straight" penalty is needed beyond `veering` and
`wandering`, which already exist.

### 7 — Stand up (not written)

The references are the two ANYmal recovery papers. Orientation is weighted **highest**
("recovers upright ASAP"), plus joint position, joint acceleration ("smooth motions"),
contact impulse ("avoid violent motions"), slippage, self-collision, action difference and
torque.

Mapped onto what exists: `upright` 1.0–2.0, `ride_height` 2.0, `posture` 1.0–1.5,
`joint_shock` −2.5e-7, `skidding` −0.1 to −0.3, `effort` as stage 5, and
`self_collision_cost` **repointed at a ground-contact sensor** to penalise slamming the
trunk down — the direct analog of the papers' impulse term, and nothing else in mjlab
covers it.

Two procedural lessons both papers state outright: **randomise the starting pose broadly**
(they drop from ~0.5 m with random joint angles, not one fixed pose), and **ramp the
penalties in from near zero** — same fix as RMA, same failure otherwise.

This is the one stage that genuinely warrants its own reward function rather than another
layer on the chain, because it is a different task, not a different command.

### 8 — Rough terrain (not written)

mjlab's `unitree_go1_rough_env_cfg` is close to a complete reference: `upright` re-pointed
at the terrain normal via `terrain_sensor_names`, clearance measured against a per-foot
`TerrainHeightSensor` rather than world Z, three −0.1 contact penalties (self,
shank-ground, trunk-ground), `illegal_contact` and `out_of_terrain_bounds` terminations
replacing the orientation-based one, and the `terrain_levels_vel` curriculum.

Lee 2020's rough-terrain weights: linear velocity 0.05, angular 0.05, base motion 0.04,
foot clearance 0.01, body collision 0.02, smoothness 0.025, torque 2e-5 — clearance and
collision stay small relative to tracking even on rough ground.

LoComposition (2026) disagrees with the whole approach and is worth reading before this
stage is built: it finds air-time and contact-count priors **reduce** rough-terrain
traversability, and gets better results from velocity tracking plus a mechanical-power
penalty and nothing else.

---

## 5. Known failure modes

**Standing still scoring well.** Five independent mechanisms produce it. Our own history
records a reward where standing beat walking. IsaacLab #458: a Solo12 given ANYmal's
weights "preferred to stand still", and a maintainer's reason is that reused weights do
not transfer because the actuator models and rewards were tuned for those robots. RMA hit
it with a fixed-weight penalty set. A σ borrowed from a faster robot **mathematically**
pays ~78% of maximum for doing nothing. And van Marum et al. state the general version
outright: *"most reward terms will be greater when a policy stands still with both feet on
the ground, than when it steps in place."* The bias is structural, not a bug you fix once.

**A term that returns zero looks exactly like a term that is passing.** This project has
been bitten four separate times: `feet_air_time` dead behind a Go1 threshold, `track_turn`
saturated at 1.4e-11 for three runs, `_going_straight` false for every backward command,
`ground_covered` paying nothing for backward walking right now. None of these throw. None
look wrong on a curve. The countermeasure is to log every term's contribution and hunt for
the ones sitting at zero — see [REWARD_AUDIT.md](REWARD_AUDIT.md).

**Foot scuffing without a clearance term.** Rudin 2021, from their own runs: "artifacts in
the behavior, such as a dragging leg or unreasonably high or low base heights."

**Jitter without an action-rate penalty.** CAPS (ICRA 2021): oscillation causes "poor
control, high power consumption, and undue system wear"; their smoothness penalty cut real
quadrotor power ~80%.

**Action-rate set too high.** Our own `push_v3` at −1.5 is the sharpest evidence
available: the robot stopped being able to catch itself. RMA's naive full-strength penalty
vector caused outright paralysis. **Fix is the ramp, not a lower constant.**

**Rushing the fall.** legged_gym clips the summed reward at ≥0 per step, its own comment
saying this "avoids early termination problems" — under net-negative reward a policy can
end the episode early rather than keep accruing penalty. **mjlab does not clip.** Gray is
currently net-positive throughout, but the two unbounded penalties ramp 10× and this is
the thing to watch.

**Too many terms to tune.** Kim et al.: "there are often more than ten reward terms" and
the tuning "often takes several months". Gray has 24.

**Simulating servos as stronger than they are.** D²-GMBC's own hardware result: rear
23 kgf·cm servos saturated during backward walking and degraded that gait direction by
**57.6%** — a failure the reward never saw because the sim did not model saturation.

**The structural finding.** Every project surveyed that actually ran RL on hobby servos
either escaped the actuator class (Stanford Pupper v3, Solo12, RealAnt all moved to
brushless or Dynamixel, citing no torque limiting and no overheat protection) or worked
around it (D²-GMBC, SpotMicroAI, rex-gym all use RL as a residual on a hand-written gait
generator rather than raw joint targets).

**Gray is in that class, not in the ANYmal/Go1 class mjlab's defaults were tuned against.**
Treat those numbers as a sound template but expect every effort, torque and action-rate
weight to need retuning by more than A1's already-20×-ANYmal adjustment, and expect the
biggest sim-to-real gap to appear as torque saturation on hard manoeuvres — fast turns,
catching a big shove, standing up — rather than as a training-time reward problem.

---

## 6. Sources

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
- Kim et al., *Not Only Rewards But Also Constraints*, [arXiv:2308.12517](https://arxiv.org/abs/2308.12517)
- van Marum et al., *Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking*, [arXiv:2404.19173](https://arxiv.org/abs/2404.19173)
- *LoComposition: Terrain-Adaptive Energy-Efficient Quadruped Locomotion without Gait Priors*, [arXiv:2606.15896](https://arxiv.org/abs/2606.15896)
- *Learning-based legged locomotion: state of the art and future perspectives*, [arXiv:2406.01152](https://arxiv.org/abs/2406.01152)
- Chamorro, [legged-gym reward-term ablation](https://wandb.ai/simonchamorro/legged-gym/reports/Legged-Locomotion-Environment-Experiments-in-Isaac-Gym--VmlldzoxODE2MDY0)
- Voelcker, [Reward Design and Termination](https://cvoelcker.de/blog/2025/reward-functions/)
- [IsaacLab issue #458](https://github.com/isaac-sim/IsaacLab/issues/458) — standing-still reward hacking
- Solo12 clearance reward, [arXiv:2309.16683](https://arxiv.org/abs/2309.16683)
