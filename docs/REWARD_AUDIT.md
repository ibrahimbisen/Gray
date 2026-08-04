# Reward audit — 3 Aug 2026

Every scoring term in the live walk task, checked against the published work. Seven
findings, ranked by how likely each is to be costing something right now. Everything
checked and found sound is recorded at the bottom, because a term that was examined and
cleared is worth knowing about too.

The config audited is the one `progress/runs/2026-08-03_14-42-29_walk_m3100_d/run.json`
was written from: **24 scoring terms** — 9 that pay, 15 that charge — with a ceiling of
9.5 per second, 190 over a 20 s episode.

**One arithmetic fact that everything below depends on.** mjlab computes
`reward = raw × weight × dt` with `dt = 0.02 s`
([reward_manager.py:116](../.venv/Lib/site-packages/mjlab/managers/reward_manager.py#L116)).
A term that fires every step accumulates 50 times per second. A term that fires **once**
— falling over — is multiplied by 0.02 and never repeats. Those two are not on the same
scale, and finding 5 is entirely about that.

---

## 1. `ground_covered` pays nothing for walking backward, and full marks for drifting while told to stand

**What the code does.** [walk_env_cfg.py:414-436](../gray/tasks/walk_env_cfg.py#L414-L436).
It measures displacement along **world X**, divides by the distance asked for, and clamps
the result to `[0, 1]`.

**Failure A — backward.** A backward command produces negative world-X displacement.
`clamp(negative, 0, 1)` is 0. A robot walking backward perfectly earns exactly the same
from this term as one that does not move at all: nothing.

How much of the command box that is:

| share of envs | command | `ground_covered` earns |
|---|---|---|
| 15% | standing (zeroed in `_update_command`) | see failure B |
| ~21% | straight, `vx ∈ [-0.35, -0.10]` | **0** |
| ~21% | straight, `vx ∈ [+0.10, +0.35]` | works |
| ~43% | general draw, half of which has `vx < 0` | **0** for about half |

Roughly **40% of all environments cannot earn this term at all.** They can only lose the
1.0 it is worth.

**Failure B — standing.** For a standing env `asked = 0`, and the code guards with
`clamp(asked, min=1e-6)`. So `progress / 1e-6` — any forward drift at all, a micron of
it — saturates the clamp and pays the **full 1.0**. A robot told to stand still is paid
maximum for creeping forward.

**Failure C — turning.** World X is not the commanded direction once the robot has turned.
The term drifts out of alignment with the command over an episode.

**Why this matters beyond the term itself.** `gray_skill_library.csv` row 8 — "Walk
backward, sideways and diagonally" — is filed `blocked_on: range`, with the note that run
25 covered 0.00 m backward in 8 s. The command box was widened through zero on 3 Aug 2026
to fix that. This term was not changed in the same edit, so widening the box alone will
not unblock row 8: the robot is now *asked* to go backward and still *paid* nothing
for doing it.

This is the same shape as the `_going_straight` positive-only trap that was caught in that
change ([walk_env_cfg.py:368](../gray/tasks/walk_env_cfg.py#L368)). That one was fixed.
This one was next to it and was missed.

**Fix.** Project displacement onto the commanded direction instead of onto world X, and
gate the term off below `MOVING` so the standing case cannot pay:

```
want = command[:, :2] / ||command[:, :2]||        # unit vector, body frame
progress = (pos_t - pos_{t-1}) · want              # signed, along the command
return clamp(progress / (||command[:, :2]|| · dt), 0, 1) * (||cmd|| > MOVING)
```

That is the same rotation `wandering` already does at
[walk_env_cfg.py:551](../gray/tasks/walk_env_cfg.py#L551).

---

## 2. `stepping` is not command-gated, so marching in place beats standing still

**What the code does.** [walk_env_cfg.py:269-277](../gray/tasks/walk_env_cfg.py#L269-L277).
Fraction of the four feet whose air time is between 0.10 s and 0.45 s. Weight 1.0. It
carries no command gate.

**The conflict.** 15% of environments are commanded to stand
(`rel_standing_envs=0.15`). For those:

- feet planted → all air times 0 → `stepping = 0`
- marching on the spot → up to `stepping = 1.0`

And nothing charges for the marching, because every gait *penalty* **is** gated on
`MOVING`: `dragging`, `swing_height` and `hard_landing` all multiply by
`moving.float()`, and `leg_swing` gates the same way. Only `skidding`, `landing_speed`,
`joint_shock`, `shaking` and `effort` push back, and together they are worth a fraction of
1.0.

So for one environment in seven the reward strictly prefers marching in place to standing
still. `track_speed` does not object — a robot marching without translating still scores
~1.0 on it.

legged_gym ships a `stand_still` reward whose entire job is the opposite of this. Gray has
neither the gate nor the counter-term.

**Fix.** Gate `stepping` on `MOVING`, the same way `leg_swing` already is. One line.

---

## 3. Three explicit gait priors, no energy term — the 2026 result says that is backwards

Gray pays for gait shape three separate ways: `stepping` (air time), `leg_swing` (joint
excursion), `swing_height` (peak height). It has **no** power or energy term anywhere in
any of the three tasks.

LoComposition ([arXiv:2606.15896](https://arxiv.org/abs/2606.15896)) tested exactly this
trade and found it the wrong way round:

- Their whole reward is **two things**: velocity tracking, and a mechanical-power penalty
  `λ_E · Σ|τ_j q̇_j|` ramped in and saturating at 0.008.
- Gait priors — they name **air time and contact counts specifically** — "reduce terrain
  traversability and prevent lower-CoT solutions". The variants *with* priors performed
  **worse**.
- Without the power term the policy "settles on an energetically expensive bounding gait".
  With it, trotting appears on its own, at **56% lower cost of transport**.

Their framing is the useful part: energy is not a tidiness penalty, it is *the signal that
picks the gait* out of the feasible set. Gray currently picks the gait by hand, with three
terms, and then has nothing selecting for efficiency.

This matters more here than on their robot, not less. Gray's binding hardware constraint
is 1.96 N·m servos with no torque limiting and no overheat protection. `effort`
(`joint_torques_l2`, −0.0002) prices torque but not *power* — a joint holding a heavy
static load and a joint doing real work look the same to it.

`electrical_power_cost` already exists in mjlab
(`mjlab.envs.mdp.rewards.electrical_power_cost`, `sum(clamp(τ·q̇, min=0))`) and
`REWARDS.md` already lists it for stages 5 and 8. It is simply not in the config.

**Suggestion, not a fix.** Add `electrical_power_cost`, ramped in the way the other
tidiness terms already are, starting near zero. Then try dropping `stepping` and see
whether the gait survives. That is a cheap experiment and it is the one the paper says
pays.

---

## 4. 24 terms is past where the literature says hand-tuning stops working

Term counts, like for like:

| | terms | note |
|---|---|---|
| LoComposition (2026) | **2** | tracking + power |
| Kim et al. (2023) | **3** rewards + 11 constraints | tracking, torque, smoothness |
| Digit standing/walking (2024) | **14** | described as "minimally constraining" |
| legged_gym / Isaac Lab defaults | 10–12 | |
| **Gray walk** | **24** | 9 pay, 15 charge |

Kim et al. ([arXiv:2308.12517](https://arxiv.org/abs/2308.12517)) state the problem
directly: *"Finding the relative weights for each of the reward terms based on the
resulting robot's motion is not trivial because there are often more than ten reward
terms"*, and that this tuning *"often takes several months"*. Their answer is to move the
regularisation terms out of the reward and into hard constraints with physically meaningful
thresholds — joint limits, contact, foot clearance — leaving three coefficients to tune.
Eleven of their twelve constraint thresholds can be read straight out of the robot
description file rather than tuned at all.

Gray's terms cluster into groups that price overlapping things:

- **"move"** — `track_speed` 2.0, `ground_covered` 1.0, `leg_swing` 1.0, `stepping` 1.0
- **"trunk steady"** — `upright` 1.0, `rocking` →−0.2, `shaking` →−0.01
- **"hold a line"** — `track_turn` 1.0, `veering` →−2.0, `wandering` −1.0
- **"smooth"** — `twitching` →−0.05, `jitter` −0.01, `joint_shock` −2.5e-7
- **"feet"** — `dragging` −2.0, `skidding` −0.5, `swing_height` −0.25,
  `landing_speed` →−0.8, `hard_landing` −1e-4

Overlap is not automatically wrong, and the code argues each pair catches a distinct
failure — that argument is generally sound and in two cases (`veering` vs `wandering`,
`twitching` vs `jitter`) it is demonstrably right, because both were found the hard way.
But five clusters of three to five terms each is where the cited failure lives, and Gray
has no mechanism for noticing when two of them start fighting other than a run going
wrong.

**Not a recommendation to cut terms blindly.** It is a recommendation to log the
per-term contribution and see which of the 24 are actually earning. `metrics.csv`
currently carries 5 of the 24 reward terms. rsl_rl already writes `Episode_Reward/*` for
every one of them to tensorboard; `WATCH` in [scripts/train.py:115](../scripts/train.py#L115)
just doesn't forward them. Widening `WATCH` is the cheapest useful thing on this list.

---

## 5. `fell_over` at −40 is worth about 5% of what falling actually costs

This is the non-obvious one, and it is pure `dt` arithmetic.

`is_terminated` fires **once**. Its contribution is `1 × −40 × 0.02 = −0.8`.

What falling actually costs is the reward that stops arriving. At convergence run `_d`
scores 137.4 per 20 s episode, so the per-step reward is `137.4 / 20 = 6.87` before dt, or
0.137 after. With `gamma = 0.99` the value of carrying on is

```
0.137 / (1 − 0.99) ≈ 13.7
```

So falling costs about **14.5**, of which the termination penalty is **0.8 — 5.5%**.

Two consequences:

- Raising `fell_over` from −20 to −40 in the push task changed the cost of falling by
  about 2.7%. It very likely did nothing. If falling needs to cost more, the lever is
  `gamma` or the ongoing reward, not this weight.
- Early in training the ratio is different. At iteration 0 the per-step reward is
  `0.589 / (19.18 × 0.02) ≈ 1.54` before dt, so continuation is worth ~3.1 and the
  penalty is ~21% of the cost. The term does real work at the start and almost none at
  the end, which is the opposite of how it reads.

Claas Voelcker's write-up of the sign/termination interaction states the general rule:
under negative rewards termination is a *reward* to the agent, under positive rewards it
is a punishment, and mixing them makes the behaviour hard to predict. Gray mixes them —
`alive` +0.5 and `fell_over` −40 — which is normal practice, but it means the two have to
be read together and neither number means much alone.

---

## 6. mjlab does not clip the summed reward at zero. legged_gym does

`RewardManager.compute()`
([reward_manager.py:116-133](../.venv/Lib/site-packages/mjlab/managers/reward_manager.py#L116-L133))
sums the weighted terms and returns them. There is no `clip(min=0)`. legged_gym has
`only_positive_rewards`, and its own comment says the clip *"avoids early termination
problems"* — under a net-negative step reward, ending the episode beats carrying on, and
the agent learns to fall on purpose.

**Checked: not currently happening.** Mean step reward is positive at iteration 0 (≈1.54)
and at convergence (≈6.87), on a 9.5 ceiling.

**But the margin shrinks by design.** The curriculum
([walk_env_cfg.py:737-750](../gray/tasks/walk_env_cfg.py#L737-L750)) raises `veering`
−0.2 → −2.0, `landing_speed` −0.1 → −0.8, `rocking` −0.02 → −0.2, `shaking` −0.001 →
−0.01 and `twitching` −0.01 → −0.05 across the first 12,000 steps. That is the one time
this could bite: a policy that has *not* learned to hold a line by iteration 500 gets
`veering` at −2.0. `veering` returns `wrap_to_pi(...)²`, which is bounded — but at π²,
so at −2.0 a robot 90° off its line pays 4.9 per step against a 9.5 ceiling, and a robot
facing backwards pays **19.7, more than twice everything the reward can pay it.**
`wandering` is worse: `clamp(off − 0.05, min=0)` has no upper bound at all and grows with
distance.

**Suggestion.** Cap both. Every other term in the file is bounded — six exponentials, two
clamps, a binary — and `RULES.md` rule 1's stop threshold is defined against a ceiling
that only the positive side actually respects.

---

## 7. `landing_speed` has no plain-English note, and `train.py` warns instead of refusing

`run.json` for the live run ships `"name": "landing_speed", "what": ""` — a blank row on
the dashboard, which the code's own comment calls *"worse than no row"*.

The cause: `landing_speed` was added at
[walk_env_cfg.py:720](../gray/tasks/walk_env_cfg.py#L720) but never added to `WALK_NOTES`.

`CLAUDE.md` says *"train.py refuses to start if one is missing"*. It does not —
[scripts/train.py:386-390](../scripts/train.py#L386-L390) prints `[warn]` and carries on.
The rule that was supposed to enforce itself printed one line into a training log and was
never seen.

**Fix.** Add the note, and make the check `raise` so the documented behaviour and the
actual behaviour agree.

---

## Checked and found sound

- **The tracking band, σ = 0.15.** Correct, and the reasoning behind it holds. The
  standing-still exploit it was written to kill is genuinely dead: at a commanded
  0.25 m/s a stationary robot now scores 0.06 instead of 0.78. Re-checked against the
  *widened* box: the only residual is envs drawn with a very small non-zero command
  (~6% of envs, worth ≤0.9 instead of 1.0), which is negligible and points the right way
  anyway. **Do not change this number.**
- **The turning band, TURN_STD = 0.80.** The reasoning in the file — set the band against
  the error the robot *makes*, not the command it is *given* — is a genuinely good
  generalisation and I did not find it stated this clearly in any of the papers.
- **The reward ceiling of 9.5 is real.** Every one of the nine positive terms genuinely
  caps at 1.0: six are `exp(−·)`, `ground_covered` and `leg_swing` are clamped, `stepping`
  is divided by 4, `alive` is binary. `RULES.md` rule 1 is safe on the positive side.
  (The negative side is finding 6.)
- **Ramping the tidiness penalties.** Matches RMA and matches this project's own `push_v3`
  failure. LoComposition ramps its power penalty the same way. Correct.
- **Exponential kernels on tracking terms.** The survey
  ([arXiv:2406.01152](https://arxiv.org/abs/2406.01152)) recommends bounded functions —
  clipping or exponential kernels — for stable training. Gray does this throughout.
- **`end_stops` −5.0 and `fell_over` −40.** Not stale inherited values; both were
  deliberately re-weighted in [push_env_cfg.py:194-204](../gray/tasks/push_env_cfg.py#L194-L204)
  with the knee-travel reasoning written out. (`fell_over` is finding 5 for a different
  reason.)
- **`jitter` (second difference) instead of a bigger `action_rate`.** This is the right
  call and the literature agrees: Kim et al. use exactly this pair — first *and* second
  difference of the PD target — as their single smoothness reward.
- **Contact-sensor slip detection rather than a height threshold.**
  [push_env_cfg.py:65-77](../gray/tasks/push_env_cfg.py#L65-L77). Correct, and the
  reasoning given is the reason.
- **`effort` at −0.0002.** Unrevisited since the stand task, but it is A1's and Go2's own
  config value and `REWARDS.md` already justifies it. Leave it; finding 3's power term is
  the better lever.

---

## The one general lesson

Every finding above except 3 and 4 is the same failure in a different place: **a term
whose value is zero looks exactly like a term that is passing.** `ground_covered` returns
0 for backward commands. `_going_straight` returned False for backward commands.
`feet_air_time` returned 0 because a Go1 threshold was never reached. `track_turn` returned
1.4e-11 for three whole runs. A blank `what` string renders as an empty row.

None of these throw. None of them look wrong on a curve. The project has now been bitten
by this four separate times, and the countermeasure is the same each time: **log every
term's actual contribution and look for the ones sitting at zero.** That is the `WATCH`
change in finding 4, and it is worth more than any individual weight on this list.

---

## Sources added by this audit

Everything in `REWARDS.md`'s source list still stands. New:

- Kim et al., *Not Only Rewards But Also Constraints*, [arXiv:2308.12517](https://arxiv.org/abs/2308.12517) — 3 reward coefficients + 11 constraints; the "more than ten terms" problem stated outright
- van Marum et al., *Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking*, [arXiv:2404.19173](https://arxiv.org/abs/2404.19173) — minimally constraining rewards; the standing-still bias stated directly; a hardware benchmark that does not depend on the reward
- *LoComposition*, [arXiv:2606.15896](https://arxiv.org/abs/2606.15896) — gait priors harm; mechanical power selects the gait; 56% lower CoT
- *Learning-based legged locomotion: state of the art and future perspectives*, [arXiv:2406.01152](https://arxiv.org/abs/2406.01152) — the taxonomy, and "there is no general set of rules one can follow"
- Chamorro, [legged-gym single-term ablation](https://wandb.ai/simonchamorro/legged-gym/reports/Legged-Locomotion-Environment-Experiments-in-Isaac-Gym--VmlldzoxODE2MDY0) — 9 terms removed one at a time; linear and angular velocity tracking dominate, everything else is marginal
- Voelcker, [Reward Design and Termination](https://cvoelcker.de/blog/2025/reward-functions/) — reward sign × termination × discount
