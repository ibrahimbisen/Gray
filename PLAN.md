# PLAN.md — stage 2, train

Agreed 3 Aug 2026. This replaces the nine-subsection structure the dashboard
used to show.

**Where we are: 1.1.2 — and 1.2 has already happened.**

**Changed 3 Aug 2026, after 1.1.1.** 1.2 was going to come after 1.1, on the
grounds that you cannot tell whether a wider box broke something if the narrow
box was already broken. That was right while the plan was "tune the reward now,
widen later". It stopped being right the moment the decision was made to widen
*and* to add height, pitch and roll — because both throw every trained policy
away, so tuning the reward first would have meant tuning it twice.

So 1.2.1 to 1.2.3 are done, and **1.1 and 1.2.4 are now the same work**: the
rounds below tune the reward directly on the final box rather than on a box that
was about to be replaced. What that costs is the thing 1.2 was ordered after 1.1
to avoid — a harder debugging job if drift is still wrong. What it buys is not
paying for 25 hours of GPU twice.

---

## The shape

One pattern, applied twice.

    train it  ->  harden it  ->  measure it

Locomotion first, then getting up. Then join them. Then the sensors, twice: once
to keep what you had, once to use them.

**Every step ends on a number, not a date.** Nobody knows how long a reward
takes to get right — that is what the gates are for.

**Two things get tuned and they fight each other.** The REWARD says what good
means. The RANGE says where it has to be good. Widening the range makes the same
reward harder to satisfy, so they cannot be finished one after the other. That
is why the range stays pinned at its narrowest until 1.1 passes: you cannot tell
whether a wider range broke something if the narrow one was already broken.

---

## 1. Locomotion

### 1.1 Make it walk

- **1.1.1** Fix `wandering` to measure the line it was sent along — **done, 3 Aug**
- **1.1.2** Round 0 — noise floor. 3 seeds, one config
- **1.1.3** Round 1 — straightness factorial. 8 runs
- **1.1.4** Round 2 — speed terms. 4 runs
- **1.1.5** Round 3 — the winner, long, on unseen seeds. 3 runs
- **1.1.6** Add handover-shaped resets to R1's mix — *preparation for step 3*

> **Gate:** all six R1 criteria pass. Currently 4 of 6 — sideways drift is 21×
> its bar and speed tracking is 1.4× its bar.

Round 0 looks like the least and matters the most. Three identical configs on
three seeds; whatever they disagree by is the noise floor, and every later
difference has to beat it before it means anything.

### 1.2 The whole command box

- **1.2.1** Fix `_going_straight` to gate on `abs(vx)` — **done, 3 Aug**
- **1.2.2** Widen `WALK_SPEED` through zero to negative — **done, 3 Aug.**
  `(-0.35, 0.35)`. `WALK_SIDE` to `±0.20`, `WALK_TURN` to `±1.00`
- **1.2.3** Let `vy` be sampled without `vx` — **done, 3 Aug**
- **1.2.4** Retune — **this is now what 1.1's rounds do**
- **1.2.5** Add height, pitch and roll to the command — **done, 3 Aug.**
  Observation 45 → 48. Ranges measured, not chosen: height 0.15–0.25 m, pitch
  15° up to 8° down, roll ±20°, all inside where `find_stance` says the legs
  run out
- **1.2.6** Make the backward bar bite. **Not done.** `verify.py` carries
  `test_speed_back: -0.35` and reads it nowhere — the walk test still runs one
  pass, forward. Until it runs two and scores the worse, a run can be marked
  passed on forward alone. `drive.py --cases "-0.35,0,0"` measures it by hand
  in the meantime

> **A third thing was found here, and it was not in the plan.** mjlab's
> `rel_forward_envs` — set to 0.8 — did `.abs().clamp(min=0.3)` on the forward
> speed. On the old box that meant four attempts in five were forced to at least
> 0.3 m/s, three quarters of them landing exactly on it: the policy never trained
> across 0.15–0.35 at all. It trained at one speed. That is a live candidate for
> why speed tracking has been failing at 0.071 against a 0.05 bar. The `abs()`
> would also have flipped every backward command positive, so widening the range
> would have bought nothing. Replaced by `gray/tasks/walk_command.py`.

> **Gate:** the same six criteria, across the full range.

Measured on run #25: backward walked 0.00 m in 8 seconds, and pure sideways
walked 3 cm. Neither was ever sampled — `WALK_SPEED` is `(0.15, 0.35)`, so `vx`
is never zero and never negative. Four library rows filed as commands are not.

**1.2.1 must come first.** Widening the range without it trains backward walking
with no straightness penalty at all.

### 1.2b Steer it toward a crawl, not a trot — **owner's call, 4 Aug 2026**

The policy currently trots: diagonal pairs, front-right with back-left, lifting
together. Confirmed by watching run 34's films.

A trot is two feet down and the body falling between steps, caught by the servos
twice per stride. A crawl is one foot up and three always planted, so the weight
never leaves the triangle they make and nothing has to be caught.

In simulation the trot is fine — better, even. On the real robot it asks
DS3218MGs at 50 Hz, with backlash nobody has measured, to arrest a falling 3.1 kg
body twice a stride. **The owner's judgement is that they will not.** He built the
robot; that call is his and it is the right way round to be wrong.

So this is a sim-to-real decision made *before* the evidence rather than after,
which is unusual here and deliberate: finding out in stage 3.3 means every policy
trained until then is trained toward a gait the hardware cannot run.

**Not yet designed.** The honest options are a duty-factor term paying for three
feet down, a contact-schedule reward, or simply a much lower top speed — a trot
is partly what you get for asking for speed. Which one, and what it costs the
other bars, is the first thing to work out. It is NOT a weight change to
`stepping`: that term is already earning the maximum a trot can give it.

---

### 1.3 Turn the dials up

- **1.3.1** Widen what is already on — friction, mass, centre of mass, servo
  gains, joint friction
- **1.3.2** Terrain — slopes first, then uneven ground
- **1.3.3** Payload — to +2 kg, off-centre
- **1.3.4** Degraded hardware — weak servo, dead servo, low battery

> **Gate:** the same six criteria, dials at full.

The dials are not off. Foot friction is randomised 0.4 to 1.2 on every attempt
right now, along with ±20% mass, ±15 mm centre of mass and ±30% servo stiffness.
This step turns them up, it does not switch them on.

None of it is a new skill. The policy never sees the word gravel — it sees the
same 45 numbers in a different pattern. This is also where sim-to-real
robustness comes from: a policy that survives the range survives the real value,
which is the correct answer to three numbers nobody can measure yet.

### 1.4 Measure

- **1.4.1** Top speed, and the speed ladder
- **1.4.2** Acceleration, sustained run
- **1.4.3** Smoothness, read off the trained policy

> **No gate.** Numbers written down, nothing trained.

"Find top speed" has no reward term. You raise the command until it fails and
write down where. The trained half of smoothness — `twitching`, `rocking`,
`shaking`, `joint_shock`, `landing_speed` — is already ramping in during 1.1.

---

## 2. Getting up

### 2.1 Make it stand up

- **2.1.1** Write the recover task — start poses, reward, terminations
- **2.1.2** Reward it for **ending where R1 starts** — stable, at ride height,
  low joint velocity — *preparation for step 3*
- **2.1.3** Train it

> **Gate:** up from 9 of 10 random ground poses, in under 3 s.

The one group that genuinely needs its own policy file. There is no commanded
velocity to track when the robot is on its back, so `track_speed`, `stepping`,
`dragging` and `wandering` are all meaningless.

### 2.2 Turn the dials up

- **2.2.1** The same dials as 1.3, applied to R2

> **Gate:** still 9 of 10, dials at full.

### 2.3 Measure

- **2.3.1** Time to stand, per start pose

> **No gate.**

---

## 3. Join them up

- **3.1** The switch rule — hysteresis and a settle time, so control cannot
  chatter at the boundary
- **3.2** The test — 100 drops: R2 stands it up, hands over, R1 walks 5 m

> **Gate:** 100 handovers with no stall and no fall-loop.

**The failure this prevents.** R2 finishes and says "upright, take it." R1
receives a robot mid-wobble, with joint velocities and a trunk height it has
never seen, because R1's resets always start it clean:

    nudge_base:  x, y ±0.01 m,  yaw ±0.1 rad,  velocity_range {}
    joint reset: velocity ±0.05 rad/s

That is a robot standing still. R1 has trained on nothing else. Hand it an
out-of-distribution state and it falls; R2 picks it up, hands over, it falls
again. No runtime rule fixes that loop.

**Which is why 1.1.6 and 2.1.2 exist.** The preparation happens during training,
in the two steps above. Only the switch rule and the test are genuinely here. Skip
the preparation and this step is where you find out.

---

## 4. Sensors — keep what you had

- **4.1** Fit them. CAD positions needed for the three where the mounting is in
  the maths
- **4.2** Model each one in the simulator, including how it fails
- **4.3** Warm-start — grow the network, copy the old weights across, initialise
  the new input columns to **zero** so day one behaves identically to the best
  policy
- **4.4** Retrain R1 and R2 from that warm start

> **Gate:** every bar that passed before still passes, with 95 inputs.

Ten sensors decided; seven add inputs. 45 → 95 numbers, which is why they go in
as one batch: each one alone costs the same full retrain as all of them
together.

The three where mounting position is a coefficient rather than documentation:
the five optical flow units (each reads `v + ω × r`), the middle IMU (offset
from the centre of mass), and the six range finders (origin and aim vector, which
the simulator turns into raycasts).

**This gate is a regression check.** It proves nothing broke. It does not say
the sensors were worth fitting — that is step 5.

---

## 5. Sensors — use them

- **5.1** Tighten the bars. They were set for a blind policy
- **5.2** Write reward terms that were impossible before:
  - **slip** — five flow units against the IMU's yaw rate. Nothing else on the
    robot can see a slip
  - **landing** — load cells turn "did it slam" from a guess into a number
  - **anticipation** — range finders make stepping over something before
    touching it reachable at all
- **5.3** Retune against the tighter bars

> **Gate:** numbers that were out of reach before now pass.

The policy currently cannot measure its own speed — `base_lin_vel` sits in the
critic group, which is thrown away after training, precisely because the real
robot has no way to measure it. So it infers its speed while `track_speed`, its
largest reward term, scores exactly that. Optical flow measures it directly.

5.2 also unblocks library rows. Some of the rows filed under "needs a camera"
are reachable with range finders instead, which is the cheap half of seeing.

---

## Shelved

| | Trigger to un-shelve |
|---|---|
| **R3** — using a foot on an object | None. Nothing depends on it, and it needs objects in the scene nothing else does. |
| **Jumping** | Stage 3.3 measures the real servo **speed** under load. Torque is not the blocker — higher-torque hobby servos are usually slower. |
| **Seeing** | A camera the *policy* reads, plus a heightmap pipeline and the worst sim-to-real gap on the list. Not the camera used to drive the robot. |
| **The real robot** | Deferred by choice, not blocked by anything. |

**On the last one.** Stage 3.3 measures three numbers the model is guessing —
servo gains, backlash, loop latency. Every policy trained before it is trained
against those guesses. One policy depends on them today. After steps 1 to 5,
two do, hardened and tuned against a robot that may not be the real one.

That is the cost of deferring it. It is a real cost and it is the owner's call.

---

## Rules while this runs

- One training process on the card at a time. RULES.md rule 4.
- Every run is verified when it finishes. A run nobody scored is a run that did
  not happen.
- Nothing in `gray/tasks/` changes while a round is in flight. A weight edited
  mid-round makes every run before it unreadable.
- Films stay on. About 4% of throughput at one clip per 100 iterations, and the
  only way to see a robot doing something the numbers do not describe.
