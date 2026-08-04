# Rules

Decisions that hold across the project, with the reasoning that produced them.
A rule goes here once, and everything follows it — rather than being re-argued
each time it comes up.

---

## 1. Stop a run when the reward reaches 96.5% of its ceiling

**The rule.** Training stops as soon as the mean reward reaches 96.5% of the most
it could possibly score. It does not run out its remaining iterations.

**Why there is a ceiling at all.** Every positive scoring term is capped at 1.0
and is multiplied by its weight and by the timestep, then summed over the
episode. So the largest score an episode can reach is:

```
ceiling  =  (sum of the positive weights)  ×  (episode length in seconds)
```

For the stand task: (height 2.0 + posture 1.5 + still 1.0 + alive 0.5) × 10 s = **50.0**

**Do not read a ceiling off this page.** It is a property of whatever the weights
are today, and the weights change — the push ceiling was 100.0 when this was
written and is 78.0 now, because the task gained terms since. `scripts/train.py`
computes it from the task's own weights and prints it as the run starts:

```
reward        ceiling 78.0 (3.9/s over 20 s)
              stopping at 96% of it = 75.3
```

That printed line is the number the run actually used. A ceiling written down
here would be a second copy that goes quietly out of date, and a stop threshold
computed from a stale ceiling either cuts a run short or never fires.

**Why that means stop.** At that point every positive term is essentially maxed —
the robot is at the right height, level, still, in the right pose, and has not
fallen. The only headroom left is in the penalties, which are worth a fraction
of a point between them. Training on is polishing something already measured as
done.

**Why 96.5 and not 99.** The last few percent are the slowest to earn and buy
the least: the curve flattens hard near the ceiling, so the difference between
96.5% and 99% can be hundreds of iterations. It is a judgement, not a law — it is
a dial (`--stop-at`) precisely because the right number may turn out to differ per
stage, and a stage that fails its bar after stopping early is telling us to raise
the number rather than to distrust the rule.

**Why it moved from 98 to 96.5** (owner's call, 3 Aug 2026). push_v4 shows what
98% was costing. It reached 97.6 of a 100.0 ceiling at iteration ~700 and was
still at 97.64 at iteration 938 — flat for 234 iterations, with zero falls and
full-length episodes the whole way. It never reached 98, so it would have run all
1500 iterations to buy nothing. At 96.5 it stops around iteration 700 and the
remaining 800 go to the next run in the queue instead.

**What the evidence looked like.** The stand run finished at 49.93 of a 50.0
ceiling — 99.86% — and had been flat there for hundreds of iterations. It
reached zero falls at roughly iteration 110 of 600. The last 490 iterations
bought a policy that was already passing its bar.

**The honest caveat.** Reward near its ceiling does not prove robustness. A
policy can be maxing every term in the conditions it was trained on and still be
fragile outside them. So the rule is stop training, **not** stop checking: the
run still has to pass `scripts/verify.py`, which measures the bar's own numbers
over the full duration across many robots. A stage is passed by the verifier, not
by the reward curve.

**In practice.** `scripts/train.py` computes the ceiling from the task's own
weights and episode length, prints it at the start, and stops when the reward
reaches 99% of it. `--stop-at 0` runs the full schedule instead.

---

## 2. A stage is passed by its bar, not by its curve

`scripts/verify.py` decides, over the full duration the bar names, across many
robots at once. The reward is a weighted sum that can read excellently while one
term quietly fails, and an episode is shorter than the bar — so a policy that
drifts after twenty seconds still shows a clean training curve.

---

## 3. Measure the machine, do not guess at it

How many robots fit on the card, how much torque a joint needs, how wide the
body is: all of these are measurable in minutes, and all of them have been wrong
when assumed. `scripts/probe_envs.py` answers the first, `scripts/find_stance.py`
the second, and both write down what they found.

---

## 4. One training process on the card at a time

**The rule.** Never start a second run — not even a three-iteration smoke test —
while another is training. Wait, or stop the first one.

**Why.** Runs are sized by `probe_envs.py` to fill the card on purpose: the push
task at 6400 robots holds 10.2 GB of 12.3 GB. There is no room for a second
process, and the failure is not a clean error. Both processes sit at 100% GPU
utilisation, allocating and stalling, and neither makes progress.

**What it looks like.** Exactly like a hung dashboard. The GPU is pinned, the fans
are loud, and nothing is being written — because nothing is happening. The monitor
is telling the truth; there is genuinely no new data.

**What the evidence looked like.** A 256-robot smoke test started at 01:47:22 on
top of push_v4. push_v4's last checkpoint was 01:47:13. Neither process advanced
for the next eighteen minutes. Killing the smoke test alone brought push_v4 back
within one checkpoint interval, unharmed — the deadlock costs time, not the run.

**In practice.** `nvidia-smi` before starting anything. If a python process is
already holding GPU memory, do not start a second one.

**This is not only about training runs.** Anything that opens a simulator counts,
because the card does not care what the second process is for:

- `Playground\pilot.bat` — driving a policy by hand
- `scripts/drive.py` — measuring what a policy does when told
- `scripts/find_stance.py --render` and the pose editor's render
- any viewer window at all

Opening the Playground viewer during a training run once took the whole machine
down and lost the run with it. The rule is the same rule: **one thing on the card,
and training wins**. Wait for the queue to drain, or pause it.

The dashboard itself is safe — it is CPU only. So is `find_stance.py` without
`--render`.
