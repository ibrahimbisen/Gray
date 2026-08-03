# Rules

Decisions that hold across the project, with the reasoning that produced them.
A rule goes here once, and everything follows it — rather than being re-argued
each time it comes up.

---

## 1. Stop a run when the reward reaches 98% of its ceiling

**The rule.** Training stops as soon as the mean reward reaches 98% of the most
it could possibly score. It does not run out its remaining iterations.

**Why there is a ceiling at all.** Every positive scoring term is capped at 1.0
and is multiplied by its weight and by the timestep, then summed over the
episode. So the largest score an episode can reach is:

```
ceiling  =  (sum of the positive weights)  ×  (episode length in seconds)
```

For the stand task: (height 2.0 + posture 1.5 + still 1.0 + alive 0.5) × 10 s = **50.0**
For the push task: the same weights × 20 s = **100.0**

**Why that means stop.** At that point every positive term is essentially maxed —
the robot is at the right height, level, still, in the right pose, and has not
fallen. The only headroom left is in the penalties, which are worth a fraction
of a point between them. Training on is polishing something already measured as
done.

**Why 98 and not 99.** The last percent is the slowest to earn and buys the
least: the curve flattens hard near the ceiling, so the difference between 98%
and 99% can be hundreds of iterations. 98% is a judgement, not a law — it is a
dial (`--stop-at`) precisely because the right number may turn out to differ per
stage, and a stage that fails its bar after stopping at 98% is telling us to
raise it rather than to distrust the rule.

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
