# Rules

Decisions that hold across the project, with the reasoning that produced them.
A rule goes here once, and everything follows it — rather than being re-argued
each time it comes up.

---

## 1. A run trains its full schedule. There is no early stop.

**The rule.** A run trains every iteration it was asked for. Nothing ends it
early except a person, a crash, or the queue runner being told to stop.

**What was here before, and why it is gone** (owner's call, 6 Aug 2026).
Training used to stop as soon as the mean reward reached 96.5% of the most an
episode could score — the sum of the positive weights times the episode length.
The argument was that at the ceiling every positive term is maxed, so the rest
of the schedule is polish. `--stop-at` was the dial, and it defaulted to on.

**The argument was wrong in one specific way: the reward is not what a run is
judged on.** Rule 2 says a stage is passed by its bar, not by its curve — and
the stop watched the curve. A walk policy sits at 96.5% of its ceiling while
drifting 5 degrees off its line and turning at three quarters of the rate it
was told, because those two faults cost a sliver of a total dominated by
staying upright and moving at all. The stop fired on a number nobody grades,
and ended runs that were still improving on the numbers everybody grades.

**It also made run lengths incomparable.** Two runs of the same config stopped
at different iterations, so a difference between them could be the change under
test or could be the 200 iterations one of them never trained. Every batch since
1.3 had to switch the stop off by hand to be readable at all, which is the sign
of a default pointing the wrong way.

**What replaces it.** Nothing automatic. Ask for the iterations you want; read
the curve afterwards; and if the numbers are still climbing at the end, continue
the run with `--init-from` rather than guessing longer next time. Continuing
costs only the extra iterations, so the choice of length is never final.

**The cost of this decision, stated plainly.** Runs that converge early now burn
card time to the end of their schedule. That is the price of never again asking
"did this run stop because it was finished, or because the reward flattened?"

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
