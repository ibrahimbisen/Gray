"""The training queue: what to run next, as data on disk.

    progress/queue.json

Two processes touch this file. The dashboard writes to it when you add or
reorder a job; scripts/runner.py reads it, takes the next job, and writes back
what happened. Neither one owns it, so every write goes through `_edit()`, which
takes a lock, re-reads, changes, and replaces atomically. A read-modify-write
without that loses whichever edit finished second - and the one that gets lost is
usually the runner marking a job finished, which then looks like a hung run.

**Why a queue at all.** RULES.md rule 4: one training process on the card at a
time. Two runs at once do not fail cleanly - both sit at 100% GPU and neither
advances, which looks exactly like a hung dashboard. A queue makes that rule
structural instead of something to remember at 2 a.m.

**Why a separate runner process rather than the dashboard spawning jobs.**
Closing the dashboard would kill training. The runner outlives it.

Nothing here imports torch or MuJoCo. The queue has to be readable even when the
training environment is broken, because that is exactly when you need to see it.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Makes every temp filename unique - see _publish().
_TMP_SEQ = itertools.count()

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "progress" / "queue.json"
LOCK = ROOT / "progress" / "queue.lock"
HEARTBEAT = ROOT / "progress" / "runner.json"

# The runner touches its heartbeat every few seconds. Anything older than this
# and we stop believing it is alive. This exists so the page can say "nothing
# will start - the runner is not running", which is otherwise indistinguishable
# from "the queue is working through it" and wastes a night either way.
HEARTBEAT_STALE_S = 30.0

# What can have happened to a job. `skipped` is a job the runner refused to
# start - a bad task name, say - kept in the list rather than deleted, because a
# job that silently vanishes is indistinguishable from one that never got added.
STATES = ("queued", "running", "done", "failed", "cancelled", "skipped")
FINISHED = ("done", "failed", "cancelled", "skipped")

# The tasks a job may name. Mirrors scripts/train.py's own list. Duplicated on
# purpose: this module must not import anything that pulls in torch, and the
# runner re-checks against the real list before it launches anything.
TASKS = ("Gray-Stand", "Gray-Push", "Gray-Walk")

# How many robots fit on the card, measured by scripts/probe_envs.py - see
# RULES.md rule 3. 6400 for the push task was 10.2 GB of 12.3.
#
# THE 4096 CRASH WAS NOT MEMORY, and this file said it was for half a day.
#
# A stand run at 4096 with filming on died 26 s in, and the obvious reading was
# that the trainer and the video renderer had run the card out of memory. The
# owner then measured it: 3072 uses 6.9 GB and 3600 uses 7.3 GB, so 4096 would
# have been about 7.8 GB of 12.0. Nowhere near.
#
# Reading the traceback properly says the same thing. The failure was
#
#     wp_cuda_graph_launch ... Warp CUDA error 600: device not ready
#
# raised from wp.capture_launch. Error 600 is cudaErrorNotReady - a CUDA GRAPH
# LAUNCH failure. An allocation failure reports a memory error and looks nothing
# like this. And the timeline was: train.py at 05:56:52, film_checkpoints.py at
# 05:56:57, crash at 05:57:23. The film process was creating its own CUDA context
# on the same device while training was mid graph-capture, which is a documented
# way to get exactly this error.
#
# So there is one ceiling rather than two, and the real fix lives in
# scripts/runner.py: filming now waits until training is actually iterating.
# Filming reads checkpoints off disk and the first one does not exist until
# iteration 25, so starting it early bought nothing and cost a run.
#
# THE NUMBER ITSELF is 4500, well under the 6400 probe_envs.py measured. This is
# a settled choice, not a dial to keep turning - the owner asked for it to stay.
#
# Measured on this card, walking, with the renderer loaded:
#
#     robots   memory        steps/s    s per iteration   3000 iters
#      3072    ~6.9 GB        48,500        1.52 s          76 min
#      4500   ~10.6 GB       ~62,000        1.74 s          87 min
#      5500   ~11.1 GB        72,000        1.89 s          94 min
#      6400   ~11.8 GB        82,000        1.87 s          94 min
#
# Two things that table says. Throughput is SUB-linear - 2.08x the robots buys
# 1.69x the steps, because the card is already pinned at 100% and the extra
# robots queue rather than run alongside. And more robots makes each iteration
# BIGGER, not faster, so wall-clock per run goes up.
#
# So the trade is: more robots means a less noisy gradient, fewer robots means a
# faster experiment loop. While the reward function is still being debugged - and
# straightness is currently about 20 deg off against a 5 deg bar, which is a
# problem of what the policy can SENSE and not one of gradient noise - turnaround
# is worth more than batch size.
# 4500 is comfortably past the ~4096 that legged-robot RL usually needs.
#
# Raised to 5000 on 4 Aug 2026 on the owner's call, alongside dropping the run
# length from 2000 iterations to 500. Those two go together: a bigger batch is a
# less noisy gradient per iteration, which is what buys the right to take fewer
# of them. Watch the first run of the night - 5000 is 11% more memory on a 12 GB
# card than the 4500 every run so far has used, and the failure mode is an
# out-of-memory abort in the first minute rather than anything subtle.
CARD_ENV_CEILING = 5000
FILMING_ENV_CEILING = CARD_ENV_CEILING

# A job, with every field defaulted. Anything the UI does not send falls back to
# the same value train.py would have used on its own.
DEFAULTS: dict[str, Any] = {
    "task": "Gray-Walk",
    "name": "",             # run name; train.py derives one if blank
    "note": "",             # why this run exists, in the owner's words
    "num_envs": 4500,
    "seed": 0,              # 0 = the task's own (42). Vary it to measure noise.
    "iterations": 0,        # 0 = the task's own default
    "stop_at": 0.965,       # RULES.md rule 1
    "no_video": False,
    "rewards": {},          # term -> weight, overriding the task
    "ramps": {},            # ramped term -> [w0, w1, w2, w3] for its curriculum
    "turn_std": None,       # track_turn's tolerance in rad/s; None keeps 0.80
    "upright_std": None,    # upright's tilt tolerance in rad; None keeps 0.45
    "gyro_noise": None,     # (bias, wander) rad on the heading the policy READS
    "no_heading_obs": False,  # train blind, the pre-4-Aug 48-input policy
    "with_off_track": False,  # ADD the cross-track input; off by default, it lost ground
    "crab_share": None,       # share of draws that are a PURE sideways step
    "spin_share": None,       # share of draws that are a PURE turn on the spot
    "dive_ends": False,       # trunk contact ends the attempt (nose_dived)
    "swing_target": None,     # metres a swing is scored against; None keeps 0.035
    "slope_deg": None,        # tilt of the ground; None/0 is the flat floor
    "init_from": "",          # run to continue from; "" starts from scratch
    "narrow_dials": "",       # world dials put back to the Gray-Push range, or "all"
    "push_speed": None,     # [min, max] m/s, or None to leave alone
    "push_spin": None,      # [min, max] rad/s
    "film": True,           # film checkpoints alongside training
    "verify": True,         # score against the stage bar when it finishes
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _blank() -> dict:
    return {"paused": False, "next_id": 1, "jobs": []}


def _read() -> dict:
    try:
        state = json.loads(QUEUE.read_text())
    except (OSError, ValueError):
        return _blank()
    if not isinstance(state, dict) or not isinstance(state.get("jobs"), list):
        return _blank()
    state.setdefault("paused", False)
    state.setdefault("next_id", len(state["jobs"]) + 1)
    return state


def _write(state: dict) -> None:
    """Replace the file atomically, so a reader never sees a half-written queue.

    Two things here are Windows, not paranoia.

    **The temp file is per-process.** Both the dashboard and the runner write
    this. Sharing one temp name means one can truncate-and-rewrite it while the
    other is publishing it, and what lands in queue.json is a truncated file.
    _read() then returns a blank queue, and the next _edit writes that blank back
    - every job gone, silently.

    **os.replace is retried.** It is atomic, but on Windows it FAILS outright
    while any other process has the destination open, because Python opens files
    without FILE_SHARE_DELETE. Measured under two writers and one reader: 59 of
    240 replaces raised PermissionError, with the lock working perfectly - the
    lock serialises writers, and readers never take it. An unretried failure
    throws out of the runner's heartbeat and marks a job failed while its
    training is still running.
    """
    _publish(QUEUE, json.dumps(state, indent=2))


def _publish(dest: Path, text: str, tries: int = 30) -> None:
    """Write `text` to `dest` atomically, retrying the replace.

    Used for both the queue and the heartbeat. Both are written by one process
    while another may be reading them, and on Windows that makes os.replace fail
    outright rather than block - Python opens files without FILE_SHARE_DELETE.
    Measured before the retry: 59 of 240 replaces raised PermissionError under
    two writers and one reader, with the lock working perfectly. The lock
    serialises writers; readers never take it, and never should.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Unique per CALL, not per process. Two threads in one process - two
    # concurrent dashboard requests, say - would otherwise share the name, and
    # whichever replaced first would delete the other's temp file out from under
    # it: "WinError 2, cannot find the file". Measured: 8 failures in 360 writes
    # across three threads before the counter was added.
    tmp = dest.with_suffix(f".{os.getpid()}.{next(_TMP_SEQ)}.tmp")
    tmp.write_text(text)
    last: Exception | None = None
    for _ in range(tries):                  # ~4 s of 50 ms tries
        try:
            os.replace(tmp, dest)
            return
        except PermissionError as exc:      # a reader has it open; it will let go
            last = exc
            time.sleep(0.05)
        except OSError as exc:
            last = exc
            break
    tmp.unlink(missing_ok=True)
    raise OSError(f"could not publish {dest.name} after retrying: {last}")


class _Lock:
    """A lock file, so the dashboard and the runner cannot interleave writes.

    O_EXCL create is the primitive - it is atomic on every filesystem we care
    about. A stale lock (the holder was killed mid-write) is broken after
    `stale_after`, because a queue that can never be written again is a worse
    failure than a rare double write.
    """

    # The timeout MUST comfortably exceed _publish's retry window (30 x 50 ms =
    # 1.5 s), because _publish runs while this lock is held. If a contended
    # replace could burn most of a waiter's patience, the waiter would time out,
    # proceed unlocked by design, read the state just before the holder's replace
    # landed, and write over it. The lost write is usually the runner marking a
    # job done - which leaves it "running", so claim() hands out nothing and the
    # whole queue stalls. 15 s against 1.5 s leaves an order of magnitude.
    def __init__(self, timeout: float = 15.0, stale_after: float = 45.0):
        self.timeout = timeout
        self.stale_after = stale_after
        self.held = False

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                LOCK.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self.held = True
                return self
            except FileExistsError:
                pass
            except OSError:
                # Transient - an indexer or scanner holding the file, say. Fall
                # through to the SAME deadline as everything else. An earlier
                # version did `continue` here with no sleep and no deadline
                # check, which spins at 100% CPU forever if it does not clear.
                pass
            try:
                stale = time.time() - LOCK.stat().st_mtime > self.stale_after
            except OSError:
                stale = False          # it vanished under us; just try again
            if stale:
                LOCK.unlink(missing_ok=True)
            elif time.time() > deadline:
                # Carry on unlocked rather than refusing to work. The cost of a
                # lost write here is one queue edit; the cost of raising is a
                # dashboard that cannot queue anything at all.
                return self
            time.sleep(0.05)

    def __exit__(self, *exc):
        # Only remove the lock if it is still OURS. Breaking a stale lock deletes
        # somebody else's file, and without this check a process that timed out
        # and proceeded unlocked would go on to delete whoever holds it now.
        if self.held:
            try:
                if LOCK.read_text().strip() == str(os.getpid()):
                    LOCK.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def _edit(fn) -> dict:
    """Re-read under the lock, apply fn, write back. Returns the new state.

    Refuses to publish over a queue it could not parse. Without that, one
    truncated read turns into a blank state that gets written straight back,
    and every queued job disappears with nothing to say why.
    """
    with _Lock():
        raw_existed = QUEUE.is_file()
        state = _read()
        if raw_existed and not state["jobs"] and QUEUE.stat().st_size > 200:
            raise OSError(
                "queue.json exists and is non-trivial but parsed as empty - "
                "refusing to overwrite it. Look at the file before retrying.")
        fn(state)
        _write(state)
        return state


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def beat(**extra) -> None:
    """The runner says it is alive. Called from its poll loop."""
    _publish(HEARTBEAT, json.dumps(
        {"pid": os.getpid(), "at": _now(), "stamp": time.time(), **extra}, indent=2))


def runner_status() -> dict:
    """Is a runner alive? Queueing work with nothing to run it does nothing at
    all, and looks identical to a queue that is simply busy."""
    try:
        beat_data = json.loads(HEARTBEAT.read_text())
        age = time.time() - float(beat_data.get("stamp", 0.0))
    except (OSError, ValueError, TypeError):
        return {"alive": False, "age_s": None, "pid": None, "at": None,
                "hint": "start it with:  python run.py --runner"}
    alive = age < HEARTBEAT_STALE_S
    return {
        "alive": alive,
        "age_s": round(age, 1),
        "pid": beat_data.get("pid"),
        "at": beat_data.get("at"),
        "hint": "" if alive else "the runner stopped - nothing queued will start. "
                                 "Restart it with:  python run.py --runner",
    }


def load() -> dict:
    """The queue, with counts the pages would otherwise all compute themselves."""
    state = _read()

    # The exact command each job will run, shown on the page. Queueing something
    # you cannot read back is just hoping.
    #
    # Caught, not raised. train_argv refuses a job whose settings it cannot pass
    # on - the right call there, and fatal here: load() runs over EVERY job on
    # every API request and on every runner poll, so one bad job would 500 the
    # whole page and stop the queue. The refusal is carried on the job instead,
    # where the page can show it and the runner can skip that one job.
    def described(job: dict) -> dict:
        try:
            return {**job, "command": command_line(job), "cannot_run": ""}
        except ValueError as exc:
            return {**job, "command": "", "cannot_run": str(exc)}

    jobs = [described(j) for j in state["jobs"]]
    state = {**state, "jobs": jobs}
    return {
        **state,
        "counts": {s: sum(1 for j in jobs if j.get("state") == s) for s in STATES},
        "queued": [j for j in jobs if j.get("state") == "queued"],
        "running": next((j for j in jobs if j.get("state") == "running"), None),
        "history": [j for j in jobs if j.get("state") in FINISHED][::-1],
        "tasks": list(TASKS),
        "defaults": DEFAULTS,
        # The card's ceiling, so the Add-a-job form can cap its own box and print
        # the real number in the hint beside it. It used to say 4096 in the HTML,
        # which was wrong from the moment this was raised to 5000.
        "env_ceiling": CARD_ENV_CEILING,
        "runner": runner_status(),
    }


def next_queued() -> dict | None:
    """The job the runner should start, or None. Respects the pause switch.

    Returns nothing while a job is already marked running - the queue is the
    thing enforcing RULES.md rule 4, so it refuses to hand out a second job even
    if asked.
    """
    state = _read()
    if state.get("paused"):
        return None
    if any(j.get("state") == "running" for j in state["jobs"]):
        return None
    return next((j for j in state["jobs"] if j.get("state") == "queued"), None)


def claim(owner: str) -> dict | None:
    """Take the next job AND mark it running, in one locked edit.

    next_queued() then a separate update() is two steps with a gap, and in that
    gap a second runner can read the same job and claim it too - two trainers on
    one card, which is the exact failure RULES.md rule 4 exists to prevent. The
    job can also be removed in that gap, so the runner would launch training for
    a job that is no longer in the queue.

    Claiming under the lock closes both. `owner` is recorded so a job says which
    runner has it.
    """
    taken: dict = {}

    def apply(state: dict) -> None:
        if state.get("paused"):
            return
        if any(j.get("state") == "running" for j in state["jobs"]):
            return
        for job in state["jobs"]:
            if job.get("state") == "queued":
                job["state"] = "running"
                job["started"] = _now()
                job["owner"] = owner
                taken.update(job)
                return

    _edit(apply)
    return taken or None


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


_FALSE_WORDS = {"", "0", "false", "no", "off", "none", "null"}


def _as_bool(value) -> bool:
    """"false", "0" and "off" are False.

    Plain bool() says any non-empty string is true, so a client that sends its
    checkboxes as strings gets no_video="false" -> True and silently trains a
    whole run with no videos.
    """
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_WORDS
    return bool(value)


def _as_int(value, fallback: int) -> int:
    try:
        return int(float(value))        # tolerate "3072.0" and 3072.0
    except (TypeError, ValueError):
        return fallback


def _as_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_pair(value):
    """A two-number range, or None. Never raises."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None                     # a single number is not a range
    if isinstance(value, str):
        value = [p for p in value.replace(",", " ").split() if p]
    try:
        pair = [float(v) for v in value][:2]
    except (TypeError, ValueError):
        return None
    return pair if len(pair) == 2 else None


def _clean(spec: dict) -> dict:
    """Coerce whatever the browser sent into the shape the runner expects.

    Every branch here is total: it returns a usable job for ANY input. That is
    not defensiveness for its own sake. This runs on a POST body, and one bad
    value used to be permanent - a `rewards` that was not a dict got stringified,
    saved, and then command_line() raised AttributeError on it forever. load()
    calls command_line() on every job, so /api/queue and /api/monitor returned
    500 for good, and the runner died on its next poll. The only way out was
    hand-editing queue.json.
    """
    job = dict(DEFAULTS)

    for key in ("task", "name", "note", "narrow_dials", "init_from"):
        if spec.get(key) is not None:
            job[key] = str(spec[key])
    for key in ("no_video", "film", "verify", "no_heading_obs", "with_off_track",
                "dive_ends"):
        if spec.get(key) is not None:
            job[key] = _as_bool(spec[key])
    for key in ("num_envs", "iterations", "seed"):
        if spec.get(key) is not None:
            job[key] = _as_int(spec[key], DEFAULTS[key])
    if spec.get("stop_at") is not None:
        job["stop_at"] = _as_float(spec["stop_at"], DEFAULTS["stop_at"])
    for key in ("turn_std", "upright_std", "crab_share", "spin_share",
                "swing_target", "slope_deg"):
        if spec.get(key) is not None:
            job[key] = _as_float(spec[key], 0.0)
    # NOTE the pair fields are listed here AND in train_argv, and they have to
    # match. A field added to one and not the other is a job that shows the
    # setting on the page and does not pass it to the trainer - which is
    # indistinguishable from a setting that did not work, and cost a whole
    # 13-run queue being built with every knob silently dropped on 4 Aug 2026.
    for key in ("push_speed", "push_spin", "gyro_noise"):
        if spec.get(key) is not None:
            job[key] = _as_pair(spec[key])

    # Only a mapping of name -> number. Anything else is dropped rather than
    # coerced: a reward override that did not survive is far better than one
    # that poisons the queue file.
    rewards = spec.get("rewards")
    clean_rewards: dict[str, float] = {}
    if isinstance(rewards, dict):
        for term, weight in rewards.items():
            try:
                clean_rewards[str(term)] = float(weight)
            except (TypeError, ValueError):
                continue
    job["rewards"] = clean_rewards
    # Same treatment for ramps, which are a term name against a LIST of stage
    # weights. A ramp that will not coerce is dropped rather than written back,
    # for the same reason: train.py would reject the whole job and the runner
    # would mark it skipped, which reads as "the queue is broken".
    ramps = spec.get("ramps")
    clean_ramps: dict[str, list[float]] = {}
    if isinstance(ramps, dict):
        for term, stages in ramps.items():
            if not isinstance(stages, (list, tuple)):
                continue
            try:
                clean_ramps[str(term)] = [float(w) for w in stages]
            except (TypeError, ValueError):
                continue
    job["ramps"] = clean_ramps
    if job["task"] not in TASKS:
        job["task"] = DEFAULTS["task"]
    job["iterations"] = max(0, job["iterations"])
    job["stop_at"] = min(max(job["stop_at"], 0.0), 1.0)

    # Robots on the card, against the ceiling scripts/probe_envs.py measured.
    # Clamped rather than rejected - a job that silently does not exist is worse
    # than one that runs slightly smaller - but the clamp is RECORDED so the page
    # can say it happened. A silent clamp looks like the settings were ignored.
    asked = max(1, int(job["num_envs"]))
    ceiling = FILMING_ENV_CEILING if job.get("film") else CARD_ENV_CEILING
    job["num_envs"] = min(asked, ceiling)
    job["clamped"] = ""
    if asked > ceiling:
        job["clamped"] = (f"asked for {asked} robots, capped at {ceiling} - the "
                          f"card's measured ceiling (probe_envs.py)")
    return job


def add(spec: dict, position: str = "end") -> dict:
    """Add a job. `position` is "end" or "next" (jump the queue).

    REFUSES A SPEC IT WOULD SILENTLY NARROW. `_clean` rebuilds every job from
    DEFAULTS and copies across only the keys it knows, which is what keeps one
    bad POST from poisoning the queue file - and it means a key it has never
    heard of vanishes without a word.

    That is how `upright_std` was lost on 4 Aug 2026. It was added to DEFAULTS,
    to train_argv and to train_argv's own guard, and not to `_clean` - so the
    value died here, the job carried the default of None, and the guard "job
    asks for X but the command line has no X" saw a job asking for nothing.
    A check that reads the job cannot catch a field the job never received; the
    only place that still knows what was WANTED is right here, before _clean.
    """
    made: dict = {}

    def apply(state: dict) -> None:
        job = _clean(spec)
        lost = [k for k, v in spec.items()
                if v not in (None, "", {}, [], False) and k not in ("position",)
                and job.get(k) in (None, "", {}, [], False)]
        if lost:
            raise ValueError(
                f"queue.py does not know how to carry {', '.join(sorted(lost))} - "
                f"the value was dropped. Add it to _clean() AND to train_argv(), "
                f"or the job will train the default while claiming otherwise.")
        job.update({
            "id": f"j{state['next_id']:04d}",
            "state": "queued",
            "created": _now(),
            "started": None,
            "finished": None,
            "run_id": None,
            "exit_code": None,
            "error": "",
            "steps": [],       # what the runner did, in order, with timings
        })

        state["next_id"] += 1
        if position == "next":
            at = next((i for i, j in enumerate(state["jobs"])
                       if j.get("state") == "queued"), len(state["jobs"]))
            state["jobs"].insert(at, job)
        else:
            state["jobs"].append(job)
        made.update(job)

    _edit(apply)
    return made


def update(job_id: str, **fields) -> dict | None:
    """Change fields on one job. The runner uses this to record progress."""
    found: dict = {}

    def apply(state: dict) -> None:
        for job in state["jobs"]:
            if job.get("id") == job_id:
                job.update(fields)
                found.update(job)
                return

    _edit(apply)
    return found or None


def edit(job_id: str, spec: dict) -> dict | None:
    """Change a QUEUED job's settings. Running and finished jobs are immutable -
    a finished job's settings are the record of what actually ran."""
    found: dict = {}

    def apply(state: dict) -> None:
        for job in state["jobs"]:
            if job.get("id") == job_id and job.get("state") == "queued":
                job.update(_clean({**job, **spec}))
                found.update(job)
                return

    _edit(apply)
    return found or None


def remove(job_id: str) -> bool:
    """Drop a job. A running job is cancelled instead - the runner notices and
    kills the process, because this module cannot reach across to it."""
    gone = {"ok": False}

    def apply(state: dict) -> None:
        for job in state["jobs"]:
            if job.get("id") != job_id:
                continue
            if job.get("state") == "running":
                job["state"] = "cancelled"
                job["finished"] = _now()
                job["error"] = "cancelled from the dashboard"
            else:
                state["jobs"] = [j for j in state["jobs"] if j.get("id") != job_id]
            gone["ok"] = True
            return

    _edit(apply)
    return gone["ok"]


def move(job_id: str, delta: int) -> bool:
    """Shift a queued job up or down by `delta` places.

    Only among the queued ones - reordering around a finished job would put it
    somewhere that means nothing.

    This SHIFTS rather than swaps. Swapping is the same thing for the plus or
    minus one the arrow buttons send, and different for anything larger: moving
    a job up two by swapping also throws whatever was two above it down to where
    the job came from, silently reordering a run nobody touched.
    """
    ok = {"ok": False}

    def apply(state: dict) -> None:
        slots = [i for i, j in enumerate(state["jobs"]) if j.get("state") == "queued"]
        order = [state["jobs"][i] for i in slots]
        here = next((n for n, j in enumerate(order) if j.get("id") == job_id), None)
        if here is None:
            return
        there = here + delta
        if not 0 <= there < len(order):
            return
        order.insert(there, order.pop(here))
        # Write the reordered queued jobs back into the same slots, so anything
        # running or finished keeps its position in the list.
        for slot, job in zip(slots, order):
            state["jobs"][slot] = job
        ok["ok"] = True

    _edit(apply)
    return ok["ok"]


def duplicate(job_id: str) -> dict | None:
    """Copy a job back onto the end of the queue.

    This is the one that makes sweeps bearable: run something, change one weight,
    queue it again - rather than retyping nine fields and getting one wrong.
    """
    state = _read()
    src = next((j for j in state["jobs"] if j.get("id") == job_id), None)
    if src is None:
        return None
    spec = {k: src.get(k, v) for k, v in DEFAULTS.items()}
    spec["name"] = ""       # a fresh timestamp names it; two runs must not collide
    return add(spec)


def set_paused(paused: bool) -> dict:
    """Pause takes the runner out of service. It never touches a running job -
    stopping training and stopping the QUEUE are different intentions."""
    return _edit(lambda s: s.__setitem__("paused", bool(paused)))


def clear_finished() -> int:
    """Drop every finished job. The runs themselves stay in progress/runs/ -
    this only tidies the queue list."""
    count = {"n": 0}

    def apply(state: dict) -> None:
        keep = [j for j in state["jobs"] if j.get("state") not in FINISHED]
        count["n"] = len(state["jobs"]) - len(keep)
        state["jobs"] = keep

    _edit(apply)
    return count["n"]


def requeue(job_id: str) -> dict | None:
    """Put a finished job back in the queue as a fresh one."""
    return duplicate(job_id)


# ---------------------------------------------------------------------------
# turning a job into a command
# ---------------------------------------------------------------------------


def train_argv(job: dict) -> list[str]:
    """The scripts/train.py arguments this job means.

    Kept here rather than in the runner so the dashboard can show the exact
    command a job will run before it runs - which is the difference between
    queueing something and hoping.
    """
    argv = [str(job.get("task") or DEFAULTS["task"]),
            "--num-envs", str(_as_int(job.get("num_envs"), DEFAULTS["num_envs"]))]
    if job.get("iterations"):
        argv += ["--iterations", str(_as_int(job["iterations"], 0))]
    if job.get("name"):
        argv += ["--name", str(job["name"])]
    if job.get("no_video"):
        argv += ["--no-video"]
    if job.get("seed"):
        argv += ["--seed", str(_as_int(job["seed"], 0))]
    argv += ["--stop-at", str(_as_float(job.get("stop_at"), DEFAULTS["stop_at"]))]
    # `.items()` only if it really is a mapping. _clean guarantees that for
    # anything added from now on, but a queue.json written before it did would
    # otherwise raise here - and load() calls this for EVERY job, so one bad
    # entry would 500 the whole API rather than spoiling one row.
    rewards = job.get("rewards")
    if isinstance(rewards, dict):
        for term, weight in rewards.items():
            argv += ["--reward", f"{term}={weight}"]
    # Ramped terms take every stage at once. train.py refuses --reward on these,
    # because a curriculum re-applies its own weight and would silently overwrite
    # it - so a run would record a weight it never trained on.
    ramps = job.get("ramps")
    if isinstance(ramps, dict):
        for term, stages in ramps.items():
            if isinstance(stages, (list, tuple)) and stages:
                argv += ["--ramp", f"{term}=" + ",".join(str(w) for w in stages)]
    for flag in ("push_speed", "push_spin", "gyro_noise"):
        pair = _as_pair(job.get(flag))
        if pair:
            argv += [f"--{flag.replace('_', '-')}", *(str(v) for v in pair)]
    # A tolerance, not a weight - it sets how sharply `track_turn` scores yaw
    # rate, which --reward cannot reach. Kept out of `rewards` deliberately: a
    # std in a dict of weights is the kind of thing that gets applied as one.
    if job.get("turn_std"):
        argv += ["--turn-std", str(_as_float(job["turn_std"], 0.0))]
    if job.get("upright_std"):
        argv += ["--upright-std", str(_as_float(job["upright_std"], 0.0))]
    if job.get("no_heading_obs"):
        argv += ["--no-heading-obs"]
    if job.get("with_off_track"):
        argv += ["--with-off-track"]
    # `is not None`, NOT truthiness, and that distinction is the whole point of
    # this field. Zero is the setting that matters most - it switches the pure
    # sideways share off and gives the draw mix as it was before 5 Aug 2026, so
    # the share can be measured against its own absence. Written like the two
    # lines above it, `crab_share: 0.0` would be falsy, the flag would never be
    # added, and the run would train at 0.15 under a name saying 0.
    if job.get("crab_share") is not None:
        argv += ["--crab-share", str(_as_float(job["crab_share"], 0.0))]
    # Same rule, same reason: 0 switches the spin share off, so `is not None`.
    if job.get("spin_share") is not None:
        argv += ["--spin-share", str(_as_float(job["spin_share"], 0.0))]
    if job.get("dive_ends"):
        argv += ["--dive-ends"]
    if job.get("swing_target"):
        argv += ["--swing-target", str(_as_float(job["swing_target"], 0.0))]
    if job.get("slope_deg"):
        argv += ["--slope-deg", str(_as_float(job["slope_deg"], 0.0))]
    if job.get("init_from"):
        argv += ["--init-from", str(job["init_from"])]
    if job.get("narrow_dials"):
        argv += ["--narrow-dials", str(job["narrow_dials"])]

    # Every knob that is set must have produced a flag. This is not paranoia: on
    # 4 Aug 2026 six runs - five hours of card time - trained the identical
    # config while the dashboard showed six different ones, because the RUNNER
    # PROCESS had been started before turn_std, gyro_noise and no_heading_obs
    # were added here. A long-running Python process holds the module it
    # imported; editing this file does nothing until the runner is restarted.
    #
    # Nothing above can detect its own absence, so the check is written against
    # the job instead: if the job asks for something and the argv does not carry
    # it, refuse the job rather than run a mislabelled experiment. A job that
    # will not start is a five-minute fix. Six that quietly agree are a day.
    text = " ".join(argv)
    # `zero_means_something` names the fields where 0 is a real setting rather
    # than "unset", so the check below tests them with `is not None`. Get this
    # wrong and the guard is blind to exactly the run it exists to protect.
    zero_means_something = ("crab_share", "spin_share")
    for key, flag in (("turn_std", "--turn-std"),
                      ("upright_std", "--upright-std"),
                      ("gyro_noise", "--gyro-noise"),
                      ("no_heading_obs", "--no-heading-obs"),
                      ("with_off_track", "--with-off-track"),
                      ("crab_share", "--crab-share"),
                      ("spin_share", "--spin-share"),
                      ("dive_ends", "--dive-ends"),
                      ("swing_target", "--swing-target"),
                      ("slope_deg", "--slope-deg"),
                      ("init_from", "--init-from"),
                      ("narrow_dials", "--narrow-dials"),
                      ("push_speed", "--push-speed"),
                      ("push_spin", "--push-spin")):
        asked = (job.get(key) is not None if key in zero_means_something
                 else bool(job.get(key)))
        if asked and flag not in text:
            raise ValueError(
                f"job {job.get('id', '?')} sets {key}={job[key]!r} but the "
                f"command line has no {flag}. The runner is running an older "
                f"copy of dashboard/queue.py than the queue was written with - "
                f"restart the runner.")
    return argv


def command_line(job: dict) -> str:
    """The same thing as a string, for showing on the page."""
    return "python scripts/train.py " + " ".join(train_argv(job))
