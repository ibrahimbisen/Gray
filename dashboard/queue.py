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
# RULES.md rule 3.
#
#   CARD    the ceiling with nothing else on the GPU.
#   FILMING the ceiling when scripts/film_checkpoints.py is also running, because
#           the video renderer needs its own GPU memory.
#
# This is not a style preference, it is a crash. A stand run queued at 4096 WITH
# filming on died 26 seconds in with "Warp CUDA error 600: device not ready" -
# the renderer and the trainer both wanted memory that was not there. The number
# is enforced here rather than remembered, because the failure arrives minutes
# later and looks like a broken model rather than a full card.
CARD_ENV_CEILING = 4096
FILMING_ENV_CEILING = 3072

# A job, with every field defaulted. Anything the UI does not send falls back to
# the same value train.py would have used on its own.
DEFAULTS: dict[str, Any] = {
    "task": "Gray-Walk",
    "name": "",             # run name; train.py derives one if blank
    "note": "",             # why this run exists, in the owner's words
    "num_envs": 3072,
    "iterations": 0,        # 0 = the task's own default
    "stop_at": 0.965,       # RULES.md rule 1
    "no_video": False,
    "rewards": {},          # term -> weight, overriding the task
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


def _publish(dest: Path, text: str, tries: int = 80) -> None:
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

    def __init__(self, timeout: float = 5.0, stale_after: float = 30.0):
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
    jobs = [{**j, "command": command_line(j)} for j in state["jobs"]]
    state = {**state, "jobs": jobs}
    return {
        **state,
        "counts": {s: sum(1 for j in jobs if j.get("state") == s) for s in STATES},
        "queued": [j for j in jobs if j.get("state") == "queued"],
        "running": next((j for j in jobs if j.get("state") == "running"), None),
        "history": [j for j in jobs if j.get("state") in FINISHED][::-1],
        "tasks": list(TASKS),
        "defaults": DEFAULTS,
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

    for key in ("task", "name", "note"):
        if spec.get(key) is not None:
            job[key] = str(spec[key])
    for key in ("no_video", "film", "verify"):
        if spec.get(key) is not None:
            job[key] = _as_bool(spec[key])
    for key in ("num_envs", "iterations"):
        if spec.get(key) is not None:
            job[key] = _as_int(spec[key], DEFAULTS[key])
    if spec.get("stop_at") is not None:
        job["stop_at"] = _as_float(spec["stop_at"], DEFAULTS["stop_at"])
    for key in ("push_speed", "push_spin"):
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
    if job["task"] not in TASKS:
        job["task"] = DEFAULTS["task"]
    job["iterations"] = max(0, job["iterations"])
    job["stop_at"] = min(max(job["stop_at"], 0.0), 1.0)

    # Robots on the card. Filming needs its own GPU memory, so the ceiling drops
    # when it is on. Clamped rather than rejected - a job that silently does not
    # exist is worse than one that runs slightly smaller - but the clamp is
    # RECORDED so the page can say it happened. A silent clamp would look like
    # the settings were ignored.
    asked = max(1, int(job["num_envs"]))
    ceiling = FILMING_ENV_CEILING if job.get("film") else CARD_ENV_CEILING
    job["num_envs"] = min(asked, ceiling)
    job["clamped"] = ""
    if asked > ceiling:
        job["clamped"] = (
            f"asked for {asked} robots, capped at {ceiling}"
            + (" because filming needs its own GPU memory - 4096 with the video "
               "renderer running crashes with CUDA error 600"
               if job.get("film") else " - the card's measured ceiling"))
    return job


def add(spec: dict, position: str = "end") -> dict:
    """Add a job. `position` is "end" or "next" (jump the queue)."""
    made: dict = {}

    def apply(state: dict) -> None:
        job = _clean(spec)
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
    argv += ["--stop-at", str(_as_float(job.get("stop_at"), DEFAULTS["stop_at"]))]
    # `.items()` only if it really is a mapping. _clean guarantees that for
    # anything added from now on, but a queue.json written before it did would
    # otherwise raise here - and load() calls this for EVERY job, so one bad
    # entry would 500 the whole API rather than spoiling one row.
    rewards = job.get("rewards")
    if isinstance(rewards, dict):
        for term, weight in rewards.items():
            argv += ["--reward", f"{term}={weight}"]
    for flag in ("push_speed", "push_spin"):
        pair = _as_pair(job.get(flag))
        if pair:
            argv += [f"--{flag.replace('_', '-')}", *(str(v) for v in pair)]
    return argv


def command_line(job: dict) -> str:
    """The same thing as a string, for showing on the page."""
    return "python scripts/train.py " + " ".join(train_argv(job))
