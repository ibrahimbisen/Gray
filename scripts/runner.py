"""Work through the training queue, one job at a time, unattended.

    python run.py --runner

Leave it running. It watches progress/queue.json, and whenever a job is queued
and the card is free it does the whole pipeline for that job:

    train  ->  film  ->  verify  ->  next job

Then it takes the next one. Queue ten runs, go to bed, wake up to ten filmed and
scored results.

**Why this is a separate process and not part of the dashboard.** Closing the
dashboard would kill training. This outlives it - you can restart the dashboard,
or close it entirely, and the queue keeps going.

**Why one at a time, structurally.** RULES.md rule 4. Two runs on one card do not
fail cleanly: both sit at 100% GPU and neither advances, which looks exactly like
a hung dashboard. dashboard/queue.py refuses to hand out a second job while one
is running, so the rule is enforced by the thing handing out work rather than by
remembering it at 2 a.m.

**What happens when it is killed.** A job left marked `running` by a runner that
died would block the queue forever, so on startup this reconciles: any job that
says it is running, when no runner is alive to be running it, is marked
interrupted and the queue moves on.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import queue  # noqa: E402

RUNS = ROOT / "progress" / "runs"
LOGS = ROOT / "progress" / "jobs"
WIN = os.name == "nt"

# How often to look for new work, and how often to check whether the job we are
# running has been cancelled from the dashboard.
POLL_IDLE_S = 3.0
POLL_BUSY_S = 2.0

# How long to wait for train.py to create its run directory before giving up on
# linking the job to it. The job still runs; it just will not have a link.
RUN_DIR_TIMEOUT_S = 300.0


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _say(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _run_dirs() -> set[str]:
    return {p.name for p in RUNS.iterdir() if p.is_dir()} if RUNS.is_dir() else set()


def _popen(argv: list[str], log: Path) -> subprocess.Popen:
    """Start a child with its own process group, so it can be interrupted cleanly.

    CREATE_NEW_PROCESS_GROUP matters: it is what lets us send CTRL_BREAK later.
    train.py catches KeyboardInterrupt and uses it to write its final status and
    flush the metrics one last time - a hard kill loses both, and the run then
    shows on the dashboard as permanently "running".
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8", errors="replace")
    handle.write(f"\n=== {_now()}  {' '.join(argv)}\n")
    handle.flush()
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if WIN else 0
    return subprocess.Popen(
        [sys.executable, *argv],
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        creationflags=flags,
        start_new_session=not WIN,
    )


def _interrupt(proc: subprocess.Popen, grace: float = 30.0) -> None:
    """Ask a child to stop, then insist. Politeness first so train.py can tidy up."""
    if proc.poll() is not None:
        return
    try:
        if WIN:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except (OSError, ValueError):
        pass
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    _say("child did not stop when asked - killing it")
    try:
        if WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, check=False)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        proc.kill()


def _cancelled(job_id: str) -> bool:
    """Has the dashboard taken this job away from us?"""
    state = queue.load()
    job = next((j for j in state["jobs"] if j.get("id") == job_id), None)
    return job is None or job.get("state") != "running"


def _wait(proc: subprocess.Popen, job_id: str, label: str) -> int:
    """Wait for a child, watching for cancellation. Returns its exit code."""
    while True:
        # Heartbeat from in here too, not only from the idle loop - otherwise a
        # six-hour training job looks exactly like a runner that died.
        queue.beat(state=label, job=job_id)
        code = proc.poll()
        if code is not None:
            return code
        if _cancelled(job_id):
            _say(f"{label}: cancelled from the dashboard")
            _interrupt(proc)
            return proc.poll() if proc.poll() is not None else -1
        time.sleep(POLL_BUSY_S)


def _find_run_dir(before: set[str], deadline: float) -> str | None:
    """The run directory train.py just created. It names it by the clock, so it
    has to be found rather than predicted."""
    while time.time() < deadline:
        new = _run_dirs() - before
        if new:
            return sorted(new)[-1]
        time.sleep(1.0)
    return None


def _step(job_id: str, name: str, started: float, code: int | None) -> None:
    """Record one pipeline step on the job, so the page can show what happened."""
    state = queue.load()
    job = next((j for j in state["jobs"] if j.get("id") == job_id), None)
    steps = list(job.get("steps") or []) if job else []
    steps.append({
        "name": name,
        "seconds": round(time.time() - started, 1),
        "exit_code": code,
        "at": _now(),
    })
    queue.update(job_id, steps=steps)


def run_job(job: dict) -> None:
    """train -> film -> verify, for one job."""
    job_id = job["id"]
    log = LOGS / f"{job_id}.log"
    queue.update(job_id, state="running", started=_now(), log=str(
        log.relative_to(ROOT)).replace("\\", "/"))
    _say(f"{job_id}  {job['task']}  {queue.command_line(job)}")

    before = _run_dirs()
    started = time.time()
    train = _popen(["scripts/train.py", *queue.train_argv(job)], log)

    # Link the job to its run as soon as the directory appears, rather than at
    # the end - the dashboard wants to show live curves while it is training.
    run_id = _find_run_dir(before, started + RUN_DIR_TIMEOUT_S)
    if run_id:
        queue.update(job_id, run_id=run_id)
        _say(f"{job_id}  -> progress/runs/{run_id}")
    else:
        _say(f"{job_id}  no run directory appeared - the job runs, but unlinked")

    # Filming reads the checkpoints training writes, so it runs alongside rather
    # than after. It is a separate process and its own failure: a broken renderer
    # should cost the videos, not the run.
    film = None
    if job.get("film") and not job.get("no_video"):
        film = _popen(["scripts/film_checkpoints.py", "--watch"],
                      LOGS / f"{job_id}.film.log")

    code = _wait(train, job_id, "train")
    _step(job_id, "train", started, code)

    if film is not None:
        # Give it a moment to catch the final checkpoint, then stop it.
        time.sleep(20.0)
        _interrupt(film, grace=60.0)

    if _cancelled(job_id):
        _say(f"{job_id}  cancelled")
        return
    if code != 0:
        queue.update(job_id, state="failed", finished=_now(), exit_code=code,
                     error=f"training exited {code} - see {log.name}")
        _say(f"{job_id}  FAILED (exit {code})")
        return

    # RULES.md rule 2: a stage is passed by its bar, not by its curve. Training
    # finishing is not the same as the stage being passed, so this always runs
    # separately and writes its own verdict onto the run.
    if job.get("verify"):
        at = time.time()
        argv = ["scripts/verify.py", job["task"]]
        # verify.py's --run names an MJLAB log directory under logs/rsl_rl/<exp>/,
        # NOT the progress/runs/ folder. The two happen to share a naming scheme
        # and usually match - but they are stamped a moment apart, so a run that
        # straddles a second boundary would get two different names and --run
        # would point at nothing. Pass it only when it actually resolves;
        # otherwise verify.py's own default - the newest log directory - is right,
        # because the runner is serial and the newest IS the one just trained.
        exp = ROOT / "logs" / "rsl_rl" / f"gray_{job['task'].split('-')[-1].lower()}"
        if run_id and (exp / run_id).is_dir():
            argv += ["--run", run_id]
        else:
            _say(f"{job_id}  verifying the newest {exp.name} run "
                 f"(no log dir named {run_id})")
        vcode = _wait(_popen(argv, log), job_id, "verify")
        _step(job_id, "verify", at, vcode)
        if vcode != 0:
            _say(f"{job_id}  verify exited {vcode} - training kept, verdict missing")

    queue.update(job_id, state="done", finished=_now(), exit_code=0)
    _say(f"{job_id}  done")


def reconcile() -> None:
    """Clear jobs left marked running by a runner that died.

    Without this the queue is blocked forever: next_queued() refuses to hand out
    work while anything says it is running, and nothing else will ever change
    that flag.
    """
    for job in queue.load()["jobs"]:
        if job.get("state") == "running":
            queue.update(job["id"], state="failed", finished=_now(),
                         error="the runner stopped while this was training - "
                               "requeue it if the run did not finish")
            _say(f"{job['id']}  was marked running with no runner alive - cleared")


def serve(once: bool = False) -> int:
    _say(f"runner up. interpreter: {sys.executable}")
    _say(f"queue: {queue.QUEUE.relative_to(ROOT)}")
    reconcile()
    idle_said = False
    while True:
        queue.beat(state="looking")
        job = queue.next_queued()
        if job is None:
            state = queue.load()
            if not idle_said:
                if state.get("paused"):
                    _say("paused - not taking new jobs")
                elif state["running"]:
                    _say(f"waiting on {state['running']['id']}")
                else:
                    _say(f"idle - {len(state['queued'])} queued")
                idle_said = True
            if once:
                return 0
            time.sleep(POLL_IDLE_S)
            continue

        idle_said = False
        if job["task"] not in queue.TASKS:
            queue.update(job["id"], state="skipped", finished=_now(),
                         error=f"no task called {job['task']!r}")
            continue
        try:
            run_job(job)
        except KeyboardInterrupt:
            queue.update(job["id"], state="cancelled", finished=_now(),
                         error="runner interrupted")
            _say("interrupted - stopping")
            return 130
        except Exception as exc:  # noqa: BLE001
            queue.update(job["id"], state="failed", finished=_now(),
                         error=f"{type(exc).__name__}: {exc}")
            _say(f"{job['id']}  runner error: {type(exc).__name__}: {exc}")
        if once:
            return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true",
                    help="run at most one job, then exit")
    args = ap.parse_args()
    try:
        return serve(once=args.once)
    except KeyboardInterrupt:
        _say("stopped")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
