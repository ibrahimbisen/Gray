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
a hung dashboard. The job is CLAIMED under the queue's lock - found and marked
running in a single edit - so two runners racing cannot both take it.

**What happens when it is killed.** Two protections. Every child is tracked and
killed in a `finally`, so Ctrl-C does not leave training burning the GPU with the
queue saying "cancelled". And on startup a job left marked `running` is cleared -
but ONLY after checking no other runner is alive, because a second runner
clearing the first one's job is how you end up with two trainers on one card.
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

# This runner's identity, recorded on a job it claims.
ME = f"{os.getpid()}@{datetime.now():%H%M%S}"

# Every child we have started and not yet reaped, so a `finally` can kill them
# all. Without this, Ctrl-C on the runner leaves training running - the children
# are in their own process groups precisely so the console signal does NOT reach
# them - and the queue says the job was cancelled while the GPU is still busy.
_CHILDREN: list[tuple[subprocess.Popen, object]] = []


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _say(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _beat(**extra) -> None:
    """Never let a heartbeat failure escape.

    queue.beat() writes a file, and on Windows that write can lose a race with a
    dashboard poll. An exception here used to propagate out of the wait loop and
    mark a job failed while its training was still running.
    """
    try:
        queue.beat(**extra)
    except OSError:
        pass


def _run_dirs() -> set[str]:
    return {p.name for p in RUNS.iterdir() if p.is_dir()} if RUNS.is_dir() else set()


def _popen(argv: list[str], log: Path) -> subprocess.Popen:
    """Start a child with its own process group, so it can be interrupted cleanly.

    CREATE_NEW_PROCESS_GROUP matters: it is what lets us send CTRL_BREAK later.
    train.py installs a SIGBREAK handler so that arrives as KeyboardInterrupt and
    its `finally` gets to write the run's final status.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8", errors="replace")
    handle.write(f"\n=== {_now()}  {' '.join(argv)}\n")
    handle.flush()
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if WIN else 0
    proc = subprocess.Popen(
        [sys.executable, *argv],
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        creationflags=flags,
        start_new_session=not WIN,
    )
    _CHILDREN.append((proc, handle))
    return proc


def _reap(proc: subprocess.Popen) -> None:
    """Drop a finished child and close its log handle.

    The handle has to be closed explicitly - it is not bound to the Popen, so it
    would leak one file descriptor per child, three per job, for as long as the
    runner lives.
    """
    for i, (p, handle) in enumerate(_CHILDREN):
        if p is proc:
            try:
                handle.close()
            except OSError:
                pass
            _CHILDREN.pop(i)
            return


def _interrupt(proc: subprocess.Popen, grace: float = 30.0, job_id: str = "") -> None:
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

    # Beat while waiting: a 30 s grace period is longer than HEARTBEAT_STALE_S,
    # so without this the dashboard declares the runner dead mid-shutdown.
    deadline = time.time() + grace
    while time.time() < deadline:
        if proc.poll() is not None:
            _reap(proc)
            return
        _beat(state="stopping", job=job_id)
        time.sleep(1.0)

    _say("child did not stop when asked - killing it")
    try:
        if WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, check=False)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        proc.kill()
    # Confirm it actually died. taskkill returns before the process is gone, so
    # without this wait the caller reads poll() as None and reports the child as
    # still alive when it is not - or worse, moves on while it really is.
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        _say(f"WARNING pid {proc.pid} survived taskkill - the GPU may still be busy")
    _reap(proc)


def _kill_all(why: str) -> None:
    for proc, _ in list(_CHILDREN):
        if proc.poll() is None:
            _say(f"{why}: stopping pid {proc.pid}")
            _interrupt(proc, grace=20.0)
    for proc, handle in list(_CHILDREN):
        try:
            handle.close()
        except OSError:
            pass
        _CHILDREN.clear()


def _cancelled(job_id: str) -> bool:
    """Has the dashboard taken this job away from us?"""
    try:
        state = queue.load()
    except OSError:
        return False        # could not read; assume not, and check again shortly
    job = next((j for j in state["jobs"] if j.get("id") == job_id), None)
    return job is None or job.get("state") != "running"


def _wait(proc: subprocess.Popen, job_id: str, label: str) -> int:
    """Wait for a child, watching for cancellation. Returns its exit code."""
    while True:
        # Heartbeat from in here too, not only from the idle loop - otherwise a
        # six-hour training job looks exactly like a runner that died.
        _beat(state=label, job=job_id)
        code = proc.poll()
        if code is not None:
            _reap(proc)
            return code
        if _cancelled(job_id):
            _say(f"{label}: cancelled from the dashboard")
            _interrupt(proc, job_id=job_id)
            return proc.poll() if proc.poll() is not None else -1
        time.sleep(POLL_BUSY_S)


def _find_run_dir(before: set[str], deadline: float, job_id: str) -> str | None:
    """The run directory train.py just created. It names it by the clock, so it
    has to be found rather than predicted.

    Beats while it waits and honours cancellation: torch and mjlab take 30-120 s
    to start, which is far longer than HEARTBEAT_STALE_S, so a silent wait here
    made the dashboard report "the runner stopped" at the start of every job.
    """
    while time.time() < deadline:
        new = _run_dirs() - before
        if new:
            return sorted(new)[-1]
        _beat(state="starting", job=job_id)
        if _cancelled(job_id):
            return None
        time.sleep(1.0)
    return None


def _training_started(run_id: str | None, job_id: str, deadline: float) -> bool:
    """Wait until training has actually completed an iteration.

    metrics.csv gets a data row only once train.py's bridge has read a scalar out
    of the tensorboard file, which cannot happen until the simulator is stepping.
    By then CUDA graph capture is long finished, so a second process starting on
    the device is safe.

    Returns False if it never gets there - cancelled, or training died on
    startup - in which case there is nothing to film anyway.
    """
    if not run_id:
        return False
    metrics = RUNS / run_id / "metrics.csv"
    while time.time() < deadline:
        try:
            # Header plus at least one row.
            if metrics.is_file() and len(metrics.read_text().splitlines()) >= 2:
                return True
        except OSError:
            pass
        _beat(state="starting", job=job_id)
        if _cancelled(job_id):
            return False
        time.sleep(2.0)
    return False


def _step(job_id: str, name: str, started: float, code: int | None) -> None:
    """Record one pipeline step on the job, so the page can show what happened."""
    try:
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
    except OSError as exc:
        _say(f"could not record the {name} step: {exc}")


def run_job(job: dict) -> None:
    """train -> film -> verify, for one job. Already claimed and marked running."""
    job_id = job["id"]
    log = LOGS / f"{job_id}.log"
    queue.update(job_id, log=str(log.relative_to(ROOT)).replace("\\", "/"))
    _say(f"{job_id}  {job['task']}  {queue.command_line(job)}")

    before = _run_dirs()
    started = time.time()
    try:
        train = _popen(["scripts/train.py", *queue.train_argv(job)], log)

        # Link the job to its run as soon as the directory appears, rather than
        # at the end - the dashboard wants live curves while it is training.
        run_id = _find_run_dir(before, started + RUN_DIR_TIMEOUT_S, job_id)
        if run_id:
            queue.update(job_id, run_id=run_id)
            _say(f"{job_id}  -> progress/runs/{run_id}")
        else:
            _say(f"{job_id}  no run directory appeared - the job runs, but unlinked")

        # Filming reads the checkpoints training writes, so it runs alongside
        # rather than after. Its own process and its own failure: a broken
        # renderer should cost the videos, not the run.
        #
        # But NOT until training is actually iterating. Starting it five seconds
        # after train.py cost a whole run: the film process created its own CUDA
        # context on the same device while training was capturing its forward
        # graph, and training died with "Warp CUDA error 600: device not ready"
        # out of wp.capture_launch. That was misread as the card running out of
        # memory and blamed on the robot count for half a day; it was a race.
        #
        # Nothing is lost by waiting. Filming works from checkpoints on disk and
        # the first one is not written until iteration 25.
        film = None
        if job.get("film") and not job.get("no_video"):
            if _training_started(run_id, job_id, started + RUN_DIR_TIMEOUT_S):
                film = _popen(["scripts/film_checkpoints.py", "--watch"],
                              LOGS / f"{job_id}.film.log")
            else:
                _say(f"{job_id}  training never reported an iteration - not filming")

        code = _wait(train, job_id, "train")
        _step(job_id, "train", started, code)

        if film is not None:
            time.sleep(20.0)        # let it catch the final checkpoint
            _interrupt(film, grace=60.0, job_id=job_id)

        if _cancelled(job_id):
            _say(f"{job_id}  cancelled")
            return
        if code != 0:
            queue.update(job_id, state="failed", finished=_now(), exit_code=code,
                         error=f"training exited {code} - see {log.name}")
            _say(f"{job_id}  FAILED (exit {code})")
            return

        # RULES.md rule 2: a stage is passed by its bar, not by its curve.
        if job.get("verify"):
            at = time.time()
            argv = ["scripts/verify.py", job["task"]]
            # verify.py's --run names an MJLAB log directory under
            # logs/rsl_rl/<exp>/, NOT the progress/runs/ folder. The two share a
            # naming scheme and usually match - but they are stamped a moment
            # apart, so a run straddling a second boundary gets two different
            # names. Pass it only when it resolves; otherwise verify.py's own
            # default (the newest log directory) is right, because this runner is
            # serial and the newest IS the one just trained.
            exp = ROOT / "logs" / "rsl_rl" / f"gray_{job['task'].split('-')[-1].lower()}"
            if run_id and (exp / run_id).is_dir():
                argv += ["--run", run_id]
            else:
                _say(f"{job_id}  verifying the newest {exp.name} run "
                     f"(no log dir named {run_id})")
            vcode = _wait(_popen(argv, log), job_id, "verify")
            _step(job_id, "verify", at, vcode)
            if vcode != 0:
                _say(f"{job_id}  verify exited {vcode} - "
                     f"training kept, the bar was not met or could not be read")

        # Re-check AFTER verify. Verifying takes minutes, and a cancellation that
        # arrived during it would otherwise be overwritten with a clean "done" -
        # the job would read as a success and the reason would be erased.
        if _cancelled(job_id):
            _say(f"{job_id}  cancelled during verify")
            return

        queue.update(job_id, state="done", finished=_now(), exit_code=0)
        _say(f"{job_id}  done")
    finally:
        # Whatever happened - a raise, a Ctrl-C, a clean finish - no child of
        # this job outlives it. An orphaned trainer holds the GPU and the next
        # job starts alongside it, which is the rule-4 deadlock.
        _kill_all(f"{job_id} finishing")


def _pid_alive(pid) -> bool:
    """Is that process id actually running?

    NOT os.kill(pid, 0): on Windows os.kill only understands SIGTERM and the
    CTRL_* events, and anything else calls TerminateProcess - so the "harmless
    existence check" every other platform uses would kill the process here.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if not WIN:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes  # noqa: PLC0415

    SYNCHRONIZE = 0x00100000
    handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return False
    # WAIT_TIMEOUT (258) means it is still running; WAIT_OBJECT_0 means it exited
    # but the handle is not yet closed everywhere.
    alive = ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 258
    ctypes.windll.kernel32.CloseHandle(handle)
    return alive


def reconcile() -> None:
    """Clear jobs left marked running by a runner that died.

    Without this the queue is blocked forever: claim() refuses to hand out work
    while anything says it is running, and nothing else will change that flag.

    But it checks the heartbeat FIRST. A second runner started while the first is
    three hours into a job would otherwise mark that job failed, see nothing
    running, and launch the next one alongside it - two trainers on one card,
    the exact failure rule 4 exists to prevent.
    """
    state = queue.load()
    stuck = [j for j in state["jobs"] if j.get("state") == "running"]
    if not stuck:
        return
    runner = state["runner"]
    # A fresh heartbeat is not proof: HEARTBEAT_STALE_S is 30 s, so a runner
    # killed a moment ago still looks alive and would block its own replacement
    # for half a minute, printing the dead pid and telling you to stop it. Ask
    # the operating system whether that process still exists.
    other = runner.get("pid")
    if runner.get("alive") and other != os.getpid() and _pid_alive(other):
        _say(f"another runner is alive (pid {other}, last seen "
             f"{runner.get('age_s')}s ago) and holds "
             f"{', '.join(j['id'] for j in stuck)}.")
        _say("refusing to start - two trainers on one card is RULES.md rule 4. "
             "Stop the other runner first.")
        raise SystemExit(1)
    if runner.get("alive"):
        _say(f"the previous runner (pid {other}) left a fresh heartbeat but is "
             f"gone - clearing its job rather than waiting the heartbeat out")
    for job in stuck:
        queue.update(job["id"], state="failed", finished=_now(),
                     error="the runner stopped while this was training - "
                           "requeue it if the run did not finish")
        _say(f"{job['id']}  was marked running with no runner alive - cleared")


def serve(once: bool = False) -> int:
    _say(f"runner {ME} up. interpreter: {sys.executable}")
    _say(f"queue: {queue.QUEUE.relative_to(ROOT)}")
    reconcile()
    idle_said = False
    while True:
        _beat(state="looking")
        try:
            job = queue.claim(ME)
        except OSError as exc:
            _say(f"could not read the queue: {exc}")
            time.sleep(POLL_IDLE_S)
            continue

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
                         error="the runner was interrupted")
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
    finally:
        _kill_all("runner exiting")


if __name__ == "__main__":
    raise SystemExit(main())
