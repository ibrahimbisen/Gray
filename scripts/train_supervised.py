#!/usr/bin/env python3
"""Run a training job to completion, surviving crashes.

    python scripts/train_supervised.py --run-name A_base --iterations 3000
    python scripts/train_supervised.py --run-name B_long --iterations 3000 -- \
        --env.scene.num-envs 12288 --agent.num-steps-per-env 48

Everything after a bare `--` is passed straight through to scripts/train_residual.py,
so any trainer flag still works.

WHY THIS EXISTS. A run died at round 108 of 3000 with

    Warp CUDA error 700: an illegal memory access was encountered

and nothing was watching, so the graphics card then sat idle. That fault is not
reproducible on demand and is not obviously caused by this project's code - it
surfaced while the video renderer was scoring a checkpoint on the same device the
trainer was using. Whatever its cause, a month of unattended 24/7 training cannot
be one random driver fault away from stopping, and a run that dies at round 108 of
3000 has thrown away every hour after it.

So: relaunch from the newest checkpoint and keep going.

WHAT IT DOES NOT DO. It does not retry a run that failed for a reason retrying
cannot fix - a typo in a flag, a missing task, an out-of-memory at this env count.
Those fail identically every time and would spin forever. The guard is
CONSECUTIVE_FAILURE_LIMIT together with the "did this attempt make progress"
check: an attempt that produces no new checkpoint counts as a failure to progress,
and enough of those in a row stops the supervisor rather than burning the night on
a loop that cannot succeed.

ROUND COUNTING IS RELATIVE, WHICH IS EASY TO GET WRONG. rsl_rl's
`learn(num_learning_iterations)` runs that many MORE rounds; it is not a target to
count up to. So a resume must be told the REMAINDER, not the original total, or a
job that crashed at round 108 of 3000 would run to 3108. The remainder is worked
out from the highest model_<n>.pt on disk, which is the only durable record of how
far the job actually got.

Each attempt writes its own timestamped log directory - that is mjlab's behaviour
and is not worked around here, because a resumed segment genuinely has different
provenance from the one before it. They share the --run-name suffix, so the
segments of one job are recognisable together.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_residual.py"
WATCHER = ROOT / "scripts" / "render_watcher.py"
EXPERIMENT_DIR = ROOT / "logs" / "rsl_rl" / "gray_residual"
TASK_ID = "Gray-Residual-Flat"

# Stop after this many attempts in a row that fail WITHOUT advancing the checkpoint
# count. Three is enough to ride out a transient driver fault and few enough that a
# genuinely broken configuration is not retried all night.
CONSECUTIVE_FAILURE_LIMIT = 3

# Wait between attempts. The GPU needs a moment to tear down a context that died
# mid-kernel; relaunching instantly onto a half-freed device is how one crash
# becomes several.
RESTART_DELAY_S = 20.0

_CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")


def _segments(run_name: str) -> list[Path]:
    """Every log directory belonging to this job, oldest first.

    Matched on the --run-name suffix mjlab appends to the timestamp, which is what
    ties the segments of one interrupted job together.
    """
    if not EXPERIMENT_DIR.is_dir():
        return []
    suffix = f"_{run_name}"
    found = [p for p in EXPERIMENT_DIR.iterdir()
             if p.is_dir() and p.name.endswith(suffix)]
    return sorted(found, key=lambda p: p.name)


def _latest_checkpoint(run_dir: Path) -> tuple[int, str] | None:
    """(round number, filename) of the highest checkpoint in `run_dir`, or None.

    By the number IN the filename rather than by mtime: mjlab also writes the
    exported .onnx and the params/ directory into this folder, and a checkpoint
    written before a later one was overwritten would sort wrongly by time.
    """
    best: tuple[int, str] | None = None
    try:
        names = os.listdir(run_dir)
    except OSError:
        return None
    for name in names:
        match = _CHECKPOINT_RE.match(name)
        if not match:
            continue
        number = int(match.group(1))
        if best is None or number > best[0]:
            best = (number, name)
    return best


def _progress(run_name: str) -> tuple[int, Path | None, str | None]:
    """(rounds completed, the directory holding it, its checkpoint filename).

    Scans every segment, not just the newest: an attempt that crashed during
    start-up leaves an empty directory behind, and resuming from that instead of
    from the segment before it would silently restart the job from zero.
    """
    best_round = 0
    best_dir: Path | None = None
    best_file: str | None = None
    for segment in _segments(run_name):
        found = _latest_checkpoint(segment)
        if found and found[0] >= best_round:
            best_round, best_file = found
            best_dir = segment
    return best_round, best_dir, best_file


def _start_watcher() -> subprocess.Popen | None:
    """Film every checkpoint this job saves, for the whole job.

    STARTED HERE BECAUSE TRAINING IS WHAT PRODUCES CHECKPOINTS. It used to be run.py's
    job, so filming only happened if the dashboard happened to be up - and after the
    duplicate watchers were cleaned up following a crash, a run reached round 1600 with
    31 saved checkpoints and not one clip of any of them. Nobody noticed, because the
    page correctly said "not scored yet" and that looked like a fresh run rather than a
    missing process.

    Started ONCE for the whole job rather than per attempt: a resumed segment must not
    add a second watcher. render_watcher holds a single-instance lock anyway, so a
    duplicate exits immediately rather than competing for the GPU - this is belt and
    braces, and the lock is the part that actually guarantees it.
    """
    if not WATCHER.exists():
        print("  no render_watcher.py, so checkpoints will not be filmed", flush=True)
        return None
    try:
        proc = subprocess.Popen(
            [sys.executable, str(WATCHER)],
            cwd=str(ROOT),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        # Filming is not worth losing the training run over.
        print(f"  could not start the video watcher ({exc}); training anyway",
              flush=True)
        return None
    print("  filming checkpoints as they are saved", flush=True)
    return proc


def _launch(passthrough: list[str], run_name: str, iterations: int,
            resume_from: tuple[Path, str] | None) -> int:
    cmd = [sys.executable, str(TRAINER), TASK_ID,
           "--agent.run-name", run_name,
           "--agent.max-iterations", str(iterations)]
    if resume_from is not None:
        run_dir, checkpoint = resume_from
        cmd += ["--agent.resume", "True",
                "--agent.load-run", run_dir.name,
                "--agent.load-checkpoint", checkpoint]
    cmd += passthrough

    print(f"  {' '.join(cmd[1:])}", flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def main(argv: list[str] | None = None) -> int:
    # Split on a bare "--" ourselves. argparse's own handling of it is inconsistent
    # across versions, and everything to its right is meant to be opaque here.
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1:]

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True,
                    help="suffix identifying this job, e.g. A_base")
    ap.add_argument("--iterations", type=int, default=3000,
                    help="total rounds to reach, counting work already done")
    args = ap.parse_args(argv)

    # Started before the first attempt and stopped after the last, so it spans the
    # whole job including any crash-resume in the middle - a watcher started per
    # attempt would be torn down and rebuilt at exactly the moment the GPU is
    # recovering from whatever killed the trainer.
    watcher = _start_watcher()
    try:
        return _supervise(args, passthrough)
    finally:
        if watcher is not None and watcher.poll() is None:
            watcher.terminate()
            print("video watcher stopped.", flush=True)


def _supervise(args, passthrough: list[str]) -> int:
    attempt = 0
    consecutive_failures = 0

    while True:
        done, run_dir, checkpoint = _progress(args.run_name)
        remaining = args.iterations - done

        if remaining <= 0:
            print(f"\n{args.run_name}: {done}/{args.iterations} rounds done. "
                  f"Finished.", flush=True)
            return 0

        attempt += 1
        resume_from = (run_dir, checkpoint) if run_dir and checkpoint else None
        where = (f"resuming from {run_dir.name}/{checkpoint}" if resume_from
                 else "starting fresh")
        print(f"\n=== {args.run_name} attempt {attempt}: {done}/{args.iterations} "
              f"rounds done, {remaining} to go, {where}", flush=True)

        code = _launch(passthrough, args.run_name, remaining, resume_from)

        done_after, _, _ = _progress(args.run_name)
        advanced = done_after > done

        if code == 0 and not advanced and remaining > 0:
            # A clean exit that saved nothing means the trainer decided there was
            # nothing to do. Retrying cannot change that.
            print(f"{args.run_name}: trainer exited cleanly without saving a "
                  f"checkpoint. Not retrying.", flush=True)
            return 1

        if code == 0:
            continue                       # loop re-checks the round count and stops

        consecutive_failures = 0 if advanced else consecutive_failures + 1
        print(f"{args.run_name}: attempt {attempt} exited with code {code}; "
              f"rounds {done} -> {done_after}"
              f"{'' if advanced else f' (no progress, {consecutive_failures} in a row)'}",
              flush=True)

        if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
            print(f"{args.run_name}: {consecutive_failures} attempts in a row made "
                  f"no progress. Stopping - this is not a transient fault.",
                  flush=True)
            return code

        time.sleep(RESTART_DELAY_S)


if __name__ == "__main__":
    raise SystemExit(main())
