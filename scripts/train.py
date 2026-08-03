"""Train a stage, and feed the dashboard while it runs.

    python scripts/train.py Gray-Stand
    python scripts/train.py Gray-Push
    python scripts/train.py Gray-Push --num-envs 2048 --iterations 2000

mjlab writes its own logs under logs/rsl_rl/. The dashboard reads
progress/runs/. This runs the training and bridges between the two, so the run
appears at http://127.0.0.1:8000 the moment it starts and the curves fill in
live - rather than being something you only see once it has finished. It also
records what the run is scored on, so the dashboard can show that per run rather
than reading a task file that may since have changed.

It avoids installing the project too: mjlab normally discovers tasks through a
packaging entry point, but importing gray.tasks here registers them just the
same, and calling mjlab's launch_training directly skips the CLI.
"""

from __future__ import annotations

import _thread
import argparse
import csv
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# rsl_rl records which commit a run came from, through GitPython, which refuses
# to import at all if it cannot find git. On Windows git is often on the shell's
# PATH but not the interpreter's, and losing a training run to that would be
# absurd - so point it at git if we can find it, and let it go quiet if we cannot.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
if "GIT_PYTHON_GIT_EXECUTABLE" not in os.environ:
    found = shutil.which("git") or next(
        (p for p in (r"C:\Program Files\Git\cmd\git.exe",
                     r"C:\Program Files (x86)\Git\cmd\git.exe")
         if Path(p).exists()), None)
    if found:
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = found

LOG_ROOT = ROOT / "logs" / "rsl_rl"
RUNS = ROOT / "progress" / "runs"

# What each task is and how it is judged. Kept here rather than in the task file
# so the dashboard shows the same words the plan does, and so a finished run
# still carries the bar it was actually held to.
TASKS = {
    "Gray-Stand": {
        "name": "Stand still",
        "stage": 1,
        "purpose": "Can a policy hold the ride-height stance? The static check says "
                   "twelve servos at 1.96 N-m have 3.57x the torque they need, so a "
                   "failure here is the reward or the setup, not the robot.",
        "bar": "30 s without falling. Trunk within 5 mm of 190 mm. "
               "Uprightness above 0.99.",
    },
    "Gray-Push": {
        "name": "Take a push",
        "stage": 2,
        "purpose": "The first task that cannot be solved by memorising a pose - the "
                   "robot has to notice it was shoved and do something. That makes it "
                   "the first real use of the potentiometers, and it builds the reflex "
                   "walking needs, where every step is a disturbance the robot creates "
                   "for itself.",
        "bar": "Survives repeated shoves from any direction over 20 s, on ground "
               "anywhere from slippery to grippy, with mass and servo gains varied.",
    },
}


def reward_notes() -> dict[str, str]:
    """Plain-English descriptions for every scoring term across the tasks."""
    from gray.tasks.push_env_cfg import PUSH_NOTES  # noqa: PLC0415
    from gray.tasks.stand_env_cfg import REWARD_NOTES  # noqa: PLC0415

    return {**REWARD_NOTES, **PUSH_NOTES}

# rsl_rl's tensorboard tags, and what to call them on the dashboard. Only the
# handful worth watching - the run page is a monitor, not an archive.
WATCH = {
    "Train/mean_reward": "reward",
    "Train/mean_episode_length": "episode_length",
    "Episode_Reward/height": "height_reward",
    "Episode_Reward/tilt": "tilt_penalty",
    "Episode_Reward/upright": "tilt_penalty",   # what the term was called before
    "Episode_Termination/tipped_over": "tipped_over",
    "Episode_Termination/collapsed": "collapsed",
    "Policy/mean_std": "exploration",
    "Perf/total_fps": "steps_per_second",
}


def newest_log_dir(after: float, timeout: float = 900.0) -> Path | None:
    """mjlab names its run directory by the clock, so find it rather than guess."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if LOG_ROOT.is_dir():
            for exp in LOG_ROOT.iterdir():
                for run in exp.iterdir() if exp.is_dir() else []:
                    if run.is_dir() and run.stat().st_mtime >= after - 5:
                        return run
        time.sleep(1.0)
    return None


def read_scalars(log_dir: Path) -> tuple[list[str], list[dict]]:
    """Pull the curves out of the tensorboard file training is writing to."""
    from tensorboard.backend.event_processing.event_accumulator import (  # noqa: PLC0415
        EventAccumulator,
    )

    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 100_000})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    rows: dict[int, dict] = {}
    used: list[str] = []
    for tag, name in WATCH.items():
        if tag not in available:
            continue
        used.append(name)
        for ev in acc.Scalars(tag):
            rows.setdefault(ev.step, {"iteration": ev.step})[name] = round(ev.value, 6)
    ordered = [rows[k] for k in sorted(rows)]
    return used, ordered


def sync_once(run_dir: Path, log_dir: Path) -> None:
    """Copy metrics, videos and checkpoint markers into the shape the dashboard reads."""
    try:
        cols, rows = read_scalars(log_dir)
        if rows:
            names = ["iteration"] + [c for c in cols if c != "iteration"]
            with (run_dir / "metrics.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in names})

        # mjlab's own recorder is not used for the dashboard: it names files by
        # env-step, which collides with the checkpoint iterations that
        # scripts/film_checkpoints.py writes, and both then show as "iteration 0".
        # Run that script with --watch alongside training instead.

        # The dashboard only counts checkpoints, so a marker per .pt is enough
        # and keeps a few hundred MB of weights out of progress/.
        for pt in log_dir.glob("model_*.pt"):
            marker = run_dir / "checkpoints" / f"{pt.stem}.npz"
            if not marker.exists():
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_bytes(b"")
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] {type(exc).__name__}: {exc}", flush=True)


def reward_ceiling(rewards) -> float:
    """The most an episode could possibly score.

    Every positive term is capped at 1.0, weighted, multiplied by the timestep
    and summed over the episode - so the largest reachable total is the positive
    weights times the episode length. See RULES.md, rule 1.
    """
    return sum(t.weight for t in rewards.values() if t.weight > 0)


def bridge(run_dir: Path, log_dir: Path, stop: threading.Event,
           target: float = 0.0) -> None:
    """Feed the dashboard, and stop training once the reward has nothing left to win."""
    while not stop.is_set():
        sync_once(run_dir, log_dir)
        if target:
            try:
                _, rows = read_scalars(log_dir)
                latest = rows[-1].get("reward") if rows else None
                if latest is not None and latest >= target:
                    print(f"\n[stop] reward {latest:.2f} reached the {target:.2f} "
                          f"target - every positive term is maxed, so the rest of the "
                          f"schedule would only polish. See RULES.md rule 1.",
                          flush=True)
                    stop.set()
                    # rsl_rl has no way to be asked to stop mid-learn, so raise
                    # into the main thread. The last checkpoint is at most 25
                    # iterations behind, and training's own finally block runs.
                    _thread.interrupt_main()
                    return
            except Exception as exc:  # noqa: BLE001
                print(f"[stop] could not read the reward: {exc}", flush=True)
        stop.wait(10.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", nargs="?", default="Gray-Stand", choices=sorted(TASKS),
                    help="which stage to train")
    ap.add_argument("--num-envs", type=int, default=3072,
                    help="robots trained at once. 4096 is this card's ceiling; "
                         "3072 leaves room for the video renderer")
    ap.add_argument("--iterations", type=int, default=0,
                    help="0 uses the task's own default")
    ap.add_argument("--name", default="")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--stop-at", type=float, default=0.98, metavar="FRACTION",
                    help="stop once the reward reaches this fraction of the most it "
                         "could score. 0 runs the full schedule. See RULES.md rule 1.")
    ap.add_argument("--push-speed", type=float, nargs=2, metavar=("MIN", "MAX"),
                    help="how hard the shoves are, m/s of instant trunk speed. "
                         "On 2.379 kg, 1 m/s is about 2.4 N-s.")
    ap.add_argument("--push-spin", type=float, nargs=2, metavar=("MIN", "MAX"),
                    help="how much spin each shove adds, rad/s about the vertical")
    ap.add_argument("--reward", action="append", default=[], metavar="NAME=WEIGHT",
                    help="change one scoring weight, e.g. --reward twitching=-1.5. "
                         "Repeatable. The change is recorded in the run, so the "
                         "dashboard shows what this run was actually scored on.")
    args = ap.parse_args()

    import gray.tasks  # noqa: F401  - registers the tasks
    from mjlab.scripts.train import TrainConfig, launch_training

    meta = TASKS[args.task]
    run_name = args.name or args.task.split("-")[-1].lower()

    cfg = TrainConfig.from_task(args.task)
    cfg.env.scene.num_envs = args.num_envs
    if args.iterations:
        cfg.agent.max_iterations = args.iterations
    iterations = cfg.agent.max_iterations
    cfg.agent.run_name = run_name

    shove = cfg.env.events.get("shove")
    if shove is None and (args.push_speed or args.push_spin):
        raise SystemExit(f"{args.task} has no shove event to change")
    if args.push_speed:
        was = shove.params["speed_range"]
        shove.params["speed_range"] = tuple(args.push_speed)
        print(f"shove         speed: {was} -> {tuple(args.push_speed)} m/s "
              f"(up to {args.push_speed[1] * 2.3787:.1f} N-s)")
    if args.push_spin:
        was = shove.params["spin_range"]
        shove.params["spin_range"] = tuple(args.push_spin)
        print(f"shove         spin: {was} -> {tuple(args.push_spin)} rad/s")

    for change in args.reward:
        term, _, weight = change.partition("=")
        if term not in cfg.env.rewards:
            raise SystemExit(
                f"no scoring term called {term!r}. This run has: "
                f"{', '.join(sorted(cfg.env.rewards))}")
        was = cfg.env.rewards[term].weight
        cfg.env.rewards[term].weight = float(weight)
        print(f"scoring       {term}: {was} -> {float(weight)}")
    cfg = TrainConfig(
        env=cfg.env, agent=cfg.agent,
        video=not args.no_video,
        video_length=250,
        # Every 25 iterations. The recorder counts calls to env.step(), NOT
        # robot-steps - multiplying by num_envs makes the interval so large it
        # only ever fires once, at step 0, which is what happened on the first
        # real run. num_steps_per_env is 24.
        video_interval=24 * 25,
        log_root=str(LOG_ROOT.relative_to(ROOT)),
        gpu_ids=[0],
    )

    # Record what this run is actually scored on, so the dashboard can show it
    # without anyone reading the task file - and so an old run still says what it
    # was scored on after the rewards have been changed.
    notes = reward_notes()
    missing = [n for n in cfg.env.rewards if n not in notes]
    if missing:
        # A term with no description shows as a blank row on the dashboard, which
        # is worse than no row - say so rather than let it slip through.
        print(f"[warn] no description for scoring terms: {', '.join(missing)}")
    scoring = sorted(
        ({"name": name, "weight": float(term.weight), "what": notes.get(name, "")}
         for name, term in cfg.env.rewards.items()),
        key=lambda r: -abs(r["weight"]),
    )

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{run_name}"
    run_dir = RUNS / run_id
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({
        "name": meta["name"],
        "purpose": meta["purpose"],
        "stage": meta["stage"],
        "stage_name": meta["name"],
        "task": args.task,
        "status": "running",
        "bar": meta["bar"],
        "started": datetime.now().isoformat(timespec="seconds"),
        "finished": None,
        "iterations_target": iterations,
        "notes": f"{args.num_envs} robots at once, 50 Hz control."
                 + (f" Shoves {shove.params['speed_range'][0]}-"
                    f"{shove.params['speed_range'][1]} m/s from any angle, spin "
                    f"{shove.params['spin_range'][1]} rad/s." if shove else ""),
        "scoring": scoring,
    }, indent=2))

    ceiling = reward_ceiling(cfg.env.rewards) * cfg.env.episode_length_s
    target = ceiling * args.stop_at if args.stop_at else 0.0

    print(f"training      {args.task} - {meta['name']}")
    print(f"              {args.num_envs} robots, up to {iterations} iterations")
    print(f"reward        ceiling {ceiling:.1f} "
          f"({reward_ceiling(cfg.env.rewards):.1f}/s over "
          f"{cfg.env.episode_length_s:.0f} s)")
    print(f"              stopping at {args.stop_at:.0%} of it = {target:.1f}"
          if target else "              running the full schedule")
    print(f"dashboard     http://127.0.0.1:8000  ->  {run_id}")
    print(f"tensorboard   tensorboard --logdir {LOG_ROOT}\n")

    started = time.time()
    stop = threading.Event()
    found: dict[str, Path] = {}

    def watch():
        log_dir = newest_log_dir(started)
        if log_dir is None:
            print("[bridge] never found the log directory; dashboard will stay empty")
            return
        found["dir"] = log_dir
        print(f"[bridge] following {log_dir}", flush=True)
        bridge(run_dir, log_dir, stop, target)

    threading.Thread(target=watch, daemon=True).start()

    status = "finished"
    try:
        launch_training(task_id=args.task, args=cfg)
    except KeyboardInterrupt:
        # Either Ctrl-C, or the reward hit its target and the bridge interrupted
        # us deliberately. Those are opposite outcomes and must not read the same.
        status = "reached target" if stop.is_set() else "cancelled"
    except Exception:
        status = "failed"
        raise
    finally:
        stop.set()
        # One last pass. A short run can finish inside the bridge's poll interval,
        # and then the dashboard would show a run with no curves at all - which is
        # exactly what happened the first time.
        if "dir" in found:
            sync_once(run_dir, found["dir"])
        meta = json.loads((run_dir / "run.json").read_text())
        meta["status"] = status
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
        (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
        print(f"\nrun {status}: {run_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
