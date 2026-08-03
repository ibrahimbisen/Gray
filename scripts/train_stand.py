"""Train stage 1 - stand still - and feed the dashboard while it runs.

    python scripts/train_stand.py                    # 8192 robots at once
    python scripts/train_stand.py --num-envs 4096    # smaller, if VRAM is tight
    python scripts/train_stand.py --iterations 1200

mjlab writes its own logs under logs/rsl_rl/. The dashboard reads
progress/runs/. This runs the training and bridges between the two, so the run
appears at http://127.0.0.1:8000 the moment it starts and the curves fill in
live - rather than being something you only see once it has finished.

It also avoids installing the project: mjlab normally discovers tasks through a
packaging entry point, but importing gray.tasks here registers Gray-Stand just
the same, and calling mjlab's launch_training directly skips the CLI.
"""

from __future__ import annotations

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


def bridge(run_dir: Path, log_dir: Path, stop: threading.Event) -> None:
    while not stop.is_set():
        sync_once(run_dir, log_dir)
        stop.wait(10.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-envs", type=int, default=8192,
                    help="robots trained at once. 8192 fills a 12 GB card")
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--name", default="stand")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    import gray.tasks  # noqa: F401  - registers Gray-Stand
    from mjlab.scripts.train import TrainConfig, launch_training

    cfg = TrainConfig.from_task("Gray-Stand")
    cfg.env.scene.num_envs = args.num_envs
    cfg.agent.max_iterations = args.iterations
    cfg.agent.run_name = args.name
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
    from gray.tasks.stand_env_cfg import REWARD_NOTES  # noqa: PLC0415

    scoring = sorted(
        ({"name": name, "weight": float(term.weight),
          "what": REWARD_NOTES.get(name, "")}
         for name, term in cfg.env.rewards.items()),
        key=lambda r: -abs(r["weight"]),
    )

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.name}"
    run_dir = RUNS / run_id
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({
        "name": "Stand still",
        "purpose": "Can a policy hold the ride-height stance? The static check says "
                   "twelve servos at 1.96 N-m have 3.58x the torque they need, so a "
                   "failure here is the reward or the setup, not the robot.",
        "stage": 1,
        "stage_name": "Stand still",
        "task": "Gray-Stand",
        "status": "running",
        "bar": "30 s without falling. Trunk within 5 mm of 164 mm. Uprightness above 0.99.",
        "started": datetime.now().isoformat(timespec="seconds"),
        "finished": None,
        "iterations_target": args.iterations,
        "notes": f"{args.num_envs} robots at once, 50 Hz control.",
        "scoring": scoring,
    }, indent=2))

    print(f"training      Gray-Stand, {args.num_envs} robots, {args.iterations} iterations")
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
        bridge(run_dir, log_dir, stop)

    threading.Thread(target=watch, daemon=True).start()

    status = "done"
    try:
        launch_training(task_id="Gray-Stand", args=cfg)
    except KeyboardInterrupt:
        status = "stopped"
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
