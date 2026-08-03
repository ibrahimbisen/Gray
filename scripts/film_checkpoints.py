"""Film every checkpoint as training saves it, so the robot can be watched learning.

    python scripts/film_checkpoints.py --watch      # follow whatever is training
    python scripts/film_checkpoints.py              # one pass over the newest run
    python scripts/film_checkpoints.py --task Gray-Push --every 4

mjlab's own recorder counts calls to env.step() rather than robot-steps, so its
interval is easy to get wrong by a factor of num_envs. This films from the saved
checkpoints instead - training writes one every 25 iterations regardless - and
with --watch each new one is filmed within half a minute of appearing.

It follows whichever run is currently being written to, across every task. An
earlier version hardcoded one task's log folder and sat watching a run that had
already finished, reporting "waiting for the next checkpoint" forever while the
real run filmed nothing.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

LOG_ROOT = ROOT / "logs" / "rsl_rl"
RUNS = ROOT / "progress" / "runs"


def iteration_of(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def experiment_to_task() -> dict[str, str]:
    """Which task writes to which log folder, asked of the registry not guessed."""
    import gray.tasks  # noqa: F401,PLC0415
    from mjlab.tasks.registry import list_tasks, load_rl_cfg  # noqa: PLC0415

    out = {}
    for task in list_tasks():
        if not task.startswith("Gray-"):
            continue
        out[load_rl_cfg(task).experiment_name] = task
    return out


def newest_run(mapping: dict[str, str], want_task: str | None) -> tuple[Path, str] | None:
    """The most recently written run directory, and the task that made it."""
    best = None
    for exp_dir in LOG_ROOT.iterdir() if LOG_ROOT.is_dir() else []:
        task = mapping.get(exp_dir.name)
        if task is None or (want_task and task != want_task):
            continue
        for run in exp_dir.iterdir():
            if not run.is_dir():
                continue
            # Sort on the checkpoints, not the folder: the folder's timestamp
            # stops moving while training is still writing into it.
            stamps = [p.stat().st_mtime for p in run.glob("model_*.pt")]
            when = max(stamps) if stamps else run.stat().st_mtime
            if best is None or when > best[0]:
                best = (when, run, task)
    return (best[1], best[2]) if best else None


class Filmer:
    """Holds one env and runner, rebuilt only when the task changes."""

    def __init__(self, seconds: float, envs: int) -> None:
        self.seconds, self.envs = seconds, envs
        self.task: str | None = None

    def _build(self, task: str) -> None:
        import mujoco  # noqa: PLC0415

        import gray.tasks  # noqa: F401,PLC0415
        from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: PLC0415

        if self.task is not None:
            self.env.close()
        env_cfg = load_env_cfg(task, play=True)
        env_cfg.scene.num_envs = self.envs
        agent_cfg = load_rl_cfg(task)
        self.env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device="cuda:0"),
                                      clip_actions=agent_cfg.clip_actions)
        self.runner = MjlabOnPolicyRunner(self.env, asdict(agent_cfg), device="cuda:0")
        self.model = self.env.unwrapped.sim.mj_model
        self.renderer = mujoco.Renderer(self.model, height=480, width=854)
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.0, -12.0, 125.0
        self.task = task
        print(f"[film] built environment for {task}", flush=True)

    def film(self, ckpt: Path, task: str, dst: Path) -> int:
        import imageio.v2 as imageio  # noqa: PLC0415
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if task != self.task:
            self._build(task)
        self.runner.load(str(ckpt), load_cfg={"actor": True}, strict=True,
                         map_location="cuda:0")
        policy = self.runner.get_inference_policy(device="cuda:0")

        obs, _ = self.env.reset()
        frames = []
        with torch.inference_mode():
            for _ in range(int(self.seconds * 50)):
                obs = self.env.step(policy(obs))[0]
                sim = self.env.unwrapped.sim.data
                self.data.qpos[:] = sim.qpos[0].detach().cpu().numpy()
                self.data.qvel[:] = sim.qvel[0].detach().cpu().numpy()
                mujoco.mj_forward(self.model, self.data)
                self.cam.lookat[:] = self.data.xpos[1]
                self.renderer.update_scene(self.data, self.cam)
                frames.append(np.asarray(self.renderer.render()))

        # Write under a temporary name and move it into place, so the dashboard
        # never serves a half-encoded file. Keep the .mp4: imageio picks its
        # writer from the extension and cannot open a file called .part.
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".writing.mp4")
        imageio.mimwrite(tmp, frames, fps=50, codec="libx264", quality=7,
                         macro_block_size=1)
        tmp.replace(dst)
        return len(frames)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", help="only follow this task")
    ap.add_argument("--every", type=int, default=1, help="film 1 in N checkpoints")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--envs", type=int, default=1)
    ap.add_argument("--watch", action="store_true",
                    help="keep running alongside training, filming each new checkpoint")
    args = ap.parse_args()

    mapping = experiment_to_task()
    filmer = Filmer(args.seconds, args.envs)
    print(f"mode     {'following whatever is training' if args.watch else 'one pass'}")
    print(f"tasks    {', '.join(sorted(mapping.values()))}\n", flush=True)

    current: Path | None = None
    while True:
        found = newest_run(mapping, args.task)
        if found is None:
            print("  (no runs yet)", flush=True)
        else:
            log_dir, task = found
            if log_dir != current:
                current = log_dir
                print(f"[film] following {task}  {log_dir.name}", flush=True)
            out_dir = RUNS / log_dir.name / "videos"
            todo = [c for i, c in enumerate(sorted(log_dir.glob("model_*.pt"),
                                                   key=iteration_of))
                    if i % args.every == 0
                    and not (out_dir / f"iter_{iteration_of(c):04d}.mp4").exists()]
            for ckpt in todo:
                it = iteration_of(ckpt)
                try:
                    n = filmer.film(ckpt, task, out_dir / f"iter_{it:04d}.mp4")
                    print(f"  iteration {it:>5}  ->  iter_{it:04d}.mp4  ({n} frames)",
                          flush=True)
                except Exception as exc:  # noqa: BLE001
                    # A checkpoint half-written as we read it is normal alongside
                    # training; it gets picked up on the next pass.
                    print(f"  iteration {it:>5}  skipped: {type(exc).__name__}: "
                          f"{str(exc)[:70]}", flush=True)
            if not todo and args.watch:
                print("  (up to date)", flush=True)

        if not args.watch:
            break
        time.sleep(20.0)

    print("\nReload the dashboard to see them.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nstopped")
